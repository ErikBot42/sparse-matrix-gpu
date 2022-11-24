use std::hint::black_box;
use std::time::Instant;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, Buffer, BufferAddress, BufferBindingType, BufferDescriptor, BufferUsages,
    ComputePipeline, ComputePipelineDescriptor, Device, DeviceDescriptor, Features, Instance,
    Maintain, MapMode, PipelineLayout, PipelineLayoutDescriptor, QuerySet, QuerySetDescriptor,
    QueryType, Queue, ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

#[cfg(test)]
mod tests;

fn main() {
    env_logger::init();
    let i = u16::MAX.into();
    let repeat_operation = 32;

    let data = black_box(SpmvData::new_random(black_box(i)));

    let now = Instant::now();
    let data_gpu = spmv_gpu(data.clone(), repeat_operation);
    let gpu_time = now.elapsed().as_millis();

    let now = Instant::now();
    let data_cpu = spmv_cpu_unchecked_indexing(data, repeat_operation);
    let cpu_time = now.elapsed().as_millis();

    println!("Data size: {i}");
    println!("Elapsed GPU: {gpu_time} ms");
    println!("Elapsed CPU: {cpu_time} ms");
    assert!(data_cpu.y == data_gpu.y, "Compution not equal");
    println!("Everything completed as expected.");
}

type DenseVector = Vec<u32>;
type ListOfLists = Vec<Vec<u32>>;

/// Spmv will do the following:
/// y = y + Ax, where A is a sparse matrix
#[derive(Clone, Debug)]
struct SpmvData {
    csr: Csr,       // outputs
    csc: Csc,       // inputs
    x: DenseVector, // delta
    y: DenseVector, // acc
}
impl SpmvData {
    fn new(lil: &Lil, x: DenseVector, y: DenseVector) -> Self {
        let csr = Csr::from_lil(lil);
        let csc = Csc::from_lil(lil);
        assert_eq!(csr.indexes.len() - 1, x.len());
        assert_eq!(y.len(), x.len());
        Self { csc, csr, x, y }
    }
    fn new_random(length: usize) -> Self {
        use rand::prelude::*;
        use rand::rngs::StdRng;
        let matrix_density = 0.1;
        let x_density = 0.8;
        let mut rng = StdRng::seed_from_u64(42);
        let (x, y): (DenseVector, DenseVector) = (0..length)
            .map(|_| {
                (
                    if rng.gen_bool(x_density) {
                        rng.gen::<u32>()
                    } else {
                        0
                    },
                    rng.gen::<u32>(),
                )
            })
            .unzip();
        let mut ll: ListOfLists = (0..length).map(|_| Vec::new()).collect();

        //let connections_to_add = ((length * length) as f64 * matrix_density) as u32;
        let connections_to_add = length * 16;

        for _ in 0..connections_to_add {
            ll[rng.gen_range(0..length)].push(rng.gen_range(0..length) as u32);
        }
        assert_eq!(ll.len(), length);
        Self::new(&Lil::new(ll), x, y)
    }
}

fn spmv_gpu(input: SpmvData, repeat_operation: usize) -> SpmvData {
    pollster::block_on(spmv_gpu_i(input, repeat_operation))
}

async fn prep_gpu() -> (Device, Queue) {
    let instance = wgpu::Instance::new(wgpu::Backends::all());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .unwrap();
    dbg!(device.limits());
    dbg!(adapter.get_info());
    (device, queue)
}
async fn spmv_gpu_i(input: SpmvData, repeat_operation: usize) -> SpmvData {
    let (device, queue) = prep_gpu().await;
    spmv_gpu_ei(&device, &queue, input, repeat_operation).await
}


async fn spmv_gpu_ei(
    device: &Device,
    queue: &Queue,
    mut input: SpmvData,
    repeat_operation: usize,
) -> SpmvData {
    let source = include_str!("shader.wgsl");
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)),
    });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: "main",
    });


    let y_slice_size = (&input.y).len() * std::mem::size_of::<u32>();
    let y_size = y_slice_size as BufferAddress;

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: y_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let create_buffer = |content: &[u32], usage: BufferUsages| -> Buffer {
        device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&content),
            usage,
        })
    };
    let default_usages = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;

    let read_only_usage = BufferUsages::STORAGE | BufferUsages::COPY_DST;

    let y_buffer = create_buffer(&input.y, default_usages);
    let x_buffer = create_buffer(&input.x, read_only_usage);
    let a_indexes_buffer = create_buffer(&input.csr.indexes, read_only_usage);
    let a_outputs_buffer = create_buffer(&input.csr.outputs, read_only_usage);


    // this is obtained from the **shader**
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: y_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: a_indexes_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: a_outputs_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        for _ in 0..repeat_operation {
            cpass.dispatch_workgroups((&input.y).len() as u32, 1, 1);
        }
    }
    encoder.copy_buffer_to_buffer(&y_buffer, 0, &staging_buffer, 0, y_size);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    let now = Instant::now();
    device.poll(wgpu::Maintain::Wait);
    println!("Elapsed {} ms during GPU poll", now.elapsed().as_millis());

    input.y = if let Some(Ok(())) = receiver.receive().await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
        let result = bytemuck::cast_slice(&data).to_vec();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        staging_buffer.unmap(); // Unmaps buffer from memory
                                // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                //   delete myPointer;
                                //   myPointer = NULL;
                                // It effectively frees the memory

        // Returns data from buffer
        result
    } else {
        panic!("failed to run compute on gpu!")
    };
    input
}

#[cfg(test)]
fn spmv_cpu_reference(mut input: SpmvData) -> SpmvData {
    for i in 0..input.y.len() {
        let from_index = input.csr.indexes[i] as usize;
        let to_index = input.csr.indexes[i + 1] as usize;
        let delta = input.x[i];
        for i in input.csr.outputs[from_index..to_index]
            .iter()
            .map(|i| *i as usize)
        {
            let r = &mut input.y[i];
            *r = r.wrapping_add(delta);
        }
    }
    input
}

fn spmv_cpu_unchecked_indexing(mut input: SpmvData, repeat_operation: usize) -> SpmvData {
    spmv_cpu_unchecked_indexing_in_place(&mut input, repeat_operation);
    input
}

fn spmv_cpu_unchecked_indexing_in_place(input: &mut SpmvData, repeat_operation: usize) {
    unsafe {
        for _ in 0..repeat_operation {
            for i in 0..input.y.len() {
                let from_index = *input.csr.indexes.get_unchecked(i) as usize;
                let to_index = *input.csr.indexes.get_unchecked(i + 1) as usize;
                let delta = *input.x.get_unchecked(i);
                for i in input
                    .csr
                    .outputs
                    .get_unchecked(from_index..to_index)
                    .iter()
                    .map(|i| *i as usize)
                {
                    let r = input.y.get_unchecked_mut(i);
                    *r = black_box(r.wrapping_add(delta));
                }
            }
        }
    }
}
use sparse::Csc;
use sparse::Csr;
use sparse::Lil;
mod sparse {

    /// List of lists
    /// Inner is exposed
    #[derive(Debug, PartialEq, Eq)]
    pub(super) struct Lil {
        pub(super) i: Vec<Vec<u32>>,
    }
    impl Lil {
        pub(super) fn new(mut i: Vec<Vec<u32>>) -> Self {
            i.iter_mut().for_each(|s| s.sort());
            Self { i }
        }
        pub(super) fn new_i(size: usize) -> Vec<Vec<u32>> {
            (0..size).map(|_| Vec::new()).collect()
        }
        #[cfg(test)]
        pub(super) fn new_random<F: rand::Rng>(rng: &mut F, density: f32, size: usize) -> Self {
            let mut tmp = Self::new_i(size);
            for _ in 0..(size as f32 * density) as usize {
                tmp[rng.gen_range(0..size)].push(rng.gen_range(0..size) as u32);
            }
            Self::new(tmp)
        }
        pub(super) fn reversed(&self) -> Self {
            let mut tmp = Self::new_i(self.i.len());
            for (i, list) in self.i.iter().enumerate() {
                for j in list {
                    tmp[*j as usize].push(i as u32)
                }
            }
            Self::new(tmp)
        }
    }

    struct Cs {}
    impl Cs {
        fn from_lil(lil: &Lil) -> (Vec<u32>, Vec<u32>) {
            let mut indexes: Vec<u32> = Vec::new();
            let mut outputs: Vec<u32> = Vec::new();
            for mut l in lil.i.iter().cloned() {
                indexes.push(outputs.len() as u32);
                outputs.append(&mut l);
            }
            indexes.push(outputs.len() as u32);
            (indexes, outputs)
        }
    }

    /// CSR without any attached data
    /// Inner is exposed
    #[derive(Clone, Debug)]
    pub(super) struct Csr {
        pub(super) indexes: Vec<u32>,
        pub(super) outputs: Vec<u32>,
    }
    impl Csr {
        // List of lists to compressed sparse row
        pub(super) fn from_lil(lil: &Lil) -> Self {
            let (indexes, outputs) = Cs::from_lil(lil);
            Self { indexes, outputs }
        }
    }

    /// CSC without any attached data
    /// Inner is exposed
    #[derive(Clone, Debug)]
    pub(super) struct Csc {
        pub(super) indexes: Vec<u32>,
        pub(super) outputs: Vec<u32>,
    }
    impl Csc {
        // List of lists to compressed sparse row
        pub(super) fn from_lil(lil: &Lil) -> Self {
            let (indexes, outputs) = Cs::from_lil(&lil.reversed());
            Self { indexes, outputs }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn test_sparse_lil() {
            use rand::prelude::*;
            use rand::rngs::StdRng;
            let matrix_density = 4.0;
            let mut rng = StdRng::seed_from_u64(42);

            let lil = Lil::new_random(&mut rng, matrix_density, 128);
            let lil_reversed = lil.reversed();
            assert_ne!(lil, lil_reversed);
            assert_eq!(lil, lil_reversed.reversed());
        }
    }
}
