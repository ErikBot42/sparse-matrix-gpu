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
fn main() {
    env_logger::init();
    let i = u16::MAX.into();
    let repeat_operation = 1;

    let data = black_box(SpmvData::new_random(black_box(i)));

    let now = Instant::now();
    let data_gpu = spmv_gpu(data.clone(), repeat_operation);
    let gpu_time = now.elapsed().as_millis();

    let now = Instant::now();
    let data_cpu = spmv_cpu_unchecked_indexing(data.clone(), repeat_operation);
    let cpu_time = now.elapsed().as_millis();

    println!("Data size: {i}");
    println!("Elapsed GPU: {gpu_time} ms");
    println!("Elapsed CPU: {cpu_time} ms");
    assert!(data_cpu.y == data_gpu.y, "Compution not equal");
    println!("Everything completed as expected.");
}

async fn prep_gpu() -> (Features, Device, Queue, Option<QuerySet>) {
    let instance = Instance::new(Backends::PRIMARY);
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    let features = adapter.features();
    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                label: None,
                features: features & Features::TIMESTAMP_QUERY,
                limits: Default::default(),
            },
            None,
        )
        .await
        .unwrap();

    dbg!(&device);
    dbg!(device.limits());
    let query_set = if features.contains(Features::TIMESTAMP_QUERY) {
        Some(device.create_query_set(&QuerySetDescriptor {
            count: 2,
            ty: QueryType::Timestamp,
            label: None,
        }))
    } else {
        None
    };
    (features, device, queue, query_set)
}

fn prepare_shader(device: &Device) -> ShaderModule {
    let start_instant = Instant::now();
    let cs_module = device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shader.wgsl"))),
    });
    println!("shader compilation {:?}", start_instant.elapsed());
    cs_module
}

// CSR without any attached data
#[derive(Clone, Debug)]
struct Csr {
    indexes: Vec<u32>,
    outputs: Vec<u32>,
}
impl Csr {
    // List of lists to compressed sparse row
    fn from_list_of_lists(ll: Vec<Vec<u32>>) -> Self {
        let ll: Vec<Vec<u32>> = ll
            .into_iter()
            .map(|x| {
                let mut x = x;
                x.sort();
                x.dedup();
                x
            })
            .collect();
        let mut indexes: Vec<u32> = Vec::new();
        let mut outputs: Vec<u32> = Vec::new();
        for mut l in ll {
            indexes.push(outputs.len() as u32);
            outputs.append(&mut l);
        }
        indexes.push(outputs.len() as u32);
        Self { indexes, outputs }
    }
}
type DenseVector = Vec<u32>;
type ListOfLists = Vec<Vec<u32>>;

/// Spmv will do the following:
/// y = y + Ax, where A is a sparse matrix
#[derive(Clone, Debug)]
struct SpmvData {
    csr: Csr,       // outputs
    x: DenseVector, // delta
    y: DenseVector, // acc
}
impl SpmvData {
    fn new(csr: Csr, x: DenseVector, y: DenseVector) -> Self {
        assert_eq!(csr.indexes.len() - 1, x.len());
        assert_eq!(y.len(), x.len());
        Self { csr, x, y }
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

        let connections_to_add = ((length * length) as f64 * matrix_density) as u32;
        let connections_to_add = length * 16;

        for _ in 0..connections_to_add {
            ll[rng.gen_range(0..length)].push(rng.gen_range(0..length) as u32);
        }
        assert_eq!(ll.len(), length);
        let csr = Csr::from_list_of_lists(ll);
        Self::new(csr, x, y)
    }
}

fn spmv_gpu(input: SpmvData, repeat_operation: usize) -> SpmvData {
    pollster::block_on(spmv_gpu_i(input, repeat_operation))
}

async fn spmv_gpu_i(input: SpmvData, repeat_operation: usize) -> SpmvData {
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

    spmv_gpu_ei(&device, &queue, input, repeat_operation).await
}

async fn spmv_gpu_ei(
    device: &Device,
    queue: &Queue,
    mut input: SpmvData,
    repeat_operation: usize,
) -> SpmvData {
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shader.wgsl"))),
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

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: "main",
    });

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
            cpass.dispatch_workgroups((&input.y).len() as u32, 32, 32);
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_unchecked_indexing() {
        for i in [10, 100, 1000] {
            dbg!(i);
            let data = SpmvData::new_random(i);
            let data1 = spmv_cpu_reference(data.clone());
            let data2 = spmv_cpu_unchecked_indexing(data.clone(), 1);
            assert_eq!(data1.y, data2.y);
        }
    }
    #[test]
    fn test_gpu() {
        for i in [10, 100, 1000] {
            dbg!(i);
            let data = SpmvData::new_random(i);
            let data1 = spmv_cpu_reference(data.clone());
            let data2 = spmv_gpu(data.clone(), 1);
            assert_eq!(data1.y, data2.y);
        }
    }
}
