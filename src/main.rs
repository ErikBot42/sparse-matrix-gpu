use std::hint::black_box;
use std::time::Instant;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupLayout, Buffer, BufferAddress, BufferUsages, ComputePipeline, Device,
    Queue, ShaderModule,
};

#[cfg(test)]
mod tests;

fn main() {
    env_logger::init();
    let i = u16::MAX.into();
    let repeat_operation = 128;

    let data = black_box(SpmvData::new_random(black_box(i)));
    let data_gpu = data.clone();
    let data_cpu = data;

    let now = Instant::now();
    let data_gpu = spmv_gpu(data_gpu, repeat_operation);
    let gpu_time = now.elapsed();

    let now = Instant::now();
    let data_cpu = spmv_cpu_csr_unchecked_indexing(data_cpu, repeat_operation);
    let cpu_time = now.elapsed();

    println!("Data size: {i}");
    println!("Elapsed GPU: {gpu_time:?}");
    println!("Elapsed CPU: {cpu_time:?}");
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
        assert_eq!(csc.indexes.len() - 1, x.len());
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
    let compute_pipeline = make_compute_pipeline(device, include_str!("shader.wgsl"), "main");

    let y_slice_size = (&input.y).len() * std::mem::size_of::<u32>();
    let y_size = y_slice_size as BufferAddress;

    let staging_buffer: Buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: y_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let default_usages = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;

    let read_only = BufferUsages::STORAGE | BufferUsages::COPY_DST;

    // this is obtained from the **shader**
    let bind_group_layout = &compute_pipeline.get_bind_group_layout(0);

    let y_buffer = create_buff(device, &input.y, default_usages);
    let bind_group = make_bind_group(
        device,
        bind_group_layout,
        [
            (0, &y_buffer),
            (1, &create_buff(device, &input.x, read_only)),
            (2, &create_buff(device, &input.csr.indexes, read_only)),
            (3, &create_buff(device, &input.csr.outputs, read_only)),
        ],
    );

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

    input.y = receiver
        .receive()
        .await
        .map(|_| {
            let data = buffer_slice.get_mapped_range();
            let result = bytemuck::cast_slice(&data).to_vec();
            drop(data); // needs to be done before unmap
            staging_buffer.unmap(); // Unmaps buffer from memory
            result
        })
        .expect("failed to run compute on gpu");
    input
}

mod gpu {}

fn make_bind_group<const LENGTH: usize>(
    device: &Device,
    bind_group_layout: &BindGroupLayout,
    resources: [(u32, &Buffer); LENGTH],
) -> BindGroup {
    let entries = resources.map(|(binding, resource)| wgpu::BindGroupEntry {
        binding,
        resource: resource.as_entire_binding(),
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: bind_group_layout,
        entries: &entries,
    })
}

fn create_buff(device: &Device, content: &[u32], usage: BufferUsages) -> Buffer {
    device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&content),
        usage,
    })
}

fn make_compute_pipeline(device: &Device, source: &str, entry_point: &str) -> ComputePipeline {
    let cs_module: ShaderModule = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)),
    });
    let compute_pipeline: ComputePipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &cs_module,
            entry_point,
        });
    compute_pipeline
}

fn spmv_cpu_csr_reference(mut input: SpmvData) -> SpmvData {
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

fn spmv_cpu_csc_reference(mut input: SpmvData) -> SpmvData {
    for (i, y_curr) in input.y.iter_mut().enumerate() {
        let from_index = input.csc.indexes[i] as usize;
        let to_index = input.csc.indexes[i + 1] as usize;
        *y_curr = input.csc.outputs[from_index..to_index]
            .iter()
            .map(|i| input.x[*i as usize])
            .fold(*y_curr, |a: u32, b: u32| a.wrapping_add(b));
    }
    input
}

fn spmv_cpu_csc_unchecked_indexing_in_place(input: &mut SpmvData) {
    unsafe {
        for (i, y_curr) in input.y.iter_mut().enumerate() {
            let from_index = *input.csc.indexes.get_unchecked(i) as usize;
            let to_index = *input.csc.indexes.get_unchecked(i + 1) as usize;
            *y_curr = input
                .csc
                .outputs
                .get_unchecked(from_index..to_index)
                .iter()
                .map(|i| *input.x.get_unchecked(*i as usize))
                .fold(*y_curr, |a: u32, b: u32| a.wrapping_add(b));
        }
    }
}

fn spmv_cpu_csr_unchecked_indexing(mut input: SpmvData, repeat_operation: usize) -> SpmvData {
    spmv_cpu_csr_unchecked_indexing_in_place(&mut input, repeat_operation);
    input
}

fn spmv_cpu_csr_unchecked_indexing_in_place(input: &mut SpmvData, repeat_operation: usize) {
    unsafe {
        for _ in 0..repeat_operation {
            for i in 0..input.y.len() {
                let from_index = *input.csr.indexes.get_unchecked(i) as usize;
                let to_index = *input.csr.indexes.get_unchecked(i + 1) as usize;
                let delta = *input.x.get_unchecked(i);
                if delta == 0 {
                    continue;
                };
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
            Self::from_edgelist(&EdgeList::from_lil(self).reversed())
        }
        pub(super) fn from_edgelist(edgelist: &EdgeList) -> Self {
            let mut tmp = Vec::new();
            for (i, j) in edgelist.edges.iter().map(|(i, j)| (*i as usize, *j)) {
                while i >= tmp.len() {
                    tmp.push(Vec::new())
                }
                tmp[i].push(j);
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

    pub(super) struct EdgeList {
        pub(super) edges: Vec<(u32, u32)>,
    }
    impl EdgeList {
        fn from_lil(lil: &Lil) -> Self {
            let mut edges: Vec<(u32, u32)> = Vec::new();
            for (i, list) in lil.i.iter().enumerate() {
                for j in list {
                    edges.push((i as u32, *j));
                }
            }
            edges.sort();
            Self { edges }
        }
        fn reversed(&self) -> Self {
            let mut edges: Vec<_> = self.edges.iter().cloned().map(|(i, j)| (j, i)).collect();
            edges.sort();
            Self { edges }
        }
    }

    #[cfg(test)]
    mod tests {
        fn test_iter() -> impl Iterator<Item = (u64, usize)> {
            itertools::iproduct!([42, 67, 43, 99], [10, 101, 1230, 42])
        }
        fn test_lil_iter() -> impl Iterator<Item = Lil> {
            test_iter().map(|(seed, size)| random_lil(seed, size))
        }
        fn random_lil(seed: u64, size: usize) -> Lil {
            use rand::prelude::*;
            use rand::rngs::StdRng;
            let matrix_density = 4.0;
            let mut rng = StdRng::seed_from_u64(seed);
            Lil::new_random(&mut rng, matrix_density, size)
        }
        use super::*;
        #[test]
        fn test_edgelist_reverse() {
            for lil in test_lil_iter() {
                let edge = EdgeList::from_lil(&lil);
                assert_eq!(edge.edges, edge.reversed().reversed().edges);
            }
        }
        #[test]
        fn test_sparse_lil() {
            for lil in test_lil_iter() {
                let lil_reversed = lil.reversed();
                assert_eq!(
                    EdgeList::from_lil(&lil).edges,
                    EdgeList::from_lil(&lil_reversed).reversed().edges
                );
                assert_ne!(lil, lil_reversed);
                assert_eq!(lil, lil_reversed.reversed());
            }
        }
    }
}
