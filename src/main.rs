use std::time::Instant;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages, ComputePipeline,
    ComputePipelineDescriptor, Device, DeviceDescriptor, Features, Instance, Maintain, MapMode,
    PipelineLayout, PipelineLayoutDescriptor, QuerySet, QuerySetDescriptor, QueryType, Queue,
    ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

async fn run() {
    env_logger::init();
    let (_, device, queue, query_set) = prep_gpu().await;

    let input_f1: &Vec<i32> = &create_input(2_i32);
    let input_f2: &Vec<i32> = &create_input(3_i32);

    let input1: &[u8] = bytemuck::cast_slice(input_f1);
    let input2: &[u8] = bytemuck::cast_slice(input_f2);

    let input_buffer1: Buffer = {
        device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: input1,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        })
    };
    let input_buffer2: Buffer = {
        device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: input2,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        })
    };

    let output_buffer: Buffer = device.create_buffer(&BufferDescriptor {
        label: None,
        size: input1.len() as u64,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Must be initialized.
    let query_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &[0; 16],
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    });

    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            BindGroupLayoutEntry {
                binding: 42,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 43,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 42,
                resource: input_buffer1.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 43,
                resource: input_buffer2.as_entire_binding(),
            },
        ],
    });

    let pipeline: ComputePipeline = {
        let compute_pipeline_layout: PipelineLayout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&compute_pipeline_layout),
            module: &prepare_shader(&device),
            entry_point: "main",
        })
    };

    {
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(input_f1.len() as u32, 1, 1);
        }

        // the operations are done ON the input buffer????
        encoder.copy_buffer_to_buffer(&input_buffer1, 0, &output_buffer, 0, input1.len() as u64);
        if let Some(query_set) = &query_set {
            encoder.resolve_query_set(query_set, 0..2, &query_buffer, 0);
        }
        queue.submit(Some(encoder.finish()));
    }

    let buf_slice = output_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buf_slice.map_async(MapMode::Read, move |v| sender.send(v).unwrap());

    // Assume that both buffers become available at the same time. A more careful
    // approach would be to wait for both notifications to be sent.
    let _query_future = query_buffer.slice(..).map_async(MapMode::Read, |_| ());

    {
        let now = std::time::Instant::now();
        device.poll(Maintain::Wait);
        println!(
            "Elapsed: {} ms during poll",
            now.elapsed().as_nanos() as f64 / 1_000_000.0
        );
    }

    if let Some(Ok(())) = receiver.receive().await {
        let data_raw = &*buf_slice.get_mapped_range();
        let data: &[i32] = bytemuck::cast_slice(data_raw);
        println!("data[..10]: {:?}, ({} elements)", &data[..10], data.len());
    } else {
        println!("could not read data");
    }
}

fn create_input(a: i32) -> Vec<i32> {
    //vec![a; 1 << 16 - 1]
    vec![a; (1 << 16) - 1]
    //vec![a; 10]
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
        for _ in 0..(((length * length) as f64 * matrix_density) as u32) {
            ll[rng.gen_range(0..length)].push(rng.gen_range(0..length) as u32);
        }
        assert_eq!(ll.len(), length);
        let csr = Csr::from_list_of_lists(ll);
        Self::new(csr, x, y)
    }
}

fn spmv_gpu(mut input: SpmvData) -> SpmvData {
    todo!()
}

fn main() {
    pollster::block_on(run());
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

fn spmv_cpu_unchecked_indexing(mut input: SpmvData) -> SpmvData {
    unsafe {
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
                *r = r.wrapping_add(delta);
            }
        }
    }
    input
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_unchecked_indexing() {
        let data = SpmvData::new_random(100);
        let data1 = spmv_cpu_reference(data.clone());
        let data2 = spmv_cpu_unchecked_indexing(data.clone());
        assert_eq!(data1.y, data2.y);
    }
}
