use std::time::Instant;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferSlice, BufferUsages,
    ComputePipelineDescriptor, Device, DeviceDescriptor, Features, Instance, Maintain, MapMode,
    PipelineLayoutDescriptor, QuerySet, QuerySetDescriptor, QueryType, Queue, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

async fn run() {
    env_logger::init();
    let (features, device, queue, query_set) = prep_gpu().await;

    let inp = vec![2.0f32; 1 << 16 - 1];
    //let inp = vec![2.0f32; 1 << 4 - 1];
    let input_f = &inp;
    let input: &[u8] = bytemuck::cast_slice(input_f);
    let input_buf: Buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: input,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    });
    let output_buf: Buffer = device.create_buffer(&BufferDescriptor {
        label: None,
        size: input.len() as u64,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // This works if the buffer is initialized, otherwise reads all 0, for some reason.
    let query_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &[0; 16],
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    });

    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let compute_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: None,
        layout: Some(&compute_pipeline_layout),
        module: &prepare_shader(&device),
        entry_point: "main",
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: input_buf.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 0);
    }
    {
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        for _ in 0..8 {
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(input_f.len() as u32, 1, 1);
        }
    }
    if let Some(query_set) = &query_set {
        encoder.write_timestamp(query_set, 1);
    }
    encoder.copy_buffer_to_buffer(&input_buf, 0, &output_buf, 0, input.len() as u64);
    if let Some(query_set) = &query_set {
        encoder.resolve_query_set(query_set, 0..2, &query_buf, 0);
    }
    queue.submit(Some(encoder.finish()));

    let buf_slice = output_buf.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buf_slice.map_async(MapMode::Read, move |v| sender.send(v).unwrap());
    let query_slice = query_buf.slice(..);

    // Assume that both buffers become available at the same time. A more careful
    // approach would be to wait for both notifications to be sent.
    let _query_future = query_slice.map_async(MapMode::Read, |_| ());
    let now = std::time::Instant::now();
    device.poll(Maintain::Wait);
    println!("Elapsed: {} ms during poll", now.elapsed().as_millis());
    if let Some(Ok(())) = receiver.receive().await {
        let data_raw = &*buf_slice.get_mapped_range();
        let data: &[f32] = bytemuck::cast_slice(data_raw);
        println!("data: {:?}", &data[..10]);
    } else {
        println!("could not read data");
    }
    print_timestamp(features, queue, query_slice);
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

fn print_timestamp(features: Features, queue: Queue, query_slice: BufferSlice) {
    if features.contains(Features::TIMESTAMP_QUERY) {
        let ts_period = queue.get_timestamp_period();
        let ts_data_raw = &*query_slice.get_mapped_range();
        let ts_data: &[u64] = bytemuck::cast_slice(ts_data_raw);
        println!(
            "compute shader elapsed: {:?}ms",
            (ts_data[1] - ts_data[0]) as f64 * ts_period as f64 * 1e-6
        );
    } else {
        println!("could not timestamp");
    }
}

fn prepare_shader(device: &Device) -> ShaderModule {
    let start_instant = Instant::now();
    let cs_module = device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });
    println!("shader compilation {:?}", start_instant.elapsed());
    cs_module
}

fn main() {
    pollster::block_on(run());
}
