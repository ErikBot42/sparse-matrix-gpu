//struct DataBuf {
//    data: array<f32>,
//}

@group(0)
@binding(42)
var<storage, read_write> first_array: array<f32>;

@group(0)
@binding(43)
var<storage, read_write> second_array: array<f32>;

@compute
@workgroup_size(1) // does not need to execute in group
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    first_array[global_id.x] = first_array[global_id.x] + second_array[global_id.x];
}
