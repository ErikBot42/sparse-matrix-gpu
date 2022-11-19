struct DataBuf {
    data: array<f32>,
}

@group(0)
@binding(42)
var<storage, read_write> v_indices: DataBuf;

@group(0)
@binding(43)
var<storage, read_write> v_indices2: DataBuf;

@compute
@workgroup_size(1) // does not need to execute in group
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    //v_indices.data[global_id.x] = f32(global_id.x) + v_indices.data[global_id.x];
    //v_indices.data[global_id.x] = 1.0 + v_indices.data[global_id.x];
    v_indices.data[global_id.x] = 1.0 + v_indices.data[global_id.x];
}
