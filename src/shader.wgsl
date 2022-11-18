struct DataBuf {
    data: array<f32>,
}

@group(0)
@binding(0)
var<storage, read_write> v_indices: DataBuf;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // TODO: a more interesting computation than this.
    var a = v_indices.data[global_id.x];
    for (var i: i32 = 1; i < 1024*16; i++) {
        a = a * f32(i);
        a = sqrt(a);
        a = a + 3.23423;
        a = a * 2.23849;
    }
    v_indices.data[global_id.x] = a;
}
