@group(0)
@binding(0)
var<storage, read_write> ys: array<atomic<u32>>;

@group(0)
@binding(1)
var<storage, read> xs: array<u32>;


@group(0)
@binding(2)
var<storage, read> indexes: array<u32>;

@group(0)
@binding(3)
var<storage, read> outputs: array<u32>;

@compute
@workgroup_size(1) // does not need to execute in group
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i: u32     = global_id.x;
    let from_index = indexes[i];
    let to_index   = indexes[i+u32(1)];
    let delta      = xs[i];
    //if delta == u32(0) {discard}
    for (var i = from_index; i<to_index; i++) {
        let output = outputs[i];
        atomicAdd(&(ys[output]), delta);
    }
}

