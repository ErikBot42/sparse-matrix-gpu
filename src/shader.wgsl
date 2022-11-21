//struct DataBuf {
//    data: array<f32>,
//}



@group(0)
@binding(0)
var<storage, read_write> ys: array<atomic<u32>>;

@group(0)
@binding(1)
var<storage, read_write> xs: array<atomic<u32>>;


@group(0)
@binding(2)
var<storage, read_write> indexes: array<atomic<u32>>;

@group(0)
@binding(3)
var<storage, read_write> outputs: array<atomic<u32>>;

@compute
@workgroup_size(1) // does not need to execute in group
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i: u32     = global_id.x;

    let from_index = indexes[i];
    let to_index   = indexes[i+u32(1)];
    let delta      = xs[i];
    for (var i = from_index; i<to_index; i++) {
        let output = outputs[i];
        atomicAdd(&(ys[output]), delta);
    }


    //for (var i = 0; i<256; i++) {
    //    //first_array[global_id.x] += u32(1);
    //    ys[i] += u32(1);
    //    xs[i] += u32(1);
    //    indexes[i] += u32(1);
    //    outputs[i] += u32(1);
    //    //atomicAdd(&(first_array[i]), u32(1));
    //    //atomicAdd(&(first_array[global_id.x]), u32(1));
    //    //first_array[global_id.x] = atomicLoad( &(first_array[0]) );
    //    //atomicAdd( &(first_array[global_id.x]), u32(1));
    //    //first_array[global_id.x] = first_array[global_id.x] + second_array[global_id.x];
    //}
}

//first_array[0] += i32(1);

