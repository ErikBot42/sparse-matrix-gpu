use super::*;
#[test]
fn test_unchecked_indexing() {
    for i in [10, 100, 1000] {
        dbg!(i);
        let data = SpmvData::new_random(i);
        let data1 = spmv_cpu_csr_reference(data.clone());
        let data2 = spmv_cpu_csr_unchecked_indexing(data.clone(), 1);
        assert_eq!(data1.y, data2.y);
    }
}
#[test]
fn test_csc() {
    for i in [10, 100, 1000] {
        let data = SpmvData::new_random(i);
        let data1 = spmv_cpu_csr_reference(data.clone());
        let data2 = spmv_cpu_csc_reference(data.clone());
        assert_eq!(data1.y, data2.y);
    }
}
#[test]
fn test_gpu() {
    for i in [10, 100, 1000] {
        dbg!(i);
        let data = SpmvData::new_random(i);
        let data1 = spmv_cpu_csr_reference(data.clone());
        let data2 = spmv_gpu(data.clone(), 1);
        assert_eq!(data1.y, data2.y);
    }
}

