
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
