/// Smoke test: initializes D3D12, uploads data, runs compute shaders, downloads results.
fn main() {
    #[cfg(target_os = "windows")]
    {
        use candle_d3d12_kernels::*;

        Gpu::enable_debug_layer();
        println!("Creating GPU context...");
        let gpu = Gpu::new(0).expect("Failed to create GPU");
        let pipelines = Pipelines::new();

        // Test 0: Upload/Download roundtrip
        println!("\n--- Test 0: Upload/Download ---");
        let test_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let test_bytes: Vec<u8> = test_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let buf = gpu.create_buffer(test_bytes.len() as u64).expect("create");
        gpu.upload_to_buffer(&test_bytes, &buf).expect("upload");
        let result = download_f32(&gpu, &buf, 4);
        println!("  {:?} -> {:?}", test_data, result);
        assert_eq!(test_data, result);

        // Test 1: Fill
        println!("\n--- Test 1: Fill ---");
        let count = 8u32;
        let buf = gpu.create_buffer((count as u64) * 4).expect("create");
        fill::call_fill_f32(&gpu, &pipelines, count, 42.0, &buf).expect("fill");
        let result = download_f32(&gpu, &buf, count);
        println!("  fill(42.0, 8) = {:?}", result);
        assert!(result.iter().all(|&v| v == 42.0));

        // Test 2: Unary neg
        println!("\n--- Test 2: Unary neg ---");
        let input = upload_f32(&gpu, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let output = gpu.create_buffer(32).expect("create");
        unary::call_unary_contiguous(&gpu, &pipelines, "neg_f32", 8, &input, &output).expect("neg");
        let result = download_f32(&gpu, &output, 8);
        println!("  neg = {:?}", result);
        assert_eq!(result, vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]);

        // Test 3: Unary exp
        println!("\n--- Test 3: Unary exp ---");
        let input = upload_f32(&gpu, &[0.0, 1.0, -1.0, 2.0]);
        let output = gpu.create_buffer(16).expect("create");
        unary::call_unary_contiguous(&gpu, &pipelines, "exp_f32", 4, &input, &output).expect("exp");
        let result = download_f32(&gpu, &output, 4);
        println!("  exp([0,1,-1,2]) = {:?}", result);
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - std::f32::consts::E).abs() < 1e-5);

        // Test 4: Binary add
        println!("\n--- Test 4: Binary add ---");
        let a = upload_f32(&gpu, &[1.0, 2.0, 3.0, 4.0]);
        let b = upload_f32(&gpu, &[10.0, 20.0, 30.0, 40.0]);
        let c = gpu.create_buffer(16).expect("create");
        binary::call_binary_contiguous(&gpu, &pipelines, "add_f32", 4, &a, &b, &c).expect("add");
        let result = download_f32(&gpu, &c, 4);
        println!("  [1,2,3,4] + [10,20,30,40] = {:?}", result);
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);

        // Test 5: Binary mul
        println!("\n--- Test 5: Binary mul ---");
        let c = gpu.create_buffer(16).expect("create");
        binary::call_binary_contiguous(&gpu, &pipelines, "mul_f32", 4, &a, &b, &c).expect("mul");
        let result = download_f32(&gpu, &c, 4);
        println!("  [1,2,3,4] * [10,20,30,40] = {:?}", result);
        assert_eq!(result, vec![10.0, 40.0, 90.0, 160.0]);

        // Test 6: Reduce sum
        println!("\n--- Test 6: Reduce sum ---");
        let input = upload_f32(&gpu, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let output = gpu.create_buffer(4).expect("create");
        reduce::call_reduce(&gpu, &pipelines, "sum_f32", 8, 1, &input, &output).expect("sum");
        let result = download_f32(&gpu, &output, 1);
        println!("  sum([1..8]) = {:?} (expected 36.0)", result);
        assert!((result[0] - 36.0).abs() < 1e-4);

        // Test 7: Reduce mean
        println!("\n--- Test 7: Reduce mean ---");
        let output = gpu.create_buffer(4).expect("create");
        reduce::call_reduce(&gpu, &pipelines, "mean_f32", 8, 1, &input, &output).expect("mean");
        let result = download_f32(&gpu, &output, 1);
        println!("  mean([1..8]) = {:?} (expected 4.5)", result);
        assert!((result[0] - 4.5).abs() < 1e-4);

        // Test 8: Affine
        println!("\n--- Test 8: Affine ---");
        let input = upload_f32(&gpu, &[1.0, 2.0, 3.0, 4.0]);
        let output = gpu.create_buffer(16).expect("create");
        affine::call_affine(&gpu, &pipelines, 4, &input, &output, 2.0, 1.0).expect("affine");
        let result = download_f32(&gpu, &output, 4);
        println!("  [1,2,3,4] * 2 + 1 = {:?}", result);
        assert_eq!(result, vec![3.0, 5.0, 7.0, 9.0]);

        // Test 9: Matmul 2x3 * 3x2
        println!("\n--- Test 9: Matmul ---");
        let a = upload_f32(&gpu, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = upload_f32(&gpu, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = gpu.create_buffer(16).expect("create");
        matmul::call_matmul(
            &gpu, &pipelines, 1, 2, 2, 3, 6, 3, 1, 6, 2, 1,
            &a, 6, &b, 6, &c,
        ).expect("matmul");
        let result = download_f32(&gpu, &c, 4);
        println!("  [[1,2,3],[4,5,6]] * [[7,8],[9,10],[11,12]] = {:?}", result);
        println!("  expected: [58, 64, 139, 154]");
        let expected = [58.0f32, 64.0, 139.0, 154.0];
        for i in 0..4 {
            assert!((result[i] - expected[i]).abs() < 1e-4, "matmul mismatch at {i}");
        }

        // Test 10: Softmax
        println!("\n--- Test 10: Softmax ---");
        let input = upload_f32(&gpu, &[1.0, 2.0, 3.0, 4.0]);
        let output = gpu.create_buffer(16).expect("create");
        reduce::call_softmax(&gpu, &pipelines, 4, 1, &input, &output).expect("softmax");
        let result = download_f32(&gpu, &output, 4);
        let sum: f32 = result.iter().sum();
        println!("  softmax([1,2,3,4]) = {:?} (sum={})", result, sum);
        assert!((sum - 1.0).abs() < 1e-4, "softmax doesn't sum to 1");
        assert!(result[3] > result[2] && result[2] > result[1] && result[1] > result[0]);

        println!("\n=== All {} tests passed! ===", 11);
    }

    #[cfg(not(target_os = "windows"))]
    println!("This test only runs on Windows.");
}

#[cfg(target_os = "windows")]
fn upload_f32(gpu: &candle_d3d12_kernels::Gpu, data: &[f32]) -> candle_d3d12_kernels::GpuBuffer {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let buf = gpu.create_buffer(bytes.len() as u64).expect("create buffer");
    gpu.upload_to_buffer(&bytes, &buf).expect("upload");
    buf
}

#[cfg(target_os = "windows")]
fn download_f32(gpu: &candle_d3d12_kernels::Gpu, buf: &candle_d3d12_kernels::GpuBuffer, count: u32) -> Vec<f32> {
    let data = gpu.download_buffer(buf, (count as u64) * 4).expect("download");
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
