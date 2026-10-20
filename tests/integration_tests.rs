use rust_wasm_simd_vector_search::{HnswIndex, Metric};

#[test]
fn test_high_dimensional_vector_search() {
    let dim = 64;
    let mut index = HnswIndex::new(dim, Metric::Cosine);

    // Create 3 semantic clusters of vectors
    // Cluster 1: Centered at base1 (indices 1..=20)
    // Cluster 2: Centered at base2 (indices 21..=40)
    // Cluster 3: Centered at base3 (indices 41..=60)
    let mut base1 = vec![0.1f32; dim];
    base1[0] = 2.0;
    base1[1] = 1.5;

    let mut base2 = vec![0.1f32; dim];
    base2[10] = 2.0;
    base2[11] = 1.5;

    let mut base3 = vec![0.1f32; dim];
    base3[20] = 2.0;
    base3[21] = 1.5;

    for i in 1..=20 {
        let mut v = base1.clone();
        v[i % dim] += (i as f32) * 0.01;
        index.insert(i as u64, v);
    }
    for i in 21..=40 {
        let mut v = base2.clone();
        v[i % dim] += (i as f32) * 0.01;
        index.insert(i as u64, v);
    }
    for i in 41..=60 {
        let mut v = base3.clone();
        v[i % dim] += (i as f32) * 0.01;
        index.insert(i as u64, v);
    }

    assert_eq!(index.len(), 60);

    // Query close to cluster 3
    let mut query = base3.clone();
    query[20] = 2.05;

    let results = index.search(&query, 5);
    println!("Query results: {:#?}", results);
    assert!(!results.is_empty(), "Search should return candidates");
    
    // All top-3 results must belong to cluster 3 (IDs 41..=60)
    for r in &results[..3] {
        assert!(
            r.id >= 41 && r.id <= 60,
            "Top results must belong to cluster 3, got ID {}",
            r.id
        );
        assert!(r.score > 0.95, "Similarity score within cluster should exceed 0.95");
    }
}

#[test]
fn test_euclidean_metric_ordering() {
    let mut index = HnswIndex::new(2, Metric::Euclidean);
    index.insert(1, vec![0.0, 0.0]);
    index.insert(2, vec![1.0, 1.0]);
    index.insert(3, vec![5.0, 5.0]);

    let query = vec![0.1, 0.1];
    let results = index.search(&query, 3);

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].id, 1, "Origin is closest to (0.1, 0.1)");
    assert_eq!(results[1].id, 2, "(1, 1) is second closest");
    assert_eq!(results[2].id, 3, "(5, 5) is furthest");
}

#[test]
fn test_dot_product_metric_ordering() {
    let mut index = HnswIndex::new(2, Metric::DotProduct);
    index.insert(1, vec![1.0, 0.0]);
    index.insert(2, vec![10.0, 0.0]);
    index.insert(3, vec![-5.0, 0.0]);

    let query = vec![1.0, 0.0];
    let results = index.search(&query, 2);

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, 2, "Higher positive dot product ranks first");
    assert_eq!(results[1].id, 1);
}
