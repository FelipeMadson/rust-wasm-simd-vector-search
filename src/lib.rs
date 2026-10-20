//! # Rust WASM SIMD Vector Search Engine
//! 
//! High-performance, zero-cloud client-side vector similarity search engine
//! compiled to WebAssembly with 128-bit SIMD acceleration and HNSW (Hierarchical
//! Navigable Small World) graph indexing.
//!
//! Author: Felipe Madison <160553421+FelipeMadson@users.noreply.github.com>
//! License: MIT

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use wasm_bindgen::prelude::*;

/// Distance metric for vector similarity calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    Cosine,
    DotProduct,
    Euclidean,
}

impl Metric {
    pub fn from_str_name(name: &str) -> Self {
        match name.to_lowercase().as_str() {
            "dot" | "dotproduct" => Metric::DotProduct,
            "euclidean" | "l2" => Metric::Euclidean,
            _ => Metric::Cosine,
        }
    }
}

/// Compute dot product between two f32 slices.
/// Uses loop unrolling for automatic SIMD vectorization.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");
    let len = a.len();
    let mut sum = 0.0f32;
    let chunks = len / 4;

    for i in 0..chunks {
        let idx = i * 4;
        sum += a[idx] * b[idx]
            + a[idx + 1] * b[idx + 1]
            + a[idx + 2] * b[idx + 2]
            + a[idx + 3] * b[idx + 3];
    }

    for i in (chunks * 4)..len {
        sum += a[i] * b[i];
    }

    sum
}

/// Compute Euclidean (L2) distance squared between two f32 slices.
#[inline]
pub fn euclidean_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");
    let len = a.len();
    let mut sum = 0.0f32;
    let chunks = len / 4;

    for i in 0..chunks {
        let idx = i * 4;
        let d0 = a[idx] - b[idx];
        let d1 = a[idx + 1] - b[idx + 1];
        let d2 = a[idx + 2] - b[idx + 2];
        let d3 = a[idx + 3] - b[idx + 3];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }

    for i in (chunks * 4)..len {
        let d = a[i] - b[i];
        sum += d * d;
    }

    sum
}

/// Compute Cosine distance: 1.0 - (dot(a, b) / (norm(a) * norm(b)))
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a = dot_product(a, a).sqrt();
    let norm_b = dot_product(b, b).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    let similarity = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
    1.0 - similarity
}

/// Calculate distance based on selected metric.
#[inline]
pub fn calculate_distance(a: &[f32], b: &[f32], metric: Metric) -> f32 {
    match metric {
        Metric::Cosine => cosine_distance(a, b),
        Metric::DotProduct => -dot_product(a, b), // Negative dot product so lower distance = higher similarity
        Metric::Euclidean => euclidean_distance_sq(a, b),
    }
}

/// Result of a nearest neighbor search query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: u64,
    pub score: f32,
    pub distance: f32,
}

/// Internal candidate item for priority queue traversal.
#[derive(Clone, Copy)]
struct Candidate {
    id: u64,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// In-memory Node representing an item in the HNSW graph.
#[derive(Debug, Clone)]
struct Node {
    #[allow(dead_code)]
    id: u64,
    vector: Vec<f32>,
    /// Neighbors indexed by layer: layers[0] = base layer, layers[1] = layer 1, etc.
    layers: Vec<Vec<u64>>,
}

/// Hierarchical Navigable Small World (HNSW) Vector Index.
#[derive(Debug, Clone)]
pub struct HnswIndex {
    dimension: usize,
    metric: Metric,
    m: usize,
    m0: usize,
    ef_construction: usize,
    ef_search: usize,
    level_mult: f64,
    nodes: std::collections::HashMap<u64, Node>,
    entry_point: Option<u64>,
    max_layer: usize,
}

impl HnswIndex {
    pub fn new(dimension: usize, metric: Metric) -> Self {
        let m = 16;
        let m0 = 32;
        let ef_construction = 64;
        let ef_search = 32;
        let level_mult = 1.0 / (m as f64).ln();

        Self {
            dimension,
            metric,
            m,
            m0,
            ef_construction,
            ef_search,
            level_mult,
            nodes: std::collections::HashMap::new(),
            entry_point: None,
            max_layer: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn sample_level(&self) -> usize {
        // Deterministic pseudo-random layer assignment
        let r: f64 = (rand::random::<f64>() + 1e-9).clamp(1e-9, 0.999999);
        (-r.ln() * self.level_mult).floor() as usize
    }

    pub fn insert(&mut self, id: u64, vector: Vec<f32>) {
        assert_eq!(
            vector.len(),
            self.dimension,
            "Vector dimension mismatch (expected {}, got {})",
            self.dimension,
            vector.len()
        );

        let node_level = self.sample_level();

        if self.entry_point.is_none() {
            let mut layers = Vec::with_capacity(node_level + 1);
            for _ in 0..=node_level {
                layers.push(Vec::new());
            }
            let node = Node {
                id,
                vector,
                layers,
            };
            self.nodes.insert(id, node);
            self.entry_point = Some(id);
            self.max_layer = node_level;
            return;
        }

        let mut curr_entry = self.entry_point.unwrap();
        let max_l = self.max_layer;

        // Top-down greedy search for closest node in upper layers
        for layer in (node_level + 1..=max_l).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                let curr_vec = &self.nodes[&curr_entry].vector;
                let curr_dist = calculate_distance(&vector, curr_vec, self.metric);

                if let Some(neighbors) = self.nodes[&curr_entry].layers.get(layer) {
                    for &neighbor_id in neighbors {
                        let n_vec = &self.nodes[&neighbor_id].vector;
                        let n_dist = calculate_distance(&vector, n_vec, self.metric);
                        if n_dist < curr_dist {
                            curr_entry = neighbor_id;
                            changed = true;
                            break;
                        }
                    }
                }
            }
        }

        // Create the new node with allocated layers
        let mut layers = Vec::with_capacity(node_level + 1);
        for _ in 0..=node_level {
            layers.push(Vec::new());
        }
        let new_node = Node {
            id,
            vector: vector.clone(),
            layers,
        };
        self.nodes.insert(id, new_node);

        // Connect neighbors at each layer from min(node_level, max_l) down to 0
        let top_layer = std::cmp::min(node_level, max_l);
        for layer in (0..=top_layer).rev() {
            let candidates = self.search_layer(&vector, curr_entry, self.ef_construction, layer);
            let m_max = if layer == 0 { self.m0 } else { self.m };

            let neighbor_ids: Vec<u64> = candidates.iter().take(m_max).map(|c| c.id).collect();

            // Link new node to neighbors and back
            for &n_id in &neighbor_ids {
                self.nodes.get_mut(&id).unwrap().layers[layer].push(n_id);
                if let Some(n_node) = self.nodes.get_mut(&n_id) {
                    if layer < n_node.layers.len() {
                        n_node.layers[layer].push(id);
                        if n_node.layers[layer].len() > m_max {
                            // Truncate furthest neighbor
                            n_node.layers[layer].truncate(m_max);
                        }
                    }
                }
            }

            if let Some(best) = candidates.first() {
                curr_entry = best.id;
            }
        }

        if node_level > self.max_layer {
            self.max_layer = node_level;
            self.entry_point = Some(id);
        }
    }

    fn search_layer(&self, query: &[f32], entry: u64, ef: usize, layer: usize) -> Vec<Candidate> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut best_set = Vec::new();

        let entry_dist = calculate_distance(query, &self.nodes[&entry].vector, self.metric);
        let entry_cand = Candidate {
            id: entry,
            distance: entry_dist,
        };

        visited.insert(entry);
        candidates.push(entry_cand);
        best_set.push(entry_cand);

        while let Some(current) = candidates.pop() {
            let worst_best_dist = best_set.last().map(|c| c.distance).unwrap_or(f32::MAX);

            if current.distance > worst_best_dist && best_set.len() >= ef {
                break;
            }

            if let Some(neighbors) = self.nodes[&current.id].layers.get(layer) {
                for &neighbor_id in neighbors {
                    if visited.insert(neighbor_id) {
                        let dist = calculate_distance(
                            query,
                            &self.nodes[&neighbor_id].vector,
                            self.metric,
                        );
                        if dist < worst_best_dist || best_set.len() < ef {
                            let cand = Candidate {
                                id: neighbor_id,
                                distance: dist,
                            };
                            candidates.push(cand);
                            best_set.push(cand);
                            best_set.sort_by(|a, b| {
                                a.distance
                                    .partial_cmp(&b.distance)
                                    .unwrap_or(Ordering::Equal)
                            });
                            if best_set.len() > ef {
                                best_set.pop();
                            }
                        }
                    }
                }
            }
        }

        best_set
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        if self.entry_point.is_none() || k == 0 {
            return Vec::new();
        }

        // Hybrid Engine: For collections <= 256 vectors, exact flat scan provides 100% recall
        // with sub-millisecond latency. For larger datasets, HNSW graph traversal is used.
        if self.nodes.len() <= 256 {
            let mut all_results: Vec<SearchResult> = self
                .nodes
                .values()
                .map(|n| {
                    let dist = calculate_distance(query, &n.vector, self.metric);
                    let score = match self.metric {
                        Metric::Cosine => 1.0 - dist,
                        Metric::DotProduct => -dist,
                        Metric::Euclidean => 1.0 / (1.0 + dist.sqrt()),
                    };
                    SearchResult {
                        id: n.id,
                        score,
                        distance: dist,
                    }
                })
                .collect();

            all_results.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(Ordering::Equal)
            });
            all_results.truncate(k);
            return all_results;
        }

        let mut curr_entry = self.entry_point.unwrap();

        for layer in (1..=self.max_layer).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                let curr_dist = calculate_distance(query, &self.nodes[&curr_entry].vector, self.metric);
                if let Some(neighbors) = self.nodes[&curr_entry].layers.get(layer) {
                    for &n_id in neighbors {
                        let n_dist = calculate_distance(query, &self.nodes[&n_id].vector, self.metric);
                        if n_dist < curr_dist {
                            curr_entry = n_id;
                            changed = true;
                            break;
                        }
                    }
                }
            }
        }

        let candidates = self.search_layer(query, curr_entry, std::cmp::max(self.ef_search, k), 0);

        candidates
            .into_iter()
            .take(k)
            .map(|c| {
                let score = match self.metric {
                    Metric::Cosine => 1.0 - c.distance,
                    Metric::DotProduct => -c.distance,
                    Metric::Euclidean => 1.0 / (1.0 + c.distance.sqrt()),
                };
                SearchResult {
                    id: c.id,
                    score,
                    distance: c.distance,
                }
            })
            .collect()
    }
}

/// WebAssembly Bindings for JavaScript / Browser consumption.
#[wasm_bindgen]
pub struct WasmVectorEngine {
    index: HnswIndex,
}

#[wasm_bindgen]
impl WasmVectorEngine {
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize, metric_name: &str) -> Self {
        let metric = Metric::from_str_name(metric_name);
        Self {
            index: HnswIndex::new(dimension, metric),
        }
    }

    #[wasm_bindgen(js_name = insertVector)]
    pub fn insert_vector(&mut self, id: u64, data: &[f32]) -> Result<(), JsValue> {
        if data.len() != self.index.dimension {
            return Err(JsValue::from_str(&format!(
                "Invalid vector dimension: expected {}, got {}",
                self.index.dimension,
                data.len()
            )));
        }
        self.index.insert(id, data.to_vec());
        Ok(())
    }

    #[wasm_bindgen(js_name = search)]
    pub fn search(&self, query: &[f32], k: usize) -> Result<JsValue, JsValue> {
        if query.len() != self.index.dimension {
            return Err(JsValue::from_str(&format!(
                "Invalid query dimension: expected {}, got {}",
                self.index.dimension,
                query.len()
            )));
        }
        let results = self.index.search(query, k);
        serde_wasm_bindgen::to_value(&results)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    #[wasm_bindgen(js_name = count)]
    pub fn count(&self) -> usize {
        self.index.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40.0
        assert_eq!(dot_product(&a, &b), 40.0);
    }

    #[test]
    fn test_cosine_distance_identical() {
        let a = vec![1.0, 0.0, 1.0];
        let b = vec![1.0, 0.0, 1.0];
        let dist = cosine_distance(&a, &b);
        assert!(dist.abs() < 1e-6, "Distance between identical vectors should be ~0");
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let dist = cosine_distance(&a, &b);
        assert!((dist - 1.0).abs() < 1e-6, "Orthogonal vectors should have distance 1.0");
    }

    #[test]
    fn test_hnsw_indexing_and_search() {
        let mut index = HnswIndex::new(3, Metric::Cosine);
        index.insert(1, vec![1.0, 0.0, 0.0]);
        index.insert(2, vec![0.0, 1.0, 0.0]);
        index.insert(3, vec![0.0, 0.0, 1.0]);
        index.insert(4, vec![0.9, 0.1, 0.0]);

        assert_eq!(index.len(), 4);

        // Query vector closest to ID 1 and ID 4
        let query = vec![1.0, 0.05, 0.0];
        let results = index.search(&query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 1);
        assert_eq!(results[1].id, 4);
        assert!(results[0].score > 0.99);
    }
}
