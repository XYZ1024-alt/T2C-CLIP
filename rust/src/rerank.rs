use std::collections::{BTreeSet, HashMap, HashSet};

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
    ndarray::{Array1, Array2},
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

const RERANK_TIE_TOLERANCE: f32 = 1e-6;

#[derive(Clone, Debug)]
struct SparseRow {
    indices: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Debug)]
struct QueryMetrics {
    average_precision: f64,
    first_match_rank: Option<usize>,
}

type TopKPyArrays<'py> = (
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
);
type CsrPyArrays<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);
type SparseIndexParts = (Vec<SparseRow>, Vec<Vec<(usize, f32)>>);

#[pyfunction]
pub fn select_topk_distances<'py>(
    py: Python<'py>,
    distances: PyReadonlyArray2<'_, f32>,
    k: usize,
) -> PyResult<TopKPyArrays<'py>> {
    let shape = distances.shape();
    let row_count = shape[0];
    let column_count = shape[1];
    if row_count == 0 || column_count == 0 {
        return Err(PyValueError::new_err(
            "distances must be a non-empty rank-2 array",
        ));
    }
    if k == 0 || k > column_count {
        return Err(PyValueError::new_err(format!(
            "k must satisfy 1 <= k <= {column_count}, got {k}"
        )));
    }
    let values = distances
        .as_slice()
        .map_err(|_| PyValueError::new_err("distances must be C-contiguous float32"))?
        .to_vec();

    let selected = py.detach(|| {
        (0..row_count)
            .into_par_iter()
            .map(|row_index| {
                let row = &values[row_index * column_count..(row_index + 1) * column_count];
                select_row(row, k, row_index)
            })
            .collect::<Result<Vec<_>, _>>()
    });
    let selected = selected.map_err(PyValueError::new_err)?;

    let mut indices = Vec::with_capacity(row_count * k);
    let mut top_distances = Vec::with_capacity(row_count * k);
    let mut maxima = Vec::with_capacity(row_count);
    for (row_indices, row_distances, maximum) in selected {
        indices.extend(row_indices.into_iter().map(|index| index as i64));
        top_distances.extend(row_distances);
        maxima.push(maximum);
    }
    let indices = Array2::from_shape_vec((row_count, k), indices)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let top_distances = Array2::from_shape_vec((row_count, k), top_distances)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        indices.into_pyarray(py),
        top_distances.into_pyarray(py),
        Array1::from_vec(maxima).into_pyarray(py),
    ))
}

#[pyfunction]
pub fn expanded_reciprocal_sets<'py>(
    py: Python<'py>,
    top_indices: PyReadonlyArray2<'_, i64>,
    k1: usize,
) -> PyResult<CsrPyArrays<'py>> {
    let shape = top_indices.shape();
    let sample_count = shape[0];
    let top_k = shape[1];
    if sample_count < 2 || top_k == 0 {
        return Err(PyValueError::new_err(
            "top_indices must contain at least two rows and one column",
        ));
    }
    if k1 == 0 {
        return Err(PyValueError::new_err("k1 must be positive"));
    }
    let required_k = required_neighbor_count(k1, 1, sample_count);
    if top_k < required_k {
        return Err(PyValueError::new_err(format!(
            "top_indices must contain at least {required_k} columns"
        )));
    }
    let top = validated_top_indices(&top_indices, sample_count)?;
    let rows = py.detach(|| build_expanded_sets(&top, top_k, k1, sample_count));

    let mut offsets = Vec::with_capacity(sample_count + 1);
    let mut flattened = Vec::new();
    offsets.push(0_i64);
    for row in rows {
        flattened.extend(row.into_iter().map(|index| index as i64));
        offsets.push(flattened.len() as i64);
    }
    Ok((
        Array1::from_vec(offsets).into_pyarray(py),
        Array1::from_vec(flattened).into_pyarray(py),
    ))
}

fn select_row(
    row: &[f32],
    k: usize,
    row_index: usize,
) -> Result<(Vec<usize>, Vec<f32>, f32), String> {
    if let Some((column_index, value)) = row
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "distances contain a non-finite value at row {row_index}, column {column_index}: {value}"
        ));
    }
    let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut order: Vec<usize> = (0..row.len()).collect();
    let compare = |left: &usize, right: &usize| {
        canonical_float(row[*left])
            .total_cmp(&canonical_float(row[*right]))
            .then_with(|| left.cmp(right))
    };
    if k < order.len() {
        order.select_nth_unstable_by(k, compare);
        order.truncate(k);
    }
    order.sort_unstable_by(compare);
    let selected_values = order.iter().map(|index| row[*index]).collect();
    Ok((order, selected_values, maximum))
}

#[pyclass(module = "t2c_clip._native", frozen)]
pub struct SparseRerankIndex {
    expanded_rows: Vec<SparseRow>,
    postings: Vec<Vec<(usize, f32)>>,
    max_distances: Vec<f32>,
    query_count: usize,
    sample_count: usize,
}

#[pymethods]
impl SparseRerankIndex {
    #[new]
    #[pyo3(signature = (
        top_indices,
        max_distances,
        affinity_offsets,
        affinity_indices,
        affinity_distances,
        query_count,
        k2
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        top_indices: PyReadonlyArray2<'_, i64>,
        max_distances: PyReadonlyArray1<'_, f32>,
        affinity_offsets: PyReadonlyArray1<'_, i64>,
        affinity_indices: PyReadonlyArray1<'_, i64>,
        affinity_distances: PyReadonlyArray1<'_, f32>,
        query_count: usize,
        k2: usize,
    ) -> PyResult<Self> {
        let top_shape = top_indices.shape();
        let sample_count = top_shape[0];
        let top_k = top_shape[1];
        if sample_count < 2 || top_k == 0 {
            return Err(PyValueError::new_err(
                "top_indices must contain at least two rows and one column",
            ));
        }
        if query_count == 0 || query_count >= sample_count {
            return Err(PyValueError::new_err(
                "query_count must leave at least one query and one gallery sample",
            ));
        }
        if k2 == 0 {
            return Err(PyValueError::new_err("k2 must be positive"));
        }
        if top_k < k2.min(sample_count) {
            return Err(PyValueError::new_err(format!(
                "top_indices must contain at least {} columns for k2",
                k2.min(sample_count)
            )));
        }

        let top = validated_top_indices(&top_indices, sample_count)?;
        let maximum_values = max_distances
            .as_slice()
            .map_err(|_| PyValueError::new_err("max_distances must be contiguous float32"))?
            .to_vec();
        if maximum_values.len() != sample_count {
            return Err(PyValueError::new_err(
                "max_distances length must match top_indices rows",
            ));
        }
        if maximum_values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(PyValueError::new_err(
                "max_distances must contain finite non-negative values",
            ));
        }

        let offsets = validated_offsets(&affinity_offsets, sample_count)?;
        let edge_indices = affinity_indices
            .as_slice()
            .map_err(|_| PyValueError::new_err("affinity_indices must be contiguous int64"))?
            .iter()
            .map(|value| {
                usize::try_from(*value)
                    .ok()
                    .filter(|index| *index < sample_count)
                    .ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "affinity_indices contains out-of-range index {value}"
                        ))
                    })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let edge_distances = affinity_distances
            .as_slice()
            .map_err(|_| PyValueError::new_err("affinity_distances must be contiguous float32"))?
            .to_vec();
        let expected_edges = *offsets.last().unwrap_or(&0);
        if edge_indices.len() != expected_edges || edge_distances.len() != expected_edges {
            return Err(PyValueError::new_err(
                "affinity offsets, indices, and distances describe different edge counts",
            ));
        }
        if edge_distances
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(PyValueError::new_err(
                "affinity_distances must contain finite non-negative values",
            ));
        }

        let built = py.detach(|| {
            build_sparse_index(
                &top,
                top_k,
                &maximum_values,
                &offsets,
                &edge_indices,
                &edge_distances,
                k2,
            )
        });
        let (expanded_rows, postings) = built.map_err(PyValueError::new_err)?;
        Ok(Self {
            expanded_rows,
            postings,
            max_distances: maximum_values
                .into_iter()
                .map(|value| value.max(1e-12))
                .collect(),
            query_count,
            sample_count,
        })
    }

    #[pyo3(signature = (
        original_distances,
        query_start,
        query_ids,
        gallery_ids,
        query_cams,
        gallery_cams,
        ranks,
        lambda_value
    ))]
    #[allow(clippy::too_many_arguments)]
    fn evaluate_block(
        &self,
        py: Python<'_>,
        original_distances: PyReadonlyArray2<'_, f32>,
        query_start: usize,
        query_ids: PyReadonlyArray1<'_, i64>,
        gallery_ids: PyReadonlyArray1<'_, i64>,
        query_cams: PyReadonlyArray1<'_, i64>,
        gallery_cams: PyReadonlyArray1<'_, i64>,
        ranks: Vec<usize>,
        lambda_value: f32,
    ) -> PyResult<(f64, Vec<u64>, usize)> {
        if !lambda_value.is_finite() || !(0.0..=1.0).contains(&lambda_value) {
            return Err(PyValueError::new_err("lambda_value must be in [0, 1]"));
        }
        if ranks.is_empty() || ranks.contains(&0) {
            return Err(PyValueError::new_err(
                "ranks must contain positive integers",
            ));
        }
        let shape = original_distances.shape();
        let block_count = shape[0];
        let gallery_count = self.sample_count - self.query_count;
        if block_count == 0 || shape[1] != gallery_count {
            return Err(PyValueError::new_err(format!(
                "original_distances must have shape (B, {gallery_count}) with B > 0"
            )));
        }
        if query_start + block_count > self.query_count {
            return Err(PyValueError::new_err(
                "query block extends beyond the configured query_count",
            ));
        }
        let distances = original_distances
            .as_slice()
            .map_err(|_| PyValueError::new_err("original_distances must be C-contiguous float32"))?
            .to_vec();
        let query_id_values = query_ids
            .as_slice()
            .map_err(|_| PyValueError::new_err("query_ids must be contiguous int64"))?
            .to_vec();
        let gallery_id_values = gallery_ids
            .as_slice()
            .map_err(|_| PyValueError::new_err("gallery_ids must be contiguous int64"))?
            .to_vec();
        let query_cam_values = query_cams
            .as_slice()
            .map_err(|_| PyValueError::new_err("query_cams must be contiguous int64"))?
            .to_vec();
        let gallery_cam_values = gallery_cams
            .as_slice()
            .map_err(|_| PyValueError::new_err("gallery_cams must be contiguous int64"))?
            .to_vec();
        require_length(&query_id_values, block_count, "query_ids")?;
        require_length(&query_cam_values, block_count, "query_cams")?;
        require_length(&gallery_id_values, gallery_count, "gallery_ids")?;
        require_length(&gallery_cam_values, gallery_count, "gallery_cams")?;

        let rows = py.detach(|| {
            (0..block_count)
                .into_par_iter()
                .map(|block_index| {
                    let query_index = query_start + block_index;
                    let row =
                        &distances[block_index * gallery_count..(block_index + 1) * gallery_count];
                    self.evaluate_query(
                        query_index,
                        row,
                        query_id_values[block_index],
                        &gallery_id_values,
                        query_cam_values[block_index],
                        &gallery_cam_values,
                        lambda_value,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
        });
        let rows = rows.map_err(PyValueError::new_err)?;
        let mut average_precision_sum = 0.0_f64;
        let mut cmc_counts = vec![0_u64; ranks.len()];
        for row in rows {
            average_precision_sum += row.average_precision;
            if let Some(first_match_rank) = row.first_match_rank {
                for (rank_index, rank) in ranks.iter().enumerate() {
                    if first_match_rank < *rank {
                        cmc_counts[rank_index] += 1;
                    }
                }
            }
        }
        Ok((average_precision_sum, cmc_counts, block_count))
    }
}

impl SparseRerankIndex {
    #[allow(clippy::too_many_arguments)]
    fn evaluate_query(
        &self,
        query_index: usize,
        original_distances: &[f32],
        query_id: i64,
        gallery_ids: &[i64],
        query_cam: i64,
        gallery_cams: &[i64],
        lambda_value: f32,
    ) -> Result<QueryMetrics, String> {
        if let Some((gallery_index, value)) = original_distances
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(format!(
                "original_distances contain a non-finite value at query {query_index}, gallery {gallery_index}: {value}"
            ));
        }
        let mut minima = vec![0.0_f32; self.sample_count];
        for (&column, &query_value) in self.expanded_rows[query_index]
            .indices
            .iter()
            .zip(&self.expanded_rows[query_index].values)
        {
            for &(row, row_value) in &self.postings[column] {
                minima[row] += query_value.min(row_value);
            }
        }

        let mut scores = Vec::with_capacity(original_distances.len());
        for (gallery_index, original_distance) in original_distances.iter().copied().enumerate() {
            let sample_index = self.query_count + gallery_index;
            let intersection = minima[sample_index];
            let jaccard = 1.0 - intersection / (2.0 - intersection).max(1e-12);
            let normalized_original = original_distance / self.max_distances[query_index];
            scores.push((1.0 - lambda_value) * jaccard + lambda_value * normalized_original);
        }
        let order = rerank_order(&scores);

        let mut valid_rank = 0_usize;
        let mut hit_count = 0_usize;
        let mut precision_sum = 0.0_f64;
        let mut first_match_rank = None;
        for gallery_index in order {
            let same_identity = gallery_ids[gallery_index] == query_id;
            if same_identity && gallery_cams[gallery_index] == query_cam {
                continue;
            }
            if same_identity {
                hit_count += 1;
                precision_sum += hit_count as f64 / (valid_rank + 1) as f64;
                first_match_rank.get_or_insert(valid_rank);
            }
            valid_rank += 1;
        }
        Ok(QueryMetrics {
            average_precision: if hit_count == 0 {
                0.0
            } else {
                precision_sum / hit_count as f64
            },
            first_match_rank,
        })
    }
}

fn required_neighbor_count(k1: usize, k2: usize, sample_count: usize) -> usize {
    let expansion_count = bankers_round_half(k1).max(1) + 1;
    (k1 + 1).max(k2).max(expansion_count).min(sample_count)
}

fn bankers_round_half(value: usize) -> usize {
    let quotient = value / 2;
    if value % 2 == 0 || quotient % 2 == 0 {
        quotient
    } else {
        quotient + 1
    }
}

fn build_expanded_sets(
    top: &[usize],
    top_k: usize,
    k1: usize,
    sample_count: usize,
) -> Vec<Vec<usize>> {
    let neighbor_count = (k1 + 1).min(sample_count);
    let expansion_count = (bankers_round_half(k1).max(1) + 1).min(sample_count);
    (0..sample_count)
        .into_par_iter()
        .map(|index| {
            let reciprocal = reciprocal_indices(top, top_k, index, neighbor_count);
            let base: HashSet<usize> = reciprocal.iter().copied().collect();
            let mut selected: BTreeSet<usize> = reciprocal.iter().copied().collect();
            for candidate in reciprocal {
                let candidate_reciprocal =
                    reciprocal_indices(top, top_k, candidate, expansion_count);
                let overlap = candidate_reciprocal
                    .iter()
                    .filter(|item| base.contains(item))
                    .count();
                if (overlap as f64) > (2.0 / 3.0) * candidate_reciprocal.len() as f64 {
                    selected.extend(candidate_reciprocal);
                }
            }
            selected.into_iter().collect()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn build_sparse_index(
    top: &[usize],
    top_k: usize,
    max_distances: &[f32],
    offsets: &[usize],
    edge_indices: &[usize],
    edge_distances: &[f32],
    k2: usize,
) -> Result<SparseIndexParts, String> {
    let sample_count = max_distances.len();
    let affinity: Vec<SparseRow> = (0..sample_count)
        .into_par_iter()
        .map(|index| {
            let start = offsets[index];
            let end = offsets[index + 1];
            let denominator = max_distances[index].max(1e-12);
            let indices = edge_indices[start..end].to_vec();
            let mut values: Vec<f32> = edge_distances[start..end]
                .iter()
                .map(|distance| (-distance / denominator).exp())
                .collect();
            let sum = values.iter().sum::<f32>().max(1e-12);
            values.iter_mut().for_each(|value| *value /= sum);
            Ok(SparseRow { indices, values })
        })
        .collect::<Result<_, String>>()?;

    let expanded_rows = if k2 <= 1 {
        affinity
    } else {
        let query_neighbor_count = k2.min(sample_count);
        (0..sample_count)
            .into_par_iter()
            .map(|index| {
                let mut accumulated: HashMap<usize, f32> = HashMap::new();
                for neighbor in top[index * top_k..index * top_k + query_neighbor_count]
                    .iter()
                    .copied()
                {
                    for (&column, &value) in affinity[neighbor]
                        .indices
                        .iter()
                        .zip(&affinity[neighbor].values)
                    {
                        *accumulated.entry(column).or_insert(0.0) += value;
                    }
                }
                let mut entries: Vec<(usize, f32)> = accumulated.into_iter().collect();
                entries.sort_unstable_by_key(|(column, _)| *column);
                SparseRow {
                    indices: entries.iter().map(|(column, _)| *column).collect(),
                    values: entries
                        .into_iter()
                        .map(|(_, value)| value / query_neighbor_count as f32)
                        .collect(),
                }
            })
            .collect()
    };

    let mut postings = vec![Vec::new(); sample_count];
    for (row_index, row) in expanded_rows.iter().enumerate() {
        for (&column, &value) in row.indices.iter().zip(&row.values) {
            postings[column].push((row_index, value));
        }
    }
    Ok((expanded_rows, postings))
}

fn validated_top_indices(
    top_indices: &PyReadonlyArray2<'_, i64>,
    sample_count: usize,
) -> PyResult<Vec<usize>> {
    top_indices
        .as_slice()
        .map_err(|_| PyValueError::new_err("top_indices must be C-contiguous int64"))?
        .iter()
        .map(|value| {
            usize::try_from(*value)
                .ok()
                .filter(|index| *index < sample_count)
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "top_indices contains out-of-range index {value}"
                    ))
                })
        })
        .collect()
}

fn validated_offsets(
    offsets: &PyReadonlyArray1<'_, i64>,
    sample_count: usize,
) -> PyResult<Vec<usize>> {
    let values = offsets
        .as_slice()
        .map_err(|_| PyValueError::new_err("affinity_offsets must be contiguous int64"))?;
    if values.len() != sample_count + 1 {
        return Err(PyValueError::new_err(format!(
            "affinity_offsets must contain {} values",
            sample_count + 1
        )));
    }
    let converted: Vec<usize> = values
        .iter()
        .map(|value| {
            usize::try_from(*value)
                .map_err(|_| PyValueError::new_err("affinity_offsets must be non-negative"))
        })
        .collect::<PyResult<_>>()?;
    if converted[0] != 0 || converted.windows(2).any(|window| window[0] > window[1]) {
        return Err(PyValueError::new_err(
            "affinity_offsets must start at zero and be non-decreasing",
        ));
    }
    Ok(converted)
}

fn reciprocal_indices(top: &[usize], top_k: usize, index: usize, count: usize) -> Vec<usize> {
    top[index * top_k..index * top_k + count]
        .iter()
        .copied()
        .filter(|candidate| top[*candidate * top_k..*candidate * top_k + count].contains(&index))
        .collect()
}

fn require_length(values: &[i64], expected: usize, name: &str) -> PyResult<()> {
    if values.len() != expected {
        return Err(PyValueError::new_err(format!(
            "{name} length {} does not match expected {expected}",
            values.len()
        )));
    }
    Ok(())
}

fn rerank_order(scores: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_unstable_by(|left, right| {
        canonical_float(scores[*left])
            .total_cmp(&canonical_float(scores[*right]))
            .then_with(|| left.cmp(right))
    });
    let mut start = 0;
    while start < order.len() {
        let anchor = scores[order[start]];
        let mut end = start + 1;
        while end < order.len() && scores[order[end]] - anchor <= RERANK_TIE_TOLERANCE {
            end += 1;
        }
        order[start..end].sort_unstable();
        start = end;
    }
    order
}

fn canonical_float(value: f32) -> f32 {
    if value == 0.0 { 0.0 } else { value }
}

#[cfg(test)]
mod tests {
    use super::{bankers_round_half, required_neighbor_count};

    #[test]
    fn half_k_uses_python_bankers_rounding() {
        assert_eq!(bankers_round_half(1), 0);
        assert_eq!(bankers_round_half(3), 2);
        assert_eq!(bankers_round_half(5), 2);
        assert_eq!(bankers_round_half(7), 4);
    }

    #[test]
    fn required_neighbors_cover_k1_and_k2() {
        assert_eq!(required_neighbor_count(20, 6, 100), 21);
        assert_eq!(required_neighbor_count(1, 8, 5), 5);
    }
}
