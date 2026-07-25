use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug)]
struct QueryMetrics {
    average_precision: f64,
    first_match_rank: Option<usize>,
}

#[pyfunction]
#[pyo3(signature = (scores, query_ids, gallery_ids, query_cams, gallery_cams, ranks))]
pub fn evaluate_scores(
    py: Python<'_>,
    scores: PyReadonlyArray2<'_, f32>,
    query_ids: PyReadonlyArray1<'_, i64>,
    gallery_ids: PyReadonlyArray1<'_, i64>,
    query_cams: PyReadonlyArray1<'_, i64>,
    gallery_cams: PyReadonlyArray1<'_, i64>,
    ranks: Vec<usize>,
) -> PyResult<(f64, Vec<u64>, usize)> {
    let shape = scores.shape();
    let query_count = shape[0];
    let gallery_count = shape[1];
    if query_count == 0 {
        return Err(PyValueError::new_err(
            "scores must contain at least one query row",
        ));
    }
    if gallery_count == 0 {
        return Err(PyValueError::new_err(
            "scores must contain at least one gallery column",
        ));
    }
    if ranks.is_empty() || ranks.contains(&0) {
        return Err(PyValueError::new_err(
            "ranks must contain positive integers",
        ));
    }

    let score_values = scores
        .as_slice()
        .map_err(|_| PyValueError::new_err("scores must be C-contiguous float32"))?
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

    require_length(&query_id_values, query_count, "query_ids")?;
    require_length(&query_cam_values, query_count, "query_cams")?;
    require_length(&gallery_id_values, gallery_count, "gallery_ids")?;
    require_length(&gallery_cam_values, gallery_count, "gallery_cams")?;

    let rows = py.detach(|| {
        (0..query_count)
            .into_par_iter()
            .map(|query_index| {
                let offset = query_index * gallery_count;
                evaluate_query(
                    &score_values[offset..offset + gallery_count],
                    query_id_values[query_index],
                    &gallery_id_values,
                    query_cam_values[query_index],
                    &gallery_cam_values,
                    query_index,
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
            for (index, rank) in ranks.iter().enumerate() {
                if first_match_rank < *rank {
                    cmc_counts[index] += 1;
                }
            }
        }
    }
    Ok((average_precision_sum, cmc_counts, query_count))
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

fn evaluate_query(
    scores: &[f32],
    query_id: i64,
    gallery_ids: &[i64],
    query_cam: i64,
    gallery_cams: &[i64],
    query_index: usize,
) -> Result<QueryMetrics, String> {
    if let Some((gallery_index, score)) = scores
        .iter()
        .copied()
        .enumerate()
        .find(|(_, score)| !score.is_finite())
    {
        return Err(format!(
            "scores contain a non-finite value at query {query_index}, gallery {gallery_index}: {score}"
        ));
    }

    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_unstable_by(|left, right| {
        canonical_score(scores[*right])
            .total_cmp(&canonical_score(scores[*left]))
            .then_with(|| left.cmp(right))
    });

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

fn canonical_score(value: f32) -> f32 {
    if value == 0.0 { 0.0 } else { value }
}

#[cfg(test)]
mod tests {
    use super::evaluate_query;

    #[test]
    fn excludes_same_identity_same_camera() {
        let result = evaluate_query(&[1.0, 0.0], 1, &[1, 2], 1, &[1, 2], 0).unwrap();
        assert_eq!(result.average_precision, 0.0);
        assert_eq!(result.first_match_rank, None);
    }

    #[test]
    fn signed_zero_scores_use_index_tie_breaking() {
        let result = evaluate_query(&[-0.0, 0.0], 1, &[1, 2], 0, &[1, 1], 0).unwrap();
        assert_eq!(result.average_precision, 1.0);
        assert_eq!(result.first_match_rank, Some(0));
    }

    #[test]
    fn uses_gallery_index_to_break_score_ties() {
        let result = evaluate_query(&[0.5, 0.5], 1, &[2, 1], 1, &[2, 2], 0).unwrap();
        assert_eq!(result.average_precision, 0.5);
        assert_eq!(result.first_match_rank, Some(1));
    }
}
