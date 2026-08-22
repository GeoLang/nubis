use crate::{Classification, Point3, PointCloud};

/// Simple ground filter: classify points whose Z is within `threshold`
/// of the lowest point in their local neighbourhood as ground.
///
/// Single-pass minimum-Z per cell, then a height threshold. Not PMF.
pub fn ground_filter_simple(cloud: &mut PointCloud, cell_size: f64, threshold: f64) {
    if cloud.is_empty() {
        return;
    }

    let (min, max) = cloud.bounds().unwrap();
    let cols = ((max.x - min.x) / cell_size).ceil() as usize + 1;
    let rows = ((max.y - min.y) / cell_size).ceil() as usize + 1;

    // Find minimum Z in each grid cell
    let mut grid_min = vec![f64::MAX; cols * rows];
    for p in cloud.points() {
        let col = ((p.x - min.x) / cell_size) as usize;
        let row = ((p.y - min.y) / cell_size) as usize;
        let idx = row * cols + col;
        if idx < grid_min.len() {
            grid_min[idx] = grid_min[idx].min(p.z);
        }
    }

    // Classify: if point Z is within threshold of cell minimum → ground
    for p in cloud.points_mut() {
        let col = ((p.x - min.x) / cell_size) as usize;
        let row = ((p.y - min.y) / cell_size) as usize;
        let idx = row * cols + col;
        if idx < grid_min.len() && (p.z - grid_min[idx]).abs() <= threshold {
            p.classification = Classification::Ground;
        }
    }
}

/// Window sizes grow as `2 * base^exponent + 1` cells.
const WINDOW_SIZE_BASE: usize = 2;
const FIRST_WINDOW_EXPONENT: u32 = 0;

/// Settings for [`ground_filter_pmf`].
///
/// `max_window_size` and the two distances are in the cloud's coordinate units,
/// `slope` is a rise over run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PmfParams {
    pub cell_size: f64,
    pub max_window_size: f64,
    pub slope: f64,
    pub initial_distance: f64,
    pub max_distance: f64,
}

impl Default for PmfParams {
    fn default() -> Self {
        Self {
            cell_size: 1.0,
            max_window_size: 33.0,
            slope: 0.15,
            initial_distance: 0.5,
            max_distance: 3.0,
        }
    }
}

/// Progressive morphological ground filter (Zhang et al. 2003).
///
/// Builds a minimum-Z surface over the cloud, then opens it with windows that grow
/// until `max_window_size`. Each window removes whatever rises above the opened
/// surface by more than the elevation threshold for that window, so small objects go
/// first and buildings go once a window is wider than they are. Points still close to
/// the surface at the end become [`Classification::Ground`], everything else keeps the
/// classification it came in with.
pub fn ground_filter_pmf(cloud: &mut PointCloud, params: &PmfParams) {
    if cloud.is_empty() || params.cell_size <= 0.0 {
        return;
    }

    let (min, max) = cloud.bounds().unwrap();
    let columns = ((max.x - min.x) / params.cell_size).ceil() as usize + 1;
    let rows = ((max.y - min.y) / params.cell_size).ceil() as usize + 1;

    let mut surface: Vec<Option<f64>> = vec![None; columns * rows];
    for point in cloud.points() {
        let Some(index) = cell_index(point, &min, params.cell_size, columns, rows) else {
            continue;
        };
        surface[index] = Some(match surface[index] {
            Some(lowest) => lowest.min(point.z),
            None => point.z,
        });
    }

    let mut threshold = params.initial_distance;
    let mut previous_window_size = 0;
    let mut exponent = FIRST_WINDOW_EXPONENT;
    loop {
        let window_size = window_size_in_cells(exponent);
        if window_size as f64 * params.cell_size > params.max_window_size {
            break;
        }
        if previous_window_size > 0 {
            threshold =
                (params.slope * (window_size - previous_window_size) as f64 * params.cell_size
                    + params.initial_distance)
                    .min(params.max_distance);
        }

        let opened = morphological_opening(&surface, columns, rows, window_size);
        for (cell, opened_height) in surface.iter_mut().zip(&opened) {
            let (Some(height), Some(opened_height)) = (*cell, *opened_height) else {
                continue;
            };
            if height - opened_height > threshold {
                *cell = Some(opened_height);
            }
        }

        previous_window_size = window_size;
        exponent += 1;
    }

    for point in cloud.points_mut() {
        let Some(index) = cell_index(point, &min, params.cell_size, columns, rows) else {
            continue;
        };
        let Some(height) = surface[index] else {
            continue;
        };
        if point.z - height <= threshold {
            point.classification = Classification::Ground;
        }
    }
}

/// Odd so the window is centred on a cell.
fn window_size_in_cells(exponent: u32) -> usize {
    2 * WINDOW_SIZE_BASE.pow(exponent) + 1
}

fn cell_index(
    point: &Point3,
    min: &Point3,
    cell_size: f64,
    columns: usize,
    rows: usize,
) -> Option<usize> {
    let column = ((point.x - min.x) / cell_size) as usize;
    let row = ((point.y - min.y) / cell_size) as usize;
    (column < columns && row < rows).then_some(row * columns + column)
}

fn morphological_opening(
    surface: &[Option<f64>],
    columns: usize,
    rows: usize,
    window_size: usize,
) -> Vec<Option<f64>> {
    let radius = window_size / 2;
    let eroded = window_extreme(surface, columns, rows, radius, f64::min);
    window_extreme(&eroded, columns, rows, radius, f64::max)
}

/// Min or max over a square window, run as a row sweep then a column sweep, which gives
/// the same answer as the square and costs a lot less. Empty cells take part in neither.
fn window_extreme(
    surface: &[Option<f64>],
    columns: usize,
    rows: usize,
    radius: usize,
    pick: fn(f64, f64) -> f64,
) -> Vec<Option<f64>> {
    let mut swept_rows = vec![None; surface.len()];
    for row in 0..rows {
        let start = row * columns;
        for column in 0..columns {
            let first = start + column.saturating_sub(radius);
            let last = start + (column + radius).min(columns - 1);
            swept_rows[start + column] =
                surface[first..=last].iter().flatten().copied().reduce(pick);
        }
    }

    let mut swept_columns = vec![None; surface.len()];
    for column in 0..columns {
        for row in 0..rows {
            let first = row.saturating_sub(radius);
            let last = (row + radius).min(rows - 1);
            swept_columns[row * columns + column] = (first..=last)
                .filter_map(|near| swept_rows[near * columns + column])
                .reduce(pick);
        }
    }
    swept_columns
}

/// Thinning: keep `fraction` of the points, spread evenly through the cloud.
///
/// Deterministic, so the same input always yields the same output.
pub fn thin_random(cloud: &PointCloud, fraction: f64) -> PointCloud {
    let n = cloud.len();
    let keep = (n as f64 * fraction.clamp(0.0, 1.0)).round() as usize;

    if keep == 0 {
        return PointCloud::new();
    }
    if keep >= n {
        return cloud.clone();
    }

    // an integer stride can only express 1/1, 1/2, 1/3..., which rounds every
    // fraction above a half up to "keep everything", so pick indices instead
    let points: Vec<Point3> = (0..keep).map(|i| cloud.points()[i * n / keep]).collect();
    PointCloud::from_points(points)
}

/// Voxel thinning: keep one point per voxel of the given size.
pub fn thin_voxel(cloud: &PointCloud, voxel_size: f64) -> PointCloud {
    use std::collections::HashMap;

    if cloud.is_empty() || voxel_size <= 0.0 {
        return cloud.clone();
    }

    let mut voxels: HashMap<(i64, i64, i64), Point3> = HashMap::new();
    for p in cloud.points() {
        let vx = (p.x / voxel_size).floor() as i64;
        let vy = (p.y / voxel_size).floor() as i64;
        let vz = (p.z / voxel_size).floor() as i64;
        voxels.entry((vx, vy, vz)).or_insert(*p);
    }

    // hash order varies between runs, so sort by voxel to keep the output stable
    let mut kept: Vec<((i64, i64, i64), Point3)> = voxels.into_iter().collect();
    kept.sort_unstable_by_key(|(voxel, _)| *voxel);

    PointCloud::from_points(kept.into_iter().map(|(_, p)| p).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_cloud() -> PointCloud {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.5, 0.5, 0.1),
            Point3::new(1.0, 1.0, 5.0), // high point
            Point3::new(1.5, 1.5, 0.2),
            Point3::new(2.0, 2.0, 0.3),
            Point3::new(2.5, 2.5, 8.0), // high point
        ];
        PointCloud::from_points(points)
    }

    #[test]
    fn test_ground_filter() {
        let mut cloud = sample_cloud();
        ground_filter_simple(&mut cloud, 2.0, 0.5);
        // Low points should be classified as ground
        assert_eq!(cloud.points()[0].classification, Classification::Ground);
        assert_eq!(cloud.points()[1].classification, Classification::Ground);
        // High points should remain unclassified
        assert_eq!(
            cloud.points()[2].classification,
            Classification::Unclassified
        );
    }

    #[test]
    fn test_thin_random() {
        let cloud = sample_cloud();
        let thinned = thin_random(&cloud, 0.5);
        assert!(thinned.len() <= cloud.len());
        assert!(!thinned.is_empty());
    }

    #[test]
    fn test_thin_voxel() {
        let cloud = PointCloud::from_points(vec![
            Point3::new(0.1, 0.1, 0.1),
            Point3::new(0.2, 0.2, 0.2),
            Point3::new(5.0, 5.0, 5.0),
        ]);
        let thinned = thin_voxel(&cloud, 1.0);
        // First two points are in same voxel, third is separate
        assert_eq!(thinned.len(), 2);
    }
}
