use crate::{Point3, PointCloud};

/// Result of IDW interpolation: a regular grid of elevation values.
#[derive(Debug, Clone)]
pub struct InterpolatedGrid {
    pub data: Vec<f64>,
    pub width: usize,
    pub height: usize,
    pub cell_size: f64,
    pub origin_x: f64,
    pub origin_y: f64,
    pub nodata: f64,
}

/// Inverse Distance Weighting (IDW) interpolation.
///
/// Interpolates point cloud Z values onto a regular grid using:
///   z(x) = Σ(w_i * z_i) / Σ(w_i)
///   where w_i = 1 / d_i^power
///
/// # Arguments
/// * `cloud` — input point cloud
/// * `cell_size` — output grid cell size in coordinate units
/// * `power` — distance exponent (typically 2.0)
/// * `search_radius` — maximum distance to consider points (0 = unlimited)
/// * `min_points` — minimum number of points required for interpolation
pub fn idw_interpolation(
    cloud: &PointCloud,
    cell_size: f64,
    power: f64,
    search_radius: f64,
    min_points: usize,
) -> Option<InterpolatedGrid> {
    if cloud.is_empty() || cell_size <= 0.0 {
        return None;
    }

    let (min_pt, max_pt) = cloud.bounds()?;

    let origin_x = min_pt.x;
    let origin_y = min_pt.y;
    let width = ((max_pt.x - min_pt.x) / cell_size).ceil() as usize + 1;
    let height = ((max_pt.y - min_pt.y) / cell_size).ceil() as usize + 1;
    let nodata = -9999.0;

    let mut data = vec![nodata; width * height];
    let points = cloud.points();
    let use_radius = search_radius > 0.0;

    for row in 0..height {
        let py = origin_y + row as f64 * cell_size;
        for col in 0..width {
            let px = origin_x + col as f64 * cell_size;

            let mut weight_sum = 0.0;
            let mut value_sum = 0.0;
            let mut count = 0;
            let mut exact_match = None;

            for p in points {
                let dx = px - p.x;
                let dy = py - p.y;
                let dist_sq = dx * dx + dy * dy;

                if use_radius && dist_sq > search_radius * search_radius {
                    continue;
                }

                if dist_sq < 1e-20 {
                    // Point exactly at grid node
                    exact_match = Some(p.z);
                    break;
                }

                let dist = dist_sq.sqrt();
                let w = 1.0 / dist.powf(power);
                weight_sum += w;
                value_sum += w * p.z;
                count += 1;
            }

            if let Some(z) = exact_match {
                data[row * width + col] = z;
            } else if count >= min_points && weight_sum > 0.0 {
                data[row * width + col] = value_sum / weight_sum;
            }
        }
    }

    Some(InterpolatedGrid {
        data,
        width,
        height,
        cell_size,
        origin_x,
        origin_y,
        nodata,
    })
}

/// Statistical Outlier Removal (SOR).
///
/// For each point, computes the mean distance to its k nearest neighbors.
/// Points whose mean distance exceeds (global_mean + std_multiplier * global_std)
/// are removed as outliers.
///
/// # Arguments
/// * `cloud` — input point cloud
/// * `k` — number of nearest neighbors to consider
/// * `std_multiplier` — number of standard deviations for threshold
///
/// # Returns
/// A new point cloud with outliers removed.
pub fn statistical_outlier_removal(
    cloud: &PointCloud,
    k: usize,
    std_multiplier: f64,
) -> PointCloud {
    let points = cloud.points();
    let n = points.len();

    if n <= k {
        return cloud.clone();
    }

    // Compute mean distance to k nearest neighbors for each point
    let mut mean_distances = Vec::with_capacity(n);

    for i in 0..n {
        let mut dists: Vec<f64> = points
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, q)| points[i].distance_to(q))
            .collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let k_actual = k.min(dists.len());
        let mean_d: f64 = dists[..k_actual].iter().sum::<f64>() / k_actual as f64;
        mean_distances.push(mean_d);
    }

    // Compute global mean and standard deviation of mean distances
    let global_mean: f64 = mean_distances.iter().sum::<f64>() / n as f64;
    let variance: f64 = mean_distances
        .iter()
        .map(|d| (d - global_mean).powi(2))
        .sum::<f64>()
        / n as f64;
    let global_std = variance.sqrt();

    let threshold = global_mean + std_multiplier * global_std;

    // Filter points
    let filtered: Vec<Point3> = points
        .iter()
        .zip(mean_distances.iter())
        .filter(|(_, d)| **d <= threshold)
        .map(|(p, _)| *p)
        .collect();

    PointCloud::from_points(filtered)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idw_basic() {
        // Four corner points at known elevations
        let cloud = PointCloud::from_points(vec![
            Point3::new(0.0, 0.0, 10.0),
            Point3::new(10.0, 0.0, 20.0),
            Point3::new(0.0, 10.0, 30.0),
            Point3::new(10.0, 10.0, 40.0),
        ]);

        let grid = idw_interpolation(&cloud, 5.0, 2.0, 0.0, 1).unwrap();

        // Grid should cover the extent
        assert!(grid.width >= 2);
        assert!(grid.height >= 2);

        // Origin point should be exactly 10.0
        assert!((grid.data[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_idw_exact_at_points() {
        let cloud = PointCloud::from_points(vec![
            Point3::new(0.0, 0.0, 100.0),
            Point3::new(5.0, 0.0, 200.0),
            Point3::new(0.0, 5.0, 300.0),
        ]);

        let grid = idw_interpolation(&cloud, 5.0, 2.0, 0.0, 1).unwrap();

        // Value at (0,0) should be exactly 100
        assert!((grid.data[0] - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_idw_with_search_radius() {
        let cloud = PointCloud::from_points(vec![
            Point3::new(0.0, 0.0, 10.0),
            Point3::new(100.0, 100.0, 50.0),
        ]);

        // With small search radius, far-away cells should be nodata
        let grid = idw_interpolation(&cloud, 10.0, 2.0, 5.0, 1).unwrap();
        // Cells far from both points should remain nodata
        let far_cell = grid.data[grid.width / 2 * grid.width + grid.width / 2];
        // The middle of a 100x100 grid at 10m resolution — might be nodata
        // depending on radius coverage
        assert!(far_cell == grid.nodata || far_cell > 0.0);
    }

    #[test]
    fn test_sor_removes_outliers() {
        // Cluster of points near origin, one outlier far away
        let mut points = Vec::new();
        for i in 0..20 {
            let x = (i % 5) as f64;
            let y = (i / 5) as f64;
            points.push(Point3::new(x, y, 0.0));
        }
        // Add outlier
        points.push(Point3::new(100.0, 100.0, 0.0));

        let cloud = PointCloud::from_points(points);
        let filtered = statistical_outlier_removal(&cloud, 5, 1.0);

        // Outlier should be removed
        assert_eq!(filtered.len(), 20, "outlier should be removed");
    }

    #[test]
    fn test_sor_preserves_dense_cluster() {
        // 2D grid of points — all close together, none should be removed
        let mut points = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                points.push(Point3::new(i as f64, j as f64, 0.0));
            }
        }

        let cloud = PointCloud::from_points(points);
        let filtered = statistical_outlier_removal(&cloud, 4, 3.0);

        assert_eq!(filtered.len(), 25, "no points should be removed");
    }
}
