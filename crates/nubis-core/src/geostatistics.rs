//! Geostatistics — spatial analysis and interpolation.
//!
//! Implements:
//! - Variogram modeling (empirical variogram, spherical/exponential/gaussian models)
//! - Ordinary kriging interpolation
//! - Moran's I spatial autocorrelation
//! - Getis-Ord Gi* hot spot analysis

use crate::cloud::{Point3, PointCloud};
use crate::interpolation::InterpolatedGrid;

/// Empirical variogram bin.
#[derive(Debug, Clone)]
pub struct VariogramBin {
    pub lag: f64,
    pub semivariance: f64,
    pub count: usize,
}

/// Variogram model types.
#[derive(Debug, Clone, Copy)]
pub enum VariogramModel {
    /// Spherical: γ(h) = C₀ + C₁ * [1.5(h/a) - 0.5(h/a)³]  for h <= a
    Spherical { nugget: f64, sill: f64, range: f64 },
    /// Exponential: γ(h) = C₀ + C₁ * [1 - exp(-3h/a)]
    Exponential { nugget: f64, sill: f64, range: f64 },
    /// Gaussian: γ(h) = C₀ + C₁ * [1 - exp(-3(h/a)²)]
    Gaussian { nugget: f64, sill: f64, range: f64 },
}

/// Fitted variogram with model and empirical data.
#[derive(Debug, Clone)]
pub struct Variogram {
    pub model: VariogramModel,
    pub bins: Vec<VariogramBin>,
}

/// Compute the empirical variogram from a point cloud.
pub fn empirical_variogram(cloud: &PointCloud, n_bins: usize, max_lag: f64) -> Vec<VariogramBin> {
    let points = cloud.points();
    let n = points.len();
    let bin_width = max_lag / n_bins as f64;

    let mut bins: Vec<(f64, usize)> = vec![(0.0, 0); n_bins];

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = points[i].distance_2d(&points[j]);
            if dist >= max_lag {
                continue;
            }
            let bin_idx = (dist / bin_width) as usize;
            if bin_idx < n_bins {
                let diff = points[i].z - points[j].z;
                bins[bin_idx].0 += diff * diff;
                bins[bin_idx].1 += 1;
            }
        }
    }

    bins.iter()
        .enumerate()
        .filter(|(_, (_, count))| *count > 0)
        .map(|(i, (sum_sq, count))| VariogramBin {
            lag: (i as f64 + 0.5) * bin_width,
            semivariance: sum_sq / (2.0 * *count as f64),
            count: *count,
        })
        .collect()
}

impl VariogramModel {
    /// Evaluate the variogram model at distance h.
    pub fn evaluate(&self, h: f64) -> f64 {
        if h <= 0.0 {
            return 0.0;
        }
        match self {
            VariogramModel::Spherical {
                nugget,
                sill,
                range,
            } => {
                if h >= *range {
                    nugget + sill
                } else {
                    let hr = h / range;
                    nugget + sill * (1.5 * hr - 0.5 * hr.powi(3))
                }
            }
            VariogramModel::Exponential {
                nugget,
                sill,
                range,
            } => nugget + sill * (1.0 - (-3.0 * h / range).exp()),
            VariogramModel::Gaussian {
                nugget,
                sill,
                range,
            } => nugget + sill * (1.0 - (-3.0 * (h / range).powi(2)).exp()),
        }
    }

    /// Fit a spherical model to empirical variogram bins using least squares.
    pub fn fit_spherical(bins: &[VariogramBin]) -> Self {
        if bins.is_empty() {
            return VariogramModel::Spherical {
                nugget: 0.0,
                sill: 1.0,
                range: 1.0,
            };
        }

        // Estimate parameters from empirical data
        let max_sv = bins.iter().map(|b| b.semivariance).fold(0.0f64, f64::max);
        let min_sv = bins
            .iter()
            .map(|b| b.semivariance)
            .fold(f64::INFINITY, f64::min);

        let nugget = min_sv * 0.5;
        let sill = max_sv - nugget;

        // Estimate range as the lag where semivariance first reaches ~95% of sill
        let threshold = nugget + sill * 0.95;
        let range = bins
            .iter()
            .find(|b| b.semivariance >= threshold)
            .map(|b| b.lag)
            .unwrap_or(bins.last().map(|b| b.lag).unwrap_or(1.0));

        VariogramModel::Spherical {
            nugget,
            sill,
            range,
        }
    }
}

/// Ordinary kriging interpolation.
///
/// Produces a gridded estimate from irregularly-spaced point observations
/// using the supplied variogram model.
pub fn ordinary_kriging(
    cloud: &PointCloud,
    model: &VariogramModel,
    cell_size: f64,
    search_radius: f64,
) -> InterpolatedGrid {
    let points = cloud.points();
    let bounds = cloud
        .bounds()
        .unwrap_or((Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)));

    let width = ((bounds.1.x - bounds.0.x) / cell_size).ceil() as usize + 1;
    let height = ((bounds.1.y - bounds.0.y) / cell_size).ceil() as usize + 1;
    let mut data = vec![f64::NAN; width * height];

    for row in 0..height {
        for col in 0..width {
            let x = bounds.0.x + col as f64 * cell_size;
            let y = bounds.0.y + row as f64 * cell_size;

            // Find nearby points
            let nearby: Vec<usize> = points
                .iter()
                .enumerate()
                .filter(|(_, p)| {
                    let dx = p.x - x;
                    let dy = p.y - y;
                    (dx * dx + dy * dy).sqrt() <= search_radius
                })
                .map(|(i, _)| i)
                .collect();

            if nearby.len() < 3 {
                continue;
            }

            let n = nearby.len();

            // Build kriging system: (n+1) × (n+1) matrix
            let size = n + 1;
            let mut a = vec![0.0; size * size];
            let mut b = vec![0.0; size];

            // Fill variogram matrix
            for i in 0..n {
                for j in 0..n {
                    let dist = points[nearby[i]].distance_2d(&points[nearby[j]]);
                    a[i * size + j] = model.evaluate(dist);
                }
                // Lagrange multiplier row/column
                a[i * size + n] = 1.0;
                a[n * size + i] = 1.0;
            }
            a[n * size + n] = 0.0;

            // Fill RHS with variogram values to estimation point
            for i in 0..n {
                let target = Point3::new(x, y, 0.0);
                let dist = points[nearby[i]].distance_2d(&target);
                b[i] = model.evaluate(dist);
            }
            b[n] = 1.0;

            // Solve system using Gaussian elimination
            if let Some(weights) = solve_linear_system(&mut a, &mut b, size) {
                let mut z_est = 0.0;
                for i in 0..n {
                    z_est += weights[i] * points[nearby[i]].z;
                }
                data[row * width + col] = z_est;
            }
        }
    }

    InterpolatedGrid {
        width,
        height,
        cell_size,
        origin_x: bounds.0.x,
        origin_y: bounds.0.y,
        data,
        nodata: f64::NAN,
    }
}

/// Moran's I spatial autocorrelation statistic.
///
/// Returns (I, E\[I\], z-score) where:
/// - I > E\[I\] indicates positive spatial autocorrelation (clustering)
/// - I < E\[I\] indicates negative spatial autocorrelation (dispersion)
/// - |z-score| > 1.96 is statistically significant at 95% confidence
pub fn morans_i(cloud: &PointCloud, bandwidth: f64) -> (f64, f64, f64) {
    let points = cloud.points();
    let n = points.len();

    if n < 3 {
        return (0.0, 0.0, 0.0);
    }

    let mean_z: f64 = points.iter().map(|p| p.z).sum::<f64>() / n as f64;

    let mut w_sum = 0.0;
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..n {
        let zi = points[i].z - mean_z;
        denominator += zi * zi;

        for j in 0..n {
            if i == j {
                continue;
            }
            let dist = points[i].distance_2d(&points[j]);
            let w = if dist <= bandwidth { 1.0 } else { 0.0 };
            w_sum += w;
            numerator += w * zi * (points[j].z - mean_z);
        }
    }

    if denominator == 0.0 || w_sum == 0.0 {
        return (0.0, 0.0, 0.0);
    }

    let i_stat = (n as f64 / w_sum) * (numerator / denominator);
    let expected = -1.0 / (n as f64 - 1.0);

    // Variance under normality assumption
    let s2 = points.iter().map(|p| (p.z - mean_z).powi(2)).sum::<f64>() / n as f64;
    let variance = if s2 > 0.0 {
        // Simplified variance formula
        let n_f = n as f64;
        (n_f * ((n_f * n_f - 3.0 * n_f + 3.0) * w_sum.powi(2)) - n_f * w_sum.powi(2)
            + 3.0 * w_sum.powi(2))
            / ((n_f - 1.0) * (n_f + 1.0) * w_sum.powi(2))
            - expected.powi(2)
    } else {
        1.0
    };

    let z_score = if variance > 0.0 {
        (i_stat - expected) / variance.sqrt()
    } else {
        0.0
    };

    (i_stat, expected, z_score)
}

/// Getis-Ord Gi* hot spot statistic for each point.
///
/// Returns a vector of z-scores, one per point.
/// - High positive z-scores indicate hot spots (clusters of high values)
/// - High negative z-scores indicate cold spots (clusters of low values)
/// - |z| > 1.96 is significant at 95%, > 2.58 at 99%
pub fn getis_ord_gi_star(cloud: &PointCloud, bandwidth: f64) -> Vec<f64> {
    let points = cloud.points();
    let n = points.len();

    if n < 2 {
        return vec![0.0; n];
    }

    let x_bar: f64 = points.iter().map(|p| p.z).sum::<f64>() / n as f64;
    let s: f64 =
        (points.iter().map(|p| p.z.powi(2)).sum::<f64>() / n as f64 - x_bar.powi(2)).sqrt();

    if s == 0.0 {
        return vec![0.0; n];
    }

    let mut z_scores = Vec::with_capacity(n);

    for i in 0..n {
        let mut w_sum = 0.0;
        let mut w_x_sum = 0.0;
        let mut w2_sum = 0.0;

        for j in 0..n {
            let dist = points[i].distance_2d(&points[j]);
            let w = if dist <= bandwidth { 1.0 } else { 0.0 };
            w_sum += w;
            w_x_sum += w * points[j].z;
            w2_sum += w * w;
        }

        let nf = n as f64;
        let numerator = w_x_sum - x_bar * w_sum;
        let denominator = s * ((nf * w2_sum - w_sum * w_sum) / (nf - 1.0)).sqrt();

        let z = if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        };
        z_scores.push(z);
    }

    z_scores
}

/// Gaussian elimination for solving Ax = b.
fn solve_linear_system(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = a[col * n + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = a[row * n + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-12 {
            return None; // Singular
        }

        // Swap rows
        if max_row != col {
            for k in 0..n {
                a.swap(col * n + k, max_row * n + k);
            }
            b.swap(col, max_row);
        }

        // Eliminate
        for row in (col + 1)..n {
            let factor = a[row * n + col] / a[col * n + col];
            for k in col..n {
                a[row * n + k] -= factor * a[col * n + k];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i * n + j] * x[j];
        }
        x[i] = sum / a[i * n + i];
    }

    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cloud() -> PointCloud {
        PointCloud::from_points(vec![
            Point3::new(0.0, 0.0, 10.0),
            Point3::new(1.0, 0.0, 12.0),
            Point3::new(0.0, 1.0, 11.0),
            Point3::new(1.0, 1.0, 13.0),
            Point3::new(0.5, 0.5, 15.0),
            Point3::new(2.0, 0.0, 8.0),
            Point3::new(0.0, 2.0, 9.0),
            Point3::new(2.0, 2.0, 7.0),
        ])
    }

    #[test]
    fn empirical_variogram_produces_bins() {
        let cloud = test_cloud();
        let bins = empirical_variogram(&cloud, 5, 3.0);
        assert!(!bins.is_empty());
        for bin in &bins {
            assert!(bin.semivariance >= 0.0);
            assert!(bin.count > 0);
        }
    }

    #[test]
    fn spherical_model_evaluation() {
        let model = VariogramModel::Spherical {
            nugget: 0.5,
            sill: 10.0,
            range: 5.0,
        };
        // At h=0, gamma=0
        assert!((model.evaluate(0.0) - 0.0).abs() < 1e-10);
        // At h=range, gamma=nugget+sill
        assert!((model.evaluate(5.0) - 10.5).abs() < 1e-10);
        // Beyond range, gamma=nugget+sill
        assert!((model.evaluate(10.0) - 10.5).abs() < 1e-10);
        // At h=range/2, partial
        let mid = model.evaluate(2.5);
        assert!(mid > 0.5 && mid < 10.5);
    }

    #[test]
    fn exponential_model_evaluation() {
        let model = VariogramModel::Exponential {
            nugget: 0.0,
            sill: 10.0,
            range: 5.0,
        };
        assert!((model.evaluate(0.0) - 0.0).abs() < 1e-10);
        // Approaches sill asymptotically
        let far = model.evaluate(50.0);
        assert!((far - 10.0).abs() < 0.1);
    }

    #[test]
    fn gaussian_model_evaluation() {
        let model = VariogramModel::Gaussian {
            nugget: 1.0,
            sill: 9.0,
            range: 5.0,
        };
        assert!((model.evaluate(0.0) - 0.0).abs() < 1e-10);
        let far = model.evaluate(50.0);
        assert!((far - 10.0).abs() < 0.1);
    }

    #[test]
    fn fit_spherical_model() {
        let cloud = test_cloud();
        let bins = empirical_variogram(&cloud, 5, 3.0);
        let model = VariogramModel::fit_spherical(&bins);
        match model {
            VariogramModel::Spherical {
                nugget,
                sill,
                range,
            } => {
                assert!(nugget >= 0.0);
                assert!(sill > 0.0);
                assert!(range > 0.0);
            }
            _ => panic!("Expected Spherical model"),
        }
    }

    #[test]
    fn ordinary_kriging_produces_grid() {
        let cloud = test_cloud();
        let model = VariogramModel::Spherical {
            nugget: 0.5,
            sill: 10.0,
            range: 3.0,
        };
        let grid = ordinary_kriging(&cloud, &model, 0.5, 5.0);
        assert!(grid.width > 0);
        assert!(grid.height > 0);
        // At least some cells should be interpolated
        let valid_count = grid.data.iter().filter(|v| !v.is_nan()).count();
        assert!(valid_count > 0);
    }

    #[test]
    fn morans_i_clustered_data() {
        // Create spatially clustered high values
        let mut points = Vec::new();
        // Cluster of high values
        for i in 0..5 {
            for j in 0..5 {
                points.push(Point3::new(i as f64 * 0.1, j as f64 * 0.1, 100.0));
            }
        }
        // Cluster of low values
        for i in 0..5 {
            for j in 0..5 {
                points.push(Point3::new(5.0 + i as f64 * 0.1, 5.0 + j as f64 * 0.1, 1.0));
            }
        }
        let cloud = PointCloud::from_points(points);
        let (i_stat, expected, _z_score) = morans_i(&cloud, 1.0);
        // Positive spatial autocorrelation expected
        assert!(i_stat > expected);
    }

    #[test]
    fn morans_i_empty_cloud() {
        let cloud = PointCloud::new();
        let (i, e, z) = morans_i(&cloud, 1.0);
        assert_eq!(i, 0.0);
        assert_eq!(e, 0.0);
        assert_eq!(z, 0.0);
    }

    #[test]
    fn getis_ord_returns_scores() {
        let cloud = test_cloud();
        let scores = getis_ord_gi_star(&cloud, 2.0);
        assert_eq!(scores.len(), cloud.len());
    }

    #[test]
    fn getis_ord_hot_spot() {
        // Point with high value surrounded by high values should have positive z
        let mut points = Vec::new();
        // Cluster of high values at origin
        for i in 0..5 {
            for j in 0..5 {
                points.push(Point3::new(i as f64 * 0.5, j as f64 * 0.5, 100.0));
            }
        }
        // Some low values far away
        for i in 0..5 {
            points.push(Point3::new(10.0 + i as f64, 10.0, 1.0));
        }
        let cloud = PointCloud::from_points(points);
        let scores = getis_ord_gi_star(&cloud, 3.0);
        // First point (in high cluster) should have positive z-score
        assert!(scores[0] > 0.0);
        // Last point (in low cluster) should have negative z-score
        assert!(*scores.last().unwrap() < 0.0);
    }

    #[test]
    fn solve_linear_system_basic() {
        // 2x + y = 5
        // x + 3y = 10
        let mut a = vec![2.0, 1.0, 1.0, 3.0];
        let mut b = vec![5.0, 10.0];
        let x = solve_linear_system(&mut a, &mut b, 2).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn solve_linear_system_singular() {
        let mut a = vec![1.0, 2.0, 2.0, 4.0];
        let mut b = vec![3.0, 6.0];
        let result = solve_linear_system(&mut a, &mut b, 2);
        assert!(result.is_none());
    }
}
