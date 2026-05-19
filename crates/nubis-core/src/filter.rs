use crate::{Classification, Point3, PointCloud};

/// Simple ground filter: classify points whose Z is within `threshold`
/// of the lowest point in their local neighbourhood as ground.
///
/// This is a simplified progressive morphological filter approach.
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

/// Random thinning: keep approximately `fraction` of points.
pub fn thin_random(cloud: &PointCloud, fraction: f64) -> PointCloud {
    // Use a simple deterministic hash-based approach for reproducibility
    let keep = (cloud.len() as f64 * fraction.clamp(0.0, 1.0)) as usize;
    let step = if keep == 0 {
        return PointCloud::new();
    } else {
        cloud.len() / keep
    };

    let points: Vec<Point3> = cloud
        .points()
        .iter()
        .step_by(step.max(1))
        .copied()
        .collect();
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

    PointCloud::from_points(voxels.into_values().collect())
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
