// Comprehensive tests for nubis-core point cloud processing.

use nubis_core::*;

// ═══════════════════════════════════════════════════════════════════════════
// Point3 tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_point3_new() {
    let p = Point3::new(1.0, 2.0, 3.0);
    assert_eq!(p.x, 1.0);
    assert_eq!(p.y, 2.0);
    assert_eq!(p.z, 3.0);
    assert_eq!(p.intensity, 0);
    assert_eq!(p.classification, Classification::Unclassified);
}

#[test]
fn test_point3_with_classification() {
    let p = Point3::new(0.0, 0.0, 0.0).with_classification(Classification::Ground);
    assert_eq!(p.classification, Classification::Ground);
}

#[test]
fn test_point3_with_intensity() {
    let p = Point3::new(0.0, 0.0, 0.0).with_intensity(255);
    assert_eq!(p.intensity, 255);
}

#[test]
fn test_point3_distance_3d() {
    let a = Point3::new(0.0, 0.0, 0.0);
    let b = Point3::new(3.0, 4.0, 0.0);
    assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
}

#[test]
fn test_point3_distance_same_point() {
    let a = Point3::new(5.0, 5.0, 5.0);
    assert!((a.distance_to(&a) - 0.0).abs() < 1e-10);
}

#[test]
fn test_point3_distance_3d_full() {
    let a = Point3::new(1.0, 2.0, 3.0);
    let b = Point3::new(4.0, 6.0, 3.0);
    // sqrt(9 + 16 + 0) = 5
    assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
}

#[test]
fn test_point3_distance_2d() {
    let a = Point3::new(0.0, 0.0, 100.0);
    let b = Point3::new(3.0, 4.0, 200.0);
    // 2D distance ignores Z
    assert!((a.distance_2d(&b) - 5.0).abs() < 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// PointCloud tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_pointcloud_new_is_empty() {
    let cloud = PointCloud::new();
    assert!(cloud.is_empty());
    assert_eq!(cloud.len(), 0);
}

#[test]
fn test_pointcloud_push() {
    let mut cloud = PointCloud::new();
    cloud.push(Point3::new(1.0, 2.0, 3.0));
    assert_eq!(cloud.len(), 1);
    assert!(!cloud.is_empty());
}

#[test]
fn test_pointcloud_from_points() {
    let points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 1.0, 1.0),
        Point3::new(2.0, 2.0, 2.0),
    ];
    let cloud = PointCloud::from_points(points);
    assert_eq!(cloud.len(), 3);
}

#[test]
fn test_pointcloud_bounds() {
    let cloud = PointCloud::from_points(vec![
        Point3::new(-5.0, -3.0, 0.0),
        Point3::new(10.0, 8.0, 15.0),
        Point3::new(2.0, 2.0, 5.0),
    ]);
    let (min, max) = cloud.bounds().unwrap();
    assert_eq!(min.x, -5.0);
    assert_eq!(min.y, -3.0);
    assert_eq!(min.z, 0.0);
    assert_eq!(max.x, 10.0);
    assert_eq!(max.y, 8.0);
    assert_eq!(max.z, 15.0);
}

#[test]
fn test_pointcloud_bounds_empty() {
    let cloud = PointCloud::new();
    assert!(cloud.bounds().is_none());
}

#[test]
fn test_pointcloud_centroid() {
    let cloud = PointCloud::from_points(vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(4.0, 0.0, 0.0),
        Point3::new(0.0, 6.0, 0.0),
    ]);
    let c = cloud.centroid().unwrap();
    assert!((c.x - 4.0 / 3.0).abs() < 1e-10);
    assert!((c.y - 2.0).abs() < 1e-10);
    assert!((c.z - 0.0).abs() < 1e-10);
}

#[test]
fn test_pointcloud_centroid_empty() {
    let cloud = PointCloud::new();
    assert!(cloud.centroid().is_none());
}

// ═══════════════════════════════════════════════════════════════════════════
// Classification tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_classification_from_u8_known() {
    assert_eq!(Classification::from_u8(0), Classification::Unclassified);
    assert_eq!(Classification::from_u8(2), Classification::Ground);
    assert_eq!(Classification::from_u8(6), Classification::Building);
    assert_eq!(Classification::from_u8(9), Classification::Water);
    assert_eq!(Classification::from_u8(11), Classification::Road);
    assert_eq!(Classification::from_u8(18), Classification::HighNoise);
}

#[test]
fn test_classification_from_u8_unnamed() {
    // codes with no name keep their value instead of folding to Unknown
    assert_eq!(Classification::from_u8(255), Classification::Other(255));
    assert_eq!(Classification::from_u8(100), Classification::Other(100));
    assert_eq!(Classification::from_u8(8), Classification::Other(8));
    // code 1 is the one that really is "unknown"
    assert_eq!(Classification::from_u8(1), Classification::Unknown);
}

// ═══════════════════════════════════════════════════════════════════════════
// Filter tests
// ═══════════════════════════════════════════════════════════════════════════

fn terrain_cloud() -> PointCloud {
    // Simulate terrain: flat ground with some buildings/trees
    let mut points = Vec::new();
    // Ground points (Z ≈ 0)
    for i in 0..10 {
        for j in 0..10 {
            points.push(Point3::new(i as f64, j as f64, 0.0 + (i as f64 * 0.01)));
        }
    }
    // Building points (Z ≈ 10)
    for i in 3..6 {
        for j in 3..6 {
            points.push(Point3::new(i as f64, j as f64, 10.0));
        }
    }
    // Tree points (Z ≈ 5)
    points.push(Point3::new(8.0, 8.0, 5.0));
    points.push(Point3::new(8.0, 8.5, 4.5));
    PointCloud::from_points(points)
}

#[test]
fn test_ground_filter_classifies_ground() {
    let mut cloud = terrain_cloud();
    ground_filter_simple(&mut cloud, 1.5, 0.5);

    // Count ground-classified points
    let ground_count = cloud
        .points()
        .iter()
        .filter(|p| p.classification == Classification::Ground)
        .count();
    // All 100 ground points should be classified as ground
    assert!(
        ground_count >= 90,
        "expected ≥90 ground, got {ground_count}"
    );
}

#[test]
fn test_ground_filter_non_ground_stays() {
    let mut cloud = terrain_cloud();
    ground_filter_simple(&mut cloud, 1.5, 0.5);

    // Building points (Z=10) should NOT be ground
    let building_points: Vec<_> = cloud.points().iter().filter(|p| p.z > 8.0).collect();
    for p in &building_points {
        assert_ne!(
            p.classification,
            Classification::Ground,
            "building point at z={} classified as ground",
            p.z
        );
    }
}

#[test]
fn test_ground_filter_empty_cloud() {
    let mut cloud = PointCloud::new();
    ground_filter_simple(&mut cloud, 1.0, 0.5);
    assert!(cloud.is_empty());
}

#[test]
fn test_thin_random_preserves_subset() {
    let cloud = terrain_cloud();
    let original_len = cloud.len();
    let thinned = thin_random(&cloud, 0.5);
    assert!(thinned.len() < original_len);
    assert!(!thinned.is_empty());
}

#[test]
fn test_thin_random_fraction_zero() {
    let cloud = terrain_cloud();
    let thinned = thin_random(&cloud, 0.0);
    assert!(thinned.is_empty() || thinned.len() <= 1);
}

#[test]
fn test_thin_random_fraction_one() {
    let cloud = terrain_cloud();
    let original_len = cloud.len();
    let thinned = thin_random(&cloud, 1.0);
    // Should keep approximately all points
    assert!(thinned.len() >= original_len / 2);
}

#[test]
fn test_thin_voxel_reduces_density() {
    let cloud = PointCloud::from_points(vec![
        Point3::new(0.1, 0.1, 0.1),
        Point3::new(0.2, 0.2, 0.2),
        Point3::new(0.3, 0.3, 0.3),
        Point3::new(5.0, 5.0, 5.0),
        Point3::new(5.1, 5.1, 5.1),
    ]);
    let thinned = thin_voxel(&cloud, 1.0);
    // 5 points in 2 voxels → 2 points
    assert_eq!(thinned.len(), 2);
}

#[test]
fn test_thin_voxel_preserves_all_if_sparse() {
    let cloud = PointCloud::from_points(vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(5.0, 5.0, 5.0),
        Point3::new(10.0, 10.0, 10.0),
    ]);
    let thinned = thin_voxel(&cloud, 1.0);
    assert_eq!(thinned.len(), 3);
}

#[test]
fn test_thin_voxel_empty_cloud() {
    let cloud = PointCloud::new();
    let thinned = thin_voxel(&cloud, 1.0);
    assert!(thinned.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// Octree tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_octree_build_and_query() {
    let points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(100.0, 100.0, 100.0),
    ];
    let tree = Octree::build(&points, 2);
    let result = tree.query_radius(&points, &Point3::new(0.5, 0.5, 0.0), 1.5);
    assert_eq!(result.len(), 3);
    assert!(!result.contains(&3)); // far point excluded
}

#[test]
fn test_octree_query_no_results() {
    let points = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)];
    let tree = Octree::build(&points, 2);
    let result = tree.query_radius(&points, &Point3::new(50.0, 50.0, 50.0), 1.0);
    assert!(result.is_empty());
}

#[test]
fn test_octree_query_all_points() {
    let points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.1, 0.1, 0.1),
        Point3::new(0.2, 0.2, 0.2),
    ];
    let tree = Octree::build(&points, 2);
    let result = tree.query_radius(&points, &Point3::new(0.1, 0.1, 0.1), 10.0);
    assert_eq!(result.len(), 3);
}

#[test]
fn test_octree_bounds() {
    let points = vec![Point3::new(-10.0, -5.0, 0.0), Point3::new(10.0, 5.0, 20.0)];
    let tree = Octree::build(&points, 2);
    let (min, max) = tree.bounds();
    assert_eq!(min.x, -10.0);
    assert_eq!(max.x, 10.0);
    assert_eq!(max.z, 20.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Interpolation tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_idw_interpolation_basic() {
    let cloud = PointCloud::from_points(vec![
        Point3::new(0.0, 0.0, 10.0),
        Point3::new(10.0, 0.0, 20.0),
        Point3::new(0.0, 10.0, 30.0),
        Point3::new(10.0, 10.0, 40.0),
    ]);
    let grid = idw_interpolation(&cloud, 5.0, 2.0, 0.0, 1).unwrap();
    assert!(grid.width > 0);
    assert!(grid.height > 0);
    // All cells should have valid data (nodata = -9999)
    for &v in &grid.data {
        assert!(v > -9000.0, "unexpected nodata at cell");
    }
}

#[test]
fn test_idw_interpolation_empty_cloud() {
    let cloud = PointCloud::new();
    let result = idw_interpolation(&cloud, 1.0, 2.0, 0.0, 1);
    assert!(result.is_none());
}

#[test]
fn test_idw_interpolation_single_point() {
    let cloud = PointCloud::from_points(vec![Point3::new(5.0, 5.0, 100.0)]);
    let grid = idw_interpolation(&cloud, 1.0, 2.0, 0.0, 1).unwrap();
    // Grid at the point location should be ~100
    assert!(grid.data.iter().any(|&v| (v - 100.0).abs() < 1.0));
}

// ═══════════════════════════════════════════════════════════════════════════
// Normals estimation tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_estimate_normals_flat_surface() {
    // Flat XY plane (Z = 0) → normals should point roughly up (0, 0, ±1)
    let mut points = Vec::new();
    for i in 0..5 {
        for j in 0..5 {
            points.push(Point3::new(i as f64, j as f64, 0.0));
        }
    }
    let cloud = PointCloud::from_points(points);
    let normals = estimate_normals(&cloud, 4);
    assert_eq!(normals.len(), 25);
    // Most normals should have |nz| close to 1 (pointing up or down)
    let vertical_count = normals.iter().filter(|n| n[2].abs() > 0.8).count();
    assert!(
        vertical_count > 15,
        "expected most normals vertical, got {vertical_count}/25"
    );
}

#[test]
fn test_estimate_normals_length() {
    let cloud = PointCloud::from_points(vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
    ]);
    let normals = estimate_normals(&cloud, 3);
    assert_eq!(normals.len(), 4);
    // Each normal should be approximately unit length
    for n in &normals {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((len - 1.0).abs() < 0.1, "normal not unit length: {len}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Statistical outlier removal tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_outlier_removal_removes_noise() {
    let mut points = Vec::new();
    // Cluster of points near origin
    for i in 0..20 {
        points.push(Point3::new((i % 5) as f64 * 0.1, (i / 5) as f64 * 0.1, 0.0));
    }
    // Add a distant outlier
    points.push(Point3::new(100.0, 100.0, 100.0));

    let cloud = PointCloud::from_points(points);
    let filtered = statistical_outlier_removal(&cloud, 5, 2.0);
    // Outlier should be removed
    assert!(filtered.len() < cloud.len());
    assert!(filtered.len() >= 19);
}

#[test]
fn test_outlier_removal_keeps_dense_cluster() {
    let mut points = Vec::new();
    for i in 0..10 {
        for j in 0..10 {
            points.push(Point3::new(i as f64, j as f64, 0.0));
        }
    }
    let cloud = PointCloud::from_points(points);
    let filtered = statistical_outlier_removal(&cloud, 5, 3.0);
    // Dense uniform grid should keep most points (edge points may be removed)
    assert!(filtered.len() >= 90, "expected ≥90, got {}", filtered.len());
}
