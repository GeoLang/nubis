//! Ground filtering, thinning, and outlier removal on inputs with a known answer.

use nubis_core::{
    Classification, PmfParams, Point3, PointCloud, ground_filter_pmf, ground_filter_simple,
    statistical_outlier_removal, thin_random, thin_voxel,
};
use std::collections::HashSet;

/// Point positions as sortable keys, so a set can be compared regardless of order.
fn position_set(cloud: &PointCloud) -> HashSet<(i64, i64, i64)> {
    cloud
        .points()
        .iter()
        .map(|p| {
            (
                (p.x * 1000.0).round() as i64,
                (p.y * 1000.0).round() as i64,
                (p.z * 1000.0).round() as i64,
            )
        })
        .collect()
}

/// Flat ground at z=0 with a 5 m tower of points at one corner.
fn ground_with_tower() -> PointCloud {
    let mut points = Vec::new();
    for y in 0..10 {
        for x in 0..10 {
            points.push(Point3::new(x as f64, y as f64, 0.0));
        }
    }
    for step in 1..=5 {
        points.push(Point3::new(0.0, 0.0, step as f64));
    }
    PointCloud::from_points(points)
}

// ── ground filter ─────────────────────────────────────────────────────────

#[test]
fn ground_filter_marks_the_flat_surface_and_not_the_tower() {
    let mut cloud = ground_with_tower();
    ground_filter_simple(&mut cloud, 2.0, 0.5);

    let ground = cloud
        .points()
        .iter()
        .filter(|p| p.classification == Classification::Ground)
        .count();
    assert_eq!(ground, 100, "every flat point and none of the tower");

    for p in cloud.points().iter().filter(|p| p.z > 0.5) {
        assert_ne!(
            p.classification,
            Classification::Ground,
            "point at z={} was called ground",
            p.z
        );
    }
}

#[test]
fn ground_filter_threshold_controls_how_much_is_ground() {
    let counts: Vec<usize> = [0.5, 2.5, 100.0]
        .iter()
        .map(|threshold| {
            let mut cloud = ground_with_tower();
            ground_filter_simple(&mut cloud, 2.0, *threshold);
            cloud
                .points()
                .iter()
                .filter(|p| p.classification == Classification::Ground)
                .count()
        })
        .collect();

    assert!(
        counts[0] < counts[1] && counts[1] < counts[2],
        "a larger threshold must not shrink the ground set: {counts:?}"
    );
    assert_eq!(counts[2], 105, "a huge threshold takes everything");
}

#[test]
fn ground_filter_keeps_the_point_count_and_positions() {
    let mut cloud = ground_with_tower();
    let before = position_set(&cloud);
    ground_filter_simple(&mut cloud, 2.0, 0.5);

    assert_eq!(cloud.len(), 105, "classification must not drop points");
    assert_eq!(position_set(&cloud), before, "positions must not move");
}

#[test]
fn ground_filter_on_a_single_point_marks_it_ground() {
    let mut cloud = PointCloud::from_points(vec![Point3::new(5.0, 5.0, 12.0)]);
    ground_filter_simple(&mut cloud, 1.0, 0.1);
    assert_eq!(cloud.points()[0].classification, Classification::Ground);
}

// ── voxel thinning ────────────────────────────────────────────────────────

#[test]
fn thin_voxel_keeps_exactly_one_point_per_occupied_voxel() {
    // 4 points in one 1-unit voxel, 1 point in another
    let cloud = PointCloud::from_points(vec![
        Point3::new(0.1, 0.1, 0.1),
        Point3::new(0.2, 0.3, 0.4),
        Point3::new(0.9, 0.9, 0.9),
        Point3::new(0.5, 0.5, 0.5),
        Point3::new(7.5, 7.5, 7.5),
    ]);
    assert_eq!(thin_voxel(&cloud, 1.0).len(), 2);
}

#[test]
fn thin_voxel_returns_points_that_came_from_the_input() {
    let cloud = ground_with_tower();
    let original = position_set(&cloud);
    let thinned = thin_voxel(&cloud, 3.0);

    assert!(thinned.len() < cloud.len());
    for key in position_set(&thinned) {
        assert!(
            original.contains(&key),
            "thinning invented a point at {key:?}"
        );
    }
}

#[test]
fn thin_voxel_returns_the_same_points_in_the_same_order_every_time() {
    // regression: the points came straight out of a HashMap, so the order shifted
    // between runs and two runs on one input wrote different files
    let cloud = ground_with_tower();
    let first: Vec<Point3> = thin_voxel(&cloud, 2.5).points().to_vec();
    for _ in 0..10 {
        assert_eq!(thin_voxel(&cloud, 2.5).points(), first.as_slice());
    }
}

#[test]
fn thin_voxel_writes_identical_bytes_on_every_run() {
    let cloud = ground_with_tower();
    let write = || {
        let mut buf = Vec::new();
        nubis_core::write_las(&thin_voxel(&cloud, 2.5), &mut buf).unwrap();
        buf
    };
    let first = write();
    for _ in 0..10 {
        assert_eq!(write(), first, "voxel thinning is not byte stable");
    }
}

#[test]
fn thin_voxel_with_a_non_positive_size_returns_the_input() {
    let cloud = ground_with_tower();
    for size in [0.0, -1.0] {
        let thinned = thin_voxel(&cloud, size);
        assert_eq!(thinned.len(), cloud.len(), "voxel size {size}");
    }
}

#[test]
fn thin_voxel_keeps_everything_when_the_voxel_is_smaller_than_the_spacing() {
    let cloud = ground_with_tower();
    assert_eq!(thin_voxel(&cloud, 0.01).len(), cloud.len());
}

#[test]
fn thin_voxel_separates_points_that_differ_only_in_z() {
    // voxels are 3D, so a ground return and a canopy return at the same xy stay apart
    let cloud = PointCloud::from_points(vec![
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(1.0, 1.0, 15.0),
    ]);
    assert_eq!(thin_voxel(&cloud, 2.0).len(), 2);
}

// ── random thinning ───────────────────────────────────────────────────────

#[test]
fn thin_random_is_reproducible() {
    let cloud = ground_with_tower();
    let first: Vec<Point3> = thin_random(&cloud, 0.3).points().to_vec();
    for _ in 0..5 {
        assert_eq!(thin_random(&cloud, 0.3).points(), first.as_slice());
    }
}

#[test]
fn thin_random_keeps_the_requested_fraction() {
    // regression: an integer stride used to round every fraction over a half up
    // to the whole cloud, so 0.6 and 0.9 both kept 100% of the points
    let cloud = ground_with_tower();
    let n = cloud.len();
    for fraction in [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9] {
        let kept = thin_random(&cloud, fraction).len();
        let expected = (n as f64 * fraction).round() as usize;
        assert_eq!(kept, expected, "asked for {fraction} of {n}");
        assert!(kept < n, "fraction {fraction} must drop something");
    }
}

#[test]
fn thin_random_clamps_a_fraction_above_one() {
    let cloud = ground_with_tower();
    assert_eq!(thin_random(&cloud, 5.0).len(), cloud.len());
}

#[test]
fn thin_random_with_a_negative_fraction_keeps_nothing() {
    let cloud = ground_with_tower();
    assert!(thin_random(&cloud, -1.0).is_empty());
}

#[test]
fn thin_random_returns_points_that_came_from_the_input() {
    let cloud = ground_with_tower();
    let original = position_set(&cloud);
    for key in position_set(&thin_random(&cloud, 0.4)) {
        assert!(original.contains(&key));
    }
}

// ── statistical outlier removal ───────────────────────────────────────────

#[test]
fn outlier_removal_drops_only_the_distant_point() {
    let mut points = Vec::new();
    for y in 0..7 {
        for x in 0..7 {
            points.push(Point3::new(x as f64, y as f64, 0.0));
        }
    }
    points.push(Point3::new(400.0, 400.0, 400.0));
    let cloud = PointCloud::from_points(points);

    let cleaned = statistical_outlier_removal(&cloud, 8, 1.5);
    assert_eq!(cleaned.len(), 49, "the planted outlier and nothing else");
    let (_, max) = cleaned.bounds().unwrap();
    assert!(max.x < 100.0, "the far point survived");
}

#[test]
fn outlier_removal_trims_grid_corners_but_keeps_the_interior() {
    // corners have fewer close neighbours than interior points, so their mean
    // neighbour distance is genuinely larger and a tight threshold clips them
    let mut points = Vec::new();
    for y in 0..8 {
        for x in 0..8 {
            points.push(Point3::new(x as f64, y as f64, 0.0));
        }
    }
    let cloud = PointCloud::from_points(points);

    let tight = statistical_outlier_removal(&cloud, 6, 2.0);
    assert_eq!(tight.len(), 60, "the four corners are trimmed");
    let survivors: HashSet<(i64, i64)> = tight
        .points()
        .iter()
        .map(|p| (p.x as i64, p.y as i64))
        .collect();
    for corner in [(0, 0), (7, 0), (0, 7), (7, 7)] {
        assert!(!survivors.contains(&corner), "corner {corner:?} survived");
    }
    for interior in [(1, 1), (3, 4), (6, 6)] {
        assert!(
            survivors.contains(&interior),
            "interior {interior:?} dropped"
        );
    }

    // a looser threshold leaves the whole grid alone
    assert_eq!(
        statistical_outlier_removal(&cloud, 6, 4.0).len(),
        cloud.len()
    );
}

#[test]
fn outlier_removal_returns_the_input_when_k_reaches_the_point_count() {
    let cloud = PointCloud::from_points(vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(90.0, 90.0, 90.0),
    ]);
    // k >= n leaves nothing to compare against, so the cloud comes back whole
    assert_eq!(statistical_outlier_removal(&cloud, 3, 1.0).len(), 3);
    assert_eq!(statistical_outlier_removal(&cloud, 50, 1.0).len(), 3);
}

#[test]
fn outlier_removal_of_an_empty_cloud_stays_empty() {
    assert!(statistical_outlier_removal(&PointCloud::new(), 5, 2.0).is_empty());
}

#[test]
fn outlier_removal_keeps_classification_and_intensity() {
    let mut points = Vec::new();
    for i in 0..30 {
        points.push(
            Point3::new((i % 6) as f64, (i / 6) as f64, 0.0)
                .with_intensity(i as u16)
                .with_classification(Classification::Building),
        );
    }
    points.push(Point3::new(300.0, 300.0, 0.0));
    let cleaned = statistical_outlier_removal(&PointCloud::from_points(points), 6, 1.5);

    assert_eq!(cleaned.len(), 30);
    for p in cleaned.points() {
        assert_eq!(p.classification, Classification::Building);
    }
    let intensities: Vec<u16> = cleaned.points().iter().map(|p| p.intensity).collect();
    assert_eq!(intensities, (0..30).collect::<Vec<u16>>());
}

// ── progressive morphological filter ──────────────────────────────────────

/// Flat ground at z=0 on a 60x60 grid, a 10x10 building at z=6, one tree point at z=8.
/// The building replaces the ground under it, the way a roof hides it from the scanner.
fn ground_with_building_and_tree() -> PointCloud {
    let building = 20..30;
    let mut points = Vec::new();
    for y in 0..60 {
        for x in 0..60 {
            let z = if building.contains(&x) && building.contains(&y) {
                6.0
            } else {
                0.0
            };
            points.push(Point3::new(x as f64, y as f64, z));
        }
    }
    points.push(Point3::new(45.0, 45.0, 8.0));
    PointCloud::from_points(points)
}

#[test]
fn pmf_separates_the_building_and_the_tree_from_the_flat_ground() {
    let mut cloud = ground_with_building_and_tree();
    ground_filter_pmf(&mut cloud, &PmfParams::default());

    for p in cloud.points() {
        let is_ground = p.classification == Classification::Ground;
        assert_eq!(
            is_ground,
            p.z == 0.0,
            "point at ({}, {}, {}) came out {:?}",
            p.x,
            p.y,
            p.z,
            p.classification
        );
    }
    let ground = cloud
        .points()
        .iter()
        .filter(|p| p.classification == Classification::Ground)
        .count();
    assert_eq!(ground, 3500, "every flat point, no roof and no tree");

    // the defect pmf fixes: a 2 unit cell that holds nothing but roof returns has its own
    // minimum, so the roof reads as ground
    let mut same_scene = ground_with_building_and_tree();
    ground_filter_simple(&mut same_scene, 2.0, 0.5);
    let roof_as_ground = same_scene
        .points()
        .iter()
        .filter(|p| p.z == 6.0 && p.classification == Classification::Ground)
        .count();
    assert!(
        roof_as_ground > 0,
        "the simple filter was expected to call part of the roof ground"
    );
}

#[test]
fn pmf_keeps_a_slope_gentler_than_the_slope_parameter() {
    let mut points = Vec::new();
    for y in 0..60 {
        for x in 0..60 {
            points.push(Point3::new(x as f64, y as f64, 0.1 * x as f64));
        }
    }
    let mut cloud = PointCloud::from_points(points);
    ground_filter_pmf(&mut cloud, &PmfParams::default());

    let ground = cloud
        .points()
        .iter()
        .filter(|p| p.classification == Classification::Ground)
        .count();
    assert_eq!(
        ground,
        cloud.len(),
        "terrain rising 0.1 per unit is under the default slope of 0.15"
    );
}

#[test]
fn pmf_on_an_empty_cloud_does_nothing() {
    let mut cloud = PointCloud::new();
    ground_filter_pmf(&mut cloud, &PmfParams::default());
    assert!(cloud.is_empty());
}
