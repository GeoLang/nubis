//! Variogram and ordinary kriging numerics, checked against values worked out by hand
//! and against the invariants kriging is supposed to hold.

use nubis_core::{
    Point3, PointCloud, VariogramModel, empirical_variogram, getis_ord_gi_star, morans_i,
    ordinary_kriging,
};

fn cloud(points: Vec<Point3>) -> PointCloud {
    PointCloud::from_points(points)
}

/// Unit square with a point at each corner, so the grid origin sits on a data point.
fn unit_square(z: [f64; 4]) -> PointCloud {
    cloud(vec![
        Point3::new(0.0, 0.0, z[0]),
        Point3::new(1.0, 0.0, z[1]),
        Point3::new(0.0, 1.0, z[2]),
        Point3::new(1.0, 1.0, z[3]),
    ])
}

fn no_nugget() -> VariogramModel {
    VariogramModel::Spherical {
        nugget: 0.0,
        sill: 10.0,
        range: 5.0,
    }
}

// ── empirical variogram ───────────────────────────────────────────────────

#[test]
fn variogram_semivariance_matches_the_hand_computed_value() {
    // three points on a line, 1 unit apart, z = 0, 2, 4.
    // one bin wide enough for the 1-unit pairs only (the 2-unit pair is at the
    // max lag and is excluded by `dist >= max_lag`).
    let c = cloud(vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 2.0),
        Point3::new(2.0, 0.0, 4.0),
    ]);
    let bins = empirical_variogram(&c, 1, 2.0);

    assert_eq!(bins.len(), 1);
    // pairs at distance 1: (0,2) and (2,4), both dz = 2
    assert_eq!(bins[0].count, 2);
    // semivariance = Σdz² / 2n = (4 + 4) / 4 = 2
    assert!(
        (bins[0].semivariance - 2.0).abs() < 1e-12,
        "got {}",
        bins[0].semivariance
    );
    // lag is the bin centre: (0 + 0.5) * 2.0
    assert!((bins[0].lag - 1.0).abs() < 1e-12, "got {}", bins[0].lag);
}

#[test]
fn variogram_of_a_flat_surface_is_zero_everywhere() {
    let mut points = Vec::new();
    for y in 0..6 {
        for x in 0..6 {
            points.push(Point3::new(x as f64, y as f64, 17.5));
        }
    }
    let bins = empirical_variogram(&cloud(points), 5, 6.0);

    assert!(!bins.is_empty());
    for bin in &bins {
        assert!(
            bin.semivariance.abs() < 1e-12,
            "flat surface gave semivariance {} at lag {}",
            bin.semivariance,
            bin.lag
        );
    }
}

#[test]
fn variogram_bin_centres_are_evenly_spaced() {
    let mut points = Vec::new();
    for i in 0..40 {
        points.push(Point3::new(i as f64 * 0.5, 0.0, i as f64));
    }
    let bins = empirical_variogram(&cloud(points), 4, 8.0);

    assert_eq!(bins.len(), 4);
    for (i, bin) in bins.iter().enumerate() {
        let expected = (i as f64 + 0.5) * 2.0;
        assert!(
            (bin.lag - expected).abs() < 1e-12,
            "bin {i} lag {}",
            bin.lag
        );
        assert!(bin.count > 0);
        assert!(bin.semivariance >= 0.0);
    }
}

#[test]
fn variogram_excludes_pairs_beyond_the_max_lag() {
    let c = cloud(vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 1.0),
        Point3::new(100.0, 0.0, 500.0),
    ]);
    let bins = empirical_variogram(&c, 2, 2.0);
    let pairs: usize = bins.iter().map(|b| b.count).sum();
    assert_eq!(pairs, 1, "only the 1-unit pair is within the max lag");
}

#[test]
fn variogram_of_a_single_point_has_no_pairs() {
    let c = cloud(vec![Point3::new(0.0, 0.0, 1.0)]);
    assert!(empirical_variogram(&c, 5, 10.0).is_empty());
}

#[test]
fn fit_spherical_falls_back_to_a_usable_model_without_data() {
    match VariogramModel::fit_spherical(&[]) {
        VariogramModel::Spherical { sill, range, .. } => {
            assert!(sill > 0.0);
            assert!(range > 0.0, "a zero range would divide by zero on evaluate");
        }
        other => panic!("expected spherical, got {other:?}"),
    }
}

#[test]
fn fitted_model_is_non_decreasing_in_distance() {
    let mut points = Vec::new();
    for i in 0..30 {
        let x = i as f64;
        points.push(Point3::new(x, 0.0, x * 0.3));
    }
    let bins = empirical_variogram(&cloud(points), 6, 12.0);
    let model = VariogramModel::fit_spherical(&bins);

    let mut previous = model.evaluate(0.0);
    for step in 1..60 {
        let value = model.evaluate(step as f64 * 0.5);
        assert!(
            value >= previous - 1e-12,
            "variogram dropped at h={}: {value} < {previous}",
            step as f64 * 0.5
        );
        previous = value;
    }
}

#[test]
fn model_evaluate_hits_the_documented_values() {
    let spherical = VariogramModel::Spherical {
        nugget: 1.0,
        sill: 9.0,
        range: 4.0,
    };
    assert!(spherical.evaluate(0.0).abs() < 1e-12, "gamma(0) must be 0");
    // at h = range the spherical model reaches nugget + sill
    assert!((spherical.evaluate(4.0) - 10.0).abs() < 1e-12);
    assert!(
        (spherical.evaluate(400.0) - 10.0).abs() < 1e-12,
        "flat past range"
    );
    // at h = range/2: 1 + 9 * (1.5*0.5 - 0.5*0.125) = 1 + 9*0.6875 = 7.1875
    assert!((spherical.evaluate(2.0) - 7.1875).abs() < 1e-12);

    let exponential = VariogramModel::Exponential {
        nugget: 0.0,
        sill: 8.0,
        range: 3.0,
    };
    assert!(exponential.evaluate(0.0).abs() < 1e-12);
    // at h = range: 8 * (1 - e^-3)
    let expected = 8.0 * (1.0 - (-3.0f64).exp());
    assert!((exponential.evaluate(3.0) - expected).abs() < 1e-12);

    let gaussian = VariogramModel::Gaussian {
        nugget: 0.0,
        sill: 8.0,
        range: 3.0,
    };
    assert!(gaussian.evaluate(0.0).abs() < 1e-12);
    assert!((gaussian.evaluate(3.0) - expected).abs() < 1e-12);
}

// ── ordinary kriging ──────────────────────────────────────────────────────

#[test]
fn kriging_of_a_constant_field_returns_that_constant() {
    // the kriging weights are constrained to sum to 1, so a field that is 42
    // everywhere must krige to 42 in every cell that gets a value
    let mut points = Vec::new();
    for y in 0..5 {
        for x in 0..5 {
            points.push(Point3::new(x as f64, y as f64, 42.0));
        }
    }
    let grid = ordinary_kriging(&cloud(points), &no_nugget(), 1.0, 3.0);

    let valued: Vec<f64> = grid.data.iter().copied().filter(|v| !v.is_nan()).collect();
    assert!(!valued.is_empty(), "no cell was estimated");
    for v in valued {
        assert!((v - 42.0).abs() < 1e-6, "constant field kriged to {v}");
    }
}

#[test]
fn kriging_reproduces_the_data_value_at_a_data_location() {
    // with a zero nugget, ordinary kriging is an exact interpolator: the grid
    // origin sits on the (0,0) point, so that cell must come back as its z
    let grid = ordinary_kriging(&unit_square([5.0, 9.0, 11.0, 20.0]), &no_nugget(), 1.0, 3.0);
    assert!(
        (grid.data[0] - 5.0).abs() < 1e-6,
        "corner cell was {} not 5.0",
        grid.data[0]
    );
}

#[test]
fn kriging_estimates_stay_within_the_range_of_the_inputs() {
    let z = [5.0, 9.0, 11.0, 20.0];
    let grid = ordinary_kriging(&unit_square(z), &no_nugget(), 0.25, 3.0);

    let lo = z.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut estimated = 0;
    for v in grid.data.iter().copied().filter(|v| !v.is_nan()) {
        // ordinary kriging can overshoot slightly, but not wildly
        assert!(
            v > lo - 5.0 && v < hi + 5.0,
            "estimate {v} is far outside the input range {lo}..{hi}"
        );
        estimated += 1;
    }
    assert!(estimated > 0);
}

#[test]
fn kriging_grid_geometry_follows_the_cloud_bounds() {
    let c = cloud(vec![
        Point3::new(10.0, 20.0, 1.0),
        Point3::new(13.0, 20.0, 2.0),
        Point3::new(10.0, 24.0, 3.0),
        Point3::new(13.0, 24.0, 4.0),
    ]);
    let grid = ordinary_kriging(&c, &no_nugget(), 1.0, 10.0);

    assert!((grid.origin_x - 10.0).abs() < 1e-12);
    assert!((grid.origin_y - 20.0).abs() < 1e-12);
    assert_eq!(grid.cell_size, 1.0);
    assert_eq!(grid.width, 4, "3 units across at 1.0 plus the closing node");
    assert_eq!(grid.height, 5, "4 units up at 1.0 plus the closing node");
    assert_eq!(grid.data.len(), grid.width * grid.height);
}

#[test]
fn kriging_leaves_cells_with_too_few_neighbours_unestimated() {
    // two far-apart pairs, and a search radius that never reaches three points
    let c = cloud(vec![
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(0.5, 0.0, 2.0),
        Point3::new(50.0, 50.0, 3.0),
        Point3::new(50.5, 50.0, 4.0),
    ]);
    let grid = ordinary_kriging(&c, &no_nugget(), 5.0, 1.0);
    assert!(
        grid.data.iter().all(|v| v.is_nan()),
        "no cell has three neighbours within the radius"
    );
}

#[test]
fn kriging_survives_duplicate_points() {
    // identical points make two rows of the kriging matrix equal, so the solve
    // must fail cleanly and leave the cell empty rather than panic or return NaN maths
    let c = cloud(vec![
        Point3::new(0.0, 0.0, 5.0),
        Point3::new(0.0, 0.0, 5.0),
        Point3::new(0.0, 0.0, 5.0),
        Point3::new(1.0, 1.0, 7.0),
    ]);
    let grid = ordinary_kriging(&c, &no_nugget(), 0.5, 5.0);
    for v in &grid.data {
        assert!(v.is_nan() || v.is_finite(), "got a non-finite estimate {v}");
    }
}

#[test]
fn kriging_survives_collinear_points() {
    let points: Vec<Point3> = (0..8)
        .map(|i| Point3::new(i as f64, 0.0, 10.0 + i as f64))
        .collect();
    let grid = ordinary_kriging(&cloud(points), &no_nugget(), 1.0, 4.0);

    assert_eq!(grid.height, 1, "a line of points is one row tall");
    for v in &grid.data {
        assert!(v.is_nan() || v.is_finite());
    }
}

#[test]
fn kriging_of_an_empty_cloud_produces_an_empty_grid() {
    let grid = ordinary_kriging(&PointCloud::new(), &no_nugget(), 1.0, 5.0);
    assert!(
        grid.data.iter().all(|v| v.is_nan()),
        "nothing to interpolate from"
    );
}

#[test]
fn kriging_respects_the_search_radius() {
    // a distant high point must not influence the cells near the origin cluster
    let mut points = vec![Point3::new(100.0, 100.0, 9_999.0)];
    for y in 0..4 {
        for x in 0..4 {
            points.push(Point3::new(x as f64, y as f64, 20.0));
        }
    }
    let grid = ordinary_kriging(&cloud(points), &no_nugget(), 1.0, 2.0);

    // the cell at the cluster corner only sees the 20.0 points
    assert!(
        (grid.data[0] - 20.0).abs() < 1e-6,
        "corner cell was {}",
        grid.data[0]
    );
}

// ── spatial statistics ────────────────────────────────────────────────────

#[test]
fn morans_i_separates_clustered_from_random() {
    // two tight clusters, one high one low: strong positive autocorrelation
    let mut clustered = Vec::new();
    for i in 0..6 {
        for j in 0..6 {
            clustered.push(Point3::new(i as f64 * 0.1, j as f64 * 0.1, 100.0));
        }
    }
    for i in 0..6 {
        for j in 0..6 {
            clustered.push(Point3::new(10.0 + i as f64 * 0.1, j as f64 * 0.1, 1.0));
        }
    }
    let (clustered_i, expected, _) = morans_i(&cloud(clustered), 1.0);
    assert!(
        clustered_i > expected,
        "clustered data should exceed the expected value"
    );

    // alternating high/low on a lattice: negative autocorrelation
    let mut checker = Vec::new();
    for i in 0..8 {
        for j in 0..8 {
            let z = if (i + j) % 2 == 0 { 100.0 } else { 1.0 };
            checker.push(Point3::new(i as f64, j as f64, z));
        }
    }
    let (checker_i, checker_expected, _) = morans_i(&cloud(checker), 1.0);
    assert!(
        checker_i < checker_expected,
        "a checkerboard should be dispersed, got {checker_i} vs {checker_expected}"
    );
}

#[test]
fn morans_i_of_a_constant_field_is_undefined_not_a_crash() {
    let points: Vec<Point3> = (0..10).map(|i| Point3::new(i as f64, 0.0, 5.0)).collect();
    let (i, e, z) = morans_i(&cloud(points), 2.0);
    assert_eq!((i, e, z), (0.0, 0.0, 0.0), "zero variance has no statistic");
}

#[test]
fn getis_ord_scores_hot_and_cold_clusters_with_opposite_signs() {
    let mut points = Vec::new();
    for i in 0..5 {
        for j in 0..5 {
            points.push(Point3::new(i as f64 * 0.5, j as f64 * 0.5, 100.0));
        }
    }
    for i in 0..5 {
        for j in 0..5 {
            points.push(Point3::new(20.0 + i as f64 * 0.5, j as f64 * 0.5, 1.0));
        }
    }
    let c = cloud(points);
    let scores = getis_ord_gi_star(&c, 2.0);

    assert_eq!(scores.len(), c.len());
    assert!(scores[0] > 0.0, "hot cluster should be positive");
    assert!(
        *scores.last().unwrap() < 0.0,
        "cold cluster should be negative"
    );
}

#[test]
fn getis_ord_of_a_constant_field_is_all_zero() {
    let points: Vec<Point3> = (0..8).map(|i| Point3::new(i as f64, 0.0, 3.0)).collect();
    let scores = getis_ord_gi_star(&cloud(points), 2.0);
    assert!(scores.iter().all(|s| *s == 0.0));
}
