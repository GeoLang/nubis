//! End-to-end tests: build LAS fixtures with nubis-core, then run the real binary against them.

use nubis_core::{Classification, Point3, PointCloud, read_las, write_las};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::{TempDir, tempdir};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_nubis"))
        .args(args)
        .output()
        .expect("failed to run nubis")
}

fn run_ok(args: &[&str]) -> String {
    let out = run(args);
    assert!(
        out.status.success(),
        "nubis {args:?} failed\nstderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("stdout is utf-8")
}

fn run_err(args: &[&str]) -> String {
    let out = run(args);
    assert!(
        !out.status.success(),
        "nubis {args:?} unexpectedly succeeded\nstdout: {}",
        String::from_utf8_lossy(&out.stdout)
    );
    String::from_utf8(out.stderr).expect("stderr is utf-8")
}

fn fixture(dir: &TempDir, name: &str, cloud: &PointCloud) -> PathBuf {
    let path = dir.path().join(name);
    let mut writer = BufWriter::new(File::create(&path).unwrap());
    write_las(cloud, &mut writer).unwrap();
    writer.flush().unwrap();
    path
}

fn load(path: &Path) -> PointCloud {
    let mut reader = BufReader::new(File::open(path).unwrap());
    read_las(&mut reader).unwrap()
}

fn s(path: &Path) -> &str {
    path.to_str().unwrap()
}

/// 10x10 grid on a 1 unit spacing, sloping in x, with every tenth point raised 12m.
fn terrain() -> PointCloud {
    let mut points = Vec::new();
    for i in 0..100 {
        let x = (i % 10) as f64;
        let y = (i / 10) as f64;
        let ground = 5.0 + x * 0.1;
        let z = if i % 10 == 0 { ground + 12.0 } else { ground };
        points.push(Point3::new(x, y, z));
    }
    PointCloud::from_points(points)
}

/// Plane sampled exactly on integer nodes, so grid nodes coincide with points.
fn plane() -> PointCloud {
    let mut points = Vec::new();
    for y in 0..5 {
        for x in 0..5 {
            let z = 10.0 + x as f64 * 0.5 + y as f64 * 0.25;
            points.push(Point3::new(x as f64, y as f64, z));
        }
    }
    PointCloud::from_points(points)
}

/// Dense cluster plus one point far outside it.
fn cluster_with_outlier() -> PointCloud {
    let mut points = Vec::new();
    for y in 0..6 {
        for x in 0..6 {
            points.push(Point3::new(x as f64, y as f64, 1.0));
        }
    }
    points.push(Point3::new(500.0, 500.0, 90.0));
    PointCloud::from_points(points)
}

struct Asc {
    ncols: usize,
    nrows: usize,
    cellsize: f64,
    nodata: f64,
    /// rows as written, north first
    rows: Vec<Vec<f64>>,
}

fn parse_asc(path: &Path) -> Asc {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    let mut header = std::collections::HashMap::new();
    let mut rows = Vec::new();

    for _ in 0..6 {
        let line = lines.next().expect("asc header line");
        let (key, value) = line.split_once(' ').expect("key value header");
        header.insert(key.to_string(), value.trim().parse::<f64>().unwrap());
    }
    for line in lines {
        rows.push(
            line.split_whitespace()
                .map(|v| v.parse::<f64>().unwrap())
                .collect::<Vec<f64>>(),
        );
    }

    Asc {
        ncols: header["ncols"] as usize,
        nrows: header["nrows"] as usize,
        cellsize: header["cellsize"],
        nodata: header["NODATA_value"],
        rows,
    }
}

#[test]
fn info_reports_header_stats_and_classes() {
    let dir = tempdir().unwrap();
    let cloud = PointCloud::from_points(vec![
        Point3::new(0.0, 0.0, 1.0).with_classification(Classification::Ground),
        Point3::new(10.0, 0.0, 3.0).with_classification(Classification::Ground),
        Point3::new(10.0, 20.0, 5.0).with_classification(Classification::Building),
    ]);
    let path = fixture(&dir, "in.las", &cloud);

    let out = run_ok(&["info", "--input", s(&path)]);

    assert!(out.contains("LAS 1.2, point format 0"), "{out}");
    assert!(out.contains("Points: 3"), "{out}");
    assert!(
        out.contains("Bounds: (0.00, 0.00, 1.00) - (10.00, 20.00, 5.00)"),
        "{out}"
    );
    assert!(out.contains("Centroid: (6.67, 6.67, 3.00)"), "{out}");
    assert!(out.contains("Z mean: 3.00"), "{out}");
    assert!(out.contains("2 Ground: 2"), "{out}");
    assert!(out.contains("6 Building: 1"), "{out}");
}

#[test]
fn ground_classify_writes_classified_points() {
    let dir = tempdir().unwrap();
    let input = fixture(&dir, "terrain.las", &terrain());
    let output = dir.path().join("ground.las");

    let out = run_ok(&[
        "ground-classify",
        "--input",
        s(&input),
        "--output",
        s(&output),
        "--cell-size",
        "2.0",
        "--threshold",
        "0.5",
    ]);
    assert!(out.contains("Ground: 90/100 points"), "{out}");

    let result = load(&output);
    assert_eq!(result.len(), 100, "no points may be dropped");

    let ground = result
        .points()
        .iter()
        .filter(|p| p.classification == Classification::Ground)
        .count();
    assert_eq!(ground, 90);

    // the raised points must not be ground
    for p in result.points() {
        if p.z > 10.0 {
            assert_eq!(p.classification, Classification::Unclassified);
        }
    }
}

#[test]
fn thin_voxel_reduces_point_count() {
    let dir = tempdir().unwrap();
    let input = fixture(&dir, "terrain.las", &terrain());
    let output = dir.path().join("thin.las");

    let out = run_ok(&[
        "thin",
        "--input",
        s(&input),
        "--output",
        s(&output),
        "--voxel-size",
        "5.0",
    ]);
    assert!(out.contains("Thinned 100 -> "), "{out}");

    let result = load(&output);
    assert!(result.len() < 100, "voxel thinning must drop points");
    assert!(!result.is_empty());
    assert!(
        out.contains(&format!("-> {} points", result.len())),
        "summary must match the written file: {out}"
    );

    // every surviving point has to come from the input
    let input_cloud = load(&input);
    for p in result.points() {
        assert!(
            input_cloud.points().iter().any(|q| q.distance_to(p) < 0.01),
            "unexpected point {p:?}"
        );
    }
}

#[test]
fn thin_random_keeps_requested_fraction() {
    let dir = tempdir().unwrap();
    let input = fixture(&dir, "terrain.las", &terrain());
    let output = dir.path().join("thin.las");

    let out = run_ok(&[
        "thin",
        "--input",
        s(&input),
        "--output",
        s(&output),
        "--method",
        "random",
        "--fraction",
        "0.25",
    ]);

    let result = load(&output);
    assert_eq!(result.len(), 25);
    assert!(out.contains("Thinned 100 -> 25 points"), "{out}");
}

#[test]
fn outlier_removal_drops_the_planted_outlier() {
    let dir = tempdir().unwrap();
    let input = fixture(&dir, "cluster.las", &cluster_with_outlier());
    let output = dir.path().join("clean.las");

    let out = run_ok(&[
        "outlier-removal",
        "--input",
        s(&input),
        "--output",
        s(&output),
        "--neighbours",
        "5",
        "--std-multiplier",
        "1.0",
    ]);
    assert!(out.contains("Removed 1/37 points, 36 kept"), "{out}");

    let result = load(&output);
    assert_eq!(result.len(), 36);
    assert!(
        result.points().iter().all(|p| p.x < 100.0 && p.y < 100.0),
        "the far point must be gone"
    );
}

#[test]
fn interpolate_idw_grid_matches_input_elevations() {
    let dir = tempdir().unwrap();
    let input = fixture(&dir, "plane.las", &plane());
    let output = dir.path().join("dem.asc");

    let out = run_ok(&[
        "interpolate-to-grid",
        "--input",
        s(&input),
        "--output",
        s(&output),
        "--cell-size",
        "1.0",
    ]);
    assert!(out.contains("Grid: 5x5"), "{out}");
    assert!(out.contains("25/25 cells filled"), "{out}");

    let asc = parse_asc(&output);
    assert_eq!(asc.ncols, 5);
    assert_eq!(asc.nrows, 5);
    assert!((asc.cellsize - 1.0).abs() < 1e-9);
    assert_eq!(asc.rows.len(), 5);
    assert!(asc.rows.iter().all(|r| r.len() == 5));

    // grid nodes coincide with input points, so values reproduce the plane.
    // rows run north to south, so the last row is y = 0.
    for (row_from_top, row) in asc.rows.iter().enumerate() {
        let y = (asc.nrows - 1 - row_from_top) as f64;
        for (col, value) in row.iter().enumerate() {
            let expected = 10.0 + col as f64 * 0.5 + y * 0.25;
            assert!(
                (value - expected).abs() < 0.01,
                "cell ({col}, {y}) is {value}, expected {expected}"
            );
        }
    }
}

#[test]
fn interpolate_idw_leaves_unreached_cells_as_nodata() {
    let dir = tempdir().unwrap();
    let cloud = PointCloud::from_points(vec![
        Point3::new(0.0, 0.0, 5.0),
        Point3::new(1.0, 0.0, 5.0),
        Point3::new(0.0, 1.0, 5.0),
        Point3::new(60.0, 60.0, 8.0),
    ]);
    let input = fixture(&dir, "sparse.las", &cloud);
    let output = dir.path().join("dem.asc");

    run_ok(&[
        "interpolate-to-grid",
        "--input",
        s(&input),
        "--output",
        s(&output),
        "--cell-size",
        "10.0",
        "--search-radius",
        "5.0",
    ]);

    let asc = parse_asc(&output);
    let nodata_cells = asc
        .rows
        .iter()
        .flatten()
        .filter(|v| (**v - asc.nodata).abs() < 1e-9)
        .count();
    assert!(nodata_cells > 0, "cells outside the radius must be nodata");
    let valued: Vec<f64> = asc
        .rows
        .iter()
        .flatten()
        .copied()
        .filter(|v| (v - asc.nodata).abs() >= 1e-9)
        .collect();
    assert!(!valued.is_empty());
    assert!(valued.iter().all(|v| (4.9..=8.1).contains(v)), "{valued:?}");
}

#[test]
fn interpolate_kriging_writes_grid_within_input_range() {
    let dir = tempdir().unwrap();
    let input = fixture(&dir, "plane.las", &plane());
    let output = dir.path().join("krige.asc");

    let out = run_ok(&[
        "interpolate-to-grid",
        "--input",
        s(&input),
        "--output",
        s(&output),
        "--method",
        "kriging",
        "--cell-size",
        "1.0",
        "--search-radius",
        "3.0",
        "--variogram-bins",
        "6",
    ]);
    assert!(out.contains("Variogram: spherical"), "{out}");
    assert!(out.contains("Grid: 5x5"), "{out}");

    let asc = parse_asc(&output);
    assert_eq!(asc.ncols, 5);
    assert_eq!(asc.nrows, 5);

    let valued: Vec<f64> = asc
        .rows
        .iter()
        .flatten()
        .copied()
        .filter(|v| (v - asc.nodata).abs() >= 1e-9)
        .collect();
    assert!(valued.len() >= 20, "most cells should be estimated");
    // input elevations span 10.0 to 13.0
    assert!(
        valued.iter().all(|v| (9.0..=14.0).contains(v)),
        "{valued:?}"
    );
}

#[test]
fn variogram_prints_bins_and_fitted_model() {
    let dir = tempdir().unwrap();
    let input = fixture(&dir, "terrain.las", &terrain());

    let out = run_ok(&["variogram", "--input", s(&input), "--bins", "5"]);

    assert!(out.contains("non-empty bins"), "{out}");
    assert!(out.contains("Fitted spherical"), "{out}");

    let bin_lines: Vec<&str> = out
        .lines()
        .skip_while(|l| !l.contains("semivariance"))
        .skip(1)
        .take_while(|l| !l.starts_with("Fitted"))
        .collect();
    assert!(!bin_lines.is_empty(), "{out}");
    for line in bin_lines {
        let cols: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(cols.len(), 3, "bin line: {line}");
        assert!(cols[0].parse::<f64>().unwrap() > 0.0);
        assert!(cols[1].parse::<f64>().unwrap() > 0.0);
        assert!(cols[2].parse::<usize>().unwrap() > 0);
    }
}

#[test]
fn empty_result_fails_without_writing_a_file() {
    let dir = tempdir().unwrap();
    let input = fixture(&dir, "terrain.las", &terrain());
    let output = dir.path().join("thin.las");

    let err = run_err(&[
        "thin",
        "--input",
        s(&input),
        "--output",
        s(&output),
        "--method",
        "random",
        "--fraction",
        "0.0",
    ]);
    assert!(err.contains("empty"), "{err}");
    assert!(!output.exists(), "no file should be left behind");
}

#[test]
fn demo_writes_a_readable_las_file() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("demo.las");

    let out = run_ok(&["demo", "--output", s(&output), "--count", "250"]);
    assert!(out.contains("Wrote 250 synthetic points"), "{out}");

    let cloud = load(&output);
    assert_eq!(cloud.len(), 250);
    let (min, max) = cloud.bounds().unwrap();
    assert!(max.z - min.z > 10.0, "demo cloud should have relief");
}

#[test]
fn pipeline_demo_ground_thin_interpolate() {
    let dir = tempdir().unwrap();
    let raw = dir.path().join("raw.las");
    let classified = dir.path().join("classified.las");
    let thinned = dir.path().join("thinned.las");
    let dem = dir.path().join("dem.asc");

    run_ok(&["demo", "--output", s(&raw), "--count", "400"]);
    run_ok(&[
        "ground-classify",
        "--input",
        s(&raw),
        "--output",
        s(&classified),
        "--cell-size",
        "4.0",
        "--threshold",
        "1.0",
    ]);
    run_ok(&[
        "thin",
        "--input",
        s(&classified),
        "--output",
        s(&thinned),
        "--voxel-size",
        "4.0",
    ]);
    run_ok(&[
        "interpolate-to-grid",
        "--input",
        s(&thinned),
        "--output",
        s(&dem),
        "--cell-size",
        "4.0",
        "--search-radius",
        "12.0",
    ]);

    let classified_cloud = load(&classified);
    assert!(
        classified_cloud
            .points()
            .iter()
            .any(|p| p.classification == Classification::Ground)
    );

    let thinned_cloud = load(&thinned);
    assert!(thinned_cloud.len() < classified_cloud.len());

    let asc = parse_asc(&dem);
    assert!(asc.ncols > 1 && asc.nrows > 1);
    assert_eq!(asc.rows.len(), asc.nrows);
}

#[test]
fn missing_input_file_fails() {
    let dir = tempdir().unwrap();
    let missing = dir.path().join("nope.las");

    let err = run_err(&["info", "--input", s(&missing)]);
    assert!(err.contains("nope.las"), "{err}");
    assert!(
        err.contains("No such file") || err.contains("cannot find"),
        "{err}"
    );
}

#[test]
fn non_las_input_fails() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("junk.las");
    std::fs::write(&path, vec![0u8; 400]).unwrap();

    let err = run_err(&["info", "--input", s(&path)]);
    assert!(err.contains("LASF"), "{err}");
}

#[test]
fn kriging_without_search_radius_fails() {
    let dir = tempdir().unwrap();
    let input = fixture(&dir, "plane.las", &plane());
    let output = dir.path().join("krige.asc");

    let err = run_err(&[
        "interpolate-to-grid",
        "--input",
        s(&input),
        "--output",
        s(&output),
        "--method",
        "kriging",
    ]);
    assert!(err.contains("--search-radius"), "{err}");
    assert!(!output.exists(), "no grid should be written on failure");
}
