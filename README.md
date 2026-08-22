# Nubis

[![CI](https://github.com/GeoLang/nubis/actions/workflows/ci.yml/badge.svg)](https://github.com/GeoLang/nubis/actions)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

Point cloud processing engine for the GeoLang GIS stack.

[Documentation](https://geolang.github.io/nubis/) · [GitHub](https://github.com/GeoLang/nubis)

## Features

- **LAS I/O** — Read and write LAS point formats 0-3 from any reader or writer, with header parsing
- **Point cloud types** — `Point3`, `PointCloud` with classification, intensity, and statistics
- **Classification** — ASPRS LAS standard codes (ground, vegetation, building, water, etc.)
- **Ground filtering** — `ground_filter_simple` takes a single-pass minimum-Z per grid cell plus a height threshold, with no opening, no window progression, no slope term and no iteration. `ground_filter_pmf` is the progressive morphological filter (Zhang et al. 2003), opening that surface with windows that grow until buildings and vegetation drop out
- **Thinning** — Random sampling and voxel-based decimation
- **IDW interpolation** — Inverse Distance Weighting gridding from scattered points
- **Normal estimation** — Per-point surface normals from local neighborhoods
- **Statistical Outlier Removal (SOR)** — Remove noise points based on mean distance to neighbors
- **Spatial indexing** — Octree with radius queries, configurable leaf size, depth-limited subdivision
- **Geostatistics** — Empirical variograms (spherical, exponential, gaussian models), Ordinary Kriging interpolation, Moran's I spatial autocorrelation, Getis-Ord Gi* hot-spot analysis

## Usage

```rust
use nubis_core::{
    Point3, PointCloud, ground_filter_simple, thin_voxel, Octree,
    idw_interpolation, estimate_normals, statistical_outlier_removal,
    read_las, write_las,
};

// Read a LAS file
let mut file = std::fs::File::open("scan.las").unwrap();
let cloud = read_las(&mut file).unwrap();

// Ground filtering
let mut cloud = PointCloud::from_points(points);
ground_filter_simple(&mut cloud, 2.0, 0.5);

// IDW interpolation to grid: cell size, power, search radius, min points
let grid = idw_interpolation(&cloud, 1.0, 2.0, 10.0, 3).unwrap();

// Normal estimation
let normals = estimate_normals(&cloud, 10);

// Statistical Outlier Removal
let cleaned = statistical_outlier_removal(&cloud, 20, 2.0);

// Spatial indexing
let tree = Octree::build(cloud.points(), 64);
let nearby = tree.query_radius(cloud.points(), &query, 5.0);
```

## CLI

`nubis` reads LAS point formats 0-3 and writes LAS 1.2 format 0. Gridded output is an
Esri ASCII grid (`.asc`) with values on the grid nodes (`xllcenter`/`yllcenter`).

```sh
# summary: header, bounds, z statistics, classification counts
nubis info --input scan.las

# ground classification
nubis ground-classify --input scan.las --output ground.las --cell-size 2.0 --threshold 0.5

# decimation, voxel (default) or random
nubis thin --input scan.las --output thin.las --voxel-size 1.0
nubis thin --input scan.las --output thin.las --method random --fraction 0.25

# keep only chosen classes, repeat --keep for several
nubis filter-class --input scan.las --output ground.las --keep ground
nubis filter-class --input scan.las --output surfaces.las --keep ground --keep building

# statistical outlier removal
nubis outlier-removal --input scan.las --output clean.las --neighbours 20 --std-multiplier 2.0

# gridding, idw (default) or ordinary kriging with a fitted spherical variogram
nubis interpolate-to-grid --input ground.las --output dem.asc --cell-size 1.0 --search-radius 10.0
nubis interpolate-to-grid --input ground.las --output dem.asc --method kriging --search-radius 10.0

# empirical variogram and fitted spherical model
nubis variogram --input scan.las --bins 10

# synthetic terrain to try the commands on
nubis demo --output demo.las --count 1000
```

Kriging needs `--search-radius` above 0, it also sets the maximum lag used to fit the variogram.
Every command prints a short summary and exits non-zero with a message on stderr on failure.

A bare-earth DEM is three steps, classify then select then grid:

```sh
nubis outlier-removal --input scan.las --output clean.las
nubis ground-classify --input clean.las --output classified.las --cell-size 3.0 --threshold 0.5
nubis filter-class --input classified.las --output bare.las --keep ground
nubis interpolate-to-grid --input bare.las --output dem.asc --cell-size 2.0 --search-radius 4.0
```

Limits worth knowing:

- `--keep` only names the classes `Classification` has variants for. Other codes round trip
  through read and write untouched, but cannot be selected by name.
- Writing uses a 1 mm scale, so a cloud spanning more than about 4295 km on any axis is
  rejected rather than written with saturated coordinates.

## License

AGPL-3.0-or-later, see [LICENSE](LICENSE).

Copyright (C) 2026 Grok Image Compression Inc.
