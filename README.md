# Nubis

[![CI](https://github.com/GeoLang/nubis/actions/workflows/ci.yml/badge.svg)](https://github.com/GeoLang/nubis/actions)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

Point cloud processing engine for the GeoLang GIS stack.

[Documentation](https://geolang.github.io/nubis/) · [GitHub](https://github.com/GeoLang/nubis)

## Features

- **LAS I/O**: `read_las` reads point record formats 0 to 3 from any seekable reader, keeping XYZ, intensity and classification. GPS time and RGB are skipped. `write_las` writes LAS 1.2 format 0 to any writer
- **Point cloud types**: `Point3`, `PointCloud` with classification, intensity, and statistics
- **Classification**: ASPRS codes as `Classification`, 13 named (ground, vegetation, building, water and others) and the rest carried as `Other(code)`
- **Ground filtering**: `ground_filter_simple` takes a single-pass minimum-Z per grid cell plus a height threshold, with no opening, no window progression, no slope term and no iteration. `ground_filter_pmf` is the progressive morphological filter (Zhang et al. 2003), opening that surface with windows that grow until buildings and vegetation drop out
- **Thinning**: `thin_voxel` keeps the first point of each voxel. `thin_random` keeps a fraction of the points at evenly spaced indices. It draws no random numbers, so the same cloud always thins to the same points
- **IDW interpolation**: inverse distance weighting onto a grid. `idw_interpolation` grids the cloud bounds. `idw_window` grids a caller-supplied `GridWindow` for tiled output and bins points at the search radius, so each node only scans nearby points
- **Normal estimation**: `estimate_normals` fits a plane to each point's k nearest neighbours, found by brute-force search
- **Statistical outlier removal**: `statistical_outlier_removal` drops points whose mean distance to their k nearest neighbours is more than a multiple of the standard deviation above the average, also by brute-force search
- **Spatial indexing**: `Octree` with radius queries, a configurable leaf size and subdivision capped at depth 20
- **Geostatistics**: empirical variograms, spherical, exponential and gaussian models (only the spherical one has a fitter, `VariogramModel::fit_spherical`), ordinary kriging onto a grid, Moran's I and Getis-Ord Gi* scores

## Usage

```rust
use nubis_core::{
    Octree, estimate_normals, ground_filter_simple, idw_interpolation, read_las,
    statistical_outlier_removal, write_las,
};

// Read a LAS file
let mut file = std::fs::File::open("scan.las").unwrap();
let mut cloud = read_las(&mut file).unwrap();

// Ground filtering: cell size, height threshold
ground_filter_simple(&mut cloud, 2.0, 0.5);

// IDW interpolation to grid: cell size, power, search radius, min points
let grid = idw_interpolation(&cloud, 1.0, 2.0, 10.0, 3).unwrap();

// Normals from the 10 nearest neighbours
let normals = estimate_normals(&cloud, 10);

// Outlier removal: 20 neighbours, 2 standard deviations
let cleaned = statistical_outlier_removal(&cloud, 20, 2.0);

// Octree with up to 64 points per leaf, indices within 5 units of the first point
let tree = Octree::build(cloud.points(), 64);
let nearby = tree.query_radius(cloud.points(), &cloud.points()[0], 5.0);

// Write LAS 1.2
let mut out = std::fs::File::create("clean.las").unwrap();
write_las(&cleaned, &mut out).unwrap();
```

## CLI

`cargo install --path crates/nubis-cli` builds the `nubis` binary. Tagged releases
attach it for Linux and macOS on x86_64 and aarch64.

`nubis` reads LAS point formats 0-3 and writes LAS 1.2 format 0. Gridded output is an
Esri ASCII grid (`.asc`) with values on the grid nodes (`xllcenter`/`yllcenter`) and
`-9999` as nodata.

```sh
# summary: header, bounds, z statistics, classification counts
nubis info --input scan.las

# ground classification, single-pass minimum-Z per cell
nubis ground-classify --input scan.las --output ground.las --cell-size 2.0 --threshold 0.5

# ground classification, progressive morphological filter
nubis pmf --input scan.las --output ground.las --cell-size 1.0 --max-window-size 33.0 \
  --slope 0.15 --initial-distance 0.5 --max-distance 3.0

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

# empirical variogram and fitted spherical model, --max-lag defaults to half the cloud diagonal
nubis variogram --input scan.las --bins 10 --max-lag 25.0

# synthetic terrain to try the commands on
nubis demo --output demo.las --count 1000
```

Kriging needs `--search-radius` above 0. It also sets the maximum lag used to fit the variogram.
Every command prints a short summary and exits non-zero with a message on stderr on failure.

A bare-earth DEM is four steps, clean then classify then select then grid:

```sh
nubis outlier-removal --input scan.las --output clean.las
nubis ground-classify --input clean.las --output classified.las --cell-size 3.0 --threshold 0.5
nubis filter-class --input classified.las --output bare.las --keep ground
nubis interpolate-to-grid --input bare.las --output dem.asc --cell-size 2.0 --search-radius 4.0
```

Limits worth knowing:

- `--keep` takes a name only for the 13 classes `Classification` has variants for. Any other
  code is selected by its bare number, 0 to 31, and round trips through read and write with
  its value intact.
- Writing uses a 1 mm scale, so a cloud spanning more than about 4295 km on any axis is
  rejected rather than written with saturated coordinates.

## License

AGPL-3.0-or-later, see [LICENSE](LICENSE).

Copyright (C) 2026 Grok Image Compression Inc.
