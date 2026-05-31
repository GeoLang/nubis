# Nubis

Point cloud processing engine for the GeoLang GIS stack.

[Documentation](https://geolang.github.io/nubis/) · [GitHub](https://github.com/GeoLang/nubis)

## Features

- **LAS I/O** — Read and write LAS point cloud files with header parsing
- **Point cloud types** — `Point3`, `PointCloud` with classification, intensity, and statistics
- **Classification** — ASPRS LAS standard codes (ground, vegetation, building, water, etc.)
- **Ground filtering** — Grid-based progressive morphological filter with configurable cell size and threshold
- **Thinning** — Random sampling and voxel-based decimation
- **IDW interpolation** — Inverse Distance Weighting gridding from scattered points
- **Normal estimation** — Per-point surface normals from local neighborhoods
- **Statistical Outlier Removal (SOR)** — Remove noise points based on mean distance to neighbors
- **Spatial indexing** — Octree with radius queries, configurable leaf size, depth-limited subdivision
- **Geostatistics** — Empirical variograms (spherical, exponential, gaussian, linear, power models), Ordinary Kriging interpolation, Moran's I spatial autocorrelation, Getis-Ord Gi* hot-spot analysis

## Usage

```rust
use nubis_core::{
    Point3, PointCloud, ground_filter_simple, thin_voxel, Octree,
    idw_interpolation, estimate_normals, statistical_outlier_removal,
    read_las, write_las,
};

// Read a LAS file
let cloud = read_las("scan.las").unwrap();

// Ground filtering
let mut cloud = PointCloud::from_points(points);
ground_filter_simple(&mut cloud, 2.0, 0.5);

// IDW interpolation to grid
let grid = idw_interpolation(&cloud, 1.0, 2.0, 12);

// Normal estimation
let normals = estimate_normals(&cloud, 10);

// Statistical Outlier Removal
let cleaned = statistical_outlier_removal(&cloud, 20, 2.0);

// Spatial indexing
let tree = Octree::build(cloud.points(), 64);
let nearby = tree.query_radius(cloud.points(), &query, 5.0);
```

## CLI

```sh
nubis info scan.las
nubis ground --input scan.las --cell-size 2.0 --threshold 0.5 --output ground.las
nubis thin --input scan.las --method voxel --size 0.5 --output thinned.las
```

## License

AGPL-3.0-or-later
