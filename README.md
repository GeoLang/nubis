# Nubis

Point cloud processing engine for the TileTopia-HQ GIS stack.

## Features

- **Point cloud types** — `Point3`, `PointCloud` with classification and intensity
- **Classification** — ASPRS LAS standard codes
- **Ground filtering** — grid-based progressive morphological filter
- **Thinning** — random sampling and voxel-based decimation
- **Spatial indexing** — octree with radius queries

## Usage

```rust
use nubis_core::{Point3, PointCloud, ground_filter_simple, thin_voxel, Octree};

let mut cloud = PointCloud::from_points(vec![
    Point3::new(0.0, 0.0, 0.0),
    Point3::new(1.0, 1.0, 5.0),
]);
ground_filter_simple(&mut cloud, 2.0, 0.5);
let thinned = thin_voxel(&cloud, 1.0);
```

## CLI

```sh
nubis info --count 10000
nubis ground --cell-size 2.0 --threshold 0.5
```

## License

AGPL-3.0-or-later
