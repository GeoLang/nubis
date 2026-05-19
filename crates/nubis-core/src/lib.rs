//! Nubis — Point cloud processing engine.
//!
//! LiDAR point cloud operations: classification, ground filtering,
//! thinning, spatial indexing, and statistics.

mod classification;
mod cloud;
mod error;
mod filter;
mod io;
mod normals;
mod octree;

pub use classification::Classification;
pub use cloud::{Point3, PointCloud};
pub use error::Error;
pub use filter::{ground_filter_simple, thin_random, thin_voxel};
pub use io::{CloudStats, LasHeader};
pub use normals::estimate_normals;
pub use octree::Octree;
