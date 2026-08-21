use crate::Point3;

/// Simple octree for spatial queries on 3D point clouds.
#[derive(Debug)]
pub struct Octree {
    root: Option<OctreeNode>,
    bounds_min: Point3,
    bounds_max: Point3,
}

#[derive(Debug)]
struct OctreeNode {
    /// Points stored in this leaf (only for leaf nodes).
    points: Vec<usize>,
    /// Child octants (if subdivided).
    children: Option<Box<[Option<OctreeNode>; 8]>>,
    bounds_min: Point3,
    bounds_max: Point3,
}

impl Octree {
    /// Build an octree from a point cloud.
    /// `max_points_per_leaf` controls when to subdivide.
    pub fn build(points: &[Point3], max_points_per_leaf: usize) -> Self {
        if points.is_empty() {
            return Self {
                root: None,
                bounds_min: Point3::new(0.0, 0.0, 0.0),
                bounds_max: Point3::new(0.0, 0.0, 0.0),
            };
        }

        let indices: Vec<usize> = (0..points.len()).collect();
        let (min, max) = node_bounds(points, &indices);
        let root = Self::build_node(points, indices, min, max, max_points_per_leaf, 0);

        Self {
            root: Some(root),
            bounds_min: min,
            bounds_max: max,
        }
    }

    fn build_node(
        points: &[Point3],
        indices: Vec<usize>,
        bounds_min: Point3,
        bounds_max: Point3,
        max_per_leaf: usize,
        depth: usize,
    ) -> OctreeNode {
        if indices.len() <= max_per_leaf || depth >= 20 {
            return OctreeNode {
                points: indices,
                children: None,
                bounds_min,
                bounds_max,
            };
        }

        let center = Point3::new(
            (bounds_min.x + bounds_max.x) / 2.0,
            (bounds_min.y + bounds_max.y) / 2.0,
            (bounds_min.z + bounds_max.z) / 2.0,
        );

        let mut buckets: [Vec<usize>; 8] = Default::default();
        for &idx in &indices {
            let p = &points[idx];
            let octant = ((p.x >= center.x) as usize)
                | (((p.y >= center.y) as usize) << 1)
                | (((p.z >= center.z) as usize) << 2);
            buckets[octant].push(idx);
        }

        // If all points end up in one bucket, just store as leaf
        if buckets.iter().filter(|b| !b.is_empty()).count() <= 1 {
            return OctreeNode {
                points: indices,
                children: None,
                bounds_min,
                bounds_max,
            };
        }

        let children: [Option<OctreeNode>; 8] = std::array::from_fn(|i| {
            if buckets[i].is_empty() {
                None
            } else {
                let bucket = std::mem::take(&mut buckets[i]);
                let (child_min, child_max) = node_bounds(points, &bucket);
                Some(Self::build_node(
                    points,
                    bucket,
                    child_min,
                    child_max,
                    max_per_leaf,
                    depth + 1,
                ))
            }
        });

        OctreeNode {
            points: Vec::new(),
            children: Some(Box::new(children)),
            bounds_min,
            bounds_max,
        }
    }

    /// Find all point indices within `radius` of `query`.
    pub fn query_radius(&self, points: &[Point3], query: &Point3, radius: f64) -> Vec<usize> {
        let mut result = Vec::new();
        if let Some(ref root) = self.root {
            Self::query_node(root, points, query, radius * radius, &mut result);
        }
        result
    }

    fn query_node(
        node: &OctreeNode,
        points: &[Point3],
        query: &Point3,
        radius_sq: f64,
        result: &mut Vec<usize>,
    ) {
        if !sphere_overlaps_aabb(query, radius_sq, &node.bounds_min, &node.bounds_max) {
            return;
        }

        for &idx in &node.points {
            let p = &points[idx];
            let dist_sq =
                (p.x - query.x).powi(2) + (p.y - query.y).powi(2) + (p.z - query.z).powi(2);
            if dist_sq <= radius_sq {
                result.push(idx);
            }
        }

        if let Some(ref children) = node.children {
            for child in children.iter().flatten() {
                Self::query_node(child, points, query, radius_sq, result);
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    pub fn bounds(&self) -> (Point3, Point3) {
        (self.bounds_min, self.bounds_max)
    }
}

fn node_bounds(points: &[Point3], indices: &[usize]) -> (Point3, Point3) {
    let mut min = Point3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut max = Point3::new(f64::MIN, f64::MIN, f64::MIN);
    for &idx in indices {
        let p = &points[idx];
        min.x = min.x.min(p.x);
        min.y = min.y.min(p.y);
        min.z = min.z.min(p.z);
        max.x = max.x.max(p.x);
        max.y = max.y.max(p.y);
        max.z = max.z.max(p.z);
    }
    (min, max)
}

fn sphere_overlaps_aabb(query: &Point3, radius_sq: f64, min: &Point3, max: &Point3) -> bool {
    let dx = query.x.clamp(min.x, max.x) - query.x;
    let dy = query.y.clamp(min.y, max.y) - query.y;
    let dz = query.z.clamp(min.z, max.z) - query.z;
    dx * dx + dy * dy + dz * dz <= radius_sq
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_octree_query() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(10.0, 10.0, 10.0),
        ];
        let tree = Octree::build(&points, 1);
        let result = tree.query_radius(&points, &Point3::new(0.5, 0.5, 0.5), 2.0);
        assert_eq!(result.len(), 2); // first two points
        assert!(result.contains(&0));
        assert!(result.contains(&1));
    }

    #[test]
    fn test_octree_empty() {
        let tree = Octree::build(&[], 10);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_octree_query_prunes_far_cluster() {
        let mut points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.2, 0.0, 0.0),
            Point3::new(0.0, 0.2, 0.0),
            Point3::new(0.0, 0.0, 0.2),
            Point3::new(0.1, 0.1, 0.1),
        ];
        let near_count = points.len();
        points.extend([
            Point3::new(100.0, 100.0, 100.0),
            Point3::new(100.2, 100.0, 100.0),
            Point3::new(100.0, 100.2, 100.0),
            Point3::new(100.0, 100.0, 100.2),
            Point3::new(100.1, 100.1, 100.1),
        ]);
        let tree = Octree::build(&points, 1);

        let near = tree.query_radius(&points, &Point3::new(0.0, 0.0, 0.0), 2.0);
        assert_eq!(near.len(), near_count);
        assert!(near.iter().all(|&i| i < near_count));

        let both = tree.query_radius(&points, &Point3::new(0.0, 0.0, 0.0), 200.0);
        assert_eq!(both.len(), points.len());
    }
}
