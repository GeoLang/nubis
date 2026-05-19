use crate::Classification;

/// A 3D point with optional classification and intensity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub intensity: u16,
    pub classification: Classification,
}

impl Point3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            x,
            y,
            z,
            intensity: 0,
            classification: Classification::Unclassified,
        }
    }

    pub fn with_classification(mut self, class: Classification) -> Self {
        self.classification = class;
        self
    }

    pub fn with_intensity(mut self, intensity: u16) -> Self {
        self.intensity = intensity;
        self
    }

    pub fn distance_to(&self, other: &Self) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2))
            .sqrt()
    }

    pub fn distance_2d(&self, other: &Self) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

/// A collection of 3D points.
#[derive(Debug, Clone)]
pub struct PointCloud {
    points: Vec<Point3>,
}

impl PointCloud {
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    pub fn from_points(points: Vec<Point3>) -> Self {
        Self { points }
    }

    pub fn push(&mut self, point: Point3) {
        self.points.push(point);
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn points(&self) -> &[Point3] {
        &self.points
    }

    pub fn points_mut(&mut self) -> &mut [Point3] {
        &mut self.points
    }

    /// Compute axis-aligned bounding box: (min, max).
    pub fn bounds(&self) -> Option<(Point3, Point3)> {
        if self.points.is_empty() {
            return None;
        }
        let mut min = Point3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut max = Point3::new(f64::MIN, f64::MIN, f64::MIN);
        for p in &self.points {
            min.x = min.x.min(p.x);
            min.y = min.y.min(p.y);
            min.z = min.z.min(p.z);
            max.x = max.x.max(p.x);
            max.y = max.y.max(p.y);
            max.z = max.z.max(p.z);
        }
        Some((min, max))
    }

    /// Compute the centroid of the point cloud.
    pub fn centroid(&self) -> Option<Point3> {
        if self.points.is_empty() {
            return None;
        }
        let n = self.points.len() as f64;
        let (sx, sy, sz) = self
            .points
            .iter()
            .fold((0.0, 0.0, 0.0), |(x, y, z), p| (x + p.x, y + p.y, z + p.z));
        Some(Point3::new(sx / n, sy / n, sz / n))
    }
}

impl Default for PointCloud {
    fn default() -> Self {
        Self::new()
    }
}
