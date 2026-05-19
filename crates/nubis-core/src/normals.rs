use crate::{Point3, PointCloud};

/// Estimate surface normals for each point using PCA on k-nearest neighbors.
/// Returns a vector of unit normal vectors (nx, ny, nz) for each point.
pub fn estimate_normals(cloud: &PointCloud, k: usize) -> Vec<[f64; 3]> {
    let points = cloud.points();
    let n = points.len();
    let mut normals = Vec::with_capacity(n);

    for i in 0..n {
        let neighbors = find_k_nearest(points, i, k);
        let normal = compute_normal_pca(points, &neighbors);
        normals.push(normal);
    }

    normals
}

/// Find k nearest neighbors of point at index `idx` (brute-force for simplicity).
fn find_k_nearest(points: &[Point3], idx: usize, k: usize) -> Vec<usize> {
    let p = &points[idx];
    let mut dists: Vec<(usize, f64)> = points
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != idx)
        .map(|(i, q)| (i, p.distance_to(q)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.iter().take(k).map(|(i, _)| *i).collect()
}

/// Compute surface normal from covariance matrix using analytical eigensolver.
fn compute_normal_pca(points: &[Point3], indices: &[usize]) -> [f64; 3] {
    if indices.len() < 2 {
        return [0.0, 0.0, 1.0]; // Default up
    }

    // Compute centroid of neighbors
    let n = indices.len() as f64;
    let cx: f64 = indices.iter().map(|&i| points[i].x).sum::<f64>() / n;
    let cy: f64 = indices.iter().map(|&i| points[i].y).sum::<f64>() / n;
    let cz: f64 = indices.iter().map(|&i| points[i].z).sum::<f64>() / n;

    // Compute 3x3 covariance matrix
    let mut cov = [[0.0f64; 3]; 3];
    for &i in indices {
        let dx = points[i].x - cx;
        let dy = points[i].y - cy;
        let dz = points[i].z - cz;
        cov[0][0] += dx * dx;
        cov[0][1] += dx * dy;
        cov[0][2] += dx * dz;
        cov[1][1] += dy * dy;
        cov[1][2] += dy * dz;
        cov[2][2] += dz * dz;
    }
    cov[1][0] = cov[0][1];
    cov[2][0] = cov[0][2];
    cov[2][1] = cov[1][2];

    // Find smallest eigenvector of symmetric 3x3 matrix
    smallest_eigenvector_3x3(&cov)
}

/// Compute smallest eigenvector of a symmetric 3x3 matrix using analytical eigenvalues.
fn smallest_eigenvector_3x3(m: &[[f64; 3]; 3]) -> [f64; 3] {
    // Compute eigenvalues using Cardano's formula for the characteristic polynomial
    let a = m[0][0];
    let b = m[1][1];
    let c = m[2][2];
    let d = m[0][1];
    let e = m[0][2];
    let f = m[1][2];

    let p1 = d * d + e * e + f * f;
    if p1 < 1e-30 {
        // Matrix is diagonal — smallest eigenvalue is min of diagonal
        let min_val = a.min(b).min(c);
        if (a - min_val).abs() < 1e-15 {
            return [1.0, 0.0, 0.0];
        } else if (b - min_val).abs() < 1e-15 {
            return [0.0, 1.0, 0.0];
        } else {
            return [0.0, 0.0, 1.0];
        }
    }

    let q = (a + b + c) / 3.0;
    let p2 = (a - q).powi(2) + (b - q).powi(2) + (c - q).powi(2) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();

    // B = (1/p) * (A - q*I)
    let inv_p = 1.0 / p;
    let b_mat = [
        [(a - q) * inv_p, d * inv_p, e * inv_p],
        [d * inv_p, (b - q) * inv_p, f * inv_p],
        [e * inv_p, f * inv_p, (c - q) * inv_p],
    ];

    let det_b = b_mat[0][0] * (b_mat[1][1] * b_mat[2][2] - b_mat[1][2] * b_mat[2][1])
        - b_mat[0][1] * (b_mat[1][0] * b_mat[2][2] - b_mat[1][2] * b_mat[2][0])
        + b_mat[0][2] * (b_mat[1][0] * b_mat[2][1] - b_mat[1][1] * b_mat[2][0]);

    let r = det_b / 2.0;
    let phi = if r <= -1.0 {
        std::f64::consts::PI / 3.0
    } else if r >= 1.0 {
        0.0
    } else {
        r.acos() / 3.0
    };

    // Eigenvalues in decreasing order
    let eig1 = q + 2.0 * p * phi.cos();
    let eig3 = q + 2.0 * p * (phi + 2.0 * std::f64::consts::PI / 3.0).cos();
    let eig2 = 3.0 * q - eig1 - eig3;

    // Find the smallest eigenvalue
    let lambda = eig1.min(eig2).min(eig3);

    // Find eigenvector for smallest eigenvalue: (A - lambda*I) * v = 0
    // Use cross product of two rows of (A - lambda*I) for robustness
    let row0 = [m[0][0] - lambda, m[0][1], m[0][2]];
    let row1 = [m[1][0], m[1][1] - lambda, m[1][2]];
    let row2 = [m[2][0], m[2][1], m[2][2] - lambda];

    // Try cross product of different row pairs, pick the one with largest magnitude
    let c01 = cross(&row0, &row1);
    let c02 = cross(&row0, &row2);
    let c12 = cross(&row1, &row2);

    let l01 = vec_len(&c01);
    let l02 = vec_len(&c02);
    let l12 = vec_len(&c12);

    let (v, l) = if l01 >= l02 && l01 >= l12 {
        (c01, l01)
    } else if l02 >= l12 {
        (c02, l02)
    } else {
        (c12, l12)
    };

    if l < 1e-15 {
        return [0.0, 0.0, 1.0];
    }
    [v[0] / l, v[1] / l, v[2] / l]
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn vec_len(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_normals_flat_plane() {
        // Points on the XY plane (z=0) — normal should be approximately [0, 0, ±1]
        let cloud = PointCloud::from_points(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.5, 0.5, 0.0),
        ]);
        let normals = estimate_normals(&cloud, 3);
        assert_eq!(normals.len(), 5);
        // All normals should be close to [0, 0, ±1] for a flat plane
        for n in &normals {
            assert!(n[2].abs() > 0.99, "expected z-normal, got {:?}", n);
        }
    }

    #[test]
    fn test_find_k_nearest() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(0.5, 0.0, 0.0),
        ];
        let neighbors = find_k_nearest(&points, 0, 2);
        // Closest to (0,0,0) should be (0.5,0,0) and (1,0,0)
        assert!(neighbors.contains(&3)); // 0.5
        assert!(neighbors.contains(&1)); // 1.0
    }
}
