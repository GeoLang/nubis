use crate::Error;
use std::io::Read;

/// LAS file header (version 1.2+).
#[derive(Debug, Clone)]
pub struct LasHeader {
    pub version_major: u8,
    pub version_minor: u8,
    pub point_format: u8,
    pub point_record_length: u16,
    pub number_of_points: u32,
    pub scale_x: f64,
    pub scale_y: f64,
    pub scale_z: f64,
    pub offset_x: f64,
    pub offset_y: f64,
    pub offset_z: f64,
    pub min_x: f64,
    pub min_y: f64,
    pub min_z: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub max_z: f64,
    pub offset_to_points: u32,
}

impl LasHeader {
    /// Read a LAS header from a byte stream.
    pub fn read<R: Read>(reader: &mut R) -> Result<Self, Error> {
        let mut buf = [0u8; 227];
        reader.read_exact(&mut buf).map_err(Error::Io)?;

        // Validate file signature "LASF"
        if &buf[0..4] != b"LASF" {
            return Err(Error::InvalidParameter(
                "not a valid LAS file (missing LASF signature)".to_string(),
            ));
        }

        let version_major = buf[24];
        let version_minor = buf[25];
        let offset_to_points = u32::from_le_bytes([buf[96], buf[97], buf[98], buf[99]]);
        let point_format = buf[104];
        let point_record_length = u16::from_le_bytes([buf[105], buf[106]]);
        let number_of_points = u32::from_le_bytes([buf[107], buf[108], buf[109], buf[110]]);

        let scale_x = f64::from_le_bytes(buf[131..139].try_into().unwrap());
        let scale_y = f64::from_le_bytes(buf[139..147].try_into().unwrap());
        let scale_z = f64::from_le_bytes(buf[147..155].try_into().unwrap());
        let offset_x = f64::from_le_bytes(buf[155..163].try_into().unwrap());
        let offset_y = f64::from_le_bytes(buf[163..171].try_into().unwrap());
        let offset_z = f64::from_le_bytes(buf[171..179].try_into().unwrap());
        let max_x = f64::from_le_bytes(buf[179..187].try_into().unwrap());
        let min_x = f64::from_le_bytes(buf[187..195].try_into().unwrap());
        let max_y = f64::from_le_bytes(buf[195..203].try_into().unwrap());
        let min_y = f64::from_le_bytes(buf[203..211].try_into().unwrap());
        let max_z = f64::from_le_bytes(buf[211..219].try_into().unwrap());
        let min_z = f64::from_le_bytes(buf[219..227].try_into().unwrap());

        Ok(Self {
            version_major,
            version_minor,
            point_format,
            point_record_length,
            number_of_points,
            scale_x,
            scale_y,
            scale_z,
            offset_x,
            offset_y,
            offset_z,
            min_x,
            min_y,
            min_z,
            max_x,
            max_y,
            max_z,
            offset_to_points,
        })
    }
}

/// Statistics about a point cloud.
#[derive(Debug, Clone)]
pub struct CloudStats {
    pub count: usize,
    pub min_x: f64,
    pub min_y: f64,
    pub min_z: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub max_z: f64,
    pub mean_z: f64,
    pub std_z: f64,
}

impl CloudStats {
    pub fn from_cloud(cloud: &crate::PointCloud) -> Option<Self> {
        if cloud.is_empty() {
            return None;
        }
        let (min, max) = cloud.bounds()?;
        let n = cloud.len() as f64;
        let mean_z: f64 = cloud.points().iter().map(|p| p.z).sum::<f64>() / n;
        let var_z: f64 = cloud
            .points()
            .iter()
            .map(|p| (p.z - mean_z).powi(2))
            .sum::<f64>()
            / n;

        Some(Self {
            count: cloud.len(),
            min_x: min.x,
            min_y: min.y,
            min_z: min.z,
            max_x: max.x,
            max_y: max.y,
            max_z: max.z,
            mean_z,
            std_z: var_z.sqrt(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_las_header_invalid_signature() {
        let data = vec![0u8; 227];
        let mut cursor = std::io::Cursor::new(data);
        let result = LasHeader::read(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_cloud_stats() {
        use crate::{Point3, PointCloud};
        let cloud = PointCloud::from_points(vec![
            Point3::new(0.0, 0.0, 10.0),
            Point3::new(1.0, 1.0, 20.0),
            Point3::new(2.0, 2.0, 30.0),
        ]);
        let stats = CloudStats::from_cloud(&cloud).unwrap();
        assert_eq!(stats.count, 3);
        assert!((stats.mean_z - 20.0).abs() < 1e-10);
        assert!((stats.min_z - 10.0).abs() < 1e-10);
        assert!((stats.max_z - 30.0).abs() < 1e-10);
    }
}
