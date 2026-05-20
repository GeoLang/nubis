use crate::{Classification, Error};
use std::io::{Read, Seek, SeekFrom};

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

/// Read a LAS file (header + point records) from a seekable stream.
///
/// Supports point record formats 0-3:
/// - Format 0: XYZ + intensity + classification (20 bytes)
/// - Format 1: Format 0 + GPS time (28 bytes)
/// - Format 2: Format 0 + RGB (26 bytes)
/// - Format 3: Format 0 + GPS time + RGB (34 bytes)
///
/// Returns a PointCloud with XYZ coordinates (scaled and offset-applied),
/// intensity, and classification.
pub fn read_las<R: Read + Seek>(reader: &mut R) -> Result<crate::PointCloud, Error> {
    let header = LasHeader::read(reader)?;

    if header.point_format > 3 {
        return Err(Error::InvalidParameter(format!(
            "unsupported point format: {} (only 0-3 supported)",
            header.point_format
        )));
    }

    // Seek to start of point data
    reader
        .seek(SeekFrom::Start(header.offset_to_points as u64))
        .map_err(Error::Io)?;

    let mut points = Vec::with_capacity(header.number_of_points as usize);
    let record_len = header.point_record_length as usize;
    let mut record_buf = vec![0u8; record_len];

    for _ in 0..header.number_of_points {
        reader.read_exact(&mut record_buf).map_err(Error::Io)?;

        // All formats start with: X(i32), Y(i32), Z(i32), intensity(u16)
        let raw_x = i32::from_le_bytes(record_buf[0..4].try_into().unwrap());
        let raw_y = i32::from_le_bytes(record_buf[4..8].try_into().unwrap());
        let raw_z = i32::from_le_bytes(record_buf[8..12].try_into().unwrap());
        let intensity = u16::from_le_bytes(record_buf[12..14].try_into().unwrap());

        // Byte 15: return number/flags, byte 16: classification
        let classification = Classification::from_u8(record_buf[15]);

        let x = raw_x as f64 * header.scale_x + header.offset_x;
        let y = raw_y as f64 * header.scale_y + header.offset_y;
        let z = raw_z as f64 * header.scale_z + header.offset_z;

        let mut pt = crate::Point3::new(x, y, z);
        pt.intensity = intensity;
        pt.classification = classification;

        points.push(pt);
    }

    Ok(crate::PointCloud::from_points(points))
}

/// Write a point cloud as a minimal LAS 1.2 file (format 0).
pub fn write_las<W: std::io::Write>(
    cloud: &crate::PointCloud,
    writer: &mut W,
) -> Result<(), Error> {
    if cloud.is_empty() {
        return Err(Error::EmptyCloud);
    }

    let (min, max) = cloud.bounds().ok_or(Error::EmptyCloud)?;

    // Use scale that gives millimeter precision
    let scale: f64 = 0.001;
    let offset_x = (min.x + max.x) / 2.0;
    let offset_y = (min.y + max.y) / 2.0;
    let offset_z = (min.z + max.z) / 2.0;

    let point_format: u8 = 0;
    let point_record_length: u16 = 20;
    let offset_to_points: u32 = 227;
    let num_points = cloud.len() as u32;

    // Write header (227 bytes)
    let mut header = vec![0u8; 227];

    // Signature
    header[0..4].copy_from_slice(b"LASF");
    // Version 1.2
    header[24] = 1;
    header[25] = 2;
    // System identifier (32 bytes at offset 26)
    header[26..32].copy_from_slice(b"Nubis\0");
    // Generating software (32 bytes at offset 58)
    header[58..64].copy_from_slice(b"Nubis\0");
    // Offset to point data
    header[96..100].copy_from_slice(&offset_to_points.to_le_bytes());
    // Point data format
    header[104] = point_format;
    // Point record length
    header[105..107].copy_from_slice(&point_record_length.to_le_bytes());
    // Number of point records
    header[107..111].copy_from_slice(&num_points.to_le_bytes());

    // Scale factors
    header[131..139].copy_from_slice(&scale.to_le_bytes());
    header[139..147].copy_from_slice(&scale.to_le_bytes());
    header[147..155].copy_from_slice(&scale.to_le_bytes());
    // Offsets
    header[155..163].copy_from_slice(&offset_x.to_le_bytes());
    header[163..171].copy_from_slice(&offset_y.to_le_bytes());
    header[171..179].copy_from_slice(&offset_z.to_le_bytes());
    // Bounds
    header[179..187].copy_from_slice(&max.x.to_le_bytes());
    header[187..195].copy_from_slice(&min.x.to_le_bytes());
    header[195..203].copy_from_slice(&max.y.to_le_bytes());
    header[203..211].copy_from_slice(&min.y.to_le_bytes());
    header[211..219].copy_from_slice(&max.z.to_le_bytes());
    header[219..227].copy_from_slice(&min.z.to_le_bytes());

    writer.write_all(&header)?;

    // Write point records (format 0: 20 bytes each)
    for pt in cloud.points() {
        let raw_x = ((pt.x - offset_x) / scale).round() as i32;
        let raw_y = ((pt.y - offset_y) / scale).round() as i32;
        let raw_z = ((pt.z - offset_z) / scale).round() as i32;

        writer.write_all(&raw_x.to_le_bytes())?;
        writer.write_all(&raw_y.to_le_bytes())?;
        writer.write_all(&raw_z.to_le_bytes())?;
        writer.write_all(&pt.intensity.to_le_bytes())?;
        // Return number + number of returns (byte 14)
        writer.write_all(&[0x11])?; // 1 return, return #1
        // Classification (byte 15)
        writer.write_all(&[pt.classification as u8])?;
        // Scan angle rank, user data, point source id (4 bytes padding)
        writer.write_all(&[0u8; 4])?;
    }

    Ok(())
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

    #[test]
    fn test_las_roundtrip() {
        use crate::{Point3, PointCloud};

        let points = vec![
            Point3::new(100.0, 200.0, 50.0)
                .with_intensity(1000)
                .with_classification(crate::Classification::Ground),
            Point3::new(100.5, 200.5, 51.0)
                .with_intensity(2000)
                .with_classification(crate::Classification::Building),
            Point3::new(101.0, 201.0, 52.0)
                .with_intensity(500)
                .with_classification(crate::Classification::HighVegetation),
        ];
        let cloud = PointCloud::from_points(points);

        // Write
        let mut buf = Vec::new();
        write_las(&cloud, &mut buf).unwrap();

        // Read back
        let mut cursor = std::io::Cursor::new(buf);
        let read_cloud = read_las(&mut cursor).unwrap();

        assert_eq!(read_cloud.len(), 3);

        // Check coordinates (should match to ~millimeter due to 0.001 scale)
        let pts = read_cloud.points();
        assert!((pts[0].x - 100.0).abs() < 0.002);
        assert!((pts[0].y - 200.0).abs() < 0.002);
        assert!((pts[0].z - 50.0).abs() < 0.002);
        assert_eq!(pts[0].intensity, 1000);

        assert!((pts[1].x - 100.5).abs() < 0.002);
        assert!((pts[2].z - 52.0).abs() < 0.002);
    }

    #[test]
    fn test_las_write_empty_cloud() {
        use crate::PointCloud;
        let cloud = PointCloud::from_points(vec![]);
        let mut buf = Vec::new();
        let result = write_las(&cloud, &mut buf);
        assert!(result.is_err());
    }
}
