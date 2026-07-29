//! LAS reading and writing against hand-built files.
//!
//! The reader claims point formats 0-3 but `write_las` only ever emits format 0,
//! so the fixtures here are assembled byte by byte to cover the other formats.

use nubis_core::{Classification, LasHeader, Point3, PointCloud, read_las, write_las};
use std::io::Cursor;

const HEADER_LEN: usize = 227;

fn record_len(point_format: u8) -> usize {
    match point_format {
        0 => 20,
        1 => 28,
        2 => 26,
        3 => 34,
        other => panic!("no record length for format {other}"),
    }
}

/// A point as it is stored on disk: raw coordinates, intensity, and the packed
/// classification byte (class in bits 0-4, flags in bits 5-7).
struct RawPoint {
    x: i32,
    y: i32,
    z: i32,
    intensity: u16,
    class_byte: u8,
}

/// Assemble a LAS 1.2 file with the given point format.
fn las_bytes(point_format: u8, scale: f64, offset: f64, points: &[RawPoint]) -> Vec<u8> {
    let len = record_len(point_format);
    let mut buf = vec![0u8; HEADER_LEN];

    buf[0..4].copy_from_slice(b"LASF");
    buf[24] = 1;
    buf[25] = 2;
    buf[94..96].copy_from_slice(&(HEADER_LEN as u16).to_le_bytes());
    buf[96..100].copy_from_slice(&(HEADER_LEN as u32).to_le_bytes());
    buf[104] = point_format;
    buf[105..107].copy_from_slice(&(len as u16).to_le_bytes());
    buf[107..111].copy_from_slice(&(points.len() as u32).to_le_bytes());
    for i in 0..3 {
        let at = 131 + i * 8;
        buf[at..at + 8].copy_from_slice(&scale.to_le_bytes());
    }
    for i in 0..3 {
        let at = 155 + i * 8;
        buf[at..at + 8].copy_from_slice(&offset.to_le_bytes());
    }

    for p in points {
        let mut rec = vec![0u8; len];
        rec[0..4].copy_from_slice(&p.x.to_le_bytes());
        rec[4..8].copy_from_slice(&p.y.to_le_bytes());
        rec[8..12].copy_from_slice(&p.z.to_le_bytes());
        rec[12..14].copy_from_slice(&p.intensity.to_le_bytes());
        rec[14] = 0x11;
        rec[15] = p.class_byte;
        buf.extend_from_slice(&rec);
    }

    buf
}

fn raw(x: i32, y: i32, z: i32, class_byte: u8) -> RawPoint {
    RawPoint {
        x,
        y,
        z,
        intensity: 0,
        class_byte,
    }
}

fn roundtrip(cloud: &PointCloud) -> PointCloud {
    let mut buf = Vec::new();
    write_las(cloud, &mut buf).expect("write");
    read_las(&mut Cursor::new(buf)).expect("read")
}

// ── the reader on files nubis does not write ──────────────────────────────

#[test]
fn reads_every_supported_point_format() {
    // same three points in each format, only the record length differs
    let points = [
        raw(100, 200, 300, 2),
        raw(-500, 1500, 42, 6),
        raw(0, 0, -750, 5),
    ];

    for format in [0u8, 1, 2, 3] {
        let bytes = las_bytes(format, 0.01, 0.0, &points);
        let cloud = read_las(&mut Cursor::new(bytes)).unwrap_or_else(|e| {
            panic!("format {format} failed to read: {e}");
        });

        assert_eq!(cloud.len(), 3, "format {format} point count");
        let p = cloud.points();
        assert!((p[0].x - 1.0).abs() < 1e-9, "format {format} x");
        assert!((p[0].y - 2.0).abs() < 1e-9, "format {format} y");
        assert!((p[0].z - 3.0).abs() < 1e-9, "format {format} z");
        assert!((p[1].x - -5.0).abs() < 1e-9, "format {format} negative x");
        assert!((p[2].z - -7.5).abs() < 1e-9, "format {format} negative z");
        assert_eq!(p[1].classification, Classification::Building);
        assert_eq!(p[2].classification, Classification::HighVegetation);
    }
}

#[test]
fn applies_scale_and_offset_from_the_header() {
    // raw 12345 at scale 0.001 with offset 500000 is 500012.345
    let bytes = las_bytes(0, 0.001, 500_000.0, &[raw(12_345, 12_345, 12_345, 2)]);
    let cloud = read_las(&mut Cursor::new(bytes)).unwrap();
    let p = cloud.points()[0];
    assert!((p.x - 500_012.345).abs() < 1e-6, "got {}", p.x);
    assert!((p.z - 500_012.345).abs() < 1e-6);
}

#[test]
fn classification_flag_bits_do_not_corrupt_the_class() {
    // bits 5-7 are the synthetic / key-point / withheld flags, not part of the class
    for flags in [0x00u8, 0x20, 0x40, 0x80, 0xe0] {
        let bytes = las_bytes(
            0,
            0.01,
            0.0,
            &[raw(0, 0, 0, 2 | flags), raw(1, 1, 1, 6 | flags)],
        );
        let cloud = read_las(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(
            cloud.points()[0].classification,
            Classification::Ground,
            "ground point misread with flags {flags:#04x}"
        );
        assert_eq!(
            cloud.points()[1].classification,
            Classification::Building,
            "building point misread with flags {flags:#04x}"
        );
    }
}

#[test]
fn class_codes_without_a_name_keep_their_value() {
    for code in [8u8, 12, 13, 20, 31] {
        let bytes = las_bytes(0, 0.01, 0.0, &[raw(0, 0, 0, code)]);
        let cloud = read_las(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(
            cloud.points()[0].classification,
            Classification::Other(code),
            "code {code} was not carried through"
        );
    }
}

#[test]
fn roundtrip_preserves_every_class_code_the_field_can_hold() {
    // the class is 5 bits, so 0-31 must all survive read, write, and read again
    let points: Vec<RawPoint> = (0..=31u8)
        .map(|code| raw(code as i32, 0, 0, code))
        .collect();
    let cloud = read_las(&mut Cursor::new(las_bytes(0, 0.01, 0.0, &points))).unwrap();
    assert_eq!(cloud.len(), 32);

    let back = roundtrip(&cloud);
    assert_eq!(back.len(), 32);
    for (i, point) in back.points().iter().enumerate() {
        let code = i as u8;
        assert_eq!(
            point.classification,
            Classification::from_u8(code),
            "code {code} did not survive the round trip"
        );
        assert_eq!(
            point.classification.to_u8(),
            code,
            "code {code} changed value"
        );
    }
}

#[test]
fn named_codes_still_map_to_their_own_variants() {
    // Other is only for codes with no name, it must not swallow the named ones
    for (code, expected) in [
        (0u8, Classification::Unclassified),
        (1, Classification::Unknown),
        (2, Classification::Ground),
        (7, Classification::LowPoint),
        (9, Classification::Water),
        (17, Classification::BridgeDeck),
        (18, Classification::HighNoise),
    ] {
        assert_eq!(Classification::from_u8(code), expected);
        assert_eq!(expected.to_u8(), code);
    }
}

#[test]
fn an_out_of_range_class_cannot_reach_the_flag_bits() {
    // the field is 5 bits wide, so a hand-built code above 31 must not corrupt
    // the synthetic / key-point / withheld flags packed into the same byte
    let cloud = PointCloud::from_points(vec![
        Point3::new(0.0, 0.0, 0.0).with_classification(Classification::Other(200)),
    ]);
    let mut buf = Vec::new();
    write_las(&cloud, &mut buf).unwrap();
    assert_eq!(buf[HEADER_LEN + 15] & 0xe0, 0, "flag bits were written");
}

#[test]
fn rejects_a_file_without_the_lasf_signature() {
    let mut bytes = las_bytes(0, 0.01, 0.0, &[raw(0, 0, 0, 2)]);
    bytes[0..4].copy_from_slice(b"XXXX");
    assert!(read_las(&mut Cursor::new(bytes)).is_err());
}

#[test]
fn rejects_point_formats_above_three() {
    let mut bytes = las_bytes(0, 0.01, 0.0, &[raw(0, 0, 0, 2)]);
    bytes[104] = 6;
    let err = read_las(&mut Cursor::new(bytes)).unwrap_err().to_string();
    assert!(err.contains("unsupported point format"), "got: {err}");
}

#[test]
fn rejects_a_file_truncated_mid_point_record() {
    let mut bytes = las_bytes(0, 0.01, 0.0, &[raw(0, 0, 0, 2), raw(1, 1, 1, 2)]);
    bytes.truncate(bytes.len() - 8); // second record is now short
    assert!(read_las(&mut Cursor::new(bytes)).is_err());
}

#[test]
fn rejects_a_header_shorter_than_the_public_block() {
    let bytes = vec![0u8; 100];
    assert!(LasHeader::read(&mut Cursor::new(bytes)).is_err());
}

#[test]
fn header_reports_the_declared_point_count_and_format() {
    let bytes = las_bytes(1, 0.01, 0.0, &[raw(0, 0, 0, 2), raw(1, 1, 1, 2)]);
    let header = LasHeader::read(&mut Cursor::new(bytes)).unwrap();
    assert_eq!(header.version_major, 1);
    assert_eq!(header.version_minor, 2);
    assert_eq!(header.point_format, 1);
    assert_eq!(header.point_record_length, 28);
    assert_eq!(header.number_of_points, 2);
    assert_eq!(header.offset_to_points, HEADER_LEN as u32);
}

// ── the writer ────────────────────────────────────────────────────────────

#[test]
fn writes_the_header_size_field() {
    // regression: leaving this at 0 makes every other LAS reader reject the file
    let cloud = PointCloud::from_points(vec![Point3::new(1.0, 2.0, 3.0)]);
    let mut buf = Vec::new();
    write_las(&cloud, &mut buf).unwrap();

    let header_size = u16::from_le_bytes([buf[94], buf[95]]);
    assert_eq!(header_size, HEADER_LEN as u16);
    let offset_to_points = u32::from_le_bytes([buf[96], buf[97], buf[98], buf[99]]);
    assert_eq!(offset_to_points, HEADER_LEN as u32);
}

#[test]
fn writes_one_fixed_size_record_per_point() {
    let cloud = PointCloud::from_points(vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 1.0, 1.0),
        Point3::new(2.0, 2.0, 2.0),
    ]);
    let mut buf = Vec::new();
    write_las(&cloud, &mut buf).unwrap();
    assert_eq!(buf.len(), HEADER_LEN + 3 * 20);
    assert_eq!(buf[104], 0, "point format");
    assert_eq!(u16::from_le_bytes([buf[105], buf[106]]), 20);
}

#[test]
fn writes_the_cloud_bounds_into_the_header() {
    let cloud = PointCloud::from_points(vec![
        Point3::new(-10.0, 5.0, 100.0),
        Point3::new(30.0, 45.0, 250.0),
    ]);
    let mut buf = Vec::new();
    write_las(&cloud, &mut buf).unwrap();

    let f = |at: usize| f64::from_le_bytes(buf[at..at + 8].try_into().unwrap());
    assert!((f(179) - 30.0).abs() < 1e-9, "max x");
    assert!((f(187) - -10.0).abs() < 1e-9, "min x");
    assert!((f(195) - 45.0).abs() < 1e-9, "max y");
    assert!((f(203) - 5.0).abs() < 1e-9, "min y");
    assert!((f(211) - 250.0).abs() < 1e-9, "max z");
    assert!((f(219) - 100.0).abs() < 1e-9, "min z");
}

#[test]
fn refuses_to_write_an_extent_the_coordinate_field_cannot_hold() {
    // coordinates are i32 counts of a 1 mm scale measured from the midpoint, so
    // anything past about 4295 km across cannot be represented and used to saturate
    let too_wide = PointCloud::from_points(vec![
        Point3::new(-3_000_000.0, 0.0, 0.0),
        Point3::new(3_000_000.0, 0.0, 0.0),
    ]);
    let mut buf = Vec::new();
    let err = write_las(&too_wide, &mut buf).unwrap_err().to_string();
    assert!(err.contains("extent"), "unhelpful error: {err}");
    assert!(err.starts_with("invalid parameter"), "got: {err}");

    // just inside the limit still writes, and the extreme points survive
    let widest = i32::MAX as f64 * 0.001 * 2.0;
    let half = widest / 2.0 - 1.0;
    let ok = PointCloud::from_points(vec![
        Point3::new(-half, 0.0, 0.0),
        Point3::new(half, 0.0, 0.0),
    ]);
    let back = roundtrip(&ok);
    assert_eq!(back.len(), 2);
    assert!((back.points()[0].x - -half).abs() < 1.0);
    assert!((back.points()[1].x - half).abs() < 1.0);
}

#[test]
fn refuses_to_write_a_non_finite_extent() {
    let cloud = PointCloud::from_points(vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(f64::INFINITY, 0.0, 0.0),
    ]);
    let mut buf = Vec::new();
    assert!(write_las(&cloud, &mut buf).is_err());
}

#[test]
fn refuses_to_write_an_empty_cloud() {
    let mut buf = Vec::new();
    assert!(write_las(&PointCloud::new(), &mut buf).is_err());
    assert!(buf.is_empty(), "nothing should reach the writer");
}

// ── round trips ───────────────────────────────────────────────────────────

#[test]
fn roundtrip_preserves_every_named_classification() {
    let classes = [
        Classification::Unclassified,
        Classification::Unknown,
        Classification::Ground,
        Classification::LowVegetation,
        Classification::MediumVegetation,
        Classification::HighVegetation,
        Classification::Building,
        Classification::LowPoint,
        Classification::Water,
        Classification::Rail,
        Classification::Road,
        Classification::BridgeDeck,
        Classification::HighNoise,
    ];
    let points: Vec<Point3> = classes
        .iter()
        .enumerate()
        .map(|(i, c)| Point3::new(i as f64, 0.0, 0.0).with_classification(*c))
        .collect();

    let back = roundtrip(&PointCloud::from_points(points));
    assert_eq!(back.len(), classes.len());
    for (point, expected) in back.points().iter().zip(classes.iter()) {
        assert_eq!(point.classification, *expected);
    }
}

#[test]
fn roundtrip_preserves_intensity() {
    let points = vec![
        Point3::new(0.0, 0.0, 0.0).with_intensity(0),
        Point3::new(1.0, 0.0, 0.0).with_intensity(1),
        Point3::new(2.0, 0.0, 0.0).with_intensity(30_000),
        Point3::new(3.0, 0.0, 0.0).with_intensity(u16::MAX),
    ];
    let back = roundtrip(&PointCloud::from_points(points));
    let got: Vec<u16> = back.points().iter().map(|p| p.intensity).collect();
    assert_eq!(got, vec![0, 1, 30_000, u16::MAX]);
}

#[test]
fn roundtrip_of_a_single_point_keeps_its_position() {
    // degenerate bounds: min == max, so the header offset lands on the point itself
    let cloud = PointCloud::from_points(vec![
        Point3::new(123.456, -78.9, 1000.5).with_classification(Classification::Water),
    ]);
    let back = roundtrip(&cloud);
    assert_eq!(back.len(), 1);
    let p = back.points()[0];
    assert!((p.x - 123.456).abs() < 1e-3, "x was {}", p.x);
    assert!((p.y - -78.9).abs() < 1e-3, "y was {}", p.y);
    assert!((p.z - 1000.5).abs() < 1e-3, "z was {}", p.z);
    assert_eq!(p.classification, Classification::Water);
}

#[test]
fn roundtrip_keeps_utm_scale_coordinates_to_the_millimetre() {
    // a realistic projected tile: 6-figure eastings, 7-figure northings, negative z
    let points = vec![
        Point3::new(499_123.456, 6_100_987.654, -412.345),
        Point3::new(499_623.999, 6_101_487.001, 87.001),
        Point3::new(500_123.001, 6_101_986.500, 1_234.567),
    ];
    let back = roundtrip(&PointCloud::from_points(points.clone()));

    for (got, want) in back.points().iter().zip(points.iter()) {
        assert!((got.x - want.x).abs() < 1e-3, "x {} vs {}", got.x, want.x);
        assert!((got.y - want.y).abs() < 1e-3, "y {} vs {}", got.y, want.y);
        assert!((got.z - want.z).abs() < 1e-3, "z {} vs {}", got.z, want.z);
    }
}

#[test]
fn roundtrip_keeps_bounds_and_count_of_a_larger_cloud() {
    let mut points = Vec::new();
    for i in 0..2_000 {
        let x = (i % 50) as f64 * 0.37;
        let y = (i / 50) as f64 * 0.41;
        points.push(Point3::new(x, y, 10.0 + (x * 0.1).sin()));
    }
    let cloud = PointCloud::from_points(points);
    let (min, max) = cloud.bounds().unwrap();

    let back = roundtrip(&cloud);
    assert_eq!(back.len(), cloud.len());
    let (bmin, bmax) = back.bounds().unwrap();
    assert!((bmin.x - min.x).abs() < 1e-3);
    assert!((bmax.x - max.x).abs() < 1e-3);
    assert!((bmin.z - min.z).abs() < 1e-3);
    assert!((bmax.z - max.z).abs() < 1e-3);
}
