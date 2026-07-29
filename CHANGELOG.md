# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- CLI works on real LAS files: `info`, `ground-classify`, `thin`, `filter-class`,
  `outlier-removal`, `interpolate-to-grid` (idw and kriging), `variogram`, `demo`.
  Grid output is an Esri ASCII grid.
- `filter-class` keeps only chosen classifications, so a classified cloud can be reduced
  to bare earth before gridding.

### Changed

- `Classification` gained an `Other(u8)` variant, so ASPRS codes the enum does not name
  (8, 12, 13-31) keep their value instead of folding to `Unknown`. The enum no longer casts
  with `as u8`, use `Classification::to_u8`.
- `idw_interpolation` averages the returns sharing a grid node instead of taking whichever
  the cloud lists first.
- `thin_voxel` returns its points in a fixed order, so repeated runs write identical files.
- Removed the unused `Variogram` type.

### Fixed

- `write_las` left the header size field at 0, which made every other LAS reader reject
  the output.
- `read_las` treated the whole classification byte as the class, so any point carrying a
  synthetic, key-point, or withheld flag was read as the wrong class.
- `thin_random` used an integer stride, so any fraction above a half kept the whole cloud.
- `write_las` silently saturated coordinates on clouds too large for the LAS scale, and now
  reports the extent it cannot represent.

## [0.1.0] - 2026-05-30

### Added

- Initial release.
