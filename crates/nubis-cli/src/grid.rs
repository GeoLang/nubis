use nubis_core::InterpolatedGrid;
use std::io::{self, Write};

/// nodata marker written to the .asc file, whatever the grid uses internally
const NODATA: f64 = -9999.0;

/// Write an Esri ASCII grid.
///
/// Cell values sit on the grid nodes, so the header uses xllcenter/yllcenter.
/// Rows run north to south, the reverse of the grid's row order.
pub fn write_ascii_grid<W: Write>(grid: &InterpolatedGrid, writer: &mut W) -> io::Result<()> {
    writeln!(writer, "ncols {}", grid.width)?;
    writeln!(writer, "nrows {}", grid.height)?;
    writeln!(writer, "xllcenter {}", grid.origin_x)?;
    writeln!(writer, "yllcenter {}", grid.origin_y)?;
    writeln!(writer, "cellsize {}", grid.cell_size)?;
    writeln!(writer, "NODATA_value {NODATA}")?;

    for row in (0..grid.height).rev() {
        for col in 0..grid.width {
            if col > 0 {
                write!(writer, " ")?;
            }
            let value = grid.data[row * grid.width + col];
            if is_nodata(value, grid.nodata) {
                write!(writer, "{NODATA}")?;
            } else {
                write!(writer, "{value:.3}")?;
            }
        }
        writeln!(writer)?;
    }

    Ok(())
}

/// Number of cells that got a value.
pub fn filled_cells(grid: &InterpolatedGrid) -> usize {
    grid.data
        .iter()
        .filter(|v| !is_nodata(**v, grid.nodata))
        .count()
}

fn is_nodata(value: f64, nodata: f64) -> bool {
    value.is_nan() || value == nodata
}
