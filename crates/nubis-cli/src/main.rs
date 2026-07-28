use clap::{Args, Parser, Subcommand, ValueEnum};
use nubis_core::{
    Classification, CloudStats, LasHeader, Point3, PointCloud, VariogramModel, empirical_variogram,
    ground_filter_simple, idw_interpolation, ordinary_kriging, read_las,
    statistical_outlier_removal, thin_random, thin_voxel, write_las,
};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Seek, Write};
use std::path::{Path, PathBuf};

mod grid;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(name = "nubis", version, about = "Point cloud processing CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Summarize a LAS file: header, bounds, statistics, classification counts
    Info {
        /// Input LAS file
        #[arg(long)]
        input: PathBuf,
    },
    /// Classify ground points and write a new LAS file
    GroundClassify {
        /// Input LAS file
        #[arg(long)]
        input: PathBuf,
        /// Output LAS file
        #[arg(long)]
        output: PathBuf,
        /// Grid cell size
        #[arg(long, default_value_t = 1.0)]
        cell_size: f64,
        /// Height above local minimum still counted as ground
        #[arg(long, default_value_t = 0.5)]
        threshold: f64,
    },
    /// Decimate a LAS file and write the result
    Thin {
        /// Input LAS file
        #[arg(long)]
        input: PathBuf,
        /// Output LAS file
        #[arg(long)]
        output: PathBuf,
        /// Thinning method
        #[arg(long, value_enum, default_value_t = ThinMethod::Voxel)]
        method: ThinMethod,
        /// Voxel edge length (voxel method)
        #[arg(long, default_value_t = 1.0)]
        voxel_size: f64,
        /// Fraction of points to keep (random method)
        #[arg(long, default_value_t = 0.5)]
        fraction: f64,
    },
    /// Remove statistical outliers and write a new LAS file
    OutlierRemoval {
        /// Input LAS file
        #[arg(long)]
        input: PathBuf,
        /// Output LAS file
        #[arg(long)]
        output: PathBuf,
        /// Neighbours used for the mean-distance test
        #[arg(long, default_value_t = 20)]
        neighbours: usize,
        /// Standard deviations above the mean distance before a point is dropped
        #[arg(long, default_value_t = 2.0)]
        std_multiplier: f64,
    },
    /// Interpolate elevations onto a regular grid, written as an Esri ASCII grid
    InterpolateToGrid(InterpolateArgs),
    /// Compute an empirical variogram and fit a spherical model
    Variogram {
        /// Input LAS file
        #[arg(long)]
        input: PathBuf,
        /// Number of lag bins
        #[arg(long, default_value_t = 10)]
        bins: usize,
        /// Largest lag distance, defaults to half the cloud diagonal
        #[arg(long)]
        max_lag: Option<f64>,
    },
    /// Write a synthetic terrain LAS file to experiment with
    Demo {
        /// Output LAS file
        #[arg(long)]
        output: PathBuf,
        /// Number of points to generate
        #[arg(long, default_value_t = 1000)]
        count: usize,
    },
}

#[derive(Args)]
struct InterpolateArgs {
    /// Input LAS file
    #[arg(long)]
    input: PathBuf,
    /// Output .asc grid
    #[arg(long)]
    output: PathBuf,
    /// Interpolation method
    #[arg(long, value_enum, default_value_t = GridMethod::Idw)]
    method: GridMethod,
    /// Output cell size
    #[arg(long, default_value_t = 1.0)]
    cell_size: f64,
    /// Distance exponent (idw)
    #[arg(long, default_value_t = 2.0)]
    power: f64,
    /// Search radius, 0 means unlimited for idw, required for kriging
    #[arg(long, default_value_t = 0.0)]
    search_radius: f64,
    /// Points needed before a cell gets a value (idw)
    #[arg(long, default_value_t = 1)]
    min_points: usize,
    /// Bins used to fit the variogram (kriging)
    #[arg(long, default_value_t = 10)]
    variogram_bins: usize,
}

#[derive(Clone, Copy, ValueEnum)]
enum ThinMethod {
    Voxel,
    Random,
}

#[derive(Clone, Copy, ValueEnum)]
enum GridMethod {
    Idw,
    Kriging,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    match Cli::parse().command {
        Commands::Info { input } => info(&input),
        Commands::GroundClassify {
            input,
            output,
            cell_size,
            threshold,
        } => ground_classify(&input, &output, cell_size, threshold),
        Commands::Thin {
            input,
            output,
            method,
            voxel_size,
            fraction,
        } => thin(&input, &output, method, voxel_size, fraction),
        Commands::OutlierRemoval {
            input,
            output,
            neighbours,
            std_multiplier,
        } => outlier_removal(&input, &output, neighbours, std_multiplier),
        Commands::InterpolateToGrid(args) => interpolate_to_grid(&args),
        Commands::Variogram {
            input,
            bins,
            max_lag,
        } => variogram(&input, bins, max_lag),
        Commands::Demo { output, count } => demo(&output, count),
    }
}

fn info(input: &Path) -> Result<()> {
    let mut reader = open(input)?;
    let header = LasHeader::read(&mut reader).map_err(|e| context(input, e))?;
    reader.rewind()?;
    let cloud = read_las(&mut reader).map_err(|e| context(input, e))?;
    let stats = CloudStats::from_cloud(&cloud).ok_or("file contains no points")?;
    let centroid = cloud.centroid().ok_or("file contains no points")?;

    println!("File: {}", input.display());
    println!(
        "LAS {}.{}, point format {}",
        header.version_major, header.version_minor, header.point_format
    );
    println!("Points: {}", stats.count);
    println!(
        "Bounds: ({:.2}, {:.2}, {:.2}) - ({:.2}, {:.2}, {:.2})",
        stats.min_x, stats.min_y, stats.min_z, stats.max_x, stats.max_y, stats.max_z
    );
    println!(
        "Centroid: ({:.2}, {:.2}, {:.2})",
        centroid.x, centroid.y, centroid.z
    );
    println!("Z mean: {:.2}, std: {:.2}", stats.mean_z, stats.std_z);

    let mut counts: BTreeMap<u8, usize> = BTreeMap::new();
    for p in cloud.points() {
        *counts.entry(p.classification as u8).or_default() += 1;
    }
    println!("Classifications:");
    for (code, count) in counts {
        println!("  {code:>2} {:?}: {count}", Classification::from_u8(code));
    }

    Ok(())
}

fn ground_classify(input: &Path, output: &Path, cell_size: f64, threshold: f64) -> Result<()> {
    let mut cloud = load_cloud(input)?;
    ground_filter_simple(&mut cloud, cell_size, threshold);

    let ground = cloud
        .points()
        .iter()
        .filter(|p| p.classification == Classification::Ground)
        .count();
    save_cloud(&cloud, output)?;

    println!(
        "Ground: {ground}/{} points ({:.1}%)",
        cloud.len(),
        percent(ground, cloud.len())
    );
    println!("Wrote {}", output.display());
    Ok(())
}

fn thin(
    input: &Path,
    output: &Path,
    method: ThinMethod,
    voxel_size: f64,
    fraction: f64,
) -> Result<()> {
    let cloud = load_cloud(input)?;
    let (thinned, label) = match method {
        ThinMethod::Voxel => (
            thin_voxel(&cloud, voxel_size),
            format!("voxel {voxel_size}"),
        ),
        ThinMethod::Random => (thin_random(&cloud, fraction), format!("random {fraction}")),
    };
    save_cloud(&thinned, output)?;

    println!(
        "Thinned {} -> {} points ({label}, {:.1}% kept)",
        cloud.len(),
        thinned.len(),
        percent(thinned.len(), cloud.len())
    );
    println!("Wrote {}", output.display());
    Ok(())
}

fn outlier_removal(
    input: &Path,
    output: &Path,
    neighbours: usize,
    std_multiplier: f64,
) -> Result<()> {
    let cloud = load_cloud(input)?;
    let cleaned = statistical_outlier_removal(&cloud, neighbours, std_multiplier);
    let removed = cloud.len() - cleaned.len();
    save_cloud(&cleaned, output)?;

    println!(
        "Removed {removed}/{} points, {} kept",
        cloud.len(),
        cleaned.len()
    );
    println!("Wrote {}", output.display());
    Ok(())
}

fn interpolate_to_grid(args: &InterpolateArgs) -> Result<()> {
    let cloud = load_cloud(&args.input)?;

    let interpolated = match args.method {
        GridMethod::Idw => idw_interpolation(
            &cloud,
            args.cell_size,
            args.power,
            args.search_radius,
            args.min_points,
        )
        .ok_or("interpolation failed: empty cloud or non-positive cell size")?,
        GridMethod::Kriging => {
            if args.search_radius <= 0.0 {
                return Err("kriging needs --search-radius greater than 0".into());
            }
            let bins = empirical_variogram(&cloud, args.variogram_bins, args.search_radius);
            if bins.is_empty() {
                return Err("no point pairs within the search radius, raise it".into());
            }
            let model = VariogramModel::fit_spherical(&bins);
            let (name, nugget, sill, range) = model_params(&model);
            println!("Variogram: {name} nugget {nugget:.3} sill {sill:.3} range {range:.2}");
            ordinary_kriging(&cloud, &model, args.cell_size, args.search_radius)
        }
    };

    let mut writer = create(&args.output)?;
    grid::write_ascii_grid(&interpolated, &mut writer).map_err(|e| context(&args.output, e))?;
    writer.flush().map_err(|e| context(&args.output, e))?;

    let cells = interpolated.width * interpolated.height;
    let filled = grid::filled_cells(&interpolated);
    println!(
        "Grid: {}x{} at {}, {filled}/{cells} cells filled ({:.1}%)",
        interpolated.width,
        interpolated.height,
        args.cell_size,
        percent(filled, cells)
    );
    println!("Wrote {}", args.output.display());
    Ok(())
}

fn variogram(input: &Path, bins: usize, max_lag: Option<f64>) -> Result<()> {
    let cloud = load_cloud(input)?;
    let max_lag = match max_lag {
        Some(lag) => lag,
        None => default_max_lag(&cloud).ok_or("file contains no points")?,
    };
    if max_lag <= 0.0 || bins == 0 {
        return Err("--max-lag and --bins must be greater than 0".into());
    }

    let empirical = empirical_variogram(&cloud, bins, max_lag);
    if empirical.is_empty() {
        return Err("no point pairs within max lag, raise --max-lag".into());
    }

    println!("Max lag: {max_lag:.2}, {} non-empty bins", empirical.len());
    println!("{:>10} {:>14} {:>9}", "lag", "semivariance", "pairs");
    for bin in &empirical {
        println!(
            "{:>10.2} {:>14.4} {:>9}",
            bin.lag, bin.semivariance, bin.count
        );
    }

    let model = VariogramModel::fit_spherical(&empirical);
    let (name, nugget, sill, range) = model_params(&model);
    println!("Fitted {name}: nugget {nugget:.3}, sill {sill:.3}, range {range:.2}");
    Ok(())
}

fn demo(output: &Path, count: usize) -> Result<()> {
    let cloud = synthetic_cloud(count);
    save_cloud(&cloud, output)?;
    println!(
        "Wrote {} synthetic points to {}",
        cloud.len(),
        output.display()
    );
    Ok(())
}

/// Sloped terrain on a 2 unit grid with every tenth point lifted into "vegetation".
fn synthetic_cloud(n: usize) -> PointCloud {
    let side = (n as f64).sqrt().ceil().max(1.0) as usize;
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i % side) as f64 * 2.0;
        let y = (i / side) as f64 * 2.0;
        let ground = 10.0 + x * 0.05 + (y * 0.1).sin();
        let z = if i % 10 == 0 { ground + 15.0 } else { ground };
        points.push(Point3::new(x, y, z).with_intensity((i % 1000) as u16));
    }
    PointCloud::from_points(points)
}

fn model_params(model: &VariogramModel) -> (&'static str, f64, f64, f64) {
    match model {
        VariogramModel::Spherical {
            nugget,
            sill,
            range,
        } => ("spherical", *nugget, *sill, *range),
        VariogramModel::Exponential {
            nugget,
            sill,
            range,
        } => ("exponential", *nugget, *sill, *range),
        VariogramModel::Gaussian {
            nugget,
            sill,
            range,
        } => ("gaussian", *nugget, *sill, *range),
    }
}

fn default_max_lag(cloud: &PointCloud) -> Option<f64> {
    let (min, max) = cloud.bounds()?;
    let dx = max.x - min.x;
    let dy = max.y - min.y;
    Some((dx * dx + dy * dy).sqrt() / 2.0)
}

fn percent(part: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    part as f64 / total as f64 * 100.0
}

fn open(path: &Path) -> Result<BufReader<File>> {
    let file = File::open(path).map_err(|e| context(path, e))?;
    Ok(BufReader::new(file))
}

fn create(path: &Path) -> Result<BufWriter<File>> {
    let file = File::create(path).map_err(|e| context(path, e))?;
    Ok(BufWriter::new(file))
}

fn load_cloud(path: &Path) -> Result<PointCloud> {
    let mut reader = open(path)?;
    read_las(&mut reader).map_err(|e| context(path, e).into())
}

fn save_cloud(cloud: &PointCloud, path: &Path) -> Result<()> {
    // checked before create so a failed run cannot leave an empty file behind
    if cloud.is_empty() {
        return Err(context(path, "nothing left to write, the result is empty").into());
    }
    let mut writer = create(path)?;
    write_las(cloud, &mut writer).map_err(|e| context(path, e))?;
    writer.flush().map_err(|e| context(path, e))?;
    Ok(())
}

fn context(path: &Path, err: impl std::fmt::Display) -> String {
    format!("{}: {err}", path.display())
}
