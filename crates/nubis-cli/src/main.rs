use clap::{Parser, Subcommand};
use nubis_core::{Point3, PointCloud, ground_filter_simple, thin_voxel};

#[derive(Parser)]
#[command(name = "nubis", version, about = "Point cloud processing CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show info about a synthetic point cloud
    Info {
        /// Number of points to generate
        #[arg(long, default_value_t = 1000)]
        count: usize,
    },
    /// Classify ground in a synthetic point cloud
    Ground {
        /// Grid cell size
        #[arg(long, default_value_t = 1.0)]
        cell_size: f64,
        /// Height threshold
        #[arg(long, default_value_t = 0.5)]
        threshold: f64,
    },
}

fn synthetic_cloud(n: usize) -> PointCloud {
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / n as f64;
        let x = t * 100.0;
        let y = (t * std::f64::consts::PI).sin() * 50.0;
        let z = if i % 10 == 0 { 20.0 } else { t * 2.0 }; // every 10th is "vegetation"
        points.push(Point3::new(x, y, z));
    }
    PointCloud::from_points(points)
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Info { count } => {
            let cloud = synthetic_cloud(count);
            let (min, max) = cloud.bounds().unwrap();
            let centroid = cloud.centroid().unwrap();
            println!("Points: {}", cloud.len());
            println!(
                "Bounds: ({:.2}, {:.2}, {:.2}) - ({:.2}, {:.2}, {:.2})",
                min.x, min.y, min.z, max.x, max.y, max.z
            );
            println!(
                "Centroid: ({:.2}, {:.2}, {:.2})",
                centroid.x, centroid.y, centroid.z
            );

            let thinned = thin_voxel(&cloud, 5.0);
            println!("After voxel thinning (5m): {} points", thinned.len());
        }
        Commands::Ground {
            cell_size,
            threshold,
        } => {
            let mut cloud = synthetic_cloud(100);
            ground_filter_simple(&mut cloud, cell_size, threshold);
            let ground_count = cloud
                .points()
                .iter()
                .filter(|p| p.classification == nubis_core::Classification::Ground)
                .count();
            println!("Ground points: {}/{}", ground_count, cloud.len());
        }
    }
}
