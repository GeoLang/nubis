/// LAS point classification codes (ASPRS standard).
///
/// Point formats 0-5 store the class in 5 bits, so codes run 0-31. Codes with no
/// name here are carried in `Other` so they survive a read/write round trip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Classification {
    Unclassified,
    Unknown,
    Ground,
    LowVegetation,
    MediumVegetation,
    HighVegetation,
    Building,
    LowPoint,
    Water,
    Rail,
    Road,
    BridgeDeck,
    HighNoise,
    /// A code this enum has no name for. Build it with [`Classification::from_u8`],
    /// which sends the named codes to their own variants.
    Other(u8),
}

impl Classification {
    pub fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Unclassified,
            1 => Self::Unknown,
            2 => Self::Ground,
            3 => Self::LowVegetation,
            4 => Self::MediumVegetation,
            5 => Self::HighVegetation,
            6 => Self::Building,
            7 => Self::LowPoint,
            9 => Self::Water,
            10 => Self::Rail,
            11 => Self::Road,
            17 => Self::BridgeDeck,
            18 => Self::HighNoise,
            other => Self::Other(other),
        }
    }

    /// The ASPRS code for this class.
    pub fn to_u8(self) -> u8 {
        match self {
            Self::Unclassified => 0,
            Self::Unknown => 1,
            Self::Ground => 2,
            Self::LowVegetation => 3,
            Self::MediumVegetation => 4,
            Self::HighVegetation => 5,
            Self::Building => 6,
            Self::LowPoint => 7,
            Self::Water => 9,
            Self::Rail => 10,
            Self::Road => 11,
            Self::BridgeDeck => 17,
            Self::HighNoise => 18,
            Self::Other(code) => code,
        }
    }
}
