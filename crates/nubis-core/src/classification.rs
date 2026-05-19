/// LAS point classification codes (ASPRS standard).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Classification {
    Unclassified = 0,
    Unknown = 1,
    Ground = 2,
    LowVegetation = 3,
    MediumVegetation = 4,
    HighVegetation = 5,
    Building = 6,
    LowPoint = 7,
    Water = 9,
    Rail = 10,
    Road = 11,
    BridgeDeck = 17,
    HighNoise = 18,
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
            _ => Self::Unknown,
        }
    }
}
