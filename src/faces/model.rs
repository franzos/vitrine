use image::RgbImage;

use crate::config::Faces;

/// A detected face in source pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    /// `x1, y1, x2, y2`; may extend past the image edges.
    pub bbox: [f32; 4],
    /// Eyes, nose tip, mouth corners; in each pair the point on the image's left comes first.
    pub landmarks: [[f32; 2]; 5],
    pub score: f32,
}

impl Detection {
    #[must_use]
    pub fn width(&self) -> f32 {
        self.bbox[2] - self.bbox[0]
    }

    #[must_use]
    pub fn height(&self) -> f32 {
        self.bbox[3] - self.bbox[1]
    }

    #[must_use]
    pub fn counts(&self, faces: &Faces) -> bool {
        #[allow(clippy::cast_precision_loss)]
        let min_side = faces.min_face_size as f32;
        self.score >= faces.min_score && self.width().min(self.height()) >= min_side
    }
}

pub trait FaceDetector {
    /// Sorted by descending score.
    fn detect(&self, image: &RgbImage) -> anyhow::Result<Vec<Detection>>;
}

pub trait FaceEmbedder {
    /// Crops from [`crate::faces::align::align`]; one vector per crop, in order.
    fn embed(&self, aligned: &[RgbImage]) -> anyhow::Result<Vec<Vec<f32>>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detection(bbox: [f32; 4], score: f32) -> Detection {
        Detection {
            bbox,
            landmarks: [[0.0; 2]; 5],
            score,
        }
    }

    #[test]
    fn count_filter_checks_score_and_shorter_side() {
        let faces = Faces {
            min_score: 0.6,
            min_face_size: 40,
            ..Faces::default()
        };
        assert!(detection([10.0, 10.0, 50.0, 60.0], 0.6).counts(&faces));
        assert!(!detection([10.0, 10.0, 50.0, 60.0], 0.59).counts(&faces));
        assert!(!detection([10.0, 10.0, 49.0, 60.0], 0.9).counts(&faces));
        assert!(!detection([10.0, 10.0, 100.0, 49.0], 0.9).counts(&faces));
    }
}
