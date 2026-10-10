//! Port of InsightFace's `face_align.norm_crop`.

use image::{Rgb, RgbImage};

pub const ALIGNED_SIZE: u32 = 112;

/// `arcface_dst` from InsightFace's `utils/face_align.py`.
pub const ARCFACE_DST: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

/// A 2x3 affine matrix mapping source to destination: `[x', y'] = [[a, b, tx], [c, d, ty]] * [x, y, 1]`.
pub type Affine = [[f64; 3]; 2];

#[must_use]
pub fn align(image: &RgbImage, landmarks: &[[f32; 2]; 5]) -> Option<RgbImage> {
    let m = similarity(landmarks, &ARCFACE_DST)?;
    Some(warp_affine(image, &m, ALIGNED_SIZE, ALIGNED_SIZE))
}

/// Umeyama, as skimage's `SimilarityTransform.estimate`; reflected sets get the best proper rotation.
#[must_use]
pub fn similarity(src: &[[f32; 2]], dst: &[[f32; 2]]) -> Option<Affine> {
    if src.len() != dst.len() || src.is_empty() {
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    let n = src.len() as f64;
    let mean = |points: &[[f32; 2]]| {
        let (x, y) = points.iter().fold((0.0, 0.0), |(x, y), p| {
            (x + f64::from(p[0]), y + f64::from(p[1]))
        });
        (x / n, y / n)
    };
    let (sx, sy) = mean(src);
    let (dx, dy) = mean(dst);
    // in 2D the optimal scaled rotation is the complex least-squares ratio of the centred sets
    let (mut dot, mut cross, mut var) = (0.0, 0.0, 0.0);
    for (s, d) in src.iter().zip(dst) {
        let (px, py) = (f64::from(s[0]) - sx, f64::from(s[1]) - sy);
        let (qx, qy) = (f64::from(d[0]) - dx, f64::from(d[1]) - dy);
        dot += px * qx + py * qy;
        cross += px * qy - py * qx;
        var += px * px + py * py;
    }
    if var <= f64::EPSILON {
        return None;
    }
    let (a, b) = (dot / var, cross / var);
    if a == 0.0 && b == 0.0 {
        return None;
    }
    Some([
        [a, -b, dx - (a * sx - b * sy)],
        [b, a, dy - (b * sx + a * sy)],
    ])
}

/// Applies `m` like `cv2.warpAffine` with bilinear interpolation and a constant black border.
#[must_use]
pub fn warp_affine(image: &RgbImage, m: &Affine, width: u32, height: u32) -> RgbImage {
    let [[a, b, tx], [c, d, ty]] = *m;
    let det = a * d - b * c;
    let (ia, ib, ic, id) = (d / det, -b / det, -c / det, a / det);
    let (itx, ity) = (-(ia * tx + ib * ty), -(ic * tx + id * ty));
    RgbImage::from_fn(width, height, |x, y| {
        let (x, y) = (f64::from(x), f64::from(y));
        sample_or_black(image, ia * x + ib * y + itx, ic * x + id * y + ity)
    })
}

fn sample_or_black(image: &RgbImage, x: f64, y: f64) -> Rgb<u8> {
    let (w, h) = (i64::from(image.width()), i64::from(image.height()));
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    #[allow(clippy::cast_possible_truncation)]
    let (x0, y0) = (x0 as i64, y0 as i64);
    let mut acc = [0.0_f64; 3];
    for (ox, oy, weight) in [
        (0, 0, (1.0 - fx) * (1.0 - fy)),
        (1, 0, fx * (1.0 - fy)),
        (0, 1, (1.0 - fx) * fy),
        (1, 1, fx * fy),
    ] {
        let (px, py) = (x0 + ox, y0 + oy);
        if weight == 0.0 || px < 0 || py < 0 || px >= w || py >= h {
            continue;
        }
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let pixel = image.get_pixel(px as u32, py as u32);
        for (sum, channel) in acc.iter_mut().zip(pixel.0) {
            *sum += weight * f64::from(channel);
        }
    }
    Rgb(acc.map(to_u8))
}

#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
pub(crate) fn to_u8(value: f64) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(m: &Affine, p: [f32; 2]) -> [f32; 2] {
        let (x, y) = (f64::from(p[0]), f64::from(p[1]));
        #[allow(clippy::cast_possible_truncation)]
        [
            (m[0][0] * x + m[0][1] * y + m[0][2]) as f32,
            (m[1][0] * x + m[1][1] * y + m[1][2]) as f32,
        ]
    }

    fn rotation(scale: f64, angle: f64, tx: f64, ty: f64) -> Affine {
        let (sin, cos) = angle.sin_cos();
        [
            [scale * cos, -scale * sin, tx],
            [scale * sin, scale * cos, ty],
        ]
    }

    fn assert_close(m: &Affine, expected: &Affine, tolerance: f64) {
        for (row, want) in m.iter().zip(expected) {
            for (got, want) in row.iter().zip(want) {
                assert!((got - want).abs() < tolerance, "{m:?} != {expected:?}");
            }
        }
    }

    #[test]
    fn known_similarity_is_recovered() {
        let truth = rotation(0.37, 0.4, 12.5, -30.25);
        let src = [
            [100.0, 220.0],
            [260.0, 215.0],
            [180.0, 300.0],
            [120.0, 380.0],
            [245.0, 376.0],
        ];
        let dst = src.map(|p| apply(&truth, p));
        let m = similarity(&src, &dst).unwrap();
        assert_close(&m, &truth, 1e-4);
    }

    #[test]
    fn noisy_points_give_least_squares_fit() {
        // centred source, so the translation is the mean of dst and the scale/rotation is closed form
        let src = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]];
        let dst = [[-2.0, 0.1], [2.0, -0.1], [0.0, -2.0], [0.0, 2.0]];
        let m = similarity(&src, &dst).unwrap();
        // dot = 2 + 2 + 2 + 2 = 8, cross = (-1)(0.1) + (1)(-0.1) = -0.2, var = 4
        assert_close(&m, &[[2.0, 0.05, 0.0], [-0.05, 2.0, 0.0]], 1e-6);
    }

    #[test]
    fn matches_skimage_estimate_norm() {
        // skimage 0.26 SimilarityTransform.estimate(lmk, arcface_dst).params[0:2]
        let lmk = [
            [251.0, 280.0],
            [349.0, 274.0],
            [303.0, 335.0],
            [262.0, 384.0],
            [342.0, 380.0],
        ];
        let m = similarity(&lmk, &ARCFACE_DST).unwrap();
        let expected = [
            [
                0.375_507_095_172_4,
                -0.010_759_763_056_01,
                -53.594_497_267_22,
            ],
            [
                0.010_759_763_056_01,
                0.375_507_095_172_4,
                -55.484_860_751_20,
            ],
        ];
        assert_close(&m, &expected, 1e-4);
    }

    #[test]
    fn mirrored_points_match_skimage() {
        // skimage 0.26 for a reflected point set: the best proper rotation, not the reflection
        let src = [[0.0, 0.0], [4.0, 1.0], [1.0, 3.0], [5.0, 5.0], [2.0, -1.0]];
        let dst = src.map(|[x, y]| [10.0 - x, y + 3.0]);
        let m = similarity(&src, &dst).unwrap();
        let expected = [
            [0.148_514_851_485, -0.534_653_465_347, 8.099_009_900_99],
            [0.534_653_465_347, 0.148_514_851_485, 3.079_207_920_792],
        ];
        assert_close(&m, &expected, 1e-6);
    }

    #[test]
    fn degenerate_points_are_rejected() {
        assert!(similarity(&[[5.0, 5.0]; 5], &ARCFACE_DST).is_none());
        assert!(similarity(&[[0.0, 0.0]], &[]).is_none());
        // skimage yields scale 0 here: the exact mirror image of a symmetric set
        let src = [[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]];
        let dst = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]];
        assert!(similarity(&src, &dst).is_none());
    }

    #[test]
    fn warp_identity_copies_and_borders_black() {
        let image = RgbImage::from_fn(4, 3, |x, y| {
            #[allow(clippy::cast_possible_truncation)]
            Rgb([(x * 10) as u8, (y * 10) as u8, 200])
        });
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let out = warp_affine(&image, &identity, 6, 3);
        for y in 0..3 {
            for x in 0..4 {
                assert_eq!(out.get_pixel(x, y), image.get_pixel(x, y));
            }
            assert_eq!(out.get_pixel(5, y).0, [0, 0, 0]);
        }
    }

    #[test]
    fn warp_shift_interpolates_and_fades_into_border() {
        let image = RgbImage::from_pixel(2, 1, Rgb([100, 100, 100]));
        // destination x = source x + 0.5
        let shift = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]];
        let out = warp_affine(&image, &shift, 3, 1);
        assert_eq!(out.get_pixel(0, 0).0, [50; 3]);
        assert_eq!(out.get_pixel(1, 0).0, [100; 3]);
        assert_eq!(out.get_pixel(2, 0).0, [50; 3]);
    }

    #[test]
    fn aligned_template_points_land_on_template() {
        let truth = rotation(3.1, -0.3, 140.0, 75.0);
        let landmarks = ARCFACE_DST.map(|p| apply(&truth, p));
        let mut image = RgbImage::new(640, 480);
        for p in &landmarks {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let (cx, cy) = (p[0].round() as i64, p[1].round() as i64);
            for y in cy - 4..=cy + 4 {
                for x in cx - 4..=cx + 4 {
                    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                    image.put_pixel(x as u32, y as u32, Rgb([255; 3]));
                }
            }
        }
        let aligned = align(&image, &landmarks).unwrap();
        assert_eq!(aligned.dimensions(), (ALIGNED_SIZE, ALIGNED_SIZE));
        for q in ARCFACE_DST {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let pixel = aligned.get_pixel(q[0].round() as u32, q[1].round() as u32);
            assert!(pixel.0[0] > 200, "{q:?}: {pixel:?}");
        }
        assert_eq!(aligned.get_pixel(56, 20).0, [0; 3]);
    }
}
