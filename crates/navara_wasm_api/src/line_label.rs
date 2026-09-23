//! Pure numeric kernel for placing text labels along a line.
//!
//! Sibling of [`crate::declutter`], and split from TypeScript for the same
//! reason: everything here is a CPU mirror of what `sdfText.vert.glsl` does
//! with the same data, and that mirror already exists on this side of the
//! boundary. Doing it in TypeScript would mean a third copy of `nvr_pxToWorld`.
//!
//! Three decisions per label, all of which depend on the camera and so cannot
//! be baked when the tile is parsed:
//!
//! - **flip** — whether to walk the label's path backwards, so a name never
//!   reads right-to-left on screen (`keepUpright`).
//! - **fit** — whether the label is short enough to sit on the line at all.
//!   With `sizeInMeters: false` a label's world length grows as the camera
//!   pulls back, so a name that fits when zoomed in can overrun its road when
//!   zoomed out.
//! - **angle** — whether the line bends too sharply under the label to stay
//!   readable (`maxAngle`).
//!
//! ## Label layout
//!
//! Labels arrive as a flat `f64` slice, [`LINE_LABEL_STRIDE`] values each, with
//! their path samples in a parallel `f32` slice of `2 * samples` values each:
//!
//! | offset | field | meaning |
//! |--------|-------|---------|
//! | 0,1,2  | anchorX/Y/Z | ECEF anchor in meters, before the height offset |
//! | 3      | addHeight | surface-normal height offset (meters) |
//! | 4      | widthEm | the text block's width, in ems |
//! | 5      | fontSize | px or meters, per `sizeInMeters` |
//! | 6      | sizeInMeters | `0.0` = px, non-zero = meters |
//! | 7      | maxAngleRad | largest cumulative turn allowed under the label |
//! | 8      | keepUpright | `0.0` = never flip, non-zero = flip when backwards |
//! | 9      | stepMeters | arc length between adjacent path samples |
//! | 10     | halfExtentMeters | real line either side of the anchor |
//! | 11     | bearingRad | the line's tangent at the anchor, clockwise from north |
//! | 12     | isFlipped | the label's current flip, for hysteresis |
//! | 13,14  | minX/maxX | the label's unrotated box along its baseline |
//! | 15,16  | minY/maxY | the same across it, +Y up, in the font's own units |
//!
//! ## Result layout
//!
//! | offset | field |
//! |--------|-------|
//! | 0      | flip — walk the path backwards |
//! | 1      | rejected — does not fit, or bends too far |
//! | 2,3,4,5| minX/maxX/minY/maxY of the *rotated* box, +Y up |

use wasm_bindgen::prelude::*;

/// Number of `f64` values per label in the packed input slice.
pub const LINE_LABEL_STRIDE: usize = 17;

/// Number of `f64` values per label in the packed output slice.
pub const LINE_LABEL_RESULT_STRIDE: usize = 6;

/// How far the sliding angle window reaches, as a multiple of the font size.
///
/// MapLibre's `checkMaxAngle` uses `3/5 * glyphSize`; a window of roughly one
/// em is the same idea — the turn that matters is the one a reader sees across
/// a couple of adjacent glyphs, not the total bend of a long gentle curve.
const ANGLE_WINDOW_EMS: f64 = 1.5;

/// How near vertical a label has to run before its horizontal direction stops
/// being a meaningful test of whether it reads forwards.
const VERTICAL_BAND: f64 = 0.15;

/// Hysteresis on the flip decision. A label sitting exactly on the boundary
/// would otherwise flip back and forth frame to frame as the camera drifts —
/// the same failure `HYSTERESIS_PX` guards against in the declutter pass.
const FLIP_HYSTERESIS: f64 = 0.03;

/// Place text labels along their lines.
///
/// `labels` is a packed `f64` slice of `n * LINE_LABEL_STRIDE` values (see the
/// module docs); `paths` holds each label's samples as `2 * samples` `f32`
/// values, east then north metres relative to its anchor. `view` and `proj` are
/// column-major 4x4 matrices.
///
/// Returns `n * LINE_LABEL_RESULT_STRIDE` values per label, in input order; see
/// the module docs for the layout.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = lineLabelPlace)]
pub fn line_label_place(
    labels: &[f64],
    paths: &[f32],
    samples_per_label: usize,
    view: &[f64],
    height_px: f64,
    fov_rad: f64,
) -> Vec<f64> {
    let n = labels.len() / LINE_LABEL_STRIDE;
    let mut out = vec![0.0; n * LINE_LABEL_RESULT_STRIDE];
    if samples_per_label < 2 {
        return out;
    }

    for i in 0..n {
        let l = &labels[i * LINE_LABEL_STRIDE..(i + 1) * LINE_LABEL_STRIDE];
        let path = &paths[i * samples_per_label * 2..(i + 1) * samples_per_label * 2];

        let meters_per_em = em_to_meters(l, view, height_px, fov_rad);
        let half_len = l[4] * meters_per_em * 0.5;

        // Which way the label runs on screen. Taken across the label's whole
        // extent rather than from the tangent at its anchor: on a curving road
        // the two disagree, and it is the overall reading direction that
        // decides whether a name comes out backwards.
        let (mut sx, mut sy) = screen_direction(l, path, samples_per_label, half_len, view);

        let flip = l[8] != 0.0 && should_flip(sx, sy, l[12] != 0.0);
        if flip {
            sx = -sx;
            sy = -sy;
        }

        let rejected = half_len <= 0.0
            || half_len > l[10]
            || exceeds_max_angle(path, samples_per_label, l[9], half_len, meters_per_em, l[7]);

        let o = i * LINE_LABEL_RESULT_STRIDE;
        out[o] = if flip { 1.0 } else { 0.0 };
        out[o + 1] = if rejected { 1.0 } else { 0.0 };
        let (bx0, bx1, by0, by1) = rotated_box(l[13], l[14], l[15], l[16], sx, sy);
        out[o + 2] = bx0;
        out[o + 3] = bx1;
        out[o + 4] = by0;
        out[o + 5] = by1;
    }

    out
}

/// Metres one em of the label spans on the ground.
///
/// Mirrors `sdfText.vert.glsl`'s `scaleFactor`: metres directly when the font
/// size is metric, otherwise `nvr_pxToWorld` at the anchor's view depth —
/// including its `|viewZ|` approximation of distance, so the CPU and the shader
/// cannot disagree about how long a label is.
fn em_to_meters(l: &[f64], view: &[f64], height_px: f64, fov_rad: f64) -> f64 {
    let font_size = l[5];
    if l[6] != 0.0 {
        return font_size;
    }
    let vz = view[2] * l[0] + view[6] * l[1] + view[10] * l[2] + view[14];
    if vz >= 0.0 {
        // Behind the camera: nothing sensible to scale by, and the declutter
        // pass will drop the label anyway.
        return 0.0;
    }
    font_size * (2.0 * (fov_rad / 2.0).tan() * -vz) / height_px
}

/// The anchor's east and north directions, projected into view space and kept
/// as 2D screen axes.
///
/// The derivation mirrors `nvr_enuBasis` in
/// `shaders/glsl/chunks/quad_orientation.glsl`, which is the basis the vertex
/// shader lays the label out in. Only the x/y components survive: view space
/// shares the screen's axes, and the label's orientation is a 2D question.
fn enu_screen_axes(l: &[f64], view: &[f64]) -> ((f64, f64), (f64, f64)) {
    let (x, y, z) = (l[0], l[1], l[2]);
    let len = (x * x + y * y + z * z).sqrt();
    if len <= 0.0 {
        return ((1.0, 0.0), (0.0, 1.0));
    }
    let (nx, ny, nz) = (x / len, y / len, z / len);

    let (ex, ey) = (-ny, nx);
    let e_len = (ex * ex + ey * ey).sqrt();
    let east = if e_len > 1e-6 {
        (ex / e_len, ey / e_len, 0.0)
    } else {
        // At the poles every tangent direction is an equally valid "east".
        (1.0, 0.0, 0.0)
    };
    let north = (
        ny * east.2 - nz * east.1,
        nz * east.0 - nx * east.2,
        nx * east.1 - ny * east.0,
    );

    // Rotate both into view space (w = 0), keeping the screen plane only.
    let rot = |v: (f64, f64, f64)| {
        (
            view[0] * v.0 + view[4] * v.1 + view[8] * v.2,
            view[1] * v.0 + view[5] * v.1 + view[9] * v.2,
        )
    };
    (rot(east), rot(north))
}

/// Unit direction the label reads in, on screen.
///
/// Measured as the chord across the label's own extent, so a road that curves
/// under the label is judged by where the text actually starts and ends rather
/// than by the tangent at its midpoint. Falls back to the anchor's bearing when
/// the label has no length yet — a label whose text has not been shaped.
fn screen_direction(
    l: &[f64],
    path: &[f32],
    samples: usize,
    half_len_meters: f64,
    view: &[f64],
) -> (f64, f64) {
    let (east, north) = enu_screen_axes(l, view);

    let step = l[9];
    let mid = samples / 2;
    let reach = if step > 0.0 {
        ((half_len_meters / step).round() as usize).clamp(1, mid.max(1))
    } else {
        1
    };
    let first = mid.saturating_sub(reach);
    let last = (mid + reach).min(samples - 1);

    let (mut de, mut dn) = (
        (path[last * 2] - path[first * 2]) as f64,
        (path[last * 2 + 1] - path[first * 2 + 1]) as f64,
    );
    if de.hypot(dn) <= 1e-6 {
        // A degenerate chord (a hairpin doubling back onto itself, or a label
        // with no extent). The bearing is clockwise from north, so it
        // decomposes as north*cos + east*sin.
        let (sin_b, cos_b) = l[11].sin_cos();
        de = sin_b;
        dn = cos_b;
    }

    let sx = de * east.0 + dn * north.0;
    let sy = de * east.1 + dn * north.1;
    let len = sx.hypot(sy);
    if len <= 1e-12 {
        // The label is edge-on to the camera; any direction reads the same.
        return (1.0, 0.0);
    }
    (sx / len, sy / len)
}

/// Whether a label running in screen direction `(sx, sy)` should be walked
/// backwards.
///
/// Normally this is just "does it read left-to-right", but a label running
/// nearly straight up or down the screen has no meaningful left-to-right to
/// test — `sx` there is noise. Those fall back to reading **bottom-to-top**,
/// the cartographic convention for a vertical street name, and the difference
/// between a name you tilt your head left to read and one you tilt it right
/// for.
///
/// The decision is asymmetric about zero so a label sitting on the boundary
/// keeps whichever way it already faces rather than flipping back and forth as
/// the camera drifts — the same failure `HYSTERESIS_PX` guards against in the
/// declutter pass.
fn should_flip(sx: f64, sy: f64, currently_flipped: bool) -> bool {
    // The hysteresis belongs on the *band*, not on the score. Inside either
    // branch the score is far from zero — a label steep enough to be judged
    // horizontally has |sx| > VERTICAL_BAND, and one judged vertically has
    // |sy| close to 1 — so the only place the answer can chatter is where the
    // two rules hand over to each other.
    let band = if currently_flipped {
        VERTICAL_BAND - FLIP_HYSTERESIS
    } else {
        VERTICAL_BAND + FLIP_HYSTERESIS
    };
    let score = if sx.abs() > band { sx } else { sy };
    score < 0.0
}

/// Screen-space AABB of the label's box once turned to run along `(dx, dy)`.
///
/// The declutter grid is axis-aligned, so a label that follows a north-south
/// road has to be handed the box it actually covers: modelling it as the
/// horizontal rectangle it would occupy unrotated over-claims across the road
/// and, worse, under-claims along it — which is exactly where labels stack up
/// and overlap.
fn rotated_box(
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    dx: f64,
    dy: f64,
) -> (f64, f64, f64, f64) {
    // The label's own +x maps to (dx, dy) and its +y to the perpendicular.
    let corner = |lx: f64, ly: f64| (lx * dx - ly * dy, lx * dy + ly * dx);
    let corners = [
        corner(min_x, min_y),
        corner(max_x, min_y),
        corner(max_x, max_y),
        corner(min_x, max_y),
    ];
    corners.iter().fold(
        (f64::MAX, f64::MIN, f64::MAX, f64::MIN),
        |(x0, x1, y0, y1), &(x, y)| (x0.min(x), x1.max(x), y0.min(y), y1.max(y)),
    )
}

/// Whether the path bends too sharply anywhere under the label.
///
/// The turn is accumulated over a sliding window rather than over the whole
/// label: a long road that curves gently is perfectly readable even though its
/// total bend is large, while a short sharp kink is not. Samples are uniform in
/// arc length, so the window is a fixed number of samples and the sliding sum
/// comes straight off a prefix sum.
fn exceeds_max_angle(
    path: &[f32],
    samples: usize,
    step_meters: f64,
    half_len_meters: f64,
    meters_per_em: f64,
    max_angle_rad: f64,
) -> bool {
    if step_meters <= 0.0 || max_angle_rad <= 0.0 {
        return false;
    }

    // Samples the label actually covers, centred on the anchor.
    let half_span_samples = (half_len_meters / step_meters).ceil() as usize;
    let mid = samples / 2;
    let first = mid.saturating_sub(half_span_samples);
    let last = (mid + half_span_samples).min(samples - 1);
    if last - first < 2 {
        return false;
    }

    // Turn at each interior sample, as a prefix sum over the covered range.
    let mut prefix = vec![0.0f64; last - first];
    for k in (first + 1)..last {
        let turn = corner_angle(path, k);
        prefix[k - first] = prefix[k - first - 1] + turn;
    }

    let window_samples = ((ANGLE_WINDOW_EMS * meters_per_em) / step_meters)
        .ceil()
        .max(1.0) as usize;
    if window_samples >= prefix.len() {
        return *prefix.last().unwrap_or(&0.0) > max_angle_rad;
    }
    for start in 0..(prefix.len() - window_samples) {
        if prefix[start + window_samples] - prefix[start] > max_angle_rad {
            return true;
        }
    }
    false
}

/// Absolute turn, in radians, between the segments meeting at sample `k`.
fn corner_angle(path: &[f32], k: usize) -> f64 {
    let p = |i: usize| (path[i * 2] as f64, path[i * 2 + 1] as f64);
    let (ax, ay) = p(k - 1);
    let (bx, by) = p(k);
    let (cx, cy) = p(k + 1);
    let (ux, uy) = (bx - ax, by - ay);
    let (vx, vy) = (cx - bx, cy - by);
    let (ul, vl) = (ux.hypot(uy), vx.hypot(vy));
    if ul <= 0.0 || vl <= 0.0 {
        return 0.0;
    }
    // atan2 of the cross and dot products is stable where acos of the
    // normalized dot loses precision near zero turn — which is most corners.
    let cross = ux * vy - uy * vx;
    let dot = ux * vx + uy * vy;
    cross.atan2(dot).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A camera hovering over the equator at longitude 0, looking straight down
    /// with screen right = east and screen up = north.
    ///
    /// Column-major, so `v[col * 4 + row]`: row 0 is `(v[0], v[4], v[8])`, the
    /// row `decide_flip` dots the tangent with. Putting east there is what makes
    /// the flip test reduce to the sign of the tangent's easting.
    fn top_down_view(distance_m: f64) -> Vec<f64> {
        vec![
            // col 0        col 1        col 2        col 3
            0.0,
            0.0,
            1.0,
            0.0, //
            1.0,
            0.0,
            0.0,
            0.0, //
            0.0,
            1.0,
            0.0,
            0.0, //
            0.0,
            0.0,
            -WGS84_EQ - distance_m,
            1.0,
        ]
    }

    /// The camera close enough that a pixel-sized label stays short.
    fn view() -> Vec<f64> {
        top_down_view(500.0)
    }

    fn label(bearing_rad: f64, keep_upright: bool, is_flipped: bool) -> [f64; LINE_LABEL_STRIDE] {
        [
            // Anchor on the equator at longitude 0, so east is ECEF +y and
            // north is ECEF +z.
            WGS84_EQ,
            0.0,
            0.0,  //
            0.0,  // addHeight
            4.0,  // widthEm
            10.0, // fontSize
            1.0,  // sizeInMeters
            std::f64::consts::FRAC_PI_4,
            if keep_upright { 1.0 } else { 0.0 },
            1.0,    // stepMeters
            1000.0, // halfExtentMeters
            bearing_rad,
            if is_flipped { 1.0 } else { 0.0 },
            // A 4-em label, one em tall, centred on its anchor.
            -2.0 * 10.0,
            2.0 * 10.0,
            0.0,
            10.0,
        ]
    }

    const WGS84_EQ: f64 = 6378137.0;

    /// A dead-straight path through the anchor, running east.
    fn straight_path(samples: usize, step: f64) -> Vec<f32> {
        directed_path(samples, step, (1.0, 0.0))
    }

    /// A dead-straight path through the anchor along the given east/north
    /// direction. The label's reading direction is taken from the path, so a
    /// fixture's bearing and its samples have to agree.
    fn directed_path(samples: usize, step: f64, dir: (f64, f64)) -> Vec<f32> {
        let mid = samples / 2;
        (0..samples)
            .flat_map(|k| {
                let t = (k as f64 - mid as f64) * step;
                [(t * dir.0) as f32, (t * dir.1) as f32]
            })
            .collect()
    }

    #[test]
    fn straight_lines_are_accepted_and_not_flipped() {
        // Bearing 90 deg = due east, which reads left-to-right on this camera.
        let l = label(std::f64::consts::FRAC_PI_2, true, false);
        let path = straight_path(32, 1.0);
        let out = line_label_place(&l, &path, 32, &view(), 1000.0, 1.0);
        assert_eq!(out[0], 0.0, "flip");
        assert_eq!(out[1], 0.0, "rejected");
    }

    #[test]
    fn westbound_lines_flip_when_keep_upright_is_on() {
        // Bearing 270 deg = due west: the text would read right-to-left.
        let west = 3.0 * std::f64::consts::FRAC_PI_2;
        let path = directed_path(32, 1.0, (-1.0, 0.0));

        let on = line_label_place(&label(west, true, false), &path, 32, &view(), 1000.0, 1.0);
        assert_eq!(on[0], 1.0, "should flip");

        let off = line_label_place(&label(west, false, false), &path, 32, &view(), 1000.0, 1.0);
        assert_eq!(off[0], 0.0, "keepUpright off must never flip");
    }

    #[test]
    fn vertical_labels_are_turned_to_read_bottom_to_top() {
        // A street running up the screen already reads bottom-to-top; one
        // running down the screen has to be turned, or the name comes out
        // needing the reader to tilt their head the wrong way. Neither has a
        // meaningful left-to-right to test, which is why the vertical
        // component decides.
        let up = line_label_place(
            &label(0.0, true, false),
            &directed_path(32, 1.0, (0.0, 1.0)),
            32,
            &view(),
            1000.0,
            1.0,
        );
        assert_eq!(up[0], 0.0, "already reads bottom-to-top");

        let down = line_label_place(
            &label(0.0, true, false),
            &directed_path(32, 1.0, (0.0, -1.0)),
            32,
            &view(),
            1000.0,
            1.0,
        );
        assert_eq!(down[0], 1.0, "should be turned to read bottom-to-top");
    }

    #[test]
    fn flip_is_sticky_where_the_two_rules_hand_over() {
        // Just steep enough that the horizontal rule would flip it, but the
        // vertical rule would not. Whichever way the label already faces has to
        // win, or a label sitting on this boundary flips every time the camera
        // drifts a pixel.
        let dir = (-0.15, 0.99f64);
        let len = (dir.0 * dir.0 + dir.1 * dir.1).sqrt();
        let path = directed_path(32, 1.0, (dir.0 / len, dir.1 / len));

        let was_upright =
            line_label_place(&label(0.0, true, false), &path, 32, &view(), 1000.0, 1.0);
        let was_flipped =
            line_label_place(&label(0.0, true, true), &path, 32, &view(), 1000.0, 1.0);

        assert_eq!(was_upright[0], 0.0);
        assert_eq!(was_flipped[0], 1.0);
    }

    #[test]
    fn the_reading_direction_comes_from_the_label_not_its_anchor() {
        // A road that runs east at the anchor but doubles back westward over
        // the label's own extent. The anchor's tangent says "reads fine"; the
        // chord across the label says otherwise, and the chord is what a reader
        // sees.
        let samples = 32;
        let mid = samples / 2;
        let path: Vec<f32> = (0..samples)
            .flat_map(|k| {
                // A shallow "V" opening westward: both ends sit west of the
                // anchor, so the label overall reads right-to-left.
                let t = (k as f64 - mid as f64) * 10.0;
                [(-t.abs()) as f32, (t * 0.05) as f32]
            })
            .collect();
        let mut l = label(std::f64::consts::FRAC_PI_2, true, false);
        l[9] = 10.0;
        let out = line_label_place(&l, &path, samples, &view(), 1000.0, 1.0);
        assert_eq!(out[0], 0.0, "a symmetric V has no net direction to flip");

        // Now a plain westward road whose anchor bearing wrongly claims east.
        let west_path = directed_path(samples, 10.0, (-1.0, 0.0));
        let out = line_label_place(&l, &west_path, samples, &view(), 1000.0, 1.0);
        assert_eq!(out[0], 1.0, "the chord should win over the bearing");
    }

    #[test]
    fn the_collision_box_turns_with_the_label() {
        // The fixture's box is 40 wide and 10 tall. Running east it stays that
        // way; running north it must come back 10 wide and 40 tall, or the
        // declutter grid models a vertical street label as a horizontal one and
        // lets its neighbours overlap it.
        let l = label(std::f64::consts::FRAC_PI_2, false, false);

        let east = line_label_place(&l, &straight_path(32, 1.0), 32, &view(), 1000.0, 1.0);
        let (ew, eh) = (east[3] - east[2], east[5] - east[4]);
        assert!((ew - 40.0).abs() < 1e-6, "east width {ew}");
        assert!((eh - 10.0).abs() < 1e-6, "east height {eh}");

        let north = line_label_place(
            &l,
            &directed_path(32, 1.0, (0.0, 1.0)),
            32,
            &view(),
            1000.0,
            1.0,
        );
        let (nw, nh) = (north[3] - north[2], north[5] - north[4]);
        assert!((nw - 10.0).abs() < 1e-6, "north width {nw}");
        assert!((nh - 40.0).abs() < 1e-6, "north height {nh}");
    }

    #[test]
    fn labels_longer_than_their_line_are_rejected() {
        let mut l = label(std::f64::consts::FRAC_PI_2, true, false);
        // 4 ems at 10 m/em is 40 m long, so 15 m of road either side is not
        // enough to hold it.
        l[10] = 15.0;
        let path = straight_path(32, 1.0);
        let out = line_label_place(&l, &path, 32, &view(), 1000.0, 1.0);
        assert_eq!(out[1], 1.0, "should be rejected");
    }

    #[test]
    fn a_sharp_kink_under_the_label_is_rejected() {
        // A right-angle turn at the anchor: 90 deg of bend inside one window,
        // well past the 45 deg limit.
        let samples = 32;
        let mid = samples / 2;
        let path: Vec<f32> = (0..samples)
            .flat_map(|k| {
                if k <= mid {
                    [((k as f64 - mid as f64) * 1.0) as f32, 0.0f32]
                } else {
                    [0.0f32, ((k - mid) as f64 * 1.0) as f32]
                }
            })
            .collect();
        let out = line_label_place(
            &label(std::f64::consts::FRAC_PI_2, true, false),
            &path,
            samples,
            &view(),
            1000.0,
            1.0,
        );
        assert_eq!(out[1], 1.0, "sharp kink should be rejected");
    }

    #[test]
    fn a_gentle_curve_is_accepted_even_when_its_total_bend_is_large() {
        // A wide arc turning 90 deg overall, but only a couple of degrees
        // within any window — exactly the case a whole-label angle sum would
        // wrongly reject, and the reason the window slides.
        let samples = 32;
        let mid = samples as f64 / 2.0;
        let radius = 400.0;
        let path: Vec<f32> = (0..samples)
            .flat_map(|k| {
                let t = (k as f64 - mid) / samples as f64 * std::f64::consts::FRAC_PI_2;
                [(radius * t.sin()) as f32, (radius * (1.0 - t.cos())) as f32]
            })
            .collect();
        let mut l = label(std::f64::consts::FRAC_PI_2, true, false);
        l[9] = radius * std::f64::consts::FRAC_PI_2 / samples as f64; // step
        let out = line_label_place(&l, &path, samples, &view(), 1000.0, 1.0);
        assert_eq!(out[1], 0.0, "gentle curve should be accepted");
    }

    #[test]
    fn a_pixel_sized_label_grows_as_the_camera_pulls_back() {
        // Same label, same road, two camera distances: metres per em scales
        // with view depth, so the far camera's label overruns the road while
        // the near one fits.
        let mut l = label(std::f64::consts::FRAC_PI_2, true, false);
        l[6] = 0.0; // sizeInMeters = false, so fontSize is pixels
        l[10] = 60.0; // 60 m of road either side
        let path = straight_path(32, 10.0);

        let near_view = top_down_view(500.0);
        let far_view = top_down_view(20_000.0);

        let near = line_label_place(&l, &path, 32, &near_view, 1000.0, 1.0);
        let far = line_label_place(&l, &path, 32, &far_view, 1000.0, 1.0);
        assert_eq!(near[1], 0.0, "close camera: label fits");
        assert_eq!(far[1], 1.0, "far camera: label overruns the road");
    }
}
