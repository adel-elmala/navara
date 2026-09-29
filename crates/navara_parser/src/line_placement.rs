//! Anchors spaced along line geometry, shared by every source that derives
//! point-like anchors (points, billboards, text) from lines.
//!
//! Everything here works in a planar, **conformal** frame whose y axis grows
//! southward: MVT tile units, or Web Mercator for sources in degrees. Conformal
//! is what makes [`tangent_to_bearing`] exact — a direction in the frame is the
//! same direction on the ground — while the frame's scale may vary with
//! latitude, which is why metre conversions take a per-anchor
//! `meters_per_unit`.

/// How an emitter derives anchors from line geometry.
///
/// Mirrors `navara_material::Placement` but is ECS-free, for the same reason
/// `LayerParseKind` mirrors `GeometryAppearanceKind`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PointPlacement {
    /// One anchor per source vertex.
    #[default]
    Point,
    /// Anchors repeated along the line at `spacing` intervals.
    Line,
    /// A single anchor at the line's arc-length midpoint.
    LineCenter,
}

impl PointPlacement {
    /// Whether this mode resamples the line rather than emitting per vertex.
    pub fn is_along_line(self) -> bool {
        matches!(self, Self::Line | Self::LineCenter)
    }
}

/// Path samples stored per along-line text anchor.
///
/// The samples are uniform in arc length, which is the whole point: the vertex
/// shader finds the segment containing a glyph with `floor(s / step)` instead
/// of walking the path, so bending a glyph costs two texel fetches rather than
/// a loop.
pub const PATH_SAMPLES: usize = 32;

// The transfer attribute's `size` is a u8 holding the two-floats-per-sample
// stride, so the count cannot exceed 127.
const _: () = assert!(PATH_SAMPLES * 2 <= u8::MAX as usize);

/// Arc length a label's sampled path covers, as a multiple of the anchor
/// spacing. Repeated labels are spaced by `spacing`, so a label that needs more
/// than this would collide with its own neighbours anyway; it is rejected
/// instead of being given a longer path.
const PATH_SPAN_SPACINGS: f64 = 2.0;

/// Most anchors one line may receive. `spacing` comes straight from the style,
/// so a vanishingly small value would otherwise ask for an unbounded number of
/// them; flooring the spacing instead of truncating the count keeps the anchors
/// spread along the whole line.
const MAX_ANCHORS_PER_LINE: f64 = 10_000.0;

/// Scalars stored per anchor alongside its path samples: the metre step between
/// samples, then the arc length of *real* line either side of the anchor.
///
/// The second matters because samples past a line's end are extrapolated (so
/// the tangent never degenerates), which leaves the renderer unable to tell
/// where the road actually stopped. Without it a name would happily run along
/// 200 m of imaginary straight road off the end of a 30 m stub.
pub const PATH_META_STRIDE: usize = 2;

/// The sampled line under one along-line text anchor.
#[derive(Clone, Debug, PartialEq)]
pub struct AnchorPath {
    /// [`PATH_SAMPLES`] east/north metre offsets from the anchor.
    pub samples: Vec<f32>,
    /// See [`PATH_META_STRIDE`].
    pub meta: [f32; PATH_META_STRIDE],
}

/// A linestring prepared for arc-length queries.
pub struct LinePath<'a> {
    verts: &'a [(f64, f64)],
    /// Prefix sums of segment lengths, one per vertex.
    cum: Vec<f64>,
}

impl<'a> LinePath<'a> {
    /// `None` when the line has fewer than two vertices or no length, which
    /// would make every arc-length query degenerate.
    pub fn new(verts: &'a [(f64, f64)]) -> Option<Self> {
        if verts.len() < 2 {
            return None;
        }
        let mut cum = Vec::with_capacity(verts.len());
        cum.push(0.0);
        let mut total = 0.0;
        for w in verts.windows(2) {
            total += (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1);
            cum.push(total);
        }
        (total > 0.0).then_some(Self { verts, cum })
    }

    /// Total arc length, in frame units.
    pub fn length(&self) -> f64 {
        self.cum[self.cum.len() - 1]
    }

    /// The segment containing arc length `s` (clamped to the line) and the
    /// fraction along it, for interpolating per-vertex values such as height.
    pub fn segment_at(&self, s: f64) -> (usize, f64) {
        let clamped = s.clamp(0.0, self.length());
        // The first segment whose end is at or past `clamped` contains it.
        // Zero-length segments (repeated vertices) are skipped: a line opening
        // with a duplicate would otherwise hand every query at its start that
        // segment, whose tangent is undefined, and the extrapolation before the
        // first anchor would run off in an arbitrary direction. The line has
        // positive length, so some segment always qualifies.
        let seg = (0..self.verts.len() - 1)
            .find(|&i| self.cum[i + 1] >= clamped && self.cum[i + 1] > self.cum[i])
            .unwrap_or(self.verts.len() - 2);
        let len = self.cum[seg + 1] - self.cum[seg];
        let t = if len > 0.0 {
            (clamped - self.cum[seg]) / len
        } else {
            0.0
        };
        (seg, t)
    }

    /// Position and unit tangent at arc length `s`.
    ///
    /// Past either end the path is **extrapolated** along that end's tangent
    /// rather than clamped: a label longer than the line it sits on then runs
    /// straight off the end, which reads correctly and — unlike a clamped,
    /// zero-length end segment — never yields a degenerate tangent for the
    /// shader to normalize.
    pub fn sample(&self, s: f64) -> ((f64, f64), (f64, f64)) {
        let (seg, t) = self.segment_at(s);
        let (a, b) = (self.verts[seg], self.verts[seg + 1]);
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let norm = dx.hypot(dy);
        let tangent = if norm > 0.0 {
            (dx / norm, dy / norm)
        } else {
            (1.0, 0.0)
        };
        // Zero whenever `s` is on the path, so this is a no-op for anchor queries.
        let overshoot = s - s.clamp(0.0, self.length());
        (
            (
                a.0 + dx * t + tangent.0 * overshoot,
                a.1 + dy * t + tangent.1 * overshoot,
            ),
            tangent,
        )
    }

    /// Arc lengths of the anchors `placement` puts on this line, `spacing`
    /// frame units apart.
    ///
    /// A line shorter than one interval — and every `LineCenter` line — still
    /// deserves its one anchor, at the midpoint. Otherwise the pattern starts
    /// half an interval in and stops half an interval before the end, so an
    /// anchor never lands on an endpoint where a label has no line left to sit
    /// on (the renderer would reject it and the repeat would silently go
    /// missing instead of being spaced evenly).
    pub fn anchors(&self, placement: PointPlacement, spacing: f64) -> impl Iterator<Item = f64> {
        debug_assert!(placement.is_along_line());
        let total = self.length();
        let spacing = self.resolve_spacing(spacing);
        let single = placement == PointPlacement::LineCenter || total < spacing;
        let count = if single {
            1
        } else {
            // A line measuring a whole number of intervals must keep its last
            // anchor, which rounding in the projected length would otherwise
            // push a hair past the stop.
            ((total - spacing) / spacing + 1e-6).floor() as usize + 1
        };
        (0..count).map(move |i| {
            if single {
                total * 0.5
            } else {
                spacing * (0.5 + i as f64)
            }
        })
    }

    /// The spacing the resampler actually uses, given the style's.
    ///
    /// A zero, negative or non-finite value has no meaningful interval, so it
    /// falls back to the whole line — one anchor, at the midpoint — rather than
    /// being divided by. A valid one is floored at [`MAX_ANCHORS_PER_LINE`].
    fn resolve_spacing(&self, spacing: f64) -> f64 {
        let total = self.length();
        if spacing.is_finite() && spacing > 0.0 {
            spacing.max(total / MAX_ANCHORS_PER_LINE)
        } else {
            total
        }
    }

    /// [`PATH_SAMPLES`] points centred on the anchor at arc length `s`, as
    /// east/north metre offsets from it, for a pattern `spacing` frame units
    /// apart. `spacing` also bounds the sampled span for
    /// [`PointPlacement::LineCenter`], whose single anchor otherwise ignores it:
    /// the sample count is fixed, so sampling a long line whole would coarsen
    /// the path under a short label to a few straight chords.
    ///
    /// Metres rather than frame units because glyph sizes are metric
    /// downstream, and relative to the anchor so the values stay small enough
    /// for `f32` — a few hundred metres, against the ~6.4e6 of an absolute
    /// ECEF coordinate. `meters_per_unit` is the frame's ground scale at the
    /// anchor.
    pub fn anchor_path(&self, s: f64, spacing: f64, meters_per_unit: f64) -> AnchorPath {
        let spacing = self.resolve_spacing(spacing);
        let step = spacing * PATH_SPAN_SPACINGS / (PATH_SAMPLES - 1) as f64;
        let half_span = step * (PATH_SAMPLES - 1) as f64 * 0.5;
        let (origin, _) = self.sample(s);
        let mut samples = Vec::with_capacity(PATH_SAMPLES * 2);
        for k in 0..PATH_SAMPLES {
            let (p, _) = self.sample(s - half_span + step * k as f64);
            // The frame's y grows southward, so north is the negated delta.
            samples.push(((p.0 - origin.0) * meters_per_unit) as f32);
            samples.push(((origin.1 - p.1) * meters_per_unit) as f32);
        }
        let half_extent = s.min(self.length() - s) * meters_per_unit;
        AnchorPath {
            samples,
            meta: [(step * meters_per_unit) as f32, half_extent as f32],
        }
    }
}

/// Convert a frame tangent to a compass bearing in degrees.
///
/// East is `+x` and north is `-y`, so clockwise-from-north matches the
/// `rotation` field's sense.
pub fn tangent_to_bearing(tangent: (f64, f64)) -> f32 {
    tangent.0.atan2(-tangent.1).to_degrees() as f32
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn rejects_degenerate_lines() {
        assert!(LinePath::new(&[(0.0, 0.0)]).is_none());
        assert!(LinePath::new(&[(1.0, 1.0), (1.0, 1.0)]).is_none());
    }

    #[test]
    fn segment_at_locates_the_containing_segment() {
        let verts = [(0.0, 0.0), (10.0, 0.0), (10.0, 30.0)];
        let path = LinePath::new(&verts).unwrap();
        assert_eq!(path.segment_at(5.0), (0, 0.5));
        assert_eq!(path.segment_at(25.0), (1, 0.5));
        // Clamped past the end, so the last vertex's values are reused.
        assert_eq!(path.segment_at(100.0), (1, 1.0));
    }

    #[test]
    fn anchors_are_centred_on_the_line() {
        let verts = [(0.0, 0.0), (1000.0, 0.0)];
        let path = LinePath::new(&verts).unwrap();
        let line: Vec<f64> = path.anchors(PointPlacement::Line, 100.0).collect();
        assert_eq!(line.len(), 10);
        assert_eq!(line[0], 50.0);
        assert_eq!(line[9], 950.0);

        let center: Vec<f64> = path.anchors(PointPlacement::LineCenter, 100.0).collect();
        assert_eq!(center, vec![500.0]);

        // Shorter than one interval: one anchor at the midpoint, not none.
        let short: Vec<f64> = path.anchors(PointPlacement::Line, 5000.0).collect();
        assert_eq!(short, vec![500.0]);
    }

    #[test]
    fn invalid_spacing_gives_one_anchor_rather_than_unbounded_many() {
        let verts = [(0.0, 0.0), (1000.0, 0.0)];
        let path = LinePath::new(&verts).unwrap();
        for spacing in [0.0, -5.0, f64::NAN, f64::INFINITY] {
            let anchors: Vec<f64> = path.anchors(PointPlacement::Line, spacing).collect();
            assert_eq!(anchors, vec![500.0], "spacing {spacing}");
            let p = path.anchor_path(500.0, spacing, 1.0);
            assert!(
                p.meta[0].is_finite() && p.meta[0] > 0.0,
                "spacing {spacing}"
            );
            assert!(p.samples.iter().all(|v| v.is_finite()), "spacing {spacing}");
        }
        // Valid but absurdly dense: capped, and still spread over the line.
        let dense: Vec<f64> = path.anchors(PointPlacement::Line, 1e-12).collect();
        assert!(dense.len() <= MAX_ANCHORS_PER_LINE as usize);
        assert!(*dense.last().unwrap() > 999.0);
    }

    #[test]
    fn a_repeated_first_vertex_does_not_bend_the_start() {
        // The line runs due south (+y) but opens with a duplicate vertex. The
        // extrapolation before its start has to continue that direction, not
        // fall back to east for the zero-length first segment.
        let verts = [(0.0, 0.0), (0.0, 0.0), (0.0, 100.0)];
        let path = LinePath::new(&verts).unwrap();
        let (pos, tangent) = path.sample(-10.0);
        assert_eq!(tangent, (0.0, 1.0));
        assert!((pos.0 - 0.0).abs() < 1e-12 && (pos.1 + 10.0).abs() < 1e-12);
        assert_eq!(path.segment_at(0.0), (1, 0.0));
    }

    #[test]
    fn bearings_are_clockwise_from_north() {
        assert!((tangent_to_bearing((0.0, -1.0)) - 0.0).abs() < 1e-5);
        assert!((tangent_to_bearing((1.0, 0.0)) - 90.0).abs() < 1e-5);
        assert!((tangent_to_bearing((0.0, 1.0)) - 180.0).abs() < 1e-5);
    }

    #[test]
    fn anchor_path_scales_to_metres() {
        let verts = [(0.0, 0.0), (1000.0, 0.0)];
        let path = LinePath::new(&verts).unwrap();
        let p = path.anchor_path(500.0, 100.0, 2.0);
        assert_eq!(p.samples.len(), PATH_SAMPLES * 2);
        let step = (100.0 * PATH_SPAN_SPACINGS / (PATH_SAMPLES - 1) as f64 * 2.0) as f32;
        assert!((p.meta[0] - step).abs() < 1e-4);
        // 500 frame units either side at 2 m each.
        assert!((p.meta[1] - 1000.0).abs() < 1e-3);
    }
}
