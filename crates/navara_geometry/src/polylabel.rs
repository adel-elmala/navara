//! Pole of inaccessibility — where a polygon's label goes.

use std::collections::BinaryHeap;

/// The point inside a polygon farthest from its boundary, to within
/// `precision`, as MapLibre places a polygon's symbol (Mapbox's `polylabel`).
///
/// `rings[0]` is the outer ring and the rest are holes, in any planar units;
/// a ring may or may not repeat its first vertex. Unlike a centroid, the result
/// is always inside the polygon, and away from narrow parts of it. A polygon
/// whose bounding box is no wider than `precision` answers its minimum corner;
/// one with no vertices answers `None`.
pub fn pole_of_inaccessibility<R: AsRef<[(f64, f64)]>>(
    rings: &[R],
    precision: f64,
) -> Option<(f64, f64)> {
    let outer = rings.first()?.as_ref();
    let (first, rest) = outer.split_first()?;
    let (mut min, mut max) = (*first, *first);
    for &(x, y) in rest {
        min = (min.0.min(x), min.1.min(y));
        max = (max.0.max(x), max.1.max(y));
    }
    // A polygon narrower than `precision` would seed `long side / short side`
    // cells for an answer no better than its corner.
    let cell_size = (max.0 - min.0).min(max.1 - min.1);
    if cell_size <= precision {
        return Some(min);
    }

    let cell = |x: f64, y: f64, half: f64| Cell::new(x, y, half, rings);
    let half = cell_size / 2.0;
    let mut queue = BinaryHeap::new();
    let mut x = min.0;
    while x < max.0 {
        let mut y = min.1;
        while y < max.1 {
            queue.push(cell(x + half, y + half, half));
            y += cell_size;
        }
        x += cell_size;
    }

    let mut best = centroid_cell(outer, rings);
    let center = cell((min.0 + max.0) / 2.0, (min.1 + max.1) / 2.0, 0.0);
    if center.distance > best.distance {
        best = center;
    }

    // The queue pops the cell that could hold the farthest point first, so
    // once that one cannot beat `best` by more than `precision`, none can.
    while let Some(c) = queue.pop() {
        if c.distance > best.distance {
            best = c;
        }
        if c.potential - best.distance <= precision {
            break;
        }
        let half = c.half / 2.0;
        for (dx, dy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
            queue.push(cell(c.x + dx * half, c.y + dy * half, half));
        }
    }
    Some((best.x, best.y))
}

/// A square of the search grid, ordered by the farthest a point inside it
/// could possibly be from the boundary.
#[derive(Clone, Copy)]
struct Cell {
    x: f64,
    y: f64,
    half: f64,
    /// Signed distance from the center to the boundary, positive inside.
    distance: f64,
    /// Upper bound on `distance` anywhere in the cell.
    potential: f64,
}

impl Cell {
    fn new<R: AsRef<[(f64, f64)]>>(x: f64, y: f64, half: f64, rings: &[R]) -> Self {
        let distance = signed_distance((x, y), rings);
        Self {
            x,
            y,
            half,
            distance,
            potential: distance + half * std::f64::consts::SQRT_2,
        }
    }
}

impl PartialEq for Cell {
    fn eq(&self, other: &Self) -> bool {
        self.potential.total_cmp(&other.potential).is_eq()
    }
}

impl Eq for Cell {}

impl PartialOrd for Cell {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cell {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.potential.total_cmp(&other.potential)
    }
}

/// The outer ring's area centroid as a starting guess, or its first vertex
/// when the ring has no area.
fn centroid_cell<R: AsRef<[(f64, f64)]>>(outer: &[(f64, f64)], rings: &[R]) -> Cell {
    let mut area = 0.0;
    let (mut cx, mut cy) = (0.0, 0.0);
    let mut prev = outer[outer.len() - 1];
    for &p in outer {
        let f = p.0 * prev.1 - prev.0 * p.1;
        cx += (p.0 + prev.0) * f;
        cy += (p.1 + prev.1) * f;
        area += f * 3.0;
        prev = p;
    }
    if area == 0.0 {
        return Cell::new(outer[0].0, outer[0].1, 0.0, rings);
    }
    Cell::new(cx / area, cy / area, 0.0, rings)
}

/// Distance from `p` to the nearest ring edge, positive inside the polygon
/// (even-odd over every ring, so holes count as outside).
fn signed_distance<R: AsRef<[(f64, f64)]>>(p: (f64, f64), rings: &[R]) -> f64 {
    let mut inside = false;
    let mut min_sq = f64::INFINITY;
    for ring in rings {
        let ring = ring.as_ref();
        let Some(&last) = ring.last() else { continue };
        let mut b = last;
        for &a in ring {
            if (a.1 > p.1) != (b.1 > p.1) && p.0 < (b.0 - a.0) * (p.1 - a.1) / (b.1 - a.1) + a.0 {
                inside = !inside;
            }
            min_sq = min_sq.min(segment_distance_sq(p, a, b));
            b = a;
        }
    }
    let d = min_sq.sqrt();
    if inside { d } else { -d }
}

/// Squared distance from `p` to the segment `a`–`b`.
fn segment_distance_sq(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (mut x, mut y) = a;
    let (dx, dy) = (b.0 - x, b.1 - y);
    if dx != 0.0 || dy != 0.0 {
        let t = ((p.0 - x) * dx + (p.1 - y) * dy) / (dx * dx + dy * dy);
        if t > 1.0 {
            (x, y) = b;
        } else if t > 0.0 {
            x += dx * t;
            y += dy * t;
        }
    }
    (p.0 - x).powi(2) + (p.1 - y).powi(2)
}

#[cfg(test)]
mod test {
    use super::*;

    fn square(min: f64, max: f64) -> Vec<(f64, f64)> {
        vec![(min, min), (max, min), (max, max), (min, max), (min, min)]
    }

    #[test]
    fn square_labels_at_its_center() {
        let (x, y) = pole_of_inaccessibility(&[square(0.0, 10.0)], 0.01).unwrap();
        assert!((x - 5.0).abs() < 0.1 && (y - 5.0).abs() < 0.1, "({x}, {y})");
    }

    #[test]
    fn open_and_closed_rings_agree() {
        let closed = square(0.0, 10.0);
        assert_eq!(
            pole_of_inaccessibility(&[&closed[..]], 0.01),
            pole_of_inaccessibility(&[&closed[..4]], 0.01),
        );
    }

    #[test]
    fn hole_pushes_the_label_off_center() {
        let rings = [square(0.0, 10.0), square(3.0, 7.0)];
        let p = pole_of_inaccessibility(&rings, 0.01).unwrap();
        // Inside the ring band, and not hugging either of its edges.
        assert!(signed_distance(p, &rings) > 1.0, "{p:?}");
    }

    #[test]
    fn concave_polygon_labels_inside_its_widest_part() {
        // An L whose area centroid falls outside the polygon, in the notch.
        let l = vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 2.0),
            (2.0, 2.0),
            (2.0, 10.0),
            (0.0, 10.0),
        ];
        let p = pole_of_inaccessibility(&[&l], 0.01).unwrap();
        assert!(signed_distance(p, &[&l]) > 0.9, "{p:?}");
    }

    #[test]
    fn polygon_without_area_answers_its_minimum_corner() {
        let line = vec![(1.0, 5.0), (4.0, 5.0), (1.0, 5.0)];
        assert_eq!(pole_of_inaccessibility(&[line], 0.01), Some((1.0, 5.0)));
    }

    #[test]
    fn sliver_narrower_than_precision_answers_its_minimum_corner() {
        let sliver = vec![(0.0, 0.0), (1000.0, 0.0), (1000.0, 1e-7), (0.0, 1e-7)];
        assert_eq!(pole_of_inaccessibility(&[sliver], 1.0), Some((0.0, 0.0)));
    }

    #[test]
    fn polygon_without_vertices_answers_none() {
        let empty: [Vec<(f64, f64)>; 1] = [Vec::new()];
        assert_eq!(pole_of_inaccessibility(&empty, 0.01), None);
        let none: [Vec<(f64, f64)>; 0] = [];
        assert_eq!(pole_of_inaccessibility(&none, 0.01), None);
    }
}
