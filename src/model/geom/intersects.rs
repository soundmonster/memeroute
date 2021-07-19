use crate::model::geom::distance::{circ_rt_dist, rt_seg_dist};
use crate::model::geom::math::{le, lt, ne, orientation, pts_same_side};
use crate::model::primitive::capsule::Capsule;
use crate::model::primitive::circle::Circle;
use crate::model::primitive::line_shape::Line;
use crate::model::primitive::path_shape::Path;
use crate::model::primitive::polygon::Polygon;
use crate::model::primitive::rect::Rt;
use crate::model::primitive::segment::Segment;
use crate::model::primitive::triangle::Tri;
use crate::model::primitive::{cap, line};

pub fn cap_intersect_rt(a: &Capsule, b: &Rt) -> bool {
    if b.contains(a.st()) || b.contains(a.en()) {
        true
    } else {
        le(rt_seg_dist(b, &a.seg()), a.r())
    }
}

pub fn circ_intersect_rt(a: &Circle, b: &Rt) -> bool {
    // Check if the circle centre is contained in the rect or
    // the distance from the boundary of the rect to the circle is less than 0.
    b.contains(a.p()) || lt(circ_rt_dist(a, b), 0.0)
}

pub fn line_intersects_line(a: &Line, b: &Line) -> bool {
    // Intersects if not parallel.
    ne(a.dir().cross(b.dir()), 0.0)
}

pub fn line_intersects_seg(_a: &Line, _b: &Segment) -> bool {
    todo!()
}

pub fn path_intersects_rt(a: &Path, b: &Rt) -> bool {
    // Check whether each capsule in the path intersects the rectangle.
    for &[st, en] in a.pts().array_windows::<2>() {
        if cap_intersect_rt(&cap(st, en, a.r()), b) {
            return true;
        }
    }
    false
}

pub fn poly_intersects_rt(a: &Polygon, b: &Rt) -> bool {
    for tri in a.tri() {
        if rt_intersects_tri(b, tri) {
            return true;
        }
    }
    false
}

pub fn rt_intersects_rt(a: &Rt, b: &Rt) -> bool {
    a.intersects(b)
}

pub fn rt_intersects_tri(a: &Rt, b: &Tri) -> bool {
    let rt = &a.pts();
    let tri = b.pts();
    // Test tri axes:
    if pts_same_side(&line(tri[0], tri[1]), rt) {
        return false;
    }
    if pts_same_side(&line(tri[1], tri[2]), rt) {
        return false;
    }
    if pts_same_side(&line(tri[2], tri[0]), rt) {
        return false;
    }
    // Test rect axes:
    if pts_same_side(&line(rt[0], rt[1]), tri) {
        return false;
    }
    if pts_same_side(&line(rt[1], rt[2]), tri) {
        return false;
    }
    if pts_same_side(&line(rt[2], rt[3]), tri) {
        return false;
    }
    if pts_same_side(&line(rt[3], rt[0]), tri) {
        return false;
    }
    true
}

pub fn rt_intersects_seg(_a: &Rt, _b: &Segment) -> bool {
    todo!()
}

pub fn seg_intersects_seg(a: &Segment, b: &Segment) -> bool {
    // Check if the segment endpoints are on opposite sides of the other segment.
    let a_st = orientation(&b.line(), a.st());
    let a_en = orientation(&b.line(), a.en());
    let b_st = orientation(&a.line(), b.st());
    let b_en = orientation(&a.line(), b.en());
    // No collinear points. Everything on different sides.
    if a_st != a_en && b_st != b_en {
        return true;
    }
    // Check collinear cases. Need to check both x and y coordinates to handle
    // vertical and horizontal segments.
    let a_rt = Rt::enclosing(a.st(), a.en());
    let b_rt = Rt::enclosing(b.st(), b.en());
    if a_st == 0 && b_rt.contains(a.st()) {
        return true;
    }
    if a_en == 0 && b_rt.contains(a.en()) {
        return true;
    }
    if b_st == 0 && a_rt.contains(b.st()) {
        return true;
    }
    if b_en == 0 && a_rt.contains(b.en()) {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::primitive::{pt, seg};
    use crate::model::tf::Tf;

    const SEG_SEG_TESTS: &[(Segment, Segment, bool)] = &[
        // Crossing
        (seg(pt(1.0, 1.0), pt(3.0, 4.0)), seg(pt(2.0, 4.0), pt(3.0, 1.0)), true),
        // Shared endpoints, not parallel
        (seg(pt(1.0, 1.0), pt(2.0, 3.0)), seg(pt(2.0, 3.0), pt(4.0, 1.0)), true),
        // Shared endpoints, parallel, one point of intersection
        (seg(pt(1.0, 1.0), pt(3.0, 2.0)), seg(pt(3.0, 2.0), pt(5.0, 3.0)), true),
        // Endpoint abutting segment, perpendicular
        (seg(pt(1.0, 1.0), pt(3.0, 3.0)), seg(pt(2.0, 4.0), pt(4.0, 2.0)), true),
        // Same segments
        (seg(pt(1.0, 1.0), pt(1.0, 1.0)), seg(pt(1.0, 1.0), pt(1.0, 1.0)), true),
        // Parallel and overlapping
        (seg(pt(1.0, 1.0), pt(3.0, 1.0)), seg(pt(2.0, 1.0), pt(4.0, 1.0)), true),
        // Parallel and contained
        (seg(pt(1.0, 1.0), pt(4.0, 1.0)), seg(pt(2.0, 1.0), pt(3.0, 1.0)), true),
        // Parallel segments with one shared endpoint overlapping
        (seg(pt(1.0, 1.0), pt(3.0, 1.0)), seg(pt(1.0, 1.0), pt(4.0, 1.0)), true),
        // Degenerate: One segment is a point, on the other segment.
        (seg(pt(1.0, 1.0), pt(3.0, 1.0)), seg(pt(2.0, 1.0), pt(2.0, 1.0)), true),
        // Degenerate: One segment is a point, on the other segment's endpoint
        (seg(pt(1.0, 1.0), pt(3.0, 1.0)), seg(pt(3.0, 1.0), pt(3.0, 1.0)), true),
        // Degenerate: Same segments and they are points
        (seg(pt(1.0, 1.0), pt(1.0, 1.0)), seg(pt(1.0, 1.0), pt(1.0, 1.0)), true),
        // Parallel, not intersecting
        (seg(pt(1.0, 3.0), pt(3.0, 1.0)), seg(pt(2.0, 4.0), pt(4.0, 2.0)), false),
        // Perpendicular, not intersecting, projection of endpoint onto other is
        // an endpoint
        (seg(pt(1.0, 1.0), pt(3.0, 3.0)), seg(pt(4.0, 2.0), pt(5.0, 1.0)), false),
        // Perpendicular, not intersecting
        (seg(pt(1.0, 1.0), pt(3.0, 3.0)), seg(pt(3.0, 1.0), pt(4.0, 0.0)), false),
        // Degenerate: Both are points, not intersecting
        (seg(pt(1.0, 1.0), pt(1.0, 1.0)), seg(pt(2.0, 1.0), pt(2.0, 1.0)), false),
        // Degenerate: One is a point, collinear with the other segment, not intersecting.
        (seg(pt(1.0, 1.0), pt(3.0, 3.0)), seg(pt(4.0, 4.0), pt(4.0, 4.0)), false),
        // Degenerate: One is a point, not intersecting.
        (seg(pt(1.0, 1.0), pt(3.0, 3.0)), seg(pt(1.0, 2.0), pt(1.0, 2.0)), false),
    ];

    fn test_seg_seg_permutations(a: &Segment, b: &Segment, res: bool) {
        // Try each permutation of orderings
        assert_eq!(seg_intersects_seg(a, b), res, "{} {} intersects? {}", a, b, res);
        assert_eq!(seg_intersects_seg(b, a), res, "{} {} intersects? {}", a, b, res);
        let a = seg(a.en(), a.st());
        let b = seg(b.en(), b.st());
        assert_eq!(seg_intersects_seg(&a, &b), res, "{} {} intersects? {}", a, b, res);
        assert_eq!(seg_intersects_seg(&b, &a), res, "{} {} intersects? {}", a, b, res);
    }

    #[test]
    fn test_seg_seg() {
        for (a, b, res) in SEG_SEG_TESTS {
            test_seg_seg_permutations(a, b, *res);
            // Negating pts should not change result.
            let a = &seg(-a.st(), -a.en());
            let b = &seg(-b.st(), -b.en());
            test_seg_seg_permutations(a, b, *res);
            // Rotating should not change result.
            let tf = Tf::rotate(42.0);
            let a = &tf.seg(a);
            let b = &tf.seg(b);
            test_seg_seg_permutations(a, b, *res);
            // Translating should not change result.
            let tf = Tf::translate(pt(-3.0, 4.0));
            let a = &tf.seg(a);
            let b = &tf.seg(b);
            test_seg_seg_permutations(a, b, *res);
            // Scaling should not change result.
            let tf = Tf::scale(pt(-0.4, 0.7));
            let a = &tf.seg(a);
            let b = &tf.seg(b);
            test_seg_seg_permutations(a, b, *res);
        }
    }
}
