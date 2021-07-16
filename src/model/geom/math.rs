use approx::{relative_eq, relative_ne};

use crate::model::pt::Pt;

pub const EP: f64 = 1e-6;

pub fn pt_eq(a: Pt, b: Pt) -> bool {
    relative_eq!(a, b, epsilon = EP)
}

pub fn f64_eq(a: f64, b: f64) -> bool {
    relative_eq!(a, b, epsilon = EP)
}

pub fn f64_ne(a: f64, b: f64) -> bool {
    relative_ne!(a, b, epsilon = EP)
}

pub fn f64_gt(a: f64, b: f64) -> bool {
    f64_ne(a, b) && a > b
}

pub fn f64_ge(a: f64, b: f64) -> bool {
    f64_eq(a, b) || a > b
}

// Return cross-product of OA and OB.
pub fn cross_at(o: Pt, a: Pt, b: Pt) -> f64 {
    (o - a).cross(o - b)
}

// Returns true if p is strictly left of line defined by ST EN.
pub fn is_strictly_left_of(p: Pt, st: Pt, en: Pt) -> bool {
    f64_gt(cross_at(st, en, p), 0.0)
}

pub fn is_left_of(p: Pt, st: Pt, en: Pt) -> bool {
    f64_ge(cross_at(st, en, p), 0.0)
}

pub fn is_collinear(a: Pt, b: Pt, c: Pt) -> bool {
    f64_eq(cross_at(a, b, c), 0.0)
}
