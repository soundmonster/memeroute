use crate::model::geom::contains::polygon_contains_rt;
use crate::model::geom::intersects::rt_intersects_polygon;
use crate::model::geom::math::f64_eq;
use crate::model::primitive::circle::Circle;
use crate::model::primitive::line_shape::Line;
use crate::model::primitive::path_shape::Path;
use crate::model::primitive::point::Pt;
use crate::model::primitive::polygon::Polygon;
use crate::model::primitive::rect::Rt;
use crate::model::primitive::segment::Segment;
use crate::model::primitive::triangle::Tri;
use crate::model::primitive::{poly, ShapeOps};

#[derive(Debug, Clone)]
pub enum Shape {
    Circle(Circle),
    Line(Line),
    Path(Path),
    Point(Pt),
    Polygon(Polygon),
    Rect(Rt),
    Segment(Segment),
    Tri(Tri),
}

impl Shape {
    pub fn intersects(&self, s: &Shape) -> bool {
        match (self, s) {
            (Shape::Circle(_), Shape::Line(_)) => todo!(),
            (Shape::Circle(_), Shape::Point(_)) => todo!(),
            (Shape::Circle(_), Shape::Segment(_)) => todo!(),
            (Shape::Circle(_), Shape::Tri(_)) => todo!(),
            (Shape::Circle(_), Shape::Circle(_)) => todo!(),
            (Shape::Circle(_), Shape::Path(_)) => todo!(),
            (Shape::Circle(_), Shape::Polygon(_)) => todo!(),
            (Shape::Circle(_), Shape::Rect(_)) => todo!(),
            (Shape::Line(_), Shape::Circle(_)) => todo!(),
            (Shape::Line(_), Shape::Line(_)) => todo!(),
            (Shape::Line(_), Shape::Path(_)) => todo!(),
            (Shape::Line(_), Shape::Point(_)) => todo!(),
            (Shape::Line(_), Shape::Polygon(_)) => todo!(),
            (Shape::Line(_), Shape::Rect(_)) => todo!(),
            (Shape::Line(_), Shape::Segment(_)) => todo!(),
            (Shape::Line(_), Shape::Tri(_)) => todo!(),
            (Shape::Path(_), Shape::Line(_)) => todo!(),
            (Shape::Path(_), Shape::Point(_)) => todo!(),
            (Shape::Path(_), Shape::Segment(_)) => todo!(),
            (Shape::Path(_), Shape::Tri(_)) => todo!(),
            (Shape::Path(_), Shape::Circle(_)) => todo!(),
            (Shape::Path(_), Shape::Path(_)) => todo!(),
            (Shape::Path(_), Shape::Polygon(_)) => todo!(),
            (Shape::Path(_), Shape::Rect(_)) => todo!(),
            (Shape::Point(_), Shape::Circle(_)) => todo!(),
            (Shape::Point(_), Shape::Line(_)) => todo!(),
            (Shape::Point(_), Shape::Path(_)) => todo!(),
            (Shape::Point(_), Shape::Point(_)) => todo!(),
            (Shape::Point(_), Shape::Polygon(_)) => todo!(),
            (Shape::Point(_), Shape::Rect(_)) => todo!(),
            (Shape::Point(_), Shape::Segment(_)) => todo!(),
            (Shape::Point(_), Shape::Tri(_)) => todo!(),
            (Shape::Polygon(_), Shape::Line(_)) => todo!(),
            (Shape::Polygon(_), Shape::Point(_)) => todo!(),
            (Shape::Polygon(_), Shape::Segment(_)) => todo!(),
            (Shape::Polygon(_), Shape::Tri(_)) => todo!(),
            (Shape::Polygon(_), Shape::Circle(_)) => todo!(),
            (Shape::Polygon(_), Shape::Path(_)) => todo!(),
            (Shape::Polygon(_), Shape::Polygon(_)) => todo!(),
            (Shape::Polygon(a), Shape::Rect(b)) => rt_intersects_polygon(b, a),
            (Shape::Rect(_), Shape::Line(_)) => todo!(),
            (Shape::Rect(_), Shape::Point(_)) => todo!(),
            (Shape::Rect(_), Shape::Segment(_)) => todo!(),
            (Shape::Rect(_), Shape::Tri(_)) => todo!(),
            (Shape::Rect(_), Shape::Circle(_)) => todo!(),
            (Shape::Rect(_), Shape::Path(_)) => todo!(),
            (Shape::Rect(a), Shape::Polygon(b)) => rt_intersects_polygon(a, b),
            (Shape::Rect(_), Shape::Rect(_)) => todo!(),
            (Shape::Segment(_), Shape::Circle(_)) => todo!(),
            (Shape::Segment(_), Shape::Line(_)) => todo!(),
            (Shape::Segment(_), Shape::Path(_)) => todo!(),
            (Shape::Segment(_), Shape::Point(_)) => todo!(),
            (Shape::Segment(_), Shape::Polygon(_)) => todo!(),
            (Shape::Segment(_), Shape::Rect(_)) => todo!(),
            (Shape::Segment(_), Shape::Segment(_)) => todo!(),
            (Shape::Segment(_), Shape::Tri(_)) => todo!(),
            (Shape::Tri(_), Shape::Circle(_)) => todo!(),
            (Shape::Tri(_), Shape::Line(_)) => todo!(),
            (Shape::Tri(_), Shape::Path(_)) => todo!(),
            (Shape::Tri(_), Shape::Point(_)) => todo!(),
            (Shape::Tri(_), Shape::Polygon(_)) => todo!(),
            (Shape::Tri(_), Shape::Rect(_)) => todo!(),
            (Shape::Tri(_), Shape::Segment(_)) => todo!(),
            (Shape::Tri(_), Shape::Tri(_)) => todo!(),
        }
    }

    pub fn contains(&self, s: &Shape) -> bool {
        match (self, s) {
            (Shape::Circle(_), Shape::Line(_)) => todo!(),
            (Shape::Circle(_), Shape::Point(_)) => todo!(),
            (Shape::Circle(_), Shape::Segment(_)) => todo!(),
            (Shape::Circle(_), Shape::Tri(_)) => todo!(),
            (Shape::Circle(_), Shape::Circle(_)) => todo!(),
            (Shape::Circle(_), Shape::Path(_)) => todo!(),
            (Shape::Circle(_), Shape::Polygon(_)) => todo!(),
            (Shape::Circle(_), Shape::Rect(_)) => todo!(),
            (Shape::Line(_), Shape::Circle(_)) => todo!(),
            (Shape::Line(_), Shape::Line(_)) => todo!(),
            (Shape::Line(_), Shape::Path(_)) => todo!(),
            (Shape::Line(_), Shape::Point(_)) => todo!(),
            (Shape::Line(_), Shape::Polygon(_)) => todo!(),
            (Shape::Line(_), Shape::Rect(_)) => todo!(),
            (Shape::Line(_), Shape::Segment(_)) => todo!(),
            (Shape::Line(_), Shape::Tri(_)) => todo!(),
            (Shape::Path(_), Shape::Line(_)) => todo!(),
            (Shape::Path(_), Shape::Point(_)) => todo!(),
            (Shape::Path(_), Shape::Segment(_)) => todo!(),
            (Shape::Path(_), Shape::Tri(_)) => todo!(),
            (Shape::Path(_), Shape::Circle(_)) => todo!(),
            (Shape::Path(_), Shape::Path(_)) => todo!(),
            (Shape::Path(_), Shape::Polygon(_)) => todo!(),
            (Shape::Path(_), Shape::Rect(_)) => todo!(),
            (Shape::Point(_), Shape::Circle(_)) => todo!(),
            (Shape::Point(_), Shape::Line(_)) => todo!(),
            (Shape::Point(_), Shape::Path(_)) => todo!(),
            (Shape::Point(_), Shape::Point(_)) => todo!(),
            (Shape::Point(_), Shape::Polygon(_)) => todo!(),
            (Shape::Point(_), Shape::Rect(_)) => todo!(),
            (Shape::Point(_), Shape::Segment(_)) => todo!(),
            (Shape::Point(_), Shape::Tri(_)) => todo!(),
            (Shape::Polygon(_), Shape::Line(_)) => todo!(),
            (Shape::Polygon(_), Shape::Point(_)) => todo!(),
            (Shape::Polygon(_), Shape::Segment(_)) => todo!(),
            (Shape::Polygon(_), Shape::Tri(_)) => todo!(),
            (Shape::Polygon(_), Shape::Circle(_)) => todo!(),
            (Shape::Polygon(_), Shape::Path(_)) => todo!(),
            (Shape::Polygon(_), Shape::Polygon(_)) => todo!(),
            (Shape::Polygon(a), Shape::Rect(b)) => polygon_contains_rt(a, b),
            (Shape::Rect(_), Shape::Line(_)) => todo!(),
            (Shape::Rect(_), Shape::Point(_)) => todo!(),
            (Shape::Rect(_), Shape::Segment(_)) => todo!(),
            (Shape::Rect(_), Shape::Tri(_)) => todo!(),
            (Shape::Rect(_), Shape::Circle(_)) => todo!(),
            (Shape::Rect(_), Shape::Path(_)) => todo!(),
            (Shape::Rect(_), Shape::Polygon(_)) => todo!(),
            (Shape::Rect(_), Shape::Rect(_)) => todo!(),
            (Shape::Segment(_), Shape::Circle(_)) => todo!(),
            (Shape::Segment(_), Shape::Line(_)) => todo!(),
            (Shape::Segment(_), Shape::Path(_)) => todo!(),
            (Shape::Segment(_), Shape::Point(_)) => todo!(),
            (Shape::Segment(_), Shape::Polygon(_)) => todo!(),
            (Shape::Segment(_), Shape::Rect(_)) => todo!(),
            (Shape::Segment(_), Shape::Segment(_)) => todo!(),
            (Shape::Segment(_), Shape::Tri(_)) => todo!(),
            (Shape::Tri(_), Shape::Circle(_)) => todo!(),
            (Shape::Tri(_), Shape::Line(_)) => todo!(),
            (Shape::Tri(_), Shape::Path(_)) => todo!(),
            (Shape::Tri(_), Shape::Point(_)) => todo!(),
            (Shape::Tri(_), Shape::Polygon(_)) => todo!(),
            (Shape::Tri(_), Shape::Rect(_)) => todo!(),
            (Shape::Tri(_), Shape::Segment(_)) => todo!(),
            (Shape::Tri(_), Shape::Tri(_)) => todo!(),
        }
    }

    pub fn filled(self) -> Shape {
        match self {
            Shape::Path(s) => {
                assert!(f64_eq(s.r(), 0.0), "path width not supported for polygons");
                poly(s.pts()).shape()
            }
            s => s,
        }
    }
}

impl ShapeOps for Shape {
    fn bounds(&self) -> Rt {
        match self {
            Shape::Circle(s) => s.bounds(),
            Shape::Line(s) => s.bounds(),
            Shape::Path(s) => s.bounds(),
            Shape::Point(s) => s.bounds(),
            Shape::Polygon(s) => s.bounds(),
            Shape::Rect(s) => s.bounds(),
            Shape::Segment(s) => s.bounds(),
            Shape::Tri(s) => s.bounds(),
        }
    }

    fn shape(self) -> Shape {
        self
    }
}
