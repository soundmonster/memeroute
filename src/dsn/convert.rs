use std::collections::HashMap;

use eyre::{eyre, Result};
use num::ToPrimitive;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

use crate::dsn::types::{
    DsnComponent, DsnDimensionUnit, DsnId, DsnImage, DsnKeepout, DsnKeepoutType, DsnNet,
    DsnPadstack, DsnPcb, DsnRect, DsnShape,
};
use crate::model::geom::{Pt, PtF, Rt};
use crate::model::pcb::{
    Arc, Circle, Component, Keepout, KeepoutType, Layer, Net, Padstack, Path, Pcb, PinRef, Polygon,
    Shape, ShapeType,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Converter {
    dsn: DsnPcb,
    pcb: Pcb,
    padstacks: HashMap<DsnId, Padstack>,
    images: HashMap<DsnId, Component>,
}

impl Converter {
    pub fn new(dsn: DsnPcb) -> Self {
        Self { dsn, pcb: Default::default(), padstacks: HashMap::new(), images: HashMap::new() }
    }

    fn mm(&self) -> Decimal {
        match self.dsn.resolution.dimension {
            DsnDimensionUnit::Inch => dec!(25.4),
            DsnDimensionUnit::Mil => dec!(0.0254),
            DsnDimensionUnit::Cm => dec!(10),
            DsnDimensionUnit::Mm => dec!(1),
            DsnDimensionUnit::Um => dec!(1000),
        }
    }

    fn coord(&self, v: Decimal) -> Decimal {
        self.mm() * Decimal::new(self.dsn.resolution.amount as i64, 0) * v
    }

    fn resolution(&self) -> Decimal {
        self.mm() / Decimal::new(self.dsn.resolution.amount as i64, 0)
    }

    fn i64(&self, v: Decimal) -> Result<i64> {
        self.coord(v)
            .to_i64()
            .ok_or_else(|| eyre!("value {} unrepresentable with given resolution", v))
    }

    fn u64(&self, v: Decimal) -> Result<u64> {
        self.coord(v)
            .to_u64()
            .ok_or_else(|| eyre!("value {} unrepresentable with given resolution", v))
    }

    fn rect(&self, v: &DsnRect) -> Result<Rt> {
        Ok(Rt {
            x: self.i64(v.rect.x)?,
            y: self.i64(v.rect.y)?,
            w: self.i64(v.rect.w)?,
            h: self.i64(v.rect.h)?,
        })
    }

    fn pt(&self, v: &PtF) -> Result<Pt> {
        Ok(Pt { x: self.i64(v.x)?, y: self.i64(v.y)? })
    }


    fn shape(&self, v: &DsnShape) -> Result<Shape> {
        Ok(match v {
            DsnShape::Rect(v) => {
                Shape { layer: v.layer_id.clone(), shape: ShapeType::Rect(self.rect(v)?) }
            }
            DsnShape::Circle(v) => Shape {
                layer: v.layer_id.clone(),
                shape: ShapeType::Circle(Circle {
                    r: self.u64(v.diameter / dec!(2))?,
                    p: self.pt(&v.p)?,
                }),
            },
            DsnShape::Polygon(v) => Shape {
                layer: v.layer_id.clone(),
                shape: ShapeType::Polygon(Polygon {
                    width: self.u64(v.aperture_width)?,
                    pts: v.pts.iter().map(|v| self.pt(v)).collect::<Result<_>>()?,
                }),
            },
            DsnShape::Path(v) => Shape {
                layer: v.layer_id.clone(),
                shape: ShapeType::Path(Path {
                    width: self.u64(v.aperture_width)?,
                    pts: v.pts.iter().map(|v| self.pt(v)).collect::<Result<_>>()?,
                }),
            },
            DsnShape::QArc(v) => Shape {
                layer: v.layer_id.clone(),
                shape: ShapeType::Arc(Arc {
                    width: self.u64(v.aperture_width)?,
                    start: self.pt(&v.start)?,
                    end: self.pt(&v.end)?,
                    center: self.pt(&v.center)?,
                }),
            },
        })
    }

    fn keepout(&self, v: &DsnKeepout) -> Result<Keepout> {
        Ok(Keepout {
            kind: match v.keepout_type {
                DsnKeepoutType::Keepout => KeepoutType::Keepout,
                DsnKeepoutType::ViaKeepout => KeepoutType::ViaKeepout,
                DsnKeepoutType::WireKeepout => KeepoutType::WireKeepout,
            },
            shape: self.shape(&v.shape)?,
        })
    }

    fn padstack(&self, v: &DsnPadstack) -> Result<Padstack> {
        Ok(Padstack {
            id: v.padstack_id.clone(),
            shapes: v.shapes.iter().map(|s| self.shape(&s.shape)).collect::<Result<_>>()?,
            attach: v.attach,
        })
    }

    fn image(&self, v: &DsnImage) -> Result<Component> {
        Ok(Component { ..Default::default() })
    }

    fn component(&self, v: &DsnComponent) -> Result<Component> {
        todo!()
    }

    fn net(&self, v: &DsnNet) -> Result<Net> {
        Ok(Net {
            id: v.net_id.clone(),
            pins: v
                .pins
                .iter()
                .map(|p| PinRef { component: p.component_id.clone(), pin: p.pin_id.clone() })
                .collect(),
        })
    }

    fn convert_padstacks(&mut self) -> Result<()> {
        for v in self.dsn.library.padstacks.iter() {
            if self.padstacks.insert(v.padstack_id.clone(), self.padstack(v)?).is_some() {
                return Err(eyre!("duplicate padstack with id {}", v.padstack_id));
            }
        }
        Ok(())
    }

    fn convert_images(&mut self) -> Result<()> {
        for v in self.dsn.library.images.iter() {
            if self.images.insert(v.image_id.clone(), self.image(v)?).is_some() {
                return Err(eyre!("duplicate image with id {}", v.padstack_id));
            }
        }
        Ok(())
    }

    pub fn convert(mut self) -> Result<Pcb> {
        self.pcb.set_id(&self.dsn.pcb_id);
        if self.dsn.unit.dimension != self.dsn.resolution.dimension {
            return Err(eyre!(
                "unit override unimplemented: {} {}",
                self.dsn.unit.dimension,
                self.dsn.resolution.dimension
            ));
        }
        self.pcb.set_resolution(self.resolution());
        self.convert_padstacks()?;
        self.convert_images()?;

        // Physical structure:
        for v in self.dsn.structure.layers.iter() {
            self.pcb.add_layer(Layer::new(&v.layer_name));
        }
        for v in self.dsn.structure.boundaries.iter() {
            self.pcb.add_boundary(self.shape(v)?);
        }
        for v in self.dsn.structure.keepouts.iter() {
            self.pcb.add_keepout(self.keepout(v)?);
        }
        for v in self.dsn.structure.vias.iter() {
            self.pcb.add_via_padstack(
                self.padstacks.get(v).ok_or_else(|| eyre!("unknown padstack id {}", v))?.clone(),
            );
        }
        for v in self.dsn.placement.components.iter() {
            self.pcb.add_component(self.component(v)?);
        }

        // Routing:
        for v in self.dsn.network.nets.iter() {
            self.pcb.add_net(self.net(v)?);
        }

        // TODO: Add wires
        // TODO: Add vias
        // TODO: Support classes for nets.
        Ok(self.pcb)
    }
}
