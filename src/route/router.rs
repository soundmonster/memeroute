use std::sync::Mutex;

use derive_more::{Deref, DerefMut, Display};
use eyre::Result;
use memega::eval::Evaluator;
use memega::evolve::cfg::{
    Crossover, Duplicates, EvolveCfg, Mutation, Niching, Replacement, Stagnation, Survival,
};
use memega::evolve::evolver::Evolver;
use memega::ops::crossover::{crossover_cycle, crossover_order, crossover_pmx};
use memega::ops::distance::kendall_tau;
use memega::ops::mutation::{mutate_insert, mutate_inversion, mutate_scramble, mutate_swap};
use memega::train::cfg::{Termination, TrainerCfg};
use memega::train::sampler::EmptyDataSampler;
use memega::train::trainer::Trainer;
use memegeom::primitive::path_shape::Path;
use memegeom::primitive::point::Pt;
use memegeom::primitive::rect::Rt;
use memegeom::primitive::shape::Shape;
use memegeom::primitive::ShapeOps;
use rand::prelude::SliceRandom;
use rand::Rng;
use spade::{DelaunayTriangulation, Point2, Triangulation};

use crate::model::pcb::{Pcb, Via, Wire};
use crate::name::Id;
use crate::route::grid::GridRouter;

pub trait RouteStrategy {
    fn route(&mut self) -> Result<RouteResult>;
}

#[must_use]
#[derive(Debug, Default, Clone)]
pub struct RouteResult {
    pub wires: Vec<Wire>,
    pub vias: Vec<Via>,
    pub debug_rts: Vec<Rt>,
    pub failed: bool,
}

impl RouteResult {
    pub fn merge(&mut self, r: RouteResult) {
        self.wires.extend(r.wires);
        self.vias.extend(r.vias);
        self.debug_rts.extend(r.debug_rts);
        self.failed |= r.failed;
    }
}

#[must_use]
#[derive(Debug)]
pub struct Router {
    pcb: Mutex<Pcb>,
}

impl Clone for Router {
    fn clone(&self) -> Self {
        Self::new(self.pcb.lock().unwrap().clone())
    }
}

impl Router {
    pub fn new(pcb: Pcb) -> Self {
        Self { pcb: Mutex::new(pcb) }
    }

    pub fn rand_net_order(&self) -> Vec<Id> {
        let mut net_order: Vec<_> = self.pcb.lock().unwrap().nets().map(|v| v.id).collect();
        //net_order.shuffle(rand::thread_rng());
        net_order.sort_unstable();
        net_order
    }

    pub fn route(&self, net_order: Vec<Id>) -> Result<RouteResult> {
        let mut grid = GridRouter::new(self.pcb.lock().unwrap().clone(), net_order);
        grid.route()
    }

    pub fn get_ratsnest(&self) -> Vec<Shape> {
        let pcb = self.pcb.lock().unwrap();
        pcb.nets()
            .map(|net| {
                let path_points: Vec<Pt> = net
                    .pins
                    .iter()
                    .map(|pin_ref| {
                        let (component, pin) = pcb.pin_ref(pin_ref).unwrap();
                        (component.tf() * pin.tf()).pt(Pt::zero())
                    })
                    .collect();
                Path::new(&path_points, 0.1).shape()
            })
            .collect()
    }

    pub fn triangulate(&self) -> Vec<Shape> {
        let pcb = self.pcb.lock().unwrap();
        let points_from_components = pcb.components().flat_map(|component| {
            let pins = component.pins().map(|pin| (component.tf() * pin.tf()).pt(Pt::zero()));
            let keepouts = component.keepouts.iter().map(|keepout| {
                let mut s = keepout.shape.shape.clone();
                s.apply(&component.tf());
                s.bounds().center()
            });
            pins.chain(keepouts)
        });
        let keepouts = pcb.keepouts().iter().map(|keepout| keepout.shape.shape.bounds().center());
        let points_from_boundaries = pcb.boundaries().iter().flat_map(|ls| {
            // TODO this needs to contain "interesting" points along the boundary, not just the corners of the bounding box
            let r = ls.shape.bounds();
            [r.tr(), r.br(), r.bl(), r.tl()]
        });

        let all_points: Vec<Pt> =
            points_from_components.chain(points_from_boundaries).chain(keepouts).collect();
        let mut triangulation: DelaunayTriangulation<_> = DelaunayTriangulation::new();
        for p in all_points.clone() {
            triangulation.insert(Point2::new(p.x, p.y));
            println!(
                "{} {} {}",
                triangulation.num_vertices(),
                triangulation.num_inner_faces(),
                triangulation.num_undirected_edges()
            );
        }

        let t_edges = triangulation.undirected_edges().map(|edge| {
            let [s, e] = edge.positions();
            Path::new(&[Pt::new(s.x, s.y), Pt::new(e.x, e.y)], 0.1).shape()
        });

        all_points
            .iter()
            .map(|c| memegeom::primitive::circ(*c, 0.3).shape())
            .chain(t_edges)
            .collect()
    }

    pub fn run_ga(&self) -> Result<RouteResult> {
        let cfg = EvolveCfg::new(32)
            .set_mutation(Mutation::Adaptive)
            .set_crossover(Crossover::Adaptive)
            .set_survival(Survival::TopProportion(0.1))
            .set_niching(Niching::None)
            .set_stagnation(Stagnation::ContinuousAfter(200))
            .set_replacement(Replacement::ReplaceChildren(0.5))
            .set_duplicates(Duplicates::DisallowDuplicates)
            .set_par_fitness(true)
            .set_par_dist(true);

        let net_order: Vec<_> = self.pcb.lock().unwrap().nets().map(|v| v.id).collect();
        let genfn = move || {
            let mut rand_order = net_order.clone();
            rand_order.shuffle(&mut rand::thread_rng());
            RouteState(rand_order)
        };

        let evolver = Evolver::new(self.clone(), cfg, genfn);
        let mut trainer = Trainer::new(
            TrainerCfg::new("memeroute").set_termination(Termination::FixedGenerations(1)),
        );
        let order = trainer.train(evolver, &EmptyDataSampler {})?.nth(0).state.0.clone();
        self.route(order)
    }
}

#[must_use]
#[derive(Debug, Display, Deref, DerefMut, Hash, Clone, PartialEq, Eq, PartialOrd)]
#[display(fmt = "{_0:?}")]
pub struct RouteState(pub Vec<Id>);

impl Evaluator for Router {
    type State = RouteState;
    const NUM_CROSSOVER: usize = 4;
    const NUM_MUTATION: usize = 4;

    fn crossover(&self, s1: &mut Self::State, s2: &mut Self::State, idx: usize) {
        match idx {
            0 => {} // Do nothing.
            1 => crossover_pmx(s1, s2),
            2 => crossover_order(s1, s2),
            3 => crossover_cycle(s1, s2),
            _ => panic!("unknown crossover strategy"),
        };
    }

    fn mutate(&self, s: &mut Self::State, rate: f64, idx: usize) {
        let mut r = rand::thread_rng();
        if r.gen::<f64>() > rate {
            return;
        }
        match idx {
            0 => mutate_swap(s),
            1 => mutate_insert(s),
            2 => mutate_scramble(s),
            3 => mutate_inversion(s),
            _ => panic!("unknown mutation strategy"),
        }
    }

    fn fitness(&self, s: &Self::State, _data: &Self::Data) -> Result<f64> {
        let res = self.route(s.0.clone()).unwrap();
        let mut cost = 0.0;
        if res.failed {
            cost += 1000.0;
        }
        cost += res.vias.len() as f64 * 10.0;
        // TODO: Count wire lengths
        Ok(1.0 / (1.0 + cost))
    }

    fn distance(&self, s1: &Self::State, s2: &Self::State) -> Result<f64> {
        Ok(kendall_tau(s1, s2)? as f64)
    }
}

pub fn apply_route_result(pcb: &mut Pcb, r: &RouteResult) {
    for wire in &r.wires {
        pcb.add_wire(wire.clone());
    }
    for via in &r.vias {
        pcb.add_via(via.clone());
    }
    for rt in &r.debug_rts {
        pcb.add_debug_rt(*rt);
    }
}
