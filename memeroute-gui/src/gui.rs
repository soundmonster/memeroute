use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use eframe::egui;
use eframe::egui::Widget;
use memeroute::dsn::pcb_to_session::PcbToSession;
use memeroute::model::pcb::Pcb;
use memeroute::route::router::{apply_route_result, Router};
use serde::{Deserialize, Serialize};

use crate::pcb::pcb_view::PcbView;

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default)]
struct State {
    filename: String,
}

impl Default for State {
    fn default() -> Self {
        Self { filename: "data/left.dsn".to_string() }
    }
}

#[derive(Debug, Clone)]
pub struct MemerouteGui {
    s: State,
    pcb: Arc<Mutex<Pcb>>,
    pcb_view: Arc<Mutex<PcbView>>,
    data_path: PathBuf,
}

impl MemerouteGui {
    pub fn new<P: AsRef<Path>>(
        pcb_input: Pcb,
        data_path: P,
        cc: &eframe::CreationContext<'_>,
    ) -> Self {
        let s: State = if let Some(storage) = cc.storage {
            eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        } else {
            State::default()
        };
        let bounds = pcb_input.bounds().clone();
        let pcb = Arc::new(Mutex::new(pcb_input));
        let pcb_view = Arc::new(Mutex::new(PcbView::new(pcb.clone(), bounds)));
        Self { s, pcb, pcb_view, data_path: data_path.as_ref().into() }
    }
}

impl eframe::App for MemerouteGui {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.s);
    }

    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Quit").clicked() {
                        frame.quit();
                    }
                });
            });
        });

        egui::SidePanel::left("side_panel").show(ctx, |ui| {
            ui.heading("Side Panel");

            if ui.button("Route").clicked() {
                let pcb = self.pcb.clone();
                let output_path = self.data_path.with_extension("ses");
                let pcb_view = self.pcb_view.clone();
                let _handle = thread::spawn(move || {
                    let router = Router::new(pcb.clone());
                    let start = Instant::now();
                    let resp = router.route(router.rand_net_order()).unwrap();
                    // let resp = router.run_ga().unwrap();
                    println!(
                        "Route result succeeded: {}, {} wires {} vias, time: {:?}",
                        !resp.failed,
                        resp.wires.len(),
                        resp.vias.len(),
                        Instant::now().duration_since(start)
                    );
                    apply_route_result(&mut pcb.clone().lock().unwrap(), &resp);

                    let ses = PcbToSession::new(pcb.lock().unwrap().clone()).convert().unwrap();
                    std::fs::write(output_path, ses).unwrap();
                    // Update pcb view.
                    pcb_view.lock().unwrap().set_pcb(pcb.clone());
                });
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.pcb_view.lock().unwrap().ui(ui);
        });
    }
}
