use eframe::egui::Widget;
use eframe::{egui, epi};
use memeroute::model::geom::Rt;
use memeroute::model::pcb::Pcb;
use serde::{Deserialize, Serialize};

use crate::pcb::pcb_view::PcbView;

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default)]
struct State {
    filename: String,
    value: f32,
}

impl Default for State {
    fn default() -> Self {
        Self { filename: "data/left.dsn".to_string(), value: 2.7 }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemerouteGui {
    s: State,
    pcb: Pcb,
    pcb_view: PcbView,
}

impl MemerouteGui {
    pub fn new(pcb: Pcb) -> Self {
        let pcb_view = PcbView::new(pcb.clone(), Rt::new(-20.0, -20.0, 200.0, 200.0));
        Self { s: Default::default(), pcb, pcb_view }
    }
}

impl epi::App for MemerouteGui {
    fn name(&self) -> &str {
        "Memeroute"
    }

    fn setup(
        &mut self,
        _ctx: &egui::CtxRef,
        _frame: &mut epi::Frame<'_>,
        storage: Option<&dyn epi::Storage>,
    ) {
        if let Some(storage) = storage {
            self.s = epi::get_value(storage, epi::APP_KEY).unwrap_or_default();
        }
    }

    fn save(&mut self, storage: &mut dyn epi::Storage) {
        epi::set_value(storage, epi::APP_KEY, &self.s);
    }

    fn update(&mut self, ctx: &egui::CtxRef, frame: &mut epi::Frame<'_>) {
        let State { filename, value, .. } = &mut self.s;

        // Examples of how to create different panels and windows.
        // Pick whichever suits you.
        // Tip: a good default choice is to just keep the `CentralPanel`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:
            egui::menu::bar(ui, |ui| {
                egui::menu::menu(ui, "File", |ui| {
                    if ui.button("Quit").clicked() {
                        frame.quit();
                    }
                });
            });
        });

        egui::SidePanel::left("side_panel").show(ctx, |ui| {
            ui.heading("Side Panel");

            ui.horizontal(|ui| {
                ui.label("Write something: ");
                ui.text_edit_singleline(filename);
            });

            ui.add(egui::Slider::new(value, 0.0..=10.0).text("value"));
            if ui.button("Increment").clicked() {
                *value += 1.0;
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.pcb_view.ui(ui);
        });
    }
}
