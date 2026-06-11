#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod audio;
mod dsp;
mod gui;
mod pitch;
mod remote;
mod tuning;

#[cfg(not(target_arch = "wasm32"))]
use eframe::egui;

/// Ignore input quieter than this (noise gate). Set low for built-in
/// laptop mics, which are quiet; the detector itself rejects unpitched noise.
pub const RMS_GATE: f32 = 0.004;
/// Frames the display holds after the note decays below the gate.
pub const HOLD_FRAMES: u32 = 12;
/// Deviation considered "in tune", in cents.
pub const IN_TUNE_CENTS: f32 = 3.0;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = audio::start()?;
    let remote = remote::start(engine.sample_rate);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 920.0])
            .with_min_inner_size([1000.0, 700.0])
            .with_title("Flamenco Tuner Pro"),
        ..Default::default()
    };
    eframe::run_native(
        "Flamenco Tuner Pro",
        options,
        Box::new(move |_cc| Ok(Box::new(gui::TunerApp::new(engine, remote)))),
    )
    .map_err(|e| format!("failed to start UI: {e}"))?;
    Ok(())
}

/// Web entry point: render into the page's canvas. The audio engine is
/// created up front (its async getUserMedia wiring finishes in the
/// background) so the dashboard appears immediately.
#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast as _;

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .expect("no window")
            .document()
            .expect("no document");
        let canvas = document
            .get_element_by_id("tuner_canvas")
            .expect("missing #tuner_canvas in index.html")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("#tuner_canvas is not a canvas");

        let result = eframe::WebRunner::new()
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(|_cc| {
                    let engine = audio::start().expect("Web Audio unavailable");
                    let remote = remote::start(engine.sample_rate);
                    Ok(Box::new(gui::TunerApp::new(engine, remote)))
                }),
            )
            .await;
        if let Err(e) = result {
            web_sys::console::error_1(&format!("failed to start UI: {e:?}").into());
        }
    });
}
