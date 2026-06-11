#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod gui;
mod pitch;
mod tuning;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;

/// Ignore input quieter than this (noise gate). Set low for built-in
/// laptop mics, which are quiet; YIN itself rejects unpitched noise.
pub const RMS_GATE: f32 = 0.004;
/// Frames the display holds after the note decays below the gate.
pub const HOLD_FRAMES: u32 = 12;
/// Deviation considered "in tune", in cents.
pub const IN_TUNE_CENTS: f32 = 3.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("no input device found — check microphone permissions")?;
    let config = device.default_input_config()?;
    let sample_rate = config.sample_rate().0 as f32;
    let channels = config.channels() as usize;

    let buffer: Arc<Mutex<VecDeque<f32>>> =
        Arc::new(Mutex::new(VecDeque::with_capacity(pitch::FFT_LEN * 2)));
    let writer = Arc::clone(&buffer);

    let err_fn = |e| eprintln!("audio stream error: {e}");
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &_| push_samples(&writer, data, channels),
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config.into(),
            move |data: &[i16], _: &_| {
                let floats: Vec<f32> = data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                push_samples(&writer, &floats, channels);
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::U16 => device.build_input_stream(
            &config.into(),
            move |data: &[u16], _: &_| {
                let floats: Vec<f32> = data
                    .iter()
                    .map(|&s| (s as f32 - 32768.0) / 32768.0)
                    .collect();
                push_samples(&writer, &floats, channels);
            },
            err_fn,
            None,
        )?,
        other => return Err(format!("unsupported sample format: {other}").into()),
    };
    stream.play()?;

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([560.0, 620.0])
            .with_min_inner_size([480.0, 540.0])
            .with_title("Flamenco Tuner"),
        ..Default::default()
    };
    eframe::run_native(
        "Flamenco Tuner",
        options,
        Box::new(move |_cc| Ok(Box::new(gui::TunerApp::new(buffer, sample_rate, stream)))),
    )
    .map_err(|e| format!("failed to start UI: {e}"))?;
    Ok(())
}

fn push_samples(buffer: &Mutex<VecDeque<f32>>, data: &[f32], channels: usize) {
    let mut buf = buffer.lock().unwrap();
    // Downmix interleaved channels to mono.
    for frame in data.chunks(channels) {
        let mono = frame.iter().sum::<f32>() / channels as f32;
        buf.push_back(mono);
    }
    while buf.len() > pitch::FFT_LEN {
        buf.pop_front();
    }
}
