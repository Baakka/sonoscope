//! Quick microphone diagnostic: prints the input device and 3 seconds of
//! peak levels. All zeros means no signal — usually denied mic permission.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let host = cpal::default_host();

    println!("input devices:");
    for d in host.input_devices()? {
        println!("  - {}", d.name().unwrap_or_else(|_| "?".into()));
    }

    let device = host
        .default_input_device()
        .ok_or("no default input device")?;
    let config = device.default_input_config()?;
    println!(
        "\nusing: {} ({} Hz, {} ch, {:?})",
        device.name()?,
        config.sample_rate().0,
        config.channels(),
        config.sample_format()
    );

    let peak = Arc::new(AtomicU32::new(0));
    let writer = Arc::clone(&peak);
    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &_| {
            let p = data.iter().fold(0.0f32, |m, s| m.max(s.abs()));
            writer.fetch_max(p.to_bits(), Ordering::Relaxed);
        },
        |e| eprintln!("stream error: {e}"),
        None,
    )?;
    stream.play()?;

    println!("\nlistening for 3 seconds — make some noise…");
    for i in 1..=6 {
        std::thread::sleep(Duration::from_millis(500));
        let p = f32::from_bits(peak.swap(0, Ordering::Relaxed));
        let bar = "█".repeat((p * 60.0).min(60.0) as usize);
        println!("  {:>4.1}s  peak {:>8.5}  {}", i as f32 * 0.5, p, bar);
    }
    Ok(())
}
