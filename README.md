# Flamenco Tuner

A native desktop tuner and audio analyzer for flamenco guitar, written in
Rust with `egui`. Listens on the default microphone, detects pitch with the
YIN algorithm, and shows a modern dashboard:

- **Tuner** — big note readout, a ±50-cent bar with a highlighted in-tune
  zone, and tune up/down guidance. Green within ±3¢, amber to ±15¢, red
  beyond.
- **Spectrum** — live FFT (Hann window, fast-attack/slow-decay averaging)
  from 0–1.5 kHz, with markers for the detected pitch (red) and the target
  pitch (green dashed) so you can see the fundamental and harmonics.
- **Pitch history** — the last 12 seconds of deviation in cents, with the
  in-tune band marked. Useful for watching a string settle after a bend or
  a cejilla change.
- **Waveform** — oscilloscope view of the analysis window.
- **Strings panel** — all six strings with capo-shifted targets; the one
  you're plucking lights up in the accuracy color.

## Flamenco specifics

- **Standard tuning** E A D G B E and **Rondeña** scordatura D A D F# B E
  (the one true altered tuning of the repertoire — *por medio* and
  *por arriba* are positions, not tunings).
- **Cejilla (capo) support** — set the fret (0–9) and all string targets
  shift so you can tune with the cejilla on.
- Adjustable A4 reference, 415–466 Hz (default 440), for matching a
  cantaor or an old recording.

## Run

```sh
cargo run --release
```

On macOS, grant the terminal microphone access when prompted (System
Settings → Privacy & Security → Microphone).

## Controls

Everything is clickable in the top bar; keyboard shortcuts:

| Key       | Action                       |
|-----------|------------------------------|
| `t`       | Toggle Standard ↔ Rondeña    |
| `↑` / `↓` | Cejilla fret up / down (0–9) |

## Mic diagnostic

If the level meter stays at zero:

```sh
cargo run --release --bin miccheck
```

prints the available input devices and three seconds of peak levels.
