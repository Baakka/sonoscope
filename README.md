# Flamenco Tuner Pro

A professional-grade audio analysis tool for flamenco guitar and voice,
written in Rust (`egui` + `cpal` + `rustfft`). A single dashboard combines a
sub-cent tuner with spectral analytics.

## Run

### Native (full feature set)

```sh
cargo run --release
```

macOS will ask for microphone access for your terminal on first run.

### Web (static page)

The same dashboard compiles to WebAssembly and runs entirely in the browser
— no backend, hostable on any static host (GitHub Pages, S3, nginx, …):

```sh
rustup target add wasm32-unknown-unknown
cargo install trunk        # or: brew install trunk

trunk serve                # develop at http://127.0.0.1:8080
trunk build --release      # emits the deployable site into dist/
```

Upload the contents of `dist/` anywhere. The page must be served over
**HTTPS** (or localhost) — browsers require a secure context for
microphone access. Mic capture uses Web Audio (getUserMedia +
AudioWorklet) with all browser processing disabled; audio starts after the
first click/tap (autoplay policy).

The layout is adaptive: below ~700 px the dashboard stacks into a single
column, so it works on phones too.

Web-build differences: the phone-as-extra-mic feature (and with it
beamforming and the vector scope) is compiled out — a static page can't
host the WebSocket server the phones stream to. Everything else — tuner,
spectrum, peak tracking, CQT spectrogram, chroma/chords, visibility
metrics — is identical.

## Dashboard

- **Tuner strip** — note readout, ±50¢ bar with in-tune zone, string badges
  (guitar) or vibrato analytics (voice: rate, depth, center offset).
- **Spectrum + peak tracking** — live FFT with peaks tracked across frames
  (adaptive threshold, parabolic interpolation, EMA-smoothed locations);
  the strongest tracks are labeled and tabulated with note names and cents.
- **CQT spectrogram** — scrolling variable-Q heatmap, C2–C7, one row per
  semitone: log-frequency bins aligned with musical pitch.
- **Chromagram** — 12 pitch-class energies folded from the CQT, with
  template-matched chord recognition (24 major/minor triads) and a running
  Krumhansl-Schmuckler key estimate.
- **Vector scope** — Lissajous/goniometer view with a phase correlation
  meter, fed by **your phone as a second microphone** (see below) or a
  stereo input device.
- **Texture metrics** — horizontal visibility graph over the level envelope
  (mean degree, density, degree histogram): a network-topology complexity
  signature — sustained notes read low, rasgueado reads high.
- **Pitch history & waveform** — 12 s deviation trace and oscilloscope.

## Pitch detection

Resonance-robust hybrid: YIN candidate dips → spectral harmonic scoring
(so a sympathetically ringing string can't steal the lock) → multi-harmonic
parabolic refinement (sub-cent) → median stabilizer. Guitar mode searches
55–600 Hz with string matching (standard + rondeña tunings, cejilla 0–9);
voice mode searches 70–1100 Hz with chromatic note matching.

## Input filters

High-pass (20–400 Hz) and low-pass (500 Hz–20 kHz) RBJ Butterworth biquads
applied in the audio callback — every analyzer sees the filtered signal.

## Phone as second microphone

The app serves `https://<your-lan-ip>:7777` (QR shown in the Scope panel
while disconnected). On the phone: scan, accept the self-signed certificate
warning, tap *Start streaming*, allow the mic. The phone's audio streams
over WebSocket, is resampled to the local rate, and pairs with the Mac mic
in the vector scope. Both clocks free-run, so phase is indicative rather
than sample-exact.

## Keyboard

| Key       | Action                          |
|-----------|---------------------------------|
| `t`       | Toggle Standard ↔ Rondeña (guitar mode) |
| `↑` / `↓` | Cejilla fret up / down (0–9)    |

## Development

```sh
cargo test --release                                  # 29 DSP/unit tests
cargo test --release report -- --ignored --nocapture  # detector accuracy table
cargo run --release --bin miccheck                    # input device diagnostic
```

Module map: `audio.rs` (capture + filter chain; cpal backend on native,
Web Audio backend on wasm behind the same API), `pitch.rs` (detector),
`dsp/` (filter, peaks, cqt, chroma, visibility, vibrato), `remote.rs`
(phone-mic HTTPS/WebSocket service), `gui/` (dashboard panels),
`tuning.rs` (note math).
