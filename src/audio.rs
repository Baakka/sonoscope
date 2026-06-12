//! Capture engine: input stream → LP/HP filter chain → ring buffers.
//!
//! Two backends behind the same `AudioEngine` API: cpal on native, Web Audio
//! (getUserMedia + AudioWorklet) on wasm. Filtering happens in the audio
//! callback so every consumer (pitch detection, FFT, waveform, scope) sees
//! the same filtered signal. Filter parameters live behind a version-stamped
//! RwLock; the callback rebuilds its biquads only when the version changes.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use crate::dsp::filter::{BUTTERWORTH_Q, Biquad};

/// Mono analysis ring: 16384 samples ≈ 341 ms at 48 kHz (CQT window).
pub const RING_LEN: usize = 16384;
/// Stereo scope ring (sample pairs). Browser capture is mono-only.
#[cfg(not(target_arch = "wasm32"))]
pub const STEREO_RING_LEN: usize = 4096;

#[derive(Clone, Copy, PartialEq)]
pub struct FilterConfig {
    pub hp_enabled: bool,
    pub hp_cutoff: f32,
    pub lp_enabled: bool,
    pub lp_cutoff: f32,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            hp_enabled: false,
            hp_cutoff: 80.0,
            lp_enabled: false,
            lp_cutoff: 8000.0,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub type StereoRing = Arc<Mutex<VecDeque<(f32, f32)>>>;

/// Mono ring with an absolute sample counter: `total` counts every sample
/// ever ingested, so positions in the stream can be compared against the
/// phone streams' counters for alignment.
#[derive(Default)]
pub struct MonoRing {
    pub buf: VecDeque<f32>,
    pub total: u64,
}

/// Independent filter state per signal path: 0 = mono mix, 1 = L, 2 = R.
struct FilterChain {
    sample_rate: f32,
    version_seen: u64,
    hp: Option<[Biquad; 3]>,
    lp: Option<[Biquad; 3]>,
}

impl FilterChain {
    fn sync(&mut self, cfg: &RwLock<FilterConfig>, version: &AtomicU64) {
        let v = version.load(Ordering::Acquire);
        if v == self.version_seen {
            return;
        }
        self.version_seen = v;
        let cfg = *cfg.read().unwrap();
        self.hp = cfg
            .hp_enabled
            .then(|| [Biquad::highpass(self.sample_rate, cfg.hp_cutoff, BUTTERWORTH_Q); 3]);
        self.lp = cfg
            .lp_enabled
            .then(|| [Biquad::lowpass(self.sample_rate, cfg.lp_cutoff, BUTTERWORTH_Q); 3]);
    }

    #[inline]
    fn process(&mut self, channel: usize, x: f32) -> f32 {
        let mut y = x;
        if let Some(hp) = &mut self.hp {
            y = hp[channel].process(y);
        }
        if let Some(lp) = &mut self.lp {
            y = lp[channel].process(y);
        }
        y
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::{AudioEngine, play_chirp, start};
#[cfg(target_arch = "wasm32")]
pub use web::{AudioEngine, start};

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    pub struct AudioEngine {
        /// Filtered mono samples, newest at the back, capped at RING_LEN.
        pub ring: Arc<Mutex<MonoRing>>,
        /// Filtered (L, R) pairs when the device is stereo; None on mono devices.
        pub stereo_ring: Option<StereoRing>,
        pub sample_rate: f32,
        pub device_name: String,
        filter_cfg: Arc<RwLock<FilterConfig>>,
        filter_version: Arc<AtomicU64>,
        // Keeps the stream alive; dropped with the engine.
        _stream: cpal::Stream,
    }

    impl AudioEngine {
        pub fn filter_config(&self) -> FilterConfig {
            *self.filter_cfg.read().unwrap()
        }

        pub fn set_filter_config(&self, cfg: FilterConfig) {
            let mut current = self.filter_cfg.write().unwrap();
            if *current != cfg {
                *current = cfg;
                self.filter_version.fetch_add(1, Ordering::Release);
            }
        }
    }

    /// Play the alignment calibration chirp on the default output device.
    /// Fire-and-forget: errors are logged, never fatal.
    pub fn play_chirp() {
        std::thread::Builder::new()
            .name("calibration-chirp".into())
            .spawn(|| {
                let run = || -> Result<(), Box<dyn std::error::Error>> {
                    let host = cpal::default_host();
                    let device = host.default_output_device().ok_or("no output device")?;
                    let config = device.default_output_config()?;
                    if config.sample_format() != cpal::SampleFormat::F32 {
                        return Err(format!(
                            "output format {} unsupported",
                            config.sample_format()
                        )
                        .into());
                    }
                    let channels = config.channels() as usize;
                    let samples = crate::dsp::align::chirp(config.sample_rate().0 as f32);
                    let total = samples.len();
                    let mut pos = 0usize;
                    let stream = device.build_output_stream(
                        &config.into(),
                        move |out: &mut [f32], _: &_| {
                            for frame in out.chunks_mut(channels) {
                                let v = if pos < total { samples[pos] } else { 0.0 };
                                pos += 1;
                                frame.fill(v);
                            }
                        },
                        |e| eprintln!("chirp output error: {e}"),
                        None,
                    )?;
                    stream.play()?;
                    std::thread::sleep(std::time::Duration::from_millis(1200));
                    Ok(())
                };
                if let Err(e) = run() {
                    eprintln!("calibration chirp failed: {e}");
                }
            })
            .ok();
    }

    pub fn start() -> Result<AudioEngine, Box<dyn std::error::Error>> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or("no input device found — check microphone permissions")?;
        let config = device.default_input_config()?;
        let sample_rate = config.sample_rate().0 as f32;
        let channels = config.channels() as usize;
        let device_name = format!(
            "{} ({} ch)",
            device.name().unwrap_or_else(|_| "unknown".into()),
            channels
        );

        let ring: Arc<Mutex<MonoRing>> = Arc::new(Mutex::new(MonoRing {
            buf: VecDeque::with_capacity(RING_LEN + 1024),
            total: 0,
        }));
        let stereo_ring: Option<StereoRing> = (channels >= 2)
            .then(|| Arc::new(Mutex::new(VecDeque::with_capacity(STEREO_RING_LEN + 1024))));
        let filter_cfg = Arc::new(RwLock::new(FilterConfig::default()));
        // Start at 1 so the chain (version_seen = 0) syncs on the first callback.
        let filter_version = Arc::new(AtomicU64::new(1));

        let cb_ring = Arc::clone(&ring);
        let cb_stereo = stereo_ring.clone();
        let cb_cfg = Arc::clone(&filter_cfg);
        let cb_version = Arc::clone(&filter_version);
        let mut chain = FilterChain {
            sample_rate,
            version_seen: 0,
            hp: None,
            lp: None,
        };

        let mut ingest = move |data: &[f32]| {
            chain.sync(&cb_cfg, &cb_version);
            let mut mono = cb_ring.lock().unwrap();
            let mut stereo = cb_stereo.as_ref().map(|s| s.lock().unwrap());
            for frame in data.chunks(channels) {
                let mix = frame.iter().sum::<f32>() / channels as f32;
                mono.buf.push_back(chain.process(0, mix));
                mono.total += 1;
                if let Some(stereo) = stereo.as_mut() {
                    let l = chain.process(1, frame[0]);
                    let r = chain.process(2, frame.get(1).copied().unwrap_or(frame[0]));
                    stereo.push_back((l, r));
                    if stereo.len() > STEREO_RING_LEN {
                        stereo.pop_front();
                    }
                }
            }
            while mono.buf.len() > RING_LEN {
                mono.buf.pop_front();
            }
        };

        let err_fn = |e| eprintln!("audio stream error: {e}");
        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config.into(),
                move |data: &[f32], _: &_| ingest(data),
                err_fn,
                None,
            )?,
            cpal::SampleFormat::I16 => device.build_input_stream(
                &config.into(),
                move |data: &[i16], _: &_| {
                    let floats: Vec<f32> =
                        data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                    ingest(&floats);
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
                    ingest(&floats);
                },
                err_fn,
                None,
            )?,
            other => return Err(format!("unsupported sample format: {other}").into()),
        };
        stream.play()?;

        Ok(AudioEngine {
            ring,
            stereo_ring,
            sample_rate,
            device_name,
            filter_cfg,
            filter_version,
            _stream: stream,
        })
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use wasm_bindgen::closure::Closure;
    use wasm_bindgen::{JsCast, JsValue};
    use wasm_bindgen_futures::JsFuture;

    pub struct AudioEngine {
        /// Filtered mono samples, newest at the back, capped at RING_LEN.
        pub ring: Arc<Mutex<MonoRing>>,
        pub sample_rate: f32,
        pub device_name: String,
        filter_cfg: Arc<RwLock<FilterConfig>>,
        filter_version: Arc<AtomicU64>,
        _ctx: web_sys::AudioContext,
    }

    impl AudioEngine {
        pub fn filter_config(&self) -> FilterConfig {
            *self.filter_cfg.read().unwrap()
        }

        pub fn set_filter_config(&self, cfg: FilterConfig) {
            let mut current = self.filter_cfg.write().unwrap();
            if *current != cfg {
                *current = cfg;
                self.filter_version.fetch_add(1, Ordering::Release);
            }
        }
    }

    /// 128-sample render quanta batched to ≥2048 before posting — the same
    /// tap the phone page uses.
    const WORKLET_JS: &str = r#"
class PcmTap extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Float32Array(4096); this.n = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    if (this.n + ch.length > this.buf.length) this.n = 0;
    this.buf.set(ch, this.n);
    this.n += ch.length;
    if (this.n >= 2048) { this.port.postMessage(this.buf.slice(0, this.n)); this.n = 0; }
    return true;
  }
}
registerProcessor('pcm-tap', PcmTap);
"#;

    /// Create the AudioContext synchronously (its sample rate is known
    /// immediately) and finish the async getUserMedia/worklet wiring in the
    /// background. Samples start flowing into the ring once the user grants
    /// the mic and the first gesture resumes the context.
    pub fn start() -> Result<AudioEngine, Box<dyn std::error::Error>> {
        let ctx =
            web_sys::AudioContext::new().map_err(|e| format!("AudioContext unavailable: {e:?}"))?;
        let sample_rate = ctx.sample_rate();

        let ring: Arc<Mutex<MonoRing>> = Arc::new(Mutex::new(MonoRing {
            buf: VecDeque::with_capacity(RING_LEN + 1024),
            total: 0,
        }));
        let filter_cfg = Arc::new(RwLock::new(FilterConfig::default()));
        // Start at 1 so the chain (version_seen = 0) syncs on the first batch.
        let filter_version = Arc::new(AtomicU64::new(1));

        resume_on_gesture(ctx.clone());
        wasm_bindgen_futures::spawn_local(setup(
            ctx.clone(),
            Arc::clone(&ring),
            Arc::clone(&filter_cfg),
            Arc::clone(&filter_version),
            sample_rate,
        ));

        Ok(AudioEngine {
            ring,
            sample_rate,
            device_name: format!("browser mic ({sample_rate:.0} Hz)"),
            filter_cfg,
            filter_version,
            _ctx: ctx,
        })
    }

    /// Autoplay policy: the context stays suspended until a user gesture.
    fn resume_on_gesture(ctx: web_sys::AudioContext) {
        let Some(win) = web_sys::window() else { return };
        let cb = Closure::<dyn FnMut()>::new(move || {
            let _ = ctx.resume();
        });
        for ev in ["pointerdown", "keydown", "touchstart"] {
            let _ = win.add_event_listener_with_callback(ev, cb.as_ref().unchecked_ref());
        }
        cb.forget();
    }

    async fn setup(
        ctx: web_sys::AudioContext,
        ring: Arc<Mutex<MonoRing>>,
        cfg: Arc<RwLock<FilterConfig>>,
        version: Arc<AtomicU64>,
        sample_rate: f32,
    ) {
        if let Err(e) = try_setup(ctx, ring, cfg, version, sample_rate).await {
            web_sys::console::error_1(
                &format!("microphone setup failed: {e:?} — check mic permissions").into(),
            );
        }
    }

    async fn try_setup(
        ctx: web_sys::AudioContext,
        ring: Arc<Mutex<MonoRing>>,
        cfg: Arc<RwLock<FilterConfig>>,
        version: Arc<AtomicU64>,
        sample_rate: f32,
    ) -> Result<(), JsValue> {
        // Load the worklet processor from a same-origin blob URL.
        let parts = js_sys::Array::of1(&JsValue::from_str(WORKLET_JS));
        let opts = web_sys::BlobPropertyBag::new();
        opts.set_type("application/javascript");
        let blob = web_sys::Blob::new_with_str_sequence_and_options(&parts, &opts)?;
        let url = web_sys::Url::create_object_url_with_blob(&blob)?;
        let loaded = JsFuture::from(ctx.audio_worklet()?.add_module(&url)?).await;
        let _ = web_sys::Url::revoke_object_url(&url);
        loaded?;

        // Raw mono mic: every browser DSP stage disabled, as on the phone page.
        let constraints = web_sys::MediaStreamConstraints::new();
        let audio = js_sys::Object::new();
        for key in ["echoCancellation", "noiseSuppression", "autoGainControl"] {
            js_sys::Reflect::set(&audio, &key.into(), &false.into())?;
        }
        js_sys::Reflect::set(&audio, &"channelCount".into(), &1.into())?;
        constraints.set_audio(&audio.into());
        let devices = web_sys::window()
            .ok_or_else(|| JsValue::from_str("no window"))?
            .navigator()
            .media_devices()?;
        let stream: web_sys::MediaStream =
            JsFuture::from(devices.get_user_media_with_constraints(&constraints)?)
                .await?
                .dyn_into()?;

        let source = ctx.create_media_stream_source(&stream)?;
        let node = web_sys::AudioWorkletNode::new(&ctx, "pcm-tap")?;
        // A muted gain keeps the graph pulled without monitoring the mic.
        let mute = ctx.create_gain()?;
        mute.gain().set_value(0.0);
        source.connect_with_audio_node(&node)?;
        node.connect_with_audio_node(&mute)?;
        mute.connect_with_audio_node(&ctx.destination())?;

        let mut chain = FilterChain {
            sample_rate,
            version_seen: 0,
            hp: None,
            lp: None,
        };
        let mut scratch: Vec<f32> = Vec::new();
        let port = node.port()?;
        let onmsg =
            Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |ev: web_sys::MessageEvent| {
                let data = js_sys::Float32Array::new(&ev.data());
                scratch.resize(data.length() as usize, 0.0);
                data.copy_to(&mut scratch);
                chain.sync(&cfg, &version);
                let mut mono = ring.lock().unwrap();
                for &x in &scratch {
                    let y = chain.process(0, x);
                    mono.buf.push_back(y);
                    mono.total += 1;
                }
                while mono.buf.len() > RING_LEN {
                    mono.buf.pop_front();
                }
            });
        port.set_onmessage(Some(onmsg.as_ref().unchecked_ref()));
        onmsg.forget();
        Ok(())
    }
}
