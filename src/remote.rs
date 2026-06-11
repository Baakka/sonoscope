//! Phone-as-extra-microphone service.
//!
//! Serves a single-page mic streamer over HTTPS (phone browsers require a
//! secure context for getUserMedia; the certificate is self-signed and
//! generated at startup — the phone shows a one-time warning to accept).
//! Any number of phones can connect at once: each WebSocket connection gets
//! its own `RemoteClient` (ring buffer, level meter, name) in a shared
//! `Registry`. Every stream is resampled to the Mac stream rate; the
//! analysis layer fuses all mics' spectra for better data.

use std::collections::VecDeque;
#[cfg(not(target_arch = "wasm32"))]
use std::net::{IpAddr, SocketAddr, UdpSocket};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(not(target_arch = "wasm32"))]
use axum::Router;
#[cfg(not(target_arch = "wasm32"))]
use axum::extract::State;
#[cfg(not(target_arch = "wasm32"))]
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
#[cfg(not(target_arch = "wasm32"))]
use axum::response::{Html, IntoResponse};
#[cfg(not(target_arch = "wasm32"))]
use axum::routing::get;

#[cfg(not(target_arch = "wasm32"))]
const PORT: u16 = 7777;
/// Per-client ring: a couple of analysis windows' worth.
pub const REMOTE_RING_LEN: usize = 16384;

#[cfg(not(target_arch = "wasm32"))]
const PHONE_PAGE: &str = include_str!("phone.html");

/// Ring with an absolute sample counter: `total` counts every resampled
/// sample ever written, so stream positions stay comparable across
/// WebSocket burst jitter (content never moves in `total` coordinates).
#[derive(Default)]
pub struct RemoteRing {
    pub buf: VecDeque<f32>,
    pub total: u64,
}

/// One connected phone.
pub struct RemoteClient {
    pub id: u64,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    name: Mutex<String>,
    /// Phone samples resampled to the local stream rate.
    pub ring: Mutex<RemoteRing>,
    level_bits: AtomicU32,
}

impl RemoteClient {
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn name(&self) -> String {
        self.name.lock().unwrap().clone()
    }

    pub fn level(&self) -> f32 {
        f32::from_bits(self.level_bits.load(Ordering::Relaxed))
    }

    /// Consistent copy of the whole ring plus its absolute counter.
    pub fn snapshot(&self) -> (Vec<f32>, u64) {
        let ring = self.ring.lock().unwrap();
        (ring.buf.iter().copied().collect(), ring.total)
    }
}

/// Shared list of connected phones. Cloning shares the same list.
#[derive(Clone, Default)]
pub struct Registry {
    clients: Arc<Mutex<Vec<Arc<RemoteClient>>>>,
    next_id: Arc<AtomicU64>,
}

impl Registry {
    // Connections only exist on native; wasm builds keep the (empty)
    // registry so the analysis/UI layers stay identical.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    fn add(&self, name: String) -> Arc<RemoteClient> {
        let client = Arc::new(RemoteClient {
            id: self.next_id.fetch_add(1, Ordering::Relaxed),
            name: Mutex::new(name),
            ring: Mutex::new(RemoteRing {
                buf: VecDeque::with_capacity(REMOTE_RING_LEN + 4096),
                total: 0,
            }),
            level_bits: AtomicU32::new(0),
        });
        self.clients.lock().unwrap().push(Arc::clone(&client));
        client
    }

    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    fn remove(&self, id: u64) {
        self.clients.lock().unwrap().retain(|c| c.id != id);
    }

    /// Snapshot of the connected clients.
    pub fn clients(&self) -> Vec<Arc<RemoteClient>> {
        self.clients.lock().unwrap().clone()
    }
}

pub struct RemoteMics {
    pub registry: Registry,
    /// Address phones should open, or the reason there isn't one.
    /// Only displayed natively (the wasm build hides the feature entirely).
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub url: Result<String, String>,
}

/// Phones can't join the static web build: a page has no server to accept
/// their WebSocket streams. The empty registry keeps everything else working.
#[cfg(target_arch = "wasm32")]
pub fn start(_target_rate: f32) -> RemoteMics {
    RemoteMics {
        registry: Registry::default(),
        url: Err("not available in the browser build — use the native app".to_string()),
    }
}

#[derive(Clone)]
#[cfg(not(target_arch = "wasm32"))]
struct Shared {
    registry: Registry,
    target_rate: f32,
}

/// Start the HTTPS/WebSocket service on a background thread.
/// Never fails the app: on error the returned `url` carries the message.
#[cfg(not(target_arch = "wasm32"))]
pub fn start(target_rate: f32) -> RemoteMics {
    let registry = Registry::default();

    let url = match lan_ip() {
        Some(ip) => Ok(format!("https://{ip}:{PORT}")),
        None => Err("no LAN address found".to_string()),
    };

    if let Ok(url_str) = &url {
        let shared = Shared {
            registry: registry.clone(),
            target_rate,
        };
        let url_str = url_str.clone();
        std::thread::Builder::new()
            .name("phone-mic-server".into())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("tokio runtime");
                if let Err(e) = rt.block_on(serve(shared)) {
                    eprintln!("phone mic service failed ({url_str}): {e}");
                }
            })
            .expect("spawn phone-mic thread");
    }

    RemoteMics { registry, url }
}

#[cfg(not(target_arch = "wasm32"))]
async fn serve(shared: Shared) -> Result<(), Box<dyn std::error::Error>> {
    let cert = rcgen::generate_simple_self_signed(vec![
        "localhost".to_string(),
        lan_ip().map_or_else(|| "localhost".into(), |ip| ip.to_string()),
    ])?;
    let tls = axum_server::tls_rustls::RustlsConfig::from_der(
        vec![cert.cert.der().to_vec()],
        cert.key_pair.serialize_der(),
    )
    .await?;

    let app = Router::new()
        .route("/", get(|| async { Html(PHONE_PAGE) }))
        .route("/ws", get(ws_handler))
        .with_state(shared);

    let addr = SocketAddr::from(([0, 0, 0, 0], PORT));
    axum_server::bind_rustls(addr, tls)
        .serve(app.into_make_service())
        .await?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
async fn ws_handler(ws: WebSocketUpgrade, State(shared): State<Shared>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_stream(socket, shared))
}

#[cfg(not(target_arch = "wasm32"))]
async fn handle_stream(mut socket: WebSocket, shared: Shared) {
    let client = shared.registry.add("phone".to_string());
    // Assume the common case until the page reports its true rate.
    let mut resampler = Resampler::new(48000.0, shared.target_rate);
    let mut out = Vec::new();

    while let Some(Ok(msg)) = socket.recv().await {
        match msg {
            Message::Text(text) => {
                // First frame: {"sr": 44100, "name": "iPhone"}
                if let Some(sr) = parse_sr(&text) {
                    resampler = Resampler::new(sr, shared.target_rate);
                }
                if let Some(name) = parse_name(&text) {
                    *client.name.lock().unwrap() = name;
                }
            }
            Message::Binary(bytes) => {
                let samples: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                let rms = (samples.iter().map(|s| s * s).sum::<f32>()
                    / samples.len().max(1) as f32)
                    .sqrt();
                client.level_bits.store(rms.to_bits(), Ordering::Relaxed);

                out.clear();
                resampler.process(&samples, &mut out);
                let mut ring = client.ring.lock().unwrap();
                ring.buf.extend(out.iter().copied());
                ring.total += out.len() as u64;
                while ring.buf.len() > REMOTE_RING_LEN {
                    ring.buf.pop_front();
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    shared.registry.remove(client.id);
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_sr(text: &str) -> Option<f32> {
    // Tiny tolerant parse of {"sr": <number>} — not worth a JSON dependency.
    let idx = text.find("\"sr\"")?;
    let rest = &text[idx + 4..];
    let num: String = rest
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    let sr: f32 = num.parse().ok()?;
    (8000.0..=192000.0).contains(&sr).then_some(sr)
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_name(text: &str) -> Option<String> {
    // Tolerant parse of "name": "<string>" — value is the next quoted run.
    let idx = text.find("\"name\"")?;
    let rest = &text[idx + 6..];
    let start = rest.find('"')? + 1;
    let end = start + rest[start..].find('"')?;
    let name: String = rest[start..end].trim().chars().take(20).collect();
    (!name.is_empty()).then_some(name)
}

/// The LAN address peers can reach us on (no traffic is actually sent).
#[cfg(not(target_arch = "wasm32"))]
fn lan_ip() -> Option<IpAddr> {
    let sock = UdpSocket::bind("0.0.0.0:0").ok()?;
    sock.connect("8.8.8.8:80").ok()?;
    sock.local_addr().ok().map(|a| a.ip())
}

/// Streaming linear resampler. `step` input samples are consumed per output
/// sample; clock drift between the two devices shows up as a slightly wrong
/// `step`, which only accumulates as buffer growth/shrinkage that the ring
/// cap absorbs.
#[cfg(not(target_arch = "wasm32"))]
pub struct Resampler {
    step: f64,
    /// Position in the virtual stream [last, input...], in input samples.
    pos: f64,
    last: f32,
    primed: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl Resampler {
    pub fn new(in_rate: f32, out_rate: f32) -> Self {
        Self {
            step: in_rate as f64 / out_rate as f64,
            pos: 0.0,
            last: 0.0,
            primed: false,
        }
    }

    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        let n = input.len();
        if n == 0 {
            return;
        }
        if !self.primed {
            self.last = input[0];
            self.primed = true;
        }
        // Virtual stream v[0] = last block's final sample, v[k] = input[k-1].
        let v = |k: usize| if k == 0 { self.last } else { input[k - 1] };
        while (self.pos.floor() as usize) < n {
            let k = self.pos.floor() as usize;
            let frac = (self.pos - k as f64) as f32;
            out.push(v(k) * (1.0 - frac) + v(k + 1) * frac);
            self.pos += self.step;
        }
        self.pos -= n as f64;
        self.last = input[n - 1];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, sr: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect()
    }

    #[test]
    fn resampler_rate_conversion() {
        let mut rs = Resampler::new(44100.0, 48000.0);
        let input = sine(100.0, 44100.0, 44100); // 1 second
        let mut out = Vec::new();
        // Feed in audio-callback-sized chunks.
        for chunk in input.chunks(2048) {
            rs.process(chunk, &mut out);
        }
        let expected = 48000.0 * (input.len() as f64 / 44100.0);
        assert!(
            (out.len() as f64 - expected).abs() < 4.0,
            "got {} samples, expected ~{expected}",
            out.len()
        );
        // Continuity: a 100 Hz sine at 48 kHz moves at most ~0.0131/sample.
        let max_step = out
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(max_step < 0.02, "discontinuity: {max_step}");
    }

    #[test]
    fn resampler_bounded_under_drift() {
        // Device clock 0.1% fast: position offset must stay bounded per
        // block (the ring cap handles the long-term rate mismatch).
        let mut rs = Resampler::new(48048.0, 48000.0);
        let mut out = Vec::new();
        for _ in 0..500 {
            // ~10 s in 1024-sample blocks
            rs.process(&[0.1f32; 1024], &mut out);
            assert!(rs.pos >= 0.0 && rs.pos < 2.0, "pos drifted: {}", rs.pos);
        }
        let expected = 500.0 * 1024.0 * (48000.0 / 48048.0);
        assert!((out.len() as f64 - expected).abs() < 8.0);
    }

    #[test]
    fn parses_sample_rate_message() {
        assert_eq!(parse_sr(r#"{"sr": 44100}"#), Some(44100.0));
        assert_eq!(parse_sr(r#"{"sr":48000.0}"#), Some(48000.0));
        assert_eq!(parse_sr("nonsense"), None);
    }

    #[test]
    fn parses_name_message() {
        assert_eq!(
            parse_name(r#"{"sr": 44100, "name": "iPhone 15"}"#),
            Some("iPhone 15".to_string())
        );
        assert_eq!(parse_name(r#"{"name":""}"#), None);
        assert_eq!(parse_name(r#"{"sr": 44100}"#), None);
    }

    #[test]
    fn registry_tracks_multiple_clients() {
        let reg = Registry::default();
        let a = reg.add("iPhone".into());
        let b = reg.add("Pixel".into());
        assert_eq!(reg.clients().len(), 2);
        assert_ne!(a.id, b.id);

        {
            let mut ring = a.ring.lock().unwrap();
            ring.buf.extend([0.5f32; 100]);
            ring.total += 100;
        }
        let (snap, total) = a.snapshot();
        assert_eq!((snap.len(), total), (100, 100));
        assert_eq!(b.snapshot().0.len(), 0);

        reg.remove(a.id);
        let left = reg.clients();
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].name(), "Pixel");
    }
}
