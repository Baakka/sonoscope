//! Vector scope (Lissajous / goniometer) + phase correlation meter, fed by
//! Mac mic + phone mic (or a true stereo input device). Custom-painted into
//! a fixed card; shows the phone connection QR while no second channel
//! exists.

use eframe::egui::{
    self, Align2, Color32, ColorImage, CornerRadius, FontId, Pos2, Rect, Stroke, TextureOptions,
    Vec2, pos2, vec2,
};

use super::draw::{self, Scale};
use super::{DIM, FLAMENCO_GOLD, FLAMENCO_RED, GREEN, TEXT, TRACK, TunerApp};

pub fn section(ui: &mut egui::Ui, app: &mut TunerApp, size: Vec2) {
    let pairs = app.stereo_pairs();
    let card = draw::card(ui, size, "VECTOR SCOPE · L = mac, R = loudest phone");

    match pairs {
        Some(pairs) => lissajous(&card, &pairs),
        None => qr_card(&card, app, ui.ctx()),
    }
}

fn lissajous(card: &draw::Card, pairs: &[(f32, f32)]) {
    let p = &card.painter;
    // Square plot area on the left, meter on the right.
    let side = (card.rect.height() - 36.0).min(card.rect.width() * 0.55);
    let plot = Rect::from_min_size(card.rect.min, vec2(side, side));
    let s = Scale::new(plot, (-0.55, 0.55), (-0.55, 0.55));

    // Frame + crosshair.
    p.rect_stroke(
        plot,
        CornerRadius::same(4),
        Stroke::new(1.0, draw::GRID),
        egui::StrokeKind::Inside,
    );
    p.line_segment(
        [s.pos(0.0, -0.55), s.pos(0.0, 0.55)],
        Stroke::new(1.0, draw::GRID),
    );
    p.line_segment(
        [s.pos(-0.55, 0.0), s.pos(0.55, 0.0)],
        Stroke::new(1.0, draw::GRID),
    );
    p.text(
        pos2(plot.center().x, plot.top() + 4.0),
        Align2::CENTER_TOP,
        "M",
        FontId::proportional(9.0),
        DIM,
    );
    p.text(
        pos2(plot.right() - 6.0, plot.center().y - 8.0),
        Align2::RIGHT_CENTER,
        "S",
        FontId::proportional(9.0),
        DIM,
    );

    // Mid/side rotation: mono content reads as a vertical line.
    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
    let trace: Vec<Pos2> = pairs
        .iter()
        .map(|&(l, r)| {
            let x = ((l - r) * inv_sqrt2).clamp(-0.55, 0.55);
            let y = ((l + r) * inv_sqrt2).clamp(-0.55, 0.55);
            s.pos(x, y)
        })
        .collect();
    draw::polyline(
        p,
        trace,
        Stroke::new(0.8, FLAMENCO_GOLD.linear_multiply(0.7)),
    );

    // Correlation meter to the right of the square.
    let r = pearson(pairs);
    let meter_left = plot.right() + 18.0;
    let track = Rect::from_min_max(
        pos2(meter_left, plot.center().y - 5.0),
        pos2(card.rect.right() - 24.0, plot.center().y + 5.0),
    );
    p.text(
        pos2(track.center().x, track.top() - 16.0),
        Align2::CENTER_BOTTOM,
        format!("phase correlation {r:+.2}"),
        FontId::proportional(11.0),
        DIM,
    );
    p.rect_filled(track, CornerRadius::same(5), TRACK);
    let x = track.left() + (r.clamp(-1.0, 1.0) + 1.0) / 2.0 * track.width();
    let color = if r > 0.4 {
        GREEN
    } else if r > -0.2 {
        FLAMENCO_GOLD
    } else {
        FLAMENCO_RED
    };
    p.rect_filled(
        Rect::from_center_size(pos2(x, track.center().y), vec2(4.0, 18.0)),
        CornerRadius::same(2),
        color,
    );
    p.text(
        pos2(track.left() - 4.0, track.center().y),
        Align2::RIGHT_CENTER,
        "−1",
        FontId::proportional(9.0),
        DIM,
    );
    p.text(
        pos2(track.right() + 4.0, track.center().y),
        Align2::LEFT_CENTER,
        "+1",
        FontId::proportional(9.0),
        DIM,
    );
    p.text(
        pos2(track.center().x, track.bottom() + 14.0),
        Align2::CENTER_CENTER,
        "two free-running mics — indicative phase",
        FontId::proportional(9.0),
        DIM,
    );
}

fn pearson(pairs: &[(f32, f32)]) -> f32 {
    let n = pairs.len() as f32;
    if n < 2.0 {
        return 0.0;
    }
    let (ml, mr) = pairs
        .iter()
        .fold((0.0, 0.0), |(a, b), &(l, r)| (a + l, b + r));
    let (ml, mr) = (ml / n, mr / n);
    let (mut num, mut dl, mut dr) = (0.0f32, 0.0f32, 0.0f32);
    for &(l, r) in pairs {
        let (a, b) = (l - ml, r - mr);
        num += a * b;
        dl += a * a;
        dr += b * b;
    }
    if dl <= 1e-12 || dr <= 1e-12 {
        0.0
    } else {
        num / (dl * dr).sqrt()
    }
}

/// QR + URL card painted while no second channel exists.
fn qr_card(card: &draw::Card, app: &mut TunerApp, ctx: &egui::Context) {
    let p = &card.painter;
    match app.remote.url.clone() {
        Ok(url) => {
            let texture = app.qr_texture.get_or_insert_with(|| {
                ctx.load_texture("phone-qr", qr_image(&url), TextureOptions::NEAREST)
            });
            let side = (card.rect.height() - 8.0).min(150.0);
            let qr_rect = Rect::from_min_size(card.rect.min, vec2(side, side));
            p.image(
                texture.id(),
                qr_rect,
                Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                Color32::WHITE,
            );
            let text_x = qr_rect.right() + 16.0;
            p.text(
                pos2(text_x, card.rect.top() + 14.0),
                Align2::LEFT_TOP,
                "Add phones as extra mics (several can join)",
                FontId::proportional(14.0),
                TEXT,
            );
            p.text(
                pos2(text_x, card.rect.top() + 36.0),
                Align2::LEFT_TOP,
                &url,
                FontId::monospace(12.0),
                FLAMENCO_GOLD,
            );
            p.text(
                pos2(text_x, card.rect.top() + 58.0),
                Align2::LEFT_TOP,
                "1. Scan the QR (same Wi-Fi)\n2. Accept the certificate warning\n3. Tap Start streaming and allow the mic",
                FontId::proportional(11.0),
                DIM,
            );
        }
        Err(e) => {
            p.text(
                card.rect.min,
                Align2::LEFT_TOP,
                format!("phone mic unavailable: {e}"),
                FontId::proportional(12.0),
                DIM,
            );
        }
    }
}

fn qr_image(url: &str) -> ColorImage {
    let code = qrcode::QrCode::new(url.as_bytes()).expect("QR encode");
    let w = code.width();
    let quiet = 2;
    let size = w + quiet * 2;
    let mut pixels = vec![Color32::WHITE; size * size];
    for (i, color) in code.to_colors().iter().enumerate() {
        if *color == qrcode::Color::Dark {
            let (x, y) = (i % w + quiet, i / w + quiet);
            pixels[y * size + x] = Color32::BLACK;
        }
    }
    ColorImage {
        size: [size, size],
        source_size: vec2(size as f32, size as f32),
        pixels,
    }
}
