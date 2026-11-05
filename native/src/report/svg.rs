// report/svg.rs
// Hand-written SVG charts: grouped bars, heat grid, stacked bars, lines, scatter
// Fixed light palette on an opaque surface so files read the same in any viewer

use std::fmt::Write;

pub const SURFACE: &str = "#fcfcfb";
pub const INK: &str = "#0b0b0b";
pub const INK2: &str = "#52514e";
pub const GRID: &str = "#e4e3df";
/// Categorical slots in fixed order.
pub const SERIES: [&str; 8] = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
];
/// Sequential blue ramp, light to dark.
const RAMP: [&str; 7] = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
];
const FONT: &str = "system-ui, -apple-system, 'Segoe UI', sans-serif";

pub fn series(i: usize) -> &'static str {
    SERIES[i % SERIES.len()]
}
fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
/// Ramp color for a value in 0..=1.
pub fn ramp(v: f64) -> &'static str {
    RAMP[((v.clamp(0., 1.) * (RAMP.len() - 1) as f64).round() as usize).min(RAMP.len() - 1)]
}
/// Label color on a ramp fill by its luminance side.
pub fn on_ramp(v: f64) -> &'static str {
    if v >= 0.45 {
        "#ffffff"
    } else {
        INK
    }
}
/// Approximate rendered width of text at a font size.
pub fn text_width(s: &str, size: f64) -> f64 {
    s.chars().count() as f64 * size * 0.56
}
/// Round tick values from 0 to at least `max`, about `n` of them.
pub fn ticks(max: f64, n: usize) -> Vec<f64> {
    let max = if max > 0. { max } else { 1. };
    let raw = max / n as f64;
    let mag = 10f64.powf(raw.log10().floor());
    let step = [1., 2., 2.5, 5., 10.]
        .iter()
        .map(|m| m * mag)
        .find(|s| *s >= raw)
        .unwrap_or(10. * mag);
    let count = (max / step).ceil() as usize;
    (0..=count).map(|i| i as f64 * step).collect()
}
/// Ticks that fit inside a scale already rounded with `nice`.
pub fn axis(max: f64, n: usize) -> Vec<f64> {
    ticks(max, n)
        .into_iter()
        .filter(|t| *t <= max * 1.0001)
        .collect()
}
/// Top tick of an axis for `ticks(max, n)`; data scales to this value.
pub fn nice(max: f64, n: usize) -> f64 {
    ticks(max, n).last().copied().unwrap_or(1.)
}
/// Compact number: 1.2K, 3.4M.
pub fn compact(v: f64) -> String {
    let a = v.abs();
    if a >= 1e6 {
        format!("{:.1}M", v / 1e6)
    } else if a >= 1e4 {
        format!("{:.0}K", v / 1e3)
    } else if a >= 1e3 {
        format!("{:.1}K", v / 1e3)
    } else if (v - v.round()).abs() < 1e-9 {
        format!("{v:.0}")
    } else if a < 1. {
        format!("{v:.3}").trim_end_matches('0').to_string()
    } else {
        format!("{v:.1}")
    }
}

pub struct Svg {
    pub w: f64,
    pub h: f64,
    body: String,
}
impl Svg {
    pub fn new(w: f64, h: f64) -> Self {
        Self {
            w,
            h,
            body: String::new(),
        }
    }
    pub fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, fill: &str) {
        let _ = write!(
            self.body,
            r#"<rect x="{x:.1}" y="{y:.1}" width="{:.1}" height="{:.1}" fill="{fill}"/>"#,
            w.max(0.),
            h.max(0.)
        );
    }
    pub fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, stroke: &str, width: f64) {
        let _ = write!(
            self.body,
            r#"<line x1="{x1:.1}" y1="{y1:.1}" x2="{x2:.1}" y2="{y2:.1}" stroke="{stroke}" stroke-width="{width}"/>"#
        );
    }
    pub fn text(&mut self, x: f64, y: f64, s: &str, size: f64, anchor: &str, fill: &str) {
        let _ = write!(
            self.body,
            r#"<text x="{x:.1}" y="{y:.1}" font-size="{size}" text-anchor="{anchor}" fill="{fill}">{}</text>"#,
            escape(s)
        );
    }
    pub fn text_rotated(&mut self, x: f64, y: f64, s: &str, size: f64, degrees: f64, fill: &str) {
        let _ = write!(
            self.body,
            r#"<text x="{x:.1}" y="{y:.1}" font-size="{size}" text-anchor="start" fill="{fill}" transform="rotate({degrees} {x:.1} {y:.1})">{}</text>"#,
            escape(s)
        );
    }
    pub fn circle(&mut self, x: f64, y: f64, r: f64, fill: &str) {
        let _ = write!(
            self.body,
            r#"<circle cx="{x:.1}" cy="{y:.1}" r="{r}" fill="{fill}" stroke="{SURFACE}" stroke-width="2"/>"#
        );
    }
    pub fn polyline(&mut self, points: &[(f64, f64)], stroke: &str) {
        if let [(x, y)] = points {
            self.circle(*x, *y, 3., stroke);
            return;
        }
        let pts = points
            .iter()
            .map(|(x, y)| format!("{x:.1},{y:.1}"))
            .collect::<Vec<_>>()
            .join(" ");
        let _ = write!(
            self.body,
            r#"<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>"#
        );
    }
    /// Column with a rounded top and a square base.
    pub fn column(&mut self, x: f64, base: f64, w: f64, h: f64, fill: &str) {
        if h <= 0. {
            return;
        }
        let r = 4f64.min(h).min(w / 2.);
        let top = base - h;
        let _ = write!(
            self.body,
            r#"<path d="M{x:.1},{base:.1} L{x:.1},{:.1} Q{x:.1},{top:.1} {:.1},{top:.1} L{:.1},{top:.1} Q{:.1},{top:.1} {:.1},{:.1} L{:.1},{base:.1} Z" fill="{fill}"/>"#,
            top + r,
            x + r,
            x + w - r,
            x + w,
            x + w,
            top + r,
            x + w
        );
    }
    /// Horizontal bar with a rounded right end and a square base.
    pub fn row_bar(&mut self, x: f64, y: f64, w: f64, h: f64, fill: &str, opacity: f64) {
        if w <= 0. {
            return;
        }
        let r = 4f64.min(w).min(h / 2.);
        let end = x + w;
        let _ = write!(
            self.body,
            r#"<path d="M{x:.1},{y:.1} L{:.1},{y:.1} Q{end:.1},{y:.1} {end:.1},{:.1} L{end:.1},{:.1} Q{end:.1},{:.1} {:.1},{:.1} L{x:.1},{:.1} Z" fill="{fill}" fill-opacity="{opacity}"/>"#,
            end - r,
            y + r,
            y + h - r,
            y + h,
            end - r,
            y + h,
            y + h
        );
    }
    pub fn legend(&mut self, x: f64, y: f64, items: &[(&str, String)]) -> f64 {
        let mut cursor = x;
        for (color, label) in items {
            self.rect(cursor, y - 9., 10., 10., color);
            self.text(cursor + 15., y, label, 11., "start", INK2);
            cursor += 15. + text_width(label, 11.) + 16.;
        }
        self.w = self.w.max(cursor + 4.);
        cursor
    }
    pub fn title(&mut self, title: &str, subtitle: &str) {
        self.text(20., 26., title, 15., "start", INK);
        if !subtitle.is_empty() {
            self.text(20., 44., subtitle, 11.5, "start", INK2);
        }
        self.w = self
            .w
            .max(text_width(title, 15.).max(text_width(subtitle, 11.5)) + 40.);
    }
    /// Complete document with an opaque surface and accessible title.
    pub fn finish(self, title: &str) -> String {
        format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0}" height="{h:.0}" viewBox="0 0 {w:.0} {h:.0}" font-family="{FONT}" role="img" aria-label="{t}"><title>{t}</title><rect width="{w:.0}" height="{h:.0}" fill="{SURFACE}"/>{body}</svg>
"#,
            w = self.w,
            h = self.h,
            t = escape(title),
            body = self.body
        )
    }
}
