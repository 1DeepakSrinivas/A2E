// report/plots.rs
// The seven report plots built from a Report
// Each plot returns None when the data it needs is missing

use super::{column, percentile, svg::*, Group, Report};
use std::collections::BTreeMap;

const CATEGORIES: [&str; 8] = [
    "simple",
    "multiple",
    "irrelevance",
    "missing_param",
    "chain",
    "recovery",
    "robustness",
    "web",
];

pub fn all(r: &Report) -> Vec<(String, String)> {
    [
        ("success_by_model.svg", success(r)),
        ("category_heatmap.svg", heatmap(r)),
        ("failure_classes.svg", failures(r)),
        ("latency.svg", latency(r)),
        ("memory.svg", memory(r)),
        ("resource_timeline.svg", timeline(r)),
        ("tokens_vs_wall.svg", scatter(r)),
    ]
    .into_iter()
    .filter_map(|(name, svg)| svg.map(|s| (name.to_string(), s)))
    .collect()
}
fn row_label(r: &Report, g: &Group) -> String {
    format!("{} · {}", g.model, g.label(r.many_contexts()))
}
fn model_legend(r: &Report) -> Vec<(&'static str, String)> {
    r.models
        .iter()
        .enumerate()
        .map(|(i, m)| (series(i), m.clone()))
        .collect()
}
/// Left axis with gridlines for a vertical chart.
fn y_grid(
    s: &mut Svg,
    left: f64,
    right: f64,
    top: f64,
    height: f64,
    max: f64,
    label: impl Fn(f64) -> String,
) {
    for t in axis(max, 4) {
        let y = top + height - t / max * height;
        s.line(left, y, right, y, GRID, 1.);
        s.text(left - 8., y + 4., &label(t), 11., "end", INK2);
    }
}

/// Grouped bars: success rate per variant, one bar per model, with bootstrap whiskers.
fn success(r: &Report) -> Option<String> {
    let many = r.many_contexts();
    let mut labels = Vec::<String>::new();
    for g in r.groups.iter().filter(|g| g.rate.is_some()) {
        let l = g.label(many);
        if !labels.contains(&l) {
            labels.push(l);
        }
    }
    if labels.is_empty() {
        return None;
    }
    let (left, top, height) = (56., 96., 250.);
    let slot = (r.models.len() as f64 * 28. + 36.).max(90.);
    let width = left + labels.len() as f64 * slot + 24.;
    let mut s = Svg::new(width.max(560.), top + height + 56.);
    s.title("Success rate by variant", "Bars show the share of scored attempts that passed; whiskers are the 95% task-bootstrap interval");
    s.legend(20., 62., &model_legend(r));
    y_grid(&mut s, left, width - 16., top, height, 1., |t| {
        format!("{:.0}%", t * 100.)
    });
    let base = top + height;
    for (c, label) in labels.iter().enumerate() {
        let gx = left + c as f64 * slot + 18.;
        let in_group = r
            .groups
            .iter()
            .filter(|g| g.rate.is_some() && &g.label(many) == label)
            .collect::<Vec<_>>();
        for (k, g) in in_group.iter().enumerate() {
            let x = gx + k as f64 * 28.;
            let rate = g.rate.unwrap_or(0.);
            s.column(x, base, 22., rate * height, series(r.model_index(&g.model)));
            let mut tip = base - rate * height;
            if let Some([lo, hi]) = g.ci {
                let (ylo, yhi) = (base - lo * height, base - hi * height);
                s.line(x + 11., ylo, x + 11., yhi, INK2, 1.);
                s.line(x + 7., ylo, x + 15., ylo, INK2, 1.);
                s.line(x + 7., yhi, x + 15., yhi, INK2, 1.);
                tip = tip.min(yhi);
            }
            s.text(
                x + 11.,
                tip - 5.,
                &format!("{:.0}%", rate * 100.),
                10.,
                "middle",
                INK,
            );
        }
        s.text(
            gx + (in_group.len() as f64 * 28. - 6.) / 2.,
            base + 18.,
            label,
            11.,
            "middle",
            INK2,
        );
    }
    s.line(left, base, width - 16., base, INK2, 1.);
    Some(s.finish("Success rate by variant"))
}

/// Heat grid: success rate for each category and each model, variant, and context.
fn heatmap(r: &Report) -> Option<String> {
    let many = r.many_contexts();
    let mut cells = BTreeMap::<(&str, usize), (usize, usize)>::new();
    for (gi, g) in r.groups.iter().enumerate() {
        for row in g.rows(&r.rows) {
            if row["pending"].as_i64() == Some(1) {
                continue;
            }
            let c = cells
                .entry((row["category"].as_str().unwrap_or(""), gi))
                .or_default();
            c.1 += 1;
            c.0 += usize::from(row["passed"].as_i64() == Some(1));
        }
    }
    let rows = CATEGORIES
        .iter()
        .filter(|c| cells.keys().any(|(k, _)| k == *c))
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return None;
    }
    let (left, top, cw, ch) = (110., 140., 44., 28.);
    let mut s = Svg::new(
        left + r.groups.len() as f64 * cw + 30.,
        top + rows.len() as f64 * ch + 24.,
    );
    s.title(
        "Success rate by category",
        "Each cell is passes over scored attempts; n/a means no scored attempts",
    );
    for (gi, g) in r.groups.iter().enumerate() {
        let x = left + gi as f64 * cw + cw / 2.;
        s.text_rotated(
            x + 4.,
            top - 8.,
            &format!("{} · {}", g.model, g.label(many)),
            11.,
            -45.,
            INK2,
        );
    }
    for (ri, cat) in rows.iter().enumerate() {
        let y = top + ri as f64 * ch;
        s.text(left - 8., y + ch / 2. + 4., cat, 11.5, "end", INK2);
        for gi in 0..r.groups.len() {
            let x = left + gi as f64 * cw;
            match cells.get(&(**cat, gi)) {
                Some((p, n)) if *n > 0 => {
                    let v = *p as f64 / *n as f64;
                    s.rect(x + 1., y + 1., cw - 2., ch - 2., ramp(v));
                    s.text(
                        x + cw / 2.,
                        y + ch / 2. + 4.,
                        &format!("{:.0}%", v * 100.),
                        10.5,
                        "middle",
                        on_ramp(v),
                    );
                }
                _ => {
                    s.rect(x + 1., y + 1., cw - 2., ch - 2., GRID);
                    s.text(x + cw / 2., y + ch / 2. + 4., "n/a", 10., "middle", INK2);
                }
            }
        }
    }
    Some(s.finish("Success rate by category"))
}

/// Horizontal stacked bars: failed attempts by failure class.
fn failures(r: &Report) -> Option<String> {
    let mut totals = BTreeMap::<String, usize>::new();
    for row in &r.rows {
        if let Some(c) = row["failure_class"].as_str() {
            *totals.entry(c.into()).or_default() += 1;
        }
    }
    if totals.is_empty() {
        return None;
    }
    let mut ranked = totals.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let classes = ranked
        .iter()
        .take(7)
        .map(|(c, _)| c.clone())
        .collect::<Vec<_>>();
    let bucket = |c: &str| classes.iter().position(|k| k == c).unwrap_or(classes.len());
    let counts = r
        .groups
        .iter()
        .map(|g| {
            let mut v = vec![0usize; classes.len() + 1];
            for row in g.rows(&r.rows) {
                if let Some(c) = row["failure_class"].as_str() {
                    v[bucket(c)] += 1;
                }
            }
            v
        })
        .collect::<Vec<_>>();
    let max = counts
        .iter()
        .map(|v| v.iter().sum::<usize>())
        .max()
        .unwrap_or(1)
        .max(1) as f64;
    let max = nice(max, 5);
    let left = r
        .groups
        .iter()
        .map(|g| text_width(&row_label(r, g), 11.))
        .fold(0., f64::max)
        + 32.;
    let (top, rh, plot_w) = (74., 26., 420.);
    let mut s = Svg::new(left + plot_w + 60., top + r.groups.len() as f64 * rh + 40.);
    s.title(
        "Failure classes",
        "Failed measured attempts per condition, by the first check or limit that failed",
    );
    let mut legend = classes
        .iter()
        .enumerate()
        .map(|(i, c)| (series(i), c.clone()))
        .collect::<Vec<_>>();
    if counts.iter().any(|v| v[classes.len()] > 0) {
        legend.push(("#898780", "other".into()));
    }
    s.legend(20., 62., &legend);
    for (gi, g) in r.groups.iter().enumerate() {
        let y = top + gi as f64 * rh;
        s.text(left - 8., y + 14., &row_label(r, g), 11., "end", INK2);
        let mut x = left;
        for (ci, n) in counts[gi].iter().enumerate() {
            if *n == 0 {
                continue;
            }
            let w = *n as f64 / max * plot_w;
            let color = if ci < classes.len() {
                series(ci)
            } else {
                "#898780"
            };
            s.rect(x, y + 2., (w - 2.).max(1.), 16., color);
            if w >= 26. {
                s.text(
                    x + (w - 2.) / 2.,
                    y + 14.,
                    &n.to_string(),
                    10.,
                    "middle",
                    "#ffffff",
                );
            }
            x += w;
        }
    }
    let base = top + r.groups.len() as f64 * rh + 6.;
    s.line(left, top - 4., left, base, INK2, 1.);
    for t in axis(max, 5) {
        let x = left + t / max * plot_w;
        s.text(x, base + 14., &compact(t), 10.5, "middle", INK2);
    }
    Some(s.finish("Failure classes"))
}

/// Median and 95th percentile wall time per condition.
fn latency(r: &Report) -> Option<String> {
    let stats = r
        .groups
        .iter()
        .map(|g| {
            let w = column(&g.rows(&r.rows), "wall_ms");
            (percentile(&w, 0.5), percentile(&w, 0.95))
        })
        .collect::<Vec<_>>();
    if stats.iter().all(|(a, _)| a.is_none()) {
        return None;
    }
    let max = nice(
        stats.iter().filter_map(|(_, p)| *p).fold(0., f64::max) / 1000.,
        5,
    );
    let left = r
        .groups
        .iter()
        .map(|g| text_width(&row_label(r, g), 11.))
        .fold(0., f64::max)
        + 32.;
    let (top, rh, plot_w) = (74., 34., 420.);
    let mut s = Svg::new(left + plot_w + 90., top + r.groups.len() as f64 * rh + 40.);
    s.title(
        "Wall time per attempt",
        "Median (solid) and 95th percentile (light), seconds",
    );
    s.legend(20., 62., &model_legend(r));
    for (gi, g) in r.groups.iter().enumerate() {
        let y = top + gi as f64 * rh;
        let color = series(r.model_index(&g.model));
        s.text(left - 8., y + 17., &row_label(r, g), 11., "end", INK2);
        if let (Some(p50), Some(p95)) = stats[gi] {
            let (w50, w95) = (p50 / 1000. / max * plot_w, p95 / 1000. / max * plot_w);
            s.row_bar(left, y + 2., w50, 11., color, 1.);
            s.row_bar(left, y + 15., w95, 11., color, 0.4);
            s.text(
                left + w50 + 6.,
                y + 11.,
                &format!("{}s", compact(p50 / 1000.)),
                10.,
                "start",
                INK,
            );
            s.text(
                left + w95 + 6.,
                y + 24.,
                &format!("{}s", compact(p95 / 1000.)),
                10.,
                "start",
                INK2,
            );
        } else {
            s.text(left + 6., y + 17., "n/a", 10.5, "start", INK2);
        }
    }
    let base = top + r.groups.len() as f64 * rh + 4.;
    s.line(left, top - 4., left, base, INK2, 1.);
    for t in axis(max, 5) {
        s.text(
            left + t / max * plot_w,
            base + 14.,
            &compact(t),
            10.5,
            "middle",
            INK2,
        );
    }
    Some(s.finish("Wall time per attempt"))
}

/// Peak combined memory per model against the configured limit.
fn memory(r: &Report) -> Option<String> {
    let peaks = r
        .models
        .iter()
        .map(|m| {
            let rows = r
                .rows
                .iter()
                .filter(|x| x["model"] == m.as_str())
                .collect::<Vec<_>>();
            column(&rows, "combined_peak_bytes")
                .into_iter()
                .filter(|b| *b > 0.)
                .fold(None, |a: Option<f64>, b| Some(a.map_or(b, |a| a.max(b))))
                .map(|b| b / 1_048_576.)
        })
        .collect::<Vec<_>>();
    if peaks.iter().all(Option::is_none) {
        return None;
    }
    let limit = r.memory_limit_mb;
    let max = peaks
        .iter()
        .flatten()
        .copied()
        .chain(limit)
        .fold(1., f64::max);
    let max = nice(max, 5);
    let left = r
        .models
        .iter()
        .map(|m| text_width(m, 11.))
        .fold(0., f64::max)
        + 32.;
    let (top, rh, plot_w) = (64., 34., 420.);
    let mut s = Svg::new(left + plot_w + 90., top + r.models.len() as f64 * rh + 44.);
    s.title(
        "Peak memory",
        "Harness plus inference RSS, highest sample over all measured attempts",
    );
    for (i, m) in r.models.iter().enumerate() {
        let y = top + i as f64 * rh;
        s.text(left - 8., y + 17., m, 11., "end", INK2);
        match peaks[i] {
            Some(mb) => {
                let w = mb / max * plot_w;
                s.row_bar(left, y + 5., w, 18., series(i), 1.);
                s.text(
                    left + w + 6.,
                    y + 18.,
                    &format!("{mb:.0} MB"),
                    10.5,
                    "start",
                    INK,
                );
            }
            None => s.text(left + 6., y + 18., "n/a", 10.5, "start", INK2),
        }
    }
    let base = top + r.models.len() as f64 * rh + 4.;
    if let Some(l) = limit {
        let x = left + l / max * plot_w;
        s.line(x, top - 8., x, base, INK, 1.);
        s.text(
            x + 4.,
            top - 12.,
            &format!("limit {l:.0} MB"),
            10.5,
            "start",
            INK,
        );
    }
    s.line(left, top - 4., left, base, INK2, 1.);
    for t in axis(max, 5) {
        s.text(
            left + t / max * plot_w,
            base + 14.,
            &compact(t),
            10.5,
            "middle",
            INK2,
        );
    }
    Some(s.finish("Peak memory"))
}

/// Memory and temperature over time for one run, as two stacked panels.
fn timeline(r: &Report) -> Option<String> {
    let t = r.timeline.as_ref()?;
    let (left, right, width) = (64., 24., 760.);
    let plot_w = width - left - right;
    let tmax = nice(t.seconds.iter().copied().fold(0., f64::max), 6);
    let x = |sec: f64| left + sec / tmax * plot_w;
    let mut s = Svg::new(width, 400.);
    s.title(
        "Resources over one run",
        &format!("Run {}: {}", t.run_id, t.label),
    );
    s.legend(
        20.,
        62.,
        &[
            (series(0), "harness RSS".into()),
            (series(1), "inference RSS".into()),
        ],
    );
    // Memory panel
    let (top, height) = (80., 130.);
    let mmax = t
        .harness_mb
        .iter()
        .chain(&t.inference_mb)
        .copied()
        .chain(r.memory_limit_mb)
        .fold(1., f64::max);
    let mmax = nice(mmax, 4);
    y_grid(&mut s, left, left + plot_w, top, height, mmax, |v| {
        format!("{} MB", compact(v))
    });
    s.polyline(
        &t.seconds
            .iter()
            .zip(&t.harness_mb)
            .map(|(a, b)| (x(*a), top + height - b / mmax * height))
            .collect::<Vec<_>>(),
        series(0),
    );
    s.polyline(
        &t.seconds
            .iter()
            .zip(&t.inference_mb)
            .map(|(a, b)| (x(*a), top + height - b / mmax * height))
            .collect::<Vec<_>>(),
        series(1),
    );
    if let Some(l) = r.memory_limit_mb {
        let y = top + height - l / mmax * height;
        s.line(left, y, left + plot_w, y, INK, 1.);
        s.text(
            left + plot_w,
            y - 4.,
            &format!("limit {l:.0} MB"),
            10.5,
            "end",
            INK,
        );
    }
    // Temperature panel
    let (top2, height2) = (250., 90.);
    let temps = t
        .temperature_c
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    if temps.is_empty() {
        s.text(
            left,
            top2 + 40.,
            "Temperature: n/a (no thermal sensor on this device)",
            11.5,
            "start",
            INK2,
        );
    } else {
        let lo = temps.iter().copied().fold(f64::MAX, f64::min).floor();
        let hi = temps
            .iter()
            .copied()
            .fold(f64::MIN, f64::max)
            .ceil()
            .max(lo + 1.);
        let hi = lo + nice(hi - lo, 3);
        for v in axis(hi - lo, 3) {
            let y = top2 + height2 - v / (hi - lo) * height2;
            s.line(left, y, left + plot_w, y, GRID, 1.);
            s.text(
                left - 8.,
                y + 4.,
                &format!("{:.0} °C", lo + v),
                11.,
                "end",
                INK2,
            );
        }
        let points = t
            .seconds
            .iter()
            .zip(&t.temperature_c)
            .filter_map(|(a, b)| {
                b.map(|b| (x(*a), top2 + height2 - (b - lo) / (hi - lo) * height2))
            })
            .collect::<Vec<_>>();
        s.polyline(&points, series(1));
    }
    let base = top2 + height2 + 6.;
    s.line(left, base, left + plot_w, base, INK2, 1.);
    for v in axis(tmax, 6) {
        s.text(x(v), base + 14., &compact(v), 10.5, "middle", INK2);
    }
    s.text(
        left + plot_w / 2.,
        base + 32.,
        "seconds",
        11.,
        "middle",
        INK2,
    );
    Some(s.finish("Resources over one run"))
}

/// Generated tokens against wall time for every measured attempt.
fn scatter(r: &Report) -> Option<String> {
    let points = r
        .rows
        .iter()
        .filter_map(|row| {
            Some((
                row["generated_tokens"].as_f64()?,
                row["wall_ms"].as_f64()? / 1000.,
                r.model_index(row["model"].as_str()?),
            ))
        })
        .collect::<Vec<_>>();
    if points.len() < 2 {
        return None;
    }
    let xmax = points.iter().map(|p| p.0).fold(0., f64::max);
    let ymax = points.iter().map(|p| p.1).fold(0., f64::max);
    let (xm, ym) = (nice(xmax, 5), nice(ymax, 4));
    let xt = axis(xm, 5);
    let (left, top, w, h) = (64., 74., 640., 300.);
    let mut s = Svg::new(left + w + 30., top + h + 54.);
    s.title(
        "Tokens against wall time",
        "Each dot is one measured attempt",
    );
    s.legend(20., 62., &model_legend(r));
    y_grid(&mut s, left, left + w, top, h, ym, |v| {
        format!("{}s", compact(v))
    });
    for t in &xt {
        s.text(
            left + t / xm * w,
            top + h + 16.,
            &compact(*t),
            10.5,
            "middle",
            INK2,
        );
    }
    s.line(left, top + h, left + w, top + h, INK2, 1.);
    s.text(
        left + w / 2.,
        top + h + 36.,
        "generated tokens",
        11.,
        "middle",
        INK2,
    );
    for (gx, wall, m) in points {
        s.circle(left + gx / xm * w, top + h - wall / ym * h, 4., series(m));
    }
    Some(s.finish("Tokens against wall time"))
}
