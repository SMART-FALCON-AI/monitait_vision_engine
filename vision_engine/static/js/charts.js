// Embedded detection-insight charts for the Charts tab.
// Data source: GET /api/detection_stats?window=... (reads inference_results in TimescaleDB).
// Two charts: per-class distribution (bar) + detections-over-time (line).
// Renders an informative empty-state when no class has Store enabled yet.

let _insightClassChart = null;
let _insightClassPie = null;
let _insightTimelineChart = null;
let _insightSizeChart = null;
let _insightConfidenceChart = null;
let _insightConfClassChart = null;
let _insightCameraScatter = null;

// 4.0.296 — pipeline_id -> ai-trainer task_id map, so clicking a chart dot annotates
// against the pipeline THAT PRODUCED it (each dot carries its `pid`), not the currently
// active pipeline. Loaded once from /api/pipelines; refreshed cheaply on chart refresh.
let _pipeTaskByIdMap = {};
async function _loadPipeTaskMap() {
    try {
        const r = await fetch('/api/pipelines');
        const d = await r.json();
        const m = {};
        for (const p of Object.values(d.pipelines || {})) {
            if (p && p.id != null && p.task_id) m[String(p.id)] = String(p.task_id);
        }
        _pipeTaskByIdMap = m;
    } catch (e) { /* keep the previous map */ }
}
function _pipeTaskById(pid) {
    return (pid != null && _pipeTaskByIdMap[String(pid)]) ? _pipeTaskByIdMap[String(pid)] : '';
}
try { _loadPipeTaskMap(); } catch (e) {}
let _insightCameraScatterEncoder = null;  // 3.25.13: kept (unused) to avoid touching unrelated old refs.
// v4.0.108 — module-level cache of the last authoritative `camera_y_order`
// returned by /api/detection_charts. `_extendScatterFromBucket` fetches
// with `scatter_only=1` which SKIPS computing camera_y_order server-side,
// so `data.camera_y_order` is empty in bucket responses. Without a cache
// the bucket path fell through to `_camToIdx(cid) = cid`, plotting dots
// at Y=raw-cam-id (1, 2, 3, 4, 5, 6) while the BASE scatter (which had
// camera_y_order in its response) plotted dots at Y=display-index
// (0, 1, 2, ...) after mapping through the same order. Net effect on
// screen: camera 6 showed as TWO rows on the Y-axis — one at its display
// index (e.g. row 0) from the base dots, another at raw id 6 from the
// bucket dots. Same for every other camera. Caching the last good order
// lets bucket dots reuse it and plot at the same Y-positions.
let _lastCameraYOrder = [];

// 4.0.180 — narrow-rectangle pointStyle for orientation-tagged defects.
// Chart.js supports pointStyle: HTMLCanvasElement — it draws the canvas
// centered at (x, y) using natural width/height (NOT scaled by `radius`).
// So per-point sizes need their own canvas; we cache by (orientation,
// rounded-size, color) to keep memory bounded even with hundreds of dots.
const _narrowRectCache = new Map();
function _narrowRectCanvas(orientation, radius, fillHex) {
    const r = Math.max(3, Math.min(20, Math.round(Number(radius) || 6)));
    const color = String(fillHex || '#94a3b8');
    const key = orientation + ':' + r + ':' + color;
    let cv = _narrowRectCache.get(key);
    if (cv) return cv;
    // 4.0.182 — sentinel names map to DEFECT ORIENTATION on the fabric, not
    // shape orientation on the canvas. The chart's X-axis IS the fabric-length
    // axis, so a "vertical defect" runs ALONG X → renders as a WIDE thin bar
    // (in harmony with X-axis). A "horizontal defect" runs ACROSS the fabric
    // (spans cameras on Y) → renders as a TALL thin bar (perpendicular to X).
    let w, h;
    if (orientation === 'vertical') {
        w = Math.round(r * 2.4);                 // wide  (defect along fabric length)
        h = Math.max(3, Math.round(r * 0.55));   // thin
    } else {
        w = Math.max(3, Math.round(r * 0.55));   // thin
        h = Math.round(r * 2.4);                 // tall  (defect across fabric width)
    }
    cv = document.createElement('canvas');
    cv.width = w; cv.height = h;
    const cctx = cv.getContext('2d');
    cctx.fillStyle = color;
    cctx.fillRect(0, 0, w, h);
    cctx.strokeStyle = 'rgba(15, 23, 42, 0.55)';
    cctx.lineWidth = 1;
    cctx.strokeRect(0.5, 0.5, w - 1, h - 1);
    _narrowRectCache.set(key, cv);
    return cv;
}
window._narrowRectCanvas = _narrowRectCanvas;
// v4.0.132 — cache shipment_length_offsets so the progressive-bucket
// loader (`scatter_only=1` responses omit this heavy field) can still
// project dots onto the length axis. Populated on every full-response
// fetch; consumed by both the initial render and _extendScatterFromBucket.
let _lastLengthOffsets = {};
// v4.0.134 — class → group map for shape selection. Populated by the
// main _refreshAdvancedChartsCore fetch; consumed by _extendScatterFromBucket
// (bucket fetches use scatter_only=1 which omits this heavy field).
let _lastClassGroups = {};

// v4.0.140 — chart-side length formatter, shared by every place that
// renders "N unit" (Score-per-shipment tooltip, bar annotation, quality
// strip, Camera×Length axis ticks, normalisation note). Takes the raw
// encoder count, a pulses-per-unit divisor (per-payload or the global
// cache), and a display label (per-payload or the global cache). Returns
// "12.3 Batch" when upu > 0, else "12,345 units" as-is.
function _fmtLenWithUpu(value, upu, label) {
    const raw = Number(value);
    if (!Number.isFinite(raw)) return '—';
    const upuN = Number(upu) || 0;
    const lab  = (label && String(label).trim()) || 'units';
    if (upuN > 0) {
        const c = raw / upuN;
        const abs = Math.abs(c);
        return (abs < 10 ? c.toFixed(1) : Math.round(c).toLocaleString()) + ' ' + lab;
    }
    return raw.toLocaleString() + ' ' + lab;
}

// v4.0.140 — pull upu + label from any of {payload row, top-level payload,
// global cache}. Order: per-row (per-shipment history), then top-level
// (current window), then global (SSE-seeded on every tick).
function _upuFromAny(row, data) {
    const from = (o) => {
        if (!o) return null;
        if (o.encoder_units_per_unit !== undefined && o.encoder_units_per_unit !== null) return Number(o.encoder_units_per_unit);
        if (o.encoder_units_per_meter !== undefined && o.encoder_units_per_meter !== null) return Number(o.encoder_units_per_meter);
        return null;
    };
    return from(row) || from(data) || Number(window._lengthCache && window._lengthCache.upu) || 0;
}
function _unitLabelFromAny(row, data) {
    return (row && row.encoder_unit)
        || (data && data.encoder_unit)
        || (window._lengthCache && window._lengthCache.unit)
        || 'units';
}

let _insightAxis = (function () { try { return localStorage.getItem('mve_insight_axis') || 'encoder'; } catch (e) { return 'encoder'; } })();   // 3.26.0 default encoder; 4.0.271 — persisted so a refresh keeps the operator's pick (was resetting, then getting overwritten Time by the stationary-guard fallback).
let _ejectionProcBar = null;
let _ejectionProcPie = null;
let _ejectionTimeline = null;
let _prodReject = null;
let _prodThroughput = null;
let _prodUptime = null;
let _prodSpc = null;
let _prodOee = null;
let _prodSpeed = null;
let _qualPareto = null;
let _qualCamera = null;
let _qualHeatmap = null;
let _qualLatency = null;
let _shipmentsLoaded = false;

// Per-class "Show" toggle (Process tab → Per-Object Configuration) mirrors into
// /api/audio_settings. The charts honor it: a class with Show=off is hidden from
// every per-class chart (class bar, confidence-by-class, camera scatter). Default
// is shown, so only classes explicitly set show=false are filtered out.
let _hiddenClasses = new Set();
async function _loadShownClasses() {
    try {
        const r = await fetch('/api/audio_settings');
        if (!r.ok) { _hiddenClasses = new Set(); return; }
        const data = await r.json();
        const m = data.audio_settings || {};
        const hidden = new Set();
        Object.keys(m).forEach(k => { if (m[k] && m[k].show === false) hidden.add(k); });
        _hiddenClasses = hidden;
    } catch (e) { _hiddenClasses = new Set(); }
}
function _isShown(cls) { return !_hiddenClasses.has(cls); }

// 4.0.245 — default the heatmap baseline to shipment_avg and the phase to p0 on
// a FRESH browser (operator spec). Only seeds when nothing is stored, so an
// explicit later choice always wins. Runs at script load, BEFORE any fetch reads
// `localStorage.getItem('mve_heatmap_baseline') || 'camera'`.
(function _seedHeatmapDefaults() {
    try {
        if (localStorage.getItem('mve_heatmap_baseline') == null) localStorage.setItem('mve_heatmap_baseline', 'shipment_avg');
        if (localStorage.getItem('mve_heatmap_phase') == null)    localStorage.setItem('mve_heatmap_phase', '0');
    } catch (_) { /* private-mode / storage-disabled — fall back to code defaults */ }
})();

const _INSIGHT_PALETTE = [
    '#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ec4899',
    '#14b8a6', '#f97316', '#6366f1', '#84cc16', '#06b6d4', '#a855f7'
];

// 4.0.26 — class → color override populated from the AI Trainer's category
// palette so chart dots match the same colour an operator sees inside the
// LSF annotate modal and the trainer's own categories tab. Keys are
// case-insensitive (matched by lower-cased category_name). Falls back to
// the hash palette below when a class is not in the trainer or trainer
// hasn't been reached yet.
let _trainerClassColor = new Map();
// Fetch trainer labels once on page load + refresh after the AI Trainer
// config is updated; results cached in-memory so per-render lookups are O(1).
async function _loadTrainerClassColors() {
    try {
        const r = await fetch('/api/ai_trainer/labels', { cache: 'no-store' });
        if (!r.ok) return;
        const d = await r.json();
        const next = new Map();
        for (const lbl of (d.labels || [])) {
            const name = String(lbl.category_name || '').trim();
            const colour = String(lbl.color || '').trim();
            if (!name || !colour) continue;
            next.set(name.toLowerCase(), colour);
        }
        _trainerClassColor = next;
        // Re-render whatever's currently on screen with the new colours.
        if (typeof refreshDetectionInsights === 'function') {
            try { refreshDetectionInsights(); } catch (e) {}
        }
    } catch (e) { /* trainer unreachable — just stay on the hash palette */ }
}
window._loadTrainerClassColors = _loadTrainerClassColors;
document.addEventListener('DOMContentLoaded', _loadTrainerClassColors);

// Stable color per class name. Prefer the AI Trainer category colour (so
// every chart dot, region chip, and LSF label share the same hue for the
// operator); fall back to the hash palette below for classes the trainer
// hasn't catalogued yet.
function _classColor(name) {
    const s = String(name || '').trim();
    if (s) {
        const c = _trainerClassColor.get(s.toLowerCase());
        if (c) return c;
    }
    let h = 0;
    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
    return _INSIGHT_PALETTE[h % _INSIGHT_PALETTE.length];
}

const _commonScaleOpts = (titleText) => ({
    responsive: true, maintainAspectRatio: false,
    plugins: {
        legend: { display: true, labels: { color: '#cbd5e1', font: { size: 10 }, boxWidth: 12 } },
        title: { display: true, text: titleText, color: '#cbd5e1', font: { size: 13 } }
    },
    scales: {
        x: { ticks: { color: '#94a3b8', font: { size: 9 }, maxTicksLimit: 8 }, grid: { display: false } },
        y: { beginAtZero: true, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148,163,184,0.1)' } }
    }
});

// 3.21.13 — Shipment Quality Score card (Phase 2 preview). Reads
// /api/shipment_quality_score for the current window + shipment, renders
// score / verdict / impact / top-defects into the #shipment-quality-card
// banner. Verdict color matches QUALITY_RELEASE / QUALITY_REINSPECT bands.
// 3.21.14 — utility: format a number of seconds into "Xh Ym" / "Ym Zs"
function _fmtDuration(sec) {
    sec = Math.max(0, Math.round(sec || 0));
    if (sec >= 3600) {
        const h = Math.floor(sec / 3600);
        const m = Math.round((sec % 3600) / 60);
        return `${h}h ${m}m`;
    }
    if (sec >= 60) {
        const m = Math.floor(sec / 60);
        const s = sec % 60;
        return `${m}m ${s}s`;
    }
    return `${sec}s`;
}

async function refreshShipmentQualityScore() {
    const win  = document.getElementById('insight-window')?.value   || '24h';
    const ship = document.getElementById('insight-shipment')?.value || '';
    const card = document.getElementById('shipment-quality-card');
    if (!card) return;
    let data;
    try {
        const r = await fetch('/api/shipment_quality_score?window=' + encodeURIComponent(win) +
                              '&shipment=' + encodeURIComponent(ship));
        data = await r.json();
    } catch (e) { console.error('shipment_quality_score fetch failed:', e); return; }

    const scoreEl   = document.getElementById('sqs-score');       // Relative
    const scoreAbsEl = document.getElementById('sqs-score-abs');    // 4.0.61 Absolute
    const verdictEl = document.getElementById('sqs-verdict');
    const normNoteEl= document.getElementById('sqs-norm-note');
    const lengthEl  = document.getElementById('sqs-length');
    const durEl     = document.getElementById('sqs-duration');
    const throughEl = document.getElementById('sqs-throughput');
    const impactPerUnitEl = document.getElementById('sqs-impact-per-unit');
    const impactUnitLabelEl = document.getElementById('sqs-impact-unit-label');
    const impactEl  = document.getElementById('sqs-impact-total');
    const countEl   = document.getElementById('sqs-count');
    const topListEl = document.getElementById('sqs-top-list');

    const score = data.score;
    const verdict = data.verdict;

    if (score === null || score === undefined) {
        if (scoreEl)   scoreEl.textContent   = '—';
        if (scoreAbsEl) scoreAbsEl.textContent = '—';
        if (verdictEl) { verdictEl.textContent = 'NO DATA'; verdictEl.style.background = 'rgba(51,65,85,0.6)'; verdictEl.style.color = 'var(--text-primary)'; }
        if (normNoteEl) normNoteEl.textContent = '—';
        if (lengthEl)  lengthEl.textContent  = '—';
        if (durEl)     durEl.textContent     = '—';
        if (throughEl) throughEl.textContent = '—';
        if (impactPerUnitEl) impactPerUnitEl.textContent = '—';
        if (impactEl)  impactEl.textContent  = '—';
        if (countEl)   countEl.textContent   = '—';
        if (topListEl) topListEl.innerHTML = '<span style="font-style:italic;">No data in this window yet.</span>';
        return;
    }

    // 4.0.61 — dual scoring: Absolute has no calibration knob so it's stable
    // across time / sites; Relative uses the calibrated score_scale_factor.
    // Fall back gracefully if the older API version doesn't emit `score_absolute`.
    if (scoreEl) scoreEl.textContent = score.toFixed(1);
    if (scoreAbsEl) {
        const abs = (data.score_absolute !== undefined && data.score_absolute !== null)
            ? data.score_absolute : score;
        scoreAbsEl.textContent = Number(abs).toFixed(1);
    }

    // Verdict color: green RELEASE / amber RE-INSPECT / red HOLD
    let bg = 'rgba(51,65,85,0.6)', fg = 'var(--text-primary)';
    if (verdict === 'RELEASE')      { bg = 'rgba(34,197,94,0.28)';  fg = '#86efac'; }
    else if (verdict === 'RE-INSPECT') { bg = 'rgba(234,179,8,0.28)'; fg = '#fcd34d'; }
    else if (verdict === 'HOLD')    { bg = 'rgba(239,68,68,0.30)';  fg = '#fca5a5'; }
    if (verdictEl) { verdictEl.textContent = verdict; verdictEl.style.background = bg; verdictEl.style.color = fg; }

    // Normalization note (so operator knows the score basis)
    if (normNoteEl) {
        const by = data.normalized_by;
        if (by === 'encoder') {
            // v4.0.140 — shared formatter. Uses per-payload upu (falls back
            // to global cache) and the operator's chosen unit label.
            normNoteEl.textContent = `Normalized by encoder span (${_fmtLenWithUpu(data.encoder_span, _upuFromAny(null, data), _unitLabelFromAny(null, data))})`;
        } else if (by === 'frame') {
            normNoteEl.textContent = `Normalized by frame count (${(data.frame_count || 0).toLocaleString()} frames) — no encoder data`;
        } else {
            normNoteEl.textContent = 'No length data available — score may not be meaningful';
        }
    }

    if (lengthEl) {
        // v4.0.140 — same formatter as the norm note above so both strings
        // never disagree on the same shipment.
        lengthEl.textContent = _fmtLenWithUpu(data.encoder_span, _upuFromAny(null, data), _unitLabelFromAny(null, data));
    }
    if (durEl)    durEl.textContent     = _fmtDuration(data.duration_sec);
    if (throughEl) {
        const t = data.throughput ?? 0;
        throughEl.textContent = t > 0 ? (t.toFixed(2) + ' ' + (data.throughput_label || 'units/sec')) : '—';
    }
    if (impactPerUnitEl) impactPerUnitEl.textContent = (data.impact_per_unit ?? 0).toFixed(3);
    if (impactUnitLabelEl) impactUnitLabelEl.textContent = (data.impact_per_unit_label || '/unit').replace(/^\//, '');
    if (impactEl) impactEl.textContent = (data.impact_total ?? 0).toFixed(1);
    if (countEl)  countEl.textContent  = (data.total_detections ?? 0).toLocaleString();

    if (topListEl) {
        const tops = data.top_defects || [];
        if (!tops.length) {
            topListEl.innerHTML = '<span style="font-style:italic;">No class has Severity > 0 yet — set in Process tab.</span>';
        } else {
            const unitLbl = (data.impact_per_unit_label || '/unit').replace(/^\//, '');
            topListEl.innerHTML = tops.map(t =>
                `<div style="display:flex; justify-content:space-between; gap:8px;">` +
                `<span style="color:var(--text-primary);">${t.class}</span>` +
                `<span><span style="color:#86efac;">impact ${t.impact.toFixed(1)}</span> · ` +
                `<span style="color:#fcd34d;">${(t.impact_per_unit ?? 0).toFixed(3)}/${unitLbl}</span> · ` +
                `${t.count.toLocaleString()} det · sev ${t.severity}</span></div>`
            ).join('');
        }
    }
}
window.refreshShipmentQualityScore = refreshShipmentQualityScore;


// 3.21.15 — server-rendered PDF download of the score card payload.
// 4.0.16 — rewritten with explicit error surfacing + a direct-navigation
// fallback. The previous blob→<a>.click() path could fail silently when
// Chrome cancelled the blob URL between createObjectURL and click (seen on
// flaky tunnels). The fallback opens the URL in a new tab, where the server's
// Content-Disposition: attachment header makes the browser save it the same
// way. Every failure mode now surfaces an alert + console.error so we never
// just "do nothing".
async function downloadQualityReport() {
    const btn = document.getElementById('sqs-download-pdf');
    const hint = document.getElementById('sqs-download-hint');
    const setBtn = (txt, disabled) => {
        if (!btn) return;
        btn.disabled = !!disabled;
        btn.textContent = txt;
    };
    setBtn('⏳ Generating…', true);
    try {
        const win  = document.getElementById('insight-window')?.value || '24h';
        const ship = document.getElementById('insight-shipment')?.value || '';
        const params = new URLSearchParams({ window: win });
        if (ship) params.set('shipment', ship);
        const url = `/api/shipment_quality_score/report.pdf?${params.toString()}`;
        console.info('[PDF] fetching', url);

        // GET probes the endpoint AND streams the PDF body in one call. If the
        // response is non-OK (503 reportlab missing, 500 backend error) we
        // surface the server's JSON `hint` field inline.
        let resp;
        try {
            resp = await fetch(url, { method: 'GET', cache: 'no-store' });
        } catch (netErr) {
            console.error('[PDF] network error:', netErr);
            alert(`Network error fetching PDF:\n${netErr.message || netErr}\n\nCheck that you're logged into MVE and the host is reachable.`);
            return;
        }
        if (!resp.ok) {
            let detail = {};
            try { detail = await resp.json(); } catch (_) { /* not JSON */ }
            console.error('[PDF] HTTP', resp.status, detail);
            alert(detail.hint
                ? `PDF library missing on the server — install it with:\n\n${detail.hint}`
                : `Report failed: HTTP ${resp.status} ${resp.statusText || ''}\nURL: ${url}`);
            return;
        }
        const blob = await resp.blob();
        if (!blob || !blob.size) {
            console.error('[PDF] empty blob, blob.size=', blob && blob.size);
            alert('Server returned an empty PDF. Check the MVE container logs for an error.');
            return;
        }
        console.info('[PDF] received', blob.size, 'bytes type=', blob.type);
        // 4.0.16 — primary path: blob URL + <a download>. Wrap in try so we
        // can fall back to direct-navigation if the browser refuses the blob.
        const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 16);
        const fname = `quality_${ship || 'all'}_${win}_${stamp}.pdf`;
        try {
            const blobUrl = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = blobUrl;
            a.download = fname;
            a.rel = 'noopener';
            a.style.display = 'none';
            document.body.appendChild(a);
            a.click();
            setTimeout(() => {
                try { a.remove(); } catch (_) {}
                URL.revokeObjectURL(blobUrl);
            }, 4000);
        } catch (downloadErr) {
            // Fallback: open the URL itself; server's Content-Disposition
            // tells the browser to download.
            console.warn('[PDF] blob download failed, falling back to direct navigation:', downloadErr);
            window.open(url, '_blank', 'noopener');
        }
        if (hint) hint.textContent = '✓ downloaded';
        setTimeout(() => { if (hint) hint.textContent = 'uses current window + shipment'; }, 3000);
    } catch (e) {
        console.error('[PDF] downloadQualityReport failed:', e);
        alert('Report generation failed: ' + (e.message || e));
    } finally {
        setBtn('📄 Download PDF', false);
    }
}
window.downloadQualityReport = downloadQualityReport;


// 3.25.12 — calibrate the global score_scale_factor so the strip stops being
// all-green. Two clicks: first to preview (no DB write), second to apply.
async function calibrateShipmentScore() {
    const btn = document.getElementById('sqs-calibrate');
    const resultEl = document.getElementById('sqs-calibrate-result');
    if (!btn || !resultEl) return;
    btn.disabled = true; btn.style.opacity = '0.6';
    const prevState = btn.dataset.calState || 'preview';
    try {
        if (prevState === 'preview') {
            // Preview: ask for the recommendation but DON'T apply.
            const r = await fetch('/api/score/calibrate', {
                method: 'POST', headers: {'Content-Type':'application/json'},
                body: JSON.stringify({ target_p50: 85, target_p5: 60, window: '7d', apply: false }),
            });
            const d = await r.json();
            if (!r.ok) {
                resultEl.style.display = 'block';
                resultEl.innerHTML = '<b>✗ ' + (d.error || ('HTTP ' + r.status)) + '</b>';
                return;
            }
            resultEl.style.display = 'block';
            resultEl.innerHTML =
                '<b>Calibration preview (' + d.shipments_sampled + ' shipments, 7d):</b><br>' +
                'Current scale: <code>' + d.current_scale_factor + '</code><br>' +
                'Recommended:   <code>' + d.recommended_scale_factor + '</code><br>' +
                'After applying:&nbsp; p5 ≈ <b>' + d.projected_score_p5 + '</b>, ' +
                'p50 ≈ <b>' + d.projected_score_p50 + '</b>, ' +
                'p95 ≈ <b>' + d.projected_score_p95 + '</b><br>' +
                '<i>Click again to apply.</i>';
            btn.textContent = '✓ Apply ' + d.recommended_scale_factor;
            btn.dataset.calState = 'apply';
            btn.dataset.calRecommended = String(d.recommended_scale_factor);
        } else {
            // Apply.
            const r = await fetch('/api/score/calibrate', {
                method: 'POST', headers: {'Content-Type':'application/json'},
                body: JSON.stringify({ target_p50: 85, target_p5: 60, window: '7d', apply: true }),
            });
            const d = await r.json();
            if (!r.ok || !d.applied) {
                resultEl.style.display = 'block';
                resultEl.innerHTML = '<b>✗ Apply failed: ' + (d.error || d.apply_error || ('HTTP ' + r.status)) + '</b>';
                return;
            }
            resultEl.innerHTML =
                '<b>✓ Applied scale = ' + d.recommended_scale_factor + '.</b><br>' +
                'New score distribution: p5 ≈ ' + d.projected_score_p5 + ', ' +
                'p50 ≈ ' + d.projected_score_p50 + ', p95 ≈ ' + d.projected_score_p95 + '.<br>' +
                '<i>Refresh the page to see the new score / strip colors.</i>';
            btn.textContent = '🎯 Calibrate score';
            btn.dataset.calState = 'preview';
            // Refresh the score card so the operator sees the new score immediately.
            if (typeof refreshShipmentQualityScore === 'function') {
                setTimeout(refreshShipmentQualityScore, 600);
            }
            if (typeof refreshQualityCharts === 'function') {
                setTimeout(refreshQualityCharts, 800);
            }
        }
    } catch (e) {
        resultEl.style.display = 'block';
        resultEl.innerHTML = '<b>✗ Network error: ' + (e.message || e) + '</b>';
    } finally {
        btn.disabled = false; btn.style.opacity = '1';
    }
}
window.calibrateShipmentScore = calibrateShipmentScore;

// 4.0.28 — Process-tab manual override for score_scale_factor. Reads the
// current value into the input field on page load + POST handler on Set.
async function _loadScoreScaleFactorIntoUI() {
    try {
        const r = await fetch('/api/score_scale_factor', { cache: 'no-store' });
        if (!r.ok) return;
        const d = await r.json();
        const v = Number(d.score_scale_factor);
        const inp = document.getElementById('score-scale-factor-input');
        const cur = document.getElementById('score-scale-current');
        if (Number.isFinite(v) && inp) inp.value = String(v);
        if (cur) cur.textContent = 'current: ' + (Number.isFinite(v) ? v : '?');
    } catch (e) { /* leave defaults */ }
}
window._loadScoreScaleFactorIntoUI = _loadScoreScaleFactorIntoUI;
document.addEventListener('DOMContentLoaded', _loadScoreScaleFactorIntoUI);

async function setScoreScaleFactor() {
    const respEl = document.getElementById('config-score_scale_factor-response');
    const inp    = document.getElementById('score-scale-factor-input');
    const cur    = document.getElementById('score-scale-current');
    if (!inp) return;
    const raw = (inp.value || '').trim();
    const v = Number(raw);
    if (!Number.isFinite(v) || v <= 0) {
        if (respEl) {
            respEl.textContent = 'Error: must be a positive number';
            respEl.className = 'control-response error';
        }
        return;
    }
    try {
        const r = await fetch('/api/score_scale_factor', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ score_scale_factor: v }),
        });
        const d = await r.json();
        if (!r.ok) {
            if (respEl) {
                respEl.textContent = 'Error: ' + (d.error || ('HTTP ' + r.status));
                respEl.className = 'control-response error';
            }
            return;
        }
        const saved = Number(d.score_scale_factor);
        if (respEl) {
            respEl.textContent = 'OK: score_scale_factor = ' + saved;
            respEl.className = 'control-response success';
            setTimeout(() => { respEl.textContent = ''; respEl.className = 'control-response'; }, 3000);
        }
        if (cur) cur.textContent = 'current: ' + saved;
        // Nudge any Charts-tab cards that read the score to refresh.
        if (typeof refreshShipmentQualityScore === 'function') {
            setTimeout(refreshShipmentQualityScore, 300);
        }
        if (typeof refreshQualityCharts === 'function') {
            setTimeout(refreshQualityCharts, 400);
        }
    } catch (e) {
        if (respEl) {
            respEl.textContent = 'Error: ' + (e.message || e);
            respEl.className = 'control-response error';
        }
    }
}
window.setScoreScaleFactor = setScoreScaleFactor;

// 4.0.32 — Process tab Colour Target editor. Reads /api/color_target on
// page load + after Save; writes back with the row inputs.
async function _loadColorTargetIntoUI() {
    try {
        const r = await fetch('/api/color_target', { cache: 'no-store' });
        if (!r.ok) return;
        const d = await r.json();
        const tgt = (d && d.color_target) || {};
        const wrap = document.getElementById('color-target-rows');
        if (!wrap) return;
        wrap.innerHTML = '';
        const keys = Object.keys(tgt).sort((a,b) => Number(a) - Number(b));
        if (keys.length === 0) {
            // Seed with a couple of empty rows so the operator has somewhere to type.
            _appendColorTargetRow(1, '', '', '');
            _appendColorTargetRow(2, '', '', '');
        } else {
            for (const cid of keys) {
                const v = tgt[cid] || {};
                _appendColorTargetRow(cid, v.L, v.a, v.b);
            }
        }
    } catch (e) { /* leave UI as-is */ }
}
window._loadColorTargetIntoUI = _loadColorTargetIntoUI;
document.addEventListener('DOMContentLoaded', _loadColorTargetIntoUI);

function _appendColorTargetRow(cam, L, a, b) {
    const wrap = document.getElementById('color-target-rows');
    if (!wrap) return;
    const row = document.createElement('div');
    row.className = 'color-target-row';
    row.style.cssText = 'display:flex; gap:8px; align-items:center;';
    row.innerHTML = `
        <span style="color:var(--text-secondary); min-width:55px; font-size:12px;">Camera</span>
        <input type="number" class="color-target-cam"  value="${cam}"               min="1" max="64" step="1" style="width:55px; padding:3px 6px; background:rgba(30,41,59,0.6); color:var(--text-primary); border:1px solid rgba(51,65,85,0.6); border-radius:4px; font-size:12px;">
        <span style="color:var(--text-secondary); font-size:12px;">L*</span>
        <input type="number" class="color-target-L"    value="${L == null ? '' : L}" min="0" max="100" step="0.1" style="width:75px; padding:3px 6px; background:rgba(30,41,59,0.6); color:var(--text-primary); border:1px solid rgba(51,65,85,0.6); border-radius:4px; font-size:12px;">
        <span style="color:var(--text-secondary); font-size:12px;">a*</span>
        <input type="number" class="color-target-a"    value="${a == null ? '' : a}" min="-128" max="127" step="0.1" style="width:75px; padding:3px 6px; background:rgba(30,41,59,0.6); color:var(--text-primary); border:1px solid rgba(51,65,85,0.6); border-radius:4px; font-size:12px;">
        <span style="color:var(--text-secondary); font-size:12px;">b*</span>
        <input type="number" class="color-target-b"    value="${b == null ? '' : b}" min="-128" max="127" step="0.1" style="width:75px; padding:3px 6px; background:rgba(30,41,59,0.6); color:var(--text-primary); border:1px solid rgba(51,65,85,0.6); border-radius:4px; font-size:12px;">
        <button onclick="this.parentElement.remove()" style="background:rgba(239,68,68,0.25); color:#fca5a5; border:1px solid rgba(239,68,68,0.5); padding:3px 8px; border-radius:4px; cursor:pointer; font-size:11px;">remove</button>
    `;
    wrap.appendChild(row);
}

function addColorTargetRow() {
    const rows = document.querySelectorAll('#color-target-rows .color-target-row');
    let nextCam = 1;
    rows.forEach(r => {
        const c = parseInt(r.querySelector('.color-target-cam').value, 10) || 0;
        if (c >= nextCam) nextCam = c + 1;
    });
    _appendColorTargetRow(nextCam, '', '', '');
}
window.addColorTargetRow = addColorTargetRow;

async function saveColorTarget() {
    const respEl = document.getElementById('config-color_target-response');
    const out = {};
    const rows = document.querySelectorAll('#color-target-rows .color-target-row');
    for (const row of rows) {
        const cam = parseInt(row.querySelector('.color-target-cam').value, 10);
        const L = parseFloat(row.querySelector('.color-target-L').value);
        const a = parseFloat(row.querySelector('.color-target-a').value);
        const b = parseFloat(row.querySelector('.color-target-b').value);
        if (!Number.isFinite(cam) || cam < 1) continue;
        if (!Number.isFinite(L) && !Number.isFinite(a) && !Number.isFinite(b)) continue;
        out[String(cam)] = {
            L: Number.isFinite(L) ? L : 0,
            a: Number.isFinite(a) ? a : 0,
            b: Number.isFinite(b) ? b : 0,
        };
    }
    try {
        const r = await fetch('/api/color_target', {
            method: 'POST', headers: {'Content-Type':'application/json'},
            body: JSON.stringify({ color_target: out }),
        });
        const d = await r.json();
        if (!r.ok) {
            if (respEl) { respEl.textContent = 'Error: ' + (d.error || 'HTTP ' + r.status); respEl.className = 'control-response error'; }
            return;
        }
        if (respEl) {
            const n = Object.keys(d.color_target || {}).length;
            respEl.textContent = `OK: saved ${n} camera target${n === 1 ? '' : 's'}.`;
            respEl.className = 'control-response success';
            setTimeout(() => { respEl.textContent = ''; respEl.className = 'control-response'; }, 3000);
        }
        if (typeof refreshDetectionInsights === 'function') refreshDetectionInsights();
    } catch (e) {
        if (respEl) { respEl.textContent = 'Error: ' + (e.message || e); respEl.className = 'control-response error'; }
    }
}
window.saveColorTarget = saveColorTarget;

// 4.0.30 — operator clicks one of the heatmap baseline toggle buttons.
// 4.0.32 adds:
//   - shipment_start: if "All shipments" is selected, auto-pick the
//     current active shipment from /api/status. Friendlier default than
//     forcing the operator to find the dropdown.
//   - reference_frame: if no reference is set yet, enter PICK MODE so the
//     next chart click captures the position. Clicking the button when a
//     reference IS set just activates the mode (no pick needed).
async function setHeatmapBaseline(mode) {
    // 4.0.174 — shipment_avg added; was missing here so clicks silently
    // returned. No auto-shipment / config-check needed for shipment_avg:
    // its delta is precomputed per cell (color_heatmap.cells.delta_from_avg)
    // and works for the "All shipments" view too.
    const VALID = ['camera', 'shipment_start', 'shipment_avg', 'target', 'reference_frame'];
    if (!VALID.includes(mode)) return;

    if (mode === 'shipment_start') {
        const shipSel = document.getElementById('insight-shipment');
        if (shipSel && !shipSel.value) {
            // Try to auto-select the current shipment.
            fetch('/api/status').then(r => r.json()).then(s => {
                const cur = s && s.shipment;
                if (cur && cur !== 'no_shipment') {
                    shipSel.value = cur;
                    if (typeof _syncTimeframeVisibility === 'function') _syncTimeframeVisibility(); // 4.0.198
                    localStorage.setItem('mve_heatmap_baseline', mode);
                    if (typeof refreshDetectionInsights === 'function') refreshDetectionInsights();
                    _kickAnomalyBaseline('shipment_start', {});
                } else {
                    alert('No active shipment — pick a specific shipment from the Shipment dropdown first.');
                }
            }).catch(() => {
                alert('Could not read current shipment. Pick one from the Shipment dropdown.');
            });
            return;
        }
    }

    if (mode === 'target') {
        // 4.0.32 — friendly hint when target hasn't been configured yet.
        // We can't tell from JS alone, so fire a HEAD on /api/color_target.
        try {
            const r = await fetch('/api/color_target', { cache: 'no-store' });
            const d = await r.json();
            const tgt = (d && d.color_target) || {};
            if (Object.keys(tgt).length === 0) {
                alert('No colour target configured. Set per-camera L*a*b* values in Process tab → 🎨 Colour Target, then come back and pick this mode.');
                return;
            }
        } catch (e) { /* fall through and try anyway */ }
        // Has targets — proceed normally.
        localStorage.setItem('mve_heatmap_baseline', mode);
        if (typeof refreshDetectionInsights === 'function') refreshDetectionInsights();
        return;
    }

    if (mode === 'reference_frame') {
        // 4.0.42 — ALWAYS enter pick mode on a Reference click. The earlier
        // "already set, just activate" shortcut skipped pick mode when a
        // reference existed, leaving _pickingBaseline=false. The chart's
        // onClick then ran openFrameInAnnotator (els is always non-empty
        // with `nearest, intersect:false` so any click finds a dot). With
        // every Reference click entering pick mode, the operator can RE-pick
        // any time, and onClick always returns early via _onBaselinePickClick.
        window._pickingBaseline = true;
        const hint = document.getElementById('hm-pick-hint');
        if (hint) {
            hint.style.display = 'inline';
            hint.textContent = '🎯 Click anywhere on the chart to set the colour reference point';
        }
        const sc = document.getElementById('insight-camera-scatter');
        if (sc) sc.style.cursor = 'crosshair';
        return;
    }

    localStorage.setItem('mve_heatmap_baseline', mode);
    if (typeof refreshDetectionInsights === 'function') {
        refreshDetectionInsights();
    }
    // 4.0.50 — Fire the anomaly-baseline rebuild for the same mode so a
    // single mode click drives both models. `target` and `reference_frame`
    // are handled elsewhere (target is colour-only; reference_frame fires
    // from _onBaselinePickClick when the operator finishes picking).
    if (mode === 'camera' || mode === 'shipment_start') {
        _kickAnomalyBaseline(mode, {});
    }
}
window.setHeatmapBaseline = setHeatmapBaseline;

// 4.0.34 — Phase filter (heatmap-only). `phase=""` means "All phases".
// Sticky in localStorage; one refetch on change so the heatmap repaints.
function setHeatmapPhase(phase) {
    const v = (phase == null) ? '' : String(phase);
    localStorage.setItem('mve_heatmap_phase', v);
    if (typeof refreshDetectionInsights === 'function') refreshDetectionInsights();
}
window.setHeatmapPhase = setHeatmapPhase;

// 4.0.177 — Colour scale mode: 'absolute' (default) uses fixed CIELAB
// tolerance bands (≤2/≤5/≤10/>10). 'relative' normalises to the visible
// span's max |Δ|. Sticky in localStorage; heatmap plugin reads the value
// each render so no refetch is needed — just a chart update.
function setHeatmapScale(mode) {
    const v = (mode === 'relative') ? 'relative' : 'absolute';
    localStorage.setItem('mve_heatmap_scale', v);
    // Repaint pill state
    const a = document.getElementById('hm-scale-absolute');
    const r = document.getElementById('hm-scale-relative');
    const on  = 'linear-gradient(135deg,#10b981,#047857)';
    const off = 'rgba(51,65,85,0.5)';
    if (a) { a.style.background = (v === 'absolute') ? on : off; a.style.color = (v === 'absolute') ? '#fff' : '#cbd5e1'; }
    if (r) { r.style.background = (v === 'relative') ? on : off; r.style.color = (v === 'relative') ? '#fff' : '#cbd5e1'; }
    const _ss = document.getElementById('hm-scale-select'); // 4.0.244 dropdown sync
    if (_ss && _ss.value !== v) _ss.value = v;
    if (_insightCameraScatter) { try { _insightCameraScatter.update('none'); } catch (_) {} }
}
window.setHeatmapScale = setHeatmapScale;
// Restore sticky state at load
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        const v = localStorage.getItem('mve_heatmap_scale') || 'absolute';
        setHeatmapScale(v);
    }, 800);
});

// 4.0.194 — Sticky window dropdown. The <select id="insight-window">
// HTML has `<option value="24h" selected>` so every hard-refresh snapped
// back to 24h regardless of the operator's last pick. Persist to
// localStorage on change, restore on DOMContentLoaded. Also wrap the
// onchange so the choice is written BEFORE refreshDetectionInsights runs
// — the standalone shipment-strip refresher (v4.0.192) reads the current
// dropdown value at fetch-time, so writing first keeps everything in sync
// on the very first refetch.
function _initInsightWindowSticky() {
    const el = document.getElementById('insight-window');
    if (!el) return;
    const saved = localStorage.getItem('mve_insight_window');
    if (saved) {
        // Only apply if the saved value is one of the actual options —
        // guards against a stale value from an older build (e.g. "12h"
        // never existed) forcing the select into an invalid state.
        const opts = Array.from(el.options).map(o => o.value);
        if (opts.includes(saved) && el.value !== saved) {
            el.value = saved;
            // Kick a refresh so the chart matches the restored value on
            // first paint. The HTML default was 24h, so if we just wrote
            // 6h into el.value, refreshDetectionInsights below fetches 6h.
            if (typeof refreshDetectionInsights === 'function') {
                try { refreshDetectionInsights(); } catch (_) {}
            }
        }
    }
    el.addEventListener('change', () => {
        localStorage.setItem('mve_insight_window', el.value);
    });
}
document.addEventListener('DOMContentLoaded', () => setTimeout(_initInsightWindowSticky, 900));
// 4.0.198 — set the initial timeframe-group visibility to match the shipment
// picker's state (hidden if a shipment is somehow pre-filled; shown for the
// default "All shipments").
document.addEventListener('DOMContentLoaded', () => setTimeout(() => {
    try { _syncTimeframeVisibility(); } catch (_) {}
}, 950));

// 4.0.185 — Colour-cell visibility toggle. Operators asked for the same
// hide/show affordance the per-class chips give for scatter dots, but for
// the whole ΔE heatmap layer. Sticky in localStorage; the colorHeatmap
// plugin (beforeDatasetsDraw) reads the flag on every paint and early-
// returns when 'hidden', so no refetch is needed — just a chart update.
function setHeatmapCells(mode) {
    const v = (mode === 'hidden') ? 'hidden' : 'shown';
    localStorage.setItem('mve_heatmap_cells', v);
    const s = document.getElementById('hm-cells-shown');
    const h = document.getElementById('hm-cells-hidden');
    const on  = 'linear-gradient(135deg,#10b981,#047857)';
    const off = 'rgba(51,65,85,0.5)';
    if (s) { s.style.background = (v === 'shown')  ? on : off; s.style.color = (v === 'shown')  ? '#fff' : '#cbd5e1'; }
    if (h) { h.style.background = (v === 'hidden') ? on : off; h.style.color = (v === 'hidden') ? '#fff' : '#cbd5e1'; }
    const _cs = document.getElementById('hm-cells-select'); // 4.0.244 dropdown sync
    if (_cs && _cs.value !== v) _cs.value = v;
    if (_insightCameraScatter) { try { _insightCameraScatter.update('none'); } catch (_) {} }
}
window.setHeatmapCells = setHeatmapCells;
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        const v = localStorage.getItem('mve_heatmap_cells') || 'shown';
        setHeatmapCells(v);
    }, 800);
});

// 4.0.161 — Parent-class filter (heatmap-only). Same shape as phase.
// `parent_class=""` = All; `_root` = whole-frame fallback rows;
// anything else = one specific parent-role class name.
function setHeatmapParentClass(parentClass) {
    const v = (parentClass == null) ? '' : String(parentClass);
    localStorage.setItem('mve_heatmap_parent_class', v);
    if (typeof refreshDetectionInsights === 'function') refreshDetectionInsights();
}
window.setHeatmapParentClass = setHeatmapParentClass;

// 4.0.35 — bucket count is shared across the colour heatmap, quality strip
// and ejection strip so the three rows line up visually. Sticky in
// localStorage. Re-renders all three (refreshAdvancedCharts orchestrates
// the heatmap + scatter; the strip loaders are called from there too).
function setBucketCount(value) {
    let n = parseInt(value, 10);
    if (!Number.isFinite(n) || n <= 0) n = 48;
    n = Math.max(4, Math.min(192, n));
    localStorage.setItem('mve_bucket_count', String(n));
    const el = document.getElementById('hm-bucket-count');
    if (el && parseInt(el.value, 10) !== n) el.value = String(n);
    if (typeof refreshDetectionInsights === 'function') refreshDetectionInsights();
}
window.setBucketCount = setBucketCount;

// v4.0.80 — per-class dot cap. Server clamps to [100, 5000]; UI options
// go up to 3000 with a warning at the top end. Higher = denser scatter but
// slower Chart.js pan/zoom above ~3000.
function _dotCap() {
    // 4.0.203 — floor lowered 100 → 10 so the operator can pick a sparse
    // per-bucket view (the "10" dropdown option). dot_cap is applied PER
    // BUCKET by the progressive ladder, so 10 = up to 10 dots per bucket.
    const raw = parseInt(localStorage.getItem('mve_dot_cap') || '750', 10);
    return Math.max(10, Math.min(5000, Number.isFinite(raw) ? raw : 750));
}
function setDotCap(value) {
    let n = parseInt(value, 10);
    if (!Number.isFinite(n)) n = 750;
    n = Math.max(10, Math.min(5000, n));
    localStorage.setItem('mve_dot_cap', String(n));
    const el = document.getElementById('hm-dot-cap');
    if (el) el.value = String(n);
    const warn = document.getElementById('hm-dot-cap-warn');
    if (warn) warn.style.display = (n >= 3000) ? 'inline' : 'none';
    if (typeof refreshDetectionInsights === 'function') refreshDetectionInsights();
}
window.setDotCap = setDotCap;
// Restore saved value on load and update warning visibility.
document.addEventListener('DOMContentLoaded', () => {
    const el = document.getElementById('hm-dot-cap');
    if (el) {
        el.value = String(_dotCap());
    }
    const warn = document.getElementById('hm-dot-cap-warn');
    if (warn) warn.style.display = (_dotCap() >= 3000) ? 'inline' : 'none';
});

// 4.0.38 — 4.0.36's sticky-legend layer was over-complication. Source of
// truth for "is this class visible" is now ONLY the Process tab "Show"
// checkbox (which 4.0.37 wires through to the chart scatter via a server
// filter). Chart.js's default per-session legend click still works for
// ad-hoc hiding but is intentionally NOT sticky.

// 4.0.35 — hydrate the bucket-count input from localStorage on first DOM
// ready so the control reflects the persisted choice before the first
// chart render.
(function _hydrateChartToolbar() {
    try {
        const apply = () => {
            const bkEl = document.getElementById('hm-bucket-count');
            const v = parseInt(localStorage.getItem('mve_bucket_count') || '', 10);
            if (bkEl && Number.isFinite(v) && v > 0) bkEl.value = String(v);
        };
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', apply, { once: true });
        } else {
            apply();
        }
    } catch (e) { /* non-fatal */ }
})();

// 4.0.34 — rebuild the phase button row from the available phases. Always
// emits the leading "All" button (id `hm-phase-all`, already in the DOM from
// status.html); the rest are appended dynamically with ids `hm-phase-<v>`.
function _renderPhaseButtons(available, activePhase) {
    const wrap = document.getElementById('hm-phase-buttons');
    if (!wrap) return;
    const sel = (localStorage.getItem('mve_heatmap_phase') || '');
    // Drop everything except the "All" button — we'll re-append the rest.
    const allBtn = document.getElementById('hm-phase-all');
    wrap.innerHTML = '';
    if (allBtn) wrap.appendChild(allBtn);
    const setBtnActive = (btn, isActive) => {
        if (!btn) return;
        btn.style.background = isActive
            ? 'linear-gradient(135deg,#10b981,#047857)'
            : 'rgba(51,65,85,0.5)';
        btn.style.color = isActive ? '#fff' : '#cbd5e1';
    };
    setBtnActive(allBtn, sel === '');
    for (const ph of (available || [])) {
        if (!ph && ph !== 0) continue;
        const id = 'hm-phase-' + String(ph);
        const btn = document.createElement('button');
        btn.id = id;
        btn.textContent = 'p' + ph;
        btn.title = 'Show colour data only from phase ' + ph + '.';
        btn.style.cssText = 'border:none; padding:3px 10px; cursor:pointer; font-size:11px; border-radius:3px; font-weight:600;';
        btn.onclick = () => setHeatmapPhase(ph);
        setBtnActive(btn, String(sel) === String(ph));
        wrap.appendChild(btn);
    }
    // If the localStorage phase isn't in the available list anymore (e.g. data
    // window scrolled past it), silently reset to 'All' so we never freeze on
    // an empty heatmap. Doesn't refetch — the current render is already 'all'.
    if (sel && !(available || []).map(String).includes(String(sel))) {
        localStorage.setItem('mve_heatmap_phase', '');
        setBtnActive(allBtn, true);
    }
    // 4.0.262 — populate the visible Phase DROPDOWN from the same available list (the
    // buttons above are now hidden, kept only for setHeatmapPhase compatibility).
    const _phSel = document.getElementById('hm-phase-select');
    if (_phSel) {
        const _phCur = (localStorage.getItem('mve_heatmap_phase') || '');
        let _o = '<option value="">🌐 All</option>';
        for (const ph of (available || [])) { if (!ph && ph !== 0) continue; _o += '<option value="' + String(ph) + '">p' + String(ph) + '</option>'; }
        _phSel.innerHTML = _o;
        const _avail = (available || []).map(String);
        _phSel.value = (_phCur && _avail.includes(String(_phCur))) ? String(_phCur) : '';
    }
}
window._renderPhaseButtons = _renderPhaseButtons;

// 4.0.161 — parent-class picker. Wraps in #hm-parent-class-wrap which
// stays hidden while `available` is empty or has ≤ 1 entry (single-parent
// or _root-only sites don't need the picker). Buttons: leading "All",
// then one per discovered parent_class name.
function _renderParentClassButtons(available) {
    // 4.0.306 — now a DROPDOWN (was a button row) to match the other toolbar pickers.
    const wrap = document.getElementById('hm-parent-class-wrap');
    const sel = document.getElementById('hm-parent-class-select');
    if (!wrap || !sel) return;
    const list = (available || []).filter(x => x != null).map(String);
    // Hide the whole row when there's nothing meaningful to pick between (≤1 parent-role class).
    if (list.length <= 1) {
        wrap.style.display = 'none';
        return;
    }
    wrap.style.display = window._mveColorCheckEnabled === true ? 'inline-flex' : 'none';
    const cur = (localStorage.getItem('mve_heatmap_parent_class') || '');
    // Rebuild options: "All" + one per discovered parent class.
    sel.innerHTML = '';
    const _add = (val, label) => {
        const o = document.createElement('option');
        o.value = val; o.textContent = label;
        sel.appendChild(o);
    };
    _add('', '🌐 All');
    for (const pc of list) _add(pc, pc === '_root' ? '🖼 whole frame' : pc);
    // Reset selection if it dropped out of the window's data, then reflect it in the dropdown.
    const _val = list.includes(cur) ? cur : '';
    if (_val !== cur) localStorage.setItem('mve_heatmap_parent_class', '');
    sel.value = _val;
}
window._renderParentClassButtons = _renderParentClassButtons;

// 4.0.32 — fired by the chart's onClick when pick mode is on. Posts the
// click X (encoder count or epoch-ms) plus a ±5% window to the backend so
// the next chart load can compute the reference baseline from frames near
// that point. Exits pick mode + activates `reference_frame` automatically.
function _onBaselinePickClick(chart, evt, axisMode) {
    if (!chart || !chart.scales || !chart.scales.x) return;
    const xScale = chart.scales.x;
    let pixelX = (evt && (evt.x != null ? evt.x : evt.offsetX));
    if (typeof pixelX !== 'number') return;
    const xVal = xScale.getValueForPixel(pixelX);
    const range = (xScale.max - xScale.min) || 1;
    const win = Math.abs(range * 0.05);   // ±5% of visible range
    const axis = (axisMode === 'encoder') ? 'encoder' : 'time';
    fetch('/api/color_reference_position', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ axis, value: xVal, window: win }),
    }).then(r => r.json()).then(d => {
        window._pickingBaseline = false;
        const hint = document.getElementById('hm-pick-hint');
        if (hint) hint.style.display = 'none';
        const sc = document.getElementById('insight-camera-scatter');
        if (sc) sc.style.cursor = 'default';
        if (d && d.status === 'ok') {
            localStorage.setItem('mve_heatmap_baseline', 'reference_frame');
            if (typeof refreshDetectionInsights === 'function') refreshDetectionInsights();
            // 4.0.50 — unified reference-frames pool: the same clicked position
            // that just rebuilt the colour baseline ALSO rebuilds the anomaly
            // worker's baseline against the same frames. Non-blocking — the
            // colour heatmap doesn't have to wait on the anomaly memory-bank
            // build (which can take a few seconds for 50 frames).
            _kickAnomalyBaseline('reference_frame', {
                axis: axis, value: xVal, window_pct: 0.05,
            });
        } else {
            alert('Could not save reference: ' + (d && d.error || 'unknown error'));
        }
    }).catch(e => {
        alert('Network error setting reference: ' + (e.message || e));
        window._pickingBaseline = false;
    });
}

// 4.0.50 — Fire-and-forget anomaly baseline rebuild. Called from the colour
// baseline set paths (reference pick + shipment_start switch + camera-mode
// switch) so a single operator gesture drives both models. Deliberately
// non-blocking: the colour heatmap redraws immediately, the anomaly bank
// finishes building a beat later, and any error is surfaced only on the
// small status chip near the base-mode buttons (not as a blocking alert).
function _kickAnomalyBaseline(mode, extras) {
    try {
        const body = Object.assign({
            mode: mode,
            phase: (localStorage.getItem('mve_heatmap_phase') || '0'),
            camera_id: '_global',
        }, extras || {});
        fetch('/api/anomaly/build-baseline', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body),
        }).then(r => r.json()).then(d => {
            const chip = document.getElementById('hm-anomaly-baseline-chip');
            if (!chip) return;
            if (d && d.success) {
                chip.textContent = `anomaly baseline: ${mode} · ${d.n_frames} frames · just now`;
                chip.style.color = '#a7f3d0';
            } else if (d && d.skipped) {
                chip.textContent = `anomaly baseline: ${d.skipped}`;
                chip.style.color = '#94a3b8';
            } else {
                chip.textContent = `anomaly baseline: failed (${(d && d.error) || 'unknown'})`;
                chip.style.color = '#fca5a5';
            }
        }).catch(e => {
            const chip = document.getElementById('hm-anomaly-baseline-chip');
            if (chip) {
                chip.textContent = `anomaly baseline: network error`;
                chip.style.color = '#fca5a5';
            }
        });
    } catch (e) { /* never let anomaly plumbing break colour path */ }
}
window._kickAnomalyBaseline = _kickAnomalyBaseline;


// 3.21.16 — Quality drift / trend chart under the score card.
// Bucketed impact-per-unit timeline with verdict-threshold guide lines.
let _sqsTrendChart = null;
async function refreshShipmentQualityTrend() {
    const win  = document.getElementById('insight-window')?.value || '24h';
    const ship = document.getElementById('insight-shipment')?.value || '';
    const chipEl = document.getElementById('sqs-trend-chip');
    const noteEl = document.getElementById('sqs-trend-note');
    const canvas = document.getElementById('sqs-trend-chart');
    if (!canvas) return;
    let data;
    try {
        const params = new URLSearchParams({ window: win, buckets: '12' });
        if (ship) params.set('shipment', ship);
        const resp = await fetch(`/api/shipment_quality_score/trend?${params}`);
        data = await resp.json();
    } catch (e) { console.warn('trend fetch failed', e); return; }

    const buckets = data.buckets || [];
    if (!buckets.length) {
        if (chipEl) {
            chipEl.textContent = '— no data —';
            chipEl.style.background = 'rgba(71,85,105,0.5)';
            chipEl.style.color = 'var(--text-primary)';
        }
        if (_sqsTrendChart) { _sqsTrendChart.destroy(); _sqsTrendChart = null; }
        return;
    }

    // Slope chip
    const label = data.slope_label || 'stable';
    const pct = data.slope_pct || 0;
    if (chipEl) {
        let arrow = '→', bg = 'rgba(71,85,105,0.6)', fg = '#e2e8f0';
        if (label === 'degrading') { arrow = '↗'; bg = 'rgba(220,38,38,0.7)';  fg = '#fff'; }
        else if (label === 'improving') { arrow = '↘'; bg = 'rgba(16,185,129,0.7)'; fg = '#fff'; }
        const sign = pct > 0 ? '+' : '';
        chipEl.textContent = `${arrow} ${label} ${sign}${pct}%`;
        chipEl.style.background = bg;
        chipEl.style.color = fg;
    }
    if (noteEl) {
        const norm = data.normalized_by || 'none';
        const unit = data.encoder_unit || 'unit';
        const basisLbl = norm === 'encoder' ? `impact/${unit}` : (norm === 'frame' ? 'impact/frame' : 'impact (no normalization)');
        noteEl.textContent = `${basisLbl} per ${data.bucket_size || 'bucket'} — first-third vs last-third`;
    }

    // Chart.js line of impact_per_unit per bucket
    const labels = buckets.map(b => {
        try { return new Date(b.bucket).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); }
        catch (_) { return ''; }
    });
    const ipus = buckets.map(b => b.impact_per_unit ?? 0);

    if (typeof Chart === 'undefined') return;
    if (_sqsTrendChart) _sqsTrendChart.destroy();
    _sqsTrendChart = new Chart(canvas, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'impact/unit',
                data: ipus,
                borderColor: '#fbbf24',
                backgroundColor: 'rgba(251,191,36,0.15)',
                borderWidth: 2,
                fill: true,
                tension: 0.35,
                pointRadius: 2,
                pointHoverRadius: 5,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const b = buckets[ctx.dataIndex] || {};
                            return [
                                `impact/unit: ${(b.impact_per_unit ?? 0).toFixed(5)}`,
                                `score: ${b.score ?? '—'}`,
                                `dets: ${(b.detections ?? 0).toLocaleString()}`,
                            ];
                        },
                    },
                },
            },
            scales: {
                x: { ticks: { color: '#94a3b8', font: { size: 9 }, maxTicksLimit: 6 }, grid: { display: false } },
                y: { beginAtZero: true, ticks: { color: '#94a3b8', font: { size: 9 } }, grid: { color: 'rgba(148,163,184,0.08)' } },
            },
        },
    });
}
window.refreshShipmentQualityTrend = refreshShipmentQualityTrend;


// 4.0.53 — chart-tab-active check. Used to gate the "Loading charts…" badge
// so it never appears on Dashboard / Cameras / etc. even if a stale interval
// fires refreshDetectionInsights() or refreshAdvancedCharts() while the
// operator is looking at a different tab. The Charts tab container id is
// legacy "tab-grafana" — Charts tab .tab-content wrapper.
function _chartsTabActive() {
    const el = document.getElementById('tab-grafana');
    return !!(el && el.classList.contains('active'));
}

// 4.0.198 — shipment picker + timeframe visibility (operator spec: shipment
// dropdown first, timeframe disappears when a specific shipment is picked).
function _syncTimeframeVisibility() {
    const ship = (document.getElementById('insight-shipment') || {}).value || '';
    const grp  = document.getElementById('insight-timeframe-group');
    const clr  = document.getElementById('insight-shipment-clear');
    if (grp) grp.style.display = ship ? 'none' : 'inline-flex';
    if (clr) clr.style.display = ship ? 'inline-block' : 'none';
}
window._syncTimeframeVisibility = _syncTimeframeVisibility;

function _onShipmentPicked() {
    // Toggle the timeframe group, then reload. When a specific shipment is
    // chosen the backend override (v4.0.196) ignores the window entirely and
    // returns the shipment's full data; the axis (v4.0.198) spans its extent.
    _syncTimeframeVisibility();
    if (typeof refreshDetectionInsights === 'function') refreshDetectionInsights();
}
window._onShipmentPicked = _onShipmentPicked;

function _clearShipment() {
    const el = document.getElementById('insight-shipment');
    if (el) el.value = '';
    _syncTimeframeVisibility();
    if (typeof refreshDetectionInsights === 'function') refreshDetectionInsights();
}
window._clearShipment = _clearShipment;


async function refreshDetectionInsights() {
    // 4.0.53 — Two bugs fixed:
    //  (a) The umbrella function called mveLoaderBegin but never a matching
    //      mveLoaderEnd on the success path (falls off at line 1040), so
    //      _mveLoadInFlight climbed by 1 every refresh — spinner circled
    //      forever. Wrap the whole body in try/finally.
    //  (b) The loader is a fixed-position badge, so a fetch fired while the
    //      operator was on Charts persisted visually after they switched to
    //      Dashboard. Gate the loader with _chartsTabActive() so it only
    //      appears/hides in the Charts tab context (id="tab-grafana").
    const _showedLoader = _chartsTabActive();
    if (_showedLoader && typeof mveLoaderBegin === 'function') mveLoaderBegin('Loading charts…');
    try {
    const windowSel = document.getElementById('insight-window');
    const window = windowSel ? windowSel.value : '24h';
    const emptyEl = document.getElementById('insight-empty');
    const chartsEl = document.getElementById('insight-charts');
    const totalEl = document.getElementById('insight-total');

    // v4.0.131 — long-window mode. At 90d/120d/1y the scatter,
    // detection_stats and detection_charts queries against inference_results
    // scan 30-100 M rows and time out no matter what — 8 M rows in 7 d on
    // khoy already needs 15 s. Skip every heavy panel and only render the
    // Score-per-shipment card, which uses /api/quality/shipments (bounded
    // at 5000 rows). Show an operator-visible hint so the empty panels
    // aren't confusing.
    const _LONG_WINDOWS = { '90d': 1, '120d': 1, '1y': 1 };
    const _isLongWindow = !!_LONG_WINDOWS[window];
    const _longNote = document.getElementById('insight-long-window-note');
    if (_longNote) _longNote.style.display = _isLongWindow ? '' : 'none';

    if (_isLongWindow) {
        // Only run the Score-per-shipment fetch. Hide the panels that
        // would just spin/empty for a large window.
        if (chartsEl) chartsEl.style.display = 'none';
        if (emptyEl)  emptyEl.style.display = 'none';
        // 4.0.203 — Score-per-shipment is window-independent; load it ONCE
        // (not forced) so switching between long windows doesn't re-fetch it.
        try { if (typeof loadScorePerShipment === 'function') await loadScorePerShipment(false); }
        catch (e) { console.warn('Score-per-shipment (long window) failed:', e); }
        return;   // finally-block still fires the loader-end
    }

    await _loadShownClasses();  // so per-class charts honor the Show toggle
    refreshShipmentQualityScore();  // 3.21.13 — Phase 2 preview: score card
    refreshShipmentQualityTrend();  // 3.21.16 — Phase 2: drift / trend chart
    refreshEjectionCharts();    // independent of detection data (Store gates separately)
    refreshProductionCharts();  // production_metrics KPIs (own data source)
    // v4.0.210 — the quality + ejection strips are now OWNED by the progressive
    // ladder (refreshAdvancedCharts, below), which streams them from the SAME
    // /api/detection_charts endpoint as the scatter, newest→oldest. So we NO
    // LONGER call refreshQualityCharts() here (that fired an instant, separate-
    // API paint BEFORE the ladder → the "blink" + the misalignment the operator
    // complained about). The 60 s timer + tab-open still call it as a periodic
    // same-API refresh once the domain exists.
    // v4.0.210 — Score-per-shipment: load it here (once-guarded) so a page
    // reload that restores the Charts tab WITHOUT a genuine tab-button click
    // still populates the card (the click handler at DOMContentLoaded was the
    // only trigger before → empty card on reload). Gated on _chartsTabActive()
    // so a background refresh on another tab can't render into a hidden 0-size
    // canvas and latch the once-guard.
    try {
        if ((typeof _chartsTabActive !== 'function' || _chartsTabActive())
            && typeof loadScorePerShipment === 'function') {
            loadScorePerShipment(false);
        }
    } catch (_spe) { /* non-fatal */ }

    let data;
    try {
        const minConf = (parseFloat(document.getElementById('insight-min-conf')?.value || '0') / 100) || 0;
        const r = await fetch('/api/detection_stats?window=' + encodeURIComponent(window) + '&min_conf=' + minConf);
        data = await r.json();
    } catch (e) {
        console.error('detection_stats fetch failed:', e);
        return;  // finally-block hides the loader
    }

    const byClass = data.by_class || {};
    const timeline = data.timeline || [];
    const hasData = (data.total || 0) > 0 || timeline.length > 0;

    if (emptyEl) emptyEl.style.display = hasData ? 'none' : 'block';
    if (chartsEl) chartsEl.style.display = hasData ? 'grid' : 'none';
    if (totalEl) totalEl.textContent = hasData
        ? `Total detections in window: ${data.total} across ${Object.keys(byClass).length} class(es)`
        : '';

    if (!hasData) return;

    // ---- Class distribution (bar) — honor per-class Show toggle ----
    const classNames = Object.keys(byClass).filter(_isShown);
    const classCounts = classNames.map(k => byClass[k]);
    const colors = classNames.map((_, i) => _INSIGHT_PALETTE[i % _INSIGHT_PALETTE.length]);

    const classCtx = document.getElementById('insight-class-chart');
    if (classCtx) {
        if (_insightClassChart) _insightClassChart.destroy();
        _insightClassChart = new Chart(classCtx, {
            type: 'bar',
            data: {
                labels: classNames,
                datasets: [{ label: 'Detections', data: classCounts, backgroundColor: colors, borderRadius: 4 }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    title: { display: true, text: 'Detections by class — click a bar for images', color: '#cbd5e1', font: { size: 13 } }
                },
                onClick: (evt, els, chart) => {
                    if (els && els.length) {
                        const lab = chart.data.labels[els[0].index];
                        if (lab) openDefectDrawer(lab);
                    }
                },
                scales: {
                    x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { display: false } },
                    y: { beginAtZero: true, ticks: { color: '#94a3b8', precision: 0 }, grid: { color: 'rgba(148,163,184,0.1)' } }
                }
            }
        });
    }

    // ---- Class distribution (doughnut) — same Show-filtered data as the bar ----
    const pieCtx = document.getElementById('insight-class-pie');
    if (pieCtx) {
        if (_insightClassPie) _insightClassPie.destroy();
        _insightClassPie = new Chart(pieCtx, {
            type: 'doughnut',
            data: {
                labels: classNames,
                datasets: [{ data: classCounts, backgroundColor: colors, borderColor: '#1e293b', borderWidth: 2 }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, position: 'right', labels: { color: '#cbd5e1', font: { size: 10 }, boxWidth: 12 } },
                    title: { display: true, text: 'Detection distribution by class — click for images', color: '#cbd5e1', font: { size: 13 } },
                    tooltip: { callbacks: { label: (c) => {
                        const total = c.dataset.data.reduce((a, b) => a + b, 0) || 1;
                        return `${c.label}: ${c.raw} (${(c.raw / total * 100).toFixed(1)}%)`;
                    } } }
                },
                onClick: (evt, els, chart) => {
                    if (els && els.length) {
                        const lab = chart.data.labels[els[0].index];
                        if (lab) openDefectDrawer(lab);
                    }
                }
            }
        });
    }

    // ---- Detections over time (line) ----
    const tlCtx = document.getElementById('insight-timeline-chart');
    if (tlCtx) {
        if (_insightTimelineChart) _insightTimelineChart.destroy();
        _insightTimelineChart = new Chart(tlCtx, {
            type: 'line',
            data: {
                labels: timeline.map(p => p.t),
                datasets: [{
                    label: 'Detections', data: timeline.map(p => p.count),
                    borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.15)',
                    fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    title: { display: true, text: 'Detections over time', color: '#cbd5e1', font: { size: 13 } }
                },
                scales: {
                    x: { ticks: { color: '#94a3b8', font: { size: 9 }, maxTicksLimit: 8 }, grid: { display: false } },
                    y: { beginAtZero: true, ticks: { color: '#94a3b8', precision: 0 }, grid: { color: 'rgba(148,163,184,0.1)' } }
                }
            }
        });
    }

    // The three richer charts come from /api/detection_charts (size, confidence,
    // camera scatter) and honor the shipment filter. Fetched separately.
    refreshAdvancedCharts();
    } finally {
        // 4.0.53 — Always balance the mveLoaderBegin from the top so the
        // spinner can't circle forever if a fetch throws or an early-return
        // path is taken (empty result, missing DOM element, etc.).
        if (_showedLoader && typeof mveLoaderEnd === 'function') mveLoaderEnd();
    }
}

// ---------------------------------------------------------------------------
// Advanced charts: object size distribution, confidence over time, camera
// scatter — all from /api/detection_charts, scoped by window + shipment.
// ---------------------------------------------------------------------------
// 4.0.46 — global safety net. A single document-level mousemove listener
// hides the chart hover preview the instant the cursor is NOT directly over
// the scatter canvas. No debounce, no Chart.js coupling, no stale-event
// problem. Runs in addition to the canvas-level mouseleave + Chart.js
// external tooltip hide paths so even if one of those misses, this catches.
(function _wireGlobalHoverHide() {
    if (typeof document === 'undefined' || document._mveHoverHideWired) return;
    document._mveHoverHideWired = true;
    document.addEventListener('mousemove', (event) => {
        const preview = document.getElementById('chart-image-preview');
        if (!preview || preview.style.display === 'none') return;
        const scatter = document.getElementById('insight-camera-scatter');
        if (!scatter) return;
        const cursorEl = document.elementFromPoint(event.clientX, event.clientY);
        if (cursorEl !== scatter) {
            preview.style.display = 'none';
        }
    });
})();

// 4.0.42 — tiny reference-counted loading badge. Shows in the top-right of
// the viewport while ANY chart fetch is in flight. Multiple concurrent
// fetches stack via the counter so the badge stays visible until they all
// resolve. Self-injecting (no HTML changes needed). Style is a small dark
// pill with a CSS-only spinner so there's no extra network request just to
// show "loading".
let _mveLoadInFlight = 0;
function _ensureGlobalLoader() {
    if (document.getElementById('mve-global-loader')) return;
    const style = document.createElement('style');
    style.textContent = '@keyframes mve-spin{to{transform:rotate(360deg)}}';
    document.head.appendChild(style);
    const el = document.createElement('div');
    el.id = 'mve-global-loader';
    // 4.0.47 — much more visible. Pinned to TOP-CENTER, bigger text, bright
    // gradient background. Earlier corner badge was too small to notice on
    // a fast fetch.
    el.style.cssText = 'position:fixed;top:62px;left:50%;transform:translateX(-50%);'
        + 'z-index:999999;display:none;padding:8px 18px;'
        + 'background:linear-gradient(135deg,#1e3a8a,#3b82f6);'
        + 'border:1px solid rgba(96,165,250,0.7);border-radius:8px;'
        + 'color:#f1f5f9;font-size:13px;font-weight:600;'
        + 'align-items:center;gap:10px;'
        + 'box-shadow:0 8px 24px rgba(0,0,0,0.55)';
    el.innerHTML = '<span style="width:16px;height:16px;border:3px solid '
        + 'rgba(241,245,249,0.35);border-top-color:#f1f5f9;border-radius:50%;'
        + 'animation:spin 0.7s linear infinite;display:inline-block"></span>'
        + ' <span id="mve-global-loader-text">Loading…</span>';
    if (document.body) document.body.appendChild(el);
    else document.addEventListener('DOMContentLoaded', () => document.body.appendChild(el), { once: true });
}
function mveLoaderBegin(label) {
    _ensureGlobalLoader();
    _mveLoadInFlight++;
    const el = document.getElementById('mve-global-loader');
    if (el) {
        el.style.display = 'inline-flex';
        el._shownAt = Date.now();
        const txt = document.getElementById('mve-global-loader-text');
        if (txt && label) txt.textContent = String(label);
    }
}
function mveLoaderEnd() {
    _mveLoadInFlight = Math.max(0, _mveLoadInFlight - 1);
    if (_mveLoadInFlight === 0) {
        const el = document.getElementById('mve-global-loader');
        if (!el) return;
        // 4.0.47 — keep the badge visible for at least 1000ms so the operator
        // actually sees it even when the fetch completes in <100ms. Earlier
        // 400ms was too short to notice on cached fetches.
        const MIN_MS = 1000;
        const remaining = (el._shownAt) ? MIN_MS - (Date.now() - el._shownAt) : 0;
        const hide = () => { if (_mveLoadInFlight === 0) el.style.display = 'none'; };
        if (remaining > 0) setTimeout(hide, remaining); else hide();
    }
}
window.mveLoaderBegin = mveLoaderBegin;
window.mveLoaderEnd   = mveLoaderEnd;

// v4.0.74 — Progressive Camera × Encoder loading, bucket-by-bucket.
//
// Operator's mental model: dots appear once and STAY. New (older) dots are
// ADDED, one time-bucket at a time, starting from the most recent bucket and
// walking back to the oldest bucket in the selected window.
//
// Implementation:
//   1. Initial render: fetch `/api/detection_charts?window=X` where X is the
//      first (newest) bucket. The initial fetch is a normal
//      `_refreshAdvancedChartsCore` so it also paints size / confidence /
//      heatmap / ejection etc. against that first-bucket data, and hides
//      the "Loading charts…" spinner as soon as it finishes.
//   2. Progressive fill: iterate bucket-by-bucket going backward. Each
//      iteration fetches `/api/detection_charts?since_ms=X&until_ms=Y&scatter_only=1`
//      — the server short-circuits to only run the two scatter SQL queries
//      (skipping the expensive size/conf/heatmap/ejection queries that were
//      already rendered from the first bucket), and returns just
//      `camera_scatter` + `camera_scatter_encoder`. Client dedups against
//      keys already on the chart and appends only truly-new points via
//      Chart.js's `dataset.data.push()` + `chart.update('none')`. No loader
//      spinner, no destroy — the scatter grows continuously.
//   3. Bucket size = target window / bins (bins from Charts-tab local storage,
//      default 48). So 24h target with 48 bins → 30 minutes per bucket → 48
//      requests each ~200-500 ms → dots keep appearing for ~10-20 seconds
//      instead of the user staring at a spinner for 4-8 seconds and then
//      seeing everything at once.
//
// Skip the bucket-by-bucket ladder when the operator has already scoped the
// query narrowly (window === '1h' → no benefit; specific shipment → a single
// query is already narrow enough).
// v4.0.74 — Progressive Camera × Encoder loading, bucket-by-bucket.
// Initial render at '1h' paints all charts (size, confidence, scatter,
// heatmap, ejection). Then iterate wider buckets using server's
// since_ms/until_ms/scatter_only param path, appending only NEW points
// to the existing scatter chart via chart.update('none'). Skip iteration
// when the operator has narrowed the view (window='1h' or a specific
// shipment).
// v4.0.80 — real bucket-by-bucket scatter loader, aligned to the Quality
// strip's bucket count. Operator complaint on v4.0.74/75: "buckets not
// synced with Quality by Time / Ejection by Time — you invented 192 buckets
// out of nowhere". Fix: use the SAME `bins` value the operator sees under
// "Buckets:" (mve_bucket_count localStorage), iterate every one of those
// buckets from newest to oldest, so one bucket's worth of scatter dots
// lines up with one Quality/Ejection strip cell as it loads.
//
// Also implements the operator's second directive: "decouple anything that
// may stop UI functioning". This handler no longer calls
// _refreshAdvancedChartsCore (which fetched EVERY chart via one big
// /api/detection_charts call and destroyed all charts together). Instead:
//   - refreshQualityCharts() fires in parallel (Score per shipment,
//     Quality strip, Ejection strip — each has its own endpoint, their
//     own loader, their own render, independent failure modes).
//   - refreshDetectionInsights (Pareto bar, timeline) is untouched — it
//     fires from its own entrypoint via its own /api/detection_stats fetch.
//   - The scatter creates its own empty chart, mounts a scatter-scoped
//     spinner over ONLY the scatter canvas, and fills bucket by bucket.
//
// If /api/detection_charts hangs entirely, only the SCATTER shows the
// spinner. Score per shipment, insights, heatmap and ejection strips
// still render from their own fast endpoints.
let _progressiveLadderTicket = 0;
// v4.0.230 — race coalescing. `_ladderActive` is true only while a progressive
// load is in flight; `_lastLoadKey` identifies the view it is loading. Multiple
// uncoordinated triggers (tab-click, trainer-colour fetch, colour-check event +
// its re-apply, the 60s strip timer) used to each restart the whole ladder and
// wipe the strip accumulators mid-load → the blank/snap at end of load. These
// two let an identical re-fire fold into the running load, and gate the 60s
// strip refresher so it never wipes accumulators the ladder is still filling.
let _ladderActive = false;
let _lastLoadKey = '';

// 4.0.163 — progressive colour-cell cache. `refreshDetectionInsights` seeds
// this from the initial /api/detection_charts response's color_heatmap.cells;
// the per-bucket loader below merges slice cells into it (dedupe key =
// `${cam}|${bin}`) and forces a chart update after each merge so cells fill
// in newest-first alongside the scatter dots.
let _progressiveColorCells = new Map();       // key -> cell
let _progressiveColorRange = null;            // {axis, lo, hi, n_bins}

// v4.0.228 — the selected window's encoder span (union of shipment_spans'
// enc_min/enc_max), stashed on `window` by refreshAdvancedCharts for the encoder
// axis. ONE validated source for its three consumers: the scatter domain, the
// strip domain, and the stationary-line guard. Returns {lo,hi} or null.
function _encWinRange() {
    const r = window._mveEncWindowRange;
    return (r && Number.isFinite(r.lo) && Number.isFinite(r.hi) && r.hi > r.lo) ? r : null;
}

function _seedProgressiveColorCache(data, axisMode) {
    // 4.0.165 — the initial /api/detection_charts?window=1h call returns
    // color cells for a 1h SLICE but bin numbers 0..47 which the chart
    // plugin spreads across the FULL X-axis width. That painted the entire
    // chart at load and made per-bucket loading invisible. Fix: DO NOT
    // seed cells from the initial payload. Start empty and let the
    // ladder loop fill in newest→oldest via bin_ref bounds anchored to
    // the TARGET window (not the 1h hydrate).
    _progressiveColorCells = new Map();
    // v4.0.210 — reset the quality + ejection accumulators in lock-step with the
    // colour cache so a new generation (window/shipment change) starts clean.
    _progressiveQualityCells = new Map();
    _progressiveEjectionCells = new Map();
    _progressiveShipmentCells = new Map();
    const useEnc = (axisMode === 'encoder');
    const raw = useEnc ? (data.color_heatmap || {}) : (data.color_heatmap_time || {});
    const bins = Math.max(1, Number(raw.n_bins) || 48);
    // v4.0.210 — a PICKED shipment's payload IS the full span (slice mode), so
    // its color_heatmap[_time] bounds ARE the correct domain — use them directly
    // for BOTH axes. Only the all-shipments case needs the [now-target, now]
    // synthesis (its payload is just the 1h hydrate, not the full window).
    const _picked = ((document.getElementById('insight-shipment') || {}).value || '').trim();
    // Compute bin_ref for the FULL target window, not the 1h hydrate.
    let lo, hi;
    if (useEnc) {
        // v4.0.228 — for all-shipments, use the window's encoder span (stashed by
        // refreshAdvancedCharts from shipment_spans) so the strips cover the whole
        // window in lock-step with the scatter, not just the 1h hydrate bounds. A
        // picked shipment keeps its own payload bounds (raw.enc_min/max = its span).
        const _ewr = _picked ? null : _encWinRange();
        if (_ewr) { lo = _ewr.lo; hi = _ewr.hi; }
        else {
            lo = Number(raw.enc_min); hi = Number(raw.enc_max);
            // 4.0.276 — the 1h hydrate can be EMPTY (line idle in the last hour) or
            // narrow, so raw.enc_min/max come back NaN/undefined → a degenerate
            // [0,0] range that bins EVERY colour cell into one block. When that
            // happens, fall back to the FULL-WINDOW encoder extent from the range
            // probe (returns the real span even when the last hour has no data).
            if (!(hi > lo)) {
                const _sr = window._mveScatterRange;
                if (_sr && Number.isFinite(_sr.encLo) && Number.isFinite(_sr.encHi) && _sr.encHi > _sr.encLo) {
                    lo = _sr.encLo; hi = _sr.encHi;
                } else { lo = Number(lo) || 0; hi = Number(hi) || 0; }
            }
        }
    } else if (_picked && Number.isFinite(Number(raw.t_min)) && Number.isFinite(Number(raw.t_max))
               && Number(raw.t_max) > Number(raw.t_min)) {
        lo = Number(raw.t_min); hi = Number(raw.t_max);   // picked shipment span
    } else if (window._mveScatterRange && Number.isFinite(window._mveScatterRange.loMs)
               && Number.isFinite(window._mveScatterRange.hiMs)) {
        // 4.0.256 — align the strip binning to the SAME data extent the scatter uses
        // (range probe), so quality/ejection/colour cells land under their dots instead
        // of over [now-window, now] (which misaligns them, piling at the far-left bin).
        lo = window._mveScatterRange.loMs; hi = window._mveScatterRange.hiMs;
    } else {
        const winSel = document.getElementById('insight-window');
        const target = (winSel && winSel.value) || '24h';
        const targetMs = (typeof _windowToMs === 'function') ? _windowToMs(target) : 24 * 3600 * 1000;
        hi = Date.now();
        lo = hi - targetMs;
    }
    _progressiveColorRange = {
        axis: useEnc ? 'encoder' : 'time',
        lo: lo, hi: hi, n_bins: bins,
    };
}


// ===========================================================================
// v4.0.210 — QUALITY + EJECTION strips from the SAME API as the scatter.
// Operator (emphatic): "get the quality/color/ejection data from the EXACT
// SAME api that you get the scatter data, so you don't misalign." These strips
// now come from /api/detection_charts (the scatter's endpoint) — either the
// main-payload quality_heatmap[_time]/ejection_heatmap[_time] (single-fetch:
// picked shipment + periodic refresh) or the per-bucket quality_slice/
// ejection_slice streamed by the ladder (all-shipments, newest→oldest). Every
// path bins over the IDENTICAL [lo,hi]/N_BINS the colour heatmap uses, so a
// vertical guideline hits the same data in the scatter, colour, quality AND
// ejection layers. Accumulators mirror _progressiveColorCells.
let _progressiveQualityCells = new Map();    // bin -> {bucket,n,score,top_class}
let _progressiveEjectionCells = new Map();   // bin -> {bucket,n,top_procedure,by_procedure}
let _progressiveShipmentCells = new Map();   // v4.0.217 — bin -> {bucket, shipment, n}

// Label for strip cell `bin` derived PURELY from the bin index + the domain
// [lo,hi]/nbins — never from a per-endpoint label field, so every strip labels
// its cells over the exact same domain as the scatter axis.
function _binLabelForStrip(bin, nbins, axis, lo, hi) {
    lo = Number(lo); hi = Number(hi);
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || !(hi > lo)) return '';
    if (axis === 'time') {
        const t = lo + ((bin + 0.5) / nbins) * (hi - lo);
        const d = new Date(t);
        const span = hi - lo;
        return (span <= 26 * 3600 * 1000)
            ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            : d.toLocaleDateString([], { month: '2-digit', day: '2-digit' }) + ' ' +
              d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    const eL = Math.round(lo + (bin / nbins) * (hi - lo));
    const eR = Math.round(lo + ((bin + 1) / nbins) * (hi - lo)) - 1;
    return `${eL.toLocaleString()} – ${eR.toLocaleString()}`;
}

function _paintQualityStrip(cells, nbins, axis, lo, hi) {
    const strip  = document.getElementById('quality-strip');
    // v4.0.218 — the per-strip time/encoder axis under the quality strip is gone
    // (operator: "remove the 01:06 PM entirely — that legend is already above the
    // scatter"). Nuke it here too so a browser still holding a CACHED old status.html
    // (which the ?v= cache-buster on the JS can't refresh) also loses the stray row.
    const axisEl = document.getElementById('quality-strip-axis');
    if (axisEl) { try { axisEl.remove(); } catch (_e) {} }
    if (!strip) return;
    // v4.0.240 — show the ↑/↓ DELTA of the score vs the previous non-empty bucket, like
    // the colour-change strip (operator: "put the number of quality... goes up = red
    // worsening, down = green"). Cell tinted RED when the number goes UP, GREEN when it
    // goes DOWN; |Δ| sets the depth. Absolute score stays in the tooltip.
    let _prev = null, _maxAbs = 0;
    const _dl = new Array(nbins).fill(null);
    for (let i = 0; i < nbins; i++) {
        const c = cells[i];
        const ok = c && (Number(c.n) > 0 || c.top_class != null) && Number.isFinite(Number(c.score));
        if (!ok) { _prev = null; continue; }
        const s = Number(c.score);
        _dl[i] = (_prev === null) ? 0 : (s - _prev);
        if (Math.abs(_dl[i]) > _maxAbs) _maxAbs = Math.abs(_dl[i]);
        _prev = s;
    }
    const html = [];
    for (let i = 0; i < nbins; i++) {
        const c = cells[i], d = _dl[i];
        if (d === null) { html.push('<div style="flex:1; background:rgba(148,163,184,0.06);" title="no data in this bucket"></div>'); continue; }
        const a = (_maxAbs > 0 ? (0.2 + 0.7 * Math.abs(d) / _maxAbs) : 0.25).toFixed(2);
        const bg = d > 0 ? `rgba(239,68,68,${a})` : d < 0 ? `rgba(34,197,94,${a})` : 'rgba(148,163,184,0.15)';
        let label = '';
        if (Math.abs(d) >= 0.05) { const m = Math.abs(d); label = (d > 0 ? '↑' : '↓') + (m >= 10 ? m.toFixed(0) : m.toFixed(1)); }
        const top = (c && c.top_class) ? `Top: ${c.top_class}` : '';
        const tip = `${_binLabelForStrip(i, nbins, axis, lo, hi)}\nScore: ${c && c.score != null ? c.score : '—'}\nΔ vs prev: ${d >= 0 ? '+' : ''}${d.toFixed(1)}\nDetections: ${c ? (c.n || 0) : 0}\n${top}`.trim();
        html.push(`<div style="flex:1; background:${bg}; cursor:default; display:flex; align-items:center; justify-content:center; font-size:9px; line-height:1; font-weight:700; color:rgba(15,23,42,0.95); text-shadow:-1px -1px 0 rgba(255,255,255,.9),1px -1px 0 rgba(255,255,255,.9),-1px 1px 0 rgba(255,255,255,.9),1px 1px 0 rgba(255,255,255,.9); overflow:hidden; white-space:nowrap;" title="${tip.replace(/"/g, '&quot;')}">${label}</div>`);
    }
    strip.innerHTML = html.join('');
}

function _paintEjectionStrip(cells, nbins, axis, lo, hi) {
    const strip  = document.getElementById('ejection-strip');
    const legend = document.getElementById('ejection-strip-legend');
    if (!strip) return;
    // v4.0.240 — ↑/↓ DELTA of the ejection count vs the previous non-empty bucket
    // (operator: "ejection goes up or down"). Cell RED when ejections go UP (worse),
    // GREEN when they go DOWN; |Δ| sets the depth. Absolute count + breakdown in tooltip.
    let _prev = null, _maxAbs = 0;
    const _dl = new Array(nbins).fill(null);
    const totals = {};
    for (let i = 0; i < nbins; i++) {
        const c = cells[i];
        if (c && Number(c.n) > 0) Object.entries(c.by_procedure || {}).forEach(([n, k]) => { totals[n] = (totals[n] || 0) + k; });
        if (!c || !Number.isFinite(Number(c.n))) { _prev = null; continue; }
        const v = Number(c.n);
        _dl[i] = (_prev === null) ? 0 : (v - _prev);
        if (Math.abs(_dl[i]) > _maxAbs) _maxAbs = Math.abs(_dl[i]);
        _prev = v;
    }
    const html = [];
    for (let i = 0; i < nbins; i++) {
        const c = cells[i], d = _dl[i];
        if (d === null) { html.push('<div style="flex:1; background:transparent;" title="no data in this bucket"></div>'); continue; }
        const a = (_maxAbs > 0 ? (0.2 + 0.7 * Math.abs(d) / _maxAbs) : 0.25).toFixed(2);
        const bg = d > 0 ? `rgba(239,68,68,${a})` : d < 0 ? `rgba(34,197,94,${a})` : 'rgba(148,163,184,0.12)';
        let label = '';
        if (Math.abs(d) >= 1) label = (d > 0 ? '↑' : '↓') + Math.abs(d).toFixed(0);
        const parts = (c && Number(c.n) > 0) ? Object.entries(c.by_procedure || {}).sort((x, y) => y[1] - x[1]).map(([n, k]) => `${n}: ${k}`).join('\n') : '';
        const tip = `${_binLabelForStrip(i, nbins, axis, lo, hi)}\nEjections: ${c ? (c.n || 0) : 0}\nΔ vs prev: ${d >= 0 ? '+' : ''}${d}\n${parts}`.trim();
        html.push(`<div style="flex:1; background:${bg}; cursor:default; display:flex; align-items:center; justify-content:center; font-size:9px; line-height:1; font-weight:700; color:rgba(15,23,42,0.95); text-shadow:-1px -1px 0 rgba(255,255,255,.9),1px -1px 0 rgba(255,255,255,.9),-1px 1px 0 rgba(255,255,255,.9),1px 1px 0 rgba(255,255,255,.9); overflow:hidden; white-space:nowrap;" title="${tip.replace(/"/g, '&quot;')}">${label}</div>`);
    }
    strip.innerHTML = html.join('');
    if (legend) {
        legend.innerHTML = Object.entries(totals)
            .sort((a, b) => b[1] - a[1])
            .map(([n, k]) => `<span style="display:inline-flex; align-items:center; gap:3px;"><span style="display:inline-block; width:10px; height:10px; background:${_procColor(n)}; border-radius:2px;"></span>${n} (${k})</span>`)
            .join('');
    }
}

// v4.0.217 — shipment lane as a bin-strip, IDENTICAL model to the strips above:
// one flex:1 cell per bin, each cell's colour keyed to its bin's dominant
// shipment (from the backend, NOT the sparse dots). A shipment that owns bins
// K..K+3 hues them the same → one solid bar; every bin of a shipment shares its
// colour. Wide same-shipment runs get a centred ID overlay; every cell shows the
// id on hover. Streams newest→oldest with the other strips.
function _shipHue(id) {
    const s = String(id || ''); let h = 0;
    for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0;
    return ((h * 137.508) % 360 + 360) % 360;
}
// v4.0.221 — solid verdict colour for a shipment id (green/amber/red/grey),
// bright to match the quality strip. Returns null when the shipment's verdict
// isn't known yet (map not loaded / shipment outside the Score-per-shipment set)
// so the caller falls back to the golden-angle hue.
function _shipVerdictColor(id) {
    const v = ((window._mveShipmentVerdicts || {})[String(id)] || '').toUpperCase();
    if (v === 'RELEASE')    return '#22c55e';
    if (v === 'RE-INSPECT') return '#eab308';
    if (v === 'HOLD')       return '#ef4444';
    if (v === 'PENDING' || v === 'NO_DATA') return '#64748b';
    return null;
}
// v4.0.218 — strip-legend info icon that uses the app's OWN tooltip component
// (.info-tooltip-sm → global #global-tooltip popup, wired in app-core.js) instead
// of a native title= bubble. Operator: "why is this tooltip a different design
// than the others in the app, like the dashboard?" Now it matches everywhere.
function _stripTipIcon(desc) {
    const safe = String(desc == null ? '' : desc)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return '<span class="info-tooltip-sm">i<span class="info-tooltip-text">' + safe + '</span></span>';
}
// v4.0.222 — rich hover text for a shipment lane block. Operator: "hovering a
// shipment should show more than the name — length, start/end, score…". Pulls
// per-shipment META cached from /api/quality/shipments (score/verdict/length/
// detections/top-defects) and computes the encoder-or-time start→end from the
// bins the shipment occupies. Multi-line (native title honours \n).
// ── Shipment-lane hover: app-styled tooltip with the FULL per-shipment record ──
// v4.0.226 — operator: "when I hover a specific shipment I need ALL the data —
// the start, the end, the detections, the length — not just the shipment name."
// The data already lived in window._mveShipmentMeta (cached from
// /api/quality/shipments), but it was only wired to the native OS `title` bubble
// AND it depended on the Score-per-shipment card having loaded first, so before
// that card rendered the hover degraded to name-only. Now: (a) an app-styled
// floating card instead of the OS bubble, (b) content computed LIVE on hover so
// it always reflects the latest meta, and (c) an eager meta load so the record
// is present even before the Score card renders.
var _shipHoverCtx = null;      // {nbins, axis, lo, hi} of the last strip paint
var _shipRangeMap = {};        // ship id -> [firstBin, lastBin]
var _shipStripBound = false;   // one-time delegated-listener guard
var _shipMetaLoadedAt = 0;     // throttle stamp for the eager fetch

// Fire-and-forget fetch that fills _mveShipmentMeta independently of the
// Score-per-shipment card. Throttled to once / 20 s. The hover reads whatever is
// in the cache at hover time, so a late fill still helps a shipment already on
// screen (no strip repaint needed — content is computed on hover).
function _ensureShipmentMetaLoaded() {
    const now = Date.now();
    if (_shipMetaLoadedAt && (now - _shipMetaLoadedAt) < 20000) return;
    _shipMetaLoadedAt = now;
    try {
        fetch('/api/quality/shipments?n=60')  // v4.0.236 — windowless; backend escalates the scan (was window=all → timeout)
            .then((r) => r.json())
            .then((d) => {
                const rows = (d && d.shipments) || [];
                window._mveShipmentMeta = window._mveShipmentMeta || {};
                window._mveShipmentVerdicts = window._mveShipmentVerdicts || {};
                for (const s of rows) {
                    if (!s || s.shipment == null) continue;
                    const id = String(s.shipment);
                    if (s.verdict) window._mveShipmentVerdicts[id] = String(s.verdict);
                    window._mveShipmentMeta[id] = {
                        score: s.score, verdict: s.verdict,
                        encoder_span: s.encoder_span, encoder_unit: s.encoder_unit,
                        encoder_min: s.encoder_min, encoder_max: s.encoder_max,  // 4.0.263 — real encoder start/end for the tooltip
                        first_t: s.first_t, last_t: s.last_t,
                        rows: s.rows, top_defects: s.top_defects
                    };
                }
            })
            .catch(() => {});
    } catch (_e) {}
}

function _shipTooltipEl() {
    let el = document.getElementById('mve-ship-tooltip');
    if (!el) {
        el = document.createElement('div');
        el.id = 'mve-ship-tooltip';
        el.style.cssText = 'position:fixed; z-index:99999; pointer-events:none; display:none; '
            + 'max-width:300px; padding:8px 11px; border-radius:8px; font-size:11px; line-height:1.55; '
            + 'background:var(--card-bg,#0f172a); color:var(--text-color,#e2e8f0); '
            + 'border:1px solid var(--border-color,#334155); box-shadow:0 8px 24px rgba(0,0,0,0.5);';
        document.body.appendChild(el);
    }
    return el;
}

function _shipFmtTs(t) {
    try { return new Date(t).toLocaleString([], { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }); }
    catch (_e) { return String(t); }
}

// Build the tooltip body for a shipment id from the current meta + strip range.
// The bin-derived Start→End always shows (even with no meta yet); verdict / score
// / length / detections / top-defects fill in once the meta is present.
function _shipmentTooltipHTML(sh) {
    const id = String(sh);
    const m = (window._mveShipmentMeta || {})[id] || null;
    const ctx = _shipHoverCtx, rng = _shipRangeMap[id];
    const esc = (t) => String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const out = ['<div style="font-weight:700; margin-bottom:5px; word-break:break-all;">' + esc(id) + '</div>'];
    const line = (label, valHtml) => {
        out.push('<div style="display:flex; gap:12px; justify-content:space-between;">'
            + '<span style="opacity:.6;">' + esc(label) + '</span>'
            + '<span style="text-align:right;">' + valHtml + '</span></div>');
    };
    if (m && (m.verdict != null || m.score != null)) {
        const vc = (typeof _shipVerdictColor === 'function' && _shipVerdictColor(id)) || '#94a3b8';
        const badge = m.verdict
            ? '<span style="display:inline-block; padding:0 6px; border-radius:6px; background:' + vc
                + '; color:#fff; font-weight:700;">' + esc(m.verdict) + '</span>'
            : '—';
        const sc = (m.score != null && isFinite(m.score)) ? ('&nbsp; score ' + esc(Number(m.score).toFixed(1))) : '';
        line('Verdict', badge + sc);
    }
    if (m && m.encoder_span != null && isFinite(m.encoder_span)) {
        // 4.0.263 — DIVIDE the raw encoder span by encoder-units-per-unit (e.g. 1000) so
        // the length is real meters, not raw counts (operator: "the length is not true
        // considering the encoder number of each unit"). _fmtLenWithUpu does the divide +
        // unit label; falls back to raw counts (tagged) when the line isn't calibrated.
        const _upu = _upuFromAny(m, null);
        const _unit = m.encoder_unit || _unitLabelFromAny(m, null);
        line('Length', esc((_upu > 0)
            ? _fmtLenWithUpu(m.encoder_span, _upu, _unit)
            : (Math.round(m.encoder_span).toLocaleString() + ' ' + _unit + ' (raw)')));
    }
    // Start → End (TIME): prefer the shipment's actual timestamps; else derive from bins.
    if (m && m.first_t != null && m.last_t != null) {
        line('Start → End', esc(_shipFmtTs(m.first_t) + '  →  ' + _shipFmtTs(m.last_t)));
    } else if (ctx && rng && ctx.axis === 'time' && isFinite(ctx.lo) && isFinite(ctx.hi) && ctx.hi > ctx.lo && ctx.nbins > 0) {
        const sv = ctx.lo + (rng[0] / ctx.nbins) * (ctx.hi - ctx.lo);
        const ev = ctx.lo + ((rng[1] + 1) / ctx.nbins) * (ctx.hi - ctx.lo);
        line('Start → End', esc(_shipFmtTs(sv) + '  →  ' + _shipFmtTs(ev)));
    }
    // 4.0.263 — ALSO show the ENCODER start → end alongside the time (operator: "I need
    // the start/stop of encoder like time"). Prefer the shipment's real encoder_min/max;
    // else derive from the strip's encoder bins.
    {
        let _encS = null, _encE = null;
        if (m && isFinite(Number(m.encoder_min)) && isFinite(Number(m.encoder_max))) {
            _encS = Number(m.encoder_min); _encE = Number(m.encoder_max);
        } else if (ctx && rng && ctx.axis !== 'time' && isFinite(ctx.lo) && isFinite(ctx.hi) && ctx.hi > ctx.lo && ctx.nbins > 0) {
            _encS = ctx.lo + (rng[0] / ctx.nbins) * (ctx.hi - ctx.lo);
            _encE = ctx.lo + ((rng[1] + 1) / ctx.nbins) * (ctx.hi - ctx.lo);
        }
        if (_encS != null && _encE != null) {
            line('Encoder', esc(Math.round(_encS).toLocaleString() + '  →  ' + Math.round(_encE).toLocaleString()));
        }
    }
    if (m && m.rows != null && isFinite(m.rows)) line('Detections', esc(Number(m.rows).toLocaleString()));
    if (m && Array.isArray(m.top_defects) && m.top_defects.length) {
        line('Top defects', esc(m.top_defects.slice(0, 3).join(', ')));
    }
    if (!m) out.push('<div style="opacity:.5; margin-top:3px;">loading full record…</div>');
    return out.join('');
}

// One-time delegated hover on the shipment strip. Delegation survives the strip's
// frequent innerHTML repaints (progressive ladder) — the listener lives on the
// container, not the cells. Cells carry data-ship; the ID-overlay labels are
// pointer-events:none so the hover falls through to the coloured cell beneath.
function _bindShipmentStripHover() {
    if (_shipStripBound) return;
    const strip = document.getElementById('shipment-strip');
    if (!strip) return;
    _shipStripBound = true;
    const tip = _shipTooltipEl();
    const place = (e) => {
        const pad = 14, w = tip.offsetWidth || 220, h = tip.offsetHeight || 90;
        let x = e.clientX + pad, y = e.clientY + pad;
        if (x + w > window.innerWidth - 8) x = e.clientX - w - pad;
        if (y + h > window.innerHeight - 8) y = e.clientY - h - pad;
        tip.style.left = Math.max(4, x) + 'px';
        tip.style.top = Math.max(4, y) + 'px';
    };
    strip.addEventListener('mousemove', (e) => {
        const sh = (e.target && e.target.getAttribute) ? e.target.getAttribute('data-ship') : null;
        if (!sh) { tip.style.display = 'none'; return; }
        tip.innerHTML = _shipmentTooltipHTML(sh);
        tip.style.display = 'block';
        place(e);
    });
    strip.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
    // v4.0.242 — click a shipment-lane cell to SELECT that shipment (operator: "when I
    // click a shipment in the lane it doesn't select it"). Reads the cell's data-ship,
    // fills the Shipment picker, and runs the picked-shipment reload.
    strip.style.cursor = 'pointer';
    strip.addEventListener('click', (e) => {
        const sh = (e.target && e.target.getAttribute) ? e.target.getAttribute('data-ship') : null;
        if (!sh) return;
        const inp = document.getElementById('insight-shipment');
        if (!inp) return;
        inp.value = sh;
        try { tip.style.display = 'none'; } catch (_) {}
        if (typeof _onShipmentPicked === 'function') _onShipmentPicked();
    });
}
function _paintShipmentStrip(cells, nbins, axis, lo, hi) {
    const strip = document.getElementById('shipment-strip');
    if (!strip) return;
    strip.style.position = 'relative';
    strip.style.overflow = 'hidden';
    strip.style.display = 'flex';
    if (!strip.style.height) strip.style.height = '16px';
    const owner = [];
    for (let i = 0; i < nbins; i++) {
        const c = cells[i];
        owner.push((c && c.shipment) ? String(c.shipment) : null);
    }
    // v4.0.222 — first/last bin each shipment occupies, for the hover start→end.
    const shRange = {};
    for (let i = 0; i < nbins; i++) {
        const s = owner[i];
        if (!s) continue;
        if (!shRange[s]) shRange[s] = [i, i];
        else if (i > shRange[s][1]) shRange[s][1] = i;
    }
    // v4.0.226 — publish the paint context + per-shipment bin range for the
    // live hover tooltip, and make sure the full per-shipment meta is loading.
    _shipHoverCtx = { nbins: nbins, axis: axis, lo: lo, hi: hi };
    _shipRangeMap = shRange;
    _ensureShipmentMetaLoaded();
    const _escAttr = (t) => String(t).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
    const cellHtml = [];
    for (let i = 0; i < nbins; i++) {
        const sh = owner[i];
        if (!sh) { cellHtml.push('<div style="flex:1 1 0; height:100%; background:transparent;"></div>'); continue; }
        // v4.0.221 — colour by the shipment's VERDICT (release/re-inspect/hold/
        // pending); fall back to a per-id hue while the verdict map is still
        // loading or for a shipment outside the Score-per-shipment set.
        const vc = _shipVerdictColor(sh);
        const bg = vc || ('hsl(' + _shipHue(sh).toFixed(0) + ', 62%, 48%)');
        // With verdict colours, two adjacent DIFFERENT shipments can share a
        // colour (e.g. both RELEASE→green). Draw a thin divider at a shipment
        // boundary so the operator can still see where one ends and the next
        // begins. A shipment's OWN width stays a single flat block.
        const boundary = (i + 1 < nbins) && owner[i + 1] && owner[i + 1] !== sh;
        const sep = boundary ? ' box-shadow: inset -1px 0 0 rgba(2,6,23,0.55);' : '';
        // v4.0.226 — carry the id on the cell; the app-styled tooltip is rendered
        // live on hover from _shipmentTooltipHTML (replaces the native title=).
        cellHtml.push('<div data-ship="' + _escAttr(String(sh)) + '" style="flex:1 1 0; height:100%; background:' + bg + ';' + sep + '"></div>');
    }
    // Centred ID overlay on same-shipment runs ≥3 bins wide.
    const labelHtml = [];
    let b = 0;
    while (b < nbins) {
        const sh = owner[b];
        let e = b; while (e < nbins && owner[e] === sh) e++;
        if (sh && (e - b) >= 3) {
            const leftPct = (b / nbins) * 100, widthPct = ((e - b) / nbins) * 100;
            labelHtml.push('<div class="mve-ship-bar" data-w-pct="' + widthPct + '" '
                + 'data-label-full="' + sh.replace(/"/g, '&quot;') + '" '
                + 'data-label-med="' + (sh.length > 8 ? sh.slice(-8) : sh) + '" '
                + 'data-label-tail="' + (sh.length > 4 ? sh.slice(-4) : sh) + '" '
                + 'style="position:absolute; top:0; height:100%; left:' + leftPct + '%; width:' + widthPct + '%; '
                + 'display:flex; align-items:center; justify-content:center; font-size:9px; font-weight:700; color:#fff; '
                + 'text-shadow:0 1px 2px rgba(0,0,0,0.75); pointer-events:none; overflow:hidden; white-space:nowrap;">'
                + '<span class="mve-ship-label" style="padding:0 3px; overflow:hidden; text-overflow:ellipsis;">' + sh + '</span></div>');
        }
        b = e;
    }
    strip.innerHTML = cellHtml.join('') + labelHtml.join('');
    if (typeof _relayoutShipmentStripLabels === 'function') { try { _relayoutShipmentStripLabels(); } catch (_e) {} }
    _bindShipmentStripHover();  // v4.0.226 — idempotent; delegation survives repaints
}

// Rebuild the dense cell arrays from the streamed accumulators + repaint. Uses
// _progressiveColorRange as the single source of the domain (identical to the
// colour heatmap), so quality/ejection cell K == colour cell K == scatter bin K.
function _repaintProgressiveStrips() {
    const rng = _progressiveColorRange;
    if (!rng || !(rng.hi > rng.lo)) return;
    const nbins = Math.max(1, Number(rng.n_bins) || 48);
    const axis = rng.axis === 'time' ? 'time' : 'encoder';
    // v4.0.234 — re-bin every accumulated cell by its ABSOLUTE encoder centre over
    // the CURRENT (auto-widening) domain, so a cell captured when the axis was
    // narrower lands in the correct slice as the axis grows — keeping the strips
    // aligned with the scatter dots bucket-by-bucket. Cells with no `_enc` (the
    // picked-shipment single-fetch path) fall back to their raw bin index.
    const qArr = new Array(nbins).fill(null), eArr = new Array(nbins).fill(null), sArr = new Array(nbins).fill(null);
    const _binOf = (enc) => { const b = Math.floor((enc - rng.lo) / (rng.hi - rng.lo) * nbins); return (b >= 0 && b < nbins) ? b : -1; };
    const _place = (c, arr) => {
        let b = -1;
        if (c && typeof c._enc === 'number') b = _binOf(c._enc);
        else if (c && Number.isFinite(Number(c.bucket))) b = Number(c.bucket);
        if (b >= 0 && b < nbins) arr[b] = c;
    };
    for (const c of _progressiveQualityCells.values()) _place(c, qArr);
    for (const c of _progressiveEjectionCells.values()) _place(c, eArr);
    for (const c of _progressiveShipmentCells.values()) _place(c, sArr);
    _paintQualityStrip(qArr, nbins, axis, rng.lo, rng.hi);
    _paintEjectionStrip(eArr, nbins, axis, rng.lo, rng.hi);
    _paintShipmentStrip(sArr, nbins, axis, rng.lo, rng.hi);
    // v4.0.211 — the COLOUR-CHANGE strip (↑/↓ Δ-between-buckets) used to be
    // painted from Core's 1h-hydrate payload → 1h of data spread across the full
    // 24h width, out of sync with the scatter. Recompute it here from the SAME
    // streamed colour cells the heatmap uses, so it fills newest→oldest and its
    // bin K covers the exact same x-range as scatter bin K. (Keeps the ↑/↓
    // number labels — _renderColorChangeStrip draws them.)
    try { _recomputeColorChangeFromProgressive(); } catch (_cce) {}
}

// v4.0.211 — rebuild the colour-change strip cells from _progressiveColorCells
// (per-(cam,bin) E), binned over the streamed domain, so it stays x-synced with
// the scatter/heatmap. Only for the all-shipments STREAMED colour case; a picked
// shipment keeps Core's span-scoped color_change_strip (already aligned).
function _recomputeColorChangeFromProgressive() {
    if (window._mveColorCheckEnabled !== true) return;
    // v4.0.213 — read from the SAME colour cells the heatmap actually paints
    // (__mveHeatmap.cells). For all-shipments that's the streamed set; for a
    // PICKED shipment that's Core's full-span set (colour isn't streamed for a
    // pick). The old version skipped picked shipments entirely, so their
    // colour-change strip lost its ↑/↓ numbers — that's the "numbers disappear
    // when I choose a shipment" bug. One source now covers both.
    const hm = _insightCameraScatter && _insightCameraScatter.data && _insightCameraScatter.data.__mveHeatmap;
    const srcCells = (hm && Array.isArray(hm.cells)) ? hm.cells : [];
    if (!srcCells.length) return;
    const rng = _progressiveColorRange;
    if (!rng || !(rng.hi > rng.lo)) return;
    const nbins = Math.max(1, Number(rng.n_bins) || 48);
    const byBin = new Map();
    for (const c of srcCells) {
        const b = Number(c.bin); if (!Number.isFinite(b)) continue;
        const e = Number(c.E) || 0, n = Number(c.n) || 0;
        if (!byBin.has(b)) byBin.set(b, { sumEw: 0, n: 0 });
        const o = byBin.get(b); o.sumEw += e * n; o.n += n;
    }
    let prevE = null;
    const cells = [];
    for (let b = 0; b < nbins; b++) {
        const o = byBin.get(b);
        if (!o || o.n <= 0) { prevE = null; continue; }  // gap → break the Δ chain
        const mE = o.sumEw / o.n;
        const d = (prevE === null) ? 0 : (mE - prevE);
        cells.push({ bin: b, E: Math.round(mE * 100) / 100, delta_prev: Math.round(d * 100) / 100, n: o.n });
        prevE = mE;
    }
    const axisMode = rng.axis === 'time' ? 'time' : 'encoder';
    const synthetic = { parent_color_check_enabled: true };
    if (axisMode === 'time') synthetic.color_change_strip_time = { n_bins: nbins, cells };
    else                      synthetic.color_change_strip      = { n_bins: nbins, cells };
    _renderColorChangeStrip(axisMode, synthetic);
}

// Render the strips DIRECTLY from a full detection_charts payload's main-payload
// heatmaps (single-fetch path: picked shipment + periodic refresh). Same bounds
// + N_BINS as data.color_heatmap → guaranteed aligned.
function _renderStripsFromPayload(data, axis) {
    if (!data) return;
    const isEnc = (axis === 'encoder');
    const q = isEnc ? (data.quality_heatmap || {})       : (data.quality_heatmap_time || {});
    const e = isEnc ? (data.ejection_heatmap || {})      : (data.ejection_heatmap_time || {});
    const s = isEnc ? (data.shipment_heatmap || {})      : (data.shipment_heatmap_time || {});
    const qBins = Math.max(1, Number(q.n_bins) || 48);
    const eBins = Math.max(1, Number(e.n_bins) || qBins);
    const sBins = Math.max(1, Number(s.n_bins) || qBins);
    const qLo = isEnc ? q.enc_min : q.t_min, qHi = isEnc ? q.enc_max : q.t_max;
    const eLo = isEnc ? e.enc_min : e.t_min, eHi = isEnc ? e.enc_max : e.t_max;
    const sLo = isEnc ? s.enc_min : s.t_min, sHi = isEnc ? s.enc_max : s.t_max;
    // Seed the accumulators too so a subsequent per-bucket repaint keeps them.
    _progressiveQualityCells = new Map();
    _progressiveEjectionCells = new Map();
    _progressiveShipmentCells = new Map();
    for (const c of (q.cells || [])) if (Number(c.n) > 0 || c.top_class != null) _progressiveQualityCells.set(Number(c.bucket), c);
    for (const c of (e.cells || [])) if (Number(c.n) > 0) _progressiveEjectionCells.set(Number(c.bucket), c);
    for (const c of (s.cells || [])) if (c.shipment != null) _progressiveShipmentCells.set(Number(c.bucket), c);
    if (Array.isArray(q.cells) && q.cells.length) _paintQualityStrip(q.cells, qBins, axis, qLo, qHi);
    if (Array.isArray(e.cells) && e.cells.length) _paintEjectionStrip(e.cells, eBins, axis, eLo, eHi);
    if (Array.isArray(s.cells) && s.cells.length) _paintShipmentStrip(s.cells, sBins, axis, sLo, sHi);
}
window._renderStripsFromPayload = _renderStripsFromPayload;

// Per-bucket streamer for the quality + ejection strips — and folds in colour
// so ONE fetch fills all three (this superseded the old colour-only streamer).
// Called newest→oldest by the ladder AND once over the full domain by the
// periodic refresh. bin_ref comes from _progressiveColorRange (same domain the
// colour heatmap + scatter axis use).
async function _extendStripsFromBucket(sinceMs, untilMs, shipment, ticket, preData, binRef) {
    // v4.0.300 — binRef (when supplied by the merged one-call path) is the STABLE
    // range the caller already used for this fetch's bin_ref, so _encOf below
    // recovers the same absolute encoder the backend binned against.
    const rng = binRef || _progressiveColorRange;
    if (!rng || !(rng.hi > rng.lo)) return;
    const colorOn = (window._mveColorCheckEnabled === true) && !shipment;
    const phase = localStorage.getItem('mve_heatmap_phase') || '';
    const parentClass = localStorage.getItem('mve_heatmap_parent_class') || '';
    const url = '/api/detection_charts?since_ms=' + Math.floor(sinceMs) +
                '&until_ms=' + Math.floor(untilMs) +
                '&shipment=' + encodeURIComponent(shipment || '') +
                '&scatter_only=1&include_strips=1' +
                (colorOn ? '&include_color_slice=1' : '') +
                '&color_axis=' + encodeURIComponent(rng.axis) +
                '&bin_ref_lo=' + rng.lo + '&bin_ref_hi=' + rng.hi +
                '&bins=' + rng.n_bins +
                '&phase=' + encodeURIComponent(phase) +
                '&parent_class=' + encodeURIComponent(parentClass);
    // v4.0.299 — retry on transient connection resets, SAME as _extendScatterFromBucket.
    // This strips+colour fetch previously had ONE try and a silent `return` on reset, so
    // over the operator's reverse-SSH tunnel a dropped bucket lost its colour slice for
    // good — the dots (which already retry) appeared, but the colour heatmap + the
    // colour-change strip never did. Matching the scatter's 3-try backoff removes that
    // asymmetry: a transient reset no longer wipes a bucket's colour.
    // v4.0.300 — skip the fetch when the merged one-call path already fetched this
    // bucket and passed the payload in via preData.
    let data = preData || null;
    if (!data) {
        for (let _try = 0; _try < 3; _try++) {
            try { const r = await fetch(url); data = await r.json(); break; }
            catch (e) {
                if (ticket != null && ticket !== _progressiveLadderTicket) return; // superseded
                if (_try === 2) { console.warn('[progressive] strips/colour fetch failed after 3 tries:', e); return; }
                await new Promise(res => setTimeout(res, 300 * (_try + 1)));
            }
        }
    }
    if (!data) return;
    if (ticket != null && ticket !== _progressiveLadderTicket) return;
    // v4.0.234 — tag each cell with its ABSOLUTE encoder centre (from the bin_ref
    // THIS fetch used) and key by it, so _repaintProgressiveStrips can re-bin over
    // the auto-widening domain and a later bucket at the same position dedups.
    // 4.0.257 — GUARD a degenerate range (rng.hi == rng.lo). On a stationary line the
    // encoder is a single value, so the original _encOf collapsed EVERY bucket to the
    // same key → all strip cells piled into bin 0 (operator: "you keep updating the
    // leftmost bucket again and again"). When degenerate, key by the raw bin INDEX
    // (distinct per bucket) exactly like the pre-v4.0.234 code that worked.
    const _degen = !(rng.hi > rng.lo);
    const _encOf = (b) => _degen ? Number(b) : (rng.lo + ((Number(b) + 0.5) / rng.n_bins) * (rng.hi - rng.lo));
    const _reBin = (enc) => {
        const r = _progressiveColorRange;
        if (!(r.hi > r.lo)) return Math.max(0, Math.min(r.n_bins - 1, Math.round(enc)));
        const b = Math.floor((enc - r.lo) / (r.hi - r.lo) * r.n_bins);
        return Math.max(0, Math.min(r.n_bins - 1, b));
    };
    // Colour (only when streaming it — picked shipment keeps Core's full set).
    if (colorOn && data.color_slice && Array.isArray(data.color_slice.cells)) {
        for (const c of data.color_slice.cells) { c._enc = _encOf(c.bin); _progressiveColorCells.set(`${c.cam}|${Math.round(c._enc)}`, c); }
        if (_insightCameraScatter && _insightCameraScatter.data && _insightCameraScatter.data.__mveHeatmap) {
            _insightCameraScatter.data.__mveHeatmap.cells = Array.from(_progressiveColorCells.values())
                .map(c => (typeof c._enc === 'number') ? Object.assign({}, c, { bin: _reBin(c._enc) }) : c);
            try { _insightCameraScatter.update('none'); } catch (_e) {}
        }
    }
    // Quality / ejection / shipment: merge only non-empty bins (an out-of-slice
    // empty cell can't wipe a bin an earlier bucket already filled).
    const qc = (data.quality_slice && Array.isArray(data.quality_slice.cells)) ? data.quality_slice.cells : [];
    for (const c of qc) if (Number(c.n) > 0 || c.top_class != null) {
        c._enc = _encOf(c.bucket); _progressiveQualityCells.set(Math.round(c._enc), c);
    }
    const ec = (data.ejection_slice && Array.isArray(data.ejection_slice.cells)) ? data.ejection_slice.cells : [];
    for (const c of ec) if (Number(c.n) > 0) { c._enc = _encOf(c.bucket); _progressiveEjectionCells.set(Math.round(c._enc), c); }
    const sc = (data.shipment_slice && Array.isArray(data.shipment_slice.cells)) ? data.shipment_slice.cells : [];
    for (const c of sc) if (c.shipment != null) { c._enc = _encOf(c.bucket); _progressiveShipmentCells.set(Math.round(c._enc), c); }
    _repaintProgressiveStrips();
}
window._extendStripsFromBucket = _extendStripsFromBucket;

// v4.0.300 — ONE call per bucket (operator's design). Keep bucketing — per-bucket
// dot_cap is what gives EVEN category spread; a single global fetch would only
// return the newest dots (clustered). But each bucket now makes a SINGLE request
// that returns ALL of that bucket's data — dots + colour cell + quality + ejection
// + shipment lane — instead of the old two (scatter, then strips/colour). Wins:
//   • Half the requests (N instead of 2N) — fewer tunnel round-trips to reset.
//   • ATOMIC: dots and colour share one fetch + one retry, so a bucket can never
//     again land "dots but no colour" (the reset-drop bug 4.0.299 only softened).
//
// Binning: the colour/strip slices are binned server-side over bin_ref_lo/hi. The
// scatter axis AUTO-WIDENS as older buckets stream, so it can't be the bin ref up
// front (this bucket's low-encoder territory isn't in it yet — that dependency is
// the whole reason it used to be two calls). We bin instead over the STABLE full-
// window extent from the range probe (window._mveEncWindowRange); _enc tagging uses
// the same ref, and _repaintProgressiveStrips re-bins by absolute _enc over the
// current display domain — so positions stay correct no matter the load order.
// Time axis is already pinned to [loMs,hiMs] up front, so its ref is the range as-is.
async function _extendBucketAll(sinceMs, untilMs, shipment, minConf, ticket) {
    if (!_insightCameraScatter) return;
    const colorOn = (window._mveColorCheckEnabled === true) && !shipment;
    const rng = _progressiveColorRange;
    const axisIsEncoder = !!(rng && rng.axis !== 'time');
    // Stable bin ref for the slices. Only for the ALL-shipments encoder window (a
    // picked shipment keeps its Core-seeded, span-scoped _progressiveColorRange;
    // _mveEncWindowRange would be stale from a prior all-shipments load).
    let stripRef = rng;
    if (!shipment && axisIsEncoder && window._mveEncWindowRange
        && Number(window._mveEncWindowRange.hi) > Number(window._mveEncWindowRange.lo)) {
        stripRef = { axis: 'encoder',
                     lo: Number(window._mveEncWindowRange.lo),
                     hi: Number(window._mveEncWindowRange.hi),
                     n_bins: (rng && rng.n_bins) || (parseInt(localStorage.getItem('mve_bucket_count') || '48', 10) || 48) };
    }
    const phase = localStorage.getItem('mve_heatmap_phase') || '';
    const parentClass = localStorage.getItem('mve_heatmap_parent_class') || '';
    const _fetchCap = (window._mveScatterFetchCap || _dotCap());
    let url = '/api/detection_charts?since_ms=' + Math.floor(sinceMs) +
              '&until_ms=' + Math.floor(untilMs) +
              '&shipment=' + encodeURIComponent(shipment || '') +
              '&min_conf=' + minConf +
              '&scatter_only=1&dot_cap=' + _fetchCap +
              '&include_strips=1';
    const _refOk = stripRef && Number(stripRef.hi) > Number(stripRef.lo);
    if (_refOk) {
        // strips (quality/ejection/shipment) always need the bin ref; colour is added only when on.
        if (colorOn) url += '&include_color_slice=1';
        url += '&color_axis=' + encodeURIComponent(stripRef.axis || 'encoder') +
               '&bin_ref_lo=' + stripRef.lo + '&bin_ref_hi=' + stripRef.hi +
               '&bins=' + stripRef.n_bins;
    }
    url += '&phase=' + encodeURIComponent(phase) + '&parent_class=' + encodeURIComponent(parentClass);
    // ONE fetch, 3-try backoff — everything for this bucket shares its fate.
    let data = null;
    for (let _try = 0; _try < 3; _try++) {
        try { const r = await fetch(url); data = await r.json(); break; }
        catch (e) {
            if (ticket != null && ticket !== _progressiveLadderTicket) return;
            if (_try === 2) { console.warn('[progressive] bucket fetch failed after 3 tries:', e); return; }
            await new Promise(res => setTimeout(res, 300 * (_try + 1)));
        }
    }
    if (!data) return;
    if (ticket != null && ticket !== _progressiveLadderTicket) return;
    // Dots FIRST (auto-widens the display axis), then strips + colour from the SAME
    // payload, binned over the stable stripRef.
    try { await _extendScatterFromBucket(sinceMs, untilMs, shipment, minConf, ticket, data); }
    catch (_se) { /* dot apply non-fatal */ }
    try { await _extendStripsFromBucket(sinceMs, untilMs, shipment, ticket, data, _refOk ? stripRef : undefined); }
    catch (_pe) { /* strip apply non-fatal */ }
}
window._extendBucketAll = _extendBucketAll;

async function refreshAdvancedCharts() {
    const winSel = document.getElementById('insight-window');
    const target = (winSel && winSel.value) || '24h';
    const shipment = document.getElementById('insight-shipment')?.value || '';
    const minConf = (parseFloat(document.getElementById('insight-min-conf')?.value || '0') / 100) || 0;

    // v4.0.230 — RACE COALESCE. If a ladder for the SAME view (window, shipment,
    // axis, bins, colour-check) is already streaming, do NOT tear it down and
    // restart it — the redundant re-fire (trainer-colour fetch, colour-check
    // re-apply, a second tab-click) folds into the running load. A genuine change
    // has a different key and still restarts. This kills the end-of-load blank/snap
    // caused by a late trigger rebuilding the chart just as the first load finishes.
    const _loadKey = target + '|' + shipment + '|' + _insightAxis + '|'
        + (localStorage.getItem('mve_bucket_count') || '48') + '|'
        + (window._mveColorCheckEnabled ? '1' : '0') + '|'
        + minConf + '|'
        + (localStorage.getItem('mve_heatmap_baseline') || '') + '|'
        + (localStorage.getItem('mve_heatmap_phase') || '') + '|'
        + (localStorage.getItem('mve_heatmap_parent_class') || '');
    if (_ladderActive && _loadKey === _lastLoadKey) return;
    // 4.0.251 — also swallow rapid DUPLICATE re-fires of the SAME view within 1.5s.
    // The console showed 3 back-to-back [core] reloads after a tab-visible reconnect,
    // each restarting the carpet from empty. A genuine change (window / shipment / axis
    // / bins / …) has a different _loadKey and still refreshes immediately.
    if (_loadKey === _lastLoadKey && window._mveLastRacFireAt
        && (Date.now() - window._mveLastRacFireAt) < 1500) return;
    window._mveLastRacFireAt = Date.now();
    _lastLoadKey = _loadKey;

    _progressiveLadderTicket += 1;
    const myTicket = _progressiveLadderTicket;
    _ladderActive = true;
    window._mveScatterRange = null;  // 4.0.256 — only the all-shipments window branch sets this
    try {

    // 4.0.202 — DO NOT call refreshQualityCharts() here. refreshAdvancedCharts
    // is only ever called from refreshDetectionInsights (line ~1405), which
    // already fires refreshQualityCharts() once before this. Calling it again
    // here made the quality-heatmap + ejection strips fetch TWICE per refresh
    // (and race each other). One call upstream is enough.

    // 4.0.203 — ONLY 1h all-shipments is a single fetch. A PICKED shipment
    // now falls through to the progressive bin-by-bin ladder below.
    //
    // Why the ladder (reversing v4.0.201's single-fetch): the backend's
    // `camera_scatter` is `ORDER BY time DESC LIMIT dot_cap` — the NEWEST N
    // dots, not stratified. So a single fetch at a low dot_cap (operator set
    // Dots/Bucket=100) returns only the newest 100 dots → they cluster at the
    // right edge. Proven: dot_cap=100 → decile distribution {8:10, 9:90};
    // dot_cap=750 → spread {2..9} only because ~all 715 dots fit under the cap.
    // The operator's mental model is correct: dot_cap is PER BUCKET. The ladder
    // splits the shipment's [t_min,t_max] into `bins` slices and fetches up to
    // dot_cap dots FOR EACH slice → dots spread across the whole span, newest→
    // oldest, with the loading badge. The v4.0.201 "jamming right" was the Core
    // render RACE (no ticket guard) stamping a stale wide axis — fixed in
    // v4.0.202, so the ladder now lands its dots on the correct [t_min,t_max].
    if (target === '1h' && !shipment) {
        // Clear stale dots immediately so nothing lingers under the new load.
        try {
            if (_insightCameraScatter && _insightCameraScatter.data
                && Array.isArray(_insightCameraScatter.data.datasets)) {
                _insightCameraScatter.data.datasets.forEach(ds => { ds.data = []; });
                _insightCameraScatter.update('none');
            }
        } catch (_) {}
        // 4.0.202 — show the scatter spinner while the single fetch runs (the
        // operator asked for a loading indicator when picking a shipment). Core
        // is fast (~0.5 s) but on a slow remote link it's worth the feedback.
        // Unmount when Core resolves — guarded by the ticket so a superseded
        // pick doesn't yank a newer pick's spinner.
        try { _mountScatterLoader('Loading dots…'); } catch (_) {}
        try { await _refreshAdvancedChartsCore(); }
        finally { if (myTicket === _progressiveLadderTicket) { try { _unmountScatterLoader(); } catch (_) {} } }
        // v4.0.210 — no ladder in the 1h single-fetch path, so paint the quality
        // + ejection strips with ONE full-1h strips fetch (same endpoint/binning
        // as the scatter). Core already seeded _progressiveColorRange.
        if (myTicket === _progressiveLadderTicket) {
            try { await _extendStripsFromBucket(Date.now() - _windowToMs('1h'), Date.now(), '', myTicket); }
            catch (_e) {}
        }
        return;
    }

    // 4.0.199 — everything else (all-shipments over a window OR a specific
    // shipment) uses the progressive bin-by-bin ladder so the operator sees
    // dots stream in newest→oldest with a live "Loading dots" badge and the
    // scatter fills edge-to-edge. For a PICKED shipment the ladder spans the
    // shipment's OWN [t_min,t_max] (from shipment_spans) rather than the
    // sliding [now-window,now]. Without this a picked shipment did a single
    // capped fetch that only returned the newest ~750 dots, so they clustered
    // at the right edge and never filled the span. (operator: "find the start
    // and stop of the shipment, divide it by the bucket count, load bin by
    // bin, show the loading dot meanwhile.")

    // Clear stale dots + any orphaned badge before starting.
    try { _unmountScatterLoader(); } catch (_) {}
    try {
        if (_insightCameraScatter && _insightCameraScatter.data
            && Array.isArray(_insightCameraScatter.data.datasets)) {
            _insightCameraScatter.data.datasets.forEach(ds => { ds.data = []; });
            _insightCameraScatter.update('none');
        }
    } catch (_) {}

    const bins = Math.max(4, Math.min(192,
        parseInt(localStorage.getItem('mve_bucket_count') || '48', 10) || 48));

    // Determine the ladder's [loMs, hiMs] time range.
    let hiMs, loMs;
    if (shipment) {
        // Authoritative span from shipment_spans (fast, ~70 ms).
        try {
            const sr = await fetch('/api/shipment_spans?window=24h&shipment=' + encodeURIComponent(shipment));
            if (myTicket !== _progressiveLadderTicket) return;
            const sj = await sr.json();
            const sp = (sj.shipment_spans || [])[0];
            if (sp && Number.isFinite(Number(sp.t_min)) && Number.isFinite(Number(sp.t_max))
                && Number(sp.t_max) > Number(sp.t_min)) {
                loMs = Number(sp.t_min); hiMs = Number(sp.t_max);
            }
        } catch (e) { console.warn('[progressive] shipment-span fetch failed:', e); }
        if (loMs == null || hiMs == null) {
            // No usable span → single Core fetch still shows the data. v4.0.210 —
            // also paint the strips from one wide strips fetch (same endpoint).
            await _refreshAdvancedChartsCore();
            if (myTicket === _progressiveLadderTicket) {
                try { await _extendStripsFromBucket(Date.now() - _windowToMs('90d'), Date.now(), shipment, myTicket); }
                catch (_e) {}
            }
            return;
        }
    } else {
        // 4.0.256 — operator spec: bucket the ACTUAL data extent of the SELECTED axis,
        // not [now-window, now] (mostly empty when the line ran only recently). Ask the
        // backend for a cheap MIN/MAX range probe (time + encoder) over the window. The
        // ladder always buckets by TIME (since/until), so loMs/hiMs use the data TIME
        // extent; the encoder extent seeds the encoder-axis domain. 4s-timeout fallback
        // to [now-window, now] so a slow probe can't stall the load.
        hiMs = Date.now();
        loMs = hiMs - _windowToMs(target);
        window._mveEncWindowRange = null;
        window._mveScatterRange = null;
        window._mveEncSpanFixed = null;   // 4.0.304 — set below when the probe yields a real encoder span
        try {
            // 4.0.304 — the range probe IS the "one simple response at the start that returns
            // the encoder MIN/MAX (two fields)". It used to be a SINGLE 4s attempt; on a flaky
            // tunnel it reset → the axis then auto-widened from partial dots and JUMPED. Because
            // the encoder is ROLL-POSITION (resets per roll, not monotonic with time), an older
            // roll with a bigger encoder rescaled the whole axis mid-load — the operator's
            // "it resets many times / inconsistent span". Retry the probe like the buckets so the
            // two fields arrive reliably, then PIN them (_mveEncSpanFixed) so the axis is set ONCE
            // and stays fixed.
            let _rangeJson = null;
            for (let _rt = 0; _rt < 3; _rt++) {
                _rangeJson = await Promise.race([
                    fetch('/api/detection_charts?range_only=1&window=' + encodeURIComponent(target)).then(r => r.json()).catch(() => null),
                    new Promise((res) => setTimeout(() => res(null), 4000)),
                ]);
                if (myTicket !== _progressiveLadderTicket) return;
                if (_rangeJson && Number.isFinite(Number(_rangeJson.data_t_min))) break;
                await new Promise(res => setTimeout(res, 300 * (_rt + 1)));
            }
            const _tmn = _rangeJson && Number(_rangeJson.data_t_min);
            const _tmx = _rangeJson && Number(_rangeJson.data_t_max);
            if (Number.isFinite(_tmn) && Number.isFinite(_tmx) && _tmx > _tmn) {
                loMs = _tmn; hiMs = _tmx;
                const _emn = Number(_rangeJson.data_enc_min), _emx = Number(_rangeJson.data_enc_max);
                window._mveScatterRange = { loMs: loMs, hiMs: hiMs, encLo: _emn, encHi: _emx };
                // Real encoder span ⇒ seed the encoder-axis domain; 0-width (stationary
                // line) stays null so Core's guard falls back to time as before.
                if (Number.isFinite(_emn) && Number.isFinite(_emx) && _emx > _emn) {
                    window._mveEncWindowRange = { lo: _emn, hi: _emx };
                    // PIN the span. The auto-widen below anchors to this fixed extent, so the
                    // encoder axis is full-width from the first bucket and never jumps mid-load.
                    window._mveEncSpanFixed = { lo: _emn, hi: _emx };
                }
            }
        } catch (_e) { /* keep the [now-window, now] fallback */ }
    }
    const bucketMs = Math.max(1000, Math.floor((hiMs - loMs) / bins));

    // Mount the scatter-scoped loader — NOT the tab-wide mveLoaderBegin.
    _mountScatterLoader('Loading dots — newest first');

    // Set up the chart (axis, strips, size/conf bands) via Core BEFORE the
    // ladder runs. For a picked shipment we let Core fetch with the shipment
    // filter (its 90-day override returns the full span; the axis-from-span
    // logic sets the [t_min,t_max] domain, and the lane strip is fed the same
    // domain → the shipment lane spans exactly the scatter). For all-shipments
    // we hydrate with a light window=1h fetch as before.
    try {
        if (shipment) {
            if (myTicket !== _progressiveLadderTicket) { _unmountScatterLoader(); return; }
            await _refreshAdvancedChartsCore();
        } else {
            const bins48 = parseInt(localStorage.getItem('mve_bucket_count') || '48', 10) || 48;
            const r = await fetch('/api/detection_charts?window=1h'
                + '&shipment='
                + '&min_conf=' + minConf
                + '&baseline=' + encodeURIComponent(localStorage.getItem('mve_heatmap_baseline') || 'camera')
                + '&phase=' + encodeURIComponent(localStorage.getItem('mve_heatmap_phase') || '')
                + '&bins=' + bins48
                + '&dot_cap=' + _dotCap());
            if (myTicket !== _progressiveLadderTicket) { _unmountScatterLoader(); return; }
            const d = await r.json();
            try { await _refreshAdvancedChartsCore(d); } catch (e) { console.warn(e); }
        }
    } catch (e) {
        console.warn('[progressive] hydrate failed:', e);
    }

    // 4.0.247 — DO NOT blank the hydrated dots any more. The old code emptied the
    // scatter here so it could "carpet" bucket-by-bucket, but that made the chart
    // depend ENTIRELY on the 48-fetch ladder completing — and on a reset-prone link
    // enough bucket fetches failed that ZERO dots landed and the chart stayed blank.
    // We now keep the hydrate's dots on screen and TOP THEM UP with one full-window
    // fetch below, so a failed refill can never wipe the chart to empty.

    // Seed the dedup set with whatever's already on the chart.
    _progressiveScatterSeen = new Set();
    if (_insightCameraScatter) {
        for (const ds of (_insightCameraScatter.data.datasets || [])) {
            for (const p of (ds.data || [])) {
                _progressiveScatterSeen.add(_scatterPointKey(p, ds.label));
            }
        }
    }

    // 4.0.248 — bucket-by-bucket ladder RESTORED. A single full-window fetch clustered
    // ALL dots at the recent edge, because camera_scatter is ORDER BY time DESC LIMIT
    // dot_cap and a parent class (TB) fills the newest N with just the last few hours.
    // The ladder gives EACH bucket its own dot_cap, so dots spread evenly across the
    // whole window. It's now robust where it used to fail: each bucket fetch retries on
    // reset (see _extendScatterFromBucket) and we no longer blank the hydrate first, so
    // a single dropped bucket can never wipe the chart to empty.
    let bucketCount = 0;
    // 4.0.305 — bucket along the SELECTED axis (operator: "bucket by each x-axis, time or
    // encoder"). ENCODER axis with a real fixed span → bucket by ENCODER: divide the pinned
    // [enc_min, enc_max] into N columns and load each by its encoder start/stop, so a column
    // fills exactly its slice and the span never re-derives (encoder isn't monotonic with
    // time, so time-bucketing scattered a bucket's dots across the whole axis). TIME axis (or
    // a stationary line where no encoder span exists) keeps the time ladder — time IS
    // monotonic, so its buckets already map 1:1 to columns.
    const _fxEnc = window._mveEncSpanFixed;
    const _useEncoderBuckets = (!shipment && _insightAxis === 'encoder'
        && _fxEnc && Number.isFinite(_fxEnc.lo) && Number.isFinite(_fxEnc.hi) && _fxEnc.hi > _fxEnc.lo);
    try {
        if (_useEncoderBuckets) {
            const _eLo = _fxEnc.lo, _eHi = _fxEnc.hi;
            const _encStep = (_eHi - _eLo) / bins;
            // Pin the strip/colour domain to the SAME fixed span, then fetch strips + colour
            // ONCE over the full window (all N bins in one aggregate — cheaper than 50 slices).
            _progressiveColorRange = { axis: 'encoder', lo: _eLo, hi: _eHi, n_bins: bins };
            try { await _extendStripsFromBucket(loMs, hiMs, shipment, myTicket); }
            catch (_e) { /* strips non-fatal */ }
            for (let _bi = bins - 1; _bi >= 0; _bi--) {   // rightmost (highest encoder) column first
                if (myTicket !== _progressiveLadderTicket) return;
                if (!_insightCameraScatter) break;
                if ((winSel.value || '24h') !== target) break;
                if (_insightAxis !== 'encoder') break;    // operator flipped the axis mid-load
                const _bLo = _eLo + _bi * _encStep;
                // widen the last (rightmost) column slightly so the exact enc_max is included
                const _bHi = (_bi === bins - 1) ? (_eHi + Math.max(1, Math.abs(_eHi) * 1e-6)) : (_eLo + (_bi + 1) * _encStep);
                try { await _extendScatterFromBucket(loMs, hiMs, shipment, minConf, myTicket, null, _bLo, _bHi); }
                catch (_e) { /* per-column failure is non-fatal */ }
                bucketCount += 1;
            }
        } else {
            let endMs = hiMs;
            while (endMs > loMs && bucketCount < bins) {
                if (myTicket !== _progressiveLadderTicket) return;
                if (!_insightCameraScatter) break;
                if (!shipment && (winSel.value || '24h') !== target) break;
                if (shipment && ((document.getElementById('insight-shipment') || {}).value || '') !== shipment) break;
                const sinceMs = Math.max(loMs, endMs - bucketMs);
                // v4.0.300 — ONE atomic call per bucket (dots + colour + 4 strips) via
                // _extendBucketAll, instead of the old TWO (scatter, then strips/colour).
                try { await _extendBucketAll(sinceMs, endMs, shipment, minConf, myTicket); }
                catch (_e) { /* per-bucket failure is non-fatal */ }
                endMs = sinceMs;
                bucketCount += 1;
            }
        }
    } finally {
        if (myTicket === _progressiveLadderTicket) {
            _unmountScatterLoader();
            try { _paintDotsDiag((_useEncoderBuckets ? 'enc-ladder:' : 'ladder:') + bucketCount + 'buckets'); } catch (_) {}
        }
    }
    } finally {
        // v4.0.230 — release the coalesce gate, but only if THIS load is still the
        // current one (a newer refresh that superseded us owns the flag now).
        if (myTicket === _progressiveLadderTicket) _ladderActive = false;
    }
}
window.refreshAdvancedCharts = refreshAdvancedCharts;

// v4.0.80 — scatter-scoped spinner: sits INSIDE the scatter canvas's
// parent, never covers the whole tab. Ensures the rest of Charts remains
// interactive even while dots are still loading.
function _mountScatterLoader(label) {
    const host = document.getElementById('insight-camera-scatter');
    if (!host || !host.parentElement) return;
    const parent = host.parentElement;
    if (getComputedStyle(parent).position === 'static') parent.style.position = 'relative';
    let el = document.getElementById('mve-scatter-loader');
    if (!el) {
        el = document.createElement('div');
        el.id = 'mve-scatter-loader';
        el.style.cssText = 'position:absolute;top:6px;right:6px;display:flex;align-items:center;'
            + 'gap:6px;background:rgba(15,23,42,0.72);color:#e2e8f0;font-size:11px;'
            + 'font-weight:600;padding:4px 8px;pointer-events:none;z-index:5;'
            + 'border-radius:12px;transition:opacity 200ms;';
        el.innerHTML = '<span style="width:12px;height:12px;border:2px solid rgba(226,232,240,0.35);'
            + 'border-top-color:#e2e8f0;border-radius:50%;animation:spin 0.7s linear infinite;'
            + 'display:inline-block"></span><span id="mve-scatter-loader-text"></span>';
        parent.appendChild(el);
    }
    const t = document.getElementById('mve-scatter-loader-text');
    if (t) t.textContent = String(label || 'Loading dots…');
    el.style.opacity = '1';
    el.style.display = 'flex';
}
function _unmountScatterLoader() {
    const el = document.getElementById('mve-scatter-loader');
    if (!el) return;
    el.style.opacity = '0';
    setTimeout(() => { if (el && el.parentElement) el.parentElement.removeChild(el); }, 220);
}

// v4.0.74 — animate the scatter's growth from newest → oldest so the operator
// sees dots progressively appear. All data was already fetched by the Core
// render above; this just re-orders the visual reveal.
let _scatterAnimTicket = 0;

// Convert a window string ('24h', '7d', ...) to milliseconds.
function _windowToMs(w) {
    const m = String(w || '').match(/^(\d+)\s*([mhdw])$/i);
    if (!m) return 24 * 3600 * 1000;
    const n = parseInt(m[1], 10);
    const u = m[2].toLowerCase();
    if (u === 'm') return n * 60 * 1000;
    if (u === 'h') return n * 3600 * 1000;
    if (u === 'd') return n * 86400 * 1000;
    if (u === 'w') return n * 7 * 86400 * 1000;
    return 24 * 3600 * 1000;
}

// Progressive-load bookkeeping.
// Global Set of scatter-point keys already visible on the chart, so that
// subsequent wider-window fetches only APPEND net-new points instead of
// double-plotting the same detection.
let _progressiveScatterSeen = new Set();

function _scatterPointKey(p, cls) {
    // Deterministic identity from operator-visible coordinates + class + image
    // path (the image path is the unique per-detection anchor produced by
    // detection.py — server never mints the same one twice). Falls back to
    // best-available fields when older detection_charts payloads omit them.
    const _cls = cls || p.cls || '';
    return String(p.x) + '|' + String(p.y ?? p.cam_id ?? '') + '|' + _cls + '|' + String(p.img || '');
}

// For a selected target window, return the ordered list of wider-than-1h
// windows to load progressively. Excludes '1h' itself (already loaded in
// Phase 1) and stops at the operator's target.
//
// v4.0.74-bucket: finer rungs so the operator sees dots grow every ~500 ms
// instead of leaping in one jump from 1h → 24h. Each rung's fetch fires
// against the DB with the same base cost as a wider one (server still scans
// inference_results with a time>NOW()-INTERVAL predicate), so the total wall
// time to reach the target IS longer than a single big query — but the
// operator sees continuous forward progress on the scatter, which is what
// they asked for ("bucket by bucket").

// v4.0.74 — silently fetch one time-bucket via `since_ms`/`until_ms` and
// append only NEW points to the existing `_insightCameraScatter`. Server
// short-circuits (`scatter_only=1`) so it skips the heavy size / confidence /
// heatmap / ejection queries — only the two stratified scatter queries
// run against the bucket's narrow slice. No loader spinner. No touch of
// the other charts. Uses the exact same downstream extend logic so the
// dedup + dataset-append behaviour is identical to the window-based path.
// v4.0.106 — PER-CLASS window-normalized dot radii. Previously each dot's
// radius was `3 + conf * 9`, a fixed [3, 12] px mapping regardless of the
// actual confidence spread. Operators reported "all dots look the same
// size" because most detections cluster in the 0.5-0.9 conf range which
// mapped to a narrow 7-11 px slice.
//
// Per-DATASET (per-class) normalization: each class' own min/max maps
// independently to [3, 12]. A class that clusters at 0.9 gets its own
// full-range spread; a class ranging 0.2-0.5 also gets its own full
// spread. The alternative (global normalization) would make the low-
// confidence class invisibly small next to the high-confidence one.
//
// Reads `p.conf` (raw confidence stored at insertion time) and writes
// `p.r` (Chart.js bubble radius). Called before every scatter update —
// both the initial `new Chart(...)` render and every progressive-bucket
// extend — so the LARGEST dot IN A CLASS is always the highest-
// confidence detection of that class in view.
function _renormalizeScatterRadii(chart) {
    if (!chart || !chart.data || !chart.data.datasets) return;
    const MIN_R = 3, MAX_R = 12;
    for (const ds of chart.data.datasets) {
        let mn = Infinity, mx = -Infinity;
        for (const p of (ds.data || [])) {
            const c = p.conf;
            if (typeof c === 'number' && !isNaN(c)) {
                if (c < mn) mn = c;
                if (c > mx) mx = c;
            }
        }
        if (!isFinite(mn)) continue;   // class has no numeric conf; skip
        if (mn === mx) {
            // Every dot in this class has identical conf (or only one dot
            // total). Use midpoint so all are visible but not misleadingly
            // large.
            const flat = (MIN_R + MAX_R) / 2;
            for (const p of (ds.data || [])) {
                if (typeof p.conf === 'number' && !isNaN(p.conf)) p.r = flat;
            }
            continue;
        }
        const range = mx - mn;
        for (const p of (ds.data || [])) {
            if (typeof p.conf === 'number' && !isNaN(p.conf)) {
                p.r = MIN_R + ((p.conf - mn) / range) * (MAX_R - MIN_R);
            } else {
                p.r = MIN_R;
            }
        }
    }
}


// 4.0.246 — VISIBLE dots diagnostic. Paints a red monospace line above the
// scatter title (shows in screenshots — no console needed). Called after the base
// render AND after the progressive ladder so we see the FINAL drawn state.
function _paintDotsDiag(tag) {
    // 4.0.262 — diagnostic REMOVED now that the loader is fixed. Just clean up any banner
    // a cached build may have left in the DOM, then no-op.
    try { const _old = document.getElementById('mve-dots-diag'); if (_old && _old.parentNode) _old.parentNode.removeChild(_old); } catch (_e) {}
}
window._paintDotsDiag = _paintDotsDiag;

async function _extendScatterFromBucket(sinceMs, untilMs, shipment, minConf, ticket, preData, encLo, encHi) {
    if (!_insightCameraScatter) return;
    let data = preData || null;   // v4.0.300 — reuse the merged one-call payload when given
    // 4.0.247 — dot_cap can be boosted by the single-fetch path (window._mveScatterFetchCap)
    // so ONE fetch over the whole window still returns dots spread across it, not just the
    // newest _dotCap().
    const _fetchCap = (window._mveScatterFetchCap || _dotCap());
    // 4.0.305 — encLo/encHi (encoder-axis bucketing): fetch the dots for ONE encoder column
    // over the FULL time window, so bucket i == the i-th encoder slice of the fixed span.
    const _encQS = (encLo != null && encHi != null && Number.isFinite(encLo) && Number.isFinite(encHi) && encHi > encLo)
        ? ('&enc_lo=' + encLo + '&enc_hi=' + encHi) : '';
    const _scatUrl = '/api/detection_charts?since_ms=' + Math.floor(sinceMs) +
                     '&until_ms=' + Math.floor(untilMs) +
                     '&shipment=' + encodeURIComponent(shipment) +
                     '&min_conf=' + minConf +
                     '&scatter_only=1' +
                     '&dot_cap=' + _fetchCap + _encQS;
    // 4.0.247 — retry on transient connection resets (ERR_CONNECTION_RESET seen on the
    // operator's tunnel). A single dropped fetch used to silently lose the dots.
    // 4.0.300 — skip the fetch when the merged one-call path already fetched this bucket.
    if (!data) {
        for (let _try = 0; _try < 3; _try++) {
            try { const r = await fetch(_scatUrl); data = await r.json(); break; }
            catch (e) {
                if (_try === 2) { console.warn('[progressive] scatter fetch failed after 3 tries:', e); return; }
                await new Promise(res => setTimeout(res, 300 * (_try + 1)));
            }
        }
    }
    if (!data) return;
    // 4.0.198 — bail if the progressive run that spawned this bucket was
    // superseded (operator picked a shipment / changed window while this
    // fetch was in flight). Without this, a late all-shipments bucket
    // response pushes stale dots onto the freshly-rebuilt picked-shipment
    // chart — the "old dots remained there" bug.
    if (ticket != null && ticket !== _progressiveLadderTicket) return;

    // v4.0.132 — same axis handling as the base renderer, using the
    // cached shipment_length_offsets (bucket responses don't include it).
    // 4.0.199 — prefer the EFFECTIVE axis mode Core resolved (window._mveUnifiedX
    // .axisMode). Core applies a stationary-line guard that flips encoder→time
    // when the encoder has no width; if the bucket loader ignored that and used
    // the raw `_insightAxis`, its dots would use encoder x=0 (all piled at the
    // left) while Core drew time — a mismatch. Fall back to _insightAxis before
    // the first Core render sets _mveUnifiedX.
    const axisMode = (window._mveUnifiedX && window._mveUnifiedX.axisMode)
                   ? window._mveUnifiedX.axisMode
                   : ((_insightAxis === 'encoder') ? 'encoder'
                     : (_insightAxis === 'length')  ? 'length'
                     : 'time');
    const lenOffsets = (data && data.shipment_length_offsets && Object.keys(data.shipment_length_offsets).length)
        ? data.shipment_length_offsets
        : _lastLengthOffsets;
    const _toLengthX = (p) => {
        const info = lenOffsets[p.ship];
        if (!info) return p.x;
        const localLen = Math.max(0, (p.x || 0) - (info.enc_min || 0));
        return (info.offset || 0) + localLen;
    };
    const rawEnc = data.camera_scatter_encoder || [];
    const points = axisMode === 'encoder'
        ? rawEnc
        : axisMode === 'length'
            ? rawEnc.map(p => Object.assign({}, p, { x: _toLengthX(p) }))
            : (data.camera_scatter || []);
    if (!Array.isArray(points) || points.length === 0) return;

    // v4.0.108 — reuse the y-axis mapping the Core renderer built. Bucket
    // fetches use `scatter_only=1` which does NOT return `camera_y_order`,
    // so we fall back to `_lastCameraYOrder` (populated by the base
    // renderer). Without this, bucket dots plotted at Y=raw-cam-id while
    // base dots plotted at Y=display-index — same camera appeared twice
    // on the Y axis.
    const camOrder = Array.isArray(data.camera_y_order) && data.camera_y_order.length > 0
        ? data.camera_y_order
        : _lastCameraYOrder;
    const idxByCam = new Map(camOrder.map(function (cid, i) { return [cid, i]; }));
    const _camToIdx = function (cid) { return idxByCam.has(cid) ? idxByCam.get(cid) : cid; };

    const dsByLabel = new Map();
    (_insightCameraScatter.data.datasets || []).forEach(function (ds) {
        dsByLabel.set(ds.label, ds);
    });

    // v4.0.134 — reuse cached class-group map for shape selection. Bucket
    // fetches use scatter_only=1 which omits class_groups; the main
    // _refreshAdvancedChartsCore fetch populated _lastClassGroups on load.
    const _SHAPE_BY_GROUP_B = {
        defect:  'circle',
        parent:  'rectRot',
        marker:  'triangle',
        context: 'crossRot',
        // 4.0.180 — same sentinels as the base renderer; resolved by the
        // dataset scriptable pointStyle to a per-size canvas.
        vertical_defect:   '_narrow:vertical',
        horizontal_defect: '_narrow:horizontal',
    };
    const _bucketPointStyleFor = (cls) =>
        _SHAPE_BY_GROUP_B[_lastClassGroups[cls] || 'context'];
    let addedCount = 0;
    // v4.0.136 — track whether we created any NEW dataset (i.e. saw a class
    // for the first time in this progressive fetch). If so, re-render the
    // HTML class-colour legend so newly-appearing classes (e.g. WIR arriving
    // in an older bucket after only TB was in the initial fetch) actually
    // show up in the legend row above the chart.
    let newDatasetAdded = false;
    for (const p of points) {
        const cls = p.cls;
        if (typeof _isShown === 'function' && !_isShown(cls)) continue;
        const built = {
            x: p.x,
            y: _camToIdx(p.y),
            cam_id: p.y,
            r: 3 + (p.r || 0) * 9,
            conf: p.r,
            cls: cls,
            img: p.img,
            ship: p.ship,
            pid: p.pid,                 // 4.0.296 — pipeline that produced this dot
            pointStyle: _bucketPointStyleFor(cls),
        };
        const key = _scatterPointKey(built, cls);
        if (_progressiveScatterSeen.has(key)) continue;
        _progressiveScatterSeen.add(key);

        let ds = dsByLabel.get(cls);
        if (!ds) {
            ds = {
                label: cls,
                data: [],
                backgroundColor: (typeof _classColor === 'function' ? _classColor(cls) : '#888') + 'cc',
                borderColor:     (typeof _classColor === 'function' ? _classColor(cls) : '#888'),
                // v4.0.134 — same scriptable point-shape/size as the main
                // dataset builder so bucket-appended dots share the shape
                // scheme (defect=circle, parent=diamond, marker=triangle,
                // context=cross).
                pointStyle:      (ctx) => {
                    const ps = ctx.raw && ctx.raw.pointStyle;
                    if (typeof ps === 'string' && ps.indexOf('_narrow:') === 0) {
                        // 4.0.180 — vertical / horizontal narrow-rectangle
                        // pointStyle. Colour from the dataset's borderColor
                        // (already reflects the class palette). Size scales
                        // with the point's own `r` (confidence-normalised).
                        const orient = ps.slice(8);
                        const col = (ctx.dataset && ctx.dataset.borderColor) || '#94a3b8';
                        return _narrowRectCanvas(orient, (ctx.raw.r || 6), col);
                    }
                    return ps || 'circle';
                },
                // 4.0.187 — cap circle radius at 60% of the raw confidence-scaled
                // r so a max-confidence circle draws at diameter = 1.2r, which is
                // exactly half the narrow-defect bar's long dimension (2.4r). Bars
                // keep the full r so orientation stays legible at a glance.
                pointRadius:     (ctx) => {
                    const r  = (ctx.raw && ctx.raw.r) || 5;
                    const ps = ctx.raw && ctx.raw.pointStyle;
                    return (typeof ps === 'string' && ps.indexOf('_narrow:') === 0) ? r : (r * 0.6);
                },
                pointHoverRadius:(ctx) => {
                    const r  = (ctx.raw && ctx.raw.r) || 5;
                    const ps = ctx.raw && ctx.raw.pointStyle;
                    return ((typeof ps === 'string' && ps.indexOf('_narrow:') === 0) ? r : (r * 0.6)) + 2;
                },
                showLine: false,
            };
            _insightCameraScatter.data.datasets.push(ds);
            dsByLabel.set(cls, ds);
            newDatasetAdded = true;
        }
        ds.data.push(built);
        addedCount++;
    }
    if (newDatasetAdded) {
        try {
            const _legendEl = document.getElementById('insight-scatter-legend');
            if (_legendEl) {
                const _datasets = _insightCameraScatter.data.datasets || [];
                _legendEl.innerHTML = _datasets.map((ds, i) => {
                    const swatch = String(
                        (typeof _classColor === 'function' ? _classColor(ds.label) : '#3b82f6')
                    );
                    const label  = String(ds.label || '').replace(/[<>&]/g, '');
                    return `<span data-ds-idx="${i}" style="display:inline-flex; align-items:center; gap:4px; cursor:pointer; user-select:none;"><span style="display:inline-block; width:12px; height:12px; background:${swatch}; border-radius:2px;"></span>${label}</span>`;
                }).join('');
                _legendEl.querySelectorAll('[data-ds-idx]').forEach(el => {
                    el.onclick = () => {
                        const idx = parseInt(el.getAttribute('data-ds-idx'), 10);
                        const ch = _insightCameraScatter;
                        if (!ch || !Number.isFinite(idx)) return;
                        const vis = ch.isDatasetVisible(idx);
                        ch.setDatasetVisibility(idx, !vis);
                        ch.update();
                        el.style.opacity = vis ? '0.4' : '1';
                        el.style.textDecoration = vis ? 'line-through' : 'none';
                    };
                });
            }
        } catch (_e) { /* legend re-render is non-fatal */ }
    }
    if (addedCount > 0) {
        // v4.0.106 — renormalize before update so newly-appended dots' sizes
        // reflect the whole visible set's min/max confidence, not just their
        // own bucket. Otherwise the first bucket's dots are frozen at their
        // initial (raw-confidence) sizes while later buckets get rescaled.
        _renormalizeScatterRadii(_insightCameraScatter);
        // v4.0.233 — AUTO-WIDEN the encoder/length x-axis to fit ALL loaded dots.
        // The window's true encoder span can't be queried reliably on a heavy line
        // (shipment_spans caps at 200k rows; shipment_quality_score TIMES OUT → null),
        // so the axis GROWS as the ladder streams older/lower-encoder dots
        // newest→oldest. No range query; every shipment in the window appears; the
        // strip domain (_progressiveColorRange) tracks it so the strips — loaded AFTER
        // the scatter — bin over the exact same range.
        try {
            // 4.0.260 — read the scatter's ACTUAL effective axis (__mveAxis, set by Core's
            // flat-encoder guard) FIRST. Before this, when _mveUnifiedX was undefined mid-render
            // it fell back to _insightAxis (the DEFAULT 'encoder'), so this block ran on a TIME
            // chart and rewrote _progressiveColorRange.axis→'encoder' with TIME bounds — the
            // strips then binned by encoder over a time range and collapsed to ONE bucket (DIAG:
            // axis=time but stripAxis=encoder). __mveAxis keeps the strips in sync with the dots.
            const _am = (_insightCameraScatter && _insightCameraScatter.__mveAxis)
                     || (window._mveUnifiedX && window._mveUnifiedX.axisMode) || _insightAxis;
            const _pickedNow = ((document.getElementById('insight-shipment') || {}).value || '').trim();
            const _xs = _insightCameraScatter.options && _insightCameraScatter.options.scales
                && _insightCameraScatter.options.scales.x;
            // 4.0.249 — auto-widen ONLY the encoder/length axis (its span is unknown).
            // The TIME axis is pinned to the full window [loMs,hiMs] up front by
            // refreshAdvancedCharts, so the ladder carpets right→left into a STABLE frame;
            // auto-widening time here rescaled the axis every bucket and made the dots
            // jump around (operator: "why not loading bucket by bucket right→left").
            if (!_pickedNow && (_am === 'encoder' || _am === 'length') && _xs) {
                let _lo = Infinity, _hi = -Infinity;
                for (const _ds of (_insightCameraScatter.data.datasets || [])) {
                    for (const _p of (_ds.data || [])) {
                        const _x = Number(_p.x);
                        if (Number.isFinite(_x)) { if (_x < _lo) _lo = _x; if (_x > _hi) _hi = _x; }
                    }
                }
                // 4.0.304 — ANCHOR to the fixed span from the range probe (the two-field
                // response, fetched once at load start). Union with the dots so nothing is
                // ever clipped, but the axis starts at the FULL extent instead of growing
                // bucket-by-bucket — which is what made it JUMP on roll-position encoders (an
                // older roll with a bigger encoder rescaled everything). "Keep the span fixed."
                const _fx = window._mveEncSpanFixed;
                if (_fx && Number.isFinite(_fx.lo) && Number.isFinite(_fx.hi) && _fx.hi > _fx.lo) {
                    _lo = Math.min(_lo, _fx.lo); _hi = Math.max(_hi, _fx.hi);
                }
                if (Number.isFinite(_lo) && Number.isFinite(_hi) && _hi > _lo) {
                    const _pad = (_hi - _lo) * 0.01;
                    _xs.min = _lo - _pad; _xs.max = _hi + _pad;
                    if (window._mveUnifiedX) { window._mveUnifiedX.xMin = _lo - _pad; window._mveUnifiedX.xMax = _hi + _pad; }
                    // Only encoder/length re-key the colour-strip domain; time strips keep
                    // the range Core seeded (the backend has no color_axis=time binning).
                    if (_am === 'encoder' || _am === 'length') {
                        const _nb = (_progressiveColorRange && _progressiveColorRange.n_bins) || (parseInt(localStorage.getItem('mve_bucket_count') || '48', 10) || 48);
                        _progressiveColorRange = { axis: 'encoder', lo: _lo - _pad, hi: _hi + _pad, n_bins: _nb };
                    }
                }
            }
        } catch (_wm) { /* auto-widen is best-effort */ }
        _insightCameraScatter.update('none'); // no animation — feels instant
        // v4.0.106 — was `console.log('[progressive] window=' + win + ...)` but
        // `win` was never a parameter of `_extendScatterFromBucket`. That
        // ReferenceError threw AFTER the scatter update — the chart WAS
        // populated in memory but the throw prevented the return, which
        // aborted `refreshAdvancedCharts`, which aborted the progressive
        // bucket loop, which meant subsequent buckets never appended their
        // dots. Net effect on-screen: user saw at most one bucket's worth
        // of dots, often none. Log the actual time-range we appended so
        // the entry is still useful.
        void addedCount; // (was a per-bucket console.log; removed — narration spam)
    }
}

async function _refreshAdvancedChartsCore(preloadedData) {
    // v4.0.101 — accept optional preloaded data so the progressive
    // hydrate can reuse its window=1h payload instead of triggering
    // another slow `window=24h` fetch here. Khoy timing: `window=24h`
    // on 1.5M rows takes ~28 s server-side (five sequential CTEs at
    // 4-8 s each) — combined with the earlier 12 s hydrate call the
    // scatter was taking ~40 s to render, which the operator read as
    // "no dots" (they moved on). With preload the scatter builds from
    // 1h data in ~200 ms; the progressive bucket loop then extends it
    // backwards. Callers with no preload keep prior behaviour.
    const window = document.getElementById('insight-window')?.value || '24h';
    const shipment = document.getElementById('insight-shipment')?.value || '';
    const minConf = (parseFloat(document.getElementById('insight-min-conf')?.value || '0') / 100) || 0;
    let data = preloadedData;
    // 4.0.202 — RACE GUARD. Capture the current refresh generation on entry.
    // `refreshAdvancedCharts` bumps `_progressiveLadderTicket` on every call,
    // so if the operator picks shipment A then quickly picks B (or changes the
    // window then picks a shipment), two Core calls overlap. Without this
    // guard, whichever /api/detection_charts fetch RESOLVES last wins the
    // destroy()+new Chart()+_mveUnifiedX write — a stale Core then stamps the
    // WRONG fixed axis domain (e.g. the sliding window) while the visible dots
    // occupy only the shipment's narrow sub-range, so the dots jam against one
    // edge. This is the "picked shipment, dots clustered at the right" bug.
    // We re-check right before mutating the chart and bail if superseded,
    // mirroring the guard `_extendScatterFromBucket` already has.
    const _myCoreTicket = (typeof _progressiveLadderTicket === 'number') ? _progressiveLadderTicket : 0;
    const _coreSuperseded = () => (typeof _progressiveLadderTicket === 'number') && _myCoreTicket !== _progressiveLadderTicket;
    // 4.0.53 — gate the loader to the Charts tab so the badge can't leak
    // into other tabs. The paired mveLoaderEnd calls below check the same flag.
    const _showedLoader = _chartsTabActive() && !preloadedData;
    if (_showedLoader) mveLoaderBegin('Loading charts…');
    try {
        if (!data) {
            // 4.0.30 — tell the server which heatmap baseline mode is active so
            // it only computes that one (saves SQL for unused modes). Falls back
            // to `camera` if nothing's been stored yet.
            const baseline = localStorage.getItem('mve_heatmap_baseline') || 'camera';
            // 4.0.34 — phase filter is heatmap-only. Empty string = all phases.
            const phase = localStorage.getItem('mve_heatmap_phase') || '';
            // 4.0.161 — parent_class filter (heatmap-only). Empty = all parents collapsed.
            const parentClass = localStorage.getItem('mve_heatmap_parent_class') || '';
            // 4.0.35 — single bucket count drives heatmap + quality + ejection strips.
            const bins = parseInt(localStorage.getItem('mve_bucket_count') || '48', 10) || 48;
            const r = await fetch('/api/detection_charts?window=' + encodeURIComponent(window) +
                                  '&shipment=' + encodeURIComponent(shipment) +
                                  '&min_conf=' + minConf +
                                  '&baseline=' + encodeURIComponent(baseline) +
                                  '&phase=' + encodeURIComponent(phase) +
                                  '&parent_class=' + encodeURIComponent(parentClass) +
                                  '&bins=' + bins +
                                  '&dot_cap=' + _dotCap());
            data = await r.json();
        }
    } catch (e) {
        console.error('detection_charts fetch failed:', e);
        if (_showedLoader) mveLoaderEnd();
        return;
    }
    // 4.0.202 — a newer refresh started while our fetch was in flight; abandon
    // this render so we don't clobber the newer one's chart + axis domain.
    if (_coreSuperseded()) {
        if (_showedLoader) mveLoaderEnd();
        return;
    }

    // 4.0.198 — populate the shipment SEARCH datalist (was a <select>).
    // `#insight-shipment` is now an <input list=insight-shipment-list>; the
    // options live in the datalist so the browser gives free substring
    // search. The input's `.value` is the selected shipment (empty = All).
    // Preserve whatever the operator has typed/selected.
    const _shipList = document.getElementById('insight-shipment-list');
    if (_shipList && Array.isArray(data.shipments)) {
        // Newest first is friendlier when there are hundreds of shipments.
        const _sorted = data.shipments.slice().sort().reverse();
        _shipList.innerHTML = _sorted.map(s => `<option value="${s}"></option>`).join('');
    }

    // v4.0.119 — RENDER THE SHIPMENT STRIP IMMEDIATELY as soon as data
    // arrives. Previously the render was near the end of the function,
    // AFTER the big scatter/heatmap/color pipelines. Any error or slow
    // path in those blocks left the strip stuck showing the static HTML
    // fallback ("JS not loaded yet") even though data was available.
    // The strip only needs shipment_spans + a light axis choice — no
    // reason to gate it on the scatter finishing.
    try {
        // v4.0.211 — derive lanes from the scatter's OWN dots (same
        // /api/detection_charts data), never from data.shipment_spans over a
        // separate [min,max]. That old early paint stretched the lanes across a
        // different domain than the scatter → the "lane out of sync / stuck to
        // the start" bug. Empty until dots exist; the ladder grows it in
        // lock-step with the dots.
        if (typeof _renderShipmentStripFromScatterDots === 'function') {
            _renderShipmentStripFromScatterDots();
        }
    } catch (_earlyStripErr) {
        console.warn('early shipment-strip render failed:', _earlyStripErr);
    }

    const sizeT = data.size_over_time || [];
    const confT = data.confidence_over_time || [];
    const scatter = data.camera_scatter || [];

    // ---- (1) Object size distribution over time — width & height p10–p90 bands + median ----
    const sizeCtx = document.getElementById('insight-size-chart');
    if (sizeCtx) {
        if (_insightSizeChart) _insightSizeChart.destroy();
        _insightSizeChart = new Chart(sizeCtx, {
            type: 'line',
            data: {
                labels: sizeT.map(p => p.t),
                datasets: [
                    { label: 'Width p90', data: sizeT.map(p => p.w_hi), borderColor: 'rgba(59,130,246,0.4)', backgroundColor: 'rgba(59,130,246,0.12)', fill: '+1', tension: 0.3, pointRadius: 0, borderWidth: 1 },
                    { label: 'Width p10', data: sizeT.map(p => p.w_lo), borderColor: 'rgba(59,130,246,0.4)', fill: false, tension: 0.3, pointRadius: 0, borderWidth: 1 },
                    { label: 'Width median', data: sizeT.map(p => p.w_mid), borderColor: '#3b82f6', fill: false, tension: 0.3, pointRadius: 0, borderWidth: 2 },
                    { label: 'Height median', data: sizeT.map(p => p.h_mid), borderColor: '#f59e0b', fill: false, tension: 0.3, pointRadius: 0, borderWidth: 2, borderDash: [4, 3] }
                ]
            },
            options: _commonScaleOpts('Object size (px) over time — width band p10–p90 + medians')
        });
    }

    // ---- (2) Confidence over time — min/avg/max band ----
    const confCtx = document.getElementById('insight-confidence-chart');
    if (confCtx) {
        if (_insightConfidenceChart) _insightConfidenceChart.destroy();
        _insightConfidenceChart = new Chart(confCtx, {
            type: 'line',
            data: {
                labels: confT.map(p => p.t),
                datasets: [
                    { label: 'Max %', data: confT.map(p => p.c_max), borderColor: 'rgba(16,185,129,0.4)', backgroundColor: 'rgba(16,185,129,0.12)', fill: '+1', tension: 0.3, pointRadius: 0, borderWidth: 1 },
                    { label: 'Min %', data: confT.map(p => p.c_min), borderColor: 'rgba(16,185,129,0.4)', fill: false, tension: 0.3, pointRadius: 0, borderWidth: 1 },
                    { label: 'Avg %', data: confT.map(p => p.c_avg), borderColor: '#10b981', fill: false, tension: 0.3, pointRadius: 0, borderWidth: 2 }
                ]
            },
            options: (() => { const o = _commonScaleOpts('Detection confidence (%) over time'); o.scales.y.max = 100; return o; })()
        });
    }

    // ---- (2b) Confidence BY CLASS over time — one line per class (Show-filtered) ----
    const cbc = data.confidence_by_class || { buckets: [], series: {} };
    const ccCtx = document.getElementById('insight-confidence-class-chart');
    if (ccCtx) {
        if (_insightConfClassChart) _insightConfClassChart.destroy();
        const clsDatasets = Object.keys(cbc.series || {})
            .filter(_isShown)
            .map(cls => ({
                label: cls,
                data: cbc.series[cls],
                borderColor: _classColor(cls),
                backgroundColor: _classColor(cls) + '22',
                fill: false, tension: 0.3, pointRadius: 0, borderWidth: 2, spanGaps: true
            }));
        _insightConfClassChart = new Chart(ccCtx, {
            type: 'line',
            data: { labels: cbc.buckets || [], datasets: clsDatasets },
            options: (() => { const o = _commonScaleOpts('Detection confidence (%) by class over time'); o.scales.y.max = 100; return o; })()
        });
    }

    // 3.25.6 — compute scatter X-window in ms so the strip beneath aligns 1:1.
    const _win = document.getElementById('insight-window')?.value || '24h';
    const _winMs = (() => { const m = String(_win).match(/^(\d+)([hd])$/); return m ? parseInt(m[1]) * (m[2]==='d'?86400:3600) * 1000 : 24*3600*1000; })();
    const _nowMs = Date.now();
    // 4.0.256 — prefer the data's actual TIME extent (range probe stashed by
    // refreshAdvancedCharts) over [now-window, now], so the time axis spans exactly the
    // data (operator: bucket the min/max of the selected axis). Falls back to the window.
    const _sr256 = window._mveScatterRange;
    const _scatterXMin = (_sr256 && Number.isFinite(_sr256.loMs)) ? _sr256.loMs : (_nowMs - _winMs);
    const _scatterXMax = (_sr256 && Number.isFinite(_sr256.hiMs)) ? _sr256.hiMs : _nowMs;

    // ---- (3) Camera × {time|encoder} scatter — single merged renderer (3.25.13).
    // Builds the bubble chart for the currently selected `_insightAxis`; toggling
    // via setInsightAxis() destroys + rebuilds with the other axis's data.
    const scCtx = document.getElementById('insight-camera-scatter');
    if (scCtx) {
        if (_insightCameraScatter) _insightCameraScatter.destroy();
        // v4.0.132 — 'length' axis maps each dot's encoder-x into a
        // cumulative-length coordinate using shipment_length_offsets from
        // the payload. Cache offsets for the progressive-bucket loader too.
        // 4.0.197 — `let` (was const) so the stationary-line guard below can
        // flip encoder/length → time when the encoder has no width.
        let axisMode = (_insightAxis === 'encoder') ? 'encoder'
                       : (_insightAxis === 'length')  ? 'length'
                       : 'time';
        // 4.0.197 — STATIONARY-LINE GUARD. On a stopped line (or unwired
        // encoder, e.g. vteam12 test box) every encoder_value is identical
        // (usually 0), so the encoder/length axis would collapse to a single
        // vertical line and the scatter renders BLANK even though the dots
        // exist. Detect zero-width encoder dots and auto-fall-back to the
        // time axis so data is never invisible. This is a render-time
        // fallback only — it does NOT touch the operator's `_insightAxis`
        // toggle, so the axis returns to encoder automatically once the line
        // moves. The shipment-lane strip already does this same fallback.
        {
            const _encXs = (data.camera_scatter_encoder || [])
                .map(p => Number(p.x)).filter(Number.isFinite);
            const _encLo = _encXs.length ? Math.min.apply(null, _encXs) : null;
            const _encHi = _encXs.length ? Math.max.apply(null, _encXs) : null;
            // 4.0.273 — honour the operator's Encoder pick unless the encoder is
            // flat EVERYWHERE. The old guard fell back to time whenever THIS
            // window's dots were sparse — but on a MOVING line the range probe
            // reports a real encoder span (e.g. 50 → 21,698,506) even when few
            // defects were detected this window, so it wrongly showed Time while
            // Encoder was selected. Check ALL width sources: the window probe
            // (_encWinRange), the range probe (_mveScatterRange.encLo/encHi), AND
            // the loaded dots. Only flip when NONE has width (truly stationary /
            // unwired line where every encoder_value is identical, e.g. vteam12).
            const _sr = window._mveScatterRange;
            const _probeWidth = _sr && Number.isFinite(_sr.encLo) && Number.isFinite(_sr.encHi) && _sr.encHi > _sr.encLo;
            const _dotWidth = (_encLo != null && _encHi != null && _encHi > _encLo);
            // 4.0.274 — if the operator EXPLICITLY picked this axis (persisted by
            // setInsightAxis in localStorage), NEVER auto-flip it to time. The
            // flip only exists to rescue the DEFAULT on a flat/unwired line — it
            // must not override a deliberate choice. This makes it deterministic:
            // "I chose Encoder → I see Encoder", regardless of whether the async
            // range probe happened to reach this particular Core call.
            let _explicitPick = null;
            try { _explicitPick = localStorage.getItem('mve_insight_axis'); } catch (e) {}
            if (axisMode !== 'time' && _explicitPick !== axisMode
                && !_encWinRange() && !_probeWidth && !_dotWidth) {
                axisMode = 'time';
            }
        }
        // v4.0.239 — reflect the EFFECTIVE axis on the Time|Encoder|Length toggle: when
        // the guard falls back encoder/length→time (stopped/unwired line), the button
        // shows Time so it matches the chart (operator: "don't show time while Encoder
        // is selected"). Follows axisMode automatically as the line moves again.
        try { if (typeof _setAxisToggleUI === 'function') _setAxisToggleUI(axisMode); } catch (_ax) {}
        const lenOffsets = (data && data.shipment_length_offsets) || {};
        if (Object.keys(lenOffsets).length > 0) _lastLengthOffsets = lenOffsets;
        const _toLengthX = (p) => {
            const info = lenOffsets[p.ship];
            if (!info) return p.x; // no offset info — fall back to raw
            const localLen = Math.max(0, (p.x || 0) - (info.enc_min || 0));
            return (info.offset || 0) + localLen;
        };
        const rawEncPoints = data.camera_scatter_encoder || [];
        const points = axisMode === 'encoder'
            ? rawEncPoints
            : axisMode === 'length'
                ? rawEncPoints.map(p => Object.assign({}, p, { x: _toLengthX(p) }))
                : (scatter || []);
        // 4.0.24 — Camera Y-axis order matches the dashboard timeline grid.
        // Single source of truth: timeline_config.camera_order +
        // .custom_camera_order, computed server-side and surfaced as
        // data.camera_y_order = [5, 1, 6, 2, 4, 3]. We plot each point's y
        // as the INDEX into this array (display position), and the Y-axis
        // tick callback maps the index BACK to the actual camera ID for
        // display. So if the operator set custom = "5,1,6,2,4,3", the
        // top-to-bottom Y labels read 5, 1, 6, 2, 4, 3 — same as the
        // timeline grid columns. Any camera that appears in the data but
        // not in the order list lands at the end (server appended).
        // v4.0.108 — save this order to a module-level cache so
        // `_extendScatterFromBucket` can reuse it. Bucket fetches use
        // `scatter_only=1` which short-circuits server-side and does NOT
        // return `camera_y_order`. Without the cache, bucket dots plot at
        // Y=raw-cam-id while base dots plot at Y=display-index — same
        // camera appears at TWO Y positions on the axis.
        const camOrder = Array.isArray(data.camera_y_order) ? data.camera_y_order : [];
        if (camOrder.length > 0) _lastCameraYOrder = camOrder.slice();
        const idxByCam = new Map(camOrder.map((cid, i) => [cid, i]));
        const camByIdx = new Map(camOrder.map((cid, i) => [i, cid]));
        const _camToIdx = (cid) => idxByCam.has(cid) ? idxByCam.get(cid) : cid;
        // v4.0.134 — class-group shape scheme. Each dot's marker shape is
        // determined by which of the 4 groups its class belongs to (defect
        // / parent / marker / context). Colour still comes from _classColor
        // so shape adds a second visual channel without competing with
        // per-class colour identity.
        const _classGroups = (data && data.class_groups) || {};
        const _SHAPE_BY_GROUP = {
            defect:  'circle',
            parent:  'rectRot',   // rotated square == diamond
            marker:  'triangle',
            context: 'crossRot',
            // 4.0.180 — sentinel strings resolved to canvas elements
            // per-point in the scriptable pointStyle so the narrow rectangle
            // scales with the point's confidence-based radius.
            vertical_defect:   '_narrow:vertical',
            horizontal_defect: '_narrow:horizontal',
        };
        const _pointStyleFor = (cls) => _SHAPE_BY_GROUP[_classGroups[cls] || 'context'];
        _lastClassGroups = _classGroups;   // cache for _extendScatterFromBucket
        const byClass = {};
        points.forEach(p => {
            if (!_isShown(p.cls)) return;
            (byClass[p.cls] = byClass[p.cls] || []).push({
                x: p.x, y: _camToIdx(p.y), cam_id: p.y,
                r: 3 + (p.r || 0) * 9, conf: p.r, cls: p.cls, img: p.img, ship: p.ship, pid: p.pid,
                pointStyle: _pointStyleFor(p.cls),
            });
        });
        const datasets = Object.keys(byClass).map(cls => ({
            label: cls, data: byClass[cls],
            backgroundColor: _classColor(cls) + 'cc', borderColor: _classColor(cls),
            // v4.0.134 — scatter-type per-point styling. Scriptable options
            // pull pointStyle + pointRadius off each point's raw data so
            // every dot can have its own shape and size in one dataset.
            pointStyle:      (ctx) => {
                // 4.0.182 — resolve `_narrow:*` sentinels to a per-size canvas.
                // Missing this branch on this dataset builder (only the
                // progressive-bucket path had it in v4.0.180) is why TB dots
                // in the INITIAL 1h fetch still rendered as circles despite
                // role=vertical_defect being set on the class.
                const ps = ctx.raw && ctx.raw.pointStyle;
                if (typeof ps === 'string' && ps.indexOf('_narrow:') === 0) {
                    const orient = ps.slice(8);
                    const col = (ctx.dataset && ctx.dataset.borderColor) || '#94a3b8';
                    return _narrowRectCanvas(orient, (ctx.raw.r || 6), col);
                }
                return ps || 'circle';
            },
            // 4.0.187 — same 60%-of-r cap for the initial-dataset builder as
            // in the bucket-loader above. Kept identical so circles aren't a
            // different size before vs. after the first bucket refresh.
            pointRadius:     (ctx) => {
                const r  = (ctx.raw && ctx.raw.r) || 5;
                const ps = ctx.raw && ctx.raw.pointStyle;
                return (typeof ps === 'string' && ps.indexOf('_narrow:') === 0) ? r : (r * 0.6);
            },
            pointHoverRadius:(ctx) => {
                const r  = (ctx.raw && ctx.raw.r) || 5;
                const ps = ctx.raw && ctx.raw.pointStyle;
                return ((typeof ps === 'string' && ps.indexOf('_narrow:') === 0) ? r : (r * 0.6)) + 2;
            },
            showLine: false,
        }));
        // 4.0.44 — title + legend rendered as HTML siblings ABOVE the canvas
        // so the colour heatmap (which paints inside chartArea) can never
        // reach them. Clicking a legend swatch still toggles the matching
        // dataset via Chart.js's setDatasetVisibility.
        try {
            const _titleEl = document.getElementById('insight-scatter-title');
            if (_titleEl) {
                // 4.0.173 — append "measuring <parent_class>" so operators can
                // see which parent-role class the colour cells were computed
                // against. Falls back to "_root" (whole frame) when no class
                // is designated as role=parent on this site.
                let measTag = '';
                if (window._mveColorCheckEnabled === true) {
                    const pc = (localStorage.getItem('mve_heatmap_parent_class') || '').trim();
                    const avail = Array.isArray(data.parent_classes_available) ? data.parent_classes_available : [];
                    let m = pc || (avail.length === 1 ? avail[0] : 'all parents');
                    if (m === '_root') m = 'whole frame (no parent class set)';
                    measTag = ' · 🎨 measuring: ' + m;
                }
                _titleEl.textContent = ((axisMode === 'time')
                    ? 'Camera × time — hover for image, click dot to open the exact frame'
                    : 'Camera × encoder (roll position) — hover for image, click dot for the exact frame') + measTag;
            }
            const _legendEl = document.getElementById('insight-scatter-legend');
            if (_legendEl) {
                _legendEl.innerHTML = datasets.map((ds, i) => {
                    const swatch = String(_classColor(ds.label) || '#3b82f6');
                    const label  = String(ds.label || '').replace(/[<>&]/g, '');
                    return `<span data-ds-idx="${i}" style="display:inline-flex; align-items:center; gap:4px; cursor:pointer; user-select:none;"><span style="display:inline-block; width:12px; height:12px; background:${swatch}; border-radius:2px;"></span>${label}</span>`;
                }).join('');
                _legendEl.querySelectorAll('[data-ds-idx]').forEach(el => {
                    el.onclick = () => {
                        const idx = parseInt(el.getAttribute('data-ds-idx'), 10);
                        const ch = _insightCameraScatter;
                        if (!ch || !Number.isFinite(idx)) return;
                        const vis = ch.isDatasetVisible(idx);
                        ch.setDatasetVisibility(idx, !vis);
                        ch.update();
                        el.style.opacity = vis ? '0.4' : '1';
                        el.style.textDecoration = vis ? 'line-through' : 'none';
                    };
                });
            }
        } catch (_e) { /* legend render is non-fatal */ }
        // 4.0.38 — no UI hint about which classes are filtered server-side via
        // Process tab Show. The operator already knows what they toggled off
        // in Process tab; surfacing it on the Charts toolbar was noise (the
        // list balloons when math inference creates hundreds of synthetic
        // feature classes). Remove any leftover chip from prior sessions.
        try {
            const stale = document.getElementById('hm-process-hidden');
            if (stale && stale.parentNode) stale.parentNode.removeChild(stale);
        } catch (_e) { /* non-fatal */ }

        // 4.0.29 — color-drift heatmap rendered as the chart's BACKGROUND so
        // scatter dots overlay on top. Aggregation comes from the backend
        // (timeline.py:detection_charts), binned to N cells per camera by
        // encoder. Cell colour = HSL hue mapped from |delta_e| (mean E − the
        // camera's median E across the window): green at 0, yellow around 4,
        // red ≥ 8. Only painted on the encoder axis — time axis doesn't have
        // a sensible encoder→bin mapping. Plugin uses Chart.js's
        // beforeDatasetsDraw hook so rectangles fill chartArea BEFORE the
        // bubble dots render, matching the user's "background colour,
        // foreground dots" sketch from the design discussion.
        // 4.0.30 — baseline mode determines what we subtract from each cell's
        // mean E to get the ΔE the heatmap colours by. Backend computes ONLY
        // the active mode (saves SQL) and tells us via `color_baseline_modes`
        // which OTHER modes have data available right now, so we can enable
        // /disable the toggle buttons. `color_baseline` is a flat
        // `{cam_id: {E, L, a, b}}` map for the selected mode.
        const _ALL_BASELINES = ['camera', 'shipment_start', 'shipment_avg', 'target', 'reference_frame'];
        const _availableBaselineModes = new Set(data.color_baseline_modes || ['camera']);
        const _activeBaselineMode = data.color_baseline_mode || 'camera';
        const _activeBaselineMap  = data.color_baseline || {};
        // 4.0.244 — keep the Baseline dropdown in sync with the active mode
        // (only if it's one of the three options the dropdown offers, so a
        // legacy mode like shipment_start/target doesn't blank the select).
        const _bsSel = document.getElementById('hm-baseline-select');
        if (_bsSel && ['camera','shipment_avg','reference_frame'].includes(_activeBaselineMode) && _bsSel.value !== _activeBaselineMode) {
            _bsSel.value = _activeBaselineMode;
        }
        // 4.0.32 — track whether a reference is set so the Reference button
        // can skip pick mode next time.
        window._referenceIsSet = _availableBaselineModes.has('reference_frame');
        const _refPos = data.color_reference_position || null;
        for (const mode of _ALL_BASELINES) {
            const el = document.getElementById('hm-baseline-' + mode);
            if (!el) continue;
            const avail = _availableBaselineModes.has(mode);
            const active = (mode === _activeBaselineMode);
            // 4.0.32 — never set el.disabled. The onClick handler is what
            // enters pick-mode / auto-selects shipment / shows a helpful hint
            // for unconfigured modes. If we disable the button at the DOM
            // level the click never fires and the operator can't get OUT of
            // the disabled state. Visual muting (opacity) still tells the
            // operator it isn't currently active.
            el.style.cursor = 'pointer';
            el.style.opacity = avail ? '1' : '0.6';
            el.style.background = active
                ? 'linear-gradient(135deg,#10b981,#047857)'
                : 'rgba(51,65,85,0.5)';
            el.style.color = active ? '#fff' : '#cbd5e1';
        }

        // 4.0.34 — render the phase toggle buttons from the discovered list.
        // Buttons: always 'All', then one per phase from `phases_available`.
        // Active selection comes from localStorage (sticky across reloads).
        _renderPhaseButtons(
            Array.isArray(data.phases_available) ? data.phases_available : [],
            (typeof data.phase === 'string') ? data.phase : '',
        );
        // 4.0.161 — parent-class picker. Hidden entirely when the site
        // has ≤ 1 parent-class in the window (single-parent or _root-only).
        _renderParentClassButtons(
            Array.isArray(data.parent_classes_available) ? data.parent_classes_available : [],
        );
        // v4.0.231 — paint strips from the payload ONLY for a picked shipment /
        // single full-span fetch (preloadedData is null there). For an all-shipments
        // WINDOW load, `data` is the 1h HYDRATE: its cells are binned over the last
        // hour but the painters lay them out full-width → smeared + misaligned with
        // the scatter, then the ladder clears+restreams them (the reported blank/snap
        // and colour-change-ahead-of-the-others). So for a window load the progressive
        // ladder OWNS all four strips; do not seed them from the 1h payload here.
        if (!preloadedData) {
            // 4.0.162 — colour-change strip (between quality + ejection).
            try { _renderColorChangeStrip(axisMode, data); } catch (_e) { /* non-fatal */ }
            // v4.0.223 — quality / ejection / shipment from the full-span payload.
            try { _renderStripsFromPayload(data, axisMode); } catch (_e) { /* non-fatal */ }
        }

        // 4.0.29c — pick the heatmap variant that matches the current axis.
        // Backend returns BOTH binned-by-encoder and binned-by-time forms so
        // the background paints in both modes. Time mode is what shows up
        // when the line is stopped (encoder collapsed to a single value)
        // but the operator still wants to see colour drift over time.
        const _hmRaw = (axisMode === 'encoder')
            ? (data.color_heatmap || {})
            : (data.color_heatmap_time || {});
        // 4.0.164 — seed the progressive colour-cell cache with whatever the
        // initial (1h) payload returned. The bucket ladder below will merge
        // older-window cells into this same cache as it walks newest→oldest.
        try { _seedProgressiveColorCache(data, axisMode); } catch (_e) {}
        const heatmapCells = Array.isArray(_hmRaw.cells) ? _hmRaw.cells : [];
        // 4.0.281 — `heatmapActive` (whether the ΔE colour layer registers at ALL) needs
        // a valid [min,max]. Reading it from the 1h-hydrate payload means an IDLE line —
        // no colour rows in the LAST HOUR — yields null → heatmapActive=false → the whole
        // colour layer vanishes, even though older buckets DO have colour. That is exactly
        // the khoy case: colour shows on a 7-day window but NOT on a 24h window whose last
        // hour was idle (confirmed: 1h enc_min/max=None, but range probe = 50..21,698,506).
        // Take the domain from _progressiveColorRange (the range probe = full data min/max,
        // already seeded just above); fall back to the payload only if the seed is degenerate.
        // NOTE: this is the ONE piece of 4.0.278 worth keeping — the cell-clearing that broke
        // colour is NOT re-applied.
        const _seedLo = (_progressiveColorRange && Number.isFinite(Number(_progressiveColorRange.lo))) ? Number(_progressiveColorRange.lo) : null;
        const _seedHi = (_progressiveColorRange && Number.isFinite(Number(_progressiveColorRange.hi))) ? Number(_progressiveColorRange.hi) : null;
        const _seedOk = (_seedLo != null && _seedHi != null && _seedHi > _seedLo);
        const _payMin = (axisMode === 'encoder')
            ? (_hmRaw.enc_min != null ? Number(_hmRaw.enc_min) : null)
            : (_hmRaw.t_min   != null ? Number(_hmRaw.t_min)   : null);
        const _payMax = (axisMode === 'encoder')
            ? (_hmRaw.enc_max != null ? Number(_hmRaw.enc_max) : null)
            : (_hmRaw.t_max   != null ? Number(_hmRaw.t_max)   : null);
        const heatmapMin = _seedOk ? _seedLo : _payMin;
        const heatmapMax = _seedOk ? _seedHi : _payMax;
        // Alias kept so the rest of the plugin body reads cleanly. These are
        // X-axis values in whichever unit the active axis uses (encoder counts
        // OR epoch-millis), so the binning math is identical.
        const heatmapEncMin = heatmapMin, heatmapEncMax = heatmapMax;
        // 4.0.281 — bins must match the streamed cells' bin count (they're binned over
        // _progressiveColorRange.n_bins). An empty 1h payload has no n_bins → the old `|| 32`
        // fallback mispositioned the streamed cells (bin K drawn at K/32 instead of K/N).
        const heatmapBins = Math.max(1, Number(_hmRaw.n_bins)
            || (_progressiveColorRange && Number(_progressiveColorRange.n_bins))
            || 32);
        const heatmapBoundsCollapsed = (heatmapEncMin != null && heatmapEncMax != null &&
                                        heatmapEncMax === heatmapEncMin);
        // 4.0.161 — read the feature flag from THIS response first (server
        // is authoritative and can't race with loadColorConfig's background
        // fetch). Falls back to window._mveColorCheckEnabled === true only
        // when the payload doesn't carry the field (older MVE). Also mirror
        // the payload value into the window flag so the Advanced-tab toggle
        // and the toolbar-visibility handler agree with what the chart shows.
        const _colorEnabledFromPayload = (typeof data.parent_color_check_enabled === 'boolean')
            ? data.parent_color_check_enabled
            : (window._mveColorCheckEnabled === true);
        if (typeof data.parent_color_check_enabled === 'boolean') {
            window._mveColorCheckEnabled = data.parent_color_check_enabled;
        }
        // 4.0.169 — when the Parent Object Color Check feature is ON, register
        // the heatmap plugin unconditionally (was: gated on `heatmapCells > 0`).
        // Empty (cam, bin) slots now paint a subtle "no data" tile so the strip
        // feels continuous across the whole window instead of vanishing on the
        // stretch where no `_color` rows exist (e.g. shipments captured before
        // color-check was toggled on). Actual data cells overlay on top.
        // 4.0.301 — register the ΔE heatmap layer whenever colour-check is ON, NOT
        // gated on a valid [min,max]. Those bounds come from the range probe, which
        // RESETS on a flaky tunnel (see the operator's console: /health, /api/system/
        // metrics, range_only all ERR_CONNECTION_RESET). When it did, heatmapEncMin/Max
        // were null at THIS one-time check, so the plugin never registered and the
        // background never painted — even though the range became valid moments later
        // (scatter auto-widen) and the colour-CHANGE strip, which re-renders every
        // ladder tick, DID show. That's the exact "strip yes, cells no" the operator
        // saw. The plugin positions cells by bin over chartArea width (heatmapBins,
        // always ≥1) and no-ops on the time axis, so it does NOT need the bounds for
        // the main paint path — registering on colour-enabled alone is safe and makes
        // the background survive a failed/slow probe just like the strip already does.
        const heatmapActive = _colorEnabledFromPayload;
        const _heatmapPlugins = heatmapActive ? [{
            id: 'colorHeatmap',
            beforeDatasetsDraw(chart) {
                const ctx = chart.ctx;
                const xScale = chart.scales && chart.scales.x;
                const yScale = chart.scales && chart.scales.y;
                if (!ctx || !xScale || !yScale) return;
                // 4.0.185 — operator can hide the whole ΔE cell layer via the
                // "Cells: 👁 Show / 🚫 Hide" pill without dropping the scatter dots.
                if ((localStorage.getItem('mve_heatmap_cells') || 'shown') === 'hidden') return;
                // 4.0.253 — skip the ΔE cells on the TIME axis. They bin by ENCODER
                // position; on a stationary line (encoder=0 → axis falls back to time)
                // every cell collapses to bin 0 and piles up at the LEFT edge, which
                // reads as "data on the left while the right isn't loaded". No encoder
                // position ⇒ no meaningful heatmap, so don't paint it.
                if (window._mveUnifiedX && window._mveUnifiedX.axisMode === 'time') return;
                const ca = chart.chartArea;
                // 4.0.164 — read cells from the mutable __mveHeatmap cache
                // so the progressive per-bucket loader can push new cells
                // without recreating the chart. Falls back to the closure-
                // captured array on the first render before __mveHeatmap
                // is attached.
                const _liveCells = (chart.data && chart.data.__mveHeatmap && Array.isArray(chart.data.__mveHeatmap.cells))
                    ? chart.data.__mveHeatmap.cells
                    : heatmapCells;
                // 4.0.259 — if EVERY ΔE cell landed in the SAME bin, the layer is just a
                // solid block at one column (happens on a flat-encoder / stationary line
                // where the encoder-binned cells all collapse to bin 0 = the left block the
                // operator kept seeing). It carries no positional info, so skip painting it.
                if (_liveCells.length > 1) {
                    const _b0 = _liveCells[0] && _liveCells[0].bin;
                    let _allSame = true;
                    for (let _i = 1; _i < _liveCells.length; _i++) { if (_liveCells[_i].bin !== _b0) { _allSame = false; break; } }
                    if (_allSame) return;
                }
                const binWidth = (heatmapEncMax - heatmapEncMin) / heatmapBins;
                // 4.0.177 — colour scale mode. In 'relative' mode we normalise the
                // 2/5/10 CIELAB tolerance bands to the span's own max |Δ|, so
                // a very stable window (all cells |Δ|<2) still spreads across
                // green/yellow/orange/red and exposes its worst few cells.
                const _colorScaleMode = (localStorage.getItem('mve_heatmap_scale') || 'absolute');
                let _spanMaxAbs = 0;
                if (_colorScaleMode === 'relative') {
                    for (const c of _liveCells) {
                        const base = _activeBaselineMap[String(c.cam)];
                        const cellE = Number(c.E) || 0;
                        const dv = (_activeBaselineMode === 'shipment_avg' && Number.isFinite(Number(c.delta_from_avg)))
                            ? Number(c.delta_from_avg)
                            : Number(c.delta_e) || 0;
                        const d = (base && Number.isFinite(base.E))
                            ? Math.abs(cellE - Number(base.E))
                            : Math.abs(dv);
                        if (d > _spanMaxAbs) _spanMaxAbs = d;
                    }
                    if (_spanMaxAbs <= 0) _spanMaxAbs = 1; // avoid div-by-zero → falls back to absolute-ish
                }
                ctx.save();
                // 4.0.169 — paint a subtle "no data" tile at every (cam, bin)
                // slot that has no `_color` sample yet. Overlaid data cells
                // repaint on top with their actual hue below. Operator sees
                // a continuous strip across the whole window even in stretches
                // where MVE was off or color-check hadn't been enabled yet.
                if (ca && heatmapBins > 0) {
                    const cw = ca.right - ca.left;
                    const bw = cw / heatmapBins;
                    const covered = new Set();
                    for (const c of _liveCells) covered.add(String(c.cam) + '|' + String(c.bin));
                    // Iterate visible cameras (Y-axis rows) × all bins
                    ctx.fillStyle   = 'rgba(148, 163, 184, 0.10)';
                    ctx.strokeStyle = 'rgba(15, 23, 42, 0.35)';
                    ctx.lineWidth   = 0.5;
                    for (const [cam, idx] of idxByCam.entries()) {
                        const yT = yScale.getPixelForValue(idx - 0.5);
                        const yB = yScale.getPixelForValue(idx + 0.5);
                        const y  = Math.min(yT, yB);
                        const h  = Math.abs(yB - yT);
                        for (let b = 0; b < heatmapBins; b++) {
                            if (covered.has(String(cam) + '|' + b)) continue;
                            const x = ca.left + b * bw;
                            ctx.fillRect(x, y, bw, h);
                            ctx.strokeRect(x + 0.25, y + 0.25, bw - 0.5, h - 0.5);
                        }
                    }
                }
                for (const cell of _liveCells) {
                    const idx = idxByCam.has(cell.cam) ? idxByCam.get(cell.cam) : null;
                    if (idx == null) continue;
                    let x, y, w, h;
                    if (ca && heatmapBins > 1) {
                        // 4.0.302 — direct pixel positioning is now the PRIMARY path. Cells are
                        // pre-binned into heatmapBins over the display range, so bin K sits at
                        // ca.left + K*chartWidth/heatmapBins — 1:1 with the scatter dots and the
                        // strips below, with NO dependence on heatmapEncMin/Max. Those bounds
                        // come from the range probe, which on a flaky tunnel can RESET or return
                        // COLLAPSED (enc_min===enc_max); the old order hit the single-bin fallback
                        // below and painted EVERY cell full-width → the whole chart went solid
                        // green (operator: "it just create one bucket"). Bin positioning can't
                        // collapse, so it's the safe default whenever there's >1 bin.
                        const chartWidth = ca.right - ca.left;
                        const binPixelWidth = chartWidth / heatmapBins;
                        x = ca.left + cell.bin * binPixelWidth;
                        w = binPixelWidth;
                    } else if (heatmapBoundsCollapsed || binWidth <= 0) {
                        // Genuine single-bin (heatmapBins<=1) or no chartArea: paint the whole
                        // chart-area row for this camera. Useful when the line is stopped.
                        x = ca ? ca.left : 0;
                        w = ca ? (ca.right - ca.left) : chart.width;
                    } else if (ca) {
                        // v4.0.149 — direct pixel positioning across chartArea.
                        // Previously used xScale.getPixelForValue(heatmapMin +
                        // bin*binWidth) which returned wrong pixels when the
                        // payload bounds and the chart X-axis bounds were even
                        // slightly out of sync (operator reported: on time
                        // axis, all 48 bins clustered on the right edge of
                        // the chart, `start and stop of each bin` looked
                        // wrong). Direct math gives an unambiguous 1:1 mapping:
                        // bin K sits at `chartArea.left + K * chartWidth /
                        // heatmapBins`, always spread across the full width,
                        // 1:1 with the strip cells below.
                        const chartWidth = ca.right - ca.left;
                        const binPixelWidth = chartWidth / heatmapBins;
                        x = ca.left + cell.bin * binPixelWidth;
                        w = binPixelWidth;
                    } else {
                        const encL = heatmapEncMin + cell.bin * binWidth;
                        const encR = heatmapEncMin + (cell.bin + 1) * binWidth;
                        const xL = xScale.getPixelForValue(encL);
                        const xR = xScale.getPixelForValue(encR);
                        x = Math.min(xL, xR);
                        w = Math.abs(xR - xL);
                    }
                    const yT = yScale.getPixelForValue(idx - 0.5);
                    const yB = yScale.getPixelForValue(idx + 0.5);
                    y = Math.min(yT, yB);
                    h = Math.abs(yB - yT);
                    // Clamp to chartArea so we don't paint over axis ticks
                    if (ca) {
                        if (x + w < ca.left || x > ca.right || y + h < ca.top || y > ca.bottom) continue;
                    }
                    // 4.0.30 — ΔE against the server-computed baseline for the
                    // active mode. Bands aligned to industrial CIELAB tolerances:
                    // ≤2 green, ≤5 yellow, ≤10 orange, >10 red.
                    // 4.0.173 — shipment_avg mode uses per-cell `delta_from_avg`
                    // (whole-shipment median baseline) instead of the SQL's
                    // default `delta_e` (which is per-shipment-start).
                    const base = _activeBaselineMap[String(cell.cam)];
                    const cellE = Number(cell.E) || 0;
                    const _cellDefaultDelta = (_activeBaselineMode === 'shipment_avg' && Number.isFinite(Number(cell.delta_from_avg)))
                        ? Number(cell.delta_from_avg)
                        : Number(cell.delta_e) || 0;
                    const de = (base && Number.isFinite(base.E))
                        ? Math.abs(cellE - Number(base.E))
                        : Math.abs(_cellDefaultDelta);
                    // 4.0.177 — pick bands based on colour-scale mode.
                    // Absolute (default): fixed 2/5/10 CIELAB tolerance bands.
                    // Relative: same 20%/50%/100% shape but on the span's own max.
                    let _b1, _b2, _b3;
                    if (_colorScaleMode === 'relative') {
                        _b1 = _spanMaxAbs * 0.20;
                        _b2 = _spanMaxAbs * 0.50;
                        _b3 = _spanMaxAbs * 1.00;   // > b3 shouldn't happen since spanMax is our max
                    } else {
                        _b1 = 2; _b2 = 5; _b3 = 10;
                    }
                    let hue;
                    if (de <= _b1)       hue = 120;        // green
                    else if (de <= _b2)  hue = 60;         // yellow
                    else if (de <= _b3)  hue = 30;         // orange
                    else                 hue = 0;          // red
                    // v4.0.149 — alpha bumped 0.35 → 0.55 so low-ΔE cells
                    // (green over the dark chart background) are visible,
                    // not just the loud red/orange ones.
                    ctx.fillStyle = `hsla(${hue}, 65%, 45%, 0.55)`;
                    ctx.fillRect(x, y, w, h);
                    // 4.0.38 — thin dark stroke so each cell is visually
                    // distinct from its neighbours. Without this, adjacent
                    // cells with similar dE blend into wide bands and the
                    // operator can't see that the bucket count matches the
                    // quality / ejection strips below.
                    ctx.strokeStyle = 'rgba(15, 23, 42, 0.55)';
                    ctx.lineWidth = 0.5;
                    ctx.strokeRect(x + 0.25, y + 0.25, w - 0.5, h - 0.5);
                    // 4.0.170 — inline signed Δ label (↑0.3 / ↓0.5), same idea
                    // as the color-change strip below. Signed value: prefer
                    // baseline_map delta (E - baseline.E) when the operator
                    // has picked Camera / Shipment start / Target /
                    // Reference mode; falls back to the SQL's `delta_e`
                    // (per-shipment-start signed avg since v4.0.163).
                    // Hidden when the cell is too narrow (< 18 px) or too
                    // short (< 12 px), or when |Δ| < 0.05 (noise floor).
                    const signedDelta = (base && Number.isFinite(base.E))
                        ? (cellE - Number(base.E))
                        : _cellDefaultDelta;
                    if (Math.abs(signedDelta) >= 0.05 && w >= 18 && h >= 12) {
                        const mag = Math.abs(signedDelta);
                        const valStr = mag >= 10 ? mag.toFixed(0) : mag.toFixed(1);
                        const label = (signedDelta > 0 ? '↑' : '↓') + valStr;
                        // 4.0.172 — bumped contrast: 11 px (was 9), full-opacity
                        // white halo outline (was 0.75), full-black ink (was
                        // 0.95). Operator reported labels felt washed out
                        // against green/orange hues at 9 px + 0.75 halo.
                        ctx.font = '700 11px system-ui, -apple-system, sans-serif';
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'top';
                        const labelY = y + 3;
                        ctx.strokeStyle = 'rgba(255, 255, 255, 1)';
                        ctx.lineWidth = 3;
                        ctx.lineJoin = 'round';
                        ctx.strokeText(label, x + w / 2, labelY);
                        ctx.fillStyle = 'rgba(0, 0, 0, 1)';
                        ctx.fillText(label, x + w / 2, labelY);
                    }
                }
                // 4.0.32 — vertical marker at the operator-picked reference
                // point. Only painted when the axis matches what was picked
                // (encoder marker on encoder axis, time marker on time axis).
                if (_refPos && _refPos.axis === axisMode &&
                    Number.isFinite(Number(_refPos.value))) {
                    const refX = xScale.getPixelForValue(Number(_refPos.value));
                    const ca = chart.chartArea;
                    if (refX >= ca.left && refX <= ca.right) {
                        ctx.save();
                        ctx.strokeStyle = 'rgba(245, 158, 11, 0.95)';
                        ctx.lineWidth = 2;
                        ctx.setLineDash([5, 4]);
                        ctx.beginPath();
                        ctx.moveTo(refX, ca.top);
                        ctx.lineTo(refX, ca.bottom);
                        ctx.stroke();
                        ctx.setLineDash([]);
                        ctx.fillStyle = 'rgba(245, 158, 11, 0.95)';
                        ctx.font = '10px sans-serif';
                        ctx.fillText('🖼 baseline', Math.min(refX + 4, ca.right - 60), ca.top + 12);
                        ctx.restore();
                    }
                }
                ctx.restore();
            }
        }] : [];
        // v4.0.111 — force encoder axis min/max to the color-heatmap's
        // enc_min/enc_max (which after v4.0.109 IS the same as the
        // quality/ejection strip's range). Without this, Chart.js auto-
        // scaled the encoder axis to fit the dot data + padding, so the
        // color heatmap bin edges got squeezed into a narrower pixel
        // range than the strip cells below (which use the strip
        // container's full width, already aligned to the plot area).
        // Operators saw the color cells as "narrower" and misaligned
        // with the strips. Locking the axis to the heatmap range makes
        // both use the SAME encoder-to-pixel mapping — 1 bucket in the
        // heatmap = 1 cell in the strip, edge-to-edge.
        //
        // Time axis already had this behaviour via _scatterXMin/Max.
        // v4.0.140 — length-mode axis shows the operator's chosen unit.
        // Underlying X values stay in cumulative encoder counts so all the
        // pixel math, heatmap bins, and shipment_length_offsets alignment
        // keep working; only the TICK labels are divided by the operator's
        // upu and appended with the operator's unit label.
        const _upuForAxis   = _upuFromAny(null, data);
        const _unitForAxis  = _unitLabelFromAny(null, data);
        const _lengthTickCb = (v) => _fmtLenWithUpu(v, _upuForAxis, _unitForAxis);
        // ---- 4.0.197 UNIFIED X DOMAIN — single source of truth for the
        // scatter axis AND every strip below it. Derived from the ACTUAL dot
        // extent (min/max over the returned points for the active axis) so:
        //   (a) the dots fill the plot area edge-to-edge — fixes "picked an
        //       old shipment, dots don't fill the span" because the axis is
        //       the shipment's own extent, NOT the sliding [now-window, now];
        //   (b) the full-width-spread strips (colour cells, quality, ejection)
        //       line up at the endpoints by construction;
        //   (c) the value-based shipment-lane strip is handed the identical
        //       [min,max] so bar edges sit exactly under the dots (zero offset).
        // Previously the axis was pinned to color_heatmap.enc_min/enc_max
        // (full-inference range incl. idle frames) in encoder mode, and to
        // Date.now()±window in time mode — both diverged from the dot cloud.
        // 4.0.198 — derive the axis from an AUTHORITATIVE, COMPLETE source,
        // NEVER from the incrementally-loaded dot set. The progressive loader
        // adds OLDER dots AFTER Core renders, so "dot-extent at render time"
        // collapsed the axis to the 1h-hydrate window (regression in 4.0.197:
        // all-shipments 24h showed a 4-minute axis). Rules:
        //   • specific shipment picked → its FULL extent from shipment_spans
        //     (complete even when dots are capped/progressive) → fills the span;
        //   • all shipments, time → the sliding window [now-window, now] so
        //     progressively-loaded older dots have room to land;
        //   • all shipments, encoder/length → heatmap enc_min/enc_max (full
        //     inference range), same reason.
        // 4.0.201 — BULLETPROOF picked-shipment axis. Core's single fetch
        // returns dots stratified across the WHOLE shipment span (verified:
        // 750 dots spread 13%→100% of a 30-min shipment), and shipment_spans
        // carries the authoritative [t_min,t_max]/[enc_min,enc_max]. Use the
        // UNION of both so the axis = the shipment's true start→end no matter
        // which source is present: min = the leftmost of (span start, first
        // dot), max = the rightmost of (span end, last dot). This is exactly
        // the operator's spec: "min = the most-left number in the span, max =
        // the most-right." No fragile shipment-id matching, no fallback to the
        // sliding window that jammed the dots against the right edge.
        const _shipSel = ((document.getElementById('insight-shipment') || {}).value) || '';
        const _spansForAxis = data.shipment_spans || [];
        const _sp0 = _spansForAxis[0] || null;   // picked-shipment response has exactly one span
        let _uxMin, _uxMax;
        const _uXs = points.map(p => Number(p.x)).filter(Number.isFinite);
        const _dotLo = _uXs.length ? Math.min.apply(null, _uXs) : null;
        const _dotHi = _uXs.length ? Math.max.apply(null, _uXs) : null;
        if (_shipSel) {
            // Span bound for the active axis (time → t_*, encoder/length → enc_*).
            const _spanLo = _sp0 ? Number(axisMode === 'time' ? _sp0.t_min : _sp0.enc_min) : NaN;
            const _spanHi = _sp0 ? Number(axisMode === 'time' ? _sp0.t_max : _sp0.enc_max) : NaN;
            const _cand = [];
            if (Number.isFinite(_spanLo)) _cand.push(_spanLo);
            if (Number.isFinite(_dotLo))  _cand.push(_dotLo);
            _uxMin = _cand.length ? Math.min.apply(null, _cand) : (_dotLo != null ? _dotLo : 0);
            const _candHi = [];
            if (Number.isFinite(_spanHi)) _candHi.push(_spanHi);
            if (Number.isFinite(_dotHi))  _candHi.push(_dotHi);
            _uxMax = _candHi.length ? Math.max.apply(null, _candHi) : (_dotHi != null ? _dotHi : _uxMin + 1);
        } else if (axisMode === 'time') {
            _uxMin = _scatterXMin; _uxMax = _scatterXMax;
        } else {
            // v4.0.228 — prefer the SELECTED WINDOW's encoder span (from
            // shipment_spans, stashed by refreshAdvancedCharts) over the 1h
            // hydrate's heatmap bounds, so 24h shows EVERY shipment in the window
            // rather than just the last hour. Union with the on-chart dots for
            // safety (dots always fall within the span).
            const _ewr = _encWinRange();
            if (_ewr) {
                _uxMin = (_dotLo != null) ? Math.min(_ewr.lo, _dotLo) : _ewr.lo;
                _uxMax = (_dotHi != null) ? Math.max(_ewr.hi, _dotHi) : _ewr.hi;
            } else {
                _uxMin = (heatmapEncMin != null) ? heatmapEncMin
                       : (_uXs.length ? _dotLo : 0);
                _uxMax = (heatmapEncMax != null) ? heatmapEncMax
                       : (_uXs.length ? _dotHi : _uxMin + 1);
            }
        }
        // Pad the domain by 1% each side so edge dots aren't clipped by the
        // plot border, then guarantee a positive span.
        if (_uxMax > _uxMin) {
            const _pad = (_uxMax - _uxMin) * 0.01;
            _uxMin -= _pad; _uxMax += _pad;
        } else {
            _uxMax = _uxMin + 1;
        }
        // Expose for the standalone strip refreshers + strip bucket loaders.
        window._mveUnifiedX = { axisMode, xMin: _uxMin, xMax: _uxMax, bins: heatmapBins };
        const xScaleCfg = axisMode === 'time'
            ? { type: 'linear', min: _uxMin, max: _uxMax,
                ticks: { color: '#94a3b8', font: { size: 9 }, maxRotation: 40, minRotation: 0, callback: (v) => {
                    // 4.0.261 — TIME axis: every tick shows DATE + time (operator: "when time
                    // is selected the x-axis numbers should be date and time"). Encoder shows raw
                    // encoder values; length shows the Advanced-tab unit (both handled below).
                    const _d = new Date(v);
                    return _d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' +
                           _d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                } },
                grid: { color: 'rgba(148,163,184,0.08)' } }
            : axisMode === 'length'
                ? { type: 'linear',
                    min: _uxMin, max: _uxMax,
                    ticks: { color: '#94a3b8', font: { size: 9 }, callback: _lengthTickCb },
                    grid: { color: 'rgba(148,163,184,0.08)' },
                    title: { display: true, text: _upuForAxis > 0 ? `Length (${_unitForAxis}, cumulative across shipments)` : 'Length (cumulative raw counts, no calibration)', color: '#94a3b8', font: { size: 10 } } }
                : { type: 'linear',
                    min: _uxMin, max: _uxMax,
                    ticks: { color: '#94a3b8', font: { size: 9 } },
                    grid: { color: 'rgba(148,163,184,0.08)' },
                    title: { display: true, text: 'Encoder (roll position)', color: '#94a3b8', font: { size: 10 } } };
        const titleText = axisMode === 'time'
            ? 'Camera × time — hover for image, click dot to open the exact frame'
            : axisMode === 'length'
                ? 'Camera × length — hover for image, click dot for the exact frame'
                : 'Camera × encoder (roll position) — hover for image, click dot for the exact frame';
        _insightCameraScatter = new Chart(scCtx, {
            // v4.0.134 — was 'bubble' (circle-only, size by r). scatter type
            // supports per-point pointStyle for the defect/parent/marker/
            // context shape scheme while keeping the same {x,y,r} data shape.
            type: 'scatter',
            // 4.0.164 — expose the mutable heatmap cache to the colorHeatmap
            // plugin via chart.data. The progressive loader replaces
            // chart.data.__mveHeatmap.cells and calls chart.update('none')
            // per bucket to repaint newest→oldest.
            //
            // 4.0.165 — start EMPTY. Prior code seeded from the 1h hydrate
            // payload's heatmapCells which painted the full chart width at
            // load (bin K → K/N*width regardless of what time it represents)
            // and made per-bucket loading visually invisible. Progressive
            // ladder now populates cells one bucket at a time, newest→oldest,
            // bin-aligned to the FULL target window (see _seedProgressiveColorCache).
            data: Object.assign({ datasets }, {
                __mveHeatmap: {
                    // 4.0.205 — for a PICKED shipment, seed with the FULL colour
                    // set from this fetch (the backend slices over the whole
                    // shipment span, so `heatmapCells` already covers every bin).
                    // The picked-shipment ladder skips per-bucket colour streaming
                    // (v4.0.204), so without this seed the heatmap stayed EMPTY →
                    // "picked shipment shows no colour". All-shipments still starts
                    // empty and fills progressively via the ladder (see 4.0.165).
                    cells: shipment ? heatmapCells : [],
                    encMin: (_progressiveColorRange && _progressiveColorRange.lo) || heatmapEncMin,
                    encMax: (_progressiveColorRange && _progressiveColorRange.hi) || heatmapEncMax,
                    bins: (_progressiveColorRange && _progressiveColorRange.n_bins) || heatmapBins,
                },
            }),
            plugins: _heatmapPlugins,
            options: {
                responsive: true, maintainAspectRatio: false,
                // 4.0.5 — explicit interaction config so the external tooltip
                // fires on hover-near-dot (Chart.js bubble default `intersect:true`
                // can miss small bubbles when the cursor sits just outside the
                // hit-circle). `mode:'nearest'` + `intersect:false` = "find the
                // closest point and treat as hovered" even when not strictly inside.
                interaction: { mode: 'nearest', intersect: false, axis: 'xy' },
                hover:       { mode: 'nearest', intersect: false, axis: 'xy' },
                // 4.0.44 — chart-wide layout padding kept at zero; the title
                // and legend live OUTSIDE the canvas now (as HTML siblings
                // above the canvas) so the colour heatmap can't reach them.
                plugins: {
                    legend: { display: false },
                    title:  { display: false },
                    tooltip: { enabled: false, external: _scatterImageTooltip }
                },
                onClick: (evt, els, chart) => {
                    // 4.0.32 — if the operator is in baseline-pick mode, the
                    // next click sets the reference point instead of opening
                    // the annotate modal. Ignores the dot hit-test so the
                    // operator can pick ANY spot, not just where a dot is.
                    // 4.0.42 — also treat "pick hint is visible" as pick mode.
                    // Belt-and-suspenders against any path that displays the
                    // hint without setting the global flag (or vice-versa).
                    const hintEl = document.getElementById('hm-pick-hint');
                    const hintVisible = hintEl && hintEl.style.display !== 'none' && hintEl.offsetParent !== null;
                    if (window._pickingBaseline || hintVisible) {
                        _onBaselinePickClick(chart, evt, axisMode);
                        return;
                    }
                    if (!els || !els.length) return;
                    const dp = chart.data.datasets[els[0].datasetIndex].data[els[0].index];
                    if (!dp) return;
                    const cls = (chart.data.datasets[els[0].datasetIndex].label) || dp.cls || '';
                    // 4.0.2 — clicking a chart dot goes STRAIGHT to the LSF editor
                    // instead of the read-only image drawer. Operator can correct boxes
                    // and ship to the trainer in two clicks, no intermediate viewer.
                    if (axisMode === 'time') {
                        openFrameInAnnotator({
                            image_path: dp.img, shipment: dp.ship, t: dp.x,
                            cls: cls, classes: cls ? [cls] : [], best_confidence: dp.r || 0,
                            pipeline_task_id: _pipeTaskById(dp.pid),   // 4.0.296 — the dot's OWN pipeline task
                        });
                    } else {
                        // encoder mode: dot has no timestamp — parse from filename if present.
                        let t = null;
                        try {
                            const fn = String(dp.img || '').split('/').pop() || '';
                            const m = fn.match(/^(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})/);
                            if (m) t = Date.UTC(+m[1], +m[2]-1, +m[3], +m[4], +m[5], +m[6]);
                        } catch (e) {}
                        // 4.0.24 — `dp.y` is the display index now; use
                        // `dp.cam_id` for the real camera ID we carry alongside.
                        openFrameInAnnotator({
                            image_path: dp.img, shipment: dp.ship, t: t,
                            cls: cls, classes: cls ? [cls] : [], best_confidence: dp.r || 0,
                            encoder: dp.x, camera_index: (dp.cam_id != null ? dp.cam_id : dp.y),
                            pipeline_task_id: _pipeTaskById(dp.pid),   // 4.0.296 — the dot's OWN pipeline task
                        });
                    }
                },
                scales: {
                    x: xScaleCfg,
                    // 4.0.24 — tick callback maps the plotted Y (display index)
                    // back to the actual camera ID for axis labels, so the axis
                    // reads the operator's configured order (e.g. 5, 1, 6, 2,
                    // 4, 3) instead of the bare numeric sort 1, 2, 3, 4, 5, 6.
                    y: {
                        title: { display: true, text: 'Camera', color: '#94a3b8' },
                        // 4.0.172 — pin the Y range to the full camera_y_order
                        // so hiding all dots of a class via the legend can't
                        // drop that class's exclusive cameras from the axis.
                        // Operator reported: click TB in legend → cam 3 row
                        // disappeared because cam 3 only had TB dots + a
                        // colour-cell row; Chart.js auto-scaled Y to fit the
                        // remaining datasets, taking the colour cells with it.
                        min: -0.5,
                        max: Math.max(0.5, (camOrder && camOrder.length ? camOrder.length : 1) - 0.5),
                        ticks: {
                            color: '#94a3b8', stepSize: 1, precision: 0,
                            callback: (v) => {
                                const cid = camByIdx.get(v);
                                return (cid != null) ? String(cid) : String(v);
                            }
                        },
                        grid: { color: 'rgba(148,163,184,0.1)' }
                    }
                },
                // Align the merged strip wrap (which contains quality + ejection strips)
                // to the scatter's chartArea on every render + resize.
                // v4.0.113 — also align the new shipment-strip-wrap and re-run
                // its label visibility after margins settle so labels hide/show
                // correctly for the final pixel width.
                animation: { onComplete: function() {
                    _alignStripToScatter(this, 'quality-strip-wrap');
                    _alignStripToScatter(this, 'shipment-strip-wrap');
                    if (typeof _relayoutShipmentStripLabels === 'function') _relayoutShipmentStripLabels();
                } },
                onResize: function() { setTimeout(() => {
                    _alignStripToScatter(this, 'quality-strip-wrap');
                    _alignStripToScatter(this, 'shipment-strip-wrap');
                    if (typeof _relayoutShipmentStripLabels === 'function') _relayoutShipmentStripLabels();
                }, 0); }
            }
        });
        // 4.0.7 — direct mousemove hover-preview, independent of Chart.js's
        // external tooltip plugin (which wasn't firing reliably).
        _attachHoverPreview(scCtx, () => _insightCameraScatter);
        // v4.0.106 — normalize the initial dot radii per-class right after
        // creation. The insertion formula (line ~1738) used the fixed
        // `3 + conf*9` for backward compat; this pass rewrites `.r` on
        // every dot to the class-local [3, 12] normalization so class-A
        // dots with conf 0.2-0.4 look distinct from class-B dots with
        // conf 0.7-0.95 (both classes get their own full-range spread).
        try {
            _renormalizeScatterRadii(_insightCameraScatter);
            _insightCameraScatter.update('none');
        } catch (e) { /* keep the chart if renormalize fails */ }
        // 4.0.246 — visible dots diagnostic (base-render state, before the ladder
        // blanks + refills). See _paintDotsDiag.
        try { _insightCameraScatter.__mveAxis = axisMode; _paintDotsDiag('core'); } catch (_diag) {}
    }
    // v4.0.113 — populate the shipment-lane strip that sits between the
    // scatter and the quality strip.
    // 4.0.197 — feed the UNIFIED domain (_uxMin/_uxMax, the actual dot
    // extent) instead of heatmapEncMin/Max + _scatterXMin/Max. This is the
    // fix for the shipment-lane-offset bug: previously the strip got the
    // color-heatmap's full-inference encoder range (or the sliding time
    // window) while the dots were drawn on a different domain, so lane bar
    // edges sat off the dots. Now both share exactly [_uxMin,_uxMax], so a
    // bar edge at encoder value E lands on the same pixel as a dot at E.
    // Pass the pair matching the ACTIVE axis; the other pair stays null so
    // the shipment-lane renderer picks the right mode.
    // 4.0.207 — derive the shipment lanes from the SCATTER'S OWN DOTS instead
    // of a separate /api/shipment_spans fetch. Operator: "show the shipment
    // lane based on the data inside the scatter chart so we're sure they're in
    // sync on the x-axis." This guarantees:
    //   • the lanes span EXACTLY the encoder/time range the dots occupy (no
    //     more lane vs scatter range mismatch), and
    //   • the lanes never "switch to 24h" on their own — they're rebuilt from
    //     whatever dots are currently on the chart, in the same [_uxMin,_uxMax]
    //     domain, so the standalone 30s refresher can't clobber them with a
    //     different-window fetch.
    try { _renderShipmentStripFromScatterDots(); }
    catch (_stripE) { /* never let strip render break the chart */ }
    // 4.0.42 — hide the loading badge now that the scatter + heatmap have
    // been built. Paired with mveLoaderBegin() at the top of this function.
    if (_showedLoader) mveLoaderEnd();
}

// 4.0.207 — build shipment lanes from the dots currently on the scatter.
// Each dot carries `ship` + `x` (encoder value / epoch-ms / cumulative
// length depending on axis). Per-shipment [min x, max x] over the dots IS
// the lane span, in the exact domain the scatter draws → guaranteed sync.
// v4.0.214 — BIN the shipment lane exactly like the quality / colour / ejection
// strips (operator: "make the shipment lane like the other strips with bins;
// each bin might have one or two shipments — give the bin to the biggest
// shipment in that bin"). We split the scatter's own [xMin,xMax] domain into
// window._mveUnifiedX.bins bins (the SAME bins the other strips + the scatter
// use), count each shipment's dots per bin, and award each bin to the shipment
// with the MOST dots in it. Bars are snapped to bin edges, so a dot that lands
// in bin K sits inside its shipment's bar in bin K — the lane can no longer
// drift from the dots the way value-based positioning did.
// v4.0.217 — the shipment lane is now a BACKEND bin-strip: its bins come from
// /api/detection_charts (shipment_heatmap / shipment_slice), computed by
// _dc_shipment_cells over the IDENTICAL [lo,hi]/N_BINS domain the colour heatmap
// + scatter use, then painted by _paintShipmentStrip via _repaintProgressiveStrips.
// Because it bins over the same domain as quality/ejection/colour, a vertical
// guideline at bin K hits the same data in every layer — guaranteed x-sync,
// exactly what the operator asked for ("make the shipment lane like the other
// strips with bins ... each bin related to each shipment should get same color").
//
// This former dots-derived renderer is retired to a thin delegate so every
// legacy caller (Core render, the 30 s standalone refresher, the axis-toggle
// repaint) just repaints the backend strip from the streamed accumulator
// instead of re-deriving lanes from the sparse dots (which drifted because the
// dots are LIMIT-capped per bucket, not a full census of each bin).
function _renderShipmentStripFromScatterDots() {
    const title = document.getElementById('shipment-strip-title');
    if (title) {
        title.style.display = '';
        title.innerHTML = '🚚 ' + _stripTipIcon('shipment lanes — each bin is its dominant shipment, coloured by that shipment’s verdict (green=release, amber=re-inspect, red=hold, grey=pending). Binned to the scatter, in sync with the dots + strips.');
    }
    try { _repaintProgressiveStrips(); } catch (_e) { /* strip repaint is best-effort */ }
}
window._renderShipmentStripFromScatterDots = _renderShipmentStripFromScatterDots;


// v4.0.115 — Shipment strip renderer with multi-lane overlap handling
// and self-derived axis bounds. One bordered box per shipment span, laid
// out across the scatter's x-axis so each box sits under the dots it
// produced. Overlapping shipments (parallel loading, rewind, merged
// rolls) stack in separate lanes so every ID stays visible. Deterministic
// hue per shipment ID via golden-angle hash. Style intentionally matches
// the operator's requested design: colored border + tinted background,
// full shipment ID in matching hue text.

// v4.0.119 — INDEPENDENT strip refresher. Hits `/api/shipment_spans`
// (a standalone endpoint that only runs the bounded CTE — ~70 ms) and
// paints the strip. Runs on page load and every 30 s afterwards,
// completely decoupled from `_refreshAdvancedChartsCore` which does the
// full 24 h fetch that can take >30 s on busy sites. Without this the
// strip could never paint until the full dashboard fetch resolved.
// v4.0.120 — also cache the last-fetched spans so an axis toggle can
// re-render INSTANTLY without waiting for a network round-trip. Toggle
// hook below in setInsightAxis re-renders from cache immediately, then
// kicks a background refresh to pick up any new shipments.
let _shipmentStripTimer = null;
let _lastShipmentSpans = [];
async function _refreshShipmentStripStandalone() {
    // 4.0.207 — the shipment lanes are now derived from the SCATTER'S OWN DOTS
    // (see _renderShipmentStripFromScatterDots). If the scatter already has
    // dots, just re-derive from them — DO NOT fetch /api/shipment_spans. This
    // is what killed the "lane suddenly switches from 6h to 24h" bug: the 30s
    // standalone timer used to re-fetch shipment_spans with the window selector
    // and clobber the dot-aligned strip. Now the strip is a pure function of
    // the dots that are actually on the chart, so it can never diverge.
    try {
        // v4.0.211 — the shipment lane is DERIVED PURELY from the scatter's own
        // dots (same /api/detection_charts data as the scatter), so it can never
        // desync on the x-axis. The old `/api/shipment_spans` fallback painted
        // stretched, full-width lanes over a DIFFERENT [min,max] than the scatter
        // (that's the "lane sticks to the start" bug), so it's GONE. If no dots
        // are on the chart yet, the lane simply stays blank until the ladder
        // starts streaming — never a mismatched fetch.
        _renderShipmentStripFromScatterDots();
    } catch (e) {
        console.warn('standalone shipment-strip refresh failed:', e);
    }
}
// v4.0.120 — instant re-render from cache in the current axisMode.
// Called by setInsightAxis on toggle so the operator sees the strip
// flip modes with no perceptible lag.
function _repaintShipmentStripFromCache() {
    try {
        // v4.0.211 — lanes are PURELY dots-derived (same /api/detection_charts
        // data as the scatter). No _lastShipmentSpans fallback — that painted a
        // stretched, x-desynced lane. Blank until dots exist.
        _renderShipmentStripFromScatterDots();
    } catch (_e) { /* swallow — strip re-render is best-effort */ }
}
window._repaintShipmentStripFromCache = _repaintShipmentStripFromCache;
window._refreshShipmentStripStandalone = _refreshShipmentStripStandalone;
// Kick off the first paint as soon as the DOM is ready.
(function _bootstrapStandaloneStrip() {
    const start = () => {
        _refreshShipmentStripStandalone();
        if (_shipmentStripTimer) clearInterval(_shipmentStripTimer);
        _shipmentStripTimer = setInterval(_refreshShipmentStripStandalone, 30000);
    };
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', start, { once: true });
    } else {
        // DOM already ready — defer to next tick so late-defined helpers exist.
        setTimeout(start, 0);
    }
})();

// Sibling helper: pick FULL / tail-8 / tail-4 / hidden based on the
// bar's current pixel width. Full-id preferred so the operator can
// eyeball the whole shipment number the way they wrote it on the roll.
function _relayoutShipmentStripLabels() {
    const strip = document.getElementById('shipment-strip');
    if (!strip) return;
    const parentW = strip.clientWidth || 0;
    if (parentW <= 0) return;
    strip.querySelectorAll('.mve-ship-bar').forEach((el) => {
        const wPct = parseFloat(el.getAttribute('data-w-pct')) || 0;
        const px   = (wPct / 100) * parentW;
        const lbl  = el.querySelector('.mve-ship-label');
        if (!lbl) return;
        const full = el.getAttribute('data-label-full') || '';
        const med  = el.getAttribute('data-label-med')  || full;
        const tail = el.getAttribute('data-label-tail') || full;
        // 4.0.167 — updated for 9 px font (was 11 px). Char width ~5.5 px.
        if (px < 14)                       { lbl.style.display = 'none'; }
        else if (px < 32)                  { lbl.style.display = ''; lbl.textContent = tail; }
        else if (px < full.length * 6)     { lbl.style.display = ''; lbl.textContent = med;  }
        else                               { lbl.style.display = ''; lbl.textContent = full; }
    });
}
window._relayoutShipmentStripLabels = _relayoutShipmentStripLabels;

// 3.25.13 — toggle the merged scatter's X-axis between time and encoder.
// Re-renders the scatter (destroy + rebuild), reloads the quality + ejection
// strips for the new axis, updates the strip titles + button styling.
// v4.0.239 — sync ONLY the Time|Encoder|Length toggle styling to a given axis,
// WITHOUT reloading. Shared by setInsightAxis AND the stationary-line guard, so the
// toggle follows the EFFECTIVE axis (shows Time when encoder/length falls back).
function _setAxisToggleUI(axis) {
    const tBtn = document.getElementById('insight-axis-toggle-time');
    const eBtn = document.getElementById('insight-axis-toggle-encoder');
    const lBtn = document.getElementById('insight-axis-toggle-length');
    const ACTIVE = 'linear-gradient(135deg,#10b981,#047857)', INACTIVE = 'rgba(51,65,85,0.5)';
    if (tBtn) { tBtn.style.background = axis === 'time'    ? ACTIVE : INACTIVE; tBtn.style.color = axis === 'time'    ? '#fff' : '#cbd5e1'; }
    if (eBtn) { eBtn.style.background = axis === 'encoder' ? ACTIVE : INACTIVE; eBtn.style.color = axis === 'encoder' ? '#fff' : '#cbd5e1'; }
    if (lBtn) { lBtn.style.background = axis === 'length'  ? ACTIVE : INACTIVE; lBtn.style.color = axis === 'length'  ? '#fff' : '#cbd5e1'; }
    // 4.0.271 — the dropdown must reflect the OPERATOR's pick (_insightAxis), NOT a
    // transient stationary-guard fallback. The guard flips encoder->time when the
    // encoder data is momentarily degenerate/empty (e.g. first paint before encoder
    // rows arrive); the old code overwrote the dropdown to Time, so on refresh the
    // operator saw "Time" even though they'd chosen Encoder. Only an EXPLICIT
    // setInsightAxis (axis === _insightAxis) is allowed to move the dropdown.
    const sel = document.getElementById('insight-axis-select');
    if (sel && axis === _insightAxis && sel.value !== axis) sel.value = axis;
}
function setInsightAxis(axis) {
    // v4.0.132 — added 'length' (cumulative length across shipments,
    // immune to PLC encoder resets).
    if (axis !== 'time' && axis !== 'encoder' && axis !== 'length') return;
    if (_insightAxis === axis) return;
    _insightAxis = axis;
    try { localStorage.setItem('mve_insight_axis', axis); } catch (e) {}   // 4.0.271 — persist the operator's pick across refreshes
    _setAxisToggleUI(axis);   // active-button styling (shared with the fallback sync)
    // Strip titles.
    // v4.0.216 — compact titles: emoji + a hover ⓘ (full description in tooltip).
    const _si = (emoji, desc) => emoji + ' ' + _stripTipIcon(desc);
    const qTitle = document.getElementById('quality-strip-title');
    if (qTitle) qTitle.innerHTML = axis === 'encoder'
        ? _si('📏', 'quality by encoder — find spots on the material with the most issues')
        : axis === 'length'
            ? _si('🧵', 'quality by cumulative length — sequential across shipments (PLC-reset immune)')
            : _si('📍', 'quality by time — hover a cell for top defect + score');
    const eTitle = document.getElementById('ejection-strip-title');
    if (eTitle) eTitle.innerHTML = axis === 'encoder'
        ? _si('⏏️', 'ejections by encoder — color = procedure; hover for breakdown')
        : axis === 'length'
            ? _si('⏏️', 'ejections by cumulative length — color = procedure; hover for breakdown')
            : _si('⏏️', 'ejections by time — color = procedure; hover for breakdown');
    // v4.0.120 — flip the shipment strip to the new axis INSTANTLY from
    // the cached spans (no network wait). Then kick a background refresh
    // to pick up any new shipments that may have started since the last
    // 30 s tick.
    if (typeof _repaintShipmentStripFromCache === 'function') _repaintShipmentStripFromCache();
    if (typeof _refreshShipmentStripStandalone === 'function') _refreshShipmentStripStandalone();
    // Re-fetch insight + re-render scatter + reload both strips for the new axis.
    if (typeof refreshDetectionInsights === 'function') refreshDetectionInsights();
}
window.setInsightAxis = setInsightAxis;

// 3.25.6 — given a Chart.js instance and a wrap div ID, inset the wrap to match the chart's plot area.
function _alignStripToScatter(scatter, wrapId) {
    try {
        const wrap = document.getElementById(wrapId);
        if (!scatter || !wrap) return;
        const ca = scatter.chartArea;
        const canvas = scatter.canvas;
        if (!ca || !canvas) return;
        const containerW = canvas.clientWidth || canvas.parentElement?.clientWidth || canvas.width;
        if (!containerW) return;
        const left  = Math.max(0, Math.round(ca.left));
        const right = Math.max(0, Math.round(containerW - ca.right));
        // v4.0.218b — inset the whole wrap to EXACTLY the plot area [chartArea.left,
        // chartArea.right]. Every strip is 100% of the wrap, so every strip starts
        // and ends on the same x as the scatter — no per-legend width to get wrong.
        // The legend icons are ABSOLUTELY positioned in the left gutter (CSS:
        // right:100%), out of flow, so they can't push any strip sideways. This is
        // what finally kills the "colour-change is more left than the others" drift:
        // a strip revealed LATE (colour-change is display:none until enabled) is
        // still exactly plot-width because it never depended on its legend's size.
        wrap.style.marginLeft  = left + 'px';
        wrap.style.marginRight = right + 'px';
    } catch (e) { /* ignore */ }
}
window._alignStripToScatter = _alignStripToScatter;

// ---------------------------------------------------------------------------
// Ejection Insights: ejections by procedure (bar + distribution doughnut) and
// ejections over time. Reads /api/ejection_stats — populated only by procedures
// with Store=ON (Process tab → Ejection Procedures). Scoped by window + shipment.
// ---------------------------------------------------------------------------
async function refreshEjectionCharts() {
    const window = document.getElementById('insight-window')?.value || '24h';
    const shipment = document.getElementById('insight-shipment')?.value || '';
    let data;
    try {
        const r = await fetch('/api/ejection_stats?window=' + encodeURIComponent(window) +
                              '&shipment=' + encodeURIComponent(shipment));
        data = await r.json();
    } catch (e) { console.error('ejection_stats fetch failed:', e); return; }

    const byProc = data.by_procedure || {};
    const timeline = data.timeline || [];
    const total = data.total || 0;
    const hasData = total > 0;

    const emptyEl = document.getElementById('ejection-empty');
    const chartsEl = document.getElementById('ejection-charts');
    const totalEl = document.getElementById('ejection-total');
    if (emptyEl) emptyEl.style.display = hasData ? 'none' : 'block';
    if (chartsEl) chartsEl.style.display = hasData ? 'grid' : 'none';
    if (totalEl) totalEl.textContent = hasData
        ? `${total} ejection(s) in window across ${Object.keys(byProc).length} procedure(s)`
        : '';
    if (!hasData) return;

    const procNames = Object.keys(byProc);
    const procCounts = procNames.map(k => byProc[k]);
    const procColors = procNames.map(n => _classColor(n));

    // ---- Ejections by procedure (bar) ----
    const barCtx = document.getElementById('ejection-proc-bar');
    if (barCtx) {
        if (_ejectionProcBar) _ejectionProcBar.destroy();
        _ejectionProcBar = new Chart(barCtx, {
            type: 'bar',
            data: { labels: procNames, datasets: [{ label: 'Ejections', data: procCounts, backgroundColor: procColors, borderRadius: 4 }] },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, title: { display: true, text: 'Ejections by procedure', color: '#cbd5e1', font: { size: 13 } } },
                scales: {
                    x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { display: false } },
                    y: { beginAtZero: true, ticks: { color: '#94a3b8', precision: 0 }, grid: { color: 'rgba(148,163,184,0.1)' } }
                }
            }
        });
    }

    // ---- Ejection distribution by procedure (doughnut) ----
    const pieCtx = document.getElementById('ejection-proc-pie');
    if (pieCtx) {
        if (_ejectionProcPie) _ejectionProcPie.destroy();
        _ejectionProcPie = new Chart(pieCtx, {
            type: 'doughnut',
            data: { labels: procNames, datasets: [{ data: procCounts, backgroundColor: procColors, borderColor: '#1e293b', borderWidth: 2 }] },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, position: 'right', labels: { color: '#cbd5e1', font: { size: 10 }, boxWidth: 12 } },
                    title: { display: true, text: 'Ejection distribution by procedure', color: '#cbd5e1', font: { size: 13 } },
                    tooltip: { callbacks: { label: (c) => {
                        const t = c.dataset.data.reduce((a, b) => a + b, 0) || 1;
                        return `${c.label}: ${c.raw} (${(c.raw / t * 100).toFixed(1)}%)`;
                    } } }
                }
            }
        });
    }

    // ---- Ejections over time (line) ----
    const tlCtx = document.getElementById('ejection-timeline');
    if (tlCtx) {
        if (_ejectionTimeline) _ejectionTimeline.destroy();
        _ejectionTimeline = new Chart(tlCtx, {
            type: 'line',
            data: {
                labels: timeline.map(p => p.t),
                datasets: [{
                    label: 'Ejections', data: timeline.map(p => p.count),
                    borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.15)',
                    fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, title: { display: true, text: 'Ejections over time', color: '#cbd5e1', font: { size: 13 } } },
                scales: {
                    x: { ticks: { color: '#94a3b8', font: { size: 9 }, maxTicksLimit: 8 }, grid: { display: false } },
                    y: { beginAtZero: true, ticks: { color: '#94a3b8', precision: 0 }, grid: { color: 'rgba(148,163,184,0.1)' } }
                }
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Production KPIs: reject-rate (OK/NG), throughput, uptime, and an SPC p-chart.
// Reads /api/production_stats (PLC OK/NG counters diffed per bucket).
// ---------------------------------------------------------------------------
async function refreshProductionCharts() {
    const window = document.getElementById('insight-window')?.value || '24h';
    const shipment = document.getElementById('insight-shipment')?.value || '';
    let data;
    try {
        const r = await fetch('/api/production_stats?window=' + encodeURIComponent(window) +
                              '&shipment=' + encodeURIComponent(shipment));
        data = await r.json();
    } catch (e) { console.error('production_stats fetch failed:', e); return; }

    const tl = data.timeline || [];
    const hasData = tl.length > 0 && ((data.total_units || 0) > 0 || (data.speed_max || 0) > 0 || (data.downtime_total_s || 0) > 0);
    const emptyEl = document.getElementById('production-empty');
    const chartsEl = document.getElementById('production-charts');
    const kpisEl = document.getElementById('production-kpis');
    const totalEl = document.getElementById('production-total');
    if (emptyEl) emptyEl.style.display = hasData ? 'none' : 'block';
    if (chartsEl) chartsEl.style.display = hasData ? 'grid' : 'none';
    if (kpisEl) kpisEl.style.display = hasData ? 'grid' : 'none';
    if (totalEl) totalEl.textContent = hasData
        ? `${data.total_units} units • ${data.total_ng} NG • reject ${data.reject_rate_overall}%`
        : '';
    if (!hasData) return;

    // ---- KPI cards: OEE, Availability, Performance, Quality, Eject/Total, Downtime, Speed ----
    const _set = (id, txt) => { const e = document.getElementById(id); if (e) e.textContent = txt; };
    const _fmtDur = (s) => {
        s = Math.round(s || 0);
        if (s < 60) return s + 's';
        if (s < 3600) return Math.floor(s / 60) + 'm ' + (s % 60) + 's';
        return Math.floor(s / 3600) + 'h ' + Math.floor((s % 3600) / 60) + 'm';
    };
    const oeeEl = document.getElementById('kpi-oee');
    if (oeeEl) oeeEl.style.color = (data.oee >= 85) ? '#10b981' : (data.oee >= 60) ? '#f59e0b' : '#ef4444';
    _set('kpi-oee', (data.oee ?? 0) + '%');
    _set('kpi-availability', (data.availability ?? 0) + '%');
    _set('kpi-performance', (data.performance ?? 0) + '%');
    _set('kpi-quality', (data.quality ?? 0) + '%');
    _set('kpi-eject', (data.eject_over_total ?? 0) + '%');
    _set('kpi-downtime', _fmtDur(data.downtime_total_s));
    _set('kpi-speed', `${data.speed_avg ?? 0} / ${data.speed_max ?? 0}`);

    const labels = tl.map(p => p.t);

    // ---- Reject rate over time: OK/NG stacked bars + reject% line ----
    const rejCtx = document.getElementById('production-reject-chart');
    if (rejCtx) {
        if (_prodReject) _prodReject.destroy();
        _prodReject = new Chart(rejCtx, {
            data: {
                labels,
                datasets: [
                    { type: 'bar', label: 'OK', data: tl.map(p => p.ok), backgroundColor: 'rgba(16,185,129,0.6)', stack: 's', yAxisID: 'y' },
                    { type: 'bar', label: 'NG', data: tl.map(p => p.ng), backgroundColor: 'rgba(239,68,68,0.7)', stack: 's', yAxisID: 'y' },
                    { type: 'line', label: 'Reject %', data: tl.map(p => p.reject_rate), borderColor: '#f59e0b', backgroundColor: 'transparent', yAxisID: 'y1', tension: 0.3, pointRadius: 0, borderWidth: 2 }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, labels: { color: '#cbd5e1', font: { size: 10 }, boxWidth: 12 } },
                    title: { display: true, text: 'Yield: OK / NG + reject rate', color: '#cbd5e1', font: { size: 13 } }
                },
                scales: {
                    x: { stacked: true, ticks: { color: '#94a3b8', font: { size: 9 }, maxTicksLimit: 8 }, grid: { display: false } },
                    y: { stacked: true, beginAtZero: true, ticks: { color: '#94a3b8', precision: 0 }, grid: { color: 'rgba(148,163,184,0.1)' }, title: { display: true, text: 'units', color: '#94a3b8' } },
                    y1: { position: 'right', beginAtZero: true, max: 100, ticks: { color: '#f59e0b' }, grid: { display: false }, title: { display: true, text: '%', color: '#f59e0b' } }
                }
            }
        });
    }

    // ---- Throughput: units processed per bucket ----
    const thrCtx = document.getElementById('production-throughput-chart');
    if (thrCtx) {
        if (_prodThroughput) _prodThroughput.destroy();
        _prodThroughput = new Chart(thrCtx, {
            type: 'bar',
            data: { labels, datasets: [{ label: 'Units', data: tl.map(p => p.units), backgroundColor: 'rgba(59,130,246,0.6)', borderRadius: 3 }] },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, title: { display: true, text: 'Throughput (units per bucket)', color: '#cbd5e1', font: { size: 13 } } },
                scales: {
                    x: { ticks: { color: '#94a3b8', font: { size: 9 }, maxTicksLimit: 8 }, grid: { display: false } },
                    y: { beginAtZero: true, ticks: { color: '#94a3b8', precision: 0 }, grid: { color: 'rgba(148,163,184,0.1)' } }
                }
            }
        });
    }

    // ---- Uptime % over time ----
    const upCtx = document.getElementById('production-uptime-chart');
    if (upCtx) {
        if (_prodUptime) _prodUptime.destroy();
        _prodUptime = new Chart(upCtx, {
            type: 'line',
            data: { labels, datasets: [{ label: 'Uptime %', data: tl.map(p => p.uptime_pct), borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.15)', fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 }] },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, title: { display: true, text: 'Line uptime (% moving)', color: '#cbd5e1', font: { size: 13 } } },
                scales: {
                    x: { ticks: { color: '#94a3b8', font: { size: 9 }, maxTicksLimit: 8 }, grid: { display: false } },
                    y: { beginAtZero: true, max: 100, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148,163,184,0.1)' } }
                }
            }
        });
    }

    // ---- SPC p-chart: reject rate with center line + per-point 3σ control limits ----
    const spcCtx = document.getElementById('production-spc-chart');
    if (spcCtx) {
        if (_prodSpc) _prodSpc.destroy();
        const pbar = (data.reject_rate_overall || 0) / 100;  // fraction
        const cl = data.reject_rate_overall || 0;            // %
        const ucl = tl.map(p => { const n = p.units || 0; if (!n) return null; const s = Math.sqrt(Math.max(pbar * (1 - pbar), 0) / n); return Math.min(100, (pbar + 3 * s) * 100); });
        const lcl = tl.map(p => { const n = p.units || 0; if (!n) return null; const s = Math.sqrt(Math.max(pbar * (1 - pbar), 0) / n); return Math.max(0, (pbar - 3 * s) * 100); });
        const rr = tl.map(p => p.reject_rate);
        const ptColors = rr.map((v, i) => (ucl[i] != null && (v > ucl[i] || v < lcl[i])) ? '#ef4444' : '#3b82f6');
        _prodSpc = new Chart(spcCtx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label: 'Reject %', data: rr, borderColor: '#3b82f6', backgroundColor: 'transparent', pointBackgroundColor: ptColors, pointRadius: 3, tension: 0, borderWidth: 2 },
                    { label: 'UCL', data: ucl, borderColor: 'rgba(239,68,68,0.7)', borderDash: [5, 4], pointRadius: 0, borderWidth: 1, fill: false, spanGaps: true },
                    { label: 'LCL', data: lcl, borderColor: 'rgba(239,68,68,0.7)', borderDash: [5, 4], pointRadius: 0, borderWidth: 1, fill: false, spanGaps: true },
                    { label: 'CL', data: tl.map(() => cl), borderColor: 'rgba(16,185,129,0.8)', borderDash: [2, 2], pointRadius: 0, borderWidth: 1, fill: false }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, labels: { color: '#cbd5e1', font: { size: 10 }, boxWidth: 12 } },
                    title: { display: true, text: 'SPC p-chart — reject rate (3σ control limits)', color: '#cbd5e1', font: { size: 13 } }
                },
                scales: {
                    x: { ticks: { color: '#94a3b8', font: { size: 9 }, maxTicksLimit: 8 }, grid: { display: false } },
                    y: { beginAtZero: true, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148,163,184,0.1)' }, title: { display: true, text: '%', color: '#94a3b8' } }
                }
            }
        });
    }

    // ---- OEE over time (Availability × Performance × Quality) ----
    const oeeCtx = document.getElementById('production-oee-chart');
    if (oeeCtx) {
        if (_prodOee) _prodOee.destroy();
        _prodOee = new Chart(oeeCtx, {
            type: 'line',
            data: { labels, datasets: [{ label: 'OEE %', data: tl.map(p => p.oee), borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.15)', fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 }] },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, title: { display: true, text: 'OEE over time (Availability × Performance × Quality)', color: '#cbd5e1', font: { size: 13 } } },
                scales: {
                    x: { ticks: { color: '#94a3b8', font: { size: 9 }, maxTicksLimit: 8 }, grid: { display: false } },
                    y: { beginAtZero: true, max: 100, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148,163,184,0.1)' }, title: { display: true, text: '%', color: '#94a3b8' } }
                }
            }
        });
    }

    // ---- Speed over time vs max (encoder units/sec) ----
    const spdCtx = document.getElementById('production-speed-chart');
    if (spdCtx) {
        if (_prodSpeed) _prodSpeed.destroy();
        const smax = data.speed_max || 0;
        _prodSpeed = new Chart(spdCtx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label: 'Speed', data: tl.map(p => p.speed), borderColor: '#8b5cf6', backgroundColor: 'rgba(139,92,246,0.15)', fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 },
                    { label: 'Max speed', data: tl.map(() => smax), borderColor: 'rgba(139,92,246,0.6)', borderDash: [5, 4], pointRadius: 0, borderWidth: 1, fill: false }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, labels: { color: '#cbd5e1', font: { size: 10 }, boxWidth: 12 } },
                    title: { display: true, text: 'Line speed vs max (encoder units/sec)', color: '#cbd5e1', font: { size: 13 } }
                },
                scales: {
                    x: { ticks: { color: '#94a3b8', font: { size: 9 }, maxTicksLimit: 8 }, grid: { display: false } },
                    y: { beginAtZero: true, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148,163,184,0.1)' } }
                }
            }
        });
    }
}


// ---------------------------------------------------------------------------
// Operator drill-down (3.21.0):
//   • hover a scatter dot  -> floating thumbnail of THAT exact annotated frame
//   • click a class/bar/pie -> side drawer with the last 24 thumbnails of that class
// Image URLs (4.0.0):
//   raw       — /api/raw_image/<path>            (file on disk; one per frame)
//   annotated — /api/render_detected/<raw_path>  (rendered on demand from
//               inference_results.detections; Redis-cached; no disk file)
// The detection pipeline no longer writes _DETECTED.jpg as of 4.0.0; legacy
// _DETECTED.jpg URLs still resolve through the render endpoint's fallback.
// ---------------------------------------------------------------------------
// Both URLs for any chart data point or drawer item.
//   `ann`  → render-on-demand annotated view (same raw under the hood)
//   `raw`  → bare raw frame
function _imgUrlsFor(pt) {
    if (!pt || !pt.img) return { ann: null, raw: null };
    // Strip both the `raw_images/` prefix and the legacy `_DETECTED.jpg` suffix
    // — the render endpoint expects a RAW path; the pipeline-stored
    // image_path may still carry `_DETECTED.jpg` from older inference_results
    // rows written before 4.0.0.
    const relRaw = String(pt.img)
        .replace(/^raw_images\//, '')
        .replace(/_DETECTED\.jpg$/, '.jpg');
    const ann = '/api/render_detected/' + encodeURI(relRaw);
    // The raw URL is the same path through /api/raw_image/.
    const raw = '/api/raw_image/' + encodeURI(relRaw);
    return { ann, raw };
}
// Attach Panzoom + wheel-to-zoom on an <img> already in the DOM. Reuses the
// Panzoom library inlined by audio.js for the timeline slideshow.
function _attachZoom(imgEl) {
    if (!imgEl || imgEl.dataset.zoomReady === '1' || typeof Panzoom !== 'function') return;
    try {
        // No `contain:` option — `contain:'inside'` silently caps zoom at 1x
        // because at scale>1 the image becomes larger than parent, which
        // violates the "stay inside" rule. CSS (max-width:100%, object-fit:
        // contain) handles the fit-on-load; flex:1 1 0 on the wrap keeps
        // the panel 50/50. Panzoom handles wheel-zoom unconstrained up to
        // maxScale.
        const pz = Panzoom(imgEl, { maxScale: 10, minScale: 1, cursor: 'zoom-in', startScale: 1, startX: 0, startY: 0 });
        pz.reset({ animate: false });
        imgEl.parentElement.addEventListener('wheel', function (e) {
            e.preventDefault();
            pz.zoomWithWheel(e);
        }, { passive: false });
        imgEl.parentElement.addEventListener('dblclick', function () { pz.reset(); });
        imgEl.dataset.zoomReady = '1';
    } catch (e) { /* Panzoom not yet loaded — no-op */ }
}
// Chart.js v4 "external" tooltip handler — shared by both camera scatters.
// 4.0.7 — direct mousemove preview handler. Replaces the Chart.js external
// tooltip path, which wasn't firing reliably across builds. Called by a
// `mousemove` listener attached to the scatter canvas in the chart-build code.

// 4.0.15 — operator-chosen hover preview size (Charts tab top toolbar).
// Persisted in localStorage so the choice survives page reloads. Anything
// that touches `.style.display = 'block'` on the preview re-applies the
// dimensions so the size sticks across re-renders.
const _HOVER_SIZES = {
    small:  { w: 420,  h: 280 },
    medium: { w: 640,  h: 420 },
    large:  { w: 900,  h: 580 },
    xlarge: { w: 1200, h: 780 },
};
function _getHoverPreviewSize() {
    const k = localStorage.getItem('mve_hover_size') || 'medium';
    return _HOVER_SIZES[k] || _HOVER_SIZES.medium;
}
function _applyHoverPreviewSize() {
    const el = document.getElementById('chart-image-preview');
    if (!el) return;
    const img = el.querySelector('img');
    const sz = _getHoverPreviewSize();
    if (img) {
        img.style.width  = sz.w + 'px';
        img.style.height = sz.h + 'px';
    }
}
function setHoverPreviewSize(key) {
    if (!_HOVER_SIZES[key]) key = 'medium';
    localStorage.setItem('mve_hover_size', key);
    _applyHoverPreviewSize();
}
window.setHoverPreviewSize = setHoverPreviewSize;
// Sync the <select> to the stored choice + apply dims, on first paint.
document.addEventListener('DOMContentLoaded', () => {
    const sel = document.getElementById('hover-preview-size');
    const k = localStorage.getItem('mve_hover_size') || 'medium';
    if (sel) sel.value = k;
    _applyHoverPreviewSize();
});

// 4.0.19 — Charts tab reading order, per user preference:
//   1. Detection Insights header + quality score card + trend row
//   2. Score per shipment   (moved up from below; currently #quality-charts-panel)
//   3. Camera × encoder scatter
//   4. Bar + Pie (detection by class, distribution)
//   5. Secondary diagnostics + timeline + size + confidence + confidence-class
//   6. Ejection Insights + Production KPIs (unchanged at bottom)
// Inner reordering (3-4-5) is done via CSS `order:` on the existing
// #insight-charts grid items in status.html. This function only handles the
// outer rearrangement: moving #quality-charts-panel BACK into
// #detection-insight-panel right after #sqs-trend-row, so it ends up between
// the quality-score card and the inner charts.
function _reorderChartsTab() {
    try {
        // v4.0.237 — Score-per-shipment is the operator's TOP chart: it's independent
        // of every other chart and loads on tab open. Move #quality-charts-panel to
        // the VERY TOP of the Charts tab, ABOVE #detection-insight-panel (which holds
        // the shipment/window/config controls, the quality-score card, and the main
        // scatter). Was tucked BELOW the quality-score card, inside Detection Insights.
        const tab    = document.getElementById('tab-grafana');
        const dins   = document.getElementById('detection-insight-panel');
        const sScore = document.getElementById('quality-charts-panel');
        if (tab && sScore && dins && dins.parentNode === tab) {
            tab.insertBefore(sScore, dins);          // Score-per-shipment first
        } else if (tab && sScore) {
            tab.insertBefore(sScore, tab.firstChild);
        }
        // v4.0.241 — REMOVE the "Shipment Quality Score + Trend over window" section
        // (operator: "remove this fucking section"). Relocate its still-wanted buttons
        // into the config bar first (Download PDF next to CSV, then Calibrate) so
        // nothing is lost, then hide the whole card. The per-shipment score card above
        // + the ↑/↓ quality-strip deltas now carry the score/trend story.
        const csvBtn = document.querySelector('#detection-insight-panel button[onclick="exportInsightCSV()"]');
        const cfgBar = csvBtn && csvBtn.parentNode;
        const pdfBtn = document.getElementById('sqs-download-pdf');
        const calBtn = document.getElementById('sqs-calibrate');
        if (cfgBar && pdfBtn) { pdfBtn.style.cssText = 'padding:4px 10px; font-size:12px; cursor:pointer; background:rgba(16,185,129,0.25); color:#6ee7b7; border:1px solid rgba(16,185,129,0.5); border-radius:4px;'; cfgBar.insertBefore(pdfBtn, csvBtn.nextSibling); }
        // v4.0.243 — Calibrate lives in the Advanced tab beside the score_scale_factor
        // Set button (operator: it's the same gain knob), NOT in the Charts toolbar.
        const setBtn = document.querySelector('button[onclick="setScoreScaleFactor()"]');
        if (setBtn && calBtn) { calBtn.style.cssText = 'padding:6px 12px; font-size:12px; cursor:pointer; background:linear-gradient(135deg,#f59e0b,#d97706); color:#fff; border:none; border-radius:6px; font-weight:600; margin-left:8px; vertical-align:middle;'; setBtn.parentNode.insertBefore(calBtn, setBtn.nextSibling); }
        const qCard = document.getElementById('shipment-quality-card');
        if (qCard) qCard.style.display = 'none';
    } catch (e) { /* never block page load on reorder errors */ }
}
document.addEventListener('DOMContentLoaded', _reorderChartsTab);
// 4.0.271 — on load, sync the X-axis dropdown to the persisted operator pick so the
// UI shows what's actually selected (not the HTML default or a guard fallback).
document.addEventListener('DOMContentLoaded', function () {
    try {
        const sel = document.getElementById('insight-axis-select');
        if (sel && sel.value !== _insightAxis) sel.value = _insightAxis;
        if (typeof _setAxisToggleUI === 'function') _setAxisToggleUI(_insightAxis);
    } catch (e) {}
});

function _showHoverPreview(pt, datasetLabel, mouseX, mouseY) {
    const el = document.getElementById('chart-image-preview');
    if (!el) return;
    _applyHoverPreviewSize();
    const urls = _imgUrlsFor(pt);
    const img = el.querySelector('img');
    const cap = el.querySelector('.cap');
    const target = urls.ann || urls.raw;
    // 4.0.8 — make the preview VISIBLE first, then start the image load.
    el.style.display = 'block';
    const cls = datasetLabel || pt.cls || '';
    const conf = ((pt.conf || 0) * 100).toFixed(0);
    // v4.0.110 — add TIMESTAMP + ENCODER to the hover caption so the operator
    // can immediately see WHEN (or at which meter/roll-position) the defect
    // happened, without having to open the image. Time comes from either
    // `pt.x` (already ms since epoch on the time axis) or from the filename
    // stem on the encoder axis (encoder axis uses `pt.x` for encoder value,
    // but we can still reconstruct timestamp by parsing the image filename
    // which is written as `YYYY-MM-DD-HH-MM-SS-uuuuuu_pN_M.jpg`).
    // Encoder comes from `pt.cam_id != null ? pt.enc : pt.x` when the axis
    // is encoder; on the time axis it isn't stored client-side so we omit.
    let tsTxt = '', encTxt = '';
    try {
        const axisMode = (_insightAxis === 'encoder') ? 'encoder' : 'time';
        // Encoder value: on the encoder axis, x IS the encoder value.
        if (axisMode === 'encoder' && typeof pt.x === 'number') {
            encTxt = ' • enc ' + Math.round(pt.x).toLocaleString();
        }
        // Timestamp: on the time axis, x IS ms-since-epoch. On the encoder
        // axis we parse it out of the image filename stem.
        let tMs = null;
        if (axisMode === 'time' && typeof pt.x === 'number') {
            tMs = pt.x;
        } else if (pt.img) {
            const fn = String(pt.img).split('/').pop() || '';
            const m = fn.match(/^(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})/);
            if (m) tMs = Date.UTC(+m[1], +m[2]-1, +m[3], +m[4], +m[5], +m[6]);
        }
        if (tMs != null) {
            const d = new Date(tMs);
            // Local-timezone display: YYYY-MM-DD HH:MM:SS
            const pad = (n) => String(n).padStart(2, '0');
            tsTxt = ' • ' + d.getFullYear() + '-' + pad(d.getMonth()+1) + '-' + pad(d.getDate())
                    + ' ' + pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds());
        }
    } catch (e) { /* never let caption formatting crash the preview */ }
    const metaTxt = `${cls} • cam ${pt.y != null ? pt.y : '?'} • conf ${conf}%${encTxt}${tsTxt}`;
    if (img && target) {
        if (img.dataset.src !== target) {
            // 4.0.10 — surface a loading state during the network round-trip so
            // operators on slow remote links don't see a black box.
            img.dataset.src = target;
            // Hide the broken-image icon while we wait by removing src first,
            // then setting the new one on a microtask. Browsers paint a fresh
            // empty img between these steps (no broken-link glyph).
            img.removeAttribute('src');
            img.style.opacity = '0.3';
            if (cap) cap.textContent = '⏳ loading…  ' + metaTxt;
            const settle = () => {
                img.style.opacity = '1';
                img.style.display = '';   // 4.0.198 — un-hide if a prior dot was missing
                if (cap) cap.textContent = `${metaTxt}   (click to edit in ai-trainer)`;
            };
            img.onload  = settle;
            // v4.0.106 — track fallback state with a dataset flag rather than
            // comparing `img.src !== urls.raw`. `img.src` returns the FULLY
            // RESOLVED absolute URL ("http://host:port/api/raw_image/..."),
            // while `urls.raw` is the relative path we set. They never match
            // as strings, so the fallback path fired forever when both the
            // annotated AND raw endpoints returned 404, spraying dozens of
            // requests per second for the SAME missing file until the tab
            // was closed. Now: one fallback attempt max, then give up.
            img.dataset.fellBack = '';
            img.onerror = function() {
                if (urls.raw && img.dataset.fellBack !== '1') {
                    img.dataset.fellBack = '1';
                    img.src = urls.raw;
                } else {
                    // 4.0.198 — operator: "if the detection image is not
                    // available, simply doesn't show the image." Old images
                    // get purged from disk by retention while detection rows
                    // persist in the DB, so 404s are expected for old
                    // shipments. Hide the <img> entirely (no broken/faded box)
                    // and leave a one-line caption instead of a graphic.
                    img.removeAttribute('src');
                    img.style.display = 'none';
                    if (cap) cap.textContent = '🗒️ image no longer stored (data kept)  ' + metaTxt;
                    img.onerror = null;
                }
            };
            // Start the load on the next tick so the empty paint above lands first.
            setTimeout(() => { img.src = target; }, 0);
        } else {
            // Already this target — keep showing it; update caption only.
            if (cap) cap.textContent = `${metaTxt}   (click to edit in ai-trainer)`;
        }
    } else if (img && !target) {
        img.removeAttribute('src'); img.dataset.src = ''; img.style.opacity = '1';
        if (cap) cap.textContent = `${cls} • cam ${pt.y != null ? pt.y : '?'}   (no image_path on this point)`;
    }
    // 4.0.8 — viewport-clamped positioning.
    const pw = el.offsetWidth  || 432;
    const ph = el.offsetHeight || 320;
    const vw = window.innerWidth  || 1920;
    const vh = window.innerHeight || 1080;
    let left = mouseX + 14;
    if (left + pw > vw - 8) left = Math.max(8, mouseX - pw - 14);
    let top = mouseY - Math.round(ph / 2);
    if (top + ph > vh - 8) top = Math.max(8, vh - ph - 8);
    if (top < 8) top = 8;
    el.style.left = left + 'px';
    el.style.top  = top + 'px';
}

// 4.0.8 — bullet-proof nearest-point lookup. Iterates every drawn element and
// computes a CSS-pixel-space distance to the cursor. Reliable even when
// Chart.js's `getElementsAtEventForMode` returns empty (which it does for some
// dense scatters and for cursors hovering between bubble centres).
function _nearestPointPx(chart, mouseX, mouseY) {
    if (!chart || !chart.canvas) return null;
    const rect = chart.canvas.getBoundingClientRect();
    const lx = mouseX - rect.left;
    const ly = mouseY - rect.top;
    let best = null;
    let bestD = Infinity;
    const datasets = chart.data.datasets || [];
    for (let dsIdx = 0; dsIdx < datasets.length; dsIdx++) {
        const meta = chart.getDatasetMeta && chart.getDatasetMeta(dsIdx);
        if (!meta || meta.hidden) continue;
        const els = meta.data || [];
        for (let i = 0; i < els.length; i++) {
            const el = els[i];
            if (!el || el.x == null || el.y == null) continue;
            const dx = el.x - lx;
            const dy = el.y - ly;
            const d = dx * dx + dy * dy;
            if (d < bestD) { bestD = d; best = { datasetIndex: dsIdx, index: i, dist: Math.sqrt(d) }; }
        }
    }
    return best;
}

function _attachHoverPreview(canvasEl, getChart) {
    if (!canvasEl || canvasEl._hoverPreviewWired) return;
    canvasEl._hoverPreviewWired = true;
    // 4.0.10 — tiny debounce so the preview only re-renders ~10× per second.
    // On slow links this stops queued requests from piling up as the cursor
    // sweeps across dense scatters.
    let _hoverPending = null;
    canvasEl.addEventListener('mousemove', (event) => {
        const ev = event;
        if (_hoverPending) return;
        _hoverPending = setTimeout(() => {
            _hoverPending = null;
            _runHoverPreview(ev, getChart);
        }, 80);
    });
    canvasEl.addEventListener('mouseleave', () => {
        // 4.0.45 — cancel any queued mousemove timer too. Without this, a
        // pending _runHoverPreview fires ~80ms AFTER the cursor leaves the
        // canvas with stale (in-bounds) event coords and the preview
        // re-appears even though the cursor is now on a different chart.
        if (_hoverPending) { clearTimeout(_hoverPending); _hoverPending = null; }
        const el = document.getElementById('chart-image-preview');
        if (el) el.style.display = 'none';
    });
    canvasEl._runHoverPreview = (ev) => _runHoverPreview(ev, getChart);
}

function _runHoverPreview(event, getChart) {
        // 4.0.40 — when the operator is in baseline-pick mode the entire
        // hover preview must stay hidden so they can SEE where to click.
        // The Chart.js external tooltip (_scatterImageTooltip) is already
        // guarded the same way; this path is the manual mousemove handler
        // and was the one still painting the preview during pick mode.
        if (window._pickingBaseline) {
            const el = document.getElementById('chart-image-preview');
            if (el) el.style.display = 'none';
            return;
        }
        const chart = getChart();
        if (!chart) return;
        // 4.0.44 — hide the preview if the cursor is outside the colour-cell
        // plot area (chartArea). The canvas extends above and below the
        // axis ticks; mouseleave only fires when leaving the FULL canvas,
        // so without this check the preview stayed visible while the cursor
        // was on the legend / axis / padding.
        // 4.0.45 — also confirm the cursor is currently OVER the scatter
        // canvas at the time this runs (debounced mousemove can fire 80ms
        // after the cursor has already moved to a different chart). Uses
        // document.elementFromPoint so we don't rely on the stale event.
        try {
            const ca = chart.chartArea;
            const rect = chart.canvas.getBoundingClientRect();
            const lx = event.clientX - rect.left;
            const ly = event.clientY - rect.top;
            const cursorEl = document.elementFromPoint(event.clientX, event.clientY);
            const stillOnCanvas = cursorEl === chart.canvas;
            const outOfChartArea = ca && (lx < ca.left || lx > ca.right || ly < ca.top || ly > ca.bottom);
            if (!stillOnCanvas || outOfChartArea) {
                const el = document.getElementById('chart-image-preview');
                if (el) el.style.display = 'none';
                return;
            }
        } catch (_e) { /* defensive — fall through to existing logic */ }
        // 1. Try Chart.js's own picker first (cheap, usually correct).
        let pick = null;
        try {
            const points = chart.getElementsAtEventForMode(event, 'nearest', { intersect: false }, false);
            if (points && points.length) pick = points[0];
        } catch (e) { /* fall through */ }
        // 2. Fallback to manual nearest in pixel space. Tighter cap (60 px)
        //    so the preview doesn't appear when cursor is clearly between dots.
        if (!pick) {
            const near = _nearestPointPx(chart, event.clientX, event.clientY);
            if (near && near.dist <= 60) pick = near;
        }
        if (!pick) {
            const el = document.getElementById('chart-image-preview');
            if (el) el.style.display = 'none';
            return;
        }
        const ds = chart.data.datasets[pick.datasetIndex] || {};
        const dp = (ds.data && ds.data[pick.index]) || {};
        // 4.0.9 — anchor the preview to the DOT'S screen position, not the
        // cursor's. Stops the preview from "floating" when cursor drifts off
        // the scatter while the manual-nearest fallback keeps a stale dot pick.
        let anchorX = event.clientX;
        let anchorY = event.clientY;
        try {
            const meta = chart.getDatasetMeta(pick.datasetIndex);
            const dotEl = meta && meta.data && meta.data[pick.index];
            const rect = chart.canvas.getBoundingClientRect();
            if (dotEl && dotEl.x != null && dotEl.y != null && rect) {
                anchorX = rect.left + dotEl.x;
                anchorY = rect.top  + dotEl.y;
            }
        } catch (e) { /* fall back to cursor coords */ }
        _showHoverPreview(dp, ds.label, anchorX, anchorY);
}

function _scatterImageTooltip(ctx) {
    // 4.0.6 — bullet-proof hover preview:
    //   * Don't gate on tt.opacity — some Chart.js builds keep it at 0 when
    //     using an external handler, which was hiding our preview even when
    //     dataPoints were present.
    //   * If dataPoints are missing, leave preview alone (don't re-hide it
    //     during mouseout transitions — that caused flicker on dense scatters).
    //   * If pt.img is missing, surface that in the caption instead of failing
    //     silently so the operator knows WHY no preview.
    // 4.0.38 — also suppress the preview entirely while the operator is in
    // baseline-pick mode. The 420x280 preview floats next to the cursor and
    // visually obscures the spot the operator is trying to aim at — even with
    // pointer-events:none making clicks pass through, the operator can't SEE
    // where to click. Hiding the preview in pick mode unblocks the flow.
    const el = document.getElementById('chart-image-preview');
    if (!el) return;
    if (window._pickingBaseline) {
        el.style.display = 'none'; return;
    }
    const tt = ctx && ctx.tooltip;
    if (!tt || !tt.dataPoints || !tt.dataPoints.length) {
        el.style.display = 'none'; return;
    }
    // 4.0.46 — Chart.js signals "hide" by setting opacity=0 but leaves the
    // last dataPoints populated. The dataPoints check above doesn't catch
    // that path; without this opacity gate the preview stayed visible after
    // the cursor left the chart even though Chart.js had already retired
    // the tooltip. Earlier comment said "don't gate on opacity"; that was
    // for the show path. Hiding on opacity=0 is correct.
    if (tt.opacity === 0) {
        el.style.display = 'none'; return;
    }
    const dp = tt.dataPoints[0];
    const pt = (dp && dp.raw) || {};
    const cls = (dp.dataset && dp.dataset.label) || pt.cls || '';
    const conf = ((pt.conf || 0) * 100).toFixed(0);
    const img = el.querySelector('img');
    const cap = el.querySelector('.cap');
    const urls = _imgUrlsFor(pt);
    const target = urls.ann || urls.raw;
    if (img) {
        if (target && img.dataset.src !== target) {
            img.dataset.src = target;
            img.style.display = '';
            img.src = target;
            img.dataset.fellBack = '';
            img.onload = function() { img.style.display = ''; };
            img.onerror = function() {
                if (urls.raw && img.dataset.fellBack !== '1') {
                    img.dataset.fellBack = '1';
                    img.src = urls.raw;
                } else {
                    // 4.0.198 — purged old image: hide instead of broken box.
                    img.removeAttribute('src');
                    img.style.display = 'none';
                    img.onerror = null;
                }
            };
        } else if (!target) {
            img.removeAttribute('src');
            img.dataset.src = '';
        }
    }
    if (cap) {
        const meta = cls + ' • cam ' + (pt.y != null ? pt.y : '?') + ' • conf ' + conf + '%';
        cap.textContent = target
            ? meta + '   (click to edit in ai-trainer)'
            : meta + '   (no image_path on this point)';
    }
    // Position next to the cursor. Use clientX/Y when caret values aren't set
    // (Chart.js drops them on some hover modes).
    // 4.0.39 — viewport clamp (mirrors the logic in _showHoverPreview at
    // line ~2050). Without it, when the cursor sits near the right or bottom
    // edge of the chart the preview lands off-canvas in the corner of the
    // viewport. With it the preview flips to the left of the cursor and
    // pins to the visible viewport box, never drifting into the corner.
    const box = ctx.chart.canvas.getBoundingClientRect();
    const cx = (tt.caretX != null) ? tt.caretX : 0;
    const cy = (tt.caretY != null) ? tt.caretY : 0;
    const mouseX = box.left + cx;
    const mouseY = box.top  + cy;
    const pw = el.offsetWidth  || 432;
    const ph = el.offsetHeight || 320;
    const vw = window.innerWidth  || 1920;
    const vh = window.innerHeight || 1080;
    let left = mouseX + 14;
    if (left + pw > vw - 8) left = Math.max(8, mouseX - pw - 14);
    let top = mouseY - Math.round(ph / 2);
    if (top + ph > vh - 8) top = Math.max(8, vh - ph - 8);
    if (top < 8) top = 8;
    el.style.left = left + 'px';
    el.style.top  = top + 'px';
    el.style.display = 'block';
}
// Click handler factory for bar/pie/pareto: drills into the drawer.

// 3.21.2 — centered modal showing the EXACT clicked frame at BIG size.
// Default view = Annotated. Toggle Raw / Both via the header buttons.
// Download buttons appear on each image. ESC or click-backdrop closes.
let _currentDefectItem = null;
let _currentDefectView = 'ann'; // 'ann' | 'raw' | 'both'

function _renderDefectView(urls) {
    const body = document.getElementById('defect-drawer-body');
    if (!body) return;
    body.innerHTML = '';
    // Force flexbox row instead of grid: flex:1 1 0 + min-width:0 on each wrap
    // guarantees equal 50/50 columns regardless of any intrinsic content size
    // (the previous grid minmax(0,1fr) was still letting one column grow when
    // the other only had a small placeholder).
    body.style.display = 'flex';
    body.style.flexDirection = 'row';
    body.style.gap = '2px';
    body.style.gridTemplateColumns = '';
    const kinds = (_currentDefectView === 'both') ? ['raw', 'ann']
                : (_currentDefectView === 'raw') ? ['raw'] : ['ann'];
    const fnameBase = String((_currentDefectItem && _currentDefectItem.image_path) || 'image').split('/').pop().replace(/\.jpg$/i, '');
    kinds.forEach(kind => {
        const url = (kind === 'raw') ? urls.raw : urls.ann;
        const wrap = document.createElement('div');
        wrap.style.cssText = 'flex:1 1 0; overflow:hidden; background:#000; position:relative; display:flex; align-items:center; justify-content:center; min-width:0; min-height:0;';
        if (!url) {
            wrap.innerHTML = '<div style="color: var(--text-secondary); font-size:13px; padding:24px;">(no ' + kind + ' image available)</div>';
        } else {
            const img = document.createElement('img');
            img.src = url; img.loading = 'eager';
            img.style.cssText = 'max-width:100%; max-height:100%; object-fit:contain; cursor:zoom-in; display:block; user-select:none;';
            wrap.appendChild(img);
            img.addEventListener('load', () => _attachZoom(img), { once: true });
            // Graceful fallback: image file was pruned by DVR retention (chart row
            // outlives the raw_images chunk on disk). Show a clear message instead
            // of the broken-image icon. (3.21.7)
            img.addEventListener('error', () => {
                img.remove();
                const ph = document.createElement('div');
                ph.style.cssText = 'color:#94a3b8; font-size:13px; text-align:center; padding:24px; max-width:80%; line-height:1.5;';
                ph.innerHTML = '<div style="font-size:32px; opacity:0.4; margin-bottom:8px;">🗑️</div>' +
                               '<div><b>Image not available</b></div>' +
                               '<div style="font-size:11px; color: var(--text-secondary); margin-top:4px;">The ' + kind + ' jpg was pruned by the disk-retention policy (chart records outlive raw-image chunks).</div>' +
                               '<div style="font-size:10px; color: var(--text-secondary); margin-top:6px; word-break:break-all;">' + url + '</div>';
                wrap.appendChild(ph);
            }, { once: true });
            const lbl = document.createElement('div');
            lbl.textContent = kind === 'raw' ? 'RAW' : 'ANNOTATED';
            lbl.style.cssText = 'position:absolute; top:8px; left:8px; padding:3px 10px; background:rgba(0,0,0,0.72); color:#fff; font-size:11px; border-radius:3px; pointer-events:none; letter-spacing:0.4px; font-weight:600;';
            wrap.appendChild(lbl);
            const dl = document.createElement('a');
            dl.href = url; dl.download = fnameBase + (kind === 'raw' ? '.jpg' : '_DETECTED.jpg');
            dl.innerHTML = '⬇ Download';
            dl.style.cssText = 'position:absolute; top:8px; right:8px; padding:4px 12px; background:rgba(45,125,70,0.88); color:#fff; font-size:11px; border-radius:3px; text-decoration:none; font-weight:500;';
            wrap.appendChild(dl);
        }
        body.appendChild(wrap);
    });
}

function _highlightViewToggle() {
    const toggle = document.getElementById('defect-view-toggle');
    if (!toggle) return;
    toggle.querySelectorAll('button').forEach(btn => {
        const active = btn.dataset.view === _currentDefectView;
        btn.style.background = active ? 'rgba(59,130,246,0.6)' : 'transparent';
        btn.style.color = active ? '#fff' : 'var(--text-secondary)';
    });
}

function openDefectDrawerForFrame(item) {
    // item: {image_path, shipment, t, classes:[...], best_confidence, cls}
    _currentDefectItem = item || null;
    // 3.26.0 — mirror to window so cross-script consumers (annotate.js) can read
    // the current frame without depending on the `let` binding's script-scope.
    window._currentDefectItem = _currentDefectItem;
    const drawer = document.getElementById('defect-drawer');
    const body   = document.getElementById('defect-drawer-body');
    const empty  = document.getElementById('defect-drawer-empty');
    const title  = document.getElementById('defect-drawer-title');
    const cnt    = document.getElementById('defect-drawer-count');
    const meta   = document.getElementById('defect-drawer-meta');
    const whyP   = document.getElementById('defect-drawer-why'); // 3.23.0
    if (!drawer || !item) return;
    body.innerHTML = '';
    if (whyP) { whyP.style.display = 'none'; whyP.innerHTML = ''; }
    if (empty) empty.style.display = 'none';
    drawer.style.display = 'flex';

    // Allow caller to pass annotated_url / raw_url explicitly (e.g. the dashboard
    // click handler uses /api/timeline_frame which works even when Store=off and
    // the annotated jpg is not persisted to raw_images/). Falls back to deriving
    // from image_path via _imgUrlsFor otherwise.
    let urls = _imgUrlsFor({img: item.image_path, ship: item.shipment});
    if (item.annotated_url) urls.ann = item.annotated_url;
    if (item.raw_url)       urls.raw = item.raw_url;
    if (!urls.ann && !urls.raw) {
        title.textContent = 'Defect frame';
        if (cnt) cnt.textContent = '';
        if (meta) meta.textContent = '';
        if (empty) { empty.textContent = 'No image path on record.'; empty.style.display = 'block'; }
        return;
    }

    const ts = item.t ? new Date(item.t).toLocaleString() : '';
    const headerCls = item.cls || (item.classes && item.classes[0]) || 'defect';
    title.textContent = headerCls + (ts ? '  —  ' + ts : '');
    if (cnt) cnt.textContent = '';

    // default = Both side-by-side if we have both URLs, otherwise pick whichever exists.
    if (urls.ann && urls.raw) _currentDefectView = 'both';
    else _currentDefectView = urls.ann ? 'ann' : 'raw';
    _renderDefectView(urls);
    _highlightViewToggle();

    if (meta) {
        // 3.21.7: yolo classes have confidence ∈ [0,1] (percentage). Math channels
        // may emit a raw metric value > 1 (channel-specific scale). Format both
        // sensibly: pct for ≤ 1, raw 2-decimal value otherwise.
        // 3.21.17: clearer label ("math metric") + always surface encoder + camera
        // index when the dot came from the encoder scatter (user didn't realize the
        // X-axis was already showing it — now it's in the drawer too).
        const cv = item.best_confidence;
        let conf = '';
        if (cv != null && cv > 0) {
            conf = (cv <= 1)
                ? '  •  conf ' + Math.round(cv * 100) + '%'
                : '  •  math metric ' + (Math.round(cv * 100) / 100);
        }
        let enc = '';
        if (item.encoder != null && Number.isFinite(item.encoder)) {
            enc = '  •  encoder ' + Math.round(item.encoder).toLocaleString();
        }
        let camIdx = '';
        if (item.camera_index != null && Number.isFinite(item.camera_index)) {
            camIdx = '  •  cam ' + Math.round(item.camera_index);
        }
        const classes = (item.classes && item.classes.length) ? item.classes.join(', ') : headerCls;
        meta.innerHTML = '<b>Defects in this frame:</b> ' + classes + conf + enc + camIdx
            + '  &nbsp;|&nbsp;  <span style="color:#94a3b8;">scroll to zoom · drag to pan · double-click to reset</span>';
    }
}

// 3.23.0 — "🤔 Why?" chip: ask the active AI model to explain the currently
// open chart-dot. The backend gathers surrounding context (last 5 min of
// detections on this class/camera, recent inference latency, capture FPS,
// active pipeline, current state) so the model has enough to diagnose
// without us forcing the operator to ask a precise question.
async function askWhyForCurrentDefect() {
    const btn  = document.getElementById('defect-why-btn');
    const panel = document.getElementById('defect-drawer-why');
    const item = _currentDefectItem;
    if (!item) return;
    if (!btn || !panel) return;

    panel.style.display = 'block';
    panel.innerHTML = '<span style="opacity:0.7;">🤔 thinking…</span>';
    btn.disabled = true; btn.style.opacity = '0.6';

    try {
        const payload = {
            metric: item.cls || (item.classes && item.classes[0]) || 'defect',
            value:  item.best_confidence ?? null,
            timestamp: item.t || null,
            camera: item.camera_index ?? null,
            encoder: item.encoder ?? null,
            shipment: item.shipment || null,
            image_path: item.image_path || null,
            window_seconds: 300,  // 5 min before + after the dot
            language: window.currentLang || 'en',  // 3.23.1 — AI replies in operator's UI language
            mode: 'dot',
        };
        const r = await fetch('/api/why', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await r.json();
        if (!r.ok) {
            panel.innerHTML = '<span style="color:#fca5a5;">' +
                (data.error || ('Request failed (' + r.status + ')')) + '</span>';
            return;
        }
        const ans = (data.answer || '').trim();
        const model = data.model || '';
        const usage = data.usage ? '<span style="opacity:0.5; font-size:11px;"> · ' + data.usage + '</span>' : '';
        panel.innerHTML = '<div style="color:#a7f3d0; font-size:11px; margin-bottom:4px;">🤔 Why? — via <b>' +
            (model || 'AI') + '</b>' + usage + '</div>' +
            '<div>' + ans.replace(/\n/g, '<br>') + '</div>';
    } catch (e) {
        panel.innerHTML = '<span style="color:#fca5a5;">Network error: ' + (e.message || e) + '</span>';
    } finally {
        btn.disabled = false; btn.style.opacity = '1';
    }
}
window.askWhyForCurrentDefect = askWhyForCurrentDefect;


// =====================================================================
// 3.25.4 — Quality charts: per-shipment bar + time/encoder 1D heatmaps
// =====================================================================

let _qualityShipmentsChart = null;


function _verdictColor(v) {
    if (v === 'RELEASE')    return 'rgba(34,197,94,0.7)';
    if (v === 'RE-INSPECT') return 'rgba(234,179,8,0.7)';
    if (v === 'HOLD')       return 'rgba(239,68,68,0.7)';
    return 'rgba(100,116,139,0.5)';
}

// 4.0.95 — Canvas pattern generator for chart fills. Renders diagonal stripes on a
// small tile with the given color, then wraps it as a repeating CanvasPattern so
// Chart.js can use it as `backgroundColor` on any bar. Used on the Score-per-shipment
// chart so the "Relative (window)" bar shares the SAME verdict color as its paired
// "Absolute" bar but is distinguished by the striped fill instead of a different hue.
function _makeStripedPattern(color, tileSize = 8, stripeWidth = 3) {
    const c = document.createElement('canvas');
    c.width = c.height = tileSize;
    const ctx = c.getContext('2d');
    // Faint base tint so bars aren't fully transparent between stripes.
    ctx.fillStyle = String(color).replace(/,[\d.]+\)/, ',0.15)');
    ctx.fillRect(0, 0, tileSize, tileSize);
    // Diagonal opaque stripes in the actual verdict color.
    ctx.strokeStyle = color;
    ctx.lineWidth = stripeWidth;
    ctx.beginPath();
    ctx.moveTo(-stripeWidth, tileSize + stripeWidth);
    ctx.lineTo(tileSize + stripeWidth, -stripeWidth);
    ctx.stroke();
    return ctx.createPattern(c, 'repeat');
}

async function refreshQualityCharts() {
    // v4.0.210 — the quality + ejection strips now come from /api/detection_charts
    // (the SAME endpoint + binning as the scatter), NEVER a separate quality API,
    // so they can never misalign. The INITIAL newest→oldest fill is owned by the
    // progressive ladder (refreshAdvancedCharts). This function is the PERIODIC /
    // fallback refresher (60 s timer, tab-open): once a scatter render has
    // established the domain, re-fetch the strips with ONE full-window slice over
    // that exact [lo,hi]/N_BINS domain and repaint. No-op before the first render.
    const rng = _progressiveColorRange;
    if (!rng || !(rng.hi > rng.lo)) return;
    // v4.0.230 — NEVER wipe + re-stream the strip accumulators while the ladder is
    // mid-load: that clears the quality/ejection/shipment cells the ladder is still
    // filling and BLANKS the strips (the 60s timer landing on a ~60s heavy-machine
    // load = the reported "goes blank at the end"). The ladder owns the strips
    // while active; this periodic refresher only runs once the load has settled.
    if (_ladderActive) return;
    const shipment = document.getElementById('insight-shipment')?.value || '';
    const win = document.getElementById('insight-window')?.value || '24h';
    const hiMs = Date.now();
    // Time window that BOUNDS the rows; the binning domain (bin_ref) is
    // _progressiveColorRange, so a vertical guideline still hits the same data.
    const loMs = shipment ? (hiMs - _windowToMs('90d')) : (hiMs - _windowToMs(win));
    // v4.0.230 — do NOT wipe the accumulators first. _extendStripsFromBucket does
    // one wide slice over the current domain and overwrites each bin, so the strips
    // update IN PLACE with no blank gap. Wiping made them flash empty for the ~1.3s
    // the wide slice takes to return on a heavy machine (the 60s "blank" flicker).
    try { await _extendStripsFromBucket(loMs, hiMs, shipment, _progressiveLadderTicket); }
    catch (_e) { /* strip refresh failures are non-fatal */ }
}
window.refreshQualityCharts = refreshQualityCharts;

// v4.0.232 — the "🔬 Defect Diagnostics" panel is retired (removed from the current
// status.html). This hides it on any machine still serving the OLD status.html (so
// a charts.js-only ripple removes it fleet-wide without shipping status.html — which
// would otherwise add the Knowledge tab to sites that have no knowledge service).
document.addEventListener('DOMContentLoaded', function () {
    var _dd = document.getElementById('quality-insight-panel');
    if (_dd) _dd.style.display = 'none';
});

// 4.0.202 — dedicated, window-INDEPENDENT loader for the Score-per-shipment
// card. Shows the last N shipments regardless of the timeframe selector.
// Called once on Charts-tab open + on the card's own Refresh button + when the
// shipment-count input changes. NOT wired into refreshDetectionInsights /
// refreshQualityCharts, so picking a shipment or changing the window no longer
// re-fetches it.
let _scorePerShipmentLoaded = false;
let _scorePerShipmentInFlight = false;
// 4.0.203 — mount a small centred "Loading…" overlay inside the Score-per-
// shipment card so the operator never sees a blank gap wondering if it broke.
function _mountScoreSpinner() {
    const canvas = document.getElementById('quality-shipments-chart');
    const host = canvas && canvas.parentElement;
    if (!host) return null;
    if (getComputedStyle(host).position === 'static') host.style.position = 'relative';
    let el = document.getElementById('mve-score-spinner');
    if (!el) {
        el = document.createElement('div');
        el.id = 'mve-score-spinner';
        el.style.cssText = 'position:absolute; inset:0; display:flex; align-items:center; justify-content:center; gap:8px; color:#cbd5e1; font-size:12px; background:rgba(15,23,42,0.35); z-index:5;';
        el.innerHTML = '<span style="width:14px;height:14px;border:2px solid #475569;border-top-color:#22c55e;border-radius:50%;display:inline-block;animation:mve-spin 0.8s linear infinite;"></span> Loading score per shipment…';
        host.appendChild(el);
        if (!document.getElementById('mve-spin-kf')) {
            const st = document.createElement('style');
            st.id = 'mve-spin-kf';
            st.textContent = '@keyframes mve-spin{to{transform:rotate(360deg)}}';
            document.head.appendChild(st);
        }
    }
    return el;
}
function _unmountScoreSpinner() {
    const el = document.getElementById('mve-score-spinner');
    if (el) el.remove();
}
async function loadScorePerShipment(force) {
    // Already loaded and not a forced reload → nothing to do.
    if (_scorePerShipmentLoaded && !force) return;
    // 4.0.203 — in-flight guard. Rapid/overlapping callers (tab-open fires it,
    // a long-window refresh forces it, DOMContentLoaded races) previously
    // stacked renders → the chart "showed up multiple times". Collapse any
    // concurrent call into the one already running.
    if (_scorePerShipmentInFlight) return;
    _scorePerShipmentInFlight = true;
    _mountScoreSpinner();
    try {
        await _loadQualityShipmentsChart();
        _scorePerShipmentLoaded = true;
    } catch (e) {
        console.warn('Score-per-shipment load failed:', e);
        _scorePerShipmentLoaded = false;
    } finally {
        _scorePerShipmentInFlight = false;
        _unmountScoreSpinner();
    }
}
window.loadScorePerShipment = loadScorePerShipment;

// v4.0.222 — self-healing watchdog for the Score-per-shipment card. Operator:
// "when I open the Charts tab it doesn't show — have a flag that checks if
// Score-per-shipment doesn't exist and loads it." Two failure modes it fixes:
//   1) the chart was BUILT while the Charts tab was hidden → its canvas was 0×0
//      → Chart.js drew nothing → it stays blank until a manual Refresh. Here we
//      just resize()+update() it (data is already in memory — no refetch).
//   2) it never loaded at all (tab-open never triggered loadScorePerShipment) →
//      force a load.
// Called from switchTab('grafana') at a few delays so it survives the canvas-
// layout + first-fetch races. Safe to call repeatedly (no spam: resizes when it
// already has data, and won't stack a fetch while one is in flight).
function _ensureScorePerShipmentLoaded() {
    try {
        const canvas = document.getElementById('quality-shipments-chart');
        if (!canvas) return;
        const inst = (window.Chart && Chart.getChart) ? Chart.getChart(canvas) : null;
        const hasData = inst && inst.data && Array.isArray(inst.data.datasets)
            && inst.data.datasets.some(ds => Array.isArray(ds.data) && ds.data.length > 0);
        if (hasData) {
            // Rendered blank because the canvas was 0×0 while hidden → just redraw.
            try { inst.resize(); inst.update('none'); } catch (_e) {}
        } else if (!_scorePerShipmentInFlight) {
            // Never populated (or genuinely empty) → (re)load. force=true bypasses
            // the _scorePerShipmentLoaded guard so a prior blank render can't block it.
            try { loadScorePerShipment(true); } catch (_e) {}
        }
    } catch (_e) { /* watchdog is best-effort */ }
}
window._ensureScorePerShipmentLoaded = _ensureScorePerShipmentLoaded;

async function _loadQualityShipmentsChart(win) {
    const canvas = document.getElementById('quality-shipments-chart');
    if (!canvas) return;
    // 4.0.96 — honor the timeline dropdown (was hardcoded 30d). Also update the
    // subtitle so the operator sees which window drove this render.
    // 4.0.97 — shipment COUNT is now operator-controlled via #sqs-shipments input
    // (persisted in localStorage.mve_sqs_count). Default 30.
    // 4.0.99 — renamed from #sqs-count to avoid id-collision with the
    // Detections span at status.html:1029 (both had id="sqs-count").
    // Collision caused Detections-count writes to spray into the input's
    // value/layout, breaking the whole Score-per-shipment card layout.
    const nInput = document.getElementById('sqs-shipments');
    const nRaw = parseInt((nInput && nInput.value) || localStorage.getItem('mve_sqs_count') || '30', 10);
    const n = Math.max(5, Math.min(200, isNaN(nRaw) ? 30 : nRaw));
    if (nInput && nInput.value !== String(n)) nInput.value = String(n);
    const subtitleEl = document.getElementById('sqs-subtitle');
    // 4.0.202 — window-INDEPENDENT: show the last N shipments regardless of the
    // timeframe selector (operator: "show the last 30 shipments, don't depend
    // on window, remove it entirely"). Subtitle no longer mentions a window.
    if (subtitleEl) subtitleEl.textContent = `· last ${n} shipments · color = verdict`;
    try {
        // v4.0.236 — no window param (operator: "remove the window from the api
        // entirely; query the last N shipments, no matter the time"). The backend
        // escalates its scan (24h→3d→7d→…) until it has N, so this never full-scans
        // / times out the way `window=all` did (which blanked the card on khoy).
        const r = await fetch(`/api/quality/shipments?n=${n}`);
        const d = await r.json();
        const rows = (d.shipments || []).slice().reverse();  // oldest left → newest right
        // v4.0.243 — TREND chip in the header: older-third vs newer-third of the scores
        // across the shown shipments (operator: "put the trend here — improving /
        // degrading / stable, based on the relative scores"). Relative rescaling is
        // monotonic within the window, so the DIRECTION is identical to absolute scores.
        try {
            const _chip = document.getElementById('sqs-list-trend');
            if (_chip) {
                const _sc = rows.map(s => Number(s && s.score)).filter(Number.isFinite);
                if (_sc.length >= 4) {
                    const _k = Math.max(1, Math.floor(_sc.length / 3));
                    const _avg = a => a.reduce((x, y) => x + y, 0) / a.length;
                    const _d = _avg(_sc.slice(_sc.length - _k)) - _avg(_sc.slice(0, _k));  // newest − oldest
                    let _txt, _bg, _fg;
                    if (_d > 0.5)      { _txt = '↗ Improving'; _bg = 'rgba(34,197,94,0.25)'; _fg = '#6ee7b7'; }
                    else if (_d < -0.5){ _txt = '↘ Degrading'; _bg = 'rgba(239,68,68,0.25)'; _fg = '#fca5a5'; }
                    else               { _txt = '→ Stable';    _bg = 'rgba(71,85,105,0.5)';  _fg = '#cbd5e1'; }
                    _chip.textContent = _txt;
                    _chip.style.background = _bg; _chip.style.color = _fg; _chip.style.display = 'inline-block';
                } else { _chip.style.display = 'none'; }
            }
        } catch (_tr) { /* trend chip is best-effort */ }
        // v4.0.221 — cache shipment→verdict so the shipment LANE can colour each
        // bin by its shipment's verdict (operator: "colour the shipment lane by
        // the verdict of that shipment"). Reuses these already-computed verdicts —
        // no extra backend cost. Repaint the lane once so it recolours immediately.
        try {
            window._mveShipmentVerdicts = window._mveShipmentVerdicts || {};
            // v4.0.222 — also cache full per-shipment META so the shipment-lane
            // hover can show length / start-end / score / verdict, not just the id.
            window._mveShipmentMeta = window._mveShipmentMeta || {};
            for (const s of rows) {
                if (s && s.shipment != null) {
                    const id = String(s.shipment);
                    if (s.verdict) window._mveShipmentVerdicts[id] = String(s.verdict);
                    window._mveShipmentMeta[id] = {
                        score: s.score, verdict: s.verdict,
                        encoder_span: s.encoder_span, encoder_unit: s.encoder_unit,
                        encoder_min: s.encoder_min, encoder_max: s.encoder_max,  // 4.0.263 — real encoder start/end for the tooltip
                        first_t: s.first_t, last_t: s.last_t,
                        rows: s.rows, top_defects: s.top_defects
                    };
                }
            }
            if (typeof _repaintProgressiveStrips === 'function') _repaintProgressiveStrips();
        } catch (_vmap) {}
        // v4.0.140 — 2-line labels: line 1 = shipment ID, line 2 = length
        // formatted through the shared helper so it agrees with tooltip
        // and bar annotation on the same shipment. Uses per-row upu+label
        // (from shipment_quality) with global cache as fallback.
        const _lengthLabel = (s) => {
            if (!Number(s.encoder_span)) return '';
            return _fmtLenWithUpu(s.encoder_span, _upuFromAny(s, d), _unitLabelFromAny(s, d));
        };
        const labels = rows.map(s => [String(s.shipment), _lengthLabel(s)]);

        // 4.0.63 — TWO bars per shipment:
        //   (a) Absolute score — 100 × (1 − impact_per_unit) — comparable across
        //       time and sites, doesn't move because you calibrated the fleet.
        //   (b) Window-relative score — rescaled so the RANGE of the currently
        //       visible shipments spans the full 0-100 axis, i.e. the worst
        //       shipment in this window → 0 and the best → 100. Operator asked
        //       for this specifically because when every absolute score sits at
        //       99-100 (a well-behaved fleet), all bars look identical and
        //       there is nothing to compare. Relative expands whatever spread
        //       exists in the window so subtle drift becomes visible.
        //
        // Rescale math: shift-and-scale by the OBSERVED spread of scores in
        // the chart. Rank the window on WHICHEVER of score_absolute /
        // raw score has real spread — v4.0.101 fallback for the case where
        // severity calibration is unset (see khoy 2026-07-12): every shipment
        // clamps to `score_absolute=100` but the raw `score` varies 90–100.
        // Before this fallback the operator saw "all Relative bars grey (all
        // tied)" and correctly complained that nothing distinguished them.
        const absVals = rows.map(s => (s.score_absolute != null ? s.score_absolute
                                     : (s.score != null ? s.score : 0)));
        const rawVals = rows.map(s => (s.score         != null ? s.score
                                     : (s.score_absolute != null ? s.score_absolute : 0)));
        const spreadOf = (arr) => {
            const v = arr.filter(x => x != null && !isNaN(x));
            return v.length ? (Math.max(...v) - Math.min(...v)) : 0;
        };
        const absSpread = spreadOf(absVals);
        const rawSpread = spreadOf(rawVals);
        // Prefer absVals when it has real spread. Otherwise fall back to raw
        // score. Only when BOTH are flat do we render the neutral stripe.
        const useRaw = absSpread < 1e-9 && rawSpread >= 1e-9;
        const relSrc = useRaw ? rawVals : absVals;
        const relSrcMin = Math.min(...relSrc.filter(v => v != null && !isNaN(v)));
        const relSrcMax = Math.max(...relSrc.filter(v => v != null && !isNaN(v)));
        const relSrcSpread = relSrcMax - relSrcMin;
        const relData = relSrc.map(v => {
            if (v == null || isNaN(v)) return 0;
            if (relSrcSpread < 1e-9) return 100;  // truly flat → all 100
            return Number((((v - relSrcMin) / relSrcSpread) * 100).toFixed(1));
        });

        const colors   = rows.map(s => _verdictColor(s.verdict));
        // 4.0.97 (revised) — Two bars per shipment, same verdict color, distinguished
        // by fill pattern. Absolute = solid; Relative = diagonal-striped.
        // 4.0.99/4.0.101 — Relative bar color signals HOW the ranking was computed:
        //   - verdict color: normal case (Absolute scores spread — real relative)
        //   - verdict color with a small annotation in the subtitle: fallback
        //     to raw `score` because Absolute was flat (calibration issue)
        //   - slate grey: NEITHER has spread — every shipment truly tied
        const _relFlat = relSrcSpread < 1e-9;
        const NEUTRAL_STRIPE_BASE = 'rgba(148,163,184,0.55)';   // slate-400
        const relColors = _relFlat
            ? rows.map(() => _makeStripedPattern(NEUTRAL_STRIPE_BASE))
            : colors.map(c => _makeStripedPattern(c));
        if (subtitleEl) {
            // 4.0.202 — window-independent: "last N shipments", no window.
            subtitleEl.textContent = _relFlat
                ? `· last ${n} shipments · color = verdict · relative = flat (all tied)`
                : (useRaw
                    ? `· last ${n} shipments · color = verdict · relative = raw score (Absolute uncalibrated)`
                    : `· last ${n} shipments · color = verdict`);
        }

        if (_qualityShipmentsChart) _qualityShipmentsChart.destroy();
        _qualityShipmentsChart = new Chart(canvas, {
            type: 'bar',
            data: {
                labels,
                datasets: [
                    { label: 'Absolute (solid)', data: absVals,
                      backgroundColor: colors,
                      borderColor: colors.map(c => String(c).replace(/,[\d.]+\)/, ',1)')),
                      borderWidth: 1, order: 1 },
                    { label: 'Relative (striped)', data: relData,
                      backgroundColor: relColors,
                      borderColor: colors.map(c => String(c).replace(/,[\d.]+\)/, ',1)')),
                      borderWidth: 1, order: 1 },
                ],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                // 4.0.99 — reserve room BELOW the x-axis for a second line under
                // each shipment ID: line 1 = shipment ID (native tick), line 2 =
                // length ("125 m"). The shipmentAnnotations plugin writes line 2
                // into this padded area. Rotated ticks are ~35–50°, so ~18px of
                // padding fits an extra 10px text row without overlapping.
                layout: { padding: { bottom: 18 } },
                // 4.0.207 — click a bar → load that shipment in Detection
                // Insights. Operator: "when I click a shipment in Score per
                // shipment, put that shipment in the search field so we can see
                // its details." Sets the picker + fires the normal pick flow
                // (hides the timeframe, loads the shipment's dots/colour).
                onClick: (evt, els) => {
                    if (els && els.length) {
                        const s = rows[els[0].index];
                        if (s && s.shipment) {
                            const inp = document.getElementById('insight-shipment');
                            if (inp) {
                                inp.value = String(s.shipment);
                                if (typeof _onShipmentPicked === 'function') _onShipmentPicked();
                                try { inp.scrollIntoView({ behavior: 'smooth', block: 'center' }); } catch (_) {}
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#e2e8f0',          // brighter for dark bg
                            font: { size: 11, weight: '500' },
                            boxWidth: 14,
                            padding: 10,
                        },
                        onClick: () => {},
                    },
                    tooltip: {
                        // 4.0.97 — mode='index' shows BOTH bars in a single tooltip
                        // when the operator hovers a shipment. Fixes prior confusion
                        // where hovering the Relative bar led with "Absolute:" text.
                        mode: 'index',
                        intersect: false,
                        // 4.0.99 — anchor tooltip ABOVE the bar (yAlign:'bottom' =
                        // caret-on-bottom = box-above-caret) and CENTER on the bar
                        // so it can grow tall without being clipped by the canvas
                        // bottom edge. Previous default 'nearest' put it beside the
                        // bar and the last 1-2 lines fell off the canvas because
                        // Chart.js rasters tooltips onto the canvas and they cannot
                        // overflow. Also compressed lines below to reduce height.
                        position: 'nearest',
                        yAlign: 'bottom',
                        xAlign: 'center',
                        titleFont: { size: 12, weight: '600' },
                        bodyFont: { size: 11 },
                        bodySpacing: 3,
                        padding: 8,
                        callbacks: {
                            title: (items) => items[0].label,
                            // Per-dataset one-liner so the operator sees which value
                            // corresponds to which swatch.
                            label: (item) => {
                                const r = rows[item.dataIndex];
                                if (item.datasetIndex === 0) {
                                    const abs = r.score_absolute ?? r.score;
                                    return `  Abs: ${abs != null ? Number(abs).toFixed(1) : '—'}`;
                                }
                                return `  Rel: ${relData[item.dataIndex]}`;
                            },
                            // 4.0.98 — Shared context lines: verdict, detections,
                            // impact-per-unit, LENGTH, DEFECTS-PER-100-UNITS,
                            // DELTA-VS-PREVIOUS shipment.
                            // 4.0.99 — merged pairs of lines so the box is shorter
                            // and never clips at the canvas bottom edge.
                            afterBody: (items) => {
                                const i = items[0].dataIndex;
                                const r = rows[i];
                                const tops = (r.top_defects || []).join(', ');
                                const impPerUnit = r.impact_per_unit;
                                const span = r.encoder_span;
                                // v4.0.140 — shared upu + unit label from the
                                // per-shipment row (shipment_quality carries
                                // its own upu snapshot) or global cache.
                                const upu = _upuFromAny(r, d);
                                const unitLabel = _unitLabelFromAny(r, d);
                                const per100 = (impPerUnit != null) ? Number(impPerUnit) * 100 : null;
                                const abs = r.score_absolute ?? r.score;
                                let deltaStr = '';
                                if (i > 0 && abs != null) {
                                    const prev = rows[i - 1].score_absolute ?? rows[i - 1].score;
                                    if (prev != null && prev !== 0) {
                                        const dv = Number(abs) - Number(prev);
                                        const sign = dv >= 0 ? '↑+' : '↓';
                                        deltaStr = `${sign}${Math.abs(dv).toFixed(1)}`;
                                    }
                                }
                                const lengthLabel = span ? _fmtLenWithUpu(span, upu, unitLabel) : '';
                                const lengthPart = lengthLabel
                                    ? (per100 != null
                                        ? `${lengthLabel}  (${per100.toFixed(2)} def/100 ${unitLabel})`
                                        : lengthLabel)
                                    : '';
                                return [
                                    `${r.verdict || '—'} · ${(r.rows || 0).toLocaleString()} det${deltaStr ? '  Δ ' + deltaStr : ''}`,
                                    lengthPart ? `Length: ${lengthPart}` : '',
                                    impPerUnit != null ? `Impact/${unitLabel}: ${Number(impPerUnit).toFixed(4)}` : '',
                                    tops ? `Top: ${tops}` : '',
                                ].filter(Boolean);
                            },
                        },
                    },
                },
                scales: {
                    x: { ticks: { color: '#94a3b8', font: { size: 9 }, maxRotation: 50, minRotation: 35 }, grid: { display: false } },
                    y: { min: 0, max: 100, ticks: { color: '#94a3b8', font: { size: 10 }, stepSize: 20 },
                         grid: { color: 'rgba(148,163,184,0.1)' },
                         title: { display: true, text: 'Score / 100', color: '#94a3b8', font: { size: 10 } } },
                },
            },
            plugins: [{
                // 4.0.99 — length label is now on line 2 UNDER the shipment ID
                // (was above the bar, where it collided with the delta arrow).
                // Delta arrow stays above the bar top.
                // Product-agnostic: uses whatever `encoder_unit` the server returned.
                id: 'shipmentAnnotations',
                afterDatasetsDraw(chart) {
                    const meta0 = chart.getDatasetMeta(0);
                    const meta1 = chart.getDatasetMeta(1);
                    const ctx = chart.ctx;
                    ctx.save();
                    ctx.textAlign = 'center';
                    ctx.font = '10px system-ui, sans-serif';
                    // v4.0.144 — `lengthY` used to position the horizontal
                    // length text under each bar; removed with that draw.
                    rows.forEach((r, i) => {
                        const bar0 = meta0.data[i]; if (!bar0) return;
                        const bar1 = meta1 && meta1.data[i];
                        const centerX = bar1 ? (bar0.x + bar1.x) / 2 : bar0.x;
                        const topY = Math.min(bar0.y, bar1 ? bar1.y : bar0.y);
                        // v4.0.144 — removed the horizontal `234 m` annotation
                        // drawn under each bar. Duplicated the length string
                        // that already renders as line 2 of the tilted X-axis
                        // label (see labels = rows.map(s => [id, lengthLabel])
                        // above), so it was pure noise. Delta arrow stays.
                        // Delta vs previous shipment — above the bar top; no
                        // longer competes with the length label since length
                        // has moved to the bottom row.
                        // 4.0.101 — mirror the Relative bar's spread-source
                        // choice: if score_absolute is flat because severities
                        // aren't calibrated, use raw score for the delta so it
                        // still surfaces trend. Before this fallback the delta
                        // was suppressed on any window where every abs score
                        // clamped to 100, which is exactly khoy's current state.
                        if (i > 0) {
                            const pickScore = (s) => (useRaw
                                ? (s.score ?? s.score_absolute)
                                : (s.score_absolute ?? s.score));
                            const abs  = pickScore(r);
                            const prev = pickScore(rows[i-1]);
                            if (abs != null && prev != null && prev !== 0) {
                                const d = Number(abs) - Number(prev);
                                if (Math.abs(d) >= 0.1) {
                                    const arrow = d >= 0 ? '▲' : '▼';
                                    const color = d >= 0 ? '#22c55e' : '#ef4444';
                                    ctx.fillStyle = color;
                                    ctx.font = 'bold 9px system-ui, sans-serif';
                                    ctx.fillText(`${arrow}${Math.abs(d).toFixed(1)}`,
                                        centerX, Math.max(12, topY - 6));
                                }
                            }
                        }
                    });
                    ctx.restore();
                },
            }],
        });
    } catch (e) { console.warn('shipments chart failed', e); }
}

// 3.25.12 — deterministic color per procedure name (golden-angle hue). Same name
// → same color across the time + encoder strip so the operator can visually
// match an ejection cluster across both axes.
function _procColor(name) {
    if (!name) return 'rgba(100,116,139,0.5)';
    let h = 0;
    for (let i = 0; i < name.length; i++) { h = ((h << 5) - h + name.charCodeAt(i)) | 0; }
    // 137.508° is the golden angle — gives well-separated hues for distinct names.
    const hue = ((h * 137.508) % 360 + 360) % 360;
    return `hsl(${hue.toFixed(0)}, 70%, 55%)`;
}

// v4.0.209 — Build the `&lo=&hi=&shipment=` query fragment that pins a strip
// endpoint to the SCATTER's exact x-domain (window._mveUnifiedX). This is the
// single source of truth for the x-range: the scatter, the colour heatmap and
// both strips all bucket over the SAME [xMin, xMax] with the SAME bin count, so
// any vertical guideline the operator reads off the chart points at the same
// encoder value / timestamp in every layer. The range is only emitted when the
// scatter's resolved axisMode matches the strip's axis (guards a stale
// time-domain from leaking into an encoder query, and vice-versa). For time,
// xMin/xMax are epoch-milliseconds; for encoder they are encoder units — both
// exactly what the backend's lo/hi params expect.



// 4.0.164 — colour-change strip. Fed from `data.color_change_strip` (encoder
// axis) or `data.color_change_strip_time` (time axis). Each bucket cell's
// colour = SIGNED delta_prev normalized to the span's max |delta|:
//   • positive delta (E went UP → warmer/lighter) → warm ramp (yellow→red)
//   • negative delta (E went DOWN → cooler/darker) → cool ramp (cyan→blue)
//   • ~zero delta → near-white/neutral
// Biggest |delta| in the span gets the deepest saturation. Empty bins →
// transparent slot so positions align with quality + ejection strips 1:1.
// Prior v4.0.162 used |delta| thresholds (green/yellow/orange/red) which
// hid whether the shift was warming or cooling. Direction matters — a
// yellowing product ≠ a darkening one.
function _colorChangeColor(delta, maxAbs) {
    if (delta == null || !Number.isFinite(delta)) return 'transparent';
    if (!Number.isFinite(maxAbs) || maxAbs <= 0) {
        return 'rgba(148,163,184,0.6)';    // neutral gray (no variation in span)
    }
    // Clamp to [-1, +1] normalized by span max.
    const f = Math.max(-1, Math.min(1, delta / maxAbs));
    const mag = Math.abs(f);
    // v4.0.221 — brightness parity with the quality/shipment strips (which are
    // fully opaque). Operator: "why are the colour-change cells not as bright as
    // the others?" The old 0.35..0.9 alpha ramp made cells translucent + dim.
    // Now 0.78..1.0 so cells read as vividly as the neighbours; magnitude is
    // still conveyed by the yellow→red / cyan→blue hue ramp below.
    const alpha = (0.78 + 0.22 * mag).toFixed(2);
    if (f >= 0) {
        // warm: interpolate from near-neutral (yellow-orange) at f=~0 to red at f=1.
        // Endpoints: (250,204,21) yellow → (239,68,68) red.
        const r = Math.round(250 - (250 - 239) * mag);
        const g = Math.round(204 - (204 -  68) * mag);
        const b = Math.round( 21 - ( 21 -  68) * mag);
        return `rgba(${r},${g},${b},${alpha})`;
    } else {
        // cool: neutral cyan at f=~0 → deep blue at f=-1.
        // Endpoints: (103,232,249) cyan → (59,130,246) blue.
        const r = Math.round(103 - (103 -  59) * mag);
        const g = Math.round(232 - (232 - 130) * mag);
        const b = Math.round(249 - (249 - 246) * mag);
        return `rgba(${r},${g},${b},${alpha})`;
    }
}
function _renderColorChangeStrip(axisMode, data) {
    const wrap = document.getElementById('color-change-strip-wrap');
    const strip = document.getElementById('color-change-strip');
    if (!wrap || !strip) return;
    const enabled = (typeof data?.parent_color_check_enabled === 'boolean')
        ? data.parent_color_check_enabled
        : (window._mveColorCheckEnabled === true);
    if (!enabled) { wrap.style.display = 'none'; return; }
    const payload = (axisMode === 'encoder')
        ? (data.color_change_strip || {})
        : (data.color_change_strip_time || {});
    const nBins = Math.max(1, Number(payload.n_bins) || 48);
    const cells = Array.isArray(payload.cells) ? payload.cells : [];
    if (!cells.length) { wrap.style.display = 'none'; return; }
    // v4.0.220 — restore the row to FLEX (not '', which falls back to block since
    // there is no .mve-strip-row CSS rule). Block dropped align-items:center, so
    // the 14px colour-change strip sat top-aligned in the row and READ as offset
    // next to the 16px strips. flex keeps it vertically centred like the siblings.
    wrap.style.display = 'flex';
    const byBin = new Map();
    let maxAbs = 0;
    for (const c of cells) {
        byBin.set(Number(c.bin), c);
        const a = Math.abs(Number(c.delta_prev) || 0);
        if (a > maxAbs) maxAbs = a;
    }
    const html = [];
    for (let i = 0; i < nBins; i++) {
        const c = byBin.get(i);
        if (!c) {
            html.push('<div style="flex:1; background:transparent;" title="no colour samples in this bucket"></div>');
        } else {
            // 4.0.164 — pass the span's max |Δ| so the divergent colour ramp
            // is normalized within THIS chart's visible cells (biggest jump
            // in the span always paints deepest, regardless of absolute
            // magnitude — matches operator's "find max/min in span" ask).
            const bg = _colorChangeColor(Number(c.delta_prev), maxAbs);
            const dp = Number(c.delta_prev) || 0;
            const arrow = dp > 0 ? '↑ warmer' : dp < 0 ? '↓ cooler' : '→ same';
            const tip = `bin ${i}\nΔ from prev: ${dp >= 0 ? '+' : ''}${dp} (${arrow})\nmean E: ${c.E}\nsamples: ${c.n}\nspan max |Δ|: ${maxAbs.toFixed(2)}`;
            // 4.0.166 — inline label: arrow + rounded value inside the cell,
            // same idea as the ▲/▼ badges on the shipment-score bars. Hidden
            // when |Δ| < 0.05 (noise) so quiet buckets stay clean. Text
            // clipped by overflow:hidden if the cell narrows below ~24 px.
            let label = '';
            if (Math.abs(dp) >= 0.05) {
                const mag = Math.abs(dp);
                const val = mag >= 10 ? mag.toFixed(0) : mag.toFixed(1);
                label = (dp > 0 ? '↑' : '↓') + val;
            }
            html.push(`<div style="flex:1; background:${bg}; cursor:default; display:flex; align-items:center; justify-content:center; font-size:9px; line-height:1; font-weight:700; color:rgba(15,23,42,0.95); text-shadow:-1px -1px 0 rgba(255,255,255,.9),1px -1px 0 rgba(255,255,255,.9),-1px 1px 0 rgba(255,255,255,.9),1px 1px 0 rgba(255,255,255,.9); overflow:hidden; white-space:nowrap;" title="${tip.replace(/"/g,'&quot;')}">${label}</div>`);
        }
    }
    strip.innerHTML = html.join('');
    // Match the encoder/time label under the quality strip.
    const title = document.getElementById('color-change-strip-title');
    if (title) title.innerHTML = '🎨 ' + _stripTipIcon('color change by ' + (axisMode === 'time' ? 'time' : 'encoder')
        + ' — warmer = E went up · cooler = E went down · deepest = biggest jump in span');
}
window._renderColorChangeStrip = _renderColorChangeStrip;



// Auto-refresh when the Charts tab opens or window changes.
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(refreshQualityCharts, 1800);
    // Refresh every 60s while the tab is visible.
    setInterval(() => {
        const grafanaTab = document.getElementById('tab-grafana');
        if (grafanaTab && grafanaTab.classList.contains('active')) refreshQualityCharts();
    }, 60000);
    // 3.25.6 — keep strip wraps inset to scatter plot area on viewport resize.
    window.addEventListener('resize', () => {
        try {
            // 3.25.13 — single merged scatter + single strip wrap.
            if (_insightCameraScatter) _alignStripToScatter(_insightCameraScatter, 'quality-strip-wrap');
        } catch (e) {}
    });
});


// 3.23.1 — Why? chip on the Dashboard Shipment Quality Score card. Asks the
// active AI to explain the CURRENT score in the operator's UI language.
async function askWhyForScore(btn) {
    const panel = document.getElementById('sqs-why-panel');
    const scoreEl = document.getElementById('sqs-score');
    const verdictEl = document.getElementById('sqs-verdict');
    if (!panel) return;
    panel.style.display = 'block';
    panel.innerHTML = '<span style="opacity:0.7;">🤔 thinking…</span>';
    if (btn) { btn.disabled = true; btn.style.opacity = '0.6'; }
    const score = scoreEl ? parseFloat(scoreEl.textContent) : null;
    const verdict = verdictEl ? (verdictEl.textContent || '').trim() : '';
    try {
        const r = await fetch('/api/why', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mode: 'score',
                metric: 'shipment_quality_score',
                value: Number.isFinite(score) ? score : null,
                window_seconds: 3600,
                shipment: (window._currentShipment || null),
                language: window.currentLang || 'en',
                extra: { verdict: verdict },
            }),
        });
        const d = await r.json();
        if (!r.ok) {
            panel.innerHTML = '<span style="color:#fca5a5;">' + (d.error || ('Request failed (' + r.status + ')')) + '</span>';
            return;
        }
        const ans = (d.answer || '').trim();
        panel.innerHTML =
            '<div style="color:#a7f3d0; font-size:10px; margin-bottom:3px;">🤔 via <b>' + (d.model || 'AI') + '</b></div>' +
            '<div>' + ans.replace(/\n/g, '<br>') + '</div>';
    } catch (e) {
        panel.innerHTML = '<span style="color:#fca5a5;">Network error: ' + (e.message || e) + '</span>';
    } finally {
        if (btn) { btn.disabled = false; btn.style.opacity = '1'; }
    }
}
window.askWhyForScore = askWhyForScore;


// 3.21.22 — upload the currently-shown raw frame to ai-trainer.monitait.com.
// Posts {image_path, class_name, shipment, camera} to /api/ai_trainer/upload
// which strips the _DETECTED suffix, validates the path, and forwards the
// bytes to the configured trainer URL.
async function uploadCurrentDefectToAiTrainer() {
    const btn = document.getElementById('defect-upload-trainer');
    const meta = document.getElementById('defect-drawer-meta');
    const item = _currentDefectItem;
    if (!item || !item.image_path) {
        if (meta) meta.innerHTML = '<span style="color:#f87171;">No frame loaded — click a chart dot first.</span>';
        return;
    }
    const origText = btn ? btn.textContent : '';
    if (btn) { btn.disabled = true; btn.textContent = '⏳ Uploading…'; }
    try {
        const body = {
            image_path: item.image_path,
            class_name: item.cls || (item.classes && item.classes[0]) || '',
            shipment: item.shipment || '',
            camera: item.camera_index != null ? String(item.camera_index) : '',
        };
        const r = await fetch('/api/ai_trainer/upload', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const d = await r.json();
        if (r.ok && d.success) {
            if (btn) btn.textContent = '✓ Sent';
            if (meta) {
                const tid = d.task_id ? ` (task_id=${d.task_id})` : '';
                meta.innerHTML = `<span style="color:#86efac;">📤 Raw image uploaded to AI Trainer${tid}.</span>`;
            }
        } else {
            if (btn) btn.textContent = '⚠️ Failed';
            const errMsg = (d && d.error) ? d.error : `HTTP ${r.status}`;
            if (meta) meta.innerHTML = `<span style="color:#f87171;">📤 Upload failed: ${errMsg}</span>`;
        }
    } catch (e) {
        if (btn) btn.textContent = '⚠️ Failed';
        if (meta) meta.innerHTML = `<span style="color:#f87171;">📤 Upload error: ${e.message || e}</span>`;
    } finally {
        // Restore button label after 3 s so the operator can upload again.
        setTimeout(() => { if (btn) { btn.disabled = false; btn.textContent = origText || '📤 Upload to AI Trainer'; } }, 3000);
    }
}
window.uploadCurrentDefectToAiTrainer = uploadCurrentDefectToAiTrainer;


function setDefectView(kind) {
    if (!['ann', 'raw', 'both'].includes(kind)) return;
    _currentDefectView = kind;
    if (_currentDefectItem) {
        let urls = _imgUrlsFor({img: _currentDefectItem.image_path, ship: _currentDefectItem.shipment});
        if (_currentDefectItem.annotated_url) urls.ann = _currentDefectItem.annotated_url;
        if (_currentDefectItem.raw_url)       urls.raw = _currentDefectItem.raw_url;
        _renderDefectView(urls);
    }
    _highlightViewToggle();
}
window.setDefectView = setDefectView;

// Class-aggregate entry point (from bar / pie / pareto clicks): fetch the
// LATEST single frame for that class and show it. No more 24-image dump.
async function openDefectDrawer(className) {
    const win  = document.getElementById('insight-window')?.value   || '24h';
    const ship = document.getElementById('insight-shipment')?.value || '';
    const drawer = document.getElementById('defect-drawer');
    const body   = document.getElementById('defect-drawer-body');
    const empty  = document.getElementById('defect-drawer-empty');
    const title  = document.getElementById('defect-drawer-title');
    const cnt    = document.getElementById('defect-drawer-count');
    if (!drawer) return;
    title.textContent = 'Latest defect: ' + className;
    cnt.textContent = 'loading…';
    body.innerHTML = '';
    empty.style.display = 'none';
    drawer.style.display = 'flex';
    try {
        const r = await fetch('/api/recent_detections?cls=' + encodeURIComponent(className) +
                              '&window=' + encodeURIComponent(win) +
                              '&shipment=' + encodeURIComponent(ship) + '&limit=1');
        const d = await r.json();
        const it = (d.items || [])[0];
        if (!it) {
            cnt.textContent = '0 image(s)';
            empty.textContent = 'No stored frame found for ' + className + ' in this window.';
            empty.style.display = 'block';
            return;
        }
        openDefectDrawerForFrame({
            image_path: it.image_path,
            shipment: it.shipment,
            t: it.t,
            classes: it.classes,
            best_confidence: it.best_confidence,
            cls: className,
        });
    } catch (e) {
        empty.textContent = 'Error loading: ' + e;
        empty.style.display = 'block';
    }
}
function closeDefectDrawer() {
    const d = document.getElementById('defect-drawer');
    if (d) { d.style.display = 'none'; document.getElementById('defect-drawer-body').innerHTML = ''; }
    const p = document.getElementById('chart-image-preview');
    if (p) p.style.display = 'none';
}
// Click outside drawer or ESC to close.
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDefectDrawer(); });
window.openDefectDrawer = openDefectDrawer;
window.closeDefectDrawer = closeDefectDrawer;

// 3.21.3 — CSV export of all stored detections for the selected shipment+window.
// Honors the Charts tab's window + shipment selectors. Server streams the file
// (server-side cursor), filename is detections_<shipment>_<window>.csv.
function exportInsightCSV() {
    const win  = document.getElementById('insight-window')?.value   || '24h';
    const ship = document.getElementById('insight-shipment')?.value || '';
    // 3.21.10: honor the min_conf slider too so the CSV matches what the charts show.
    const minConf = (parseFloat(document.getElementById('insight-min-conf')?.value || '0') / 100) || 0;
    // 4.0.59: unwind checkbox → inverts the length column (max_encoder − encoder)
    // so unwind lines read "meters left to unwind" from full-roll = 0.
    const unwind = document.getElementById('insight-unwind')?.checked ? 'true' : 'false';
    const url = '/api/export_csv?window=' + encodeURIComponent(win) +
                '&shipment=' + encodeURIComponent(ship) +
                '&min_conf=' + minConf +
                '&unwind=' + unwind;
    // navigate via hidden anchor so the browser handles the download with the
    // Content-Disposition filename (instead of trying to render in a new tab)
    const a = document.createElement('a');
    a.href = url; a.download = '';
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
}
window.exportInsightCSV = exportInsightCSV;

// Auto-refresh when the Charts (grafana) tab becomes active. switchTab() lives in
// app-core.js; we hook the click rather than patch it to stay decoupled.
document.addEventListener('DOMContentLoaded', function () {
    const chartsTabBtn = document.querySelector('button[onclick="switchTab(\'grafana\')"]');
    if (chartsTabBtn) {
        chartsTabBtn.addEventListener('click', function () {
            // small delay so the tab is visible before Chart.js measures the canvas
            setTimeout(refreshDetectionInsights, 150);
            // 4.0.202 — load the Score-per-shipment card ONCE per session on the
            // first Charts-tab open. It's window-independent (last N shipments),
            // so it does NOT re-run on window/shipment changes — only here, its
            // own Refresh button, or a shipment-count change. `loadScorePerShipment`
            // guards against re-loading unless forced.
            setTimeout(function () { try { loadScorePerShipment(false); } catch (_) {} }, 200);
        });
    }
});

// 4.0.159 — Hide the "Color baseline" + "Phase" pill row in the chart toolbar
// when the global "Parent Object Color Check" toggle is off. app-core.js's
// loadColorConfig fires this event on page load AND whenever the operator
// saves the Advanced-tab checkbox. Refresh the chart on the flip so the
// heatmap layer appears / disappears without a manual reload.
function _mveApplyColorCheckVisibility(enabled) {
    const bar = document.getElementById('hm-color-toolbar');
    if (bar) bar.style.display = enabled ? 'inline-flex' : 'none';
    // 4.0.161 — also hide the parent-class picker (even if it would
    // otherwise be visible because of ≥ 2 discovered parent classes).
    const pc = document.getElementById('hm-parent-class-wrap');
    if (pc && !enabled) pc.style.display = 'none';
    // 4.0.162 — hide the colour-change strip when the feature is off.
    // When it comes back on, next refresh's _renderColorChangeStrip
    // reveals it if the payload carries cells.
    const ccs = document.getElementById('color-change-strip-wrap');
    if (ccs && !enabled) ccs.style.display = 'none';
    if (typeof refreshDetectionInsights === 'function') {
        try { refreshDetectionInsights(); } catch (_) {}
    }
}
document.addEventListener('mve:color-check-toggled', function (ev) {
    _mveApplyColorCheckVisibility(!!(ev && ev.detail && ev.detail.enabled));
});
// Cover the case where charts.js loads AFTER app-core.js already fired the
// event (window._mveColorCheckEnabled will be set).
document.addEventListener('DOMContentLoaded', function () {
    setTimeout(function () {
        if (typeof window._mveColorCheckEnabled === 'boolean') {
            _mveApplyColorCheckVisibility(window._mveColorCheckEnabled);
        }
    }, 1400);
});
