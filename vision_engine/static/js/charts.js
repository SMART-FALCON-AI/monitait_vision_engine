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
let _insightCameraScatterEncoder = null;  // 3.25.13: kept (unused) to avoid touching unrelated old refs.
let _insightAxis = 'encoder';   // 3.26.0 — default to encoder (was 'time'); operators on roll-based lines care about position more than wall-clock.
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
            normNoteEl.textContent = `Normalized by encoder span (${(data.encoder_span || 0).toLocaleString()} ${data.encoder_unit || 'units'})`;
        } else if (by === 'frame') {
            normNoteEl.textContent = `Normalized by frame count (${(data.frame_count || 0).toLocaleString()} frames) — no encoder data`;
        } else {
            normNoteEl.textContent = 'No length data available — score may not be meaningful';
        }
    }

    if (lengthEl) lengthEl.textContent  = (data.encoder_span ?? 0).toLocaleString() + ' ' + (data.encoder_unit || 'units');
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
    const VALID = ['camera', 'shipment_start', 'target', 'reference_frame'];
    if (!VALID.includes(mode)) return;

    if (mode === 'shipment_start') {
        const shipSel = document.getElementById('insight-shipment');
        if (shipSel && !shipSel.value) {
            // Try to auto-select the current shipment.
            fetch('/api/status').then(r => r.json()).then(s => {
                const cur = s && s.shipment;
                if (cur && cur !== 'no_shipment') {
                    shipSel.value = cur;
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
    const raw = parseInt(localStorage.getItem('mve_dot_cap') || '750', 10);
    return Math.max(100, Math.min(5000, Number.isFinite(raw) ? raw : 750));
}
function setDotCap(value) {
    let n = parseInt(value, 10);
    if (!Number.isFinite(n)) n = 750;
    n = Math.max(100, Math.min(5000, n));
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
}
window._renderPhaseButtons = _renderPhaseButtons;

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

    await _loadShownClasses();  // so per-class charts honor the Show toggle
    refreshShipmentQualityScore();  // 3.21.13 — Phase 2 preview: score card
    refreshShipmentQualityTrend();  // 3.21.16 — Phase 2: drift / trend chart
    refreshEjectionCharts();    // independent of detection data (Store gates separately)
    refreshProductionCharts();  // production_metrics KPIs (own data source)
    refreshQualityCharts();     // inference_results diagnostics (Pareto/heatmap/camera/latency)

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

async function refreshAdvancedCharts() {
    const winSel = document.getElementById('insight-window');
    const target = (winSel && winSel.value) || '24h';
    const shipment = document.getElementById('insight-shipment')?.value || '';
    const minConf = (parseFloat(document.getElementById('insight-min-conf')?.value || '0') / 100) || 0;

    _progressiveLadderTicket += 1;
    const myTicket = _progressiveLadderTicket;

    // Kick the OTHER Charts-tab panels off in parallel — they have their
    // own endpoints (quality/shipments, quality/heatmap, ejection_axis)
    // and their own render paths. Never await them.
    try { if (typeof refreshQualityCharts === 'function') refreshQualityCharts(); } catch (e) {}

    // Narrow-scope views are already fast enough — single fetch, no ladder.
    if (target === '1h' || !!shipment) {
        return _refreshAdvancedChartsCore();
    }

    // Read the operator's chosen bucket count (same input that drives the
    // Quality by Time / Ejection by Time strip resolution). Default 48
    // matches those strips; the max 192 the operator uses gives 192
    // aligned scatter-buckets over the target window.
    const bins = Math.max(4, Math.min(192,
        parseInt(localStorage.getItem('mve_bucket_count') || '48', 10) || 48));
    const targetMs = _windowToMs(target);
    const bucketMs = Math.max(60 * 1000, Math.floor(targetMs / bins));
    const nowMs = Date.now();

    // Mount the scatter-scoped loader — NOT the tab-wide mveLoaderBegin.
    // If /api/detection_charts hangs later, the rest of the tab stays clean.
    _mountScatterLoader('Loading dots — newest first');

    // Hydrate size + confidence bands + camera_y_order from a tiny fast
    // fetch. We ONLY use it for those inputs; the scatter data returned by
    // this call is IGNORED — the bucket loop owns the scatter.
    let camYOrder = [];
    try {
        const bins48 = parseInt(localStorage.getItem('mve_bucket_count') || '48', 10) || 48;
        const r = await fetch('/api/detection_charts?window=1h'
            + '&shipment=' + encodeURIComponent(shipment)
            + '&min_conf=' + minConf
            + '&baseline=' + encodeURIComponent(localStorage.getItem('mve_heatmap_baseline') || 'camera')
            + '&phase=' + encodeURIComponent(localStorage.getItem('mve_heatmap_phase') || '')
            + '&bins=' + bins48
            + '&dot_cap=' + _dotCap());
        if (myTicket !== _progressiveLadderTicket) return;
        const d = await r.json();
        camYOrder = Array.isArray(d.camera_y_order) ? d.camera_y_order : [];
        // Populate size + confidence charts only. Leaves the scatter alone.
        if (typeof _renderSizeConfidenceBandsOnly === 'function') {
            try { _renderSizeConfidenceBandsOnly(d); } catch (e) { console.warn(e); }
        } else {
            // Fallback: reuse the existing Core render just to get the size
            // and confidence charts up on first tab activation. It also
            // populates the scatter with 1h data, which the bucket loop
            // dedupes against below.
            try { await _refreshAdvancedChartsCore(); } catch (e) { console.warn(e); }
        }
    } catch (e) {
        console.warn('[progressive] hydrate failed:', e);
    }

    // Seed the dedup set with whatever's already on the chart (e.g. from
    // the 1h Core hydrate above, or from a previous poll cycle).
    _progressiveScatterSeen = new Set();
    if (_insightCameraScatter) {
        for (const ds of (_insightCameraScatter.data.datasets || [])) {
            for (const p of (ds.data || [])) {
                _progressiveScatterSeen.add(_scatterPointKey(p, ds.label));
            }
        }
    }

    // Iterate ONE Quality-strip-aligned bucket at a time, newest first.
    // Each fetch covers exactly `bucketMs` — the same width one
    // Quality/Ejection strip cell covers underneath. Operator sees dots
    // appear ONE strip-cell-width-per-fetch.
    let endMs = nowMs;
    const stopMs = nowMs - targetMs;
    let bucketCount = 0;
    try {
        while (endMs > stopMs && bucketCount < bins) {
            if (myTicket !== _progressiveLadderTicket) return;
            if (!_insightCameraScatter) break;
            if ((winSel.value || '24h') !== target) break; // operator changed window

            const sinceMs = Math.max(stopMs, endMs - bucketMs);
            await _extendScatterFromBucket(sinceMs, endMs, shipment, minConf);
            endMs = sinceMs;
            bucketCount += 1;
        }
    } finally {
        if (myTicket === _progressiveLadderTicket) _unmountScatterLoader();
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
function _animateScatterAppearance() {
    if (!_insightCameraScatter) return;
    const chart = _insightCameraScatter;
    const dsCount = (chart.data.datasets || []).length;
    if (dsCount === 0) return;

    // Snapshot every point together with its dataset index. Skip animation
    // entirely for tiny scatters — the pop-in effect would be noise for a
    // dozen dots.
    const allPoints = [];
    for (let i = 0; i < dsCount; i++) {
        const ds = chart.data.datasets[i];
        for (const p of (ds.data || [])) {
            allPoints.push({ dsIdx: i, point: p });
        }
    }
    if (allPoints.length < 30) return;

    // Sort newest→oldest. Time-axis x is epoch ms; encoder-axis x is the
    // raw encoder value — both increase over time, so higher x = newer.
    allPoints.sort((a, b) => (b.point.x || 0) - (a.point.x || 0));

    // Clear every dataset's data — chart stays alive, axes/legend intact.
    for (let i = 0; i < dsCount; i++) {
        chart.data.datasets[i].data = [];
    }
    chart.update('none');

    // Progressive fill: ~15 chunks over ~3 seconds. Cheap: no DB, no fetch.
    const CHUNKS = 15;
    const TOTAL_MS = 3000;
    const chunkSize = Math.max(3, Math.ceil(allPoints.length / CHUNKS));
    const delayMs = Math.floor(TOTAL_MS / CHUNKS);

    // Each animation is tagged with a monotonic ticket. If refreshAdvancedCharts
    // fires again before this animation finishes (operator changed window /
    // shipment / min_conf), the ticket bumps and the older loop bails out
    // instead of interleaving with the newer render's dots.
    _scatterAnimTicket += 1;
    const myTicket = _scatterAnimTicket;

    let idx = 0;
    function _pump() {
        if (myTicket !== _scatterAnimTicket) return;   // superseded
        if (!_insightCameraScatter) return;            // chart destroyed
        const chart = _insightCameraScatter;
        const end = Math.min(idx + chunkSize, allPoints.length);
        for (let i = idx; i < end; i++) {
            const { dsIdx, point } = allPoints[i];
            const ds = chart.data.datasets[dsIdx];
            if (ds && ds.data) ds.data.push(point);
        }
        chart.update('none');
        idx = end;
        if (idx < allPoints.length) {
            setTimeout(_pump, delayMs);
        }
    }
    setTimeout(_pump, 60);  // brief empty flash so the reveal is visible
}

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
function _progressiveLadder(target) {
    // Rungs are ordered smallest→largest and are POWERS of the 30-min
    // dashboard bucket size (48 buckets over 24 h → 30 min each).
    const rungs = ['2h', '4h', '6h', '12h', '24h', '48h', '7d', '30d'];
    const idx = rungs.indexOf(target);
    if (idx < 0) return ['6h', target]; // custom window — do at least one intermediate step
    return rungs.slice(0, idx + 1);
}

// v4.0.74 — silently fetch one time-bucket via `since_ms`/`until_ms` and
// append only NEW points to the existing `_insightCameraScatter`. Server
// short-circuits (`scatter_only=1`) so it skips the heavy size / confidence /
// heatmap / ejection queries — only the two stratified scatter queries
// run against the bucket's narrow slice. No loader spinner. No touch of
// the other charts. Uses the exact same downstream extend logic so the
// dedup + dataset-append behaviour is identical to the window-based path.
async function _extendScatterFromBucket(sinceMs, untilMs, shipment, minConf) {
    if (!_insightCameraScatter) return;
    let data;
    try {
        const r = await fetch('/api/detection_charts?since_ms=' + Math.floor(sinceMs) +
                              '&until_ms=' + Math.floor(untilMs) +
                              '&shipment=' + encodeURIComponent(shipment) +
                              '&min_conf=' + minConf +
                              '&scatter_only=1' +
                              '&dot_cap=' + _dotCap());
        data = await r.json();
    } catch (e) {
        console.warn('[progressive] bucket fetch failed [' + new Date(sinceMs).toISOString() + ' → ' + new Date(untilMs).toISOString() + ']:', e);
        return;
    }

    const axisMode = (_insightAxis === 'encoder') ? 'encoder' : 'time';
    const points = axisMode === 'encoder'
        ? (data.camera_scatter_encoder || [])
        : (data.camera_scatter || []);
    if (!Array.isArray(points) || points.length === 0) return;

    // Reuse the same y-axis mapping the Core renderer built for '1h' — camera
    // display order came from data.camera_y_order there. If a wider window
    // surfaces a camera that wasn't in the '1h' data, fall back to plotting
    // by raw cam id (Chart.js scales handle it, just no reordered position).
    const camOrder = Array.isArray(data.camera_y_order) ? data.camera_y_order : [];
    const idxByCam = new Map(camOrder.map(function (cid, i) { return [cid, i]; }));
    const _camToIdx = function (cid) { return idxByCam.has(cid) ? idxByCam.get(cid) : cid; };

    const dsByLabel = new Map();
    (_insightCameraScatter.data.datasets || []).forEach(function (ds) {
        dsByLabel.set(ds.label, ds);
    });

    let addedCount = 0;
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
            ship: p.ship
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
                borderColor:     (typeof _classColor === 'function' ? _classColor(cls) : '#888')
            };
            _insightCameraScatter.data.datasets.push(ds);
            dsByLabel.set(cls, ds);
        }
        ds.data.push(built);
        addedCount++;
    }
    if (addedCount > 0) {
        _insightCameraScatter.update('none'); // no animation — feels instant
        console.log('[progressive] window=' + win + ': added ' + addedCount + ' new points');
    }
}

async function _refreshAdvancedChartsCore() {
    const window = document.getElementById('insight-window')?.value || '24h';
    const shipment = document.getElementById('insight-shipment')?.value || '';
    const minConf = (parseFloat(document.getElementById('insight-min-conf')?.value || '0') / 100) || 0;
    let data;
    // 4.0.53 — gate the loader to the Charts tab so the badge can't leak
    // into other tabs. The paired mveLoaderEnd calls below check the same flag.
    const _showedLoader = _chartsTabActive();
    if (_showedLoader) mveLoaderBegin('Loading charts…');
    try {
        // 4.0.30 — tell the server which heatmap baseline mode is active so
        // it only computes that one (saves SQL for unused modes). Falls back
        // to `camera` if nothing's been stored yet.
        const baseline = localStorage.getItem('mve_heatmap_baseline') || 'camera';
        // 4.0.34 — phase filter is heatmap-only. Empty string = all phases.
        const phase = localStorage.getItem('mve_heatmap_phase') || '';
        // 4.0.35 — single bucket count drives heatmap + quality + ejection strips.
        const bins = parseInt(localStorage.getItem('mve_bucket_count') || '48', 10) || 48;
        const r = await fetch('/api/detection_charts?window=' + encodeURIComponent(window) +
                              '&shipment=' + encodeURIComponent(shipment) +
                              '&min_conf=' + minConf +
                              '&baseline=' + encodeURIComponent(baseline) +
                              '&phase=' + encodeURIComponent(phase) +
                              '&bins=' + bins +
                              '&dot_cap=' + _dotCap());
        data = await r.json();
    } catch (e) {
        console.error('detection_charts fetch failed:', e);
        if (_showedLoader) mveLoaderEnd();
        return;
    }

    // Populate shipment dropdown once (preserve current selection)
    const shipSel = document.getElementById('insight-shipment');
    if (shipSel && Array.isArray(data.shipments)) {
        const cur = shipSel.value;
        const opts = ['<option value="">All shipments</option>']
            .concat(data.shipments.map(s => `<option value="${s}">${s}</option>`));
        shipSel.innerHTML = opts.join('');
        shipSel.value = cur; // keep selection if still present
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
    const _scatterXMin = _nowMs - _winMs;
    const _scatterXMax = _nowMs;

    // ---- (3) Camera × {time|encoder} scatter — single merged renderer (3.25.13).
    // Builds the bubble chart for the currently selected `_insightAxis`; toggling
    // via setInsightAxis() destroys + rebuilds with the other axis's data.
    const scCtx = document.getElementById('insight-camera-scatter');
    if (scCtx) {
        if (_insightCameraScatter) _insightCameraScatter.destroy();
        const axisMode = (_insightAxis === 'encoder') ? 'encoder' : 'time';
        const points = axisMode === 'encoder'
            ? (data.camera_scatter_encoder || [])
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
        const camOrder = Array.isArray(data.camera_y_order) ? data.camera_y_order : [];
        const idxByCam = new Map(camOrder.map((cid, i) => [cid, i]));
        const camByIdx = new Map(camOrder.map((cid, i) => [i, cid]));
        const _camToIdx = (cid) => idxByCam.has(cid) ? idxByCam.get(cid) : cid;
        const byClass = {};
        points.forEach(p => {
            if (!_isShown(p.cls)) return;
            (byClass[p.cls] = byClass[p.cls] || []).push({
                x: p.x, y: _camToIdx(p.y), cam_id: p.y,
                r: 3 + (p.r || 0) * 9, conf: p.r, cls: p.cls, img: p.img, ship: p.ship
            });
        });
        const datasets = Object.keys(byClass).map(cls => ({
            label: cls, data: byClass[cls],
            backgroundColor: _classColor(cls) + 'cc', borderColor: _classColor(cls)
        }));
        // 4.0.44 — title + legend rendered as HTML siblings ABOVE the canvas
        // so the colour heatmap (which paints inside chartArea) can never
        // reach them. Clicking a legend swatch still toggles the matching
        // dataset via Chart.js's setDatasetVisibility.
        try {
            const _titleEl = document.getElementById('insight-scatter-title');
            if (_titleEl) {
                _titleEl.textContent = (axisMode === 'time')
                    ? 'Camera × time — hover for image, click dot to open the exact frame'
                    : 'Camera × encoder (roll position) — hover for image, click dot for the exact frame';
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
        const _ALL_BASELINES = ['camera', 'shipment_start', 'target', 'reference_frame'];
        const _availableBaselineModes = new Set(data.color_baseline_modes || ['camera']);
        const _activeBaselineMode = data.color_baseline_mode || 'camera';
        const _activeBaselineMap  = data.color_baseline || {};
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

        // 4.0.29c — pick the heatmap variant that matches the current axis.
        // Backend returns BOTH binned-by-encoder and binned-by-time forms so
        // the background paints in both modes. Time mode is what shows up
        // when the line is stopped (encoder collapsed to a single value)
        // but the operator still wants to see colour drift over time.
        const _hmRaw = (axisMode === 'encoder')
            ? (data.color_heatmap || {})
            : (data.color_heatmap_time || {});
        const heatmapCells = Array.isArray(_hmRaw.cells) ? _hmRaw.cells : [];
        const heatmapMin = (axisMode === 'encoder')
            ? (_hmRaw.enc_min != null ? Number(_hmRaw.enc_min) : null)
            : (_hmRaw.t_min   != null ? Number(_hmRaw.t_min)   : null);
        const heatmapMax = (axisMode === 'encoder')
            ? (_hmRaw.enc_max != null ? Number(_hmRaw.enc_max) : null)
            : (_hmRaw.t_max   != null ? Number(_hmRaw.t_max)   : null);
        // Alias kept so the rest of the plugin body reads cleanly. These are
        // X-axis values in whichever unit the active axis uses (encoder counts
        // OR epoch-millis), so the binning math is identical.
        const heatmapEncMin = heatmapMin, heatmapEncMax = heatmapMax;
        const heatmapBins = Math.max(1, Number(_hmRaw.n_bins) || 32);
        const heatmapBoundsCollapsed = (heatmapEncMin != null && heatmapEncMax != null &&
                                        heatmapEncMax === heatmapEncMin);
        const heatmapActive = (
            heatmapCells.length > 0 &&
            heatmapEncMin != null && heatmapEncMax != null
        );
        const _heatmapPlugins = heatmapActive ? [{
            id: 'colorHeatmap',
            beforeDatasetsDraw(chart) {
                const ctx = chart.ctx;
                const xScale = chart.scales && chart.scales.x;
                const yScale = chart.scales && chart.scales.y;
                if (!ctx || !xScale || !yScale) return;
                const ca = chart.chartArea;
                const binWidth = (heatmapEncMax - heatmapEncMin) / heatmapBins;
                ctx.save();
                for (const cell of heatmapCells) {
                    const idx = idxByCam.has(cell.cam) ? idxByCam.get(cell.cam) : null;
                    if (idx == null) continue;
                    let x, y, w, h;
                    if (heatmapBoundsCollapsed || binWidth <= 0) {
                        // Single-bin fallback: paint the whole chart-area row
                        // for this camera. Useful when line is stopped.
                        x = ca ? ca.left : 0;
                        w = ca ? (ca.right - ca.left) : chart.width;
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
                    const base = _activeBaselineMap[String(cell.cam)];
                    const cellE = Number(cell.E) || 0;
                    const de = (base && Number.isFinite(base.E))
                        ? Math.abs(cellE - Number(base.E))
                        : Math.abs(Number(cell.delta_e) || 0);
                    let hue;
                    if (de <= 2)       hue = 120;          // green
                    else if (de <= 5)  hue = 60;           // yellow
                    else if (de <= 10) hue = 30;           // orange
                    else               hue = 0;            // red
                    ctx.fillStyle = `hsla(${hue}, 65%, 45%, 0.35)`;
                    ctx.fillRect(x, y, w, h);
                    // 4.0.38 — thin dark stroke so each cell is visually
                    // distinct from its neighbours. Without this, adjacent
                    // cells with similar dE blend into wide bands and the
                    // operator can't see that the bucket count matches the
                    // quality / ejection strips below.
                    ctx.strokeStyle = 'rgba(15, 23, 42, 0.55)';
                    ctx.lineWidth = 0.5;
                    ctx.strokeRect(x + 0.25, y + 0.25, w - 0.5, h - 0.5);
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
        const xScaleCfg = axisMode === 'time'
            ? { type: 'linear', min: _scatterXMin, max: _scatterXMax,
                ticks: { color: '#94a3b8', font: { size: 9 }, callback: (v) => new Date(v).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}) },
                grid: { color: 'rgba(148,163,184,0.08)' } }
            : { type: 'linear',
                ticks: { color: '#94a3b8', font: { size: 9 } },
                grid: { color: 'rgba(148,163,184,0.08)' },
                title: { display: true, text: 'Encoder (roll position)', color: '#94a3b8', font: { size: 10 } } };
        const titleText = axisMode === 'time'
            ? 'Camera × time — hover for image, click dot to open the exact frame'
            : 'Camera × encoder (roll position) — hover for image, click dot for the exact frame';
        _insightCameraScatter = new Chart(scCtx, {
            type: 'bubble',
            data: { datasets },
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
                animation: { onComplete: function() { _alignStripToScatter(this, 'quality-strip-wrap'); } },
                onResize: function() { setTimeout(() => _alignStripToScatter(this, 'quality-strip-wrap'), 0); }
            }
        });
        // 4.0.7 — direct mousemove hover-preview, independent of Chart.js's
        // external tooltip plugin (which wasn't firing reliably).
        _attachHoverPreview(scCtx, () => _insightCameraScatter);
    }
    // 4.0.42 — hide the loading badge now that the scatter + heatmap have
    // been built. Paired with mveLoaderBegin() at the top of this function.
    if (_showedLoader) mveLoaderEnd();
}

// 3.25.13 — toggle the merged scatter's X-axis between time and encoder.
// Re-renders the scatter (destroy + rebuild), reloads the quality + ejection
// strips for the new axis, updates the strip titles + button styling.
function setInsightAxis(axis) {
    if (axis !== 'time' && axis !== 'encoder') return;
    if (_insightAxis === axis) return;
    _insightAxis = axis;
    // Active-button styling.
    const tBtn = document.getElementById('insight-axis-toggle-time');
    const eBtn = document.getElementById('insight-axis-toggle-encoder');
    if (tBtn) {
        tBtn.style.background = axis === 'time' ? 'linear-gradient(135deg,#10b981,#047857)' : 'rgba(51,65,85,0.5)';
        tBtn.style.color = axis === 'time' ? '#fff' : '#cbd5e1';
    }
    if (eBtn) {
        eBtn.style.background = axis === 'encoder' ? 'linear-gradient(135deg,#10b981,#047857)' : 'rgba(51,65,85,0.5)';
        eBtn.style.color = axis === 'encoder' ? '#fff' : '#cbd5e1';
    }
    // Strip titles.
    const qTitle = document.getElementById('quality-strip-title');
    if (qTitle) qTitle.textContent = axis === 'encoder'
        ? '📏 quality by encoder — find spots on the material with the most issues'
        : '📍 quality by time — hover a cell for top defect + score';
    const eTitle = document.getElementById('ejection-strip-title');
    if (eTitle) eTitle.textContent = axis === 'encoder'
        ? '⏏️ ejections by encoder — color = procedure; hover for breakdown'
        : '⏏️ ejections by time — color = procedure; hover for breakdown';
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
// Defect Diagnostics: Pareto (Show-filtered), defects-by-camera, defect-location
// heatmap (bbox-center density), inference latency. Reads /api/quality_charts.
// ---------------------------------------------------------------------------
async function refreshQualityCharts() {
    const window = document.getElementById('insight-window')?.value || '24h';
    const shipment = document.getElementById('insight-shipment')?.value || '';
    const minConf = (parseFloat(document.getElementById('insight-min-conf')?.value || '0') / 100) || 0;
    let data;
    try {
        const r = await fetch('/api/quality_charts?window=' + encodeURIComponent(window) +
                              '&shipment=' + encodeURIComponent(shipment) +
                              '&min_conf=' + minConf);
        data = await r.json();
    } catch (e) { console.error('quality_charts fetch failed:', e); return; }

    const byClass = data.by_class || {};
    const byCamera = data.by_camera || {};
    const hm = data.heatmap || { gw: 32, gh: 20, max: 0, cells: [] };
    const latency = data.latency_over_time || [];
    const hasData = Object.keys(byClass).length > 0;

    const emptyEl = document.getElementById('quality-empty');
    const chartsEl = document.getElementById('quality-charts');
    const totalEl = document.getElementById('quality-total');
    if (emptyEl) emptyEl.style.display = hasData ? 'none' : 'block';
    if (chartsEl) chartsEl.style.display = hasData ? 'grid' : 'none';
    if (totalEl) totalEl.textContent = hasData ? `${hm.cells.length} active grid cells • ${Object.keys(byCamera).length} camera(s)` : '';
    if (!hasData) return;

    // ---- Pareto of defects (Show-filtered): bars desc + cumulative % line ----
    const paretoCtx = document.getElementById('quality-pareto-chart');
    if (paretoCtx) {
        if (_qualPareto) _qualPareto.destroy();
        const names = Object.keys(byClass).filter(_isShown).sort((a, b) => byClass[b] - byClass[a]);
        const counts = names.map(n => byClass[n]);
        const grand = counts.reduce((a, b) => a + b, 0) || 1;
        let run = 0;
        const cum = counts.map(c => { run += c; return Math.round(run / grand * 1000) / 10; });
        _qualPareto = new Chart(paretoCtx, {
            data: {
                labels: names,
                datasets: [
                    { type: 'bar', label: 'Count', data: counts, backgroundColor: names.map(n => _classColor(n)), borderRadius: 3, yAxisID: 'y' },
                    { type: 'line', label: 'Cumulative %', data: cum, borderColor: '#f59e0b', backgroundColor: 'transparent', yAxisID: 'y1', tension: 0, pointRadius: 2, borderWidth: 2 }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, labels: { color: '#cbd5e1', font: { size: 10 }, boxWidth: 12 } },
                    title: { display: true, text: 'Pareto of defects — click a class for images', color: '#cbd5e1', font: { size: 13 } }
                },
                onClick: (evt, els, chart) => {
                    if (els && els.length) {
                        const lab = chart.data.labels[els[0].index];
                        if (lab) openDefectDrawer(lab);
                    }
                },
                scales: {
                    x: { ticks: { color: '#94a3b8', font: { size: 9 } }, grid: { display: false } },
                    y: { beginAtZero: true, ticks: { color: '#94a3b8', precision: 0 }, grid: { color: 'rgba(148,163,184,0.1)' } },
                    y1: { position: 'right', beginAtZero: true, max: 100, ticks: { color: '#f59e0b' }, grid: { display: false }, title: { display: true, text: '%', color: '#f59e0b' } }
                }
            }
        });
    }

    // ---- Defects by camera ----
    const camCtx = document.getElementById('quality-camera-chart');
    if (camCtx) {
        if (_qualCamera) _qualCamera.destroy();
        const cams = Object.keys(byCamera).sort((a, b) => (parseInt(a) || 0) - (parseInt(b) || 0));
        _qualCamera = new Chart(camCtx, {
            type: 'bar',
            data: { labels: cams.map(c => 'Cam ' + c), datasets: [{ label: 'Detections', data: cams.map(c => byCamera[c]), backgroundColor: 'rgba(139,92,246,0.6)', borderRadius: 3 }] },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, title: { display: true, text: 'Defects by camera / station', color: '#cbd5e1', font: { size: 13 } } },
                scales: {
                    x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { display: false } },
                    y: { beginAtZero: true, ticks: { color: '#94a3b8', precision: 0 }, grid: { color: 'rgba(148,163,184,0.1)' } }
                }
            }
        });
    }

    // ---- Defect location heatmap (bbox-center density) ----
    const heatCtx = document.getElementById('quality-heatmap-chart');
    if (heatCtx) {
        if (_qualHeatmap) _qualHeatmap.destroy();
        const maxc = hm.max || 1;
        const pts = (hm.cells || []).map(c => ({ x: c.x + 0.5, y: c.y + 0.5, r: 5 + (c.c / maxc) * 16, _c: c.c }));
        const bg = (hm.cells || []).map(c => `rgba(239,68,68,${(0.15 + 0.85 * (c.c / maxc)).toFixed(3)})`);
        _qualHeatmap = new Chart(heatCtx, {
            type: 'bubble',
            data: { datasets: [{ label: 'Defect density', data: pts, backgroundColor: bg, borderColor: 'rgba(239,68,68,0.3)' }] },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    title: { display: true, text: 'Defect location heatmap (bbox centers, frame layout)', color: '#cbd5e1', font: { size: 13 } },
                    tooltip: { callbacks: { label: (c) => `${c.raw._c} defect(s) here` } }
                },
                scales: {
                    x: { type: 'linear', min: 0, max: hm.gw, ticks: { display: false }, grid: { color: 'rgba(148,163,184,0.08)' }, title: { display: true, text: '← frame width →', color: '#94a3b8', font: { size: 10 } } },
                    y: { type: 'linear', min: 0, max: hm.gh, reverse: true, ticks: { display: false }, grid: { color: 'rgba(148,163,184,0.08)' }, title: { display: true, text: '↑ frame height ↓', color: '#94a3b8', font: { size: 10 } } }
                }
            }
        });
    }

    // ---- Inference latency over time (avg / max ms) ----
    const latCtx = document.getElementById('quality-latency-chart');
    if (latCtx) {
        if (_qualLatency) _qualLatency.destroy();
        _qualLatency = new Chart(latCtx, {
            type: 'line',
            data: {
                labels: latency.map(p => p.t),
                datasets: [
                    { label: 'Max ms', data: latency.map(p => p.max), borderColor: 'rgba(239,68,68,0.5)', backgroundColor: 'transparent', tension: 0.3, pointRadius: 0, borderWidth: 1 },
                    { label: 'Avg ms', data: latency.map(p => p.avg), borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.12)', fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, labels: { color: '#cbd5e1', font: { size: 10 }, boxWidth: 12 } },
                    title: { display: true, text: 'Inference latency over time (ms)', color: '#cbd5e1', font: { size: 13 } }
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
function _imgUrlFromPath(p) {
    if (!p) return null;
    return '/api/raw_image/' + encodeURI(String(p).replace(/^raw_images\//, ''));
}
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
        const trend  = document.getElementById('sqs-trend-row');
        const sScore = document.getElementById('quality-charts-panel');
        if (trend && sScore && trend.parentNode) {
            // insertBefore(node, sibling) — if sibling is null it appends; nextSibling
            // is the element right after trend, which is what we want.
            trend.parentNode.insertBefore(sScore, trend.nextSibling);
        }
    } catch (e) { /* never block page load on reorder errors */ }
}
document.addEventListener('DOMContentLoaded', _reorderChartsTab);

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
    const metaTxt = `${cls} • cam ${pt.y != null ? pt.y : '?'} • conf ${conf}%`;
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
                if (cap) cap.textContent = `${metaTxt}   (click to edit in ai-trainer)`;
            };
            img.onload  = settle;
            img.onerror = function() {
                if (urls.raw && img.src !== urls.raw) {
                    // Fall back to raw and let onload settle.
                    img.src = urls.raw;
                } else {
                    img.style.opacity = '0.3';
                    if (cap) cap.textContent = '✗ image unavailable  ' + metaTxt;
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
            img.src = target;
            img.onerror = function() {
                if (urls.raw && img.src !== urls.raw) {
                    img.src = urls.raw;
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
function _onChartClickOpenDrawer(chart) {
    return function (evt, els) {
        if (!els || !els.length) return;
        const idx = els[0].index;
        const lab = chart.data.labels && chart.data.labels[idx];
        if (lab) openDefectDrawer(lab);
    };
}

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

function _scoreColor(score) {
    if (score == null || isNaN(score)) return 'rgba(100,116,139,0.5)';
    if (score >= 85) return '#22c55e';
    if (score >= 60) return '#facc15';
    return '#ef4444';
}

function _verdictColor(v) {
    if (v === 'RELEASE')    return 'rgba(34,197,94,0.7)';
    if (v === 'RE-INSPECT') return 'rgba(234,179,8,0.7)';
    if (v === 'HOLD')       return 'rgba(239,68,68,0.7)';
    return 'rgba(100,116,139,0.5)';
}

async function refreshQualityCharts() {
    // 3.25.5/13 — strips share the merged scatter's X-axis. Follow Detection
    // Insights' window selector and the current _insightAxis (time vs encoder)
    // so quality + ejection strips both align to whatever the scatter shows.
    const insightWin = document.getElementById('insight-window');
    const qualityWin = document.getElementById('quality-charts-window');
    const win = (insightWin && insightWin.value) || (qualityWin && qualityWin.value) || '24h';
    const axis = _insightAxis || 'time';
    await Promise.all([
        _loadQualityShipmentsChart(),
        _loadQualityHeatmap(axis, win),
        _loadEjectionAxis(axis, win),
    ]);
}
window.refreshQualityCharts = refreshQualityCharts;

async function _loadQualityShipmentsChart() {
    const canvas = document.getElementById('quality-shipments-chart');
    if (!canvas) return;
    try {
        const r = await fetch('/api/quality/shipments?n=30&window=30d');
        const d = await r.json();
        const rows = (d.shipments || []).slice().reverse();  // oldest left → newest right
        const labels = rows.map(s => s.shipment);

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
        // Rescale math: shift-and-scale by the OBSERVED spread of absolute
        // scores in the chart. When every shipment has the same absolute
        // score (spread == 0), we short-circuit to 100 across the board (no
        // meaningful differences to highlight — flat is the correct render).
        const absVals   = rows.map(s => (s.score_absolute != null ? s.score_absolute
                                       : (s.score != null ? s.score : 0)));
        const validAbs  = absVals.filter(v => v != null && !isNaN(v));
        const absMin    = validAbs.length ? Math.min(...validAbs) : 0;
        const absMax    = validAbs.length ? Math.max(...validAbs) : 100;
        const absSpread = absMax - absMin;
        const relData = absVals.map(v => {
            if (v == null || isNaN(v)) return 0;
            if (absSpread < 1e-9) return 100;  // all identical → flat max
            return Number((((v - absMin) / absSpread) * 100).toFixed(1));
        });

        const colors   = rows.map(s => _verdictColor(s.verdict));
        const relColor = 'rgba(59,130,246,0.55)';  // steel blue for relative bar

        if (_qualityShipmentsChart) _qualityShipmentsChart.destroy();
        _qualityShipmentsChart = new Chart(canvas, {
            type: 'bar',
            data: {
                labels,
                datasets: [
                    { label: 'Absolute', data: absVals,
                      backgroundColor: colors,
                      borderColor: colors.map(c => String(c).replace('0.7', '1')),
                      borderWidth: 1, order: 1 },
                    { label: 'Relative (window)', data: relData,
                      backgroundColor: relColor,
                      borderColor: 'rgba(59,130,246,1)', borderWidth: 1, order: 1 },
                ],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: { color: '#cbd5e1', font: { size: 10 }, boxWidth: 12 },
                    },
                    tooltip: {
                        callbacks: {
                            title: (items) => items[0].label,
                            label:  (item) => {
                                const r = rows[item.dataIndex];
                                const tops = (r.top_defects || []).join(', ');
                                const abs = r.score_absolute ?? r.score;
                                return [
                                    `Absolute: ${abs != null ? Number(abs).toFixed(1) : '—'}`,
                                    `Relative (this window): ${relData[item.dataIndex]}`,
                                    `Verdict: ${r.verdict || '—'}`,
                                    `Detections: ${r.rows || 0}`,
                                    tops ? `Top defects: ${tops}` : '',
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

async function _loadEjectionAxis(axis, win) {
    // 3.25.13 — unified IDs (single strip + single legend regardless of axis).
    const stripId  = 'ejection-strip';
    const legendId = 'ejection-strip-legend';
    const strip  = document.getElementById(stripId);
    const legend = document.getElementById(legendId);
    if (!strip) return;
    try {
        // 4.0.35 — bucket count synced with colour heatmap.
        const _bk = parseInt(localStorage.getItem('mve_bucket_count') || '48', 10) || 48;
        const r = await fetch(`/api/quality/ejection_axis?axis=${axis}&window=${encodeURIComponent(win)}&buckets=${_bk}`);
        const d = await r.json();
        const cells = d.buckets || [];
        if (!cells.length) {
            const hint = d.note || 'no ejection events in window';
            strip.innerHTML = `<div style="color:var(--text-secondary); font-size:10px; padding:2px; flex:1; text-align:center; font-style:italic;">${hint}</div>`;
            if (legend) legend.innerHTML = '';
            return;
        }
        // 3.25.13 — local-TZ label formatting on the time axis (matches scatter).
        const _fmtEjCellLabel = (c) => {
            if (!c) return '';
            if (axis === 'time' && c.ts) {
                const d = new Date(c.ts);
                const win24h = (win === '24h' || win === '1h' || win === '6h');
                return win24h
                    ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                    : d.toLocaleDateString([], { month: '2-digit', day: '2-digit' }) + ' ' +
                      d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            }
            return c.label || '';
        };
        // Cells: colored by top_procedure; empty buckets stay transparent.
        strip.innerHTML = cells.map(c => {
            const color = c.top_procedure ? _procColor(c.top_procedure) : 'transparent';
            const parts = Object.entries(c.by_procedure || {})
                .sort((a, b) => b[1] - a[1])
                .map(([n, k]) => `${n}: ${k}`).join('\n');
            const tip = `${_fmtEjCellLabel(c)}\nTotal ejections: ${c.n}\n${parts}`.trim();
            return `<div style="flex:1; background:${color}; cursor:default;" title="${tip.replace(/"/g, '&quot;')}"></div>`;
        }).join('');
        // Legend: chip per procedure that fired in the window (deterministic colors).
        if (legend) {
            const totals = {};
            cells.forEach(c => {
                Object.entries(c.by_procedure || {}).forEach(([n, k]) => { totals[n] = (totals[n] || 0) + k; });
            });
            legend.innerHTML = Object.entries(totals)
                .sort((a, b) => b[1] - a[1])
                .map(([n, k]) => `<span style="display:inline-flex; align-items:center; gap:3px;"><span style="display:inline-block; width:10px; height:10px; background:${_procColor(n)}; border-radius:2px;"></span>${n} (${k})</span>`)
                .join('');
        }
    } catch (e) { console.warn(`ejection_axis ${axis} failed`, e); }
}


async function _loadQualityHeatmap(axis, win) {
    // 3.25.13 — unified IDs (single strip regardless of axis).
    const stripId = 'quality-strip';
    const axisId  = 'quality-strip-axis';
    const strip   = document.getElementById(stripId);
    const axisEl  = document.getElementById(axisId);
    if (!strip) return;
    try {
        // 4.0.35 — bucket count synced with colour heatmap.
        const _bk = parseInt(localStorage.getItem('mve_bucket_count') || '48', 10) || 48;
        const r = await fetch(`/api/quality/heatmap?axis=${axis}&window=${encodeURIComponent(win)}&buckets=${_bk}`);
        const d = await r.json();
        const cells = d.buckets || [];
        if (!cells.length) {
            const hint = d.note || 'no data in window';
            strip.innerHTML = `<div style="color:var(--text-secondary); font-size:11px; padding:6px; flex:1; text-align:center; font-style:italic;">${hint}</div>`;
            if (axisEl) axisEl.innerHTML = '';
            return;
        }
        // 3.25.13 — for the time axis, format labels in the OPERATOR'S local TZ
        // (the scatter does the same via toLocaleTimeString). Backend already
        // sends `ts` as ISO, so we just re-format here. Encoder axis labels are
        // numeric position ranges and don't need this.
        const _fmtCellLabel = (c) => {
            if (!c) return '';
            if (axis === 'time' && c.ts) {
                const d = new Date(c.ts);
                const win24h = (win === '24h' || win === '1h' || win === '6h');
                return win24h
                    ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                    : d.toLocaleDateString([], { month: '2-digit', day: '2-digit' }) + ' ' +
                      d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            }
            return c.label || '';
        };
        strip.innerHTML = cells.map(c => {
            const color = _scoreColor(c.score);
            const top = c.top_class ? `Top: ${c.top_class}` : '';
            const tip = `${_fmtCellLabel(c)}\nScore: ${c.score}\nDetections: ${c.n}\n${top}`.trim();
            return `<div style="flex:1; background:${color}; cursor:default;" title="${tip.replace(/"/g, '&quot;')}"></div>`;
        }).join('');
        if (axisEl) {
            const first = _fmtCellLabel(cells[0]);
            const mid   = _fmtCellLabel(cells[Math.floor(cells.length / 2)]);
            const last  = _fmtCellLabel(cells[cells.length - 1]);
            axisEl.innerHTML = `<span>${first}</span><span>${mid}</span><span>${last}</span>`;
        }
    } catch (e) { console.warn(`heatmap ${axis} failed`, e); }
}

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
        });
    }
});
