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
let _insightCameraScatterEncoder = null;
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

// Stable color per class name (hash → palette index) so the same class keeps
// its color across charts and refreshes.
function _classColor(name) {
    let h = 0;
    for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
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

    const scoreEl   = document.getElementById('sqs-score');
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

    if (scoreEl) scoreEl.textContent = score.toFixed(1);

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
// Builds the URL with the same window + shipment the card is showing, opens
// the endpoint via a hidden anchor so the browser handles Content-Disposition
// (works in IFrames, doesn't trigger pop-up blockers).
async function downloadQualityReport() {
    const btn = document.getElementById('sqs-download-pdf');
    const hint = document.getElementById('sqs-download-hint');
    if (btn) { btn.disabled = true; btn.textContent = '⏳ Generating…'; }
    try {
        const win  = document.getElementById('insight-window')?.value || '24h';
        const ship = document.getElementById('insight-shipment')?.value || '';
        const params = new URLSearchParams({ window: win });
        if (ship) params.set('shipment', ship);
        const url = `/api/shipment_quality_score/report.pdf?${params}`;

        // Probe first so we can surface a 503 (reportlab missing) inline
        // instead of showing the user a raw error PDF.
        const head = await fetch(url, { method: 'GET' });
        if (!head.ok) {
            const detail = await head.json().catch(() => ({}));
            const msg = detail.hint
                ? `PDF lib missing — run on server:\n${detail.hint}`
                : `Report failed: HTTP ${head.status}`;
            alert(msg);
            return;
        }
        const blob = await head.blob();
        const blobUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = blobUrl;
        // Browser also honors server Content-Disposition; setting download
        // gives a sensible fallback name if the server omits it.
        const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 16);
        a.download = `quality_${ship || 'all'}_${win}_${stamp}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(blobUrl), 4000);
        if (hint) hint.textContent = 'downloaded ✓';
        setTimeout(() => { if (hint) hint.textContent = 'uses current window + shipment'; }, 3000);
    } catch (e) {
        console.error('downloadQualityReport failed:', e);
        alert('Report generation failed: ' + (e.message || e));
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '📄 Download PDF'; }
    }
}
window.downloadQualityReport = downloadQualityReport;


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


async function refreshDetectionInsights() {
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
        return;
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
}

// ---------------------------------------------------------------------------
// Advanced charts: object size distribution, confidence over time, camera
// scatter — all from /api/detection_charts, scoped by window + shipment.
// ---------------------------------------------------------------------------
async function refreshAdvancedCharts() {
    const window = document.getElementById('insight-window')?.value || '24h';
    const shipment = document.getElementById('insight-shipment')?.value || '';
    const minConf = (parseFloat(document.getElementById('insight-min-conf')?.value || '0') / 100) || 0;
    let data;
    try {
        const r = await fetch('/api/detection_charts?window=' + encodeURIComponent(window) +
                              '&shipment=' + encodeURIComponent(shipment) +
                              '&min_conf=' + minConf);
        data = await r.json();
    } catch (e) { console.error('detection_charts fetch failed:', e); return; }

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

    // ---- (3) Camera × TIME scatter — hover a dot to preview that exact frame ----
    const scCtx = document.getElementById('insight-camera-scatter');
    if (scCtx) {
        if (_insightCameraScatter) _insightCameraScatter.destroy();
        const byClass = {};
        scatter.forEach(p => {
            if (!_isShown(p.cls)) return;
            (byClass[p.cls] = byClass[p.cls] || []).push({
                x: p.x, y: p.y, r: 3 + (p.r || 0) * 9, conf: p.r, cls: p.cls, img: p.img, ship: p.ship
            });
        });
        const datasets = Object.keys(byClass).map(cls => ({
            label: cls, data: byClass[cls],
            backgroundColor: _classColor(cls) + 'cc', borderColor: _classColor(cls)
        }));
        _insightCameraScatter = new Chart(scCtx, {
            type: 'bubble',
            data: { datasets },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, labels: { color: '#cbd5e1', font: { size: 10 }, boxWidth: 12 } },
                    title: { display: true, text: 'Camera × time — hover for image, click dot to open the exact frame', color: '#cbd5e1', font: { size: 13 } },
                    tooltip: { enabled: false, external: _scatterImageTooltip }
                },
                onClick: (evt, els, chart) => {
                    if (!els || !els.length) return;
                    const dp = chart.data.datasets[els[0].datasetIndex].data[els[0].index];
                    if (!dp) return;
                    const cls = (chart.data.datasets[els[0].datasetIndex].label) || dp.cls || '';
                    openDefectDrawerForFrame({
                        image_path: dp.img, shipment: dp.ship, t: dp.x,
                        cls: cls, classes: cls ? [cls] : [], best_confidence: dp.r || 0,
                    });
                },
                scales: {
                    x: { type: 'linear', ticks: { color: '#94a3b8', font: { size: 9 }, callback: (v) => new Date(v).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}) }, grid: { color: 'rgba(148,163,184,0.08)' } },
                    y: { title: { display: true, text: 'Camera', color: '#94a3b8' }, ticks: { color: '#94a3b8', stepSize: 1, precision: 0 }, grid: { color: 'rgba(148,163,184,0.1)' } }
                }
            }
        });
    }

    // ---- (3b) Camera × ENCODER scatter — defect map by roll position ----
    const scEncCtx = document.getElementById('insight-camera-scatter-encoder');
    const scatterEnc = data.camera_scatter_encoder || [];
    if (scEncCtx) {
        if (_insightCameraScatterEncoder) _insightCameraScatterEncoder.destroy();
        const byClassE = {};
        scatterEnc.forEach(p => {
            if (!_isShown(p.cls)) return;
            (byClassE[p.cls] = byClassE[p.cls] || []).push({
                x: p.x, y: p.y, r: 3 + (p.r || 0) * 9, conf: p.r, cls: p.cls, img: p.img, ship: p.ship
            });
        });
        const datasetsE = Object.keys(byClassE).map(cls => ({
            label: cls, data: byClassE[cls],
            backgroundColor: _classColor(cls) + 'cc', borderColor: _classColor(cls)
        }));
        _insightCameraScatterEncoder = new Chart(scEncCtx, {
            type: 'bubble',
            data: { datasets: datasetsE },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, labels: { color: '#cbd5e1', font: { size: 10 }, boxWidth: 12 } },
                    title: { display: true, text: 'Camera × encoder (roll position) — hover for image, click dot for the exact frame', color: '#cbd5e1', font: { size: 13 } },
                    tooltip: { enabled: false, external: _scatterImageTooltip }
                },
                onClick: (evt, els, chart) => {
                    if (!els || !els.length) return;
                    const dp = chart.data.datasets[els[0].datasetIndex].data[els[0].index];
                    if (!dp) return;
                    const cls = (chart.data.datasets[els[0].datasetIndex].label) || dp.cls || '';
                    // encoder-scatter has no timestamp on the dot; derive from filename if present
                    let t = null;
                    try {
                        const fn = String(dp.img || '').split('/').pop() || '';
                        const m = fn.match(/^(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})/);
                        if (m) t = Date.UTC(+m[1], +m[2]-1, +m[3], +m[4], +m[5], +m[6]);
                    } catch (e) {}
                    // 3.21.17: surface encoder value + camera index in the drawer.
                    // dp.x = encoder, dp.y = camera index (Y axis labelled "Camera").
                    openDefectDrawerForFrame({
                        image_path: dp.img, shipment: dp.ship, t: t,
                        cls: cls, classes: cls ? [cls] : [], best_confidence: dp.r || 0,
                        encoder: dp.x, camera_index: dp.y,
                    });
                },
                scales: {
                    x: { type: 'linear', ticks: { color: '#94a3b8', font: { size: 9 } }, grid: { color: 'rgba(148,163,184,0.08)' }, title: { display: true, text: 'Encoder (roll position)', color: '#94a3b8', font: { size: 10 } } },
                    y: { title: { display: true, text: 'Camera', color: '#94a3b8' }, ticks: { color: '#94a3b8', stepSize: 1, precision: 0 }, grid: { color: 'rgba(148,163,184,0.1)' } }
                }
            }
        });
    }
}

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
// Image URL: /api/raw_image/<path> (image_path stored on the row, stripping the
// "raw_images/" prefix). Drawer uses /api/recent_detections.
// ---------------------------------------------------------------------------
function _imgUrlFromPath(p) {
    if (!p) return null;
    return '/api/raw_image/' + encodeURI(String(p).replace(/^raw_images\//, ''));
}
// Both URLs (annotated + raw) for any chart data point or drawer item.
// Annotated lives at raw_images/<frame_id>_DETECTED.jpg; raw lives at
// raw_images/<ship>/<YYYY-MM-DD_HH>/<frame_id>.jpg — same frame_id, derived.
function _imgUrlsFor(pt) {
    if (!pt || !pt.img) return { ann: null, raw: null };
    const ann = _imgUrlFromPath(pt.img);
    const fname = (String(pt.img).split('/').pop() || '').replace(/_DETECTED\.jpg$/, '.jpg');
    const m = fname.match(/^(\d{4}-\d{2}-\d{2})-(\d{2})-/);
    let raw = null;
    if (m && pt.ship) {
        const chunk = m[1] + '_' + m[2];
        raw = '/api/raw_image/' + encodeURI(pt.ship + '/' + chunk + '/' + fname);
    }
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
function _scatterImageTooltip(ctx) {
    const tt = ctx.tooltip;
    const el = document.getElementById('chart-image-preview');
    if (!el) return;
    if (!tt || tt.opacity === 0 || !tt.dataPoints || !tt.dataPoints.length) {
        el.style.display = 'none'; return;
    }
    const dp = tt.dataPoints[0];
    const pt = dp.raw || {};
    const urls = _imgUrlsFor(pt);
    if (!urls.ann && !urls.raw) { el.style.display = 'none'; return; }
    const imgs = el.querySelectorAll('img');
    const cap = el.querySelector('.cap');
    // imgs[0] = raw, imgs[1] = annotated
    if (imgs[0] && urls.raw && imgs[0].dataset.src !== urls.raw) { imgs[0].dataset.src = urls.raw; imgs[0].src = urls.raw; }
    if (imgs[1] && urls.ann && imgs[1].dataset.src !== urls.ann) { imgs[1].dataset.src = urls.ann; imgs[1].src = urls.ann; }
    const cls = (dp.dataset && dp.dataset.label) || pt.cls || '';
    const conf = ((pt.conf || 0) * 100).toFixed(0);
    cap.textContent = cls + ' • cam ' + pt.y + ' • conf ' + conf + '%   (raw | annotated — click for big view)';
    const box = ctx.chart.canvas.getBoundingClientRect();
    el.style.left = (box.left + window.pageXOffset + tt.caretX + 14) + 'px';
    el.style.top  = Math.max(8, (box.top + window.pageYOffset + tt.caretY - 120)) + 'px';
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
            wrap.innerHTML = '<div style="color:#64748b; font-size:13px; padding:24px;">(no ' + kind + ' image available)</div>';
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
                               '<div style="font-size:11px; color:#64748b; margin-top:4px;">The ' + kind + ' jpg was pruned by the disk-retention policy (chart records outlive raw-image chunks).</div>' +
                               '<div style="font-size:10px; color:#475569; margin-top:6px; word-break:break-all;">' + url + '</div>';
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
    const winEl = document.getElementById('quality-charts-window');
    const win = (winEl && winEl.value) || '24h';
    await Promise.all([
        _loadQualityShipmentsChart(),
        _loadQualityHeatmap('time', win),
        _loadQualityHeatmap('encoder', win),
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
        const data   = rows.map(s => s.score != null ? s.score : 0);
        const colors = rows.map(s => _verdictColor(s.verdict));
        if (_qualityShipmentsChart) _qualityShipmentsChart.destroy();
        _qualityShipmentsChart = new Chart(canvas, {
            type: 'bar',
            data: { labels, datasets: [{ data, backgroundColor: colors,
                borderColor: colors.map(c => c.replace('0.7', '1')), borderWidth: 1 }] },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            title: (items) => items[0].label,
                            label:  (item) => {
                                const r = rows[item.dataIndex];
                                const tops = (r.top_defects || []).join(', ');
                                return [
                                    `Score: ${r.score ?? '—'}`,
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

async function _loadQualityHeatmap(axis, win) {
    const stripId = axis === 'encoder' ? 'quality-encoder-strip' : 'quality-time-strip';
    const axisId  = axis === 'encoder' ? 'quality-encoder-axis'  : 'quality-time-axis';
    const strip   = document.getElementById(stripId);
    const axisEl  = document.getElementById(axisId);
    if (!strip) return;
    try {
        const r = await fetch(`/api/quality/heatmap?axis=${axis}&window=${encodeURIComponent(win)}&buckets=48`);
        const d = await r.json();
        const cells = d.buckets || [];
        if (!cells.length) {
            const hint = d.note || 'no data in window';
            strip.innerHTML = `<div style="color:var(--text-secondary); font-size:11px; padding:6px; flex:1; text-align:center; font-style:italic;">${hint}</div>`;
            if (axisEl) axisEl.innerHTML = '';
            return;
        }
        strip.innerHTML = cells.map(c => {
            const color = _scoreColor(c.score);
            const top = c.top_class ? `Top: ${c.top_class}` : '';
            const tip = `${c.label}\nScore: ${c.score}\nDetections: ${c.n}\n${top}`.trim();
            return `<div style="flex:1; background:${color}; cursor:default;" title="${tip.replace(/"/g, '&quot;')}"></div>`;
        }).join('');
        if (axisEl) {
            const first = cells[0]?.label || '';
            const mid   = cells[Math.floor(cells.length / 2)]?.label || '';
            const last  = cells[cells.length - 1]?.label || '';
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
    const url = '/api/export_csv?window=' + encodeURIComponent(win) +
                '&shipment=' + encodeURIComponent(ship) +
                '&min_conf=' + minConf;
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
