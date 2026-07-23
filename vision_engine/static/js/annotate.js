/* 3.26.0 — Label Studio Frontend 1.4.0 embed for in-line YOLO box correction.
 *
 * Wiring overview:
 *   1. Operator clicks ✏️ Annotate on the defect drawer.
 *   2. openAnnotateModal() reads the current frame from window._currentDefectItem
 *      (charts.js), fetches /api/frame_detections + /api/ai_trainer/labels +
 *      /api/ai_trainer/class_map in parallel.
 *   3. If any MVE class on the frame has no mapping, show the inline mapping
 *      panel; otherwise hide it.
 *   4. Build the <View> config string and the prefill array exactly the way
 *      ai-trainer.monitait.com does it (Training.vue:501-638 in
 *      monitait_all_services_deployment).
 *   5. Mount `new LabelStudio('label-studio', {...})`. The save flow lives in
 *      submitAnnotations() and walks annotation.serializeAnnotation().
 *
 * This file MUST load AFTER /static/vendor/label-studio-1.4.0/js/main.js so
 * the global `LabelStudio` constructor is defined.
 */

let _annotateLS = null;          // current LabelStudio instance
let _annotateFrame = null;       // { image_path, image_url, image_w, image_h, detections }
let _annotateLabels = [];        // [{id, category_name, color}]
let _annotateClassMap = {};      // { mve_class_name: trainer_category_id }
let _annotateTaskId = null;      // trainer task id (from /api/ai_trainer/config)

function _setStatus(msg, isErr) {
    const el = document.getElementById('annotate-status');
    if (!el) return;
    el.textContent = msg || '';
    el.style.color = isErr ? '#fca5a5' : '#fcd34d';
}

function _enableSubmit(on) {
    const btn = document.getElementById('annotate-submit');
    if (!btn) return;
    btn.disabled = !on;
    btn.style.opacity = on ? '1' : '0.5';
}

// 4.0.2 — entry point used by chart-dot clicks. Sets the global frame state
// (so LSF prefill + save flow can read it) and opens the annotate modal in one
// call. Replaces the old `openDefectDrawerForFrame → click ✏️ Annotate` two-step.
function openFrameInAnnotator(item) {
    window._currentDefectItem = item || null;
    try { _currentDefectItem = window._currentDefectItem; } catch (e) {}
    openAnnotateModal();
}
window.openFrameInAnnotator = openFrameInAnnotator;

// 4.0.2 — bundle download: one raw + one render-on-demand DETECTED, both via
// the standard /api/raw_image and /api/render_detected endpoints. Each link
// gets a hidden `<a download>` so Chrome accepts both without a popup prompt.
function downloadFrameBundle() {
    if (!window._currentDefectItem || !window._currentDefectItem.image_path) {
        alert('No frame selected'); return;
    }
    const rel = String(window._currentDefectItem.image_path)
        .replace(/^raw_images\//, '')
        .replace(/_DETECTED\.jpg$/, '.jpg');
    const stem = rel.split('/').pop().replace(/\.jpg$/i, '');
    const rawUrl = '/api/raw_image/' + encodeURI(rel);
    const annUrl = '/api/render_detected/' + encodeURI(rel) + '?download=1';
    function _kick(url, name) {
        const a = document.createElement('a');
        a.href = url; a.download = name; a.style.display = 'none';
        document.body.appendChild(a); a.click(); setTimeout(() => a.remove(), 1000);
    }
    _kick(rawUrl, stem + '.jpg');
    setTimeout(() => _kick(annUrl, stem + '_DETECTED.jpg'), 300);
    _setStatus('📥 Downloading raw + DETECTED…');
}
window.downloadFrameBundle = downloadFrameBundle;

async function openAnnotateModal() {
    if (!window.window._currentDefectItem || !window._currentDefectItem.image_path) {
        alert('No frame selected.'); return;
    }
    if (typeof LabelStudio !== 'function') {
        // 3.26.0 follow-up — the most common cause is a stale status.html cached
        // from before the deploy that added the LSF <script> tag. Show a clearer
        // diagnostic so the operator knows exactly which knob to turn.
        const scripts = Array.from(document.scripts).map(s => s.src).filter(s => s.indexOf('label-studio') >= 0);
        const probe = document.createElement('script');
        const msg = [
            'Label Studio editor did not initialize.',
            '',
            'Diagnostic:',
            '  typeof LabelStudio = ' + typeof LabelStudio,
            '  LSF <script> tags found: ' + (scripts.length ? scripts.join(', ') : '(none — status.html is cached from before 3.26.0)'),
            '',
            'Fix:',
            '  1. Hard-reload the page (Ctrl+Shift+R / Cmd+Shift+R) to drop the cached HTML.',
            '  2. If still broken, open DevTools → Network and check that',
            '     /static/vendor/label-studio-1.4.0/js/main.js loads with status 200.',
        ].join('\n');
        alert(msg);
        return;
    }
    const modal = document.getElementById('annotate-modal');
    if (!modal) return;
    modal.style.display = 'flex';
    // 4.0.14 — lock the body scroll while the LSF modal is open so mouse wheel
    // on the editor doesn't drag the dashboard underneath up and down.
    document.body.style.overflow = 'hidden';
    // Also block any wheel that bubbles up past the editor — Konva swallows
    // wheel events that are over the canvas; this catches every other wheel
    // inside the modal body and stops it from scrolling the document.
    if (!modal._wheelLockWired) {
        modal.addEventListener('wheel', (e) => { e.preventDefault(); }, { passive: false });
        modal._wheelLockWired = true;
    }
    _setStatus('Loading frame + labels…');
    _enableSubmit(false);

    try {
        const [frameRes, labelsRes, mapRes, cfgRes] = await Promise.all([
            fetch('/api/frame_detections?image_path=' + encodeURIComponent(window._currentDefectItem.image_path)),
            fetch('/api/ai_trainer/labels'),
            fetch('/api/ai_trainer/class_map'),
            fetch('/api/ai_trainer/config'),
        ]);
        const frameD  = await frameRes.json();
        const labelsD = await labelsRes.json();
        const mapD    = await mapRes.json();
        const cfgD    = await cfgRes.json();

        if (!frameRes.ok)  throw new Error(frameD.error  || 'frame load failed');
        if (!labelsRes.ok) throw new Error(labelsD.error || 'labels load failed — check Advanced → AI Trainer');

        _annotateFrame    = frameD;
        _annotateLabels   = labelsD.labels || [];
        _annotateClassMap = mapD.class_map || {};
        _annotateTaskId   = (cfgD.task_id || labelsD.task_id || '').toString();

        // Header meta — 4.0.15 appends px/mm calibration when present
        const meta = document.getElementById('annotate-meta');
        if (meta) {
            const ppm = _annotateFrame && _annotateFrame.px_per_mm;
            const scaleStr = (ppm && ppm > 0)
                ? ` · scale ${Number(ppm).toFixed(2)} px/mm (1 px ≈ ${(1/ppm).toFixed(3)} mm)`
                : ' · uncalibrated (set px/mm on Cameras tab to see mm)';
            meta.textContent =
                `${(window._currentDefectItem.image_path || '').split('/').pop()} · task ${_annotateTaskId || '?'} · ${_annotateLabels.length} categories${scaleStr}`;
        }

        // Find every MVE class present on the frame and check it's mapped.
        // 4.0.29 — exclude synthetic detection entries (names starting with
        // `_`, e.g. `_color` carrying frame L*a*b* metrics). They aren't YOLO
        // classes, never live in the trainer's category list, and showing them
        // in the "unmapped classes" panel asked the operator to manually map
        // a fake class to a trainer category. Same prefix convention is used
        // in the chart aggregation SQL (timeline.py:detection_charts).
        const mveClassesOnFrame = Array.from(new Set(
            (_annotateFrame.detections || [])
                .map(d => String(d.name || ''))
                .filter(n => n && !n.startsWith('_'))
        ));

        // 4.0.14 — auto-map MVE classes that already have a trainer category
        // with the same name (case-insensitive). The operator only sees the
        // mapping panel for truly novel classes. Auto-matches get persisted on
        // first save so the autosuggestion only happens once per class.
        const _norm = s => String(s || '').trim().toLowerCase();
        const _byNorm = new Map(_annotateLabels.map(l => [_norm(l.category_name), l]));
        let _autoMatched = 0;
        for (const mveCls of mveClassesOnFrame) {
            if (_annotateClassMap[mveCls]) continue;       // already mapped
            const hit = _byNorm.get(_norm(mveCls));
            if (hit) {
                _annotateClassMap[mveCls] = Number(hit.id);
                _autoMatched++;
            }
        }
        // Persist the auto-matched pairs in the background — fire and forget; if
        // the trainer is briefly unreachable the operator just sees the suggestion
        // again next time, no UX impact.
        if (_autoMatched > 0) {
            fetch('/api/ai_trainer/class_map', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ class_map: _annotateClassMap }),
            }).catch(() => {});
        }

        const unmapped = mveClassesOnFrame.filter(c => !_annotateClassMap[c]);
        _renderClassMapPanel(mveClassesOnFrame, unmapped);

        // Wait for the image to load so LSF gets the right natural dimensions.
        // LSF reads the image via the `data.image` URL; we just need to ensure
        // it's reachable (HEAD probe via Image()).
        await _waitForImage(_annotateFrame.image_url);

        _mountLabelStudio();
        _setStatus('Edit boxes — click Submit when done.');
        _enableSubmit(unmapped.length === 0);
    } catch (e) {
        _setStatus('✗ ' + (e.message || e), true);
    }
}
window.openAnnotateModal = openAnnotateModal;

function closeAnnotateModal() {
    // 4.0.12 — close = hide + destroy. We don't reuse the instance anymore (the
    // assignTask path leaked Konva state into the next frame); destroying here
    // reclaims memory and keeps the next open's fresh `new LabelStudio()`
    // clean. The 2 MB LSF JS bundle remains in browser cache so re-init is
    // ~300–500 ms after the very first cold-start.
    const modal = document.getElementById('annotate-modal');
    if (modal) modal.style.display = 'none';
    // 4.0.14 — restore body scroll.
    document.body.style.overflow = '';
    if (_annotateLS) {
        try { _annotateLS.destroy(); } catch (e) {}
        _annotateLS = null;
    }
    const el = document.getElementById('label-studio');
    if (el) el.innerHTML = '';
}
window.closeAnnotateModal = closeAnnotateModal;

function _waitForImage(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload  = () => { _annotateFrame.image_w = img.naturalWidth;
                              _annotateFrame.image_h = img.naturalHeight;
                              resolve(); };
        img.onerror = () => reject(new Error('Image not reachable: ' + url));
        img.src = url;
    });
}

function _labelValueFor(catId) {
    // The trainer wraps each Label as `value="${category_name} - ${id}"` (see
    // Training.vue:621). We emit the same string so prefill rectangles match
    // an existing <Label>'s `value` attribute exactly.
    const c = _annotateLabels.find(l => Number(l.id) === Number(catId));
    if (!c) return null;
    return `${c.category_name} - ${c.id}`;
}

function _buildLSConfig() {
    // Mirrors the trainer's <View>/<Image>/<RectangleLabels>/<Label/> shape.
    // 4.0.13 — `zoom="true" zoomControl="true"` enables mouse-wheel zoom + pan
    // (Konva captures the wheel events on the Image stage and the page no longer
    // scrolls when the cursor is over the editor).
    const labelTags = _annotateLabels.map(c =>
        `<Label alias="${c.id}" value="${c.category_name} - ${c.id}" background="${c.color || '#94a3b8'}"/>`
    ).join('');
    return `
        <View>
          <Image name="img" value="$image" zoom="true" zoomControl="true" rotateControl="false" negativeZoom="true"/>
          <RectangleLabels name="tag" toName="img">
            ${labelTags}
          </RectangleLabels>
        </View>`;
}

function _buildPrefill() {
    // Convert MVE detection bboxes (pixel xmin/ymin/xmax/ymax) to LSF % rectangles,
    // mapping MVE class name → trainer category id via _annotateClassMap.
    const W = _annotateFrame.image_w, H = _annotateFrame.image_h;
    if (!W || !H) return [];
    const out = [];
    for (const d of (_annotateFrame.detections || [])) {
        const mveName = String(d.name || '');
        const catId = _annotateClassMap[mveName];
        if (catId == null) continue;       // unmapped → operator adds it manually
        const lblValue = _labelValueFor(catId);
        if (!lblValue) continue;            // category disappeared from trainer
        const xmin = Number(d.xmin || 0), ymin = Number(d.ymin || 0);
        const xmax = Number(d.xmax || 0), ymax = Number(d.ymax || 0);
        const x_pct = (xmin / W) * 100;
        const y_pct = (ymin / H) * 100;
        const w_pct = ((xmax - xmin) / W) * 100;
        const h_pct = ((ymax - ymin) / H) * 100;
        if (w_pct <= 0 || h_pct <= 0) continue;
        out.push({
            from_name: 'tag', to_name: 'img', source: '$image',
            type: 'rectanglelabels',
            value: { x: x_pct, y: y_pct, width: w_pct, height: h_pct, rotation: 0,
                     rectanglelabels: [lblValue] },
        });
    }
    return out;
}

function _mountLabelStudio() {
    // 4.0.12 — fresh LabelStudio instance per click. The earlier `assignTask`
    // reuse path leaked old Konva shapes + image into the new frame (user saw
    // a stale dark dot persisting between dots). We let the BROWSER cache the
    // 2 MB LSF bundle JS (that's the heavy part); React + MobX + Konva all
    // already in memory means a fresh instance still snaps up in ~300–500 ms
    // on second click vs ~1–3 s on the very first cold-start of the page.
    if (_annotateLS) {
        try { _annotateLS.destroy(); } catch (e) {}
        _annotateLS = null;
    }
    // Clear the mount point so React + Konva don't see ghost DOM from the
    // previous instance.
    const mountEl = document.getElementById('label-studio');
    if (mountEl) mountEl.innerHTML = '';

    const config = _buildLSConfig();
    const prefill = _buildPrefill();
    _annotateLS = new LabelStudio('label-studio', {
        config,
        interfaces: [
            'annotations:add-new', 'annotations:delete', 'annotations:menu',
            'controls', 'panel', 'predictions:menu', 'side-column',
            'skip', 'submit', 'update',
        ],
        task: {
            id: Date.now(),
            data: { image: _annotateFrame.image_url },
            annotations: [{ result: prefill }],
            predictions: [],
        },
        onSubmitAnnotation: () => { submitAnnotations(); },
        onUpdateAnnotation: () => { submitAnnotations(); },
    });
    // 4.0.25 — LSF 1.4.0 requires Ctrl+wheel to zoom by default. ai-trainer
    // zooms on plain wheel because it runs in a non-scrollable host; the
    // embed inside our modal does too (the modal does NOT scroll the
    // page). So plain wheel SHOULD zoom here too, matching operator muscle
    // memory from the trainer. We synthesize a Ctrl+wheel event off the
    // user's plain wheel event and dispatch it onto whatever canvas LSF
    // mounted (Konva renders the image into a <canvas>); LSF's own
    // listener then handles the zoom maths. Holding Ctrl manually still
    // works because we short-circuit when ctrlKey is already true.
    _attachLSFWheelZoom(mountEl);
}

function _attachLSFWheelZoom(mountEl) {
    if (!mountEl || mountEl._wheelZoomWired) return;
    mountEl._wheelZoomWired = true;
    mountEl.addEventListener('wheel', (ev) => {
        if (ev.ctrlKey) return;     // operator IS holding Ctrl — let native handler fire
        // LSF's image canvas is a Konva <canvas> mounted somewhere inside
        // #label-studio. Find the canvas under the cursor (or any canvas
        // descendant as fallback) and dispatch a synthetic Ctrl+wheel onto it.
        const cnv = (ev.target && ev.target.closest && ev.target.closest('canvas'))
                    || mountEl.querySelector('canvas');
        if (!cnv) return;
        ev.preventDefault();
        ev.stopPropagation();
        const synth = new WheelEvent('wheel', {
            deltaX: ev.deltaX, deltaY: ev.deltaY, deltaZ: ev.deltaZ,
            deltaMode: ev.deltaMode,
            clientX: ev.clientX, clientY: ev.clientY,
            screenX: ev.screenX, screenY: ev.screenY,
            ctrlKey: true,
            bubbles: true, cancelable: true, view: window,
        });
        cnv.dispatchEvent(synth);
    }, { passive: false, capture: true });
}

async function submitAnnotations() {
    if (!_annotateLS) { _setStatus('No editor — reopen the modal.', true); return; }
    const annStore = _annotateLS.store && _annotateLS.store.annotationStore;
    const cur = annStore && annStore.selected;
    if (!cur) { _setStatus('No active annotation.', true); return; }
    _setStatus('Saving + shipping to trainer…');
    _enableSubmit(false);

    // 1. Upload the raw image so the trainer has a TaskImage row to attach
    //    annotations to. We reuse the existing /api/ai_trainer/upload endpoint
    //    which knows how to POST multipart with the JWT.
    let imageId = null;
    try {
        const upRes = await fetch('/api/ai_trainer/upload', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                image_path:  _annotateFrame.image_path,
                shipment:    window._currentDefectItem.shipment || '',
                class_name:  '',
                camera:      window._currentDefectItem.camera_index || null,
                task_id:     _annotateTaskId || undefined,
            }),
        });
        const upData = await upRes.json();
        if (!upRes.ok || !upData.success) {
            throw new Error(upData.error || ('upload HTTP ' + upRes.status));
        }
        // 3.26.3 — backend now parses the trainer's response and exposes the
        // first TaskImage.id straight on `image_id`. Old fallbacks kept for
        // belt-and-braces in case the trainer format changes.
        if (upData.image_id != null) {
            imageId = Number(upData.image_id);
        } else {
            const tr = upData.trainer_response_json || upData.trainer_response;
            if (Array.isArray(tr) && tr.length && tr[0].id != null) imageId = Number(tr[0].id);
            else if (tr && Array.isArray(tr.results) && tr.results.length) imageId = Number(tr.results[0].id);
        }
        if (!imageId) throw new Error('upload succeeded but no TaskImage id returned — backend parsed: ' + JSON.stringify(upData).slice(0, 300));
    } catch (e) {
        _setStatus('✗ Upload failed: ' + (e.message || e), true);
        _enableSubmit(true);
        return;
    }

    // 4.0.6 — use `annotation.serializeAnnotation()` (the same call the trainer's
    // own Training.vue:752 uses) instead of iterating `regions` + calling
    // `region.serialize()` per region. The latter silently drops prefilled
    // rectangles in some LSF builds; the former returns the full result array
    // including every box the operator sees.
    const W = _annotateFrame.image_w, H = _annotateFrame.image_h;
    const tidNum = parseInt(_annotateTaskId, 10);
    let serResults = [];
    try {
        const ser = cur.serializeAnnotation && cur.serializeAnnotation();
        if (Array.isArray(ser)) serResults = ser;
        else if (ser && Array.isArray(ser.result)) serResults = ser.result;
    } catch (e) {
        console.warn('serializeAnnotation failed, falling back to regions:', e);
    }
    if (!serResults.length && cur.regions && cur.regions.length) {
        // Fallback: serialize regions one at a time.
        for (const region of cur.regions) {
            try {
                const r = region.serialize && region.serialize();
                if (r) serResults.push(r);
            } catch (e) {}
        }
    }
    const payload = [];
    for (const r of serResults) {
        if (!r || r.type !== 'rectanglelabels') continue;
        const v = r.value;
        if (!v) continue;
        const lblValue = (v.rectanglelabels || [])[0];
        if (!lblValue) continue;
        // 4.0.14 — accept BOTH label-value forms LSF emits:
        //   * Prefilled boxes we constructed: "STAIN - 1383" → match trailing num
        //   * User-drawn boxes: LSF puts the alias (just "1383") → match all-digits
        // Without this fallback every user-correction got silently dropped, only
        // the original YOLO prefill went to the trainer.
        const s = String(lblValue).trim();
        let catId = null;
        const m1 = s.match(/-\s*(\d+)\s*$/);
        if (m1) catId = parseInt(m1[1], 10);
        else if (/^\d+$/.test(s)) catId = parseInt(s, 10);
        if (catId == null) continue;
        const x_pct = Number(v.x), y_pct = Number(v.y);
        const w_pct = Number(v.width), h_pct = Number(v.height);
        payload.push({
            ignore: true, is_crowd: false,
            bbox: [
                Math.round(x_pct / 100 * W),
                Math.round(y_pct / 100 * H),
                Math.round(w_pct / 100 * W),
                Math.round(h_pct / 100 * H),
            ],
            area: w_pct * h_pct,
            image_id: imageId,
            category_id: catId,
            normalized_percent_points: [x_pct, y_pct, w_pct, h_pct],
            segmentation: [],
            task_id: tidNum,
        });
    }
    if (!payload.length) {
        const why = serResults.length
            ? `(${serResults.length} region${serResults.length === 1 ? '' : 's'} but none had a label — pick a class from the Labels panel before saving)`
            : '(editor returned 0 regions — draw at least one box and label it)';
        _setStatus('Nothing to save ' + why, true);
        _enableSubmit(true);
        return;
    }
    try {
        const r = await fetch('/api/ai_trainer/annotate', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ annotations: payload }),
        });
        const d = await r.json();
        if (!r.ok) throw new Error(d.error || ('save HTTP ' + r.status));
        _setStatus(`✓ Sent ${payload.length} annotation${payload.length === 1 ? '' : 's'} to trainer (image ${imageId}).`);
    } catch (e) {
        _setStatus('✗ Save failed: ' + (e.message || e), true);
    } finally {
        _enableSubmit(true);
    }
}
window.submitAnnotations = submitAnnotations;

function _renderClassMapPanel(mveClasses, unmapped) {
    const el = document.getElementById('annotate-classmap');
    if (!el) return;
    // v4.0.158 — defensive filter: drop any underscore-prefixed class names
    // (synthetic math-inference outputs like `_color`, `_stitch`). Two
    // upstream call sites already filter, but if any future caller forgets,
    // this belt-and-suspenders keeps the operator from ever seeing synthetic
    // classes in the "unmapped" panel.
    unmapped = (unmapped || []).filter(n => n && !String(n).startsWith('_'));
    if (!unmapped.length) {
        el.style.display = 'none';
        el.innerHTML = '';
        return;
    }
    el.style.display = 'block';
    const opts = _annotateLabels.map(c =>
        `<option value="${c.id}">${c.category_name} - ${c.id}</option>`
    ).join('');
    const rows = unmapped.map(cls => `
        <span style="display:inline-flex; align-items:center; gap:6px; margin-right:14px;">
            <span style="font-weight:700; min-width:90px;">${cls}</span>
            <span>→</span>
            <select data-mve-class="${cls}" class="annotate-classmap-pick"
                style="background:rgba(15,23,42,0.6); color:#fff; border:1px solid rgba(245,158,11,0.4); padding:3px 6px; border-radius:3px; font-size:11px;">
                <option value="">— pick category —</option>
                ${opts}
            </select>
        </span>
    `).join('');
    el.innerHTML = `
        <div style="margin-bottom:6px;"><b>⚠ Unmapped classes</b> — the editor can't pre-fill these boxes until you map each one to a trainer category. Maps persist for the site.</div>
        ${rows}
        <button onclick="saveClassMapAndRemount()" style="background:linear-gradient(135deg,#f59e0b,#d97706); color:#fff; border:none; padding:5px 12px; cursor:pointer; font-size:11px; border-radius:4px; font-weight:600; margin-top:8px;">Save mapping + reload prefill</button>
    `;
}

async function saveClassMapAndRemount() {
    const picks = document.querySelectorAll('.annotate-classmap-pick');
    const next = Object.assign({}, _annotateClassMap);
    let any = false;
    picks.forEach(p => {
        const k = p.getAttribute('data-mve-class');
        const v = p.value;
        if (k && v) { next[k] = parseInt(v, 10); any = true; }
    });
    if (!any) { _setStatus('Pick at least one category before saving.', true); return; }
    _setStatus('Saving mapping…');
    try {
        const r = await fetch('/api/ai_trainer/class_map', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ class_map: next }),
        });
        const d = await r.json();
        if (!r.ok) throw new Error(d.error || ('HTTP ' + r.status));
        _annotateClassMap = d.class_map || next;
        // v4.0.158 — mirror the underscore-prefix filter from the initial-open
        // pass (line ~156). Prior code only filtered empty strings here, so
        // synthetic math-inference classes (`_color`, `_stitch`, …) that got
        // filtered on modal open would REAPPEAR in the unmapped list the
        // moment the operator clicked "Save mapping + reload prefill".
        const mveClasses = Array.from(new Set(
            (_annotateFrame.detections || [])
                .map(d => String(d.name || ''))
                .filter(n => n && !n.startsWith('_'))
        ));
        const unmapped = mveClasses.filter(c => !_annotateClassMap[c]);
        _renderClassMapPanel(mveClasses, unmapped);
        _mountLabelStudio();
        _enableSubmit(unmapped.length === 0);
        _setStatus(unmapped.length ? `${unmapped.length} class still unmapped.` : '✓ Mapping saved.');
    } catch (e) {
        _setStatus('✗ ' + (e.message || e), true);
    }
}
window.saveClassMapAndRemount = saveClassMapAndRemount;

// ESC closes the modal (matches the defect drawer's behaviour).
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && document.getElementById('annotate-modal')?.style.display === 'flex') {
        closeAnnotateModal();
    }
});
