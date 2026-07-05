// ===== Audio Alert System =====
// Production line alarm system for real-time object detection
let audioSettings = {
    narrate: true,         // Narration controlled per-object
    beep: true,            // Beep controlled per-object
    volume: 0.5,
    showBoundingBoxes: true, // Bounding boxes controlled per-object
    enabledObjects: {},    // { "socket": true, ... } - per-object beep
    narrateObjects: {},    // { "socket": true, ... } - per-object narrate
    objectBeepSounds: {},  // { "socket": "sine", ... }
    showObjects: {},       // { "socket": true, ... } - per-object show bbox
    objectConfidence: {},  // { "socket": 1, ... } - min confidence %
    storeObjects: {}       // { "socket": true, ... } - per-object DB persistence; SERVER-SIDE source of truth at /api/store_objects, mirrored here for UI rendering. Default OFF (no DB write).
};
let audioContext = null;
let detectedObjectClasses = new Set();  // Track all detected objects
let lastNarrationTime = 0;
let lastBeepTime = 0;
const NARRATION_COOLDOWN = 2000;  // 2 seconds between same narrations
const BEEP_COOLDOWN = 1000;       // 1 second between beeps

// Load audio settings from localStorage
function loadAudioSettings() {
    const saved = localStorage.getItem('audioSettings');
    if (saved) {
        try {
            audioSettings = { ...audioSettings, ...JSON.parse(saved) };
        } catch (e) {
            console.error('Error loading audio settings:', e);
        }
    }
    updateAudioUI();
}

// Save audio settings to localStorage
function saveAudioSettings() {
    localStorage.setItem('audioSettings', JSON.stringify(audioSettings));
}

// Update UI to reflect current settings
function updateAudioUI() {
    // Global toggles always enabled (controlled per-object)
    audioSettings.narrate = true;
    audioSettings.beep = true;
    audioSettings.showBoundingBoxes = true;
}

// Fetch model classes from YOLO service
async function fetchModelClasses() {
    const btn = document.querySelector('button[onclick="fetchModelClasses()"]');
    if (btn) { btn.textContent = '⏳ Loading...'; btn.disabled = true; }
    try {
        const resp = await fetch('/api/model_classes');
        const data = await resp.json();
        if (data.classes && data.classes.length > 0) {
            data.classes.forEach(name => {
                if (!detectedObjectClasses.has(name)) {
                    detectedObjectClasses.add(name);
                    if (audioSettings.showObjects[name] === undefined) audioSettings.showObjects[name] = true;
                    if (audioSettings.enabledObjects[name] === undefined) audioSettings.enabledObjects[name] = true;
                    if (audioSettings.narrateObjects[name] === undefined) audioSettings.narrateObjects[name] = true;
                    if (audioSettings.objectConfidence[name] === undefined) audioSettings.objectConfidence[name] = 1;
                    if (audioSettings.storeObjects[name] === undefined) audioSettings.storeObjects[name] = false;
                }
            });
            saveAudioSettings();
            updateObjectsList();
            if (btn) { btn.textContent = `✓ ${data.classes.length} classes`; setTimeout(() => { btn.textContent = 'Fetch Classes'; btn.disabled = false; }, 2000); }
        } else {
            if (btn) { btn.textContent = 'No classes found'; setTimeout(() => { btn.textContent = 'Fetch Classes'; btn.disabled = false; }, 2000); }
        }
    } catch (e) {
        console.error('Could not fetch model classes:', e);
        if (btn) { btn.textContent = '✗ Error'; setTimeout(() => { btn.textContent = 'Fetch Classes'; btn.disabled = false; }, 2000); }
    }
}

// Initialize Web Audio API (must be called after user interaction)
function initAudioContext() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioContext.state === 'suspended') {
        audioContext.resume();
    }
}

// Audio unlock — browsers require a user gesture before playing sound.
// Show a one-click overlay on first load to capture the gesture.
// Once unlocked, it stays unlocked for the session.
(function() {
    function unlockAudio() {
        initAudioContext();
        if ('speechSynthesis' in window) {
            const silent = new SpeechSynthesisUtterance('');
            silent.volume = 0;
            window.speechSynthesis.speak(silent);
        }
        console.log('[Audio] Unlocked via user gesture');
    }

    function isAudioUnlocked() {
        return audioContext && audioContext.state === 'running';
    }

    // Try silent unlock first (works if site is in Chrome's autoplay allowlist)
    // For industrial kiosks: no blocking overlay — show a small toast instead
    window.addEventListener('load', function() {
        initAudioContext();
        // Give browser a moment to resolve autoplay policy
        setTimeout(function() {
            if (isAudioUnlocked()) {
                console.log('[Audio] Unlocked automatically (allowlisted)');
                return;
            }
            // Not allowlisted — show small non-blocking toast at bottom
            const toast = document.createElement('div');
            toast.id = 'audio-unlock-overlay';
            toast.style.cssText =
                'position:fixed;bottom:20px;left:50%;transform:translateX(-50%);z-index:99999;' +
                'background:#1e293b;border:1px solid #3b82f6;border-radius:8px;' +
                'padding:10px 20px;color:#e2e8f0;font-family:Inter,sans-serif;font-size:13px;' +
                'cursor:pointer;opacity:0.9;transition:opacity 0.3s;';
            toast.textContent = '🔊 Tap to enable audio alerts';
            toast.addEventListener('click', function() {
                unlockAudio();
                toast.remove();
            }, { once: true });
            document.body.appendChild(toast);
            // Auto-fade after 8 seconds (doesn't block anything)
            setTimeout(function() {
                if (toast.parentNode) {
                    toast.style.opacity = '0';
                    setTimeout(function() { if (toast.parentNode) toast.remove(); }, 500);
                }
            }, 8000);
        }, 500);
    });

    // Also unlock on any interaction as fallback (e.g. tab click)
    function onGesture() {
        unlockAudio();
        if (isAudioUnlocked()) {
            document.removeEventListener('click', onGesture, true);
            document.removeEventListener('touchstart', onGesture, true);
            // Remove overlay if still visible
            const overlay = document.getElementById('audio-unlock-overlay');
            if (overlay) overlay.remove();
        }
    }
    document.addEventListener('click', onGesture, true);
    document.addEventListener('touchstart', onGesture, true);
})();

// Toggle global bounding boxes
function toggleGlobalBbox() {
    const bboxEl = document.getElementById('global-show-bbox');
    audioSettings.showBoundingBoxes = bboxEl.checked;
    saveAudioSettings();
    console.log('[Detection] Bounding boxes', audioSettings.showBoundingBoxes ? 'enabled' : 'disabled');
}

// Toggle narration
function toggleNarration() {
    const narrateEl = document.getElementById('audio-narrate');
    audioSettings.narrate = narrateEl.checked;
    saveAudioSettings();
    console.log('[Audio] Narration', audioSettings.narrate ? 'enabled' : 'disabled');
}

// Toggle beep
function toggleBeep() {
    const beepEl = document.getElementById('audio-beep');
    audioSettings.beep = beepEl.checked;

    // Initialize audio context when beep is enabled
    if (audioSettings.beep) {
        initAudioContext();
    }

    saveAudioSettings();
    console.log('[Audio] Beep', audioSettings.beep ? 'enabled' : 'disabled');
}

// Update volume
function updateVolume() {
    const volumeEl = document.getElementById('audio-volume');
    const volumeDisplay = document.getElementById('audio-volume-display');
    audioSettings.volume = volumeEl.value / 100;
    if (volumeDisplay) volumeDisplay.textContent = Math.round(audioSettings.volume * 100) + '%';
    saveAudioSettings();
}

// Toggle object in beep list
function toggleObjectBeep(objectName) {
    audioSettings.enabledObjects[objectName] = !audioSettings.enabledObjects[objectName];
    saveAudioSettings();
    updateObjectsList();
    console.log('[Audio] Object', objectName, audioSettings.enabledObjects[objectName] ? 'enabled' : 'disabled');
    if (typeof syncObjectAudioToServer === 'function') syncObjectAudioToServer(objectName);
}

// Play a beep sound using Web Audio API (non-blocking)
function playAlarmBeep() {
    if (!audioContext || !audioSettings.beep) return;

    // Run audio in next event loop to avoid blocking main thread
    setTimeout(() => {
        try {
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();

            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);

            oscillator.type = 'sine';
            oscillator.frequency.value = 880;  // A5 - attention-grabbing frequency

            gainNode.gain.setValueAtTime(audioSettings.volume, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);

            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.2);
        } catch (e) {
            console.error('Error playing beep:', e);
        }
    }, 0);
}

// Narrate detected object using Web Speech API (non-blocking)
function narrateObject(objectName) {
    if (!audioSettings.narrate) return;

    // Cooldown check
    const now = Date.now();
    if (now - lastNarrationTime < NARRATION_COOLDOWN) {
        console.log('[Audio] Narration on cooldown');
        return;
    }
    lastNarrationTime = now;

    // Run speech synthesis in next event loop to avoid blocking main thread
    setTimeout(() => {
        // Check if browser supports speech synthesis
        if ('speechSynthesis' in window) {
            // Cancel any ongoing speech
            window.speechSynthesis.cancel();

            // Create utterance
            const utterance = new SpeechSynthesisUtterance(objectName);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.volume = audioSettings.volume;

            console.log('[Audio] Narrating:', objectName);
            window.speechSynthesis.speak(utterance);
        } else {
            console.warn('[Audio] Text-to-speech not supported in this browser');
        }
    }, 0);
}

// Knowledge base for the info-icon tooltip on each object filter card.
// Math channel descriptions live in i18n.js under translations.<lang>.mathTips
// — they're translatable per language. English is the canonical fallback;
// any missing key falls back to English. YOLO classes get a generic
// "trained-model class" message regardless of language.
//
// Terms used here are intentionally generic (no fabric-specific jargon like
// warp/weft/loom): MVE is a general vision-inspection platform — same
// channels apply to denim, tire-cord, knit, plastic film, glass, metal,
// PCB, etc.

function _t(key, fallback) {
    try {
        const dict = (typeof translations !== 'undefined') ? translations : {};
        const cur = (typeof currentLang !== 'undefined') ? currentLang : 'en';
        return (dict[cur] && dict[cur][key]) || (dict.en && dict.en[key]) || fallback;
    } catch (e) {
        return fallback;
    }
}

function _mathTip(key) {
    try {
        const dict = (typeof translations !== 'undefined') ? translations : {};
        const cur = (typeof currentLang !== 'undefined') ? currentLang : 'en';
        const tipsCur = dict[cur] && dict[cur].mathTips;
        const tipsEn  = dict.en && dict.en.mathTips;
        return (tipsCur && tipsCur[key]) || (tipsEn && tipsEn[key]) || null;
    } catch (e) {
        return null;
    }
}

function _formatTip(lines) {
    return lines.map(s => '<div>' + s + '</div>').join('');
}

function getObjectInfo(name) {
    // 1. Exact-match math channels — look up the i18n key
    const direct = _mathTip(name);
    if (direct) return _formatTip(Array.isArray(direct) ? direct : [direct]);

    // 2. Pattern-based math channels — parameterized names use a template key
    //    plus the rank/index substituted into a placeholder.
    let m;
    const patterns = [
        [/^fft_row_peak_(\d+)_energy$/,             'tip_fft_row_peak_K_energy'],
        [/^fft_row_peak_(\d+)_period_px$/,          'tip_fft_row_peak_K_period_px'],
        [/^fft_col_peak_(\d+)_energy$/,             'tip_fft_col_peak_K_energy'],
        [/^fft_col_peak_(\d+)_period_px$/,          'tip_fft_col_peak_K_period_px'],
        [/^fft2d_peak_(\d+)_energy$/,               'tip_fft2d_peak_K_energy'],
        [/^fft2d_peak_(\d+)_period_px$/,            'tip_fft2d_peak_K_period_px'],
        [/^fft2d_peak_(\d+)_angle_deg$/,            'tip_fft2d_peak_K_angle_deg'],
        [/^fft2d_peak_(\d+)_tilt_from_horizontal$/, 'tip_fft2d_peak_K_tilt_from_horizontal'],
        [/^fft2d_peak_(\d+)_tilt_from_vertical$/,   'tip_fft2d_peak_K_tilt_from_vertical'],
        [/^tile_/,                                  'tip_tile_generic'],
    ];
    for (const [rx, key] of patterns) {
        if ((m = rx.exec(name))) {
            const lines = _mathTip(key);
            if (lines) {
                const k = m[1] || '';
                const subbed = (Array.isArray(lines) ? lines : [lines])
                    .map(s => s.replace(/\{K\}/g, k));
                return _formatTip(subbed);
            }
        }
    }

    // 3. Fallback: YOLO class
    const yoloTip = _mathTip('tip_yolo_class');
    if (yoloTip) return _formatTip(Array.isArray(yoloTip) ? yoloTip : [yoloTip]);
    return _formatTip([
        '<b>Trained-model class</b>',
        'Defined in the loaded model weights (model.names).',
        'Rule patterns: present, count_greater, area_greater, color_delta.'
    ]);
}

// Update the objects list in the UI (merged show/confidence + audio)
function updateObjectsList() {
    const listEl = document.getElementById('audio-objects-list');
    if (!listEl) return;

    listEl.innerHTML = '';

    if (detectedObjectClasses.size === 0) {
        listEl.innerHTML = '<div style="padding: 15px; background: rgba(15, 23, 42, 0.5); border-radius: 6px; text-align: center; color: var(--text-secondary); font-style: italic;">No objects detected yet. Start detection to see objects here.</div>';
        return;
    }

    // 3.21.24 — busy-first sort + optional active/search filter
    const activeCounts = audioSettings._activeCounts || {};
    const activeOnlyEl = document.getElementById('per-object-active-only');
    const searchEl     = document.getElementById('per-object-search');
    const countChip    = document.getElementById('per-object-count');
    const activeOnly = activeOnlyEl ? activeOnlyEl.checked : false;
    const searchQ    = (searchEl ? searchEl.value.trim().toLowerCase() : '');

    let sortedObjects = Array.from(detectedObjectClasses);
    // Sort: classes with detections in the active-window first (by count desc), then idle alphabetically
    sortedObjects.sort((a, b) => {
        const ca = activeCounts[a] || 0;
        const cb = activeCounts[b] || 0;
        if (ca !== cb) return cb - ca;
        return a.localeCompare(b);
    });
    if (activeOnly) {
        sortedObjects = sortedObjects.filter(n => (activeCounts[n] || 0) > 0);
    }
    if (searchQ) {
        sortedObjects = sortedObjects.filter(n => n.toLowerCase().includes(searchQ));
    }
    if (countChip) {
        const totalClasses = detectedObjectClasses.size;
        const activeCount = Object.keys(activeCounts).length;
        const win = audioSettings._activeWindow || '1h';
        countChip.textContent = `Showing ${sortedObjects.length} of ${totalClasses} · ${activeCount} active in last ${win}`;
    }

    sortedObjects.forEach(objectName => {
        // 3.21.25 — explicit opt-in: only True renders as checked.
        // Matches the strict draw rule in services/draw_filters.py.
        const isShown = audioSettings.showObjects[objectName] === true;
        const confidence = audioSettings.objectConfidence[objectName] !== undefined ? audioSettings.objectConfidence[objectName] : 1;
        const isNarrate = audioSettings.narrateObjects[objectName] || false;
        const isBeep = audioSettings.enabledObjects[objectName] || false;
        const isStore = audioSettings.storeObjects[objectName] || false;
        const isColorE = !!audioSettings.colorEObjects?.[objectName];          // 3.22.2 — track CIELAB ΔE drift over time
        const isArea   = !!audioSettings.areaObjects?.[objectName];            // 3.22.3 — show bbox-area percentiles
        const severity = audioSettings.severityObjects?.[objectName] ?? 0;  // 3.21.12 — per-class severity (0–100, defect impact weight)
        // 4.0.26 — per-class semantic ROLE. Replaces the global
        // `parent_object_list` text-box with a per-card dropdown.
        //   context (default) — informational only, severity allowed but default 0
        //   defect            — counted in OK/NG + quality score
        //   parent            — defines the inspection region; severity disabled (always 0)
        //   marker            — encoder-axis landmark (e.g. stitch = roll boundary)
        // Backend will derive the active parent list from any class with
        // role==='parent', falling back to `_root` (whole frame) if none.
        const role = audioSettings.roleObjects?.[objectName] || 'context';
        const sevDisabled = role === 'parent';
        const baseline = audioSettings.confBaselines?.[objectName];          // 3.21.12 — read-only baseline {p50,p95} (auto-learned)
        const drift    = audioSettings.colorDrift?.[objectName];             // 3.22.2 — read-only ΔE drift {p5_de, p50_de, p95_de, by_camera}
        const areaSt   = audioSettings.areaStats?.[objectName];              // 3.22.3 — read-only bbox area stats {p5, p50, p95, by_camera}
        const beepSound = audioSettings.objectBeepSounds?.[objectName] || 'sine';
        const safeName = objectName.replace(/'/g, "\\'");

        const card = document.createElement('div');
        card.style.cssText = `padding: 10px; background: rgba(51, 65, 85, 0.4); border-radius: 6px; border: 1px solid ${isShown ? 'rgba(34,197,94,0.4)' : 'rgba(51, 65, 85, 0.6)'};`;

        const infoHtml = getObjectInfo(objectName);
        card.innerHTML = `
            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 8px;">
                <div style="font-weight: 600; color: var(--text-primary); font-size: 14px; flex: 1;">${objectName}</div>
                <button onclick="askWhyForClass('${safeName}', this)" title="Ask the AI why this class is behaving this way right now" style="background:linear-gradient(135deg,#10b981,#047857); color:#fff; border:none; padding:1px 7px; cursor:pointer; font-size:11px; border-radius:3px; font-weight:600;">🤔</button>
                <span class="info-tooltip-sm" style="cursor: help;">i<span class="info-tooltip-text" style="text-align: left; min-width: 280px;">${infoHtml}</span></span>
            </div>
            <div class="per-class-why-${safeName.replace(/[^a-zA-Z0-9]/g,'_')}" style="display:none; font-size:11px; color:#d1fae5; background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.3); border-radius:4px; padding:6px 8px; margin-bottom:6px; line-height:1.45;"></div>
            <div style="display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center; margin-bottom: 8px;">
                <label style="display: flex; align-items: center; gap: 4px; cursor: pointer; font-size: 12px; white-space: nowrap;" title="Show bounding box">
                    <input type="checkbox" ${isShown ? 'checked' : ''} onchange="toggleObjectShow('${safeName}')" style="width: 14px; height: 14px; cursor: pointer;">
                    Show
                </label>
                <label style="display: flex; align-items: center; gap: 4px; cursor: pointer; font-size: 12px; white-space: nowrap;" title="Voice narration">
                    <input type="checkbox" ${isNarrate ? 'checked' : ''} onchange="toggleObjectNarrate('${safeName}')" style="width: 14px; height: 14px; cursor: pointer;">
                    Narrate
                </label>
                <label style="display: flex; align-items: center; gap: 4px; cursor: pointer; font-size: 12px; white-space: nowrap;" title="Beep alert">
                    <input type="checkbox" ${isBeep ? 'checked' : ''} onchange="toggleObjectBeep('${safeName}')" style="width: 14px; height: 14px; cursor: pointer;">
                    Beep
                </label>
                <label style="display: flex; align-items: center; gap: 4px; cursor: pointer; font-size: 12px; white-space: nowrap;" title="Persist this class's detections to the database (off = drop, on = write to inference_results)">
                    <input type="checkbox" ${isStore ? 'checked' : ''} onchange="toggleObjectStore('${safeName}')" style="width: 14px; height: 14px; cursor: pointer;">
                    Store
                </label>
                <label style="display: flex; align-items: center; gap: 4px; cursor: pointer; font-size: 12px; white-space: nowrap;" title="ColorE — extract CIELAB color from this class's bbox on every detection. Drives the 🎨 line below.">
                    <input type="checkbox" ${isColorE ? 'checked' : ''} onchange="toggleObjectColorE('${safeName}')" style="width: 14px; height: 14px; cursor: pointer;">
                    ColorE
                </label>
                <label style="display: flex; align-items: center; gap: 4px; cursor: pointer; font-size: 12px; white-space: nowrap;" title="Area — store bbox area on detection + show p5/p50/p95 below.">
                    <input type="checkbox" ${isArea ? 'checked' : ''} onchange="toggleObjectArea('${safeName}')" style="width: 14px; height: 14px; cursor: pointer;">
                    Area
                </label>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 6px; flex-wrap: wrap;">
                <span style="font-size: 12px; color: var(--text-secondary); white-space: nowrap;" title="Semantic role of this class.&#10;&#10;Context — informational only, no special wiring.&#10;Defect — counted in OK/NG and weighted into the quality score by Severity.&#10;Parent — marks the inspection region (the physical product). When at least one class has role=Parent, frames are only scored if a parent is detected; if none of the configured classes are Parent, the whole frame (_root) is the parent.&#10;Marker — landmark on the encoder axis (e.g. a stitch marks roll boundaries). Surfaces on the encoder strip for spatial reference, doesn't affect scoring.">Role:</span>
                <select onchange="updateObjectRole('${safeName}', this.value)"
                    style="padding: 3px 6px; background: rgba(30,41,59,0.6); color: var(--text-primary); border: 1px solid rgba(51,65,85,0.6); border-radius: 4px; font-size: 12px; cursor: pointer;"
                    title="Role of this class in the system.">
                    <option value="context" ${role==='context'?'selected':''}>📋 Context</option>
                    <option value="defect"  ${role==='defect' ?'selected':''}>⚠️ Defect</option>
                    <option value="parent"  ${role==='parent' ?'selected':''}>🌳 Parent</option>
                    <option value="marker"  ${role==='marker' ?'selected':''}>📍 Marker</option>
                </select>
                <span style="font-size: 12px; color: var(--text-secondary); white-space: nowrap; margin-left: 6px;">Min conf:</span>
                <input type="number" value="${confidence}" min="0" max="100" step="1"
                    onchange="updateObjectConfidence('${safeName}', this.value)"
                    style="width: 55px; padding: 3px 5px; background: rgba(30,41,59,0.6); color: var(--text-primary); border: 1px solid rgba(51,65,85,0.6); border-radius: 4px; font-size: 12px; text-align: center;">
                <span style="font-size: 12px; color: var(--text-secondary);">%</span>
                <span style="font-size: 12px; color: ${sevDisabled?'rgba(148,163,184,0.4)':'var(--text-secondary)'}; white-space: nowrap; margin-left: 6px;" title="Severity weight 0–100. Used to compute defect impact score (severity × confidence × area). Higher = this class hurts shipment quality more. Disabled when role=Parent because parents define the inspection area, not a defect impact.">Severity:</span>
                <input type="number" value="${severity}" min="0" max="100" step="1" ${sevDisabled?'disabled':''}
                    onchange="updateObjectSeverity('${safeName}', this.value)"
                    style="width: 55px; padding: 3px 5px; background: ${sevDisabled?'rgba(30,41,59,0.25)':'rgba(30,41,59,0.6)'}; color: ${sevDisabled?'rgba(148,163,184,0.4)':'var(--text-primary)'}; border: 1px solid rgba(51,65,85,0.6); border-radius: 4px; font-size: 12px; text-align: center; cursor: ${sevDisabled?'not-allowed':'auto'};"
                    title="${sevDisabled?'Disabled: role=Parent defines inspection area, not impact':'Per-class defect impact weight (0=cosmetic, 100=critical)'}">
            </div>
            ${baseline ? `<div style="display:flex; align-items:center; gap:4px; font-size: 11px; color: var(--text-secondary); margin-bottom: 6px; padding: 3px 6px; background: rgba(15,23,42,0.4); border-radius: 3px;">
                <span style="flex:1;">📊 normal conf: ${baseline.p5 !== undefined ? `${(baseline.p5*100).toFixed(0)} · <b>${(baseline.p50*100).toFixed(0)}</b> · ${(baseline.p95*100).toFixed(0)}% (p5–p50–p95)` : `${(baseline.p50*100).toFixed(0)}–${(baseline.p95*100).toFixed(0)}% (p50–p95)`} · n=${baseline.n||0}</span>
                <span class="info-tooltip-sm" style="cursor:help;">i<span class="info-tooltip-text" style="text-align:left; min-width:300px;"><b>Normal confidence</b><br>YOLO returns a score 0.0–1.0 for each detection — how sure the model is it found this class.<br><br><b>Formula:</b> percentile_cont over confidence values, last 7 days, this class.<br><br><b>p5</b> = the noise floor (only 5% of detections are below this).<br><b>p50</b> = median; this is what's "typical".<br><b>p95</b> = the strong-detection band.<br><br><b>Tune Min conf above p5</b> to filter noise, but well below p50 so you keep real detections.</span></span>
            </div>
            ${baseline.by_camera && Object.keys(baseline.by_camera).length > 0 ? `<div style="font-size: 10px; color: var(--text-secondary); margin-bottom: 6px; padding: 3px 6px; background: rgba(15,23,42,0.25); border-radius: 3px; line-height: 1.6;" title="Per-camera percentiles. p5 · p50 · p95.">
                ${Object.entries(baseline.by_camera).sort((a,b)=>a[0].localeCompare(b[0],undefined,{numeric:true})).map(([cam, c]) => `<span style="display:inline-block; margin-right:8px;">cam ${cam}: ${(c.p5*100).toFixed(0)} · <b>${(c.p50*100).toFixed(0)}</b> · ${(c.p95*100).toFixed(0)}% n=${c.n>=1000 ? (c.n/1000).toFixed(1)+'k' : c.n}</span>`).join('')}
            </div>` : ''}` : ''}
            ${isColorE && drift && drift.L ? `<div style="display:flex; align-items:center; gap:4px; font-size: 11px; color: var(--text-secondary); margin-bottom: 6px; padding: 3px 6px; background: rgba(15,23,42,0.4); border-radius: 3px;">
                <span style="flex:1;">🎨 color &nbsp; L: ${drift.L.p5.toFixed(0)} · <b>${drift.L.p50.toFixed(0)}</b> · ${drift.L.p95.toFixed(0)} &nbsp; a: ${drift.a.p5.toFixed(0)} · <b>${drift.a.p50.toFixed(0)}</b> · ${drift.a.p95.toFixed(0)} &nbsp; b: ${drift.b.p5.toFixed(0)} · <b>${drift.b.p50.toFixed(0)}</b> · ${drift.b.p95.toFixed(0)} &nbsp; (p5–p50–p95) · n=${drift.n>=1000?(drift.n/1000).toFixed(1)+'k':drift.n}</span>
                <span class="info-tooltip-sm" style="cursor:help;">i<span class="info-tooltip-text" style="text-align:left; min-width:360px;"><b>CIELAB color (L*a*b*)</b> — absolute color coordinates, sampled from the center 50% of each detection's bbox to skip edge artifacts.<br><br><b>L* (lightness):</b> 0 = pure black, 100 = pure white.<br><b>a*:</b> negative = green, positive = red.<br><b>b*:</b> negative = blue, positive = yellow.<br><br><b>Formula:</b> mean BGR → cv2.cvtColor(BGR_to_LAB) → normalize to CIE scale → percentile_cont per channel.<br><br><b>Drift diagnosis:</b><br>• L moving = exposure change, dye fade, lighting brightness<br>• a moving = green/red dye shift<br>• b moving = blue/yellow dye, lighting color temperature</span></span>
            </div>
            ${drift.E ? `<div style="display:flex; align-items:center; gap:4px; font-size: 11px; color: var(--text-secondary); margin-bottom: 6px; padding: 3px 6px; background: rgba(15,23,42,0.4); border-radius: 3px;">
                <span style="flex:1;">✴ E (abs): ${drift.E.p5.toFixed(1)} · <b>${drift.E.p50.toFixed(1)}</b> · ${drift.E.p95.toFixed(1)} (p5–p50–p95)</span>
                <span class="info-tooltip-sm" style="cursor:help;">i<span class="info-tooltip-text" style="text-align:left; min-width:340px;"><b>E — absolute color magnitude</b>, a single scalar per detection.<br><br><b>Formula:</b> E = √(L² + a² + b²).<br>Computed at detection time and stored on the detection (no moving baseline).<br><br><b>Use in ejection procedures:</b><br>• <code>E_greater</code> with field <code>E</code> — e.g. eject when E &gt; 100<br>• <code>E_less</code> with field <code>E</code> — e.g. eject when E &lt; 30<br>• <code>E_between</code> with <code>E_min</code> + <code>E_max</code> — keep only inside the band<br><br>Pair the p5/p50/p95 above with the thresholds: setting E_less ≈ p5 and E_greater ≈ p95 catches both ends of color drift in one rule.</span></span>
            </div>` : ''}
            ${drift.by_camera && Object.keys(drift.by_camera).length > 0 ? `<div style="font-size: 10px; color: var(--text-secondary); margin-bottom: 6px; padding: 3px 6px; background: rgba(15,23,42,0.25); border-radius: 3px; line-height: 1.6;" title="Per-camera CIELAB. Compare same channel across cameras to spot per-camera lighting / lens / dye-batch issues.">
                ${Object.entries(drift.by_camera).sort((a,b)=>a[0].localeCompare(b[0],undefined,{numeric:true})).map(([cam, c]) => `<div>cam ${cam}: L ${c.L.p5.toFixed(0)}·<b>${c.L.p50.toFixed(0)}</b>·${c.L.p95.toFixed(0)} · a ${c.a.p5.toFixed(0)}·<b>${c.a.p50.toFixed(0)}</b>·${c.a.p95.toFixed(0)} · b ${c.b.p5.toFixed(0)}·<b>${c.b.p50.toFixed(0)}</b>·${c.b.p95.toFixed(0)} · n=${c.n>=1000?(c.n/1000).toFixed(1)+'k':c.n}</div>`).join('')}
            </div>` : ''}
            ` : (isColorE ? `<div style="font-size: 10px; color: var(--text-secondary); margin-bottom: 6px; padding: 3px 6px; background: rgba(15,23,42,0.25); border-radius: 3px; font-style: italic;">🎨 ColorE on — gathering CIELAB samples (5+ detections needed). Stats appear here once data flows.</div>` : '')}
            ${isArea && areaSt ? `<div style="display:flex; align-items:center; gap:4px; font-size: 11px; color: var(--text-secondary); margin-bottom: 6px; padding: 3px 6px; background: rgba(15,23,42,0.4); border-radius: 3px;">
                <span style="flex:1;">📐 area: ${_fmtArea(areaSt.p5)} · <b>${_fmtArea(areaSt.p50)}</b> · ${_fmtArea(areaSt.p95)} px² (p5–p50–p95) · n=${areaSt.n>=1000?(areaSt.n/1000).toFixed(1)+'k':areaSt.n}</span>
                <span class="info-tooltip-sm" style="cursor:help;">i<span class="info-tooltip-text" style="text-align:left; min-width:300px;"><b>Bounding-box area</b> in pixels² (px²).<br><br><b>Formula:</b> area = (xmax − xmin) × (ymax − ymin).<br>Stored on the detection at inference time when ColorE/Area is on for this class (otherwise computed on-the-fly from the bbox).<br><br><b>p5</b> = smallest "typical" detection.<br><b>p50</b> = median size.<br><b>p95</b> = largest "typical" detection.<br><br><b>Outlier reading:</b><br>• Far below p5 → likely false positive (model fired on noise — raise Min conf)<br>• Far above p95 → two objects merged into one bbox, or yarn/material issue worth inspecting</span></span>
            </div>
            ${areaSt.by_camera && Object.keys(areaSt.by_camera).length > 0 ? `<div style="font-size: 10px; color: var(--text-secondary); margin-bottom: 6px; padding: 3px 6px; background: rgba(15,23,42,0.25); border-radius: 3px; line-height: 1.6;" title="Per-camera bbox area p5/p50/p95. Use when one camera frames objects differently (closer / farther / cropped).">
                ${Object.entries(areaSt.by_camera).sort((a,b)=>a[0].localeCompare(b[0],undefined,{numeric:true})).map(([cam, c]) => `<span style="display:inline-block; margin-right:8px;">cam ${cam}: ${_fmtArea(c.p5)} · <b>${_fmtArea(c.p50)}</b> · ${_fmtArea(c.p95)} n=${c.n>=1000?(c.n/1000).toFixed(1)+'k':c.n}</span>`).join('')}
            </div>` : ''}
            ` : (isArea ? `<div style="font-size: 10px; color: var(--text-secondary); margin-bottom: 6px; padding: 3px 6px; background: rgba(15,23,42,0.25); border-radius: 3px; font-style: italic;">📐 Area on — gathering samples (5+ detections needed). Stats appear here once data flows.</div>` : '')}
            <div style="display: flex; gap: 6px; align-items: center;">
                <select onchange="updateObjectBeepSound('${safeName}', this.value)" style="flex: 1; padding: 3px 6px; background: rgba(30,41,59,0.6); color: var(--text-primary); border: 1px solid rgba(51,65,85,0.6); border-radius: 4px; font-size: 11px;">
                    <option value="sine" ${beepSound === 'sine' ? 'selected' : ''}>🔔 Sine</option>
                    <option value="square" ${beepSound === 'square' ? 'selected' : ''}>📢 Square</option>
                    <option value="sawtooth" ${beepSound === 'sawtooth' ? 'selected' : ''}>🎺 Sawtooth</option>
                    <option value="triangle" ${beepSound === 'triangle' ? 'selected' : ''}>🎵 Triangle</option>
                </select>
                <button onclick="testObjectBeep('${safeName}')" style="padding: 3px 8px; background: var(--secondary-color); color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 11px;">▶</button>
            </div>
        `;

        listEl.appendChild(card);
    });
}

// Per-class server sync: mirror localStorage toggles into /api/audio_settings
// so an AI agent (or another browser) can read/change them. Fire-and-forget;
// failure to sync logs but doesn't block the UI.
async function syncObjectAudioToServer(objectName) {
    const payload = {
        class_name: objectName,
        show: audioSettings.showObjects[objectName] === true,
        narrate: !!audioSettings.narrateObjects[objectName],
        beep: !!audioSettings.enabledObjects[objectName],
        min_confidence: (audioSettings.objectConfidence[objectName] ?? 1) / 100,  // UI uses 0-100, server uses 0-1
        severity: audioSettings.severityObjects?.[objectName] ?? 0,  // 3.21.12 — 0-100, per-class impact weight
        role:    audioSettings.roleObjects?.[objectName] || 'context', // 4.0.26 — context | defect | parent | marker
        color_e: !!audioSettings.colorEObjects?.[objectName],         // 3.22.2 — track CIELAB ΔE drift over time
        area:    !!audioSettings.areaObjects?.[objectName],           // 3.22.3 — show bbox-area percentiles on the card
    };
    try {
        await fetch('/api/audio_settings', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
    } catch (e) { console.error('audio_settings POST failed:', e); }
}

// Toggle show/hide bounding box for object
function toggleObjectShow(objectName) {
    const current = audioSettings.showObjects[objectName] !== false;
    audioSettings.showObjects[objectName] = !current;
    saveAudioSettings();
    updateObjectsList();
    syncObjectAudioToServer(objectName);
}

// Toggle narrate for object
function toggleObjectNarrate(objectName) {
    audioSettings.narrateObjects[objectName] = !audioSettings.narrateObjects[objectName];
    saveAudioSettings();
    updateObjectsList();
    syncObjectAudioToServer(objectName);
}

// Update confidence threshold for object
function updateObjectConfidence(objectName, value) {
    audioSettings.objectConfidence[objectName] = Math.max(0, Math.min(100, parseFloat(value) || 1));
    saveAudioSettings();
    syncObjectAudioToServer(objectName);
}

// 3.21.12 — Update per-class severity weight (0-100). Used for impact-score
// computation in chart endpoints and CSV export.
function updateObjectSeverity(objectName, value) {
    if (!audioSettings.severityObjects) audioSettings.severityObjects = {};
    audioSettings.severityObjects[objectName] = Math.max(0, Math.min(100, parseInt(value, 10) || 0));
    saveAudioSettings();
    syncObjectAudioToServer(objectName);
}
window.updateObjectSeverity = updateObjectSeverity;

// 4.0.26 — Update per-class semantic role.
// On change, re-render the cards so the severity input disables/enables to
// match the new role (Parent disables severity; everything else enables it).
// Setting role=Parent also forces severity to 0 so the impact-score numbers
// stay clean even if the operator had set a non-zero severity earlier.
function updateObjectRole(objectName, value) {
    const VALID = ['context', 'defect', 'parent', 'marker'];
    const role = VALID.includes(value) ? value : 'context';
    if (!audioSettings.roleObjects) audioSettings.roleObjects = {};
    audioSettings.roleObjects[objectName] = role;
    if (role === 'parent') {
        if (!audioSettings.severityObjects) audioSettings.severityObjects = {};
        audioSettings.severityObjects[objectName] = 0;
    }
    saveAudioSettings();
    syncObjectAudioToServer(objectName);
    if (typeof updateObjectsList === 'function') updateObjectsList();
}
window.updateObjectRole = updateObjectRole;

// On page load, pull server-side audio_settings so the UI matches what
// services/* actually uses (localStorage may be stale across browsers).
async function loadAudioSettingsFromServer() {
    try {
        const r = await fetch('/api/audio_settings');
        if (!r.ok) return;
        const data = await r.json();
        const server = data.audio_settings || {};
        if (!audioSettings.severityObjects) audioSettings.severityObjects = {};
        Object.keys(server).forEach(k => {
            const s = server[k];
            if (s.show !== undefined)    audioSettings.showObjects[k] = !!s.show;
            if (s.narrate !== undefined) audioSettings.narrateObjects[k] = !!s.narrate;
            if (s.beep !== undefined)    audioSettings.enabledObjects[k] = !!s.beep;
            if (s.min_confidence !== undefined) audioSettings.objectConfidence[k] = Math.round(s.min_confidence * 100);
            if (s.severity !== undefined) audioSettings.severityObjects[k] = parseInt(s.severity, 10) || 0;
            // 4.0.26 — per-class semantic role (context | defect | parent | marker)
            if (s.role !== undefined) {
                if (!audioSettings.roleObjects) audioSettings.roleObjects = {};
                audioSettings.roleObjects[k] = String(s.role);
            }
            // 3.22.2 — per-class ColorE (CIELAB ΔE tracking) toggle
            if (s.color_e !== undefined) {
                if (!audioSettings.colorEObjects) audioSettings.colorEObjects = {};
                audioSettings.colorEObjects[k] = s.color_e === true;
            }
            // 3.22.3 — per-class Area (bbox-area percentiles) display toggle
            if (s.area !== undefined) {
                if (!audioSettings.areaObjects) audioSettings.areaObjects = {};
                audioSettings.areaObjects[k] = s.area === true;
            }
        });
        saveAudioSettings();
        if (typeof updateObjectsList === 'function') updateObjectsList();
    } catch (e) { /* not fatal */ }
}
document.addEventListener('DOMContentLoaded', loadAudioSettingsFromServer);


// 3.22.2 — Per-class CIELAB ΔE drift. Reads /api/color_drift and stores by class.
// Refreshed on demand (e.g. when ColorE checkbox is toggled on).
async function loadColorDriftFromServer() {
    try {
        const r = await fetch('/api/color_drift?window=7d');
        if (!r.ok) return;
        const d = await r.json();
        audioSettings.colorDrift = d.classes || {};
        if (typeof updateObjectsList === 'function') updateObjectsList();
    } catch (e) { /* not fatal */ }
}
document.addEventListener('DOMContentLoaded', () => setTimeout(loadColorDriftFromServer, 1500));
window.loadColorDriftFromServer = loadColorDriftFromServer;


// 3.22.2 — Toggle per-class ColorE (ΔE drift tracking). Same shape as
// toggleObjectShow / toggleObjectNarrate / etc.
function toggleObjectColorE(objectName) {
    if (!audioSettings.colorEObjects) audioSettings.colorEObjects = {};
    audioSettings.colorEObjects[objectName] = !audioSettings.colorEObjects[objectName];
    saveAudioSettings();
    updateObjectsList();
    syncObjectAudioToServer(objectName);
    // Reload the drift baselines so the new class shows up once data flows
    setTimeout(loadColorDriftFromServer, 200);
}
window.toggleObjectColorE = toggleObjectColorE;


// 3.22.3 — Per-class bbox-area percentiles. Reads /api/area_stats and stores by class.
// Always loaded (cheap; no server-side gating) — the per-class card decides whether
// to display based on the Area checkbox.
async function loadAreaStatsFromServer() {
    try {
        const r = await fetch('/api/area_stats?window=7d');
        if (!r.ok) return;
        const d = await r.json();
        audioSettings.areaStats = d.classes || {};
        if (typeof updateObjectsList === 'function') updateObjectsList();
    } catch (e) { /* not fatal */ }
}
document.addEventListener('DOMContentLoaded', () => setTimeout(loadAreaStatsFromServer, 1700));
window.loadAreaStatsFromServer = loadAreaStatsFromServer;


// 3.22.3 — Toggle per-class Area display. Pure display preference: stats are already
// derivable from stored xmin/xmax/ymin/ymax, no extraction needed.
function toggleObjectArea(objectName) {
    if (!audioSettings.areaObjects) audioSettings.areaObjects = {};
    audioSettings.areaObjects[objectName] = !audioSettings.areaObjects[objectName];
    saveAudioSettings();
    updateObjectsList();
    syncObjectAudioToServer(objectName);
}
window.toggleObjectArea = toggleObjectArea;


// =====================================================================
// 3.25.2 — AI Severity suggester
// =====================================================================

let _severitySuggestions     = null;  // last class suggestions from /api/suggest_severities
let _severityProcSuggestions = null;  // 3.25.10 — last procedure suggestions (same response)

function openSeveritySuggestPanel() {
    const p = document.getElementById('severity-suggest-panel');
    if (p) p.style.display = 'block';
    const meta = document.getElementById('severity-suggest-meta');
    if (meta) meta.textContent = '— ask the AI to score every class AND every ejection procedure on a 0–100 severity scale based on counts, confidence, name keywords, rule shape, and your business context.';
    const results = document.getElementById('severity-suggest-results');
    if (results) results.innerHTML = '';
    document.getElementById('severity-apply-all-btn').disabled = true;
    document.getElementById('severity-apply-all-btn').style.opacity = '0.5';
}
window.openSeveritySuggestPanel = openSeveritySuggestPanel;

function closeSeveritySuggestPanel() {
    const p = document.getElementById('severity-suggest-panel');
    if (p) p.style.display = 'none';
}
window.closeSeveritySuggestPanel = closeSeveritySuggestPanel;

async function runSeveritySuggest() {
    const ctxEl = document.getElementById('severity-context-input');
    const runBtn = document.getElementById('severity-suggest-run-btn');
    const applyBtn = document.getElementById('severity-apply-all-btn');
    const statusEl = document.getElementById('severity-suggest-status');
    const resultsEl = document.getElementById('severity-suggest-results');
    if (!resultsEl) return;

    runBtn.disabled = true; runBtn.style.opacity = '0.5';
    statusEl.textContent = '🤔 thinking — this can take 20–60s depending on the model…';
    resultsEl.innerHTML = '';

    try {
        const r = await fetch('/api/suggest_severities', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                business_context: (ctxEl?.value || '').trim(),
                language: window.currentLang || 'en',
            }),
        });
        const d = await r.json();
        if (!r.ok) {
            // 3.25.2 — surface upstream provider errors clearly (rate limit, 500, etc.)
            const msg = d.upstream_error
                ? ('AI provider error: ' + d.upstream_error.slice(0, 200) + '  · ' + (d.hint || ''))
                : (d.error || ('Request failed (' + r.status + ')'));
            statusEl.textContent = '✗ ' + msg;
            return;
        }
        if (d.parse_error) {
            statusEl.textContent = '✗ AI response was not valid JSON. Raw preview: ' + (d.raw_preview || '').slice(0, 120);
            return;
        }
        _severitySuggestions     = d.suggestions || [];
        _severityProcSuggestions = d.procedure_suggestions || [];   // 3.25.10
        const nC = _severitySuggestions.length;
        const nP = _severityProcSuggestions.length;
        statusEl.textContent = `✓ ${nC} class + ${nP} procedure suggestion${(nC + nP) === 1 ? '' : 's'} from ${d.model || 'AI'}.`;
        _renderSeveritySuggestions(_severitySuggestions, _severityProcSuggestions);
        if (nC > 0 || nP > 0) {
            applyBtn.disabled = false;
            applyBtn.style.opacity = '1';
        }
    } catch (e) {
        statusEl.textContent = '✗ Network error: ' + (e.message || e);
    } finally {
        runBtn.disabled = false; runBtn.style.opacity = '1';
    }
}
window.runSeveritySuggest = runSeveritySuggest;

function _tierColor(tier) {
    switch ((tier || '').toUpperCase()) {
        case 'CRITICAL': return '#f87171';
        case 'SERIOUS':  return '#fb923c';
        case 'MODERATE': return '#fcd34d';
        case 'COSMETIC': return '#86efac';
        default:         return '#94a3b8';   // NONE / unknown
    }
}

function _renderSeveritySuggestions(rows, procRows) {
    const el = document.getElementById('severity-suggest-results');
    if (!el) return;
    rows     = rows     || [];
    procRows = procRows || [];
    if (!rows.length && !procRows.length) {
        el.innerHTML = '<i>No suggestions returned.</i>';
        return;
    }
    // 3.25.10 — shared row-renderer for both tables. `kind` decides the label
    // column (Class vs Procedure) and which apply handler to call.
    function _section(title, items, kind) {
        if (!items.length) return '';
        const labelCol = kind === 'procedure' ? 'Procedure' : 'Class';
        const handler  = kind === 'procedure' ? 'applyOneProcedureSeveritySuggestion' : 'applyOneSeveritySuggestion';
        const header = `
            <tr style="color:#94a3b8; background:rgba(15,23,42,0.55); position:sticky; top:0;">
                <th style="text-align:left; padding:5px 8px;">${labelCol}</th>
                <th style="text-align:center; padding:5px 8px;">Current</th>
                <th style="text-align:center; padding:5px 8px;">Suggested</th>
                <th style="text-align:center; padding:5px 8px;">Tier</th>
                <th style="text-align:left; padding:5px 8px;">Reason</th>
                <th style="text-align:center; padding:5px 8px;">Apply</th>
            </tr>`;
        const body = items.map((r, idx) => {
            const name = (kind === 'procedure' ? r.procedure : r.class) || '';
            const same = r.current_severity === r.suggested_severity;
            const arrow = r.suggested_severity > r.current_severity ? '↑' :
                          r.suggested_severity < r.current_severity ? '↓' : '=';
            const arrowColor = arrow === '↑' ? '#86efac' : arrow === '↓' ? '#fcd34d' : '#94a3b8';
            const nameDisp = (kind === 'procedure') ? `⏏ ${name}` : name;
            return `
                <tr style="border-top:1px solid rgba(51,65,85,0.4);" data-${kind}="${name}">
                    <td style="padding:4px 8px; font-weight:600;">${nameDisp}</td>
                    <td style="text-align:center; padding:4px 8px;">${r.current_severity}</td>
                    <td style="text-align:center; padding:4px 8px; font-weight:700;">
                        ${r.suggested_severity} <span style="color:${arrowColor}; font-weight:400;">${arrow}</span>
                    </td>
                    <td style="text-align:center; padding:4px 8px; color:${_tierColor(r.tier)}; font-weight:600;">${r.tier || '?'}</td>
                    <td style="padding:4px 8px; color:var(--text-secondary); font-size:11px;">${(r.reason || '').replace(/</g, '&lt;')}</td>
                    <td style="text-align:center; padding:4px 8px;">
                        <button onclick="${handler}(${idx})" ${same ? 'disabled' : ''} style="background:${same?'rgba(51,65,85,0.4)':'var(--primary-color)'}; color:#fff; border:none; padding:3px 9px; cursor:${same?'default':'pointer'}; font-size:11px; border-radius:3px; font-weight:600;">${same ? '=' : '✓'}</button>
                    </td>
                </tr>`;
        }).join('');
        return `
            <div style="margin-bottom:8px; font-size:11px; color:#a7f3d0; font-weight:600; letter-spacing:0.4px; text-transform:uppercase;">${title}</div>
            <table style="width:100%; border-collapse:collapse; margin-bottom:14px; color: var(--text-primary);">${header}${body}</table>`;
    }
    el.innerHTML =
        _section(`Detection classes (${rows.length})`, rows, 'class') +
        _section(`Ejection procedures (${procRows.length})`, procRows, 'procedure');
}

async function applyOneSeveritySuggestion(idx) {
    if (!_severitySuggestions || !_severitySuggestions[idx]) return;
    const row = _severitySuggestions[idx];
    await _postSeverityUpdates([{ class: row.class, severity: row.suggested_severity }]);
    // Reflect locally
    audioSettings.severityObjects = audioSettings.severityObjects || {};
    audioSettings.severityObjects[row.class] = row.suggested_severity;
    row.current_severity = row.suggested_severity;
    _renderSeveritySuggestions(_severitySuggestions, _severityProcSuggestions);
    if (typeof updateObjectsList === 'function') updateObjectsList();
}
window.applyOneSeveritySuggestion = applyOneSeveritySuggestion;

// 3.25.10 — apply a single procedure-severity suggestion. Same shape as the class
// path but hits /api/apply_procedure_severities and reflects into the in-memory
// procedures list so the Process tab updates without a reload.
async function applyOneProcedureSeveritySuggestion(idx) {
    if (!_severityProcSuggestions || !_severityProcSuggestions[idx]) return;
    const row = _severityProcSuggestions[idx];
    await _postProcedureSeverityUpdates([{ procedure: row.procedure, severity: row.suggested_severity }]);
    // Reflect locally so the Process tab severity input updates without a reload.
    if (Array.isArray(procedures)) {
        const p = procedures.find(pp => pp && pp.name === row.procedure);
        if (p) p.severity = row.suggested_severity;
    }
    row.current_severity = row.suggested_severity;
    _renderSeveritySuggestions(_severitySuggestions, _severityProcSuggestions);
    if (typeof renderProcedures === 'function') renderProcedures();
}
window.applyOneProcedureSeveritySuggestion = applyOneProcedureSeveritySuggestion;

async function applyAllSeveritySuggestions() {
    const classUpdates = (_severitySuggestions || [])
        .filter(r => r.current_severity !== r.suggested_severity)
        .map(r => ({ class: r.class, severity: r.suggested_severity }));
    const procUpdates  = (_severityProcSuggestions || [])
        .filter(r => r.current_severity !== r.suggested_severity)
        .map(r => ({ procedure: r.procedure, severity: r.suggested_severity }));
    if (!classUpdates.length && !procUpdates.length) {
        document.getElementById('severity-suggest-status').textContent = 'Nothing to change.';
        return;
    }
    let appliedCls = 0, appliedProc = 0;
    if (classUpdates.length) {
        const r = await _postSeverityUpdates(classUpdates);
        appliedCls = r?.applied ?? classUpdates.length;
        audioSettings.severityObjects = audioSettings.severityObjects || {};
        _severitySuggestions.forEach(r => {
            audioSettings.severityObjects[r.class] = r.suggested_severity;
            r.current_severity = r.suggested_severity;
        });
        if (typeof updateObjectsList === 'function') updateObjectsList();
    }
    if (procUpdates.length) {
        const r = await _postProcedureSeverityUpdates(procUpdates);
        appliedProc = r?.applied ?? procUpdates.length;
        if (Array.isArray(procedures)) {
            (_severityProcSuggestions || []).forEach(row => {
                const p = procedures.find(pp => pp && pp.name === row.procedure);
                if (p) p.severity = row.suggested_severity;
                row.current_severity = row.suggested_severity;
            });
        } else {
            (_severityProcSuggestions || []).forEach(row => { row.current_severity = row.suggested_severity; });
        }
        if (typeof renderProcedures === 'function') renderProcedures();
    }
    _renderSeveritySuggestions(_severitySuggestions, _severityProcSuggestions);
    document.getElementById('severity-suggest-status').textContent =
        `✓ Applied ${appliedCls} class + ${appliedProc} procedure update${(appliedCls + appliedProc) === 1 ? '' : 's'}.`;
}
window.applyAllSeveritySuggestions = applyAllSeveritySuggestions;

// 3.25.2 — "🚀 Auto-tune all" — runs suggest_severities, then applies every
// non-equal suggestion in one go. Skips the review step. Operator gets a
// confirmation toast at the end.
async function autoTuneAllSeverities() {
    if (!confirm('Auto-tune ALL classes AND ejection procedures? The AI will see your shipment score history + ejection event counts and recalibrate every Severity value. This overwrites your existing values.')) return;
    const btn = document.getElementById('severity-autotune-btn');
    const statusEl = document.getElementById('severity-suggest-status');
    if (btn) { btn.disabled = true; btn.style.opacity = '0.6'; }
    if (statusEl) statusEl.textContent = '🤔 thinking — AI analyzing classes + procedures + shipment score history…';
    try {
        await runSeveritySuggest();
        const nC = (_severitySuggestions || []).length;
        const nP = (_severityProcSuggestions || []).length;
        if (!nC && !nP) return;
        await applyAllSeveritySuggestions();
        if (statusEl) statusEl.textContent = `✓ Auto-tuned ${nC} classes + ${nP} procedures.`;
    } catch (e) {
        if (statusEl) statusEl.textContent = '✗ Auto-tune failed: ' + (e.message || e);
    } finally {
        if (btn) { btn.disabled = false; btn.style.opacity = '1'; }
    }
}
window.autoTuneAllSeverities = autoTuneAllSeverities;


async function _postSeverityUpdates(updates) {
    try {
        const r = await fetch('/api/apply_severities', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ updates }),
        });
        return await r.json();
    } catch (e) {
        return { error: e.message };
    }
}

// 3.25.10 — companion to _postSeverityUpdates for ejection procedures.
async function _postProcedureSeverityUpdates(updates) {
    try {
        const r = await fetch('/api/apply_procedure_severities', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ updates }),
        });
        return await r.json();
    } catch (e) {
        return { error: e.message };
    }
}


// 3.23.1 — "🤔" chip on each per-class card. Asks /api/why with mode="class"
// so the AI explains the CURRENT behaviour of this class instead of a
// specific dot. Result lands in the small panel below the card header.
async function askWhyForClass(className, btn) {
    const safe = className.replace(/[^a-zA-Z0-9]/g, '_');
    const panel = document.querySelector('.per-class-why-' + safe);
    if (!panel) return;
    panel.style.display = 'block';
    panel.innerHTML = '<span style="opacity:0.7;">🤔 thinking…</span>';
    if (btn) { btn.disabled = true; btn.style.opacity = '0.6'; }
    try {
        const r = await fetch('/api/why', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mode: 'class',
                metric: className,
                window_seconds: 3600,
                language: window.currentLang || 'en',
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
window.askWhyForClass = askWhyForClass;


// 3.22.3 — Compact pixel-area formatter. Operators read "12k" faster than "12,450".
function _fmtArea(px2) {
    const n = Number(px2) || 0;
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + 'M';
    if (n >= 10_000)    return (n / 1000).toFixed(0) + 'k';
    if (n >= 1000)      return (n / 1000).toFixed(1) + 'k';
    return String(Math.round(n));
}
window._fmtArea = _fmtArea;

// 3.21.12 — Fetch per-class confidence baselines (auto-learned p50/p95 over
// last 7 days of stored detections). Surfaces in the per-class card as a
// read-only badge to help operators set min_conf and judge anomalies.
async function loadConfBaselinesFromServer() {
    try {
        const r = await fetch('/api/conf_baselines');
        if (!r.ok) return;
        const d = await r.json();
        audioSettings.confBaselines = d.baselines || {};
        if (typeof updateObjectsList === 'function') updateObjectsList();
    } catch (e) { /* not fatal — baseline is optional UI hint */ }
}
document.addEventListener('DOMContentLoaded', () => setTimeout(loadConfBaselinesFromServer, 800));
window.loadConfBaselinesFromServer = loadConfBaselinesFromServer;


// 3.21.24 — Active-class lookup. Drives the Process tab filter bar's
// "Show only active" toggle + busy-first sort. The selected window
// comes from the #per-object-active-window dropdown.
async function loadActiveClasses() {
    try {
        const winEl = document.getElementById('per-object-active-window');
        const win = winEl ? winEl.value : '1h';
        const r = await fetch('/api/active_classes?window=' + encodeURIComponent(win));
        if (!r.ok) return;
        const d = await r.json();
        audioSettings._activeCounts = d.counts || {};
        audioSettings._activeNames  = d.names  || [];   // sorted by total count desc
        audioSettings._activeWindow = d.window || win;
        if (typeof updateObjectsList === 'function') updateObjectsList();
    } catch (e) { /* not fatal */ }
}
document.addEventListener('DOMContentLoaded', () => setTimeout(loadActiveClasses, 900));
window.loadActiveClasses = loadActiveClasses;


// 3.21.24 — Set the per-class Min-Conf slider to that camera's p5 baseline
// (or the overall p5 when no camera is specified). One-click "calibrate
// against historical noise floor — anything above this is real".
function suggestMinConfFromBaseline(objectName, camera) {
    const b = audioSettings.confBaselines?.[objectName];
    if (!b) return;
    let p5;
    if (camera != null && b.by_camera && b.by_camera[String(camera)]) {
        p5 = b.by_camera[String(camera)].p5;
    } else {
        p5 = b.p5;
    }
    if (typeof p5 !== 'number') return;
    const ui = Math.max(0, Math.min(100, Math.round(p5 * 100)));
    audioSettings.objectConfidence[objectName] = ui;
    saveAudioSettings();
    updateObjectsList();
    syncObjectAudioToServer(objectName);
}
window.suggestMinConfFromBaseline = suggestMinConfFromBaseline;

// Toggle DB persistence for object. POSTs to server because detection.py reads
// this server-side to gate writes to inference_results. Default OFF.
async function toggleObjectStore(objectName) {
    const next = !audioSettings.storeObjects[objectName];
    audioSettings.storeObjects[objectName] = next;
    saveAudioSettings();           // mirror locally so UI doesn't snap back on re-render
    updateObjectsList();
    try {
        await fetch('/api/store_objects', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({class_name: objectName, store: next})
        });
    } catch (e) {
        console.error('store_objects POST failed:', e);
    }
}

// On page load, pull server-side store flags so the UI matches what detection.py
// actually uses for gating (localStorage may be stale across browsers/machines).
async function loadStoreObjectsFromServer() {
    try {
        const r = await fetch('/api/store_objects');
        if (!r.ok) return;
        const data = await r.json();
        const serverMap = data.store_objects || {};
        Object.keys(serverMap).forEach(k => { audioSettings.storeObjects[k] = !!serverMap[k]; });
        saveAudioSettings();
    } catch (e) { /* not fatal */ }
}
document.addEventListener('DOMContentLoaded', loadStoreObjectsFromServer);

// ============== EJECTION PROCEDURES ==============
let procedures = [];

function addProcedure() {
    procedures.push({
        id: 'proc_' + Date.now(),
        name: 'New Procedure',
        enabled: true,
        logic: 'any',
        rules: [],
        // 3.25.8 — Severity (0–100) for ejection events. Each stored ejection
        // adds (severity / 100) to the shipment-quality impact, same scale as
        // per-class detection severity. Default 0 = doesn't affect score.
        severity: 0,
    });
    renderProcedures();
}

function removeProcedure(procId) {
    procedures = procedures.filter(p => p.id !== procId);
    renderProcedures();
}

function addRule(procId) {
    const proc = procedures.find(p => p.id === procId);
    if (!proc) return;
    const firstClass = detectedObjectClasses.size > 0 ? Array.from(detectedObjectClasses).sort()[0] : '';
    proc.rules.push({ object: firstClass, condition: 'count_equals', min_confidence: 30, count: 1, area: 10000, max_delta_e: 5.0, reference_mode: 'previous' });
    renderProcedures();
}

function removeRule(procId, ruleIndex) {
    const proc = procedures.find(p => p.id === procId);
    if (!proc) return;
    proc.rules.splice(ruleIndex, 1);
    renderProcedures();
}

function updateProcedureField(procId, field, value) {
    const proc = procedures.find(p => p.id === procId);
    if (!proc) return;
    proc[field] = value;
    if (field === 'enabled' || field === 'cameras') renderProcedures();
}

// Per-procedure DB persistence of ejection events. Persists immediately (like the
// per-class Store) so charts pick it up without a separate Save click.
function toggleProcedureStore(procId, checked) {
    const proc = procedures.find(p => p.id === procId);
    if (!proc) return;
    proc.store = checked;
    renderProcedures();
    saveProcedures();
}

function updateRuleField(procId, ruleIndex, field, value) {
    const proc = procedures.find(p => p.id === procId);
    if (!proc || !proc.rules[ruleIndex]) return;
    if (field === 'min_confidence' || field === 'count' || field === 'area') {
        proc.rules[ruleIndex][field] = Math.max(0, parseInt(value) || 0);
        if (field === 'min_confidence') proc.rules[ruleIndex][field] = Math.min(100, proc.rules[ruleIndex][field]);
    } else if (field === 'max_delta_e') {
        proc.rules[ruleIndex][field] = Math.max(0, parseFloat(value) || 5.0);
    } else {
        proc.rules[ruleIndex][field] = value;
    }
    if (field === 'condition' || field === 'reference_mode') renderProcedures();
}

async function captureColorReference(className) {
    try {
        const resp = await fetch('/api/color-reference/' + encodeURIComponent(className), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ capture: true })
        });
        const data = await resp.json();
        if (data.success) {
            alert('Fixed reference captured for "' + className + '": L*=' + data.lab[0] + ', a*=' + data.lab[1] + ', b*=' + data.lab[2]);
        } else {
            alert('Failed: ' + (data.error || 'Unknown error'));
        }
    } catch (e) {
        alert('Error capturing color reference: ' + e.message);
    }
}

function renderProcedures() {
    const container = document.getElementById('procedures-list');
    if (!container) return;

    if (procedures.length === 0) {
        container.innerHTML = '<div style="padding: 15px; background: rgba(15, 23, 42, 0.5); border-radius: 6px; text-align: center; color: var(--text-secondary); font-style: italic;">No ejection procedures configured.</div>';
        return;
    }

    container.innerHTML = procedures.map(proc => {
        const rulesHtml = proc.rules.map((rule, ri) => {
            const cond = rule.condition || 'count_equals';
            const isColor = cond === 'color_delta';
            const isArea = cond.startsWith('area_');
            const isCount = !isColor && !isArea;
            const countInput = !isCount ? '' : `
                <input type="number" value="${rule.count != null ? rule.count : 1}" min="0" step="1"
                    onchange="updateRuleField('${proc.id}', ${ri}, 'count', this.value)"
                    title="Number of detected objects to compare against"
                    style="width: 45px; padding: 3px 5px; background: rgba(30,41,59,0.6); color: var(--text-primary); border: 1px solid rgba(51,65,85,0.6); border-radius: 4px; font-size: 11px; text-align: center;">`;
            const areaInput = !isArea ? '' : `
                <input type="number" value="${rule.area != null ? rule.area : 10000}" min="0" step="100"
                    onchange="updateRuleField('${proc.id}', ${ri}, 'area', this.value)"
                    title="Area threshold in pixels (width × height of bounding box)"
                    style="width: 65px; padding: 3px 5px; background: rgba(30,41,59,0.6); color: var(--text-primary); border: 1px solid rgba(51,65,85,0.6); border-radius: 4px; font-size: 11px; text-align: center;">
                <span style="font-size: 10px; color: var(--text-secondary);">px</span>`;
            const colorControls = !isColor ? '' : `
                <input type="number" value="${rule.max_delta_e != null ? rule.max_delta_e : 5.0}" min="0" step="0.5"
                    onchange="updateRuleField('${proc.id}', ${ri}, 'max_delta_e', this.value)"
                    title="ΔE threshold — color difference limit. Values: <1 imperceptible, 2-3 noticeable, >5 clearly different"
                    style="width: 50px; padding: 3px 5px; background: rgba(30,41,59,0.6); color: var(--text-primary); border: 1px solid rgba(51,65,85,0.6); border-radius: 4px; font-size: 11px; text-align: center;">
                <select onchange="updateRuleField('${proc.id}', ${ri}, 'reference_mode', this.value)"
                    title="Color reference mode: Previous = last product, Average = rolling avg of last 20, Fixed = user-captured golden sample"
                    style="padding: 3px 6px; background: rgba(30,41,59,0.6); color: var(--text-primary); border: 1px solid rgba(51,65,85,0.6); border-radius: 4px; font-size: 11px;">
                    <option value="previous" ${(rule.reference_mode || 'previous') === 'previous' ? 'selected' : ''}>vs Previous</option>
                    <option value="running_avg" ${rule.reference_mode === 'running_avg' ? 'selected' : ''}>vs Average</option>
                    <option value="fixed" ${rule.reference_mode === 'fixed' ? 'selected' : ''}>vs Fixed</option>
                </select>
                ${rule.reference_mode === 'fixed' ? `
                    <button onclick="captureColorReference('${rule.object}')"
                        style="padding: 2px 6px; background: rgba(34,197,94,0.3); color: #22c55e; border: 1px solid rgba(34,197,94,0.4); border-radius: 4px; cursor: pointer; font-size: 10px;"
                        title="Capture current color as fixed reference">Capture</button>
                ` : ''}`;
            return `
            <div style="display: flex; gap: 6px; align-items: center; padding: 6px; background: rgba(15, 23, 42, 0.4); border-radius: 4px; flex-wrap: wrap;">
                <select onchange="updateRuleField('${proc.id}', ${ri}, 'object', this.value)"
                    title="Target object class to monitor"
                    style="padding: 3px 6px; background: rgba(30,41,59,0.6); color: var(--text-primary); border: 1px solid rgba(51,65,85,0.6); border-radius: 4px; font-size: 11px; min-width: 100px;">
                    ${(() => {
                        // Always include the rule's saved object even if no live detection
                        // has populated detectedObjectClasses for it yet. Otherwise the
                        // dropdown renders empty on a fresh page-load (rules look unbound).
                        const names = new Set(detectedObjectClasses);
                        if (rule.object) names.add(rule.object);
                        return Array.from(names).sort().map(name =>
                            '<option value="' + name + '" ' + (rule.object === name ? 'selected' : '') + '>' + name + '</option>'
                        ).join('');
                    })()}
                </select>
                <select onchange="updateRuleField('${proc.id}', ${ri}, 'condition', this.value)"
                    title="Condition type: Count = number of objects, Area = bounding box size in pixels, Color ΔE = color difference from reference"
                    style="padding: 3px 6px; background: rgba(30,41,59,0.6); color: var(--text-primary); border: 1px solid rgba(51,65,85,0.6); border-radius: 4px; font-size: 11px;">
                    <option value="count_equals" ${cond === 'count_equals' ? 'selected' : ''}>Count =</option>
                    <option value="count_greater" ${cond === 'count_greater' ? 'selected' : ''}>Count &gt;</option>
                    <option value="count_less" ${cond === 'count_less' ? 'selected' : ''}>Count &lt;</option>
                    <option value="area_greater" ${cond === 'area_greater' ? 'selected' : ''}>Area &gt;</option>
                    <option value="area_less" ${cond === 'area_less' ? 'selected' : ''}>Area &lt;</option>
                    <option value="area_equals" ${cond === 'area_equals' ? 'selected' : ''}>Area =</option>
                    <option value="color_delta" ${cond === 'color_delta' ? 'selected' : ''}>Color ΔE &gt;</option>
                </select>
                ${countInput}
                ${areaInput}
                ${colorControls}
                <span style="font-size: 11px; color: var(--text-secondary);">min:</span>
                <input type="number" value="${rule.min_confidence}" min="0" max="100" step="1"
                    onchange="updateRuleField('${proc.id}', ${ri}, 'min_confidence', this.value)"
                    title="Minimum detection confidence (%) — only detections above this threshold are considered"
                    style="width: 50px; padding: 3px 5px; background: rgba(30,41,59,0.6); color: var(--text-primary); border: 1px solid rgba(51,65,85,0.6); border-radius: 4px; font-size: 11px; text-align: center;">
                <span style="font-size: 11px; color: var(--text-secondary);">%</span>
                <button onclick="removeRule('${proc.id}', ${ri})"
                    style="padding: 2px 6px; background: rgba(239,68,68,0.3); color: #ef4444; border: 1px solid rgba(239,68,68,0.4); border-radius: 4px; cursor: pointer; font-size: 11px;"
                    title="Remove rule">X</button>
            </div>`;
        }).join('');

        return `
            <div style="padding: 12px; background: rgba(51, 65, 85, 0.4); border-radius: 6px; margin-bottom: 8px; border: 1px solid ${proc.enabled ? 'rgba(239,68,68,0.4)' : 'rgba(51,65,85,0.6)'};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; gap: 8px; flex-wrap: wrap;">
                    <input type="text" value="${proc.name}"
                        onchange="updateProcedureField('${proc.id}', 'name', this.value)"
                        title="Procedure name — a descriptive label for this ejection rule set"
                        style="flex: 1; min-width: 120px; padding: 4px 8px; background: rgba(30,41,59,0.6); color: var(--text-primary); border: 1px solid rgba(51,65,85,0.6); border-radius: 4px; font-size: 13px; font-weight: 600;">
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <label style="display: flex; align-items: center; gap: 4px; font-size: 12px; color: ${proc.enabled ? '#ef4444' : 'var(--text-secondary)'};"
                            title="Enable or disable this procedure — disabled procedures are ignored during evaluation">
                            <input type="checkbox" ${proc.enabled ? 'checked' : ''}
                                onchange="updateProcedureField('${proc.id}', 'enabled', this.checked)"
                                style="width: 14px; height: 14px; cursor: pointer;">
                            Enabled
                        </label>
                        <label style="display: flex; align-items: center; gap: 4px; font-size: 12px; color: ${proc.store ? '#10b981' : 'var(--text-secondary)'};"
                            title="Store this procedure's ejection events to the database (powers the Ejection Insights charts). Off = eject still fires but isn't logged.">
                            <input type="checkbox" ${proc.store ? 'checked' : ''}
                                onchange="toggleProcedureStore('${proc.id}', this.checked)"
                                style="width: 14px; height: 14px; cursor: pointer;">
                            Store
                        </label>
                        <!-- 3.25.8 — Severity for this ejection procedure. Counts like per-class severity:
                             each stored ejection adds (severity / 100) to the shipment quality impact. -->
                        <label style="display: flex; align-items: center; gap: 4px; font-size: 12px; color: var(--text-secondary);"
                            title="Severity 0–100 for this ejection. Each stored event adds (severity/100) to the shipment quality impact — same scale as per-class severity. 0 = doesn't affect score.">
                            <span>Sev</span>
                            <input type="number" min="0" max="100" step="1" value="${proc.severity != null ? proc.severity : 0}"
                                onchange="updateProcedureField('${proc.id}', 'severity', Math.max(0, Math.min(100, parseInt(this.value) || 0))); saveProcedures();"
                                style="width: 48px; padding: 3px 5px; background: rgba(30,41,59,0.6); color: var(--text-primary); border: 1px solid rgba(51,65,85,0.6); border-radius: 4px; font-size: 11px; text-align: center;">
                        </label>
                        <select onchange="updateProcedureField('${proc.id}', 'logic', this.value)"
                            title="Rule logic: ANY = eject if at least one rule matches (OR), ALL = eject only if every rule matches (AND)"
                            style="padding: 3px 6px; background: rgba(30,41,59,0.6); color: var(--text-primary); border: 1px solid rgba(51,65,85,0.6); border-radius: 4px; font-size: 11px;">
                            <option value="any" ${proc.logic === 'any' ? 'selected' : ''}>ANY rule (OR)</option>
                            <option value="all" ${proc.logic === 'all' ? 'selected' : ''}>ALL rules (AND)</option>
                        </select>
                        <span style="font-size: 10px; color: var(--text-secondary);">Cams:</span>
                        <input type="text" value="${(proc.cameras || []).join(',')}"
                            onchange="updateProcedureField('${proc.id}', 'cameras', this.value.split(',').map(s=>parseInt(s.trim())).filter(n=>!isNaN(n)))"
                            placeholder="all"
                            title="Camera IDs (e.g. 1,2) or leave empty for all"
                            style="width: 50px; padding: 3px 5px; background: rgba(30,41,59,0.6); color: var(--text-primary); border: 1px solid rgba(51,65,85,0.6); border-radius: 4px; font-size: 11px; text-align: center;">
                        <button onclick="removeProcedure('${proc.id}')"
                            title="Delete this entire procedure and all its rules"
                            style="padding: 3px 8px; background: rgba(239,68,68,0.2); color: #ef4444; border: 1px solid rgba(239,68,68,0.4); border-radius: 4px; cursor: pointer; font-size: 12px;">Delete</button>
                    </div>
                </div>
                <div style="display: flex; flex-direction: column; gap: 4px; margin-bottom: 8px;">
                    ${rulesHtml || '<div style="padding: 8px; text-align: center; color: var(--text-secondary); font-size: 12px; font-style: italic;">No rules. Add a rule to define when this procedure triggers.</div>'}
                </div>
                <button onclick="addRule('${proc.id}')"
                    title="Add a new rule to this procedure — rules are combined using the logic (AND/OR) above"
                    style="padding: 3px 10px; background: rgba(59,130,246,0.2); color: var(--primary-color); border: 1px solid rgba(59,130,246,0.4); border-radius: 4px; cursor: pointer; font-size: 11px;">+ Add Rule</button>
            </div>
        `;
    }).join('');
}

async function saveProcedures() {
    const statusEl = document.getElementById('procedures-save-status');
    try {
        statusEl.textContent = 'Saving...';
        statusEl.style.color = 'var(--text-secondary)';
        const response = await fetch('/api/procedures', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ procedures: procedures })
        });
        const result = await response.json();
        if (response.ok && result.success) {
            statusEl.textContent = 'Saved!';
            statusEl.style.color = 'var(--success-color)';
        } else {
            statusEl.textContent = 'Error: ' + (result.error || 'Unknown');
            statusEl.style.color = '#ef4444';
        }
    } catch (e) {
        statusEl.textContent = 'Save failed';
        statusEl.style.color = '#ef4444';
        console.error('Error saving procedures:', e);
    }
    setTimeout(() => { statusEl.textContent = ''; }, 3000);
}

async function loadProcedures() {
    try {
        const response = await fetch('/api/procedures');
        const data = await response.json();
        if (data.procedures) {
            procedures = data.procedures;
            renderProcedures();
        }
    } catch (e) {
        console.error('Error loading procedures:', e);
    }
}

// Show all objects (enable all: show + narrate + beep)
function showAllObjects() {
    detectedObjectClasses.forEach(objectName => {
        audioSettings.showObjects[objectName] = true;
        audioSettings.enabledObjects[objectName] = true;
        audioSettings.narrateObjects[objectName] = true;
    });
    saveAudioSettings();
    updateObjectsList();
}

// Hide all objects (disable all: show + narrate + beep)
function hideAllObjects() {
    detectedObjectClasses.forEach(objectName => {
        audioSettings.showObjects[objectName] = false;
        audioSettings.enabledObjects[objectName] = false;
        audioSettings.narrateObjects[objectName] = false;
    });
    saveAudioSettings();
    updateObjectsList();
}

// Update beep sound for specific object
function updateObjectBeepSound(objectName, soundType) {
    if (!audioSettings.objectBeepSounds) {
        audioSettings.objectBeepSounds = {};
    }
    audioSettings.objectBeepSounds[objectName] = soundType;
    saveAudioSettings();
}

// Test beep for specific object
function testObjectBeep(objectName) {
    initAudioContext();
    const soundType = audioSettings.objectBeepSounds?.[objectName] || 'sine';
    playBeepWithType(soundType);
    // Also narrate the object name
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(objectName);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = audioSettings.volume;
        window.speechSynthesis.speak(utterance);
    }
}

// Play beep with specific wave type
function playBeepWithType(waveType) {
    if (!audioContext) return;

    setTimeout(() => {
        try {
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();

            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);

            oscillator.type = waveType;
            oscillator.frequency.value = 880;

            gainNode.gain.setValueAtTime(audioSettings.volume, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);

            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.2);
        } catch (e) {
            console.error('Error playing beep:', e);
        }
    }, 0);
}

// Handle detection from timeline data (non-blocking)
function processDetectionForAudio(detectionData) {
    if (!detectionData || !detectionData.detections) {
        return;
    }

    // Process in next event loop to avoid blocking main thread
    setTimeout(() => {
        // Extract unique object classes from detections
        detectionData.detections.forEach(det => {
            if (det.class_name && det.class_name !== 'Unknown') {
                const className = det.class_name;

                // Add to detected classes set
                const wasNew = !detectedObjectClasses.has(className);
                detectedObjectClasses.add(className);

                // Update UI if new object detected
                if (wasNew) {
                    // Initialize defaults for new objects
                    if (audioSettings.enabledObjects[className] === undefined) {
                        audioSettings.enabledObjects[className] = true; // beep enabled by default
                    }
                    if (audioSettings.narrateObjects[className] === undefined) {
                        audioSettings.narrateObjects[className] = true; // narrate enabled by default
                    }
                    if (audioSettings.showObjects[className] === undefined) {
                        audioSettings.showObjects[className] = true; // show bbox by default
                    }
                    if (audioSettings.objectConfidence[className] === undefined) {
                        audioSettings.objectConfidence[className] = 1; // 1% default
                    }
                    saveAudioSettings();
                    updateObjectsList();
                }

                // Min-confidence gate: skip narrate/beep entirely for this detection
                // if its confidence is below the per-class floor. UI value is 0-100;
                // detection.confidence is 0-1. (Server applies the same gate before
                // emitting the event — this is a defense-in-depth client filter.)
                const minConfPct = audioSettings.objectConfidence[className] ?? 1;
                const detConfPct = (det.confidence ?? 0) * 100;
                if (detConfPct < minConfPct) {
                    return;  // skip — below threshold
                }

                // Narrate if global narration ON AND per-object narrate ON
                if (audioSettings.narrate && audioSettings.narrateObjects[className]) {
                    narrateObject(className);
                }

                // Beep if global beep ON AND per-object beep ON
                if (audioSettings.beep && audioSettings.enabledObjects[className]) {
                    const now = Date.now();
                    if (now - lastBeepTime >= BEEP_COOLDOWN) {
                        lastBeepTime = now;
                        const soundType = audioSettings.objectBeepSounds?.[className] || 'sine';
                        playBeepWithType(soundType);
                    }
                }
            }
        });
    }, 0);
}

// Test audio button
function testAudio() {
    initAudioContext();

    if (audioSettings.narrate) {
        narrateObject('Test detection');
    }

    if (audioSettings.beep) {
        playAlarmBeep();
    }

    // If nothing is enabled, play beep anyway to test volume
    if (!audioSettings.narrate && !audioSettings.beep) {
        audioSettings.beep = true;  // Temporarily enable
        playAlarmBeep();
        audioSettings.beep = false;
    }
}

// Tab switching functionality
function switchTab(tabName) {
    // 4.0.16 — kill any stuck floating UI from the OUTGOING tab before swapping.
    // Specifically the Charts hover preview (z-index 999998): if the operator
    // clicks a tab button while the preview is up, `mouseleave` on the chart
    // canvas never fires, so the floating thumbnail stays painted on top of
    // every subsequent tab.
    try {
        const hov = document.getElementById('chart-image-preview');
        if (hov) hov.style.display = 'none';
    } catch (e) { /* never block tab-switch on UI-cleanup errors */ }

    // Hide all tab contents
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(tab => tab.classList.remove('active'));

    // Remove active class from all buttons
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(btn => btn.classList.remove('active'));

    // Show selected tab content
    const selectedTab = document.getElementById('tab-' + tabName);
    if (selectedTab) {
        selectedTab.classList.add('active');
    }

    // Add active class to clicked button
    event.target.classList.add('active');

    // Save selected tab to localStorage
    localStorage.setItem('selectedTab', tabName);

    // Load chat history when switching to AI tab
    if (tabName === 'ai') {
        setTimeout(() => loadChatHistory(), 100);
    }

    // Lazy-load/unload iframes for gallery and grafana tabs
    if (typeof loadIframeForTab === 'function') {
        loadIframeForTab(tabName);
    }
}

// Restore last selected tab on page load (skip iframe tabs to prevent blocking)
document.addEventListener('DOMContentLoaded', function() {
    const savedTab = localStorage.getItem('selectedTab');
    const iframeTabs = ['gallery', 'grafana'];
    if (savedTab && !iframeTabs.includes(savedTab)) {
        const tabButton = document.querySelector(`.tab-button[onclick="switchTab('${savedTab}')"]`);
        if (tabButton) {
            tabButton.click();
        }
    }
});

// ========================================
// Timeline Slideshow Functionality (Embedded)
// ========================================

// Panzoom library v4.6.0 (embedded inline)
((t,e)=>{"object"==typeof exports&&"undefined"!=typeof module?module.exports=e():"function"==typeof define&&define.amd?define(e):(t="undefined"!=typeof globalThis?globalThis:t||self).Panzoom=e()})(this,function(){var a,X=function(){return(X=Object.assign||function(t){for(var e,n=1,o=arguments.length;n<o;n++)for(var r in e=arguments[n])Object.prototype.hasOwnProperty.call(e,r)&&(t[r]=e[r]);return t}).apply(this,arguments)},i=("undefined"!=typeof window&&(window.NodeList&&!NodeList.prototype.forEach&&(NodeList.prototype.forEach=Array.prototype.forEach),"function"!=typeof window.CustomEvent)&&(window.CustomEvent=function(t,e){e=e||{bubbles:!1,cancelable:!1,detail:null};var n=document.createEvent("CustomEvent");return n.initCustomEvent(t,e.bubbles,e.cancelable,e.detail),n}),"undefined"!=typeof document&&!!document.documentMode);var c=["webkit","moz","ms"],l={};function Y(t){if(l[t])return l[t];var e=a=a||document.createElement("div").style;if(t in e)return l[t]=t;for(var n=t[0].toUpperCase()+t.slice(1),o=c.length;o--;){var r="".concat(c[o]).concat(n);if(r in e)return l[t]=r}}function o(t,e){return parseFloat(e[Y(t)])||0}function s(t,e,n){void 0===n&&(n=window.getComputedStyle(t));t="border"===e?"Width":"";return{left:o("".concat(e,"Left").concat(t),n),right:o("".concat(e,"Right").concat(t),n),top:o("".concat(e,"Top").concat(t),n),bottom:o("".concat(e,"Bottom").concat(t),n)}}function C(t,e,n){t.style[Y(e)]=n}function N(t){var e=t.parentNode,n=window.getComputedStyle(t),o=window.getComputedStyle(e),r=t.getBoundingClientRect(),a=e.getBoundingClientRect();return{elem:{style:n,width:r.width,height:r.height,top:r.top,bottom:r.bottom,left:r.left,right:r.right,margin:s(t,"margin",n),border:s(t,"border",n)},parent:{style:o,width:a.width,height:a.height,top:a.top,bottom:a.bottom,left:a.left,right:a.right,padding:s(e,"padding",o),border:s(e,"border",o)}}}var T={down:"mousedown",move:"mousemove",up:"mouseup mouseleave"};function L(t,e,n,o){T[t].split(" ").forEach(function(t){e.addEventListener(t,n,o)})}function V(t,e,n){T[t].split(" ").forEach(function(t){e.removeEventListener(t,n)})}function G(t,e){for(var n=t.length;n--;)if(t[n].pointerId===e.pointerId)return n;return-1}function I(t,e){if(e.touches)for(var n=0,o=0,r=e.touches;o<r.length;o++){var a=r[o];a.pointerId=n++,I(t,a)}else-1<(n=G(t,e))&&t.splice(n,1),t.push(e)}function R(t){for(var e,n=(t=t.slice(0)).pop();e=t.pop();)n={clientX:(e.clientX-n.clientX)/2+n.clientX,clientY:(e.clientY-n.clientY)/2+n.clientY};return n}function W(t){var e;return t.length<2?0:(e=t[0],t=t[1],Math.sqrt(Math.pow(Math.abs(t.clientX-e.clientX),2)+Math.pow(Math.abs(t.clientY-e.clientY),2)))}"undefined"!=typeof window&&("function"==typeof window.PointerEvent?T={down:"pointerdown",move:"pointermove",up:"pointerup pointerleave pointercancel"}:"function"==typeof window.TouchEvent&&(T={down:"touchstart",move:"touchmove",up:"touchend touchcancel"}));var Z=/^http:[\w\.\/]+svg$/;var q={animate:!1,canvas:!1,cursor:"move",disablePan:!1,disableZoom:!1,disableXAxis:!1,disableYAxis:!1,duration:200,easing:"ease-in-out",exclude:[],excludeClass:"panzoom-exclude",handleStartEvent:function(t){t.preventDefault(),t.stopPropagation()},maxScale:4,minScale:.125,overflow:"hidden",panOnlyWhenZoomed:!1,pinchAndPan:!1,relative:!1,setTransform:function(t,e,n){var o=e.x,r=e.y,a=e.isSVG;C(t,"transform","scale(".concat(e.scale,") translate(").concat(o,"px, ").concat(r,"px)")),a&&i&&(e=window.getComputedStyle(t).getPropertyValue("transform"),t.setAttribute("transform",e))},startX:0,startY:0,startScale:1,step:.3,touchAction:"none"};function t(u,f){if(!u)throw new Error("Panzoom requires an element as an argument");if(1!==u.nodeType)throw new Error("Panzoom requires an element with a nodeType of 1");if(!(t=>{for(var e=t;e&&e.parentNode;){if(e.parentNode===document)return 1;e=e.parentNode instanceof ShadowRoot?e.parentNode.host:e.parentNode}})(u))throw new Error("Panzoom should be called on elements that have been attached to the DOM");f=X(X({},q),f);t=u;var t,l=Z.test(t.namespaceURI)&&"svg"!==t.nodeName.toLowerCase(),n=u.parentNode;n.style.overflow=f.overflow,n.style.userSelect="none",n.style.touchAction=f.touchAction,(f.canvas?n:u).style.cursor=f.cursor,u.style.userSelect="none",u.style.touchAction=f.touchAction,C(u,"transformOrigin","string"==typeof f.origin?f.origin:l?"0 0":"50% 50%");var r,a,i,c,s,d,m=0,h=0,v=1,p=!1;function g(t,e,n){n.silent||(n=new CustomEvent(t,{detail:e}),u.dispatchEvent(n))}function y(o,r,t){var a={x:m,y:h,scale:v,isSVG:l,originalEvent:t};return requestAnimationFrame(function(){var t,e,n;"boolean"==typeof r.animate&&(r.animate?(t=u,e=r,n=Y("transform"),C(t,"transition","".concat(n," ").concat(e.duration,"ms ").concat(e.easing))):C(u,"transition","none")),r.setTransform(u,a,r),g(o,a,r),g("panzoomchange",a,r)}),a}function w(t,e,n,o){var r,a,i,c,l,s,d,o=X(X({},f),o),p={x:m,y:h,opts:o};return!o.force&&(o.disablePan||o.panOnlyWhenZoomed&&v===o.startScale)||(t=parseFloat(t),e=parseFloat(e),o.disableXAxis||(p.x=(o.relative?m:0)+t),o.disableYAxis||(p.y=(o.relative?h:0)+e),o.contain&&(e=((r=(e=(t=N(u)).elem.width/v)*n)-e)/2,i=((a=(i=t.elem.height/v)*n)-i)/2,"inside"===o.contain?(c=(-t.elem.margin.left-t.parent.padding.left+e)/n,l=(t.parent.width-r-t.parent.padding.left-t.elem.margin.left-t.parent.border.left-t.parent.border.right+e)/n,p.x=Math.max(Math.min(p.x,l),c),s=(-t.elem.margin.top-t.parent.padding.top+i)/n,d=(t.parent.height-a-t.parent.padding.top-t.elem.margin.top-t.parent.border.top-t.parent.border.bottom+i)/n,p.y=Math.max(Math.min(p.y,d),s)):"outside"===o.contain&&(c=(-(r-t.parent.width)-t.parent.padding.left-t.parent.border.left-t.parent.border.right+e)/n,l=(e-t.parent.padding.left)/n,p.x=Math.max(Math.min(p.x,l),c),s=(-(a-t.parent.height)-t.parent.padding.top-t.parent.border.top-t.parent.border.bottom+i)/n,d=(i-t.parent.padding.top)/n,p.y=Math.max(Math.min(p.y,d),s))),o.roundPixels&&(p.x=Math.round(p.x),p.y=Math.round(p.y))),p}function b(t,e){var n,o,r,a,e=X(X({},f),e),i={scale:v,opts:e};return!e.force&&e.disableZoom||(n=f.minScale,o=f.maxScale,e.contain&&(a=(e=N(u)).elem.width/v,r=e.elem.height/v,1<a)&&1<r&&(a=(e.parent.width-e.parent.border.left-e.parent.border.right)/a,e=(e.parent.height-e.parent.border.top-e.parent.border.bottom)/r,"inside"===f.contain?o=Math.min(o,a,e):"outside"===f.contain&&(n=Math.max(n,a,e))),i.scale=Math.min(Math.max(t,n),o)),i}function x(t,e,n,o){t=w(t,e,v,n);return m!==t.x||h!==t.y?(m=t.x,h=t.y,y("panzoompan",t.opts,o)):{x:m,y:h,scale:v,isSVG:l,originalEvent:o}}function S(t,e,n){var o,r,e=b(t,e),a=e.opts;if(a.force||!a.disableZoom)return t=e.scale,e=m,o=h,a.focal&&(e=((r=a.focal).x/t-r.x/v+m*t)/t,o=(r.y/t-r.y/v+h*t)/t),r=w(e,o,t,{relative:!1,force:!0}),m=r.x,h=r.y,v=t,y("panzoomzoom",a,n)}function e(t,e){e=X(X(X({},f),{animate:!0}),e);return S(v*Math.exp((t?1:-1)*e.step),e)}function E(t,e,n,o){var r=N(u),a=r.parent.width-r.parent.padding.left-r.parent.padding.right-r.parent.border.left-r.parent.border.right,i=r.parent.height-r.parent.padding.top-r.parent.padding.bottom-r.parent.border.top-r.parent.border.bottom,c=e.clientX-r.parent.left-r.parent.padding.left-r.parent.border.left-r.elem.margin.left,e=e.clientY-r.parent.top-r.parent.padding.top-r.parent.border.top-r.elem.margin.top,r=(l||(c-=r.elem.width/v/2,e-=r.elem.height/v/2),{x:c/a*(a*t),y:e/i*(i*t)});return S(t,X(X({},n),{animate:!1,focal:r}),o)}S(f.startScale,{animate:!1,force:!0}),setTimeout(function(){x(f.startX,f.startY,{animate:!1,force:!0})});var M=[];function o(t){((t,e)=>{for(var n,o,r=t;null!=r;r=r.parentNode)if(n=r,o=e.excludeClass,1===n.nodeType&&-1<" ".concat((n.getAttribute("class")||"").trim()," ").indexOf(" ".concat(o," "))||-1<e.exclude.indexOf(r))return 1})(t.target,f)||(I(M,t),p=!0,f.handleStartEvent(t),g("panzoomstart",{x:r=m,y:a=h,scale:v,isSVG:l,originalEvent:t},f),t=R(M),i=t.clientX,c=t.clientY,s=v,d=W(M))}function A(t){var e,n,o;p&&void 0!==r&&void 0!==a&&void 0!==i&&void 0!==c&&(I(M,t),e=R(M),n=1<M.length,o=v,n&&(0===d&&(d=W(M)),E(o=b((W(M)-d)*f.step/80+s).scale,e,{animate:!1},t)),n&&!f.pinchAndPan||x(r+(e.clientX-i)/o,a+(e.clientY-c)/o,{animate:!1},t))}function P(t){1===M.length&&g("panzoomend",{x:m,y:h,scale:v,isSVG:l,originalEvent:t},f);var e=M;if(t.touches)for(;e.length;)e.pop();else{t=G(e,t);-1<t&&e.splice(t,1)}p&&(p=!1,r=a=i=c=void 0)}var O=!1;function z(){O||(O=!0,L("down",f.canvas?n:u,o),L("move",document,A,{passive:!0}),L("up",document,P,{passive:!0}))}return f.noBind||z(),{bind:z,destroy:function(){O=!1,V("down",f.canvas?n:u,o),V("move",document,A),V("up",document,P)},eventNames:T,getPan:function(){return{x:m,y:h}},getScale:function(){return v},getOptions:function(){var t,e=f,n={};for(t in e)e.hasOwnProperty(t)&&(n[t]=e[t]);return n},handleDown:o,handleMove:A,handleUp:P,pan:x,reset:function(t){var t=X(X(X({},f),{animate:!0,force:!0}),t),e=(v=b(t.startScale,t).scale,w(t.startX,t.startY,v,t));return m=e.x,h=e.y,y("panzoomreset",t)},resetStyle:function(){n.style.overflow="",n.style.userSelect="",n.style.touchAction="",n.style.cursor="",u.style.cursor="",u.style.userSelect="",u.style.touchAction="",C(u,"transformOrigin","")},setOptions:function(t){for(var e in t=void 0===t?{}:t)t.hasOwnProperty(e)&&(f[e]=t[e]);(t.hasOwnProperty("cursor")||t.hasOwnProperty("canvas"))&&(n.style.cursor=u.style.cursor="",(f.canvas?n:u).style.cursor=f.cursor),t.hasOwnProperty("overflow")&&(n.style.overflow=t.overflow),t.hasOwnProperty("touchAction")&&(n.style.touchAction=t.touchAction,u.style.touchAction=t.touchAction)},setStyle:function(t,e){return C(u,t,e)},zoom:S,zoomIn:function(t){return e(!0,t)},zoomOut:function(t){return e(!1,t)},zoomToPoint:E,zoomWithWheel:function(t,e){t.preventDefault();var e=X(X(X({},f),e),{animate:!1}),n=0===t.deltaY&&t.deltaX?t.deltaX:t.deltaY;return E(b(v*Math.exp((n<0?1:-1)*e.step/3),e).scale,t,e,t)}}}return t.defaultOptions=q,t});

// Initialize timeline slideshow (single image with auto-refresh)
(function() {
    const slide = document.getElementById("timeline-slide");
    const wrapper = document.getElementById("timeline-slide-wrapper");
    const counter = document.getElementById("timeline-counter");
    const toggleBtn = document.getElementById("timeline-toggle-auto");

    let autoUpdate = true;
    window.timelineAutoUpdate = true;
    let refreshInterval;

    // Initialize Panzoom on the timeline image
    const panzoom = Panzoom(slide, { maxScale: 40 });

    // Timeline pagination (navigate through pages of concatenated frames)
    let currentPage = 0;
    let totalPages = 0;
    let totalFrames = 0;
    let rowsPerPage = 10;

    async function updatePageCount() {
        try {
            const response = await fetch('/api/timeline_count');
            const data = await response.json();
            totalFrames = data.total_frames;
            totalPages = data.total_pages;
            rowsPerPage = data.rows_per_page;
            updateCounter();
        } catch (error) {
            console.error('Error fetching page count:', error);
        }
    }

    function updateCounter() {
        if (totalPages > 0) {
            if (autoUpdate) {
                counter.textContent = `1 / ${totalPages}`;
            } else {
                counter.textContent = `${currentPage + 1} / ${totalPages}`;
            }
        } else {
            counter.textContent = "0 / 0";
        }
    }

    function loadPage(page) {
        if (page < 0 || page >= totalPages) return;
        currentPage = page;
        // Tell WebSocket server which page we want (for future auto-pushes)
        if (typeof timelineWsSendPage === 'function') {
            timelineWsSendPage(page);
        }
        // When paused, WebSocket onmessage drops images, so always use HTTP for manual nav
        if (!autoUpdate) {
            const timestamp = new Date().getTime();
            slide.src = `/timeline_image?page=${page}&t=${timestamp}`;
            fetch('/api/timeline_meta?page=' + page)
                .then(r => r.json())
                .then(meta => { if (meta.type === 'timeline_meta') window._timelineMeta = meta; })
                .catch(() => {});
        }
        updateCounter();
        panzoom.reset();
    }

    // Auto-refresh functionality (reload latest page)
    async function refreshImage() {
        if (autoUpdate) {
            await updatePageCount();
            currentPage = 0; // Always show latest page
            // WebSocket handles auto-push for page 0; send page just in case
            if (typeof timelineWsSendPage === 'function') {
                timelineWsSendPage(0);
            } else {
                const timestamp = new Date().getTime();
                slide.src = `/timeline_image?page=0&t=${timestamp}`;
            }
            updateCounter();
        }
    }

    // Start auto-refresh
    function startAutoRefresh() {
        refreshInterval = setInterval(refreshImage, 5000);
    }

    // Stop auto-refresh
    function stopAutoRefresh() {
        if (refreshInterval) {
            clearInterval(refreshInterval);
            refreshInterval = null;
        }
    }

    // Auto-resume after 30s of inactivity when paused
    let _autoResumeTimer = null;
    function scheduleAutoResume() {
        if (_autoResumeTimer) clearTimeout(_autoResumeTimer);
        _autoResumeTimer = setTimeout(() => {
            if (!autoUpdate) {
                autoUpdate = true;
                window.timelineAutoUpdate = true;
                toggleBtn.textContent = "Stop";
                toggleBtn.classList.remove("stopped");
                currentPage = 0;
                startAutoRefresh();
                refreshImage();
                panzoom.reset();
            }
        }, 30000);
    }

    // Toggle auto-update
    toggleBtn.onclick = () => {
        autoUpdate = !autoUpdate;
        window.timelineAutoUpdate = autoUpdate;
        if (autoUpdate) {
            if (_autoResumeTimer) clearTimeout(_autoResumeTimer);
            toggleBtn.textContent = "Stop";
            toggleBtn.classList.remove("stopped");
            currentPage = 0;
            startAutoRefresh();
            refreshImage();
        } else {
            toggleBtn.textContent = "Resume";
            toggleBtn.classList.add("stopped");
            stopAutoRefresh();
            updateCounter(); // show current page instead of "Paused"
            scheduleAutoResume();
        }
    };

    // Navigation buttons (page navigation)
    async function pauseAndNav(getPage) {
        stopAutoRefresh();
        autoUpdate = false;
        window.timelineAutoUpdate = false;
        toggleBtn.textContent = "Resume";
        toggleBtn.classList.add("stopped");
        await updatePageCount(); // ensure totalPages is fresh
        loadPage(getPage());
        scheduleAutoResume();
    }

    // 3.21.6 — guard with `?.` because the < > pagination buttons were removed
    // in 3.21.2 (lightweight dashboard) but these bindings remained. Without the
    // guard, the FIRST null deref throws and the whole IIFE aborts BEFORE the
    // wheel listener gets attached — that was the real cause of "zoom dead".
    document.getElementById("timeline-first")?.addEventListener('click', () => {
        pauseAndNav(() => totalPages - 1); // Oldest page
    });

    document.getElementById("timeline-prev")?.addEventListener('click', () => {
        pauseAndNav(() => Math.min(totalPages - 1, currentPage + 1)); // Older page
    });

    document.getElementById("timeline-next")?.addEventListener('click', () => {
        pauseAndNav(() => Math.max(0, currentPage - 1)); // Newer page
    });

    document.getElementById("timeline-last")?.addEventListener('click', () => {
        pauseAndNav(() => 0); // Latest page
    });

    // Update page count periodically
    setInterval(updatePageCount, 5000);
    updatePageCount();

    // Zoom controls (also guarded — present today but cheap insurance)
    document.getElementById("timeline-zoom-in")?.addEventListener('click', () => { panzoom.zoomIn(); scheduleZoomReset(); });
    document.getElementById("timeline-zoom-out")?.addEventListener('click', () => { panzoom.zoomOut(); scheduleZoomReset(); });
    document.getElementById("timeline-reset-zoom")?.addEventListener('click', panzoom.reset);

    // Auto-reset zoom after 30s of inactivity (industrial kiosk — no one to un-zoom)
    let _zoomResetTimer = null;
    function scheduleZoomReset() {
        if (_zoomResetTimer) clearTimeout(_zoomResetTimer);
        _zoomResetTimer = setTimeout(() => {
            if (panzoom.getScale() !== 1) {
                panzoom.reset();
                console.log('[Timeline] Zoom auto-reset after inactivity');
            }
        }, 30000);
    }

    // Wheel zoom (3.21.5: reverted to the proven HEAD baseline that worked in
    // 3.19 — Panzoom's internal preventDefault is sufficient on a regular div;
    // the {passive:false} + explicit preventDefault we tried in 3.21.2 was
    // redundant and appears to have broken zoom on the user's browser).
    wrapper.addEventListener("wheel", function(e) {
        panzoom.zoomWithWheel(e);
        scheduleZoomReset();
    });

    // Double-click to toggle zoom
    slide.addEventListener("dblclick", () => {
        panzoom.zoom(panzoom.getScale() === 1 ? 2 : 1);
        scheduleZoomReset();
    });

    // Click-to-view: click on a frame to view in gallery or download raw image
    (function() {
        let mouseDownPos = null;
        let popup = null;

        function removePopup() {
            if (popup && popup.parentNode) popup.parentNode.removeChild(popup);
            popup = null;
        }

        slide.addEventListener("pointerdown", (e) => {
            mouseDownPos = { x: e.clientX, y: e.clientY };
        });

        slide.addEventListener("pointerup", (e) => {
            if (!mouseDownPos) return;
            const dx = e.clientX - mouseDownPos.x;
            const dy = e.clientY - mouseDownPos.y;
            mouseDownPos = null;

            // Only treat as click if mouse moved less than 5px (not a pan)
            if (Math.sqrt(dx * dx + dy * dy) > 5) return;

            const meta = window._timelineMeta;
            if (!meta || !meta.columns || !meta.columns.length || !meta.thumb_width) return;

            // Map click to natural image coordinates
            const rect = slide.getBoundingClientRect();
            const natW = slide.naturalWidth;
            const natH = slide.naturalHeight;
            if (!natW || !natH) return;

            const scaleX = natW / rect.width;
            const scaleY = natH / rect.height;
            const natX = (e.clientX - rect.left) * scaleX;
            const natY = (e.clientY - rect.top) * scaleY;

            // Determine column and camera row
            const colIndex = Math.floor(natX / meta.thumb_width);
            const rowY = natY - meta.header_height;
            if (rowY < 0 || colIndex < 0 || colIndex >= meta.columns.length) return;
            const camIndex = Math.floor(rowY / meta.thumb_height);
            if (camIndex < 0 || camIndex >= meta.num_cameras) return;

            const col = meta.columns[colIndex];
            const camIds = meta.cam_ids || Object.keys(col.d_paths);
            if (camIndex >= camIds.length) return;
            const camId = String(camIds[camIndex]);
            const dPath = col.d_paths[camId];

            if (!dPath) return;

            // Remove any legacy popup (we now reuse the unified defect modal)
            removePopup();

            // Build URLs that work even when Store=off (annotated jpg may not be
            // persisted to raw_images/, so we use the live /api/timeline_frame).
            const frameUrl = `/api/timeline_frame?cam=${camId}&col=${colIndex}&page=${currentPage}&path=${encodeURIComponent(dPath)}&t=${Date.now()}`;
            const rawUrl   = `/api/raw_image/${encodeURI(dPath)}.jpg`;
            const ejectTag = col.should_eject ? 'EJECT' : 'OK';
            const tags = [ejectTag, 'cam ' + camId];
            if (col.encoder != null) tags.push('enc ' + col.encoder);

            // Route through the unified centered modal (3.21.4) — same Annotated /
            // Raw / Both toggle + Download buttons + Panzoom that the charts use.
            if (typeof window.openDefectDrawerForFrame === 'function') {
                window.openDefectDrawerForFrame({
                    annotated_url: frameUrl,
                    raw_url:       rawUrl,
                    shipment:      (dPath.split('/')[0] || ''),
                    t:             col.ts ? col.ts * 1000 : null,
                    cls:           tags.join(' • '),
                    classes:       tags,
                    best_confidence: 0,
                });
            }
        });

        // Also dismiss on Escape
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape") removePopup();
        });
    })();

    // Start auto-refresh on load
    updatePageCount();
    startAutoRefresh();
    refreshImage();
})();

// ========================================
// Timeline Configuration Controls
// ========================================
(function() {
    const qualitySlider = document.getElementById('timeline-quality');
    const qualityValue = document.getElementById('timeline-quality-value');
    const rowsSlider = document.getElementById('timeline-rows');
    const rowsValue = document.getElementById('timeline-rows-value');
    const bufferSlider = document.getElementById('timeline-buffer');
    const bufferValue = document.getElementById('timeline-buffer-value');
    const applyBtn = document.getElementById('timeline-apply-config');

    // Show/hide custom order input based on radio selection
    document.querySelectorAll('input[name="camera-order"]').forEach(radio => {
        radio.addEventListener('change', () => {
            const container = document.getElementById('custom-camera-order-container');
            if (container) container.style.display = radio.value === 'custom' ? 'block' : 'none';
        });
    });

    // Load current in-memory configuration from server
    async function loadSavedConfig() {
        try {
            const response = await fetch('/api/timeline_config');
            const cfg = await response.json();
            if (cfg.image_quality) {
                qualitySlider.value = cfg.image_quality;
                qualityValue.textContent = cfg.image_quality + '%';
            }
            if (cfg.num_rows) {
                rowsSlider.value = cfg.num_rows;
                rowsValue.textContent = cfg.num_rows;
            }
            // buffer slider removed in 3.21.2 — storage now bound to num_rows
            if (cfg.buffer_size && bufferSlider) {
                bufferSlider.value = cfg.buffer_size;
                if (bufferValue) bufferValue.textContent = cfg.buffer_size;
            }
            // Restore camera order radio + custom input
            if (cfg.camera_order) {
                const radio = document.querySelector(`input[name="camera-order"][value="${cfg.camera_order}"]`);
                if (radio) radio.checked = true;
                const container = document.getElementById('custom-camera-order-container');
                if (container) container.style.display = cfg.camera_order === 'custom' ? 'block' : 'none';
            }
            if (cfg.custom_camera_order) {
                const input = document.getElementById('custom-camera-order');
                if (input) input.value = cfg.custom_camera_order;
            }
            if (cfg.object_filters) {
                const filters = cfg.object_filters;
                Object.keys(filters).forEach(name => {
                    detectedObjectClasses.add(name);
                    audioSettings.showObjects[name] = filters[name].show !== false;
                    audioSettings.objectConfidence[name] = Math.round((filters[name].min_confidence || 0.01) * 100);
                });
                updateObjectsList();
            }
        } catch (error) {
            console.error('Error loading timeline config:', error);
        }
    }

    // Load config on page load
    loadSavedConfig();

    // Update slider value displays
    qualitySlider.addEventListener('input', () => {
        qualityValue.textContent = qualitySlider.value + '%';
    });

    rowsSlider.addEventListener('input', () => {
        rowsValue.textContent = rowsSlider.value;
    });

    if (bufferSlider) bufferSlider.addEventListener('input', () => {
        if (bufferValue) bufferValue.textContent = bufferSlider.value;
    });

    // Apply configuration
    applyBtn.addEventListener('click', async () => {
        const cameraOrder = document.querySelector('input[name="camera-order"]:checked').value;
        const quality = qualitySlider.value;
        const rows = rowsSlider.value;
        const buffer = bufferSlider ? bufferSlider.value : rows;   // fallback: 3.21.2 ties storage to rows
        const responseDiv = document.getElementById('timeline-config-response');

        applyBtn.textContent = '⏳ Applying...';
        applyBtn.disabled = true;
        responseDiv.textContent = 'Applying configuration...';
        responseDiv.style.color = '#666';

        try {
            const rotation = parseInt(document.getElementById('timeline-rotation').value) || 0;

            // Build per-object filters from audioSettings
            const objectFilters = {};
            detectedObjectClasses.forEach(name => {
                objectFilters[name] = {
                    show: audioSettings.showObjects[name] !== false,
                    min_confidence: (audioSettings.objectConfidence[name] !== undefined ? audioSettings.objectConfidence[name] : 1) / 100
                };
            });

            // 3.21.22 — object_filters is no longer written from the UI.
            // The Process tab toggle alone is the canonical Show source
            // (audio_settings.<class>.show). Sending object_filters here
            // would just re-create the drift that we ripped out server-side.
            // Backend tolerates the field's absence — it was already optional.
            const response = await fetch('/api/timeline_config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    show_bounding_boxes: audioSettings.showBoundingBoxes !== false,
                    camera_order: cameraOrder,
                    custom_camera_order: cameraOrder === 'custom' ? (document.getElementById('custom-camera-order').value || '') : '',
                    image_quality: parseInt(quality),
                    num_rows: parseInt(rows),
                    buffer_size: parseInt(buffer),
                    image_rotation: rotation,
                })
            });

            const result = await response.json();

            if (response.ok && result.success) {
                applyBtn.textContent = '✓ Applied!';
                applyBtn.style.background = 'var(--success-color)';
                responseDiv.textContent = '✓ Configuration applied successfully! Timeline will update on next refresh.';
                responseDiv.style.color = 'var(--success-color)';

                // Auto-save to DATA_FILE
                await saveAllServiceConfig();

                // Refresh timeline image after config change
                setTimeout(() => {
                    const timestamp = new Date().getTime();
                    document.getElementById('timeline-slide').src = `/timeline_image?t=${timestamp}`;

                    applyBtn.textContent = '✓ Apply Timeline Configuration';
                    applyBtn.style.background = 'var(--primary-color)';
                    applyBtn.disabled = false;
                }, 1500);
            } else {
                throw new Error(result.error || 'Failed to apply configuration');
            }
        } catch (error) {
            console.error('Error applying timeline configuration:', error);
            applyBtn.textContent = '⚠️ Error - Retry';
            applyBtn.style.background = 'var(--danger-color)';
            responseDiv.textContent = `⚠️ Error: ${error.message}`;
            responseDiv.style.color = 'var(--danger-color)';

            setTimeout(() => {
                applyBtn.textContent = '✓ Apply Timeline Configuration';
                applyBtn.style.background = 'var(--primary-color)';
                applyBtn.disabled = false;
            }, 3000);
        }
    });
})();

// Shipment ID editing functionality
// 3.25.0 — single edit row (input + 🎲 + ✓/✗) inside a shipment-edit-row container;
// shows when the display value is clicked, hides on Cancel/Save.
function editShipmentId() {
    const shipmentValue = document.getElementById('shipment-value');
    const editRow       = document.getElementById('shipment-edit-row');
    const shipmentInput = document.getElementById('shipment-input');
    const shipmentText  = document.getElementById('shipment-text');

    if (shipmentValue) shipmentValue.style.display = 'none';
    if (editRow)       editRow.style.display = 'flex';
    if (shipmentInput) {
        shipmentInput.value = (shipmentText && shipmentText.textContent !== '-')
            ? shipmentText.textContent : '';
        shipmentInput.focus();
        shipmentInput.select();
    }
}

function cancelEditShipment() {
    const shipmentValue = document.getElementById('shipment-value');
    const editRow       = document.getElementById('shipment-edit-row');
    if (shipmentValue) shipmentValue.style.display = 'flex';
    if (editRow)       editRow.style.display = 'none';
}

// 4.0.51 — Dice is now FULLY FRONTEND. The backend endpoint was fine in
// isolation, but the operator observed 2-5 second delays under heavy CPU
// load because the request-thread had to wait behind capture/inference for
// a slot. That's a false-scarcity queue — the "unique code" doesn't need
// server state; the local timestamp is already monotonic and unique per
// second, and a fractional decisecond digit makes even sub-second retries
// safe.
//
// Format kept structurally similar to the old backend one:
//   yymmddHHMMSSd
// where d is the decisecond (0-9) of the local clock. Result: 13 chars,
// same digit width as the old backend format (yymmddXXYYZZZ) so any
// downstream logic that assumed 13-char shipment IDs keeps working.
//
// Zero HTTP round-trip → sub-millisecond response → dice is instantaneous
// even when the box is at 100% CPU with the disk queue full. Falls back
// to the (slow) backend endpoint only if Date is broken somehow — that
// should never happen in a real browser.
function generateShipmentCode() {
    const shipmentInput = document.getElementById('shipment-input');
    if (!shipmentInput) return;
    try {
        const now = new Date();
        const yy = String(now.getFullYear() % 100).padStart(2, '0');
        const mm = String(now.getMonth() + 1).padStart(2, '0');
        const dd = String(now.getDate()).padStart(2, '0');
        const h  = String(now.getHours()).padStart(2, '0');
        const mi = String(now.getMinutes()).padStart(2, '0');
        const s  = String(now.getSeconds()).padStart(2, '0');
        const ds = String(Math.floor(now.getMilliseconds() / 100));  // deciseconds
        shipmentInput.value = `${yy}${mm}${dd}${h}${mi}${s}${ds}`;
        shipmentInput.focus();
    } catch (e) {
        // Only reached if Date itself is broken (never happens). Alert
        // instead of a silent bad ID.
        alert('Dice failed: ' + (e.message || e));
    }
}
window.generateShipmentCode = generateShipmentCode;

async function saveShipmentId() {
    const shipmentInput = document.getElementById('shipment-input');
    const newShipmentId = shipmentInput.value.trim() || 'no_shipment';

    try {
        const response = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                shipment: newShipmentId
            })
        });

        if (response.ok) {
            const shipmentText = document.getElementById('shipment-text');
            shipmentText.textContent = newShipmentId;
            cancelEditShipment();

            // Show success message
            const statusBox = shipmentInput.closest('.status-box');
            const successMsg = document.createElement('div');
            successMsg.style.cssText = 'color: var(--success-color); font-size: 12px; margin-top: 4px;';
            successMsg.textContent = '✓ Shipment ID updated';
            statusBox.appendChild(successMsg);
            setTimeout(() => successMsg.remove(), 3000);
        } else {
            let detail = '';
            try { const err = await response.json(); detail = err.detail || JSON.stringify(err); } catch(_) {}
            console.error('Failed to update shipment ID:', response.status, detail);
            alert('Failed to update shipment ID: ' + (detail || response.status));
        }
    } catch (error) {
        console.error('Error updating shipment ID:', error);
        alert('Error updating shipment ID: ' + error.message);
    }
}

// Allow Enter key to save
document.addEventListener('DOMContentLoaded', function() {
    const shipmentInput = document.getElementById('shipment-input');
    if (shipmentInput) {
        shipmentInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                saveShipmentId();
            } else if (e.key === 'Escape') {
                cancelEditShipment();
            }
        });
    }
});

// Service Configuration functions
// Populate the "Last:" saved timestamp from the persisted config on page load,
// so it survives a refresh. The save itself writes service_config.saved_at; we
// read it back here instead of relying on the in-DOM client timestamp (which is
// lost on reload — that's why it showed "Never" after refresh).
async function loadLastSavedTime() {
    try {
        const r = await fetch('/api/cameras/config');
        const d = await r.json();
        const ts = (d.config && d.config.saved_at) || d.saved_at;
        const el = document.getElementById('last-saved-time');
        if (el && ts) el.textContent = ts;
    } catch (e) { /* leave the i18n "Never" default */ }
}
document.addEventListener('DOMContentLoaded', loadLastSavedTime);

async function saveAllServiceConfig() {
    var _b = _btnLoading();
    try {
        // Save all configuration (cameras + settings) using the correct endpoint
        const response = await fetch('/api/cameras/config/save', { method: 'POST' });

        if (response.ok) {
            const data = await response.json();
            // Prefer the server-persisted saved_at; fall back to client time.
            const timestamp = (data.config && data.config.saved_at) || new Date().toISOString();
            document.getElementById('last-saved-time').textContent = timestamp;
            alert('✓ All settings saved successfully!');
        } else {
            const data = await response.json();
            alert(`❌ Failed to save settings: ${data.error || 'Unknown error'}`);
        }
    } catch (error) {
        console.error('Error saving config:', error);
        alert('❌ Error saving settings');
    } finally { _btnDone(_b); }
}

async function loadServiceConfig() {
    if (!confirm('Load saved service configuration? This will reload the page.')) return;
    try {
        const response = await fetch('/api/cameras/config/load', { method: 'POST' });
        if (response.ok) {
            alert('✓ Configuration loaded! Reloading page...');
            location.reload();
        } else {
            const data = await response.json();
            alert(`❌ Failed to load configuration: ${data.error || 'Unknown error'}`);
        }
    } catch (error) {
        console.error('Error loading config:', error);
        alert('❌ Error loading configuration');
    }
}

async function exportServiceConfig() {
    try {
        const response = await fetch('/api/export_service_config');
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `monitaqc_config_${new Date().toISOString().split('T')[0]}.json`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } else {
            alert('❌ Failed to export configuration');
        }
    } catch (error) {
        console.error('Error exporting config:', error);
        alert('❌ Error exporting configuration');
    }
}

function importServiceConfig() {
    // Trigger file input for the working import function below
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        try {
            const text = await file.text();
            const config = JSON.parse(text);
            const response = await fetch('/api/cameras/config/upload', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config })
            });
            const result = await response.json();
            if (response.ok && result.success) {
                alert(`✓ Config imported! (${result.cameras_loaded} cameras loaded)`);
                location.reload();
            } else {
                alert(`❌ Failed to import: ${result.error || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Error importing config:', error);
            alert(`❌ Error: ${error.message}`);
        }
    };
    input.click();
}

// AI Assistant Functions
const aiProviderConfigs = {
    'claude': { label: 'Anthropic API Key:', placeholder: 'sk-ant-...', link: 'https://console.anthropic.com/', linkText: 'Anthropic Console' },
    'chatgpt': { label: 'OpenAI API Key:', placeholder: 'sk-...', link: 'https://platform.openai.com/api-keys', linkText: 'OpenAI Platform' },
    'gemini': { label: 'Google API Key:', placeholder: 'AI...', link: 'https://makersuite.google.com/app/apikey', linkText: 'Google AI Studio' },
    'local': { label: 'Model Endpoint:', placeholder: 'http://localhost:11434', link: 'https://ollama.ai/', linkText: 'Ollama Documentation' }
};

let aiModelsData = { models: {}, active: null };

function updateAIModelFields() {
    const provider = document.getElementById('ai-model-provider').value;
    const cfg = aiProviderConfigs[provider];
    document.getElementById('api-key-label').textContent = cfg.label;
    document.getElementById('ai-api-key').placeholder = cfg.placeholder;
    document.getElementById('ai-api-link').href = cfg.link;
    document.getElementById('ai-api-link').textContent = cfg.linkText;
}

async function loadAIModels() {
    try {
        const response = await fetch('/api/ai_config');
        const data = await response.json();
        aiModelsData = data;
        renderAIModelsList(data);
        updateAIActiveDisplay(data.active, data.models);
    } catch (error) {
        console.error('Error loading AI models:', error);
    }
}

function updateAIActiveDisplay(activeName, models) {
    const nameEl = document.getElementById('ai-active-model-name');
    const badgeEl = document.getElementById('ai-active-model-badge');
    if (activeName && models && models[activeName]) {
        const provider = models[activeName].provider;
        const providerLabels = { claude: 'Claude', chatgpt: 'ChatGPT', gemini: 'Gemini', local: 'Local' };
        nameEl.textContent = activeName;
        badgeEl.textContent = (providerLabels[provider] || provider).toUpperCase();
        badgeEl.style.backgroundColor = '#28a745';
    } else {
        nameEl.textContent = 'None';
        badgeEl.textContent = 'NOT SET';
        badgeEl.style.backgroundColor = '#6c757d';
    }
}

function renderAIModelsList(data) {
    const container = document.getElementById('ai-models-list');
    const modelNames = Object.keys(data.models || {});
    if (modelNames.length === 0) {
        container.innerHTML = '<span style="color: var(--text-secondary); font-style: italic;">No models configured. Add one below.</span>';
        return;
    }
    container.innerHTML = '';
    const providerLabels = { claude: 'Claude', chatgpt: 'ChatGPT', gemini: 'Gemini', local: 'Local' };
    modelNames.forEach(name => {
        const model = data.models[name];
        const isActive = data.active === name;
        const btn = document.createElement('button');
        btn.className = 'camera-btn ' + (isActive ? 'camera-btn-success' : 'camera-btn-secondary');
        btn.style.cssText = 'display: flex; flex-direction: column; align-items: center; padding: 8px 12px; min-width: 140px; cursor: pointer; position: relative;';
        btn.innerHTML = `
            <span style="font-weight: bold;">${name}</span>
            <span style="font-size: 10px; opacity: 0.8;">${providerLabels[model.provider] || model.provider}</span>
            <span style="font-size: 9px; opacity: 0.7;">${model.api_key_masked || '***'}</span>
            ${isActive ? '<span style="font-size: 9px; color: #90ee90;">ACTIVE</span>' : ''}
        `;
        btn.onclick = () => activateAIModel(name);
        btn.title = isActive ? 'Currently active' : 'Click to activate';

        // Delete button (small X in corner)
        const delBtn = document.createElement('span');
        delBtn.textContent = 'x';
        delBtn.style.cssText = 'position: absolute; top: 2px; right: 6px; font-size: 12px; cursor: pointer; opacity: 0.6; color: #ff6b6b;';
        delBtn.title = 'Delete this model';
        delBtn.onclick = (e) => { e.stopPropagation(); deleteAIModel(name); };
        btn.appendChild(delBtn);

        container.appendChild(btn);
    });
}

async function saveAIModel() {
    const name = document.getElementById('ai-model-name').value.trim();
    const provider = document.getElementById('ai-model-provider').value;
    const apiKey = document.getElementById('ai-api-key').value.trim();
    // 3.21.23 — optional overrides
    const baseUrl = (document.getElementById('ai-base-url')?.value || '').trim();
    const modelId = (document.getElementById('ai-model-id')?.value || '').trim();

    if (!name) { alert('Please enter a model name'); return; }
    if (!apiKey) { alert('Please enter an API key or endpoint'); return; }
    var _b = _btnLoading();
    try {
        const response = await fetch('/api/ai_config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, provider, api_key: apiKey, base_url: baseUrl, model_id: modelId })
        });
        const data = await response.json();
        if (response.ok) {
            clearAIModelForm();
            loadAIModels();
        } else {
            alert('Failed to save: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally { _btnDone(_b); }
}

async function activateAIModel(name) {
    try {
        const response = await fetch('/api/ai_config/activate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        if (response.ok) { loadAIModels(); }
        else { const d = await response.json(); alert('Failed: ' + (d.error || 'Unknown')); }
    } catch (error) { alert('Error: ' + error.message); }
}

async function deleteAIModel(name) {
    if (!confirm('Delete model "' + name + '"?')) return;
    try {
        const response = await fetch('/api/ai_config/' + encodeURIComponent(name), { method: 'DELETE' });
        if (response.ok) { loadAIModels(); }
        else { const d = await response.json(); alert('Failed: ' + (d.error || 'Unknown')); }
    } catch (error) { alert('Error: ' + error.message); }
}

function clearAIModelForm() {
    document.getElementById('ai-model-name').value = '';
    document.getElementById('ai-api-key').value = '';
    document.getElementById('ai-model-provider').selectedIndex = 0;
    const bu = document.getElementById('ai-base-url'); if (bu) bu.value = '';
    const mi = document.getElementById('ai-model-id'); if (mi) mi.value = '';
    updateAIModelFields();
}

// Database Profile Functions
let dbProfilesData = { profiles: {}, active: null };

async function loadDBProfiles() {
    try {
        const response = await fetch('/api/db_config');
        const data = await response.json();
        dbProfilesData = data;
        renderDBProfilesList(data);
        updateDBActiveDisplay(data.active, data.profiles);
    } catch (error) {
        console.error('Error loading DB profiles:', error);
    }
}

function updateDBActiveDisplay(activeName, profiles) {
    const nameEl = document.getElementById('db-active-profile-name');
    const badgeEl = document.getElementById('db-active-profile-badge');
    if (activeName && profiles && profiles[activeName]) {
        const p = profiles[activeName];
        nameEl.textContent = activeName;
        badgeEl.textContent = `${p.host}:${p.port}`;
        badgeEl.style.backgroundColor = '#28a745';
    } else {
        nameEl.textContent = 'None';
        badgeEl.textContent = 'NOT SET';
        badgeEl.style.backgroundColor = '#6c757d';
    }
}

function renderDBProfilesList(data) {
    const container = document.getElementById('db-profiles-list');
    const names = Object.keys(data.profiles || {});
    if (names.length === 0) {
        container.innerHTML = '<span style="color: var(--text-secondary); font-style: italic;">No profiles configured. Add one below.</span>';
        return;
    }
    container.innerHTML = '';
    names.forEach(name => {
        const p = data.profiles[name];
        const isActive = data.active === name;
        const btn = document.createElement('button');
        btn.className = 'camera-btn ' + (isActive ? 'camera-btn-success' : 'camera-btn-secondary');
        btn.style.cssText = 'display: flex; flex-direction: column; align-items: center; padding: 8px 12px; min-width: 140px; cursor: pointer; position: relative;';
        btn.innerHTML = `
            <span style="font-weight: bold;">${name}</span>
            <span style="font-size: 10px; opacity: 0.8;">${p.host}:${p.port}</span>
            <span style="font-size: 9px; opacity: 0.7;">${p.database} (${p.user})</span>
            ${isActive ? '<span style="font-size: 9px; color: #90ee90;">ACTIVE</span>' : ''}
        `;
        btn.onclick = () => activateDBProfile(name);
        btn.title = isActive ? 'Currently active' : 'Click to activate';

        const delBtn = document.createElement('span');
        delBtn.textContent = 'x';
        delBtn.style.cssText = 'position: absolute; top: 2px; right: 6px; font-size: 12px; cursor: pointer; opacity: 0.6; color: #ff6b6b;';
        delBtn.title = 'Delete this profile';
        delBtn.onclick = (e) => { e.stopPropagation(); deleteDBProfile(name); };
        btn.appendChild(delBtn);

        container.appendChild(btn);
    });
}

async function saveDBProfile() {
    const name = document.getElementById('db-profile-name').value.trim();
    const host = document.getElementById('db-host').value.trim();
    const port = parseInt(document.getElementById('db-port').value) || 5432;
    const database = document.getElementById('db-database').value.trim();
    const user = document.getElementById('db-user').value.trim();
    const password = document.getElementById('db-password').value.trim();

    if (!name) { alert('Please enter a profile name'); return; }
    if (!host) { alert('Please enter a host'); return; }
    var _b = _btnLoading();
    try {
        const response = await fetch('/api/db_config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, host, port, database, user, password })
        });
        const data = await response.json();
        if (response.ok) {
            clearDBProfileForm();
            loadDBProfiles();
        } else {
            alert('Failed to save: ' + (data.error || 'Unknown error'));
        }
    } catch (error) { alert('Error: ' + error.message); }
    finally { _btnDone(_b); }
}

async function activateDBProfile(name) {
    try {
        const response = await fetch('/api/db_config/activate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        if (response.ok) { loadDBProfiles(); }
        else { const d = await response.json(); alert('Failed: ' + (d.error || 'Unknown')); }
    } catch (error) { alert('Error: ' + error.message); }
}

async function deleteDBProfile(name) {
    if (!confirm('Delete profile "' + name + '"?')) return;
    try {
        const response = await fetch('/api/db_config/' + encodeURIComponent(name), { method: 'DELETE' });
        if (response.ok) { loadDBProfiles(); }
        else { const d = await response.json(); alert('Failed: ' + (d.error || 'Unknown')); }
    } catch (error) { alert('Error: ' + error.message); }
}

function clearDBProfileForm() {
    document.getElementById('db-profile-name').value = '';
    document.getElementById('db-host').value = '';
    document.getElementById('db-port').value = '5432';
    document.getElementById('db-database').value = '';
    document.getElementById('db-user').value = '';
    document.getElementById('db-password').value = '';
}

// Load chat history from localStorage
function loadChatHistory() {
    const chatMessages = document.getElementById('ai-chat-messages');
    const history = JSON.parse(localStorage.getItem('ai_chat_history') || '[]');

    // Clear existing messages (except welcome message)
    const welcomeMsg = chatMessages.querySelector('div');
    chatMessages.innerHTML = '';
    if (welcomeMsg) chatMessages.appendChild(welcomeMsg);

    // Restore messages
    history.forEach(msg => {
        const msgDiv = document.createElement('div');
        if (msg.type === 'user') {
            msgDiv.style.cssText = 'margin-bottom: 10px; padding: 10px; background: rgba(59, 130, 246, 0.2); border: 1px solid rgba(59, 130, 246, 0.4); border-radius: 8px; text-align: right; color: var(--text-primary);';
            msgDiv.innerHTML = `<strong>You:</strong> ${msg.content}`;
        } else if (msg.type === 'ai') {
            msgDiv.style.cssText = 'margin-bottom: 10px; padding: 15px; background: rgba(30, 41, 59, 0.6); border-radius: 8px; border-left: 4px solid var(--primary-color); color: var(--text-primary); line-height: 1.6;';
            msgDiv.innerHTML = `<div style="margin-bottom: 8px;"><strong style="color: var(--primary-color);">🤖 AI:</strong></div><div>${msg.content}</div>`;
        } else if (msg.type === 'error') {
            msgDiv.style.cssText = 'margin-bottom: 10px; padding: 10px; background: rgba(239, 68, 68, 0.2); border: 1px solid var(--danger-color); border-radius: 8px; color: var(--danger-color);';
            msgDiv.innerHTML = `<strong>Error:</strong> ${msg.content}`;
        }
        chatMessages.appendChild(msgDiv);
    });

    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Save message to history
function saveChatMessage(type, content) {
    const history = JSON.parse(localStorage.getItem('ai_chat_history') || '[]');
    history.push({ type, content, timestamp: new Date().toISOString() });

    // Keep only last 50 messages to avoid localStorage quota
    if (history.length > 50) {
        history.splice(0, history.length - 50);
    }

    localStorage.setItem('ai_chat_history', JSON.stringify(history));
}

// Clear chat history
function clearChatHistory() {
    if (confirm('Clear all chat history?')) {
        localStorage.removeItem('ai_chat_history');
        location.reload();
    }
}

async function sendAIMessage() {
    const input = document.getElementById('ai-input');
    const message = input.value.trim();

    if (!message) return;

    const chatMessages = document.getElementById('ai-chat-messages');

    // Add user message
    const userMsg = document.createElement('div');
    userMsg.style.cssText = 'margin-bottom: 10px; padding: 10px; background: rgba(59, 130, 246, 0.2); border: 1px solid rgba(59, 130, 246, 0.4); border-radius: 8px; text-align: right; color: var(--text-primary);';
    userMsg.innerHTML = `<strong>You:</strong> ${message}`;
    chatMessages.appendChild(userMsg);

    // Save to history
    saveChatMessage('user', message);

    // Clear input
    input.value = '';

    // Add loading indicator
    const loadingMsg = document.createElement('div');
    loadingMsg.id = 'ai-loading';
    loadingMsg.style.cssText = 'margin-bottom: 10px; padding: 10px; background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(51, 65, 85, 0.6); border-radius: 8px; color: var(--text-secondary);';
    loadingMsg.innerHTML = '<strong>AI:</strong> <span style="animation: pulse 1.5s infinite;">Thinking...</span>';
    chatMessages.appendChild(loadingMsg);

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch('/api/ai_query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: message })
        });

        const data = await response.json();

        // Remove loading indicator
        loadingMsg.remove();

        // Add AI response
        const aiMsg = document.createElement('div');
        aiMsg.style.cssText = 'margin-bottom: 10px; padding: 15px; background: rgba(30, 41, 59, 0.6); border-radius: 8px; border-left: 4px solid var(--primary-color); color: var(--text-primary); line-height: 1.6;';

        // Format the response with better line breaks and preserve markdown-like formatting
        let formattedResponse = (data.response || data.error || 'No response')
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/• /g, '&nbsp;&nbsp;• ');

        aiMsg.innerHTML = `<div style="margin-bottom: 8px;"><strong style="color: var(--primary-color);">🤖 AI:</strong></div><div>${formattedResponse}</div>`;
        chatMessages.appendChild(aiMsg);

        // Save to history
        saveChatMessage('ai', formattedResponse);

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;

    } catch (error) {
        console.error('Error querying AI:', error);
        loadingMsg.remove();

        const errorMsg = document.createElement('div');
        errorMsg.style.cssText = 'margin-bottom: 10px; padding: 10px; background: rgba(239, 68, 68, 0.2); border: 1px solid var(--danger-color); border-radius: 8px; color: var(--danger-color);';
        const errorContent = 'Failed to get AI response. Please check your configuration.';
        errorMsg.innerHTML = `<strong>Error:</strong> ${errorContent}`;
        chatMessages.appendChild(errorMsg);

        // Save to history
        saveChatMessage('error', errorContent);

        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

// Load history, AI models, and DB profiles on page load
document.addEventListener('DOMContentLoaded', function() {
    loadAIModels();
    loadDBProfiles();
    // Load history if on AI tab
    if (document.getElementById('tab-ai').style.display !== 'none') {
        loadChatHistory();
    }
});
