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
    objectConfidence: {}   // { "socket": 1, ... } - min confidence %
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
                }
            });
            saveAudioSettings();
            updateObjectsList();
        }
    } catch (e) {
        console.error('Could not fetch model classes:', e);
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

// Update the objects list in the UI (merged show/confidence + audio)
function updateObjectsList() {
    const listEl = document.getElementById('audio-objects-list');
    if (!listEl) return;

    listEl.innerHTML = '';

    if (detectedObjectClasses.size === 0) {
        listEl.innerHTML = '<div style="padding: 15px; background: rgba(15, 23, 42, 0.5); border-radius: 6px; text-align: center; color: var(--text-secondary); font-style: italic;">No objects detected yet. Start detection to see objects here.</div>';
        return;
    }

    const sortedObjects = Array.from(detectedObjectClasses).sort();
    sortedObjects.forEach(objectName => {
        const isShown = audioSettings.showObjects[objectName] !== false;
        const confidence = audioSettings.objectConfidence[objectName] !== undefined ? audioSettings.objectConfidence[objectName] : 1;
        const isNarrate = audioSettings.narrateObjects[objectName] || false;
        const isBeep = audioSettings.enabledObjects[objectName] || false;
        const beepSound = audioSettings.objectBeepSounds?.[objectName] || 'sine';
        const safeName = objectName.replace(/'/g, "\\'");

        const card = document.createElement('div');
        card.style.cssText = `padding: 10px; background: rgba(51, 65, 85, 0.4); border-radius: 6px; border: 1px solid ${isShown ? 'rgba(34,197,94,0.4)' : 'rgba(51, 65, 85, 0.6)'};`;

        card.innerHTML = `
            <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 8px; font-size: 14px;">${objectName}</div>
            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 6px;">
                <span style="font-size: 12px; color: var(--text-secondary); white-space: nowrap;">Min conf:</span>
                <input type="number" value="${confidence}" min="0" max="100" step="1"
                    onchange="updateObjectConfidence('${safeName}', this.value)"
                    style="width: 55px; padding: 3px 5px; background: rgba(30,41,59,0.6); color: var(--text-primary); border: 1px solid rgba(51,65,85,0.6); border-radius: 4px; font-size: 12px; text-align: center;">
                <span style="font-size: 12px; color: var(--text-secondary);">%</span>
            </div>
            <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 6px;">
                <label style="display: flex; align-items: center; gap: 4px; cursor: pointer; font-size: 12px;" title="Show bounding box">
                    <input type="checkbox" ${isShown ? 'checked' : ''} onchange="toggleObjectShow('${safeName}')" style="width: 14px; height: 14px; cursor: pointer;">
                    Show
                </label>
                <label style="display: flex; align-items: center; gap: 4px; cursor: pointer; font-size: 12px;" title="Voice narration">
                    <input type="checkbox" ${isNarrate ? 'checked' : ''} onchange="toggleObjectNarrate('${safeName}')" style="width: 14px; height: 14px; cursor: pointer;">
                    Narrate
                </label>
                <label style="display: flex; align-items: center; gap: 4px; cursor: pointer; font-size: 12px;" title="Beep alert">
                    <input type="checkbox" ${isBeep ? 'checked' : ''} onchange="toggleObjectBeep('${safeName}')" style="width: 14px; height: 14px; cursor: pointer;">
                    Beep
                </label>
            </div>
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

// Toggle show/hide bounding box for object
function toggleObjectShow(objectName) {
    const current = audioSettings.showObjects[objectName] !== false;
    audioSettings.showObjects[objectName] = !current;
    saveAudioSettings();
    updateObjectsList();
}

// Toggle narrate for object
function toggleObjectNarrate(objectName) {
    audioSettings.narrateObjects[objectName] = !audioSettings.narrateObjects[objectName];
    saveAudioSettings();
    updateObjectsList();
}

// Update confidence threshold for object
function updateObjectConfidence(objectName, value) {
    audioSettings.objectConfidence[objectName] = Math.max(0, Math.min(100, parseFloat(value) || 1));
    saveAudioSettings();
}

// ============== EJECTION PROCEDURES ==============
let procedures = [];

function addProcedure() {
    procedures.push({
        id: 'proc_' + Date.now(),
        name: 'New Procedure',
        enabled: true,
        logic: 'any',
        rules: []
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
    proc.rules.push({ object: firstClass, condition: 'present', min_confidence: 30 });
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
}

function updateRuleField(procId, ruleIndex, field, value) {
    const proc = procedures.find(p => p.id === procId);
    if (!proc || !proc.rules[ruleIndex]) return;
    if (field === 'min_confidence') {
        proc.rules[ruleIndex][field] = Math.max(0, Math.min(100, parseInt(value) || 0));
    } else {
        proc.rules[ruleIndex][field] = value;
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
        const rulesHtml = proc.rules.map((rule, ri) => `
            <div style="display: flex; gap: 6px; align-items: center; padding: 6px; background: rgba(15, 23, 42, 0.4); border-radius: 4px; flex-wrap: wrap;">
                <select onchange="updateRuleField('${proc.id}', ${ri}, 'object', this.value)"
                    style="padding: 3px 6px; background: rgba(30,41,59,0.6); color: var(--text-primary); border: 1px solid rgba(51,65,85,0.6); border-radius: 4px; font-size: 11px; min-width: 100px;">
                    ${Array.from(detectedObjectClasses).sort().map(name =>
                        '<option value="' + name + '" ' + (rule.object === name ? 'selected' : '') + '>' + name + '</option>'
                    ).join('')}
                </select>
                <select onchange="updateRuleField('${proc.id}', ${ri}, 'condition', this.value)"
                    style="padding: 3px 6px; background: rgba(30,41,59,0.6); color: var(--text-primary); border: 1px solid rgba(51,65,85,0.6); border-radius: 4px; font-size: 11px;">
                    <option value="present" ${rule.condition === 'present' ? 'selected' : ''}>Present</option>
                    <option value="not_present" ${rule.condition === 'not_present' ? 'selected' : ''}>Not Present</option>
                </select>
                <span style="font-size: 11px; color: var(--text-secondary);">min:</span>
                <input type="number" value="${rule.min_confidence}" min="0" max="100" step="1"
                    onchange="updateRuleField('${proc.id}', ${ri}, 'min_confidence', this.value)"
                    style="width: 50px; padding: 3px 5px; background: rgba(30,41,59,0.6); color: var(--text-primary); border: 1px solid rgba(51,65,85,0.6); border-radius: 4px; font-size: 11px; text-align: center;">
                <span style="font-size: 11px; color: var(--text-secondary);">%</span>
                <button onclick="removeRule('${proc.id}', ${ri})"
                    style="padding: 2px 6px; background: rgba(239,68,68,0.3); color: #ef4444; border: 1px solid rgba(239,68,68,0.4); border-radius: 4px; cursor: pointer; font-size: 11px;"
                    title="Remove rule">X</button>
            </div>
        `).join('');

        return `
            <div style="padding: 12px; background: rgba(51, 65, 85, 0.4); border-radius: 6px; margin-bottom: 8px; border: 1px solid ${proc.enabled ? 'rgba(239,68,68,0.4)' : 'rgba(51,65,85,0.6)'};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; gap: 8px; flex-wrap: wrap;">
                    <input type="text" value="${proc.name}"
                        onchange="updateProcedureField('${proc.id}', 'name', this.value)"
                        style="flex: 1; min-width: 120px; padding: 4px 8px; background: rgba(30,41,59,0.6); color: var(--text-primary); border: 1px solid rgba(51,65,85,0.6); border-radius: 4px; font-size: 13px; font-weight: 600;">
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <label style="display: flex; align-items: center; gap: 4px; font-size: 12px; color: ${proc.enabled ? '#ef4444' : 'var(--text-secondary)'};">
                            <input type="checkbox" ${proc.enabled ? 'checked' : ''}
                                onchange="updateProcedureField('${proc.id}', 'enabled', this.checked)"
                                style="width: 14px; height: 14px; cursor: pointer;">
                            Enabled
                        </label>
                        <select onchange="updateProcedureField('${proc.id}', 'logic', this.value)"
                            style="padding: 3px 6px; background: rgba(30,41,59,0.6); color: var(--text-primary); border: 1px solid rgba(51,65,85,0.6); border-radius: 4px; font-size: 11px;">
                            <option value="any" ${proc.logic === 'any' ? 'selected' : ''}>ANY rule (OR)</option>
                            <option value="all" ${proc.logic === 'all' ? 'selected' : ''}>ALL rules (AND)</option>
                        </select>
                        <button onclick="removeProcedure('${proc.id}')"
                            style="padding: 3px 8px; background: rgba(239,68,68,0.2); color: #ef4444; border: 1px solid rgba(239,68,68,0.4); border-radius: 4px; cursor: pointer; font-size: 12px;">Delete</button>
                    </div>
                </div>
                <div style="display: flex; flex-direction: column; gap: 4px; margin-bottom: 8px;">
                    ${rulesHtml || '<div style="padding: 8px; text-align: center; color: var(--text-secondary); font-size: 12px; font-style: italic;">No rules. Add a rule to define when this procedure triggers.</div>'}
                </div>
                <button onclick="addRule('${proc.id}')"
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
        const timestamp = new Date().getTime();
        slide.src = `/timeline_image?page=${page}&t=${timestamp}`;
        updateCounter();
        panzoom.reset();
    }

    // Auto-refresh functionality (reload latest page)
    async function refreshImage() {
        if (autoUpdate) {
            await updatePageCount();
            currentPage = 0; // Always show latest page
            const timestamp = new Date().getTime();
            slide.src = `/timeline_image?page=0&t=${timestamp}`;
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

    // Toggle auto-update
    toggleBtn.onclick = () => {
        autoUpdate = !autoUpdate;
        if (autoUpdate) {
            toggleBtn.textContent = "Stop";
            toggleBtn.classList.remove("stopped");
            currentPage = 0;
            startAutoRefresh();
            refreshImage();
        } else {
            toggleBtn.textContent = "Resume";
            toggleBtn.classList.add("stopped");
            stopAutoRefresh();
            counter.textContent = "Paused";
        }
    };

    // Navigation buttons (page navigation)
    document.getElementById("timeline-first").addEventListener('click', () => {
        stopAutoRefresh();
        autoUpdate = false;
        toggleBtn.textContent = "Resume";
        loadPage(totalPages - 1); // Oldest page
    });

    document.getElementById("timeline-prev").addEventListener('click', () => {
        stopAutoRefresh();
        autoUpdate = false;
        toggleBtn.textContent = "Resume";
        loadPage(Math.min(totalPages - 1, currentPage + 1)); // Older page
    });

    document.getElementById("timeline-next").addEventListener('click', () => {
        stopAutoRefresh();
        autoUpdate = false;
        toggleBtn.textContent = "Resume";
        loadPage(Math.max(0, currentPage - 1)); // Newer page
    });

    document.getElementById("timeline-last").addEventListener('click', () => {
        stopAutoRefresh();
        autoUpdate = false;
        toggleBtn.textContent = "Resume";
        loadPage(0); // Latest page
    });

    // Update page count periodically
    setInterval(updatePageCount, 5000);
    updatePageCount();

    // Zoom controls
    document.getElementById("timeline-zoom-in").addEventListener('click', panzoom.zoomIn);
    document.getElementById("timeline-zoom-out").addEventListener('click', panzoom.zoomOut);
    document.getElementById("timeline-reset-zoom").addEventListener('click', panzoom.reset);

    // Wheel zoom
    wrapper.addEventListener("wheel", panzoom.zoomWithWheel);

    // Double-click to toggle zoom
    slide.addEventListener("dblclick", () => {
        panzoom.zoom(panzoom.getScale() === 1 ? 2 : 1);
    });

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

    // Load saved configuration from server
    async function loadSavedConfig() {
        try {
            const response = await fetch('/api/timeline_count');
            const data = await response.json();
            if (data.rows_per_page) {
                rowsSlider.value = data.rows_per_page;
                rowsValue.textContent = data.rows_per_page;
            }
            // Note: Quality is stored in timeline_config but not returned by timeline_count API
            // We'll load it from the data file via getData API
            const configResponse = await fetch('/api/getData');
            const configData = await configResponse.json();
            if (configData.timeline_config) {
                if (configData.timeline_config.image_quality) {
                    qualitySlider.value = configData.timeline_config.image_quality;
                    qualityValue.textContent = configData.timeline_config.image_quality + '%';
                }
                if (configData.timeline_config.num_rows) {
                    rowsSlider.value = configData.timeline_config.num_rows;
                    rowsValue.textContent = configData.timeline_config.num_rows;
                }
                if (configData.timeline_config.buffer_size) {
                    bufferSlider.value = configData.timeline_config.buffer_size;
                    bufferValue.textContent = configData.timeline_config.buffer_size;
                }
                // Load per-object filters into audioSettings
                if (configData.timeline_config.object_filters) {
                    const filters = configData.timeline_config.object_filters;
                    Object.keys(filters).forEach(name => {
                        detectedObjectClasses.add(name);
                        audioSettings.showObjects[name] = filters[name].show !== false;
                        audioSettings.objectConfidence[name] = Math.round((filters[name].min_confidence || 0.01) * 100);
                    });
                    updateObjectsList();
                }
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

    bufferSlider.addEventListener('input', () => {
        bufferValue.textContent = bufferSlider.value;
    });

    // Apply configuration
    applyBtn.addEventListener('click', async () => {
        const cameraOrder = document.querySelector('input[name="camera-order"]:checked').value;
        const quality = qualitySlider.value;
        const rows = rowsSlider.value;
        const buffer = bufferSlider.value;
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

            const response = await fetch('/api/timeline_config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    show_bounding_boxes: audioSettings.showBoundingBoxes !== false,
                    camera_order: cameraOrder,
                    image_quality: parseInt(quality),
                    num_rows: parseInt(rows),
                    buffer_size: parseInt(buffer),
                    image_rotation: rotation,
                    object_filters: objectFilters
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
function editShipmentId() {
    const shipmentValue = document.getElementById('shipment-value');
    const shipmentInput = document.getElementById('shipment-input');
    const shipmentButtons = document.getElementById('shipment-buttons');
    const shipmentText = document.getElementById('shipment-text');

    // Hide the display value, show input and buttons
    shipmentValue.style.display = 'none';
    shipmentInput.style.display = 'block';
    shipmentButtons.style.display = 'flex';

    // Set current value in input
    shipmentInput.value = shipmentText.textContent;
    shipmentInput.focus();
    shipmentInput.select();
}

function cancelEditShipment() {
    const shipmentValue = document.getElementById('shipment-value');
    const shipmentInput = document.getElementById('shipment-input');
    const shipmentButtons = document.getElementById('shipment-buttons');

    // Show the display value, hide input and buttons
    shipmentValue.style.display = 'flex';
    shipmentInput.style.display = 'none';
    shipmentButtons.style.display = 'none';
}

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
            const statusBox = shipmentInput.closest('.status-box-item');
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
async function saveAllServiceConfig() {
    var _b = _btnLoading();
    try {
        // Save all configuration (cameras + settings) using the correct endpoint
        const response = await fetch('/api/cameras/config/save', { method: 'POST' });

        if (response.ok) {
            const data = await response.json();
            const timestamp = new Date().toISOString();
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
        container.innerHTML = '<span style="color: #666; font-style: italic;">No models configured. Add one below.</span>';
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

    if (!name) { alert('Please enter a model name'); return; }
    if (!apiKey) { alert('Please enter an API key or endpoint'); return; }
    var _b = _btnLoading();
    try {
        const response = await fetch('/api/ai_config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, provider, api_key: apiKey })
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
        container.innerHTML = '<span style="color: #666; font-style: italic;">No profiles configured. Add one below.</span>';
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
