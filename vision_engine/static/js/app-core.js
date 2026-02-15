// Update UI with status data (used by SSE stream)
function updateStatusUI(data) {
    document.getElementById('encoder-value').textContent = data.encoder_value || 0;
    document.getElementById('speed-value').textContent = (data.ppm || 0).toFixed ? (data.ppm || 0).toFixed(2) : data.ppm || 0;
    document.getElementById('pps-value').textContent = data.pps || 0;
    document.getElementById('ok-counter').textContent = data.ok_counter || 0;
    document.getElementById('ng-counter').textContent = data.ng_counter || 0;
    document.getElementById('downtime-value').textContent = data.downtime_seconds || 0;
    document.getElementById('analog-value').textContent = data.analog_value || 0;
    document.getElementById('power-value').textContent = data.power_value || 0;
    const shipmentText = document.getElementById('shipment-text');
    if (shipmentText) {
        shipmentText.textContent = data.shipment || 'no_shipment';
    }
    document.getElementById('ejector-queue').textContent = data.ejector_queue_length || 0;
    document.getElementById('ejector-running').textContent = data.ejector_running ? 'Yes' : 'No';
    document.getElementById('ejector-enabled').textContent = data.ejector_enabled ? 'Yes' : 'No';
    document.getElementById('ejector-offset').textContent = data.ejector_offset || 0;

    // Update movement status
    const movementIndicator = document.getElementById('movement-indicator');
    const movementValue = document.getElementById('movement-value');
    if (data.is_moving) {
        movementIndicator.className = 'movement-indicator movement-moving';
        movementValue.textContent = 'Moving';
    } else {
        movementIndicator.className = 'movement-indicator movement-stopped';
        movementValue.textContent = 'Stopped';
    }

    // Update U/B/Warning status circles
    const status = data.status || {};
    const uEl = document.getElementById('U-status');
    const bEl = document.getElementById('B-status');
    const wEl = document.getElementById('warning-status');
    if (uEl) uEl.className = 'status-circle U' + (status.U ? ' active' : '');
    if (bEl) bEl.className = 'status-circle B' + (status.B ? ' active' : '');
    if (wEl) wEl.className = 'status-circle warning' + (status.warning ? ' active' : '');

    // Update serial device status
    const serialDevice = data.serial_device || {};
    const serialDot = document.getElementById('serial-device-dot');
    const serialText = document.getElementById('serial-device-text');
    const serialInfo = document.getElementById('serial-device-info');
    if (serialDot && serialText && serialInfo) {
        if (serialDevice.connected) {
            serialDot.style.background = '#28a745';
            serialText.textContent = 'Connected';
            serialText.style.color = '#28a745';
            serialInfo.textContent = `${serialDevice.port} @ ${serialDevice.baudrate} (${serialDevice.mode})`;
        } else {
            serialDot.style.background = '#dc3545';
            serialText.textContent = 'Disconnected';
            serialText.style.color = '#dc3545';
            serialInfo.textContent = `${serialDevice.port} not accessible`;
        }
    }

    // Update device status in sidebar
    const deviceStatus = document.getElementById('device-status');
    if (deviceStatus) {
        if (serialDevice.connected) {
            deviceStatus.textContent = 'Device connected';
            deviceStatus.style.color = 'var(--success-color)';
        } else {
            deviceStatus.textContent = 'Serial not connected';
            deviceStatus.style.color = 'var(--text-secondary)';
        }
    }

    // Update verbose configuration values
    const v = data.verbose_data || {};
    const setVerbose = (id, key) => {
        const el = document.getElementById(id);
        if (el) el.textContent = (v[key] === undefined || v[key] === null) ? '-' : v[key];
    };
    setVerbose('v-ood', 'OOD');
    setVerbose('v-odp', 'ODP');
    setVerbose('v-odl', 'ODL');
    setVerbose('v-oef', 'OEF');
    setVerbose('v-nod', 'NOD');
    setVerbose('v-ndp', 'NDP');
    setVerbose('v-ndl', 'NDL');
    setVerbose('v-nef', 'NEF');
    setVerbose('v-ext', 'EXT');
    setVerbose('v-bud', 'BUD');
    setVerbose('v-dwt', 'DWT');
}

// Server-Sent Events for real-time status updates
let statusEventSource = null;

function startStatusStream() {
    if (statusEventSource) {
        statusEventSource.close();
    }

    statusEventSource = new EventSource('/api/status/stream');
    const dataSource = document.getElementById('dataSource');

    statusEventSource.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);

            // Debug: Log detection events
            if (data.detection_event) {
                console.log('[SSE] Detection event received:', data.detection_event);
            }

            updateStatusUI(data);

            // Update cameras from SSE
            if (data.cameras) {
                updateCameraTable(data.cameras);
            }

            // Update inference stats from SSE
            if (data.inference) {
                const inf = data.inference;
                const avgTimeEl = document.getElementById('avg-inference-time');
                const avgIntervalEl = document.getElementById('avg-frame-interval');
                const fpsEl = document.getElementById('inference-fps');
                if (avgTimeEl) avgTimeEl.textContent = inf.avg_time_ms.toFixed(1) + ' ms';
                if (avgIntervalEl) avgIntervalEl.textContent = inf.avg_interval_ms.toFixed(1) + ' ms';
                if (fpsEl) fpsEl.textContent = inf.fps.toFixed(1) + ' FPS';

                // Process detections for audio alerts
                if (inf.last_detection) {
                    processDetectionForAudio(inf.last_detection);
                }
            }

            // Update health/infrastructure from SSE
            if (data.health) {
                const updateInfraStatus = (prefix, available) => {
                    const dot = document.getElementById(`${prefix}-dot`);
                    const text = document.getElementById(`${prefix}-text`);
                    if (dot && text) {
                        dot.style.background = available ? '#28a745' : '#6c757d';
                        text.textContent = available ? 'Connected' : 'Not Available';
                        text.style.color = available ? '#28a745' : '#6c757d';
                    }
                };
                updateInfraStatus('redis', data.health.redis === 'connected');

                // Update DB status from SSE health
                const dbStatusSSE = document.getElementById('db-status');
                if (dbStatusSSE && data.health.db) {
                    const dbOk = data.health.db === 'connected';
                    dbStatusSSE.textContent = dbOk ? 'Connected' : 'Disconnected';
                    dbStatusSSE.style.background = dbOk ? '#28a745' : '#dc3545';
                }

                const restartWarning = document.getElementById('gradio-restart-warning');
                if (restartWarning) {
                    restartWarning.style.display = data.health.gradio_needs_restart ? 'inline' : 'none';
                }
            }

            // Process detection events for audio alerts
            if (data.detection_event && data.detection_event.details && data.detection_event.details.detections) {
                processDetectionForAudio(data.detection_event.details);
            }

            // Process detections for audio alerts (if detections come directly in data)
            if (data.detections) {
                processDetectionForAudio(data);
            }
        } catch (e) {
            console.error('Error parsing SSE data:', e);
        }
    };

    statusEventSource.onerror = function(error) {
        console.log('SSE reconnecting...');
        if (dataSource) { dataSource.textContent = 'RECONNECTING'; dataSource.className = 'data-source'; }
    };

    statusEventSource.onopen = function() {
        console.log('SSE connected');
        if (dataSource) { dataSource.textContent = 'LIVE'; dataSource.className = 'data-source serial'; }
    };
}

// Timeline WebSocket — event-driven push (replaces polling)
let timelineWs = null;
let timelineWsPage = 0;
function connectTimelineWs() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    timelineWs = new WebSocket(`${proto}//${location.host}/ws/timeline`);
    timelineWs.binaryType = 'arraybuffer';
    timelineWs.onmessage = (e) => {
        if (typeof e.data === 'string') return; // skip text messages
        const blob = new Blob([e.data], {type: 'image/jpeg'});
        const url = URL.createObjectURL(blob);
        const img = document.getElementById('timeline-slide');
        if (img) {
            if (img._blobUrl) URL.revokeObjectURL(img._blobUrl);
            img._blobUrl = url;
            img.src = url;
        }
    };
    timelineWs.onclose = () => {
        timelineWs = null;
        setTimeout(connectTimelineWs, 2000);
    };
    timelineWs.onerror = () => {}; // onclose will handle reconnect
}
function startTimelineRefresh() { connectTimelineWs(); }
// Send page number to server when user navigates timeline
function timelineWsSendPage(page) {
    timelineWsPage = page;
    if (timelineWs && timelineWs.readyState === WebSocket.OPEN) {
        timelineWs.send(String(page));
    }
}

// Fetch health data (infrastructure status) - called periodically
function fetchHealth() {
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            const restartWarning = document.getElementById('gradio-restart-warning');
            if (restartWarning) {
                restartWarning.style.display = data.gradio_needs_restart ? 'inline' : 'none';
            }
            if (data.cameras) {
                updateCameraTable(data.cameras);
            }

            // Update Redis status
            const redisDot = document.getElementById('redis-dot');
            const redisText = document.getElementById('redis-text');
            if (redisDot && redisText) {
                const redisOk = data.redis === 'connected';
                redisDot.style.background = redisOk ? '#28a745' : '#6c757d';
                redisText.textContent = redisOk ? 'Connected' : 'Not Available';
                redisText.style.color = redisOk ? '#28a745' : '#6c757d';
            }

            // Update DB status
            const dbStatusEl = document.getElementById('db-status');
            if (dbStatusEl) {
                const dbOk = data.db === 'connected';
                dbStatusEl.textContent = dbOk ? 'Connected' : 'Disconnected';
                dbStatusEl.style.background = dbOk ? '#28a745' : '#dc3545';
            }

            // Update Pipeline status (YOLO/Gradio)
            const pipelineDot = document.getElementById('pipeline-dot');
            const pipelineText = document.getElementById('pipeline-text');
            if (pipelineDot && pipelineText) {
                const pipelineOk = data.yolo === 'connected';
                const gradioStatus = data.gradio_status || '';

                if (pipelineOk) {
                    // Show more detailed status for Gradio
                    if (gradioStatus === 'healthy') {
                        pipelineDot.style.background = '#28a745';
                        pipelineText.textContent = 'Healthy';
                        pipelineText.style.color = '#28a745';
                    } else if (gradioStatus === 'warning' || gradioStatus === 'idle') {
                        pipelineDot.style.background = '#ffc107';
                        pipelineText.textContent = gradioStatus === 'idle' ? 'Idle' : 'Warning';
                        pipelineText.style.color = '#856404';
                    } else {
                        pipelineDot.style.background = '#28a745';
                        pipelineText.textContent = 'Connected';
                        pipelineText.style.color = '#28a745';
                    }
                } else {
                    // Not connected
                    if (gradioStatus === 'offline' || gradioStatus === 'stale') {
                        pipelineDot.style.background = '#dc3545';
                        pipelineText.textContent = gradioStatus === 'stale' ? 'Stale' : 'Offline';
                        pipelineText.style.color = '#dc3545';
                    } else {
                        pipelineDot.style.background = '#6c757d';
                        pipelineText.textContent = 'Not Available';
                        pipelineText.style.color = '#6c757d';
                    }
                }
            }
        })
        .catch(error => console.error('Error fetching health:', error));
}

// Legacy polling fallback (not used with SSE)
function fetchStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => updateStatusUI(data))
        .catch(error => console.error('Error fetching status:', error));
}

// Separate function to fetch config - only called once on page load
function fetchConfig() {
    fetch('/config')
        .then(response => response.json())
        .then(data => {
            // Update counter service configuration inputs with current values
            if (data.ejector) {
                document.getElementById('ejector-enabled-input').value = data.ejector.enabled ? 'true' : 'false';
                document.getElementById('ejector-offset-input').value = data.ejector.offset || 0;
                document.getElementById('ejector-duration-input').value = data.ejector.duration || 0.4;
                document.getElementById('ejector-poll-input').value = data.ejector.poll_interval || 0.03;
            }
            if (data.capture) {
                document.getElementById('time-between-packages-input').value = data.capture.time_between_packages || 0.305;
                document.getElementById('capture-mode-input').value = data.capture.mode || 'single';
            }
            // Image processing configuration
            if (data.image_processing) {
                document.getElementById('parent-object-list-input').value = (data.image_processing.parent_object_list || []).join(',');
                document.getElementById('remove-raw-image-input').value = data.image_processing.remove_raw_image_when_dm_decoded ? 'true' : 'false';
            }
            // DataMatrix configuration
            if (data.datamatrix) {
                document.getElementById('dm-chars-sizes-input').value = (data.datamatrix.chars_sizes || [13,19,26]).join(',');
                document.getElementById('dm-confidence-input').value = data.datamatrix.confidence_threshold || 0.8;
                document.getElementById('dm-overlap-input').value = data.datamatrix.overlap_threshold || 0.2;
            }
            // Class count check configuration
            if (data.class_count_check) {
                document.getElementById('check-class-enabled-input').value = data.class_count_check.enabled ? 'true' : 'false';
                document.getElementById('check-class-classes-input').value = (data.class_count_check.classes || []).join(',');
                document.getElementById('check-class-confidence-input').value = data.class_count_check.confidence || 0.5;
            }
            // Light control configuration
            if (data.light_control) {
                document.getElementById('light-status-check-input').value = data.light_control.status_check_enabled ? 'true' : 'false';
            }
            // Histogram configuration
            if (data.histogram) {
                document.getElementById('histogram-enabled-input').value = data.histogram.enabled ? 'true' : 'false';
                document.getElementById('histogram-save-image-input').value = data.histogram.save_image ? 'true' : 'false';
            }
            // Store annotation configuration
            if (data.store_annotation) {
                const saEl = document.getElementById('store-annotation-enabled-input');
                if (saEl) saEl.value = data.store_annotation.enabled ? 'true' : 'false';
                const pgEl = document.getElementById('postgres-info');
                if (pgEl) pgEl.textContent =
                    `PostgreSQL: ${data.store_annotation.postgres_host}:${data.store_annotation.postgres_port}/${data.store_annotation.postgres_db}`;
            }
            // Serial configuration
            if (data.serial) {
                document.getElementById('watcher-usb-input').value = data.serial.port || '/dev/ttyUSB0';
                document.getElementById('baud-rate-input').value = data.serial.baud_rate || 57600;
                document.getElementById('serial-mode-input').value = data.serial.mode || 'legacy';
            }
            // Infrastructure configuration
            if (data.infrastructure) {
                document.getElementById('redis-host-input').value = data.infrastructure.redis_host || 'redis';
                document.getElementById('redis-port-input').value = data.infrastructure.redis_port || 6379;
                document.getElementById('yolo-url-input').value = data.infrastructure.yolo_url || 'http://yolo_inference:4442/v1/object-detection/yolov5s/detect/';
                document.getElementById('gradio-confidence-input').value = data.infrastructure.gradio_confidence || 0.25;
            }
        })
        .catch(error => console.error('Error fetching config:', error));
}

// Data file functions
async function loadDataFile() {
    try {
        const response = await fetch('/api/data-file');
        const data = await response.json();

        if (response.ok) {
            document.getElementById('data-file-path').textContent = data.file_path;
            document.getElementById('data-file-content').value = data.content;
        } else {
            document.getElementById('data-file-path').textContent = 'Error loading';
            document.getElementById('data-file-content').value = 'Error: ' + (data.detail || 'Unknown error');
        }
    } catch (error) {
        document.getElementById('data-file-path').textContent = 'Error';
        document.getElementById('data-file-content').value = 'Error: ' + error.message;
    }
}

async function saveDataFile() {
    const responseEl = document.getElementById('data-file-response');
    const content = document.getElementById('data-file-content').value;

    try {
        // Validate JSON first
        JSON.parse(content);

        const response = await fetch('/api/data-file', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ content: content })
        });

        const data = await response.json();

        if (response.ok) {
            responseEl.textContent = `Saved! ${data.entries_count} entries. Backup: ${data.backup_path}`;
            responseEl.className = 'control-response success';
        } else {
            responseEl.textContent = `Error: ${data.detail || 'Unknown error'}`;
            responseEl.className = 'control-response error';
        }

        setTimeout(() => {
            responseEl.textContent = '';
            responseEl.className = 'control-response';
        }, 5000);
    } catch (error) {
        responseEl.textContent = `Error: ${error.message}`;
        responseEl.className = 'control-response error';
        setTimeout(() => {
            responseEl.textContent = '';
            responseEl.className = 'control-response';
        }, 5000);
    }
}

function formatDataFile() {
    const textarea = document.getElementById('data-file-content');
    const responseEl = document.getElementById('data-file-response');

    try {
        const parsed = JSON.parse(textarea.value);
        textarea.value = JSON.stringify(parsed, null, 2);
        responseEl.textContent = 'JSON formatted successfully';
        responseEl.className = 'control-response success';
        setTimeout(() => {
            responseEl.textContent = '';
            responseEl.className = 'control-response';
        }, 2000);
    } catch (error) {
        responseEl.textContent = `Invalid JSON: ${error.message}`;
        responseEl.className = 'control-response error';
        setTimeout(() => {
            responseEl.textContent = '';
            responseEl.className = 'control-response';
        }, 3000);
    }
}

function updateCameraTable(cameras) {
    const tbody = document.getElementById('camera-table-body');
    const container = document.getElementById('legacy-camera-status-container');
    tbody.innerHTML = '';

    const cameraNames = {
        'cam_1': 'Camera 1',
        'cam_2': 'Camera 2',
        'cam_3': 'Camera 3',
        'cam_4': 'Camera 4'
    };

    // Only show the table if there are cameras
    const hasCameras = Object.keys(cameras).length > 0;
    container.style.display = hasCameras ? 'block' : 'none';

    if (!hasCameras) {
        return;
    }

    for (const [key, status] of Object.entries(cameras)) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${cameraNames[key] || key}</td>
            <td class="${status ? 'camera-ok' : 'camera-fail'}">${status ? 'OK' : 'FAIL'}</td>
        `;
        tbody.appendChild(row);
    }
}

async function fetchGradioModels() {
    try {
        const response = await fetch('/api/gradio/models');
        const data = await response.json();

        if (data.models && Array.isArray(data.models)) {
            const modelSelect = document.getElementById('gradio-model-select');
            modelSelect.innerHTML = '';  // Clear existing options

            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                modelSelect.appendChild(option);
            });

            console.log(`Loaded ${data.models.length} models from ${data.source}`);
        }
    } catch (error) {
        console.error('Failed to fetch Gradio models:', error);
    }
}

function updateAPITypeFields() {
    const apiTypeEl = document.getElementById('api-type-select');
    if (!apiTypeEl) return;  // Element not found, exit early

    const apiType = apiTypeEl.value;
    const urlLabel = document.getElementById('yolo-url-label');
    const urlInput = document.getElementById('yolo-url-input');
    const modelSelector = document.getElementById('gradio-model-selector');
    const confidenceField = document.getElementById('gradio-confidence-field');

    if (apiType === 'gradio') {
        urlLabel.textContent = 'Gradio API URL (HuggingFace Space)';
        urlInput.value = 'https://smartfalcon-ai-industrial-defect-detection.hf.space';
        urlInput.placeholder = 'https://smartfalcon-ai-industrial-defect-detection.hf.space';
        modelSelector.style.display = 'block';  // Show model selector
        confidenceField.style.display = 'block';  // Show confidence field
        // Fetch models from Gradio
        fetchGradioModels();
        // Show helper text
        const responseDiv = document.getElementById('config-yolo_url-response');
        responseDiv.innerHTML = '<span style="color: #0c5460;">💡 HuggingFace Space URL set - click "Set URL" to apply</span>';
    } else {
        urlLabel.textContent = 'YOLO Inference URL';
        urlInput.value = 'http://yolo_inference:4442/v1/object-detection/yolov5s/detect/';
        urlInput.placeholder = 'http://yolo_inference:4442/v1/object-detection/yolov5s/detect/';
        modelSelector.style.display = 'none';  // Hide model selector
        confidenceField.style.display = 'none';  // Hide confidence field
        const responseDiv = document.getElementById('config-yolo_url-response');
        responseDiv.innerHTML = '';
    }
}

async function updateConfig(key, value) {
    const responseId = `config-${key}-response`;
    const responseEl = document.getElementById(responseId);
    var _b = _btnLoading();

    try {
        const response = await fetch('/api/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ [key]: value })
        });

        const data = await response.json();

        if (responseEl) {
            if (response.ok) {
                responseEl.textContent = `OK: ${key} = ${value}`;
                responseEl.className = 'control-response success';

                // Check if serial was reinitialized
                if (data.serial_reinitialized) {
                    responseEl.textContent = `OK: ${key} = ${value} - Serial connection reinitialized`;
                }

                setTimeout(() => {
                    responseEl.textContent = '';
                    responseEl.className = 'control-response';
                }, 3000);
            } else {
                responseEl.textContent = `Error: ${data.detail || 'Unknown error'}`;
                responseEl.className = 'control-response error';
                setTimeout(() => {
                    responseEl.textContent = '';
                    responseEl.className = 'control-response';
                }, 3000);
            }
        }
    } catch (error) {
        if (responseEl) {
            responseEl.textContent = `Error: ${error.message}`;
            responseEl.className = 'control-response error';
            setTimeout(() => {
                responseEl.textContent = '';
                responseEl.className = 'control-response';
            }, 3000);
        }
    } finally { _btnDone(_b); }
}

async function sendCommand(command, value = null) {
    const responseId = `cmd-${command}-response`;
    const responseEl = document.getElementById(responseId);
    var _b = _btnLoading();
    try {
        let url = `/${command}`;
        if (value !== null) {
            url += `?value=${value}`;
        }

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        const data = await response.json();

        if (responseEl) {
            if (response.ok) {
                responseEl.textContent = `OK: ${data.command}`;
                responseEl.className = 'control-response success';
            } else {
                responseEl.textContent = `Error: ${data.detail || 'Unknown error'}`;
                responseEl.className = 'control-response error';
            }

            setTimeout(() => {
                responseEl.textContent = '';
                responseEl.className = 'control-response';
            }, 3000);
        }
    } catch (error) {
        if (responseEl) {
            responseEl.textContent = `Error: ${error.message}`;
            responseEl.className = 'control-response error';
            setTimeout(() => {
                responseEl.textContent = '';
                responseEl.className = 'control-response';
            }, 3000);
        }
    } finally { _btnDone(_b); }
}

// ===== CAMERA MONITORING FUNCTIONS =====
// Direct camera control through counter service API
let cameraData = {};
let cameraRefreshInterval = null;
let isCameraUpdating = false;

async function fetchCameraStatus() {
    if (isCameraUpdating) return;

    const statusEl = document.getElementById('camera-refresh-status');
    try {
        statusEl.textContent = 'Updating...';
        statusEl.className = 'camera-refresh-status';

        const response = await fetch('/api/cameras');
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || 'Failed to get camera status');
        }

        const cameras = await response.json();

        if (cameras.error) {
            throw new Error(cameras.error);
        }

        cameraData = {};
        cameras.forEach(c => cameraData[c.id] = c);

        renderCameraCards();

        statusEl.textContent = 'Active';
        statusEl.className = 'camera-refresh-status';
    } catch (error) {
        console.error('Error fetching camera status:', error);
        statusEl.textContent = 'Error';
        statusEl.className = 'camera-refresh-status error';

        const grid = document.getElementById('camera-grid');
        grid.innerHTML = `
            <div class="camera-loading" style="color: #dc3545;">
                ⚠️ Could not load camera status<br>
                <small style="color: #666;">${error.message}</small>
            </div>
        `;
    }
}

function renderCameraCards() {
    const grid = document.getElementById('camera-grid');
    const cameraIds = Object.keys(cameraData).sort((a, b) => parseInt(a) - parseInt(b));

    if (cameraIds.length === 0) {
        grid.innerHTML = '<div class="camera-loading">No cameras detected</div>';
        return;
    }

    grid.innerHTML = '';
    cameraIds.forEach(cameraId => {
        const camera = cameraData[cameraId];
        const card = createCameraCard(cameraId, camera);
        grid.appendChild(card);
    });
}

function tip(text) {
    return `<span class="info-tooltip">i<span class="info-tooltip-text">${text}</span></span>`;
}

function createCameraCard(cameraId, camera) {
    const div = document.createElement('div');
    div.className = 'camera-card';
    div.id = `camera-card-${cameraId}`;

    const config = camera.config || {};
    const isConnected = camera.connected;
    const isIPCamera = camera.type === 'ip';
    const cameraName = camera.name || `Camera ${cameraId}`;

    div.innerHTML = `
        <div class="camera-card-header">
            <span class="camera-card-title">Camera ${cameraId}: ${cameraName}</span>
            <span class="camera-status-badge ${isConnected ? 'running' : 'stopped'}">
                ${isConnected ? 'CONNECTED' : 'DISCONNECTED'}
            </span>
        </div>

        <div class="camera-message" id="camera-msg-${cameraId}"></div>

        <div class="camera-card-content">
            <div class="camera-image-container">
                ${isConnected ?
                    `<img id="camera-img-${cameraId}"
                          src="/api/camera/${cameraId}/snapshot?t=${Date.now()}"
                          class="camera-image camera-live-feed"
                          data-camera-id="${cameraId}"
                          alt="${cameraName}"
                          onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                     <div class="camera-no-image" style="display:none;">📷 Camera feed unavailable</div>` :
                    `<div class="camera-no-image">Camera not connected</div>`
                }
            </div>
            <div class="camera-info">
                ${isIPCamera ? `
                <div class="camera-info-item">
                    <span class="camera-info-label">Type</span>
                    <span class="camera-info-value">IP Camera (${camera.ip || 'N/A'})</span>
                </div>
                ` : ''}
                <div class="camera-info-item">
                    <span class="camera-info-label">Path</span>
                    <span class="camera-info-value" style="font-size: 11px;">${isIPCamera && camera.camera_path ? camera.camera_path : camera.path || '-'}</span>
                </div>
                <div class="camera-info-item">
                    <span class="camera-info-label">Resolution</span>
                    <span class="camera-info-value">${config.width || '-'}x${config.height || '-'}</span>
                </div>
                <div class="camera-info-item">
                    <span class="camera-info-label">FPS</span>
                    <span class="camera-info-value">${config.fps || '-'}</span>
                </div>
                ${!isIPCamera ? `
                <div class="camera-info-item">
                    <span class="camera-info-label">Exposure</span>
                    <span class="camera-info-value">${config.exposure || '-'}</span>
                </div>
                <div class="camera-info-item">
                    <span class="camera-info-label">Gain</span>
                    <span class="camera-info-value">${config.gain || '-'}</span>
                </div>
                <div class="camera-info-item">
                    <span class="camera-info-label">Brightness</span>
                    <span class="camera-info-value">${config.brightness || '-'}</span>
                </div>
                ` : ''}
            </div>
        </div>

        ${isIPCamera ? `
        <div class="camera-config-section">
            <div class="camera-roi-section" style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #333;">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <label style="margin-right: 10px; font-weight: bold;">IP Camera Settings ${tip('Exposure, gain, and brightness must be configured through the camera\'s web interface.')}</label>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <label style="margin-right: 10px; font-weight: bold;">ROI ${tip('Region of Interest — crop the frame to only process a specific area.')}</label>
                    <input type="checkbox" id="cam-cfg-roi-enabled-${cameraId}" ${config.roi_enabled ? 'checked' : ''} ${!isConnected ? 'disabled' : ''}>
                    <span style="margin-left: 5px; font-size: 11px;">Enable</span>
                </div>
                <div class="camera-config-grid">
                    <div class="camera-config-item">
                        <label>X Min</label>
                        <input type="number" id="cam-cfg-roi-xmin-${cameraId}" value="${config.roi_xmin || 0}" min="0" max="1280"
                               onfocus="pauseCameraRefresh()" onblur="resumeCameraRefresh()" ${!isConnected ? 'disabled' : ''}>
                    </div>
                    <div class="camera-config-item">
                        <label>Y Min</label>
                        <input type="number" id="cam-cfg-roi-ymin-${cameraId}" value="${config.roi_ymin || 0}" min="0" max="720"
                               onfocus="pauseCameraRefresh()" onblur="resumeCameraRefresh()" ${!isConnected ? 'disabled' : ''}>
                    </div>
                    <div class="camera-config-item">
                        <label>X Max</label>
                        <input type="number" id="cam-cfg-roi-xmax-${cameraId}" value="${config.roi_xmax || 1280}" min="0" max="1280"
                               onfocus="pauseCameraRefresh()" onblur="resumeCameraRefresh()" ${!isConnected ? 'disabled' : ''}>
                    </div>
                    <div class="camera-config-item">
                        <label>Y Max</label>
                        <input type="number" id="cam-cfg-roi-ymax-${cameraId}" value="${config.roi_ymax || 720}" min="0" max="720"
                               onfocus="pauseCameraRefresh()" onblur="resumeCameraRefresh()" ${!isConnected ? 'disabled' : ''}>
                    </div>
                </div>
            </div>
            <div class="camera-actions" style="margin-top: 15px;">
                <button class="camera-btn camera-btn-secondary" onclick="restartCamera(${cameraId})" ${!isConnected ? 'disabled' : ''}>Restart</button>
                <button class="camera-btn camera-btn-primary" onclick="applyCameraConfig(${cameraId})" ${!isConnected ? 'disabled' : ''}>Apply ROI</button>
                <button class="camera-btn camera-btn-success" onclick="saveCameraConfig(${cameraId})">💾 Save</button>
            </div>
        </div>
        ` : `
        <div class="camera-config-section">
            <div class="camera-config-grid">
                <div class="camera-config-item">
                    <label>Exposure ${tip('Sensor exposure time. Higher = brighter but more motion blur.')}</label>
                    <input type="number" id="cam-cfg-exposure-${cameraId}" value="${config.exposure || 100}" min="1" max="100000"
                           onfocus="pauseCameraRefresh()" onblur="resumeCameraRefresh()" ${!isConnected ? 'disabled' : ''}>
                </div>
                <div class="camera-config-item">
                    <label>Gain ${tip('Sensor amplification. Higher = brighter but more noise.')}</label>
                    <input type="number" id="cam-cfg-gain-${cameraId}" value="${config.gain || 100}" min="0" max="255"
                           onfocus="pauseCameraRefresh()" onblur="resumeCameraRefresh()" ${!isConnected ? 'disabled' : ''}>
                </div>
                <div class="camera-config-item">
                    <label>Brightness ${tip('Overall image brightness offset (0-255).')}</label>
                    <input type="number" id="cam-cfg-brightness-${cameraId}" value="${config.brightness || 100}" min="0" max="255"
                           onfocus="pauseCameraRefresh()" onblur="resumeCameraRefresh()" ${!isConnected ? 'disabled' : ''}>
                </div>
                <div class="camera-config-item">
                    <label>Contrast ${tip('Difference between light and dark areas (0-255).')}</label>
                    <input type="number" id="cam-cfg-contrast-${cameraId}" value="${config.contrast || 0}" min="0" max="255"
                           onfocus="pauseCameraRefresh()" onblur="resumeCameraRefresh()" ${!isConnected ? 'disabled' : ''}>
                </div>
                <div class="camera-config-item">
                    <label>Saturation ${tip('Color intensity. 0 = grayscale, 255 = vivid colors.')}</label>
                    <input type="number" id="cam-cfg-saturation-${cameraId}" value="${config.saturation || 50}" min="0" max="255"
                           onfocus="pauseCameraRefresh()" onblur="resumeCameraRefresh()" ${!isConnected ? 'disabled' : ''}>
                </div>
                <div class="camera-config-item">
                    <label>FPS ${tip('Frames per second. Higher = smoother but more CPU/bandwidth.')}</label>
                    <input type="number" id="cam-cfg-fps-${cameraId}" value="${config.fps || 30}" min="1" max="120"
                           onfocus="pauseCameraRefresh()" onblur="resumeCameraRefresh()" ${!isConnected ? 'disabled' : ''}>
                </div>
            </div>
            <div class="camera-roi-section" style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #333;">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <label style="margin-right: 10px; font-weight: bold;">ROI ${tip('Region of Interest — crop the frame to only process a specific area.')}</label>
                    <input type="checkbox" id="cam-cfg-roi-enabled-${cameraId}" ${config.roi_enabled ? 'checked' : ''} ${!isConnected ? 'disabled' : ''}>
                    <span style="margin-left: 5px; font-size: 11px;">Enable</span>
                </div>
                <div class="camera-config-grid">
                    <div class="camera-config-item">
                        <label>X Min</label>
                        <input type="number" id="cam-cfg-roi-xmin-${cameraId}" value="${config.roi_xmin || 0}" min="0" max="1280"
                               onfocus="pauseCameraRefresh()" onblur="resumeCameraRefresh()" ${!isConnected ? 'disabled' : ''}>
                    </div>
                    <div class="camera-config-item">
                        <label>Y Min</label>
                        <input type="number" id="cam-cfg-roi-ymin-${cameraId}" value="${config.roi_ymin || 0}" min="0" max="720"
                               onfocus="pauseCameraRefresh()" onblur="resumeCameraRefresh()" ${!isConnected ? 'disabled' : ''}>
                    </div>
                    <div class="camera-config-item">
                        <label>X Max</label>
                        <input type="number" id="cam-cfg-roi-xmax-${cameraId}" value="${config.roi_xmax || 1280}" min="0" max="1280"
                               onfocus="pauseCameraRefresh()" onblur="resumeCameraRefresh()" ${!isConnected ? 'disabled' : ''}>
                    </div>
                    <div class="camera-config-item">
                        <label>Y Max</label>
                        <input type="number" id="cam-cfg-roi-ymax-${cameraId}" value="${config.roi_ymax || 720}" min="0" max="720"
                               onfocus="pauseCameraRefresh()" onblur="resumeCameraRefresh()" ${!isConnected ? 'disabled' : ''}>
                    </div>
                </div>
            </div>
            <div class="camera-actions">
                <button class="camera-btn camera-btn-secondary" onclick="restartCamera(${cameraId})" ${!isConnected ? 'disabled' : ''}>Restart</button>
                <button class="camera-btn camera-btn-primary" onclick="applyCameraConfig(${cameraId})" ${!isConnected ? 'disabled' : ''}>Apply</button>
                <button class="camera-btn camera-btn-success" onclick="saveCameraConfig(${cameraId})">💾 Save</button>
            </div>
        </div>
        `}
    `;

    return div;
}

function pauseCameraRefresh() {
    isCameraUpdating = true;
    const statusEl = document.getElementById('camera-refresh-status');
    if (statusEl) {
        statusEl.textContent = 'Paused';
        statusEl.className = 'camera-refresh-status paused';
    }
}

function resumeCameraRefresh() {
    setTimeout(() => {
        isCameraUpdating = false;
        const statusEl = document.getElementById('camera-refresh-status');
        if (statusEl) {
            statusEl.textContent = 'Active';
            statusEl.className = 'camera-refresh-status';
        }
    }, 200);
}

function showCameraMessage(cameraId, message, type) {
    const msgEl = document.getElementById(`camera-msg-${cameraId}`);
    if (!msgEl) return;

    msgEl.textContent = message;
    msgEl.className = `camera-message ${type}`;

    setTimeout(() => {
        msgEl.className = 'camera-message';
        msgEl.textContent = '';
    }, 3000);
}

async function restartCamera(cameraId) {
    try {
        showCameraMessage(cameraId, 'Restarting camera...', 'info');

        const response = await fetch(`/api/camera/${cameraId}/restart`, {
            method: 'POST'
        });
        const result = await response.json();

        if (response.ok && result.success) {
            showCameraMessage(cameraId, 'Camera restarted successfully!', 'success');
            setTimeout(() => fetchCameraStatus(), 2000);
        } else {
            showCameraMessage(cameraId, result.error || 'Failed to restart camera', 'error');
        }
    } catch (error) {
        showCameraMessage(cameraId, `Error: ${error.message}`, 'error');
    }
}

async function applyCameraConfig(cameraId) {
    pauseCameraRefresh();
    showCameraMessage(cameraId, 'Applying configuration...', 'info');

    try {
        // Build config object with only available fields
        const config = {
            roi_enabled: document.getElementById(`cam-cfg-roi-enabled-${cameraId}`).checked,
            roi_xmin: parseInt(document.getElementById(`cam-cfg-roi-xmin-${cameraId}`).value) || 0,
            roi_ymin: parseInt(document.getElementById(`cam-cfg-roi-ymin-${cameraId}`).value) || 0,
            roi_xmax: parseInt(document.getElementById(`cam-cfg-roi-xmax-${cameraId}`).value) || 1280,
            roi_ymax: parseInt(document.getElementById(`cam-cfg-roi-ymax-${cameraId}`).value) || 720
        };

        // Add USB camera settings if they exist (will be null for IP cameras)
        const fpsEl = document.getElementById(`cam-cfg-fps-${cameraId}`);
        const exposureEl = document.getElementById(`cam-cfg-exposure-${cameraId}`);
        const gainEl = document.getElementById(`cam-cfg-gain-${cameraId}`);
        const brightnessEl = document.getElementById(`cam-cfg-brightness-${cameraId}`);
        const contrastEl = document.getElementById(`cam-cfg-contrast-${cameraId}`);
        const saturationEl = document.getElementById(`cam-cfg-saturation-${cameraId}`);

        if (fpsEl) config.fps = parseInt(fpsEl.value) || 30;
        if (exposureEl) config.exposure = parseInt(exposureEl.value) || 100;
        if (gainEl) config.gain = parseInt(gainEl.value) || 100;
        if (brightnessEl) config.brightness = parseInt(brightnessEl.value) || 100;
        if (contrastEl) config.contrast = parseInt(contrastEl.value) || 0;
        if (saturationEl) config.saturation = parseInt(saturationEl.value) || 50;

        const response = await fetch(`/api/camera/${cameraId}/config`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        const result = await response.json();

        if (response.ok && result.success) {
            showCameraMessage(cameraId, 'Configuration applied!', 'success');
            setTimeout(() => fetchCameraStatus(), 1000);
        } else {
            showCameraMessage(cameraId, result.error || 'Failed to apply config', 'error');
        }
    } catch (error) {
        showCameraMessage(cameraId, `Error: ${error.message}`, 'error');
    }

    setTimeout(resumeCameraRefresh, 2000);
}

async function saveCameraConfig(cameraId) {
    var _b = _btnLoading();
    try {
        showCameraMessage(cameraId, 'Saving camera configuration...', 'info');

        // First apply the current settings
        await applyCameraConfig(cameraId);

        // Then save all config to file
        const response = await fetch('/api/cameras/config/save', { method: 'POST' });
        const result = await response.json();

        if (response.ok && result.success) {
            showCameraMessage(cameraId, 'Camera settings saved!', 'success');
        } else {
            showCameraMessage(cameraId, result.error || 'Failed to save config', 'error');
        }
    } catch (error) {
        showCameraMessage(cameraId, `Error: ${error.message}`, 'error');
    } finally { _btnDone(_b); }
}

function refreshCameras() {
    fetchCameraStatus();
}

function updateCameraImages() {
    if (isCameraUpdating) return;

    Object.keys(cameraData).forEach(cameraId => {
        const camera = cameraData[cameraId];
        if (camera.connected) {
            const imgEl = document.getElementById(`camera-img-${cameraId}`);
            if (imgEl) {
                // Update snapshot to create live feed effect
                imgEl.src = `/api/camera/${cameraId}/snapshot?t=${Date.now()}`;
            }
        }
    });
}

// Camera WebSocket live feeds — only active when Cameras tab is visible
let cameraWsConnections = {}; // { cameraId: WebSocket }
function startLiveFeedUpdates() {
    openCameraWebSockets();
}

function openCameraWebSockets() {
    const camerasTab = document.getElementById('tab-cameras');
    if (!camerasTab || !camerasTab.classList.contains('active')) return;

    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    document.querySelectorAll('.camera-live-feed').forEach(img => {
        const cameraId = img.getAttribute('data-camera-id');
        if (!cameraId || cameraWsConnections[cameraId]) return;
        if (cameraData[cameraId] && !cameraData[cameraId].connected) return;

        const ws = new WebSocket(`${proto}//${location.host}/ws/camera/${cameraId}`);
        ws.binaryType = 'arraybuffer';
        ws.onmessage = (e) => {
            if (typeof e.data === 'string') return;
            const blob = new Blob([e.data], {type: 'image/jpeg'});
            const url = URL.createObjectURL(blob);
            if (img._blobUrl) URL.revokeObjectURL(img._blobUrl);
            img._blobUrl = url;
            img.src = url;
        };
        ws.onclose = () => {
            delete cameraWsConnections[cameraId];
            // Auto-reconnect if tab is still active
            const tab = document.getElementById('tab-cameras');
            if (tab && tab.classList.contains('active')) {
                setTimeout(() => openCameraWebSockets(), 2000);
            }
        };
        ws.onerror = () => {};
        cameraWsConnections[cameraId] = ws;
    });
}

function stopLiveFeedUpdates() {
    Object.keys(cameraWsConnections).forEach(id => {
        const ws = cameraWsConnections[id];
        if (ws && ws.readyState <= WebSocket.OPEN) ws.close();
    });
    cameraWsConnections = {};
}

// Start live feed updates when cameras are loaded
window.addEventListener('load', () => {
    setTimeout(startLiveFeedUpdates, 1000);
    // Global tooltip — rendered at <body> level, outside all stacking contexts
    const _gTip = document.getElementById('global-tooltip');
    document.addEventListener('mouseenter', (e) => {
        if (!e.target || typeof e.target.closest !== 'function') return;
        const trigger = e.target.closest('.info-tooltip, .info-tooltip-sm');
        if (!trigger || !_gTip) return;
        const src = trigger.querySelector('.info-tooltip-text');
        if (!src) return;
        _gTip.textContent = src.textContent;
        _gTip.style.display = 'block';
        const rect = trigger.getBoundingClientRect();
        const tw = _gTip.offsetWidth;
        const th = _gTip.offsetHeight;
        let top = rect.top - th - 8;
        if (top < 4) top = rect.bottom + 8;
        let left = rect.left + rect.width / 2 - tw / 2;
        left = Math.max(4, Math.min(left, window.innerWidth - tw - 4));
        _gTip.style.top = top + 'px';
        _gTip.style.left = left + 'px';
    }, true);
    document.addEventListener('mouseleave', (e) => {
        if (!e.target || typeof e.target.closest !== 'function') return;
        const trigger = e.target.closest('.info-tooltip, .info-tooltip-sm');
        if (trigger && _gTip) _gTip.style.display = 'none';
    }, true);
});

// ===== CAMERA CONFIG PERSISTENCE FUNCTIONS =====
function showConfigMessage(message, type) {
    const msgEl = document.getElementById('camera-config-message');
    msgEl.textContent = message;
    msgEl.style.display = 'block';
    msgEl.style.backgroundColor = type === 'success' ? '#d4edda' : type === 'error' ? '#f8d7da' : '#d1ecf1';
    msgEl.style.color = type === 'success' ? '#155724' : type === 'error' ? '#721c24' : '#0c5460';
    setTimeout(() => { msgEl.style.display = 'none'; }, 5000);
}

async function saveServiceConfig() {
    try {
        showConfigMessage('Saving all service configuration...', 'info');
        const response = await fetch('/api/cameras/config/save', { method: 'POST' });
        const result = await response.json();
        if (response.ok && result.success) {
            showConfigMessage(`All config saved! (${result.cameras_saved} cameras, ${result.states_saved || 0} states + service settings)`, 'success');
            checkServiceConfigStatus();
            refreshStates();  // Refresh states display
        } else {
            showConfigMessage(result.error || 'Failed to save', 'error');
        }
    } catch (error) {
        showConfigMessage(`Error: ${error.message}`, 'error');
    }
}

async function loadServiceConfig() {
    try {
        showConfigMessage('Loading all service configuration...', 'info');
        const response = await fetch('/api/cameras/config/load', { method: 'POST' });
        const result = await response.json();
        if (response.ok && result.success) {
            showConfigMessage(`All config loaded! (${result.cameras_loaded} cameras, ${result.states_loaded || 0} states + service settings)`, 'success');
            setTimeout(() => fetchCameraStatus(), 500);
            refreshStates();  // Refresh states display
        } else {
            showConfigMessage(result.error || 'Failed to load', 'error');
        }
    } catch (error) {
        showConfigMessage(`Error: ${error.message}`, 'error');
    }
}

async function downloadServiceConfig() {
    try {
        const response = await fetch('/api/cameras/config');
        const data = await response.json();
        if (!data.exists || !data.config) {
            showConfigMessage('No saved configuration to export', 'error');
            return;
        }
        const blob = new Blob([JSON.stringify(data.config, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `service_config_${new Date().toISOString().slice(0,10)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showConfigMessage('All configuration exported!', 'success');
    } catch (error) {
        showConfigMessage(`Error: ${error.message}`, 'error');
    }
}


async function checkServiceConfigStatus() {
    try {
        const response = await fetch('/api/cameras/config');
        const data = await response.json();
        const statusEl = document.getElementById('camera-config-status');
        if (statusEl) {
            if (data.exists && data.config) {
                statusEl.textContent = `Last saved: ${data.config.saved_at || 'unknown'}`;
            } else {
                statusEl.textContent = 'No saved config';
            }
        }
    } catch (error) {
        const statusEl = document.getElementById('camera-config-status');
        if (statusEl) statusEl.textContent = '';
    }
}

// ===== STATE MANAGEMENT FUNCTIONS =====
let statesData = {};

function showStateMessage(message, type) {
    const msgEl = document.getElementById('state-message');
    msgEl.textContent = message;
    msgEl.style.display = 'block';
    msgEl.style.backgroundColor = type === 'success' ? '#d4edda' : type === 'error' ? '#f8d7da' : '#d1ecf1';
    msgEl.style.color = type === 'success' ? '#155724' : type === 'error' ? '#721c24' : '#0c5460';
    setTimeout(() => { msgEl.style.display = 'none'; }, 5000);
}

async function refreshStates() {
    try {
        const response = await fetch('/api/states');
        const data = await response.json();
        statesData = data;

        // Update current state display
        const currentState = data.current_state;
        document.getElementById('current-state-name').textContent = currentState ? currentState.name : 'None';

        // Update capture state badge
        const captureState = data.capture_state || 'idle';
        const badge = document.getElementById('capture-state-badge');
        badge.textContent = captureState.toUpperCase();
        const badgeColors = {
            'idle': '#6c757d',
            'ready': '#17a2b8',
            'capturing': '#28a745',
            'processing': '#ffc107',
            'error': '#dc3545'
        };
        badge.style.backgroundColor = badgeColors[captureState] || '#6c757d';

        // Render states list
        renderStatesList(data.states || {});

    } catch (error) {
        console.error('Error fetching states:', error);
        showStateMessage('Error loading states: ' + error.message, 'error');
    }
}

function renderStatesList(states) {
    const container = document.getElementById('states-list');
    const stateNames = Object.keys(states);

    if (stateNames.length === 0) {
        container.innerHTML = '<span style="color: #666; font-style: italic;">No states configured</span>';
        return;
    }

    container.innerHTML = '';
    stateNames.forEach(name => {
        const state = states[name];
        const isActive = statesData.current_state && statesData.current_state.name === name;

        // Get phases info for display
        const phases = state.phases || [];
        const phaseCount = phases.length;
        const allCameras = [...new Set(phases.flatMap(p => p.cameras || []))].sort();

        // Get trigger thresholds from first phase (or state-level fallback)
        const firstPhase = phases.length > 0 ? phases[0] : {};
        let steps = firstPhase.steps !== undefined ? firstPhase.steps : (state.steps !== undefined ? state.steps : 1);
        let analog = firstPhase.analog !== undefined ? firstPhase.analog : (state.analog !== undefined ? state.analog : -1);
        const stepsDisplay = steps < 0 ? 'loop' : steps;
        const analogDisplay = analog < 0 ? 'off' : analog;

        // Get light mode and delay info from phases
        const phasesInfo = phases.map((p, idx) => {
            const lightMode = p.light_mode || 'U_ON_B_OFF';
            const delay = p.delay !== undefined ? p.delay : 0;
            return `${lightMode}:${delay}s`;
        }).join(', ');

        const btn = document.createElement('button');
        btn.className = 'camera-btn ' + (isActive ? 'camera-btn-success' : 'camera-btn-secondary');
        btn.style.cssText = 'display: flex; flex-direction: column; align-items: center; padding: 8px 12px; min-width: 140px; cursor: pointer;';
        btn.innerHTML = `
            <span style="font-weight: bold;">${name}</span>
            <span style="font-size: 10px; opacity: 0.8;">${phaseCount} phase${phaseCount !== 1 ? 's' : ''} | Cams: ${allCameras.join(',')}</span>
            <span style="font-size: 10px; opacity: 0.8;">Steps: ${stepsDisplay} | Analog: ${analogDisplay}</span>
            <span style="font-size: 9px; opacity: 0.7;">${phasesInfo || 'No phases'}</span>
        `;
        btn.onclick = () => activateState(name);
        btn.ondblclick = () => loadStateToEditor(name, state);  // Double-click to edit
        btn.title = 'Click to activate, double-click to edit\nLight modes: ' + phasesInfo;
        container.appendChild(btn);
    });
}

function loadStateToEditor(name, state) {
    // Load state into editor form
    document.getElementById('state-name-input').value = name;
    document.getElementById('state-enabled-input').value = state.enabled ? 'true' : 'false';

    // Clear existing phases and rebuild
    const container = document.getElementById('phases-container');
    container.innerHTML = '';

    const phases = state.phases || [];
    phases.forEach((phase, idx) => {
        const steps = phase.steps !== undefined ? phase.steps : (state.steps !== undefined ? state.steps : -1);
        const analog = phase.analog !== undefined ? phase.analog : (state.analog !== undefined ? state.analog : -1);
        addPhaseRow(idx, phase.light_mode, phase.delay, phase.cameras, steps, analog);
    });

    showStateMessage(`Loaded state '${name}' into editor`, 'info');
}

async function activateState(stateName) {
    try {
        showStateMessage('Activating state...', 'info');
        const response = await fetch(`/api/states/${stateName}/activate`, { method: 'POST' });
        const result = await response.json();

        if (response.ok && result.success) {
            showStateMessage(`State '${stateName}' activated!`, 'success');
            refreshStates();
        } else {
            showStateMessage(result.error || 'Failed to activate state', 'error');
        }
    } catch (error) {
        showStateMessage('Error: ' + error.message, 'error');
    }
}

async function triggerCapture() {
    try {
        showStateMessage('Triggering capture...', 'info');
        const response = await fetch('/api/states/trigger-capture', { method: 'POST' });
        const result = await response.json();

        if (response.ok && result.success) {
            showStateMessage(`Capture triggered! Count: ${result.capture_count}`, 'success');
            refreshStates();
        } else {
            showStateMessage(result.error || 'Failed to trigger capture', 'error');
        }
    } catch (error) {
        showStateMessage('Error: ' + error.message, 'error');
    }
}

// Phase management functions
let phaseCounter = 1;  // Start at 1 since we have 1 default phase

function addPhase() {
    addPhaseRow(phaseCounter, 'U_ON_B_OFF', 0.1, [1, 2, 3, 4]);
    phaseCounter++;
}

function addPhaseRow(idx, lightMode, delay, cameras, steps = 1, analog = -1) {
    const container = document.getElementById('phases-container');
    const div = document.createElement('div');
    div.className = 'phase-row';
    div.id = `phase-row-${idx}`;
    div.style.cssText = 'display: grid; grid-template-columns: 1.2fr 0.8fr 1fr 0.8fr 0.8fr auto; gap: 6px; align-items: end; margin-bottom: 8px; padding: 8px; background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(51, 65, 85, 0.4); border-radius: 4px;';

    const camerasStr = Array.isArray(cameras) ? cameras.join(',') : cameras;

    div.innerHTML = `
        <div>
            <label style="font-size: 10px; color: #666;">Light Mode ${tip('Which lights to turn on/off during this capture phase.')}</label>
            <select class="control-input phase-light" style="margin: 0; padding: 4px; font-size: 12px;">
                <option value="U_ON_B_OFF" ${lightMode === 'U_ON_B_OFF' ? 'selected' : ''}>U On, B Off</option>
                <option value="U_OFF_B_ON" ${lightMode === 'U_OFF_B_ON' ? 'selected' : ''}>U Off, B On</option>
                <option value="U_ON_B_ON" ${lightMode === 'U_ON_B_ON' ? 'selected' : ''}>U On, B On</option>
                <option value="U_OFF_B_OFF" ${lightMode === 'U_OFF_B_OFF' ? 'selected' : ''}>U Off, B Off</option>
            </select>
        </div>
        <div>
            <label style="font-size: 10px; color: #666;">Delay (s) ${tip('Wait time in seconds after setting lights, before capturing.')}</label>
            <input type="number" class="control-input phase-delay" value="${delay}" min="0" step="0.01" style="margin: 0; padding: 4px; font-size: 12px;">
        </div>
        <div>
            <label style="font-size: 10px; color: #666;">Cameras ${tip('Comma-separated camera IDs to capture in this phase.')}</label>
            <input type="text" class="control-input phase-cameras" value="${camerasStr}" placeholder="1,2,3,4" style="margin: 0; padding: 4px; font-size: 12px;">
        </div>
        <div>
            <label style="font-size: 10px; color: #666;">Steps (-1=loop) ${tip('Capture every N encoder steps. -1 = continuous capture loop.')}</label>
            <input type="number" class="control-input phase-steps" value="${steps}" min="-1" step="1" style="margin: 0; padding: 4px; font-size: 12px;">
        </div>
        <div>
            <label style="font-size: 10px; color: #666;">Analog (-1=off) ${tip('Analog sensor threshold. -1 = disabled. N = trigger when value >= N.')}</label>
            <input type="number" class="control-input phase-analog" value="${analog}" min="-1" step="1" style="margin: 0; padding: 4px; font-size: 12px;">
        </div>
        <button class="camera-btn camera-btn-danger" onclick="removePhase('${idx}')" style="padding: 4px 8px; font-size: 11px;">✕</button>
    `;
    container.appendChild(div);
}

function removePhase(idx) {
    const row = document.getElementById(`phase-row-${idx}`);
    if (row) {
        row.remove();
    }
    // Check if we have at least one phase
    const remaining = document.querySelectorAll('.phase-row').length;
    if (remaining === 0) {
        addPhaseRow(phaseCounter++, 'U_ON_B_OFF', 0.1, [1, 2, 3, 4]);
    }
}

function collectPhases() {
    const phases = [];
    const rows = document.querySelectorAll('.phase-row');
    rows.forEach(row => {
        const lightMode = row.querySelector('.phase-light').value;
        const delay = parseFloat(row.querySelector('.phase-delay').value) || 0;
        const camerasStr = row.querySelector('.phase-cameras').value.trim();
        const cameras = camerasStr ? camerasStr.split(',').map(c => parseInt(c.trim())).filter(c => !isNaN(c)) : [];
        const steps = parseInt(row.querySelector('.phase-steps').value) || -1;
        const analog = parseInt(row.querySelector('.phase-analog').value) || -1;

        // Allow phases without cameras (e.g., light-off phase)
        phases.push({
            light_mode: lightMode,
            delay: delay,
            cameras: cameras,
            steps: steps,
            analog: analog
        });
    });
    return phases;
}

async function createOrUpdateState() {
    var _b = _btnLoading();
    const responseEl = document.getElementById('state-config-response');
    const name = document.getElementById('state-name-input').value.trim();

    if (!name) {
        responseEl.textContent = 'State name is required';
        responseEl.className = 'control-response error';
        setTimeout(() => { responseEl.textContent = ''; responseEl.className = 'control-response'; }, 3000);
        return;
    }

    const phases = collectPhases();

    if (phases.length === 0) {
        responseEl.textContent = 'At least one phase is required';
        responseEl.className = 'control-response error';
        setTimeout(() => { responseEl.textContent = ''; responseEl.className = 'control-response'; }, 3000);
        return;
    }

    // Warn if no phase has cameras (but still allow it)
    const hasCapture = phases.some(p => p.cameras.length > 0);
    if (!hasCapture) {
        console.warn('Warning: No phase has cameras configured - no images will be captured');
    }

    const stateData = {
        name: name,
        phases: phases,
        enabled: document.getElementById('state-enabled-input').value === 'true'
    };

    try {
        const response = await fetch('/api/states', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(stateData)
        });
        const result = await response.json();

        if (response.ok && result.success) {
            responseEl.textContent = `State '${name}' created/updated with ${phases.length} phase(s)!`;
            responseEl.className = 'control-response success';
            refreshStates();
        } else {
            responseEl.textContent = result.error || 'Failed to create state';
            responseEl.className = 'control-response error';
        }
    } catch (error) {
        responseEl.textContent = 'Error: ' + error.message;
        responseEl.className = 'control-response error';
    } finally { _btnDone(_b); }
    setTimeout(() => { responseEl.textContent = ''; responseEl.className = 'control-response'; }, 3000);
}

async function deleteCurrentState() {
    const responseEl = document.getElementById('state-config-response');
    const name = document.getElementById('state-name-input').value.trim();

    if (!name) {
        responseEl.textContent = 'Enter state name to delete';
        responseEl.className = 'control-response error';
        setTimeout(() => { responseEl.textContent = ''; responseEl.className = 'control-response'; }, 3000);
        return;
    }

    if (name === 'default') {
        responseEl.textContent = 'Cannot delete default state';
        responseEl.className = 'control-response error';
        setTimeout(() => { responseEl.textContent = ''; responseEl.className = 'control-response'; }, 3000);
        return;
    }

    if (!confirm(`Delete state '${name}'?`)) return;

    try {
        const response = await fetch(`/api/states/${name}`, { method: 'DELETE' });
        const result = await response.json();

        if (response.ok && result.success) {
            responseEl.textContent = `State '${name}' deleted!`;
            responseEl.className = 'control-response success';
            document.getElementById('state-name-input').value = '';
            refreshStates();
        } else {
            responseEl.textContent = result.error || 'Failed to delete state';
            responseEl.className = 'control-response error';
        }
    } catch (error) {
        responseEl.textContent = 'Error: ' + error.message;
        responseEl.className = 'control-response error';
    }
    setTimeout(() => { responseEl.textContent = ''; responseEl.className = 'control-response'; }, 3000);
}

// ===== INFERENCE PIPELINE MANAGEMENT =====
let pipelineData = {};

async function refreshPipelineConfig() {
    try {
        const response = await fetch('/api/pipelines');
        const data = await response.json();
        pipelineData = data;

        // Update current pipeline/model display
        const currentPipeline = pipelineData.current_pipeline || 'default';
        const currentModel = pipelineData.current_model;

        document.getElementById('current-pipeline-name').textContent = currentPipeline;
        document.getElementById('current-model-name').textContent = currentModel ? currentModel.name : 'None';

        // Render lists
        renderPipelinesList(pipelineData.pipelines || {}, currentPipeline);
        renderModelsList(pipelineData.models || {});
        renderModelsChecklist(pipelineData.models || {});

    } catch (error) {
        console.error('Error loading pipeline config:', error);
    }
}

async function fetchModelsFromGradio() {
    const url = document.getElementById('model-url-input').value.trim();
    const select = document.getElementById('model-gradio-name-input');
    const btn = event.target;

    if (!url) {
        alert('Please enter a Gradio URL first');
        return;
    }

    // Show loading state
    const originalText = btn.textContent;
    btn.textContent = 'Loading...';
    btn.disabled = true;

    try {
        const response = await fetch(`/api/gradio/models?url=${encodeURIComponent(url)}`);
        const data = await response.json();

        if (data.models && data.models.length > 0) {
            // Update select with fetched models
            select.innerHTML = '';
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                select.appendChild(option);
            });

            if (data.note) {
                console.log('Note:', data.note);
            }
            if (data.source) {
                console.log('Models fetched from:', data.source);
            }
        } else {
            alert('No models found from API. Using default list.');
        }
    } catch (error) {
        console.error('Error fetching models:', error);
        alert('Failed to fetch models. Using default list.');
    } finally {
        btn.textContent = originalText;
        btn.disabled = false;
    }
}

function handleModelTypeChange() {
    const type = document.getElementById('model-type-input').value;
    const select = document.getElementById('model-gradio-name-input');

    if (type === 'yolo') {
        select.innerHTML = '<option value="N/A">N/A</option>';
    } else if (type === 'gradio') {
        select.innerHTML = `
            <option value="Data Matrix">Data Matrix</option>
            <option value="Dental Implant">Dental Implant</option>
            <option value="Ball Pen">Ball Pen</option>
            <option value="Knit Up">Knit Up</option>
            <option value="Knit Back">Knit Back</option>
            <option value="Jean Back">Jean Back</option>
            <option value="Jean Up">Jean Up</option>
            <option value="Tire Cord">Tire Cord</option>
            <option value="predict">predict</option>
            <option value="N/A">N/A</option>
        `;
    }
}

function renderPipelinesList(pipelines, currentPipeline) {
    const container = document.getElementById('pipelines-list');
    const pipelineNames = Object.keys(pipelines);

    if (pipelineNames.length === 0) {
        container.innerHTML = '<span style="color: #666; font-style: italic;">No pipelines configured</span>';
        return;
    }

    container.innerHTML = '';
    pipelineNames.forEach(name => {
        const pipeline = pipelines[name];
        const isActive = name === currentPipeline;
        const pipelineDiv = document.createElement('div');
        pipelineDiv.style.cssText = `padding: 10px; background: ${isActive ? 'rgba(59, 130, 246, 0.15)' : 'rgba(30, 41, 59, 0.6)'}; border: 2px solid ${isActive ? 'var(--primary-color)' : 'var(--border-color)'}; border-radius: 6px; min-width: 200px;`;

        const phaseModels = pipeline.phases.map(p => p.model_id).join(' -> ') || 'No models';
        pipelineDiv.innerHTML = `
            <div>
                <strong style="font-size: 14px; color: var(--text-primary);">${pipeline.name}</strong>
                ${isActive ? '<span style="margin-left: 8px; padding: 2px 6px; background: var(--primary-color); color: white; border-radius: 3px; font-size: 10px;">ACTIVE</span>' : ''}
                <div style="margin-top: 4px; font-size: 11px; color: var(--text-secondary);">${pipeline.description || 'No description'}</div>
                <div style="margin-top: 4px; font-size: 10px; color: var(--text-secondary);">Phases: ${phaseModels}</div>
                <div style="margin-top: 8px; display: flex; gap: 4px;">
                    ${!isActive ? `<button onclick="activatePipeline('${name}')" style="padding: 4px 8px; background: #007bff; color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 11px;">Activate</button>` : ''}
                    <button onclick="loadPipelineForEdit('${name}')" style="padding: 4px 8px; background: #6c757d; color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 11px;">Edit</button>
                </div>
            </div>
        `;
        container.appendChild(pipelineDiv);
    });
}

function renderModelsList(models) {
    const container = document.getElementById('models-list');
    const modelIds = Object.keys(models);

    if (modelIds.length === 0) {
        container.innerHTML = '<span style="color: #666; font-style: italic;">No models configured</span>';
        return;
    }

    container.innerHTML = '';
    modelIds.forEach(id => {
        const model = models[id];
        const modelDiv = document.createElement('div');
        modelDiv.style.cssText = 'padding: 10px; background: rgba(30, 41, 59, 0.6); border: 1px solid var(--border-color); border-radius: 4px;';
        modelDiv.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div style="flex: 1;">
                    <strong style="font-size: 14px; color: var(--text-primary);">${model.name}</strong>
                    <span style="margin-left: 8px; padding: 2px 6px; background: ${model.model_type === 'gradio' ? '#17a2b8' : '#28a745'}; color: white; border-radius: 3px; font-size: 10px;">${model.model_type.toUpperCase()}</span>
                    <div style="margin-top: 4px; font-size: 11px; color: var(--text-secondary);">
                        Model: ${model.model_name} | Confidence: ${model.confidence_threshold}
                    </div>
                    <div style="margin-top: 2px; font-size: 10px; color: var(--text-secondary); word-break: break-all;">${model.inference_url}</div>
                </div>
                <div style="display: flex; gap: 4px; margin-left: 10px;">
                    <button onclick="loadModelForEdit('${id}')" style="padding: 4px 8px; background: #6c757d; color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 11px;">Edit</button>
                </div>
            </div>
        `;
        container.appendChild(modelDiv);
    });
}

function renderModelsChecklist(models) {
    const container = document.getElementById('pipeline-models-checklist');
    const modelIds = Object.keys(models);

    if (modelIds.length === 0) {
        container.innerHTML = '<span style="color: #666; font-style: italic;">No models available</span>';
        return;
    }

    container.innerHTML = '';
    modelIds.forEach(id => {
        const model = models[id];
        const label = document.createElement('label');
        label.style.cssText = 'display: flex; align-items: center; gap: 6px; padding: 4px 8px; background: rgba(30, 41, 59, 0.6); border: 1px solid var(--border-color); border-radius: 4px; cursor: pointer; color: var(--text-primary);';
        label.innerHTML = `
            <input type="checkbox" name="pipeline-model" value="${id}">
            <span>${model.name}</span>
        `;
        container.appendChild(label);
    });
}

function loadPipelineForEdit(pipelineName) {
    const pipeline = pipelineData.pipelines[pipelineName];
    if (!pipeline) return;

    document.getElementById('pipeline-name-input').value = pipeline.name;
    document.getElementById('pipeline-desc-input').value = pipeline.description || '';

    // Check the models in this pipeline
    const checkboxes = document.querySelectorAll('input[name="pipeline-model"]');
    const pipelineModelIds = pipeline.phases.map(p => p.model_id);
    checkboxes.forEach(cb => {
        cb.checked = pipelineModelIds.includes(cb.value);
    });
}

function loadModelForEdit(modelId) {
    const model = pipelineData.models[modelId];
    if (!model) return;

    document.getElementById('model-name-input').value = model.name;
    document.getElementById('model-type-input').value = model.model_type || 'gradio';
    document.getElementById('model-url-input').value = model.inference_url;
    document.getElementById('model-confidence-input').value = model.confidence_threshold;

    const modelSelect = document.getElementById('model-gradio-name-input');
    const modelValue = model.model_name;
    let found = false;
    for (let option of modelSelect.options) {
        if (option.value === modelValue) {
            modelSelect.value = modelValue;
            found = true;
            break;
        }
    }
    if (!found && modelValue) {
        const option = document.createElement('option');
        option.value = modelValue;
        option.textContent = modelValue;
        modelSelect.appendChild(option);
        modelSelect.value = modelValue;
    }
}

async function activatePipeline(pipelineName) {
    const responseEl = document.getElementById('pipeline-response');
    try {
        const response = await fetch(`/api/pipelines/activate/${pipelineName}`, { method: 'POST' });
        const result = await response.json();

        if (response.ok) {
            responseEl.textContent = `Pipeline '${pipelineName}' activated!`;
            responseEl.className = 'control-response success';
            refreshPipelineConfig();
        } else {
            responseEl.textContent = result.error || 'Failed to activate pipeline';
            responseEl.className = 'control-response error';
        }
    } catch (error) {
        responseEl.textContent = 'Error: ' + error.message;
        responseEl.className = 'control-response error';
    }
    setTimeout(() => { responseEl.textContent = ''; responseEl.className = 'control-response'; }, 3000);
}

async function createOrUpdatePipeline() {
    const pipelineName = document.getElementById('pipeline-name-input').value.trim();
    const description = document.getElementById('pipeline-desc-input').value.trim();
    const responseEl = document.getElementById('pipeline-response');

    if (!pipelineName) {
        responseEl.textContent = 'Please enter a pipeline name';
        responseEl.className = 'control-response error';
        return;
    }

    // Get selected models
    const checkboxes = document.querySelectorAll('input[name="pipeline-model"]:checked');
    const phases = Array.from(checkboxes).map((cb, idx) => ({
        model_id: cb.value,
        enabled: true,
        order: idx
    }));

    if (phases.length === 0) {
        responseEl.textContent = 'Please select at least one model for the pipeline';
        responseEl.className = 'control-response error';
        return;
    }

    try {
        const response = await fetch('/api/pipelines', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: pipelineName,
                description: description,
                phases: phases,
                enabled: true
            })
        });

        const result = await response.json();
        if (response.ok) {
            responseEl.textContent = `Pipeline '${pipelineName}' saved!`;
            responseEl.className = 'control-response success';
            refreshPipelineConfig();
        } else {
            responseEl.textContent = result.error || 'Failed to save pipeline';
            responseEl.className = 'control-response error';
        }
    } catch (error) {
        responseEl.textContent = 'Error: ' + error.message;
        responseEl.className = 'control-response error';
    }
    setTimeout(() => { responseEl.textContent = ''; responseEl.className = 'control-response'; }, 3000);
}

async function deletePipeline() {
    const pipelineName = document.getElementById('pipeline-name-input').value.trim();
    const responseEl = document.getElementById('pipeline-response');

    if (!pipelineName || !confirm(`Delete pipeline '${pipelineName}'?`)) return;

    try {
        const response = await fetch(`/api/pipelines/${pipelineName}`, { method: 'DELETE' });
        const result = await response.json();

        if (response.ok) {
            responseEl.textContent = `Pipeline '${pipelineName}' deleted!`;
            responseEl.className = 'control-response success';
            document.getElementById('pipeline-name-input').value = '';
            refreshPipelineConfig();
        } else {
            responseEl.textContent = result.error || 'Failed to delete pipeline';
            responseEl.className = 'control-response error';
        }
    } catch (error) {
        responseEl.textContent = 'Error: ' + error.message;
        responseEl.className = 'control-response error';
    }
    setTimeout(() => { responseEl.textContent = ''; responseEl.className = 'control-response'; }, 3000);
}

async function createOrUpdateModel() {
    const modelName = document.getElementById('model-name-input').value.trim();
    const modelType = document.getElementById('model-type-input').value;
    const modelGradioName = document.getElementById('model-gradio-name-input').value;
    const modelConfidence = parseFloat(document.getElementById('model-confidence-input').value);
    const modelUrl = document.getElementById('model-url-input').value.trim();
    const responseEl = document.getElementById('model-response');

    if (!modelName || !modelUrl) {
        responseEl.textContent = 'Please fill in Model Name and URL';
        responseEl.className = 'control-response error';
        return;
    }

    // Generate model_id from name (lowercase, underscores)
    const modelId = modelName.toLowerCase().replace(/\s+/g, '_').replace(/[^a-z0-9_]/g, '');

    try {
        const response = await fetch('/api/models', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_id: modelId,
                name: modelName,
                model_type: modelType,
                model_name: modelGradioName,
                confidence_threshold: modelConfidence,
                inference_url: modelUrl
            })
        });

        const result = await response.json();
        if (response.ok) {
            responseEl.textContent = `Model '${modelName}' saved!`;
            responseEl.className = 'control-response success';
            refreshPipelineConfig();
        } else {
            responseEl.textContent = result.error || 'Failed to save model';
            responseEl.className = 'control-response error';
        }
    } catch (error) {
        responseEl.textContent = 'Error: ' + error.message;
        responseEl.className = 'control-response error';
    }
    setTimeout(() => { responseEl.textContent = ''; responseEl.className = 'control-response'; }, 3000);
}

async function deleteModel() {
    const modelName = document.getElementById('model-name-input').value.trim();
    const responseEl = document.getElementById('model-response');

    if (!modelName) {
        responseEl.textContent = 'Please enter a model name to delete';
        responseEl.className = 'control-response error';
        return;
    }

    const modelId = modelName.toLowerCase().replace(/\s+/g, '_').replace(/[^a-z0-9_]/g, '');

    if (!confirm(`Delete model '${modelName}'?`)) return;

    try {
        const response = await fetch(`/api/models/${modelId}`, { method: 'DELETE' });
        const result = await response.json();

        if (response.ok) {
            responseEl.textContent = `Model '${modelName}' deleted!`;
            responseEl.className = 'control-response success';
            document.getElementById('model-name-input').value = '';
            refreshPipelineConfig();
        } else {
            responseEl.textContent = result.error || 'Failed to delete model';
            responseEl.className = 'control-response error';
        }
    } catch (error) {
        responseEl.textContent = 'Error: ' + error.message;
        responseEl.className = 'control-response error';
    }
    setTimeout(() => { responseEl.textContent = ''; responseEl.className = 'control-response'; }, 3000);
}

// Legacy compatibility - keep old function name
function refreshInferenceConfig() {
    refreshPipelineConfig();
}

// ===== Camera Discovery Functions =====
async function scanForCameras() {
    const subnet = document.getElementById('subnet-input').value.trim();
    const scanButton = document.getElementById('scan-button');
    const statusContainer = document.getElementById('discovery-status-container');
    const camerasContainer = document.getElementById('discovered-cameras-container');

    // Validate subnet
    if (!subnet || !/^\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(subnet)) {
        statusContainer.innerHTML = '<div class="discovery-status error">Invalid subnet format. Use format like: 192.168.0</div>';
        return;
    }

    // Disable button and show scanning status
    scanButton.disabled = true;
    scanButton.textContent = '🔄 Scanning...';
    statusContainer.innerHTML = '<div class="discovery-status scanning">⏳ Scanning network ' + subnet + '.0/24 for camera devices... (Quick port scan, ~30 seconds)</div>';
    camerasContainer.innerHTML = '';

    try {
        const response = await fetch('/api/cameras/discover', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ subnet })
        });

        const result = await response.json();

        if (response.ok && result.success) {
            statusContainer.innerHTML = `<div class="discovery-status success">✓ Scan complete! Found ${result.count} potential camera device(s) on subnet ${result.subnet}.0/24</div>`;

            if (result.cameras && result.cameras.length > 0) {
                displayDiscoveredCameras(result.cameras);
            } else {
                camerasContainer.innerHTML = '<p style="color: #666; text-align: center; padding: 20px;">No cameras found on this network. Check if cameras are powered on and connected to the network.</p>';
            }
        } else {
            statusContainer.innerHTML = `<div class="discovery-status error">❌ Error: ${result.error || 'Scan failed'}</div>`;
            camerasContainer.innerHTML = '';
        }
    } catch (error) {
        console.error('Discovery error:', error);
        statusContainer.innerHTML = `<div class="discovery-status error">❌ Error: ${error.message}</div>`;
        camerasContainer.innerHTML = '';
    } finally {
        scanButton.disabled = false;
        scanButton.textContent = '🔎 Scan Network';
    }
}

function displayDiscoveredCameras(cameras) {
    const container = document.getElementById('discovered-cameras-container');
    container.innerHTML = '<div class="discovered-cameras-list">' + cameras.map((cam, idx) => {
        const pathsHtml = cam.paths.map((p, pidx) => `
            <div style="margin-top: 10px; padding: 10px; background: #fff; border: 1px solid #e0e0e0; border-radius: 4px;">
                <div style="font-size: 11px; color: #666; font-family: monospace; margin-bottom: 5px;">Path ${pidx + 1}: ${p.path}</div>
                <div style="display: flex; gap: 10px; align-items: center;">
                    <input type="text" id="camera-${idx}-path-${pidx}-username" placeholder="Username" value="admin" style="padding: 6px; border: 1px solid #ddd; border-radius: 4px; font-size: 13px; width: 120px;">
                    <input type="password" id="camera-${idx}-path-${pidx}-password" placeholder="Password" style="padding: 6px; border: 1px solid #ddd; border-radius: 4px; font-size: 13px; width: 120px;">
                    <button class="test-camera-button" onclick='testCameraPath(${idx}, ${pidx}, "${p.url}", "${cam.ip}", "${p.path}")'>
                        🎥 Test & Preview
                    </button>
                </div>
            </div>
        `).join('');

        return `
            <div class="discovered-camera-item" id="discovered-camera-${idx}" style="display: block; grid-template-columns: none;">
                <div style="margin-bottom: 10px;">
                    <div class="discovered-camera-ip">📹 ${cam.ip} (${cam.protocol}:${cam.port})</div>
                    <div style="font-size: 12px; color: #666; margin-top: 5px;">Found ${cam.paths.length} possible camera path(s) - test each with credentials:</div>
                </div>
                ${pathsHtml}
            </div>
        `;
    }).join('') + '</div>';
}

async function testCameraPath(camIdx, pathIdx, url, ip, path) {
    const username = document.getElementById(`camera-${camIdx}-path-${pathIdx}-username`).value.trim();
    const password = document.getElementById(`camera-${camIdx}-path-${pathIdx}-password`).value;
    const button = event.target;

    button.disabled = true;
    button.textContent = '⏳ Testing...';

    try {
        const response = await fetch('/api/cameras/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url, username, password })
        });

        const result = await response.json();

        if (response.ok && result.success) {
            // Show camera preview (pass both display URL and full URL)
            showCameraPreview(ip, path, result.image, result.resolution, result.authenticated_url, result.full_url);
            button.textContent = '✓ Success!';
            button.style.background = '#218838';
            setTimeout(() => {
                button.textContent = '🎥 Test & Preview';
                button.style.background = '#28a745';
            }, 2000);
        } else {
            alert(`Camera test failed: ${result.error || 'Unable to connect'}\n\nTry checking:\n- Username and password are correct\n- This specific path works for your camera model\n- Camera is powered on and connected\n- Network connectivity`);
            button.textContent = '❌ Failed';
            button.style.background = '#dc3545';
            setTimeout(() => {
                button.textContent = '🎥 Test & Preview';
                button.style.background = '#28a745';
            }, 2000);
        }
    } catch (error) {
        console.error('Test error:', error);
        alert(`Error testing camera: ${error.message}`);
        button.textContent = '❌ Error';
        button.style.background = '#dc3545';
        setTimeout(() => {
            button.textContent = '🎥 Test & Preview';
            button.style.background = '#28a745';
        }, 2000);
    } finally {
        button.disabled = false;
    }
}

// Store current camera details for saving
let currentCameraDetails = null;

function showCameraPreview(ip, path, imageData, resolution, authenticatedUrl, fullUrl) {
    const modal = document.getElementById('camera-preview-modal');
    const title = document.getElementById('preview-camera-title');
    const info = document.getElementById('preview-camera-info');
    const image = document.getElementById('preview-camera-image');
    const nameInput = document.getElementById('save-camera-name');

    // Store camera details for saving (use fullUrl which has credentials)
    currentCameraDetails = {
        ip: ip,
        path: path,
        url: fullUrl || authenticatedUrl,  // Use full URL with credentials for saving
        resolution: resolution
    };

    // Auto-suggest camera name
    nameInput.value = `Camera ${ip}`;

    title.textContent = `Camera Preview - ${ip}`;
    info.innerHTML = `
        <div style="margin-bottom: 10px;">
            <strong>IP:</strong> ${ip}<br>
            <strong>Path:</strong> <span style="font-family: monospace; font-size: 12px;">${path}</span><br>
            <strong>URL:</strong> <span style="font-family: monospace; font-size: 12px;">${authenticatedUrl || 'N/A'}</span><br>
            <strong>Resolution:</strong> ${resolution.width} x ${resolution.height}
        </div>
    `;
    image.src = imageData;

    modal.style.display = 'flex';
}

function closeCameraPreview(event) {
    if (!event || event.target.id === 'camera-preview-modal') {
        document.getElementById('camera-preview-modal').style.display = 'none';
        currentCameraDetails = null;
    }
}

async function saveDiscoveredCamera() {
    if (!currentCameraDetails) {
        alert('No camera details available to save');
        return;
    }

    const nameInput = document.getElementById('save-camera-name');
    const cameraName = nameInput.value.trim() || `Camera ${currentCameraDetails.ip}`;
    const saveButton = document.getElementById('save-camera-button');

    saveButton.disabled = true;
    saveButton.textContent = '💾 Saving...';

    try {
        const response = await fetch('/api/cameras/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: cameraName,
                ip: currentCameraDetails.ip,
                url: currentCameraDetails.url,
                path: currentCameraDetails.path,
                resolution: currentCameraDetails.resolution
            })
        });

        const result = await response.json();

        if (response.ok && result.success) {
            saveButton.textContent = '✓ Saved!';
            saveButton.style.background = '#28a745';

            // Show success message
            alert(`Camera "${cameraName}" saved successfully!\n\nYou can now use this camera in the Camera Monitoring section.`);

            // Refresh camera status to show the new camera
            setTimeout(() => {
                fetchCameraStatus();
                closeCameraPreview();
            }, 1000);
        } else {
            alert(`Failed to save camera: ${result.error || 'Unknown error'}`);
            saveButton.textContent = '💾 Save Camera';
            saveButton.style.background = '#007bff';
        }
    } catch (error) {
        console.error('Save error:', error);
        alert(`Error saving camera: ${error.message}`);
        saveButton.textContent = '💾 Save Camera';
        saveButton.style.background = '#007bff';
    } finally {
        saveButton.disabled = false;
    }
}

// Initial camera status fetch (SSE will handle updates)
fetchCameraStatus();
checkServiceConfigStatus();  // Check if saved service config exists

// Fetch and update inference stats
async function fetchInferenceStats() {
    try {
        const response = await fetch('/api/inference/stats');
        const data = await response.json();

        // Update state name
        const stateNameEl = document.getElementById('inference-state-name');
        if (stateNameEl) {
            const stateName = data.state_name || 'unknown';
            stateNameEl.textContent = stateName;
        }

        // Update service name
        const serviceNameEl = document.getElementById('inference-service-name');
        if (serviceNameEl) {
            serviceNameEl.textContent = data.service_type || '...';
        }

        // Update inference latency (per-frame processing time)
        const timeValueEl = document.getElementById('inference-time-value');
        if (timeValueEl) {
            if (data.avg_inference_time_ms > 0) {
                timeValueEl.textContent = data.avg_inference_time_ms + ' ms';
            } else {
                timeValueEl.textContent = 'N/A';
            }
        }

        // Update frame interval (time between completed frames) with FPS
        const intervalValueEl = document.getElementById('frame-interval-value');
        if (intervalValueEl) {
            if (data.avg_frame_interval_ms > 0) {
                const intFps = (1000 / data.avg_frame_interval_ms).toFixed(1);
                intervalValueEl.textContent = data.avg_frame_interval_ms + ' ms (' + intFps + '/s)';
            } else {
                intervalValueEl.textContent = 'N/A';
            }
        }

        // Update inference FPS (data.inference_fps is already in FPS from backend)
        const inferenceFpsValueEl = document.getElementById('inference-fps-value');
        if (inferenceFpsValueEl) {
            if (data.inference_fps > 0) {
                const infFps = parseFloat(data.inference_fps);
                inferenceFpsValueEl.textContent = infFps.toFixed(2) + ' FPS';

                // Color-code based on FPS performance
                if (infFps >= 10) {
                    inferenceFpsValueEl.style.color = 'var(--success-color)';
                } else if (infFps >= 3) {
                    inferenceFpsValueEl.style.color = '#FFA500';
                } else if (infFps >= 1) {
                    inferenceFpsValueEl.style.color = 'var(--warning-color)';
                } else {
                    inferenceFpsValueEl.style.color = 'var(--danger-color)';
                }

                // Trigger inference heartbeat animation
                pulseHeartbeat('inference-heartbeat');
            } else {
                inferenceFpsValueEl.textContent = 'N/A';
            }
        }

        // Update capture FPS (based on camera capture rate)
        const captureFpsValueEl = document.getElementById('capture-fps-value');
        if (captureFpsValueEl) {
            if (data.capture_fps !== undefined && data.capture_fps > 0) {
                captureFpsValueEl.textContent = data.capture_fps.toFixed(2) + ' FPS';
                // Trigger capture heartbeat animation
                pulseHeartbeat('capture-heartbeat');
            } else {
                captureFpsValueEl.textContent = 'N/A';
            }
        }

        // Queue Status Monitoring - Compare Capture vs Inference FPS
        updateQueueStatus(data);
    } catch (error) {
        console.error('Error fetching inference stats:', error);
    }
}

// Queue Status Monitor - Shows autoscaler level + worker counts + system capacity
function updateQueueStatus(data) {
    const queueBar = document.getElementById('queue-status-bar');
    const queueIcon = document.getElementById('queue-icon');
    const queueMessage = document.getElementById('queue-message');
    const queueRatio = document.getElementById('queue-ratio');

    if (!queueBar) return;

    const as = data.autoscaler || {};
    const sc = data.system_capacity || {};
    const diskLvl = as.disk_level || '—';
    const infLvl = as.inf_level || '—';
    const diskW = as.disk_writers || 0;
    const infW = as.inference_workers || 0;
    const diskPct = as.disk_queue_pct || 0;
    const diskQLen = as.disk_queue_len || 0;
    const diskQMax = as.disk_queue_max || 2000;
    const infQ = as.inf_queue_len || 0;
    const infQMax = as.inf_queue_max || 1000;
    const infHot = as.inf_hot || 0;
    const infCold = as.inf_cold || 0;
    const maxDiskW = sc.max_disk_writers || 32;
    const maxInfW = sc.max_inference_workers || 24;
    const cpuCores = sc.cpu_physical || '?';
    const cpuLogical = sc.cpu_logical || '?';
    const ramGB = sc.ram_total_gb || '?';

    // Determine worst level between disk and inference
    const levels = { 'OK': 0, 'WARNING': 1, 'CRITICAL': 2 };
    let worstScore = Math.max(levels[diskLvl] || 0, levels[infLvl] || 0);

    // Also check FPS ratio: if inference is significantly slower than capture, flag it
    const infFps = data.inference ? data.inference.fps : 0;
    const capFps = data.capture_fps || 0;
    let fpsWarning = '';
    if (infFps > 0 && capFps > 0) {
        const ratio = capFps / infFps;
        if (ratio > 10) {
            worstScore = Math.max(worstScore, 2);
            fpsWarning = ` — Cap/Inf ratio: ${ratio.toFixed(0)}x`;
        } else if (ratio > 3) {
            worstScore = Math.max(worstScore, 1);
            fpsWarning = ` — Cap/Inf ratio: ${ratio.toFixed(1)}x`;
        }
    }

    queueBar.style.display = 'flex';

    if (worstScore >= 2) {
        // CRITICAL
        queueBar.style.background = 'rgba(239, 68, 68, 0.2)';
        queueBar.style.border = '1px solid var(--danger-color)';
        queueBar.style.color = 'var(--danger-color)';
        queueIcon.textContent = '🚨';
        const critReason = fpsWarning ? 'inference too slow' + fpsWarning : (diskPct > 25 ? `disk queue ${diskPct}%` : `${infHot} urgent frames waiting`);
        queueMessage.textContent = 'CRITICAL — ' + critReason;
    } else if (worstScore >= 1) {
        // WARNING
        queueBar.style.background = 'rgba(251, 191, 36, 0.2)';
        queueBar.style.border = '1px solid var(--warning-color)';
        queueBar.style.color = 'var(--warning-color)';
        queueIcon.textContent = '⚠️';
        const warnReason = fpsWarning ? 'inference falling behind' + fpsWarning : (diskPct > 5 ? `disk queue ${diskPct}%` : `${infHot} urgent frames waiting`);
        queueMessage.textContent = 'WARNING — ' + warnReason;
    } else {
        // OK
        queueBar.style.background = 'rgba(34, 197, 94, 0.2)';
        queueBar.style.border = '1px solid var(--success-color)';
        queueBar.style.color = 'var(--success-color)';
        queueIcon.textContent = '✓';
        queueMessage.textContent = 'OK — inference keeping up';
    }

    // (Chart counts are drawn directly on the canvas by updateQueueChart / updateInfQueueChart)

    // Update queue usage charts
    updateQueueChart('disk-queue-chart', _diskQueueHistory, diskQLen, diskQMax, '#3b82f6');
    updateInfQueueChart('inf-queue-chart', _infHotHistory, _infColdHistory, infHot, infCold, infQMax);
}

// Queue history buffers for charts (60 data points = ~60 seconds of history)
const _diskQueueHistory = [];
const _infHotHistory = [];
const _infColdHistory = [];
const _QUEUE_HISTORY_MAX = 60;

function updateQueueChart(canvasId, history, currentValue, maxValue, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    // Push new value as percentage
    const pct = maxValue > 0 ? (currentValue / maxValue) * 100 : 0;
    history.push(pct);
    if (history.length > _QUEUE_HISTORY_MAX) history.shift();

    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth * (window.devicePixelRatio || 1);
    const h = canvas.height = 50 * (window.devicePixelRatio || 1);
    ctx.clearRect(0, 0, w, h);

    if (history.length < 2) return;

    const stepX = w / (_QUEUE_HISTORY_MAX - 1);

    // Fill area
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let i = 0; i < history.length; i++) {
        const x = (i + _QUEUE_HISTORY_MAX - history.length) * stepX;
        const y = h - (history[i] / 100) * h;
        ctx.lineTo(x, y);
    }
    ctx.lineTo(w, h);
    ctx.closePath();
    ctx.fillStyle = color + '33';
    ctx.fill();

    // Line
    ctx.beginPath();
    for (let i = 0; i < history.length; i++) {
        const x = (i + _QUEUE_HISTORY_MAX - history.length) * stepX;
        const y = h - (history[i] / 100) * h;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5 * (window.devicePixelRatio || 1);
    ctx.stroke();

    // Current value text (bottom-right)
    ctx.fillStyle = color;
    ctx.font = `${9 * (window.devicePixelRatio || 1)}px Inter, sans-serif`;
    ctx.textAlign = 'right';
    ctx.fillText(`${currentValue}/${maxValue}`, w - 4, h - 4);
}

function updateInfQueueChart(canvasId, hotHistory, coldHistory, hotVal, coldVal, maxVal) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const hotPct = maxVal > 0 ? (hotVal / maxVal) * 100 : 0;
    const coldPct = maxVal > 0 ? (coldVal / maxVal) * 100 : 0;
    hotHistory.push(hotPct);
    coldHistory.push(coldPct);
    if (hotHistory.length > _QUEUE_HISTORY_MAX) hotHistory.shift();
    if (coldHistory.length > _QUEUE_HISTORY_MAX) coldHistory.shift();

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width = canvas.offsetWidth * dpr;
    const h = canvas.height = 50 * dpr;
    ctx.clearRect(0, 0, w, h);

    if (hotHistory.length < 2) return;
    const stepX = w / (_QUEUE_HISTORY_MAX - 1);
    const len = hotHistory.length;

    // Cold area (blue, bottom layer — stacked under hot)
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let i = 0; i < len; i++) {
        const x = (i + _QUEUE_HISTORY_MAX - len) * stepX;
        const y = h - ((coldHistory[i] + hotHistory[i]) / 100) * h;
        ctx.lineTo(x, y);
    }
    ctx.lineTo(w, h);
    ctx.closePath();
    ctx.fillStyle = '#3b82f633';
    ctx.fill();

    // Hot area (red/orange, top layer)
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let i = 0; i < len; i++) {
        const x = (i + _QUEUE_HISTORY_MAX - len) * stepX;
        const y = h - (hotHistory[i] / 100) * h;
        ctx.lineTo(x, y);
    }
    ctx.lineTo(w, h);
    ctx.closePath();
    ctx.fillStyle = '#ef444466';
    ctx.fill();

    // Hot line
    ctx.beginPath();
    for (let i = 0; i < len; i++) {
        const x = (i + _QUEUE_HISTORY_MAX - len) * stepX;
        const y = h - (hotHistory[i] / 100) * h;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 1.5 * dpr;
    ctx.stroke();

    // Labels (bottom-right)
    ctx.font = `${9 * dpr}px Inter, sans-serif`;
    ctx.textAlign = 'right';
    ctx.fillStyle = '#ef4444';
    ctx.fillText(`🔥${hotVal}`, w - 4, h - 14 * dpr);
    ctx.fillStyle = '#3b82f6';
    ctx.fillText(`❄${coldVal}`, w - 4, h - 4);
}

// Heartbeat pulse animation
function pulseHeartbeat(elementId) {
    const heartbeat = document.getElementById(elementId);
    if (!heartbeat) return;

    // Fade in quickly, fade out slowly
    heartbeat.style.transition = 'opacity 0.15s ease-in';
    heartbeat.style.opacity = '1';

    setTimeout(() => {
        heartbeat.style.transition = 'opacity 0.5s ease-out';
        heartbeat.style.opacity = '0';
    }, 150);
}

// Poll for latest detections (fallback for audio when SSE isn't working)
let lastProcessedDetectionTime = 0;
async function pollLatestDetections() {
    console.log('[Polling] pollLatestDetections() called');
    try {
        const response = await fetch('/api/latest_detections');
        console.log('[Polling] Response received:', response.status);
        const data = await response.json();
        console.log('[Polling] Data:', data);

        if (data.has_detection && data.event) {
            const event = data.event;
            // Only process if we haven't seen this timestamp before
            if (event.timestamp !== lastProcessedDetectionTime) {
                lastProcessedDetectionTime = event.timestamp;
                console.log('[Polling] New detection event:', event);

                // Process for audio
                if (event.details && event.details.detections) {
                    processDetectionForAudio(event.details);
                }
            }
        }
    } catch (e) {
        console.error('[Polling] Error fetching detections:', e);
    }
}

// Color based on usage percentage: green → yellow → red
function usageColor(pct) {
    if (pct < 50) return '#4ade80';   // green
    if (pct < 75) return '#facc15';   // yellow
    if (pct < 90) return '#fb923c';   // orange
    return '#f87171';                  // red
}

// Fetch and update system metrics
async function fetchSystemMetrics() {
    try {
        const response = await fetch('/api/system/metrics');
        const data = await response.json();

        // Update CPU
        const cpuValueEl = document.getElementById('system-cpu-value');
        const cpuCoresEl = document.getElementById('system-cpu-cores');
        if (cpuValueEl && data.cpu) {
            cpuValueEl.textContent = `${data.cpu.percent}%`;
            cpuValueEl.style.color = usageColor(data.cpu.percent);
            if (cpuCoresEl) {
                cpuCoresEl.textContent = `${data.cpu.cores_logical} cores`;
            }
        }

        // Update RAM
        const ramValueEl = document.getElementById('system-ram-value');
        const ramDetailsEl = document.getElementById('system-ram-details');
        if (ramValueEl && data.memory) {
            ramValueEl.textContent = `${data.memory.percent}%`;
            ramValueEl.style.color = usageColor(data.memory.percent);
            if (ramDetailsEl) {
                ramDetailsEl.textContent = `${data.memory.used_gb}GB / ${data.memory.total_gb}GB`;
            }
        }

        // Update Disk
        const diskValueEl = document.getElementById('system-disk-value');
        const diskDetailsEl = document.getElementById('system-disk-details');
        if (diskValueEl && data.disk) {
            diskValueEl.textContent = `${data.disk.percent}%`;
            diskValueEl.style.color = usageColor(data.disk.percent);
            if (diskDetailsEl) {
                diskDetailsEl.textContent = `${data.disk.free_gb}GB free`;
            }
        }
    } catch (error) {
        console.error('Error fetching system metrics:', error);
    }
}

// Start Server-Sent Events for real-time status updates (replaces all polling)
startStatusStream();

// Load audio settings from localStorage
loadAudioSettings();

// Pre-populate object classes from model
fetchModelClasses();

// Load ejection procedures
loadProcedures();

// Start timeline image refresh
startTimelineRefresh();

// Initial inference stats fetch and set up refresh interval
fetchInferenceStats();
setInterval(fetchInferenceStats, 2000);  // Update every 2 seconds

// Initial system metrics fetch and set up refresh interval
fetchSystemMetrics();
setInterval(fetchSystemMetrics, 3000);  // Update every 3 seconds

// Polling disabled - using SSE for audio alerts now
// pollLatestDetections();
// setInterval(pollLatestDetections, 1000);

// Initial config fetch (once on page load)
fetchConfig();
fetchHealth();  // Get YOLO status
loadDataFile();
refreshStates();
refreshInferenceConfig();

// Refresh health status periodically (for YOLO status)
setInterval(fetchHealth, 10000);

// Set default API type fields (Gradio with HuggingFace URL)
updateAPITypeFields();

// Apply saved language preference
applyLanguage(currentLang);

// ── Info Tooltips for all configuration fields ──
(function() {
    const tooltips = {
        // Image Processing
        'parent_objects': 'Objects that must be detected as parents before child objects are processed. Use _root when no parent is needed.',
        'remove_raw': 'Delete the original raw image file after DataMatrix has been successfully decoded to save disk space.',

        // DataMatrix
        'valid_dm_sizes': 'Accepted character lengths for DataMatrix codes. Codes with other lengths will be rejected.',
        'confidence_threshold': 'Minimum confidence score (0-1) for DataMatrix detection. Lower = more detections but more false positives.',
        'overlap_threshold': 'Maximum allowed overlap (0-1) between detected DataMatrix regions. Used to filter duplicate detections.',

        // Feature Toggles
        'histogram_feature': 'Analyze color/brightness distribution of captured images for quality metrics.',
        'save_histogram': 'Save histogram analysis images to disk alongside raw captures.',
        'class_count_check': 'Verify that all specified classes have equal detection counts per package. Flags NG if counts differ.',
        'classes_to_check': 'Object classes to include in the equal-count check.',
        'class_confidence': 'Minimum confidence for detections to be counted in the class count check.',
        'light_status': 'Verify light state via serial response before capturing. Slower but ensures correct lighting.',

        // Hardware - PWM
        'upper_pwm': 'PWM duty cycle for the upper light bar. 0 = off, 255 = maximum brightness.',
        'bottom_pwm': 'PWM duty cycle for the bottom light bar. 0 = off, 255 = maximum brightness.',
        'warning_led': 'Control the warning LED indicator on the hardware unit.',

        // OK Configuration
        'ok_counter_adj': 'Manually adjust the OK counter by adding or subtracting a value.',
        'ok_config': 'Settings for the ejector/marker action when a package passes inspection (OK).',

        // NG Configuration
        'ng_counter_adj': 'Manually adjust the NG (reject) counter by adding or subtracting a value.',
        'ng_config': 'Settings for the ejector/marker action when a package fails inspection (NG).',

        // Shared OK/NG fields (matched by label text below)
        'offset_delay': 'Delay in milliseconds after encoder trigger before activating the ejector.',
        'duration_pulses': 'How long the ejector stays active, measured in pulses (each pulse = 16 microseconds).',
        'duration_percent': 'Ejector activation duration as a percentage of the interval between two packages.',
        'encoder_factor': 'Scaling factor applied to encoder counts for timing calculations.',

        // Counter Service
        'ejector_enabled': 'Enable or disable the physical ejector mechanism.',
        'ejector_offset': 'Encoder count delay between detection and ejection. Adjusts for physical distance.',
        'ejector_duration': 'How long the ejector solenoid stays activated in seconds.',
        'ejector_poll': 'How frequently the system checks for packages to eject. Lower = more responsive.',
        'time_between': 'Minimum time gap between two consecutive package captures. Prevents double-captures.',
        'capture_mode': 'Single = one capture per trigger. Multiple = capture from all cameras per trigger.',

        // Serial Port
        'serial_port': 'System device path for the serial connection to the hardware controller.',
        'baud_rate': 'Communication speed for the serial port. Must match the hardware controller setting.',
        'serial_mode': 'Normal mode uses the new protocol. Legacy mode for older hardware controllers.',

        // Watcher
        'downtime_threshold': 'Seconds of no encoder movement before the system reports a downtime event.',
        'external_reset': 'Allow the hardware controller to reset counters via serial command.',
        'verbose_mode': 'Request detailed configuration readback from the hardware controller.',

        // Timeline
        'camera_order': 'Display order of camera columns in the timeline view.',
        'image_rotation': 'Rotate all timeline images by the specified angle.',

        // Redis
        'redis_host': 'Hostname or IP of the Redis server used for inter-service communication.',
        'redis_port': 'Port number for the Redis connection (default: 6379).',

        // AI
        'model_name': 'A friendly name for this AI model configuration.',
        'provider': 'The AI service provider (Claude, ChatGPT, Gemini, or local Ollama).',
        'api_key': 'Authentication key for the AI provider API.',

        // Database
        'profile_name': 'A name to identify this database connection profile.',
        'host': 'Hostname or IP address of the TimescaleDB/PostgreSQL server.',
        'port': 'Port number for the database connection (default: 5432).',
        'database': 'Name of the database to connect to.',
        'user': 'Username for database authentication.',
        'password': 'Password for database authentication.',

        // State Management
        'state_name': 'Unique name for this capture state configuration.',
        'enabled': 'Whether this state is active and can be selected.'
    };

    function createTooltipIcon(text) {
        const span = document.createElement('span');
        span.className = 'info-tooltip';
        span.innerHTML = 'i<span class="info-tooltip-text">' + text + '</span>';
        return span;
    }

    // Apply tooltips to all .control-label elements with data-i18n
    document.querySelectorAll('.control-label[data-i18n]').forEach(label => {
        const key = label.getAttribute('data-i18n');
        if (tooltips[key]) {
            label.appendChild(createTooltipIcon(tooltips[key]));
        }
    });

    // Apply tooltips to labels by text content (for fields without data-i18n)
    const textTooltips = {
        'Image Quality': 'JPEG compression quality for timeline images. Higher = better quality but larger files.',
        'Rows per Page': 'Number of capture rows shown per page in the timeline view.',
        'Total Frames Stored': 'Maximum number of frames kept in the timeline history buffer.',
        'Pipeline Name': 'Unique identifier for this inference pipeline.',
        'Description': 'Optional description of what this pipeline does.',
        'Models in Pipeline': 'Select which inference models run in this pipeline, in order.',
        'Model Name': 'A friendly name to identify this model configuration.',
        'Inference URL': 'The HTTP endpoint for the inference server (Gradio or YOLO).',
        'Model Name (for Gradio)': 'The specific model name exposed by the Gradio API.',
        'Min Confidence': 'Minimum detection confidence (0-1). Lower = more detections, more false positives.',
        'Type': 'Inference backend type: Gradio (HuggingFace) or YOLO (local).',
        'Light Mode': 'Which lights to turn on/off during this capture phase.',
        'Delay (s)': 'Wait time in seconds after setting lights, before capturing.',
        'Cameras': 'Comma-separated camera IDs to capture in this phase.',
        'Steps (-1=loop)': 'Capture every N encoder steps. -1 = continuous capture loop.',
        'Analog (-1=off)': 'Analog sensor threshold. -1 = disabled. N = trigger when value >= N.'
    };
    document.querySelectorAll('.control-label, label').forEach(label => {
        if (label.querySelector('.info-tooltip')) return;
        const text = label.childNodes[0]?.textContent?.trim() || label.textContent.trim();
        for (const [match, tipText] of Object.entries(textTooltips)) {
            if (text.startsWith(match)) {
                label.appendChild(createTooltipIcon(tipText));
                break;
            }
        }
    });
})();
