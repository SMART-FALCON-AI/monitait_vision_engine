// ===== Support chat (4.0.381) =====
// Operator-facing support channel. MVE renders this chat itself and relays through
// /api/support/** to a Matrix room; the operator never logs in to anything. A human
// answers from Element on the configured homeserver.
//
// Contract: the Support tab must NEVER throw into the rest of the dashboard. Every
// failure renders as an inline banner and the poller keeps going.

var _supportPollTimer = null;
var _supportSeen = {};          // event_id -> true, so a poll never double-renders
var _supportRoomReady = false;
var _supportPending = null;     // transcript carried in from the AI tab, if any

function _supportEl(id) { return document.getElementById(id); }

function _supportEsc(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function _supportTime(ts) {
    try { return new Date(ts).toLocaleString(); } catch (e) { return ''; }
}

function _supportBanner(kind, text) {
    var box = _supportEl('support-banner');
    if (!box) return;
    if (!text) { box.style.display = 'none'; return; }
    var color = kind === 'error' ? '#ef4444' : (kind === 'warn' ? '#f59e0b' : '#22c55e');
    box.style.display = 'block';
    box.style.borderLeft = '4px solid ' + color;
    box.innerHTML = _supportEsc(text);
}

// --- status -----------------------------------------------------------------

function refreshSupportStatus() {
    return fetch('/api/support/status')
        .then(function (r) { return r.json(); })
        .then(function (s) {
            var dot = _supportEl('support-conn-dot');
            var label = _supportEl('support-conn-label');
            var setup = _supportEl('support-setup');
            var chat = _supportEl('support-chat-wrap');
            _supportRoomReady = !!(s && s.connected);

            if (dot) dot.style.background = _supportRoomReady ? '#22c55e' : '#ef4444';
            if (label) {
                label.textContent = _supportRoomReady
                    ? ((s.site || '') + ' · ' + (s.user_id || ''))
                    : (t('support_offline') || 'Not connected');
            }
            // Not configured at all → show the setup hint instead of a broken chat.
            if (setup) setup.style.display = (s && s.configured) ? 'none' : 'block';
            if (chat) chat.style.display = (s && s.configured) ? 'flex' : 'none';

            if (s && s.configured && !s.connected) {
                _supportBanner('error', (t('support_conn_failed') || 'Cannot reach support') +
                    ': ' + (s.error || ''));
            } else {
                _supportBanner(null, null);
            }
            return s;
        })
        .catch(function (e) {
            _supportBanner('error', 'status: ' + e);
        });
}

// --- messages ---------------------------------------------------------------

function _supportRenderMessage(m) {
    var list = _supportEl('support-messages');
    if (!list || _supportSeen[m.event_id]) return;
    _supportSeen[m.event_id] = true;

    var mine = !!m.mine;
    var row = document.createElement('div');
    row.style.cssText = 'display:flex; margin-bottom:10px; ' +
        (mine ? 'justify-content:flex-end;' : 'justify-content:flex-start;');

    var bubble = document.createElement('div');
    bubble.style.cssText = 'max-width:78%; padding:9px 12px; border-radius:10px; ' +
        'font-size:13px; line-height:1.5; white-space:pre-wrap; word-break:break-word; ' +
        (mine
            ? 'background:var(--primary-color); color:#fff; border-bottom-right-radius:2px;'
            : 'background:rgba(51,65,85,0.75); color:var(--text-primary); border-bottom-left-radius:2px;');

    var who = document.createElement('div');
    who.style.cssText = 'font-size:10px; opacity:0.7; margin-bottom:3px;';
    who.textContent = (mine ? (t('support_you') || 'You') : (m.sender_name || 'support')) +
        ' · ' + _supportTime(m.ts);

    bubble.appendChild(who);
    bubble.appendChild(_supportBody(m));
    row.appendChild(bubble);
    list.appendChild(row);
}

// 4.0.383 - an attachment's `body` is only its FILENAME; the bytes come from the
// server-side proxy at m.media_url. Images render inline, anything else renders as
// a labelled link, so a screenshot from support is actually visible in the tab.
function _supportBody(m) {
    var wrap = document.createElement('div');

    if (m.media_url && (m.msgtype === 'm.image' ||
                        (m.mimetype || '').indexOf('image/') === 0)) {
        var img = document.createElement('img');
        img.src = m.media_url;
        img.alt = m.body || 'image';
        img.loading = 'lazy';
        img.style.cssText = 'display:block; max-width:100%; max-height:320px; ' +
            'border-radius:6px; margin-top:2px; cursor:zoom-in; background:rgba(0,0,0,0.2);';
        img.onclick = function () { window.open(m.media_url, '_blank'); };
        // A broken fetch must still tell the operator SOMETHING was sent.
        img.onerror = function () {
            wrap.textContent = (m.body || 'image') + ' \u2014 ' +
                (t('support_media_failed') || 'attachment could not be loaded');
        };
        wrap.appendChild(img);
        return wrap;
    }

    if (m.media_url) {
        var a = document.createElement('a');
        a.href = m.media_url;
        a.target = '_blank';
        a.rel = 'noopener';
        a.textContent = '\uD83D\uDCCE ' + (m.body || 'attachment');
        a.style.cssText = 'color:inherit; text-decoration:underline;';
        wrap.appendChild(a);
        return wrap;
    }

    wrap.textContent = m.body;
    return wrap;
}

function refreshSupportMessages(scroll) {
    if (!_supportRoomReady) return Promise.resolve();
    return fetch('/api/support/messages?limit=60')
        .then(function (r) { return r.json(); })
        .then(function (d) {
            if (!d || !d.ok) {
                if (d && d.error) _supportBanner('error', d.error);
                return;
            }
            var list = _supportEl('support-messages');
            var atBottom = !list || (list.scrollHeight - list.scrollTop - list.clientHeight < 60);
            var empty = _supportEl('support-empty');
            if (empty) empty.style.display = (d.messages && d.messages.length) ? 'none' : 'block';
            (d.messages || []).forEach(_supportRenderMessage);
            if (list && (scroll || atBottom)) list.scrollTop = list.scrollHeight;
        })
        .catch(function (e) { _supportBanner('error', 'messages: ' + e); });
}

// --- sending ----------------------------------------------------------------

function sendSupportMessage() {
    var input = _supportEl('support-input');
    var btn = _supportEl('support-send-btn');
    if (!input) return;
    var text = (input.value || '').trim();
    if (!text) return;

    // 4.0.384 - no operator decision here any more. Every message carries a one-line
    // site header from the backend, and support pulls detail with !diag when it needs it.
    var payload = { text: text };
    if (_supportPending) {           // transcript escalated from the AI tab
        payload.context = _supportPending;
        _supportPending = null;
        var note = _supportEl('support-context-note');
        if (note) note.style.display = 'none';
    }

    input.value = '';
    if (btn) btn.disabled = true;
    _supportBanner(null, null);

    fetch('/api/support/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
        .then(function (r) { return r.json(); })
        .then(function (d) {
            if (!d || !d.ok) {
                _supportBanner('error', (t('support_send_failed') || 'Could not send') +
                    ': ' + ((d && d.error) || '?'));
                input.value = text;   // give the operator their text back
                return;
            }
            return refreshSupportMessages(true);
        })
        .catch(function (e) {
            _supportBanner('error', 'send: ' + e);
            input.value = text;
        })
        .then(function () { if (btn) btn.disabled = false; });
}

// --- attachment upload (operator -> support) --------------------------------
// 4.0.403 — send a screenshot / log / exported config INTO the support room.
// multipart POST to /api/support/upload; the backend uploads to Matrix and posts
// an m.image/m.file message, which refreshSupportMessages() then renders inline.
function uploadSupportFile(input) {
    if (!input || !input.files || !input.files.length) return;
    var file = input.files[0];
    var statusEl = _supportEl('support-upload-status');
    var attachBtn = _supportEl('support-attach-btn');
    var sendBtn = _supportEl('support-send-btn');
    var MAX = 25 * 1024 * 1024;   // keep in sync with MVE_SUPPORT_MAX_UPLOAD_MB (default 25)

    function setStatus(msg, isErr) {
        if (!statusEl) return;
        statusEl.style.display = msg ? 'block' : 'none';
        statusEl.textContent = msg || '';
        statusEl.style.color = isErr ? 'var(--danger-color, #ef4444)' : 'var(--text-secondary)';
    }
    function done() {
        if (attachBtn) attachBtn.disabled = false;
        if (sendBtn) sendBtn.disabled = false;
        input.value = '';   // reset so the SAME file can be picked again
    }

    if (file.size > MAX) {
        setStatus((t('support_file_too_big') || 'File too large (max 25 MB)'), true);
        input.value = '';
        return;
    }

    if (attachBtn) attachBtn.disabled = true;
    if (sendBtn) sendBtn.disabled = true;
    setStatus((t('support_uploading') || 'Uploading') + ' ' + file.name + ' …', false);

    var fd = new FormData();
    fd.append('file', file, file.name);   // do NOT set Content-Type; browser adds the boundary

    fetch('/api/support/upload', { method: 'POST', body: fd })
        .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d }; }); })
        .then(function (res) {
            var d = res.d || {};
            if (!res.ok || !d.ok) {
                setStatus((t('support_upload_failed') || 'Upload failed') + ': ' + (d.error || '?'), true);
                done();
                return;
            }
            setStatus(null, false);
            done();
            return refreshSupportMessages(true);
        })
        .catch(function (e) {
            setStatus((t('support_upload_failed') || 'Upload failed') + ': ' + e, true);
            done();
        });
}

// --- escalation from the AI tab ---------------------------------------------

// Called by the "Ask a human about this" button in the AI Assistant. Grabs the
// visible AI conversation, switches to Support, and stages it as context so the
// operator's next message arrives WITH what the AI already tried.
function escalateToSupport() {
    var lines = [];
    try {
        var pane = document.getElementById('ai-chat-messages');
        if (pane) {
            pane.querySelectorAll('.chat-message, .message, div[data-role]').forEach(function (n) {
                var txt = (n.innerText || '').trim();
                if (txt) lines.push(txt);
            });
            if (!lines.length) {
                var whole = (pane.innerText || '').trim();
                if (whole) lines.push(whole);
            }
        }
    } catch (e) { /* an unreadable transcript must not block escalation */ }

    var transcript = lines.join('\n').slice(-3000);
    _supportPending = transcript
        ? ((t('support_from_ai') || '[Escalated from AI Assistant]') + '\n' + transcript)
        : null;

    var btn = document.querySelector('.tab-button[data-tab-name="support"]');
    if (btn) { btn.click(); } else if (typeof switchTab === 'function') {
        try { switchTab('support'); } catch (e) { /* no event object — ignore */ }
    }

    var note = _supportEl('support-context-note');
    if (note) note.style.display = _supportPending ? 'block' : 'none';
    var input = _supportEl('support-input');
    if (input) setTimeout(function () { input.focus(); }, 150);
}

// --- lifecycle --------------------------------------------------------------

function openSupportTab() {
    refreshSupportStatus().then(function () { refreshSupportMessages(true); });
    if (_supportPollTimer) clearInterval(_supportPollTimer);
    // Poll only while the tab is open — no background traffic to the homeserver.
    _supportPollTimer = setInterval(function () {
        var pane = document.getElementById('tab-support');
        if (!pane || !pane.classList.contains('active')) {
            clearInterval(_supportPollTimer);
            _supportPollTimer = null;
            return;
        }
        refreshSupportMessages(false);
    }, 5000);
}
