// Dynamic iframe management - iframes are CREATED when tab opens and DESTROYED when leaving
// This prevents any phantom iframe from blocking the page
var iframeConfig = {
    'gallery': { containerId: 'gallery-iframe-container', port: 5000 },
    'grafana': { containerId: 'grafana-iframe-container', port: 3000 }
};

function loadIframeForTab(tabName) {
    // Create iframe for active tab
    var cfg = iframeConfig[tabName];
    if (cfg) {
        var url = 'http://' + window.location.hostname + ':' + cfg.port;
        var container = document.getElementById(cfg.containerId);
        if (container && !container.querySelector('iframe')) {
            var iframe = document.createElement('iframe');
            iframe.src = url;
            iframe.style.cssText = 'width: 100%; height: 100%; border: 2px solid var(--border-color); border-radius: 8px;';
            iframe.allow = 'fullscreen';
            container.appendChild(iframe);
        }
    }
    // DESTROY iframes in other tabs completely
    Object.keys(iframeConfig).forEach(function(t) {
        if (t !== tabName) {
            var cont = document.getElementById(iframeConfig[t].containerId);
            if (cont) { cont.innerHTML = ''; }
        }
    });
}
