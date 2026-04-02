document.addEventListener('DOMContentLoaded', () => {
  const videoFeed      = document.getElementById('videoFeed');
  const startBtn       = document.getElementById('startBtn');
  const stopBtn        = document.getElementById('stopBtn');
  const statusDot      = document.getElementById('systemStatus');
  const statusText     = document.getElementById('statusText');
  const terminal       = document.getElementById('terminalLog');
  const offlineOverlay = document.getElementById('offlineOverlay');
  const scanLine       = document.getElementById('scanLine');
  const feedLabel      = document.getElementById('feedLabel');
  const hudTL          = document.getElementById('hudTL');
  const hudConf        = document.getElementById('hudConf');
  const logCountEl     = document.getElementById('logCount');

  // Both topbar stats and diag panel update in sync
  const latencyEls = [
    document.getElementById('latencyVal'),
    document.getElementById('diagLatency')
  ];
  const confEls = [
    document.getElementById('confidenceVal'),
    document.getElementById('diagConf')
  ];

  const VIDEO_URL = "/video_feed";
  let logCount = 0;
  let processingTimer = null;

  // ── Helpers ───────────────────────────────────
  function ts() {
    return new Date().toLocaleTimeString('en-US', { hour12: false });
  }

  function log(msg, type = 'system') {
    logCount++;
    const p = document.createElement('p');
    p.className = `log-entry ${type}`;
    p.innerText = `[${ts()}] > ${msg}`;
    terminal.appendChild(p);
    terminal.scrollTop = terminal.scrollHeight;
    logCountEl.textContent = `${logCount} ENTRIES`;
  }

  // ── Live clock ────────────────────────────────
  function tickClock() {
    document.getElementById('hudClock').textContent = ts();
  }
  tickClock();
  setInterval(tickClock, 1000);

  // ── Simulated live metrics ────────────────────
  function simulateProcessing() {
    if (startBtn.disabled === false) return; // system stopped

    const lat  = Math.floor(Math.random() * 26 + 20);
    const conf = Math.floor(Math.random() * 15 + 84);

    latencyEls.forEach(el => el.textContent = `${lat}ms`);
    confEls.forEach(el => el.textContent = `${conf}%`);
    hudConf.textContent = `CONF ${conf}%`;

    processingTimer = setTimeout(simulateProcessing, 1600 + Math.random() * 900);
  }

  // ── Start ─────────────────────────────────────
  startBtn.addEventListener('click', () => {
    log('Initializing system...', 'system');
    log('Connecting to camera feed...', 'system');

    startBtn.disabled = true;
    stopBtn.disabled  = false;

    statusDot.classList.add('active');
    statusText.textContent = 'SYSTEM ACTIVE';
    statusText.style.color = 'var(--green)';

    offlineOverlay.classList.add('hidden');
    scanLine.classList.add('active');
    feedLabel.textContent = 'CAM_00 · LIVE';
    hudTL.textContent     = 'INPUT · ACTIVE';

    // Timestamp prevents browser from caching the stream
    videoFeed.src = `${VIDEO_URL}?t=${Date.now()}`;
    videoFeed.classList.add('active');

    log('Stream connected.', 'success');
    log('Model: YOLO26s · Conf threshold: 0.50', 'success');
    log('Inference pipeline active.', 'system');

    simulateProcessing();
  });

  // ── Stop ──────────────────────────────────────
  stopBtn.addEventListener('click', () => {
    log('Termination signal sent...', 'error');

    startBtn.disabled = false;
    stopBtn.disabled  = true;

    statusDot.classList.remove('active');
    statusText.textContent = 'SYSTEM STANDBY';
    statusText.style.color = '';

    offlineOverlay.classList.remove('hidden');
    scanLine.classList.remove('active');
    feedLabel.textContent = 'CAM_00 · OFFLINE';
    hudTL.textContent     = 'INPUT · --';

    videoFeed.src = '';
    videoFeed.classList.remove('active');

    clearTimeout(processingTimer);
    latencyEls.forEach(el => el.textContent = '--ms');
    confEls.forEach(el => el.textContent = '--%');
    hudConf.textContent = 'CONF --';

    log('Feed disconnected.', 'system');
    log('Standby mode.', 'system');
  });

  // ── Boot messages ─────────────────────────────
  log('NeuroSign interface loaded.', 'success');
  log('Model: YOLO26s · Status: READY', 'system');
  log('Awaiting initialization command...', 'system');
});