document.addEventListener('DOMContentLoaded', () => {
  const videoFeed      = document.getElementById('videoFeed');
  const startBtn       = document.getElementById('startBtn');
  const stopBtn        = document.getElementById('stopBtn');
  const clearBtn       = document.getElementById('clearBtn');
  const statusDot      = document.getElementById('systemStatus');
  const statusText     = document.getElementById('statusText');
  const terminal       = document.getElementById('terminalLog');
  const offlineOverlay = document.getElementById('offlineOverlay');
  const scanLine       = document.getElementById('scanLine');
  const feedLabel      = document.getElementById('feedLabel');
  const hudTL          = document.getElementById('hudTL');
  const hudConf        = document.getElementById('hudConf');
  const logCountEl     = document.getElementById('logCount');

  // Transcript elements
  const wordPreview    = document.getElementById('wordPreview');
  const transcriptBody = document.getElementById('transcriptBody');
  const transcriptEmpty= document.getElementById('transcriptEmpty');
  const sentenceCount  = document.getElementById('sentenceCount');
  const saveStatus     = document.getElementById('saveStatus');
  const diagNlp        = document.getElementById('diagNlp');
  const diagWords      = document.getElementById('diagWords');

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
  let transcriptPollTimer = null;
  let knownSentenceCount = 0;
  let lastWordPreview = '';

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
    if (startBtn.disabled === false) return;
    const lat  = Math.floor(Math.random() * 26 + 20);
    const conf = Math.floor(Math.random() * 15 + 84);
    latencyEls.forEach(el => el.textContent = `${lat}ms`);
    confEls.forEach(el    => el.textContent = `${conf}%`);
    hudConf.textContent = `CONF ${conf}%`;
    processingTimer = setTimeout(simulateProcessing, 1600 + Math.random() * 900);
  }

  // ── Transcript: append a single sentence card ─
  function appendSentenceCard(sentence, index) {
    if (transcriptEmpty) transcriptEmpty.style.display = 'none';

    const card = document.createElement('div');
    card.className = 'transcript-card';
    card.innerHTML = `
      <span class="tc-index">#${index + 1}</span>
      <span class="tc-text">${sentence}</span>
    `;

    // Animate in
    card.style.opacity = '0';
    card.style.transform = 'translateY(8px)';
    transcriptBody.appendChild(card);
    transcriptBody.scrollTop = transcriptBody.scrollHeight;

    requestAnimationFrame(() => {
      card.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
      card.style.opacity = '1';
      card.style.transform = 'translateY(0)';
    });
  }

  // ── Transcript: poll backend ──────────────────
  async function pollTranscript() {
    try {
      const res = await fetch('/get_sentence');
      if (!res.ok) return;
      const data = await res.json();

      // Update NLP status badge
      if (data.nlp_ready) {
        diagNlp.textContent = 'NLTK · READY';
        diagNlp.className = 'diag-val ok';
      } else {
        diagNlp.textContent = 'INITIALIZING';
        diagNlp.className = 'diag-val';
      }

      // Update word buffer preview
      const preview = data.current_preview || '---';
      if (preview !== lastWordPreview) {
        lastWordPreview = preview;
        wordPreview.textContent = preview;
        wordPreview.classList.remove('pulse-anim');
        void wordPreview.offsetWidth; // reflow
        wordPreview.classList.add('pulse-anim');
        diagWords.textContent = data.current_words.length;
      }

      // Append any brand-new sentences
      const sentences = data.all_sentences || [];
      if (sentences.length > knownSentenceCount) {
        for (let i = knownSentenceCount; i < sentences.length; i++) {
          appendSentenceCard(sentences[i], i);
          log(`NLP sentence: "${sentences[i]}"`, 'success');
        }
        knownSentenceCount = sentences.length;
        sentenceCount.textContent = `${knownSentenceCount} sentence${knownSentenceCount !== 1 ? 's' : ''}`;

        // Flash save status
        saveStatus.style.color = 'var(--green)';
        setTimeout(() => { saveStatus.style.color = ''; }, 2000);
      }

    } catch (_) { /* server may not be ready yet */ }
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

    videoFeed.src = `${VIDEO_URL}?t=${Date.now()}`;
    videoFeed.classList.add('active');

    log('Stream connected.', 'success');
    log('Model: YOLO11l · Conf threshold: 0.50', 'success');
    log('Inference + NLP pipeline active.', 'system');

    simulateProcessing();

    // Start polling transcript every 1 second
    transcriptPollTimer = setInterval(pollTranscript, 1000);
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
    clearInterval(transcriptPollTimer);

    latencyEls.forEach(el => el.textContent = '--ms');
    confEls.forEach(el    => el.textContent = '--%');
    hudConf.textContent = 'CONF --';

    log('Feed disconnected.', 'system');
    log('Standby mode.', 'system');
  });

  // ── Clear transcript ──────────────────────────
  clearBtn.addEventListener('click', async () => {
    try {
      await fetch('/clear_sentence', { method: 'POST' });
      transcriptBody.innerHTML = '';
      transcriptBody.appendChild(transcriptEmpty);
      transcriptEmpty.style.display = '';
      wordPreview.textContent  = '---';
      lastWordPreview           = '';
      knownSentenceCount        = 0;
      diagWords.textContent     = '0';
      sentenceCount.textContent = '0 sentences';
      log('Transcript cleared.', 'system');
    } catch (e) {
      log('Failed to clear transcript.', 'error');
    }
  });

  // ── Boot messages ─────────────────────────────
  log('NeuroSign v2.0 interface loaded.', 'success');
  log('Model: YOLO11l · Status: READY', 'system');
  log('NLP: Grammar correction model loading...', 'system');
  log('Awaiting initialization command...', 'system');
});