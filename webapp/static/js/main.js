'use strict';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const uploadZone    = document.getElementById('uploadZone');
const fileInput     = document.getElementById('fileInput');
const previewBox    = document.getElementById('previewBox');
const previewImg    = document.getElementById('previewImg');
const previewInfo   = document.getElementById('previewInfo');
const removeBtn     = document.getElementById('removeBtn');
const processBtn    = document.getElementById('processBtn');
const compareAllBtn = document.getElementById('compareAllBtn');
const loadingOverlay= document.getElementById('loadingOverlay');
const loadingText   = document.getElementById('loadingText');
const statusRow     = document.getElementById('statusRow');

// Result panel
const resultPanel   = document.getElementById('resultPanel');
const resultTitle   = document.getElementById('resultTitle');
const beforeImg     = document.getElementById('beforeImg');
const afterImg      = document.getElementById('afterImg');
const downloadBtn   = document.getElementById('downloadBtn');
const inScore       = document.getElementById('inScore');
const outScore      = document.getElementById('outScore');
const improvement   = document.getElementById('improvement');
const improveCard   = document.getElementById('improveCard');
const inferTime     = document.getElementById('inferTime');

// Comparison slider
const compWrapper   = document.getElementById('comparisonWrapper');
const compBefore    = document.getElementById('compBefore');
const compDivider   = document.getElementById('compDivider');

// All models panel
const allPanel      = document.getElementById('allPanel');
const allGrid       = document.getElementById('allGrid');
const compareBody   = document.getElementById('compareBody');

// ── State ─────────────────────────────────────────────────────────────────────
let currentFile = null;

const MODEL_LABELS = {
  restormer:  'Restormer',
  realesrgan: 'Real-ESRGAN',
  diffir:     'DiffIR',
};

// ── Upload zone ───────────────────────────────────────────────────────────────
uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });
removeBtn.addEventListener('click', clearFile);

function setFile(file) {
  if (!file.type.startsWith('image/')) { alert('Please upload an image file.'); return; }
  currentFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewInfo.textContent = `${file.name}  ·  ${(file.size / 1024).toFixed(0)} KB`;
  uploadZone.style.display = 'none';
  previewBox.style.display = 'block';
  processBtn.disabled = false;
  compareAllBtn.disabled = false;
  resultPanel.style.display = 'none';
  allPanel.style.display = 'none';
}

function clearFile() {
  currentFile = null;
  previewImg.src = '';
  previewBox.style.display = 'none';
  uploadZone.style.display = '';
  fileInput.value = '';
  processBtn.disabled = true;
  compareAllBtn.disabled = true;
  resultPanel.style.display = 'none';
  allPanel.style.display = 'none';
}

// ── Model card selection ──────────────────────────────────────────────────────
document.querySelectorAll('.model-card').forEach(card => {
  card.addEventListener('click', () => {
    document.querySelectorAll('.model-card').forEach(c => c.classList.remove('active'));
    card.classList.add('active');
    card.querySelector('input[type=radio]').checked = true;
  });
});
// Mark first card active on load
document.querySelector('.model-card').classList.add('active');

function selectedModel() {
  return document.querySelector('input[name=model]:checked')?.value ?? 'restormer';
}

// ── Process single model ──────────────────────────────────────────────────────
processBtn.addEventListener('click', async () => {
  if (!currentFile) return;
  const model = selectedModel();

  showLoading(`Running ${MODEL_LABELS[model]}… (CPU bisa 1–5 menit, harap tunggu)`);
  allPanel.style.display = 'none';

  const fd = new FormData();
  fd.append('image', currentFile);
  fd.append('model', model);

  try {
    const res = await fetch('/api/predict', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) { alert('Error: ' + (data.error ?? res.statusText)); return; }
    showSingleResult(data);
  } catch (err) {
    alert('Network error: ' + err.message);
  } finally {
    hideLoading();
  }
});

function showSingleResult(data) {
  resultTitle.textContent = `Result — ${MODEL_LABELS[data.model] ?? data.model}`;

  // images
  beforeImg.src = data.input_b64;
  afterImg.src  = data.output_b64;

  // Set wrapper height using natural image dimensions (reliable, not layout-dependent)
  afterImg.onload = () => {
    if (afterImg.naturalWidth > 0) {
      const ratio = afterImg.naturalHeight / afterImg.naturalWidth;
      const h = Math.round(compWrapper.offsetWidth * ratio);
      compWrapper.style.height = Math.min(h, 520) + 'px';
    }
  };

  // metrics
  inScore.textContent     = data.input_score.toFixed(1);
  outScore.textContent    = data.output_score.toFixed(1);
  inferTime.textContent   = data.inference_time + 's';

  const pct = data.improvement;
  improvement.textContent = (pct >= 0 ? '+' : '') + pct + '%';
  improveCard.style.borderColor = pct >= 0 ? 'var(--success)' : 'var(--danger)';
  improveCard.style.background  = pct >= 0 ? '#f0fdf4' : '#fff5f5';

  // download
  downloadBtn.href = data.output_b64;
  downloadBtn.download = `deblurred_${data.model}.png`;

  // show panel
  resultPanel.style.display = 'block';
  resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // reset slider to 50%
  compBefore.style.width = '50%';
  compDivider.style.left = '50%';
}

// Init slider once on page load
initSlider();

// ── Compare all ───────────────────────────────────────────────────────────────
compareAllBtn.addEventListener('click', async () => {
  if (!currentFile) return;

  showLoading('Running all 3 models… (CPU bisa 5–15 menit, harap tunggu)');
  resultPanel.style.display = 'none';

  const fd = new FormData();
  fd.append('image', currentFile);

  try {
    const res = await fetch('/api/predict_all', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) { alert('Error: ' + (data.error ?? res.statusText)); return; }
    showAllResults(data);
  } catch (err) {
    alert('Network error: ' + err.message);
  } finally {
    hideLoading();
  }
});

function showAllResults(data) {
  allGrid.innerHTML = '';
  compareBody.innerHTML = '';

  // Input card
  const inputCard = createModelCard('Input (Blur)', data.input_b64, `Score: ${data.input_score.toFixed(1)}`, null);
  allGrid.appendChild(inputCard);

  const modelKeys = ['restormer', 'realesrgan', 'diffir'];

  modelKeys.forEach(key => {
    const m = data.models[key];
    const label = MODEL_LABELS[key];

    if (m.error) {
      const card = document.createElement('div');
      card.className = 'all-model-card';
      card.innerHTML = `
        <div style="height:200px;display:flex;align-items:center;justify-content:center;background:#f8fafc;color:var(--text-muted);font-size:.8rem;padding:12px;text-align:center">
          ${label}<br><small style="color:var(--danger)">${m.error}</small>
        </div>
        <div class="all-model-info"><span class="all-model-name">${label}</span><span class="all-model-score">Not available</span></div>`;
      allGrid.appendChild(card);
    } else {
      const pct = m.improvement;
      const card = createModelCard(
        label, m.output_b64,
        `Score: ${m.output_score.toFixed(1)}  |  ${pct >= 0 ? '+' : ''}${pct}%`,
        m.output_b64, label
      );
      allGrid.appendChild(card);
    }

    // Table row
    const tr = document.createElement('tr');
    if (m.error) {
      tr.innerHTML = `<td>${label}</td><td>${data.input_score.toFixed(1)}</td><td colspan="3" style="color:var(--danger)">${m.error}</td>`;
    } else {
      const pct = m.improvement;
      const cls = pct >= 0 ? 'positive' : 'negative';
      tr.innerHTML = `
        <td>${label}</td>
        <td>${data.input_score.toFixed(1)}</td>
        <td>${m.output_score.toFixed(1)}</td>
        <td class="td-improve ${cls}">${pct >= 0 ? '+' : ''}${pct}%</td>
        <td>${m.inference_time}s</td>`;
    }
    compareBody.appendChild(tr);
  });

  allPanel.style.display = 'block';
  allPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function createModelCard(name, imgSrc, scoreText, downloadSrc, dlName) {
  const card = document.createElement('div');
  card.className = 'all-model-card';

  const img = document.createElement('img');
  img.src = imgSrc;
  img.alt = name;

  const info = document.createElement('div');
  info.className = 'all-model-info';
  info.innerHTML = `<span class="all-model-name">${name}</span><span class="all-model-score">${scoreText}</span>`;

  if (downloadSrc && dlName) {
    const a = document.createElement('a');
    a.href = downloadSrc;
    a.download = `deblurred_${dlName}.png`;
    a.className = 'btn-download';
    a.style.cssText = 'font-size:.7rem;padding:4px 10px;';
    a.textContent = '↓';
    info.appendChild(a);
  }

  card.appendChild(img);
  card.appendChild(info);
  return card;
}

// ── Before/After slider ───────────────────────────────────────────────────────
function initSlider() {
  let dragging = false;
  compBefore.style.width = '50%';
  compDivider.style.left = '50%';

  function setPosition(x) {
    const rect = compWrapper.getBoundingClientRect();
    let pct = ((x - rect.left) / rect.width) * 100;
    pct = Math.max(2, Math.min(98, pct));
    compBefore.style.width  = pct + '%';
    compDivider.style.left  = pct + '%';
  }

  compWrapper.addEventListener('mousedown',  e => { dragging = true; setPosition(e.clientX); });
  window.addEventListener('mouseup',   () => { dragging = false; });
  window.addEventListener('mousemove', e => { if (dragging) setPosition(e.clientX); });

  compWrapper.addEventListener('touchstart', e => { dragging = true; setPosition(e.touches[0].clientX); }, { passive: true });
  window.addEventListener('touchend',   () => { dragging = false; });
  window.addEventListener('touchmove',  e => { if (dragging) setPosition(e.touches[0].clientX); }, { passive: true });
}

// ── Loading helpers ───────────────────────────────────────────────────────────
function showLoading(msg) {
  loadingText.textContent = msg ?? 'Processing…';
  loadingOverlay.style.display = 'flex';
}
function hideLoading() {
  loadingOverlay.style.display = 'none';
}

// ── Model status (header chips) ───────────────────────────────────────────────
async function fetchStatus() {
  statusRow.innerHTML = `<span class="status-chip"><span class="status-dot loading"></span> Checking models…</span>`;
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    statusRow.innerHTML = '';

    Object.entries(data).forEach(([key, info]) => {
      const chip = document.createElement('span');
      chip.className = 'status-chip';
      const dot = document.createElement('span');
      dot.className = 'status-dot ' + (info.loaded ? 'ok' : 'error');
      chip.appendChild(dot);
      chip.appendChild(document.createTextNode(' ' + (MODEL_LABELS[key] ?? key)));
      if (!info.loaded && info.error) chip.title = info.error;
      statusRow.appendChild(chip);

      // update card dots too
      const cardDot = document.getElementById('dot-' + key);
      if (cardDot) cardDot.className = 'model-status-dot ' + (info.loaded ? 'ok' : 'error');
    });
  } catch {
    statusRow.innerHTML = `<span class="status-chip"><span class="status-dot error"></span> Status unavailable</span>`;
  }
}

fetchStatus();
