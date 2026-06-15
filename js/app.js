/* ── State ─────────────────────────────────────────────────── */
let selectedModel = 'lr_tfidf';
let modelsData = [];

/* ── Init ──────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', async () => {
  await checkHealth();
  await loadModels();
  setupTextarea();
});

/* ── Health check ──────────────────────────────────────────── */
async function checkHealth() {
  const status = document.getElementById('apiStatus');
  try {
    const r = await fetch('/health');
    const d = await r.json();
    const dot = status.querySelector('.dot');
    dot.classList.add('online');
    const loaded = d.models_loaded?.length || 0;
    status.innerHTML = `<span class="dot online"></span> API en ligne · ${loaded} modèle(s) chargé(s)`;
  } catch {
    status.innerHTML = `<span class="dot offline"></span> API hors ligne`;
  }
}

/* ── Load models ───────────────────────────────────────────── */
async function loadModels() {
  try {
    const r = await fetch('/models');
    modelsData = await r.json();
    renderModelGrid();
    renderModelsList();
  } catch (e) {
    showToast('Impossible de charger les modèles', 'error');
  }
}

function renderModelGrid() {
  const grid = document.getElementById('modelGrid');
  grid.innerHTML = modelsData.map(m => `
    <button class="model-option ${m.key === selectedModel ? 'selected' : ''}"
            onclick="selectModel('${m.key}')">
      <div class="mo-radio"></div>
      <div class="mo-info">
        <div class="mo-name">
          ${m.name}
          <span class="mo-badge ${m.contextual ? 'ctx' : 'off'}">${m.contextual ? 'Contextuel' : 'Statique'}</span>
          <span class="mo-badge ${m.loaded ? 'loaded' : ''}">${m.loaded ? '✓ Chargé' : 'Mock'}</span>
        </div>
        <div class="mo-rep">${m.representation}</div>
        ${m.f1_score ? `<div class="mo-f1">F1 : ${(m.f1_score * 100).toFixed(0)}%</div>` : ''}
      </div>
    </button>
  `).join('');
}

const RANK_EMOJIS = ['🥇','🥈','🥉','4️⃣'];
function renderModelsList() {
  const list = document.getElementById('modelsList');
  list.innerHTML = modelsData.map((m, i) => `
    <div class="model-info-item">
      <div class="mi-rank">${RANK_EMOJIS[i] || (i+1)}</div>
      <div class="mi-body">
        <div class="mi-name">${m.name}</div>
        <div class="mi-desc">${m.description}</div>
        <div class="mi-tags">
          <span class="mi-tag mi-tag-rep">${m.representation}</span>
          <span class="mi-tag ${m.contextual ? 'mi-tag-ctx' : 'mi-tag-no'}">${m.contextual ? '✓ Contextuel' : '✗ Non contextuel'}</span>
          <span class="mi-loaded ${m.loaded ? 'yes' : 'no'}">${m.loaded ? '✓ Prêt' : '○ Fichier manquant'}</span>
        </div>
      </div>
      ${m.f1_score ? `<div class="mi-f1">${(m.f1_score * 100).toFixed(0)}%</div>` : ''}
    </div>
  `).join('');
}

function selectModel(key) {
  selectedModel = key;
  renderModelGrid();
}

/* ── Textarea ───────────────────────────────────────────────── */
function setupTextarea() {
  const ta = document.getElementById('inputText');
  const cc = document.getElementById('charCount');
  ta.addEventListener('input', () => {
    cc.textContent = ta.value.length;
    if (ta.value.length > 460) cc.style.color = '#ef4444';
    else cc.style.color = '';
  });
}

function setExample(text) {
  document.getElementById('inputText').value = text;
  document.getElementById('charCount').textContent = text.length;
}

function clearAll() {
  document.getElementById('inputText').value = '';
  document.getElementById('charCount').textContent = '0';
  document.getElementById('resultCard').style.display = 'none';
  document.getElementById('compareCard').style.display = 'none';
}

/* ── Predict ────────────────────────────────────────────────── */
async function runPredict() {
  const text = document.getElementById('inputText').value.trim();
  if (!text) { showToast('Veuillez entrer un texte', 'error'); return; }

  showOverlay(true);
  try {
    const r = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, model: selectedModel }),
    });
    if (!r.ok) {
      const e = await r.json();
      throw new Error(e.detail || 'Erreur serveur');
    }
    const data = await r.json();
    renderResult(data);
    document.getElementById('compareCard').style.display = 'none';
  } catch (e) {
    showToast(e.message, 'error');
  } finally {
    showOverlay(false);
  }
}

function renderResult(data) {
  const card = document.getElementById('resultCard');
  const isPos = data.label === 'positive';
  const conf = Math.round(data.confidence * 100);

  card.className = `card result-card ${isPos ? 'pos' : 'neg'}`;
  card.style.display = 'block';

  document.getElementById('resultIcon').textContent = isPos ? '😊' : '😞';

  const labelEl = document.getElementById('resultLabel');
  labelEl.textContent = isPos ? 'POSITIF' : 'NÉGATIF';
  labelEl.className = `result-label ${isPos ? 'pos' : 'neg'}`;

  document.getElementById('resultModel').textContent = data.model;
  document.getElementById('resultConfidence').textContent = `${conf}%`;
  document.getElementById('resultLatency').textContent = `${data.latency_ms} ms`;
  document.getElementById('resultConf2').textContent = `${conf}%`;

  const bar = document.getElementById('confidenceBar');
  bar.className = `confidence-bar-fill ${isPos ? 'pos' : 'neg'}`;
  // For positive: fill from left. For negative: invert.
  bar.style.width = isPos ? `${conf}%` : `${100 - conf}%`;
  // Adjust position for neg
  if (!isPos) {
    bar.style.marginLeft = `${conf}%`;
  } else {
    bar.style.marginLeft = '0';
  }

  const steps = document.getElementById('stepsList');
  steps.innerHTML = (data.preprocessing_steps || []).map(s => `<li>${s}</li>`).join('');

  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ── Compare ────────────────────────────────────────────────── */
async function runCompare() {
  const text = document.getElementById('inputText').value.trim();
  if (!text) { showToast('Veuillez entrer un texte', 'error'); return; }

  showOverlay(true);
  try {
    const r = await fetch('/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, model: selectedModel }),
    });
    if (!r.ok) throw new Error('Erreur serveur');
    const data = await r.json();
    renderCompare(data);
    document.getElementById('resultCard').style.display = 'none';
  } catch (e) {
    showToast(e.message, 'error');
  } finally {
    showOverlay(false);
  }
}

function renderCompare(data) {
  const card = document.getElementById('compareCard');
  card.style.display = 'block';

  const prev = document.getElementById('compareText');
  prev.textContent = `"${data.text.substring(0, 80)}${data.text.length > 80 ? '…' : ''}"`;

  const grid = document.getElementById('compareGrid');
  grid.innerHTML = Object.entries(data.comparisons).map(([key, res]) => {
    if (res.error) return `<div class="compare-item"><div class="ci-model">${key}</div><div class="ci-label" style="color:var(--text3)">Erreur</div><div class="ci-conf">${res.error}</div></div>`;
    const isPos = res.label === 'positive';
    const conf = Math.round(res.confidence * 100);
    const modelMeta = modelsData.find(m => m.name === res.model) || {};
    return `
      <div class="compare-item ${isPos ? 'pos' : 'neg'}">
        <div class="ci-model">${res.model}</div>
        <div class="ci-label ${isPos ? 'pos' : 'neg'}">${isPos ? '😊 POSITIF' : '😞 NÉGATIF'}</div>
        <div class="ci-bar-track"><div class="ci-bar-fill ${isPos ? 'pos' : 'neg'}" style="width:${conf}%"></div></div>
        <div class="ci-conf">Confiance : ${conf}%</div>
        <div class="ci-latency">⏱ ${res.latency_ms} ms · F1: ${modelMeta.f1_score ? (modelMeta.f1_score*100).toFixed(0)+'%' : '—'}</div>
      </div>
    `;
  }).join('');

  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ── Helpers ────────────────────────────────────────────────── */
function showOverlay(show) {
  document.getElementById('overlay').classList.toggle('active', show);
  document.getElementById('btnPredict').disabled = show;
  document.getElementById('btnCompare').disabled = show;
}

let toastTimer;
function showToast(msg, type = '') {
  document.querySelectorAll('.toast').forEach(t => t.remove());
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  document.body.appendChild(t);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.remove(), 3500);
}

/* ── Keyboard shortcut Ctrl+Enter ───────────────────────────── */
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') runPredict();
});
