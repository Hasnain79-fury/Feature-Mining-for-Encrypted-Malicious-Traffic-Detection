/**
 * Traffic Guardian — Admin Explainability frontend.
 * Standalone page: talks only to this admin service (127.0.0.1:8643),
 * never to the live backend (8642) or the Chrome extension.
 */

const ADMIN_URL = window.location.origin;

let currentFile = null;   // kept in memory; re-submitted for each explain call
let currentFlows = [];

document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  setInterval(checkHealth, 30000);

  document.getElementById('uploadBtn').addEventListener('click', () => {
    document.getElementById('pcapFileInput').click();
  });

  document.getElementById('pcapFileInput').addEventListener('change', onFileSelected);
});

async function checkHealth() {
  const dot = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  try {
    const res = await fetch(`${ADMIN_URL}/health`, { signal: AbortSignal.timeout(3000) });
    const data = await res.json();
    const online = data.status === 'ok' && data.explainer_loaded;
    dot.className = `status-dot ${online ? 'online' : 'offline'}`;
    text.textContent = online ? 'Explainer Loaded' : 'Models Not Loaded';
  } catch (_) {
    dot.className = 'status-dot offline';
    text.textContent = 'Admin Service Offline';
  }
}

function setUploadStatus(msg, kind) {
  const el = document.getElementById('uploadStatus');
  el.textContent = msg;
  el.className = `upload-status ${kind || ''}`;
}

async function onFileSelected(e) {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;

  currentFile = file;
  currentFlows = [];
  document.getElementById('flowsPanel').classList.add('hidden');
  document.getElementById('explainPanel').classList.add('hidden');

  const btn = document.getElementById('uploadBtn');
  btn.disabled = true;
  setUploadStatus(`Analyzing ${file.name}…`, '');

  try {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${ADMIN_URL}/admin/list_flows`, { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Server returned ${res.status}`);

    currentFlows = data.flows;
    setUploadStatus(`Found ${data.flows_found} flow(s), listing ${data.flows.length}.`, 'success');
    renderFlows(data.flows);
  } catch (err) {
    setUploadStatus(`Failed: ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
  }
}

function renderFlows(flows) {
  const panel = document.getElementById('flowsPanel');
  const container = document.getElementById('flowsList');

  if (flows.length === 0) {
    container.innerHTML = '<div class="empty-state">No flows with enough packets found in this capture.</div>';
    panel.classList.remove('hidden');
    return;
  }

  container.innerHTML = flows.map(f => {
    const pillClass = f.verdict === 'malicious' ? 'malicious' : (f.verdict === 'error' ? 'error' : 'safe');
    const pillText = f.verdict === 'malicious'
      ? `${(f.malicious_prob * 100).toFixed(0)}% MAL`
      : (f.verdict === 'error' ? 'ERROR' : 'Safe');
    return `
      <div class="flow-item" data-flow-index="${f.flow_index}">
        <div>
          <div class="flow-label">${f.label}</div>
          <div class="flow-meta">${f.packet_count} packets</div>
        </div>
        <span class="verdict-pill ${pillClass}">${pillText}</span>
      </div>`;
  }).join('');

  container.querySelectorAll('.flow-item').forEach(el => {
    el.addEventListener('click', () => selectFlow(Number(el.dataset.flowIndex)));
  });

  panel.classList.remove('hidden');
}

async function selectFlow(flowIndex) {
  document.querySelectorAll('.flow-item').forEach(el => {
    el.classList.toggle('selected', Number(el.dataset.flowIndex) === flowIndex);
  });

  const panel = document.getElementById('explainPanel');
  panel.classList.remove('hidden');
  document.getElementById('explainSummary').innerHTML = '<div class="empty-state">Computing SHAP explanation…</div>';
  document.getElementById('xgbMalicious').innerHTML = '';
  document.getElementById('xgbBenign').innerHTML = '';
  document.getElementById('rfMalicious').innerHTML = '';
  document.getElementById('rfBenign').innerHTML = '';

  try {
    const formData = new FormData();
    formData.append('file', currentFile);
    const res = await fetch(`${ADMIN_URL}/admin/explain_flow?flow_index=${flowIndex}`, {
      method: 'POST', body: formData,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Server returned ${res.status}`);
    renderExplanation(data);
  } catch (err) {
    document.getElementById('explainSummary').innerHTML =
      `<div class="empty-state">Explanation failed: ${err.message}</div>`;
  }
}

function renderExplanation(data) {
  const pillClass = data.verdict === 'malicious' ? 'malicious' : 'safe';
  document.getElementById('explainSummary').innerHTML = `
    <div class="summary-item">
      <span class="summary-label">Verdict</span>
      <span class="summary-value"><span class="verdict-pill ${pillClass}" style="margin-left:0">${data.verdict}</span></span>
    </div>
    <div class="summary-item">
      <span class="summary-label">Malicious Prob.</span>
      <span class="summary-value">${(data.malicious_prob * 100).toFixed(1)}%</span>
    </div>
    <div class="summary-item">
      <span class="summary-label">Confidence</span>
      <span class="summary-value">${(data.confidence * 100).toFixed(1)}%</span>
    </div>
    <div class="summary-item">
      <span class="summary-label">Branches (L/R/X)</span>
      <span class="summary-value" style="font-size:12px">
        ${(data.branches.lstm * 100).toFixed(0)}% / ${(data.branches.resnet * 100).toFixed(0)}% / ${(data.branches.xgboost * 100).toFixed(0)}%
      </span>
    </div>`;

  renderContribList('xgbMalicious', data.xgb_explanation.top_malicious, 'malicious');
  renderContribList('xgbBenign', data.xgb_explanation.top_benign, 'benign');
  renderContribList('rfMalicious', data.rf_explanation.top_malicious, 'malicious');
  renderContribList('rfBenign', data.rf_explanation.top_benign, 'benign');
}

function renderContribList(containerId, items, direction) {
  const container = document.getElementById(containerId);
  if (!items || items.length === 0) {
    container.innerHTML = '<div class="empty-state" style="padding:8px">None</div>';
    return;
  }
  const maxAbs = Math.max(...items.map(it => Math.abs(it.shap_value)), 0.0001);
  container.innerHTML = items.map(it => {
    const widthPct = Math.min(100, Math.max(4, (Math.abs(it.shap_value) / maxAbs) * 100));
    const medianTag = it.always_median
      ? '<span class="median-badge" title="Always at training median regardless of input — not session-specific">median-filled</span>'
      : '';
    return `
      <div class="contrib-row">
        <span class="contrib-name" title="${it.feature}">${it.feature}</span>
        <div class="contrib-bar-bg"><div class="contrib-bar ${direction}" style="width:${widthPct}%"></div></div>
        <span class="contrib-value">${it.shap_value.toFixed(4)}</span>
        ${medianTag}
      </div>`;
  }).join('');
}
