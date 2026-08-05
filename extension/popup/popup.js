/**
 * Popup Script — Traffic Guardian
 * Renders dashboard, history, and settings from background state.
 */

document.addEventListener('DOMContentLoaded', init);

// ── State ──
let currentTab = 'monitor';
let refreshTimer = null;

async function init() {
  setupTabs();
  setupSettings();
  await refresh();
  // Auto-refresh every 3 seconds
  refreshTimer = setInterval(refresh, 3000);
}

// ── Tab Navigation ──
function setupTabs() {
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
      tab.classList.add('active');
      const tabId = tab.dataset.tab;
      document.getElementById(`tab-${tabId}`).classList.add('active');
      currentTab = tabId;

      if (tabId === 'history') loadHistory();
      if (tabId === 'sessions') loadSessions();
    });
  });
}

// ── Main refresh ──
async function refresh() {
  try {
    const status = await sendMessage({ type: 'GET_STATUS' });
    if (status) {
      updateConnectionStatus(status.backendOnline);
      updateShield(status);
      updateStats(status);
      updateDomainList(status.sessions || []);
    }
  } catch (e) {
    updateConnectionStatus(false);
  }
}

// ── Connection Status ──
function updateConnectionStatus(online) {
  const dot = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  dot.className = `status-dot ${online ? 'online' : 'offline'}`;
  text.textContent = online ? 'Backend Online' : 'Backend Offline';
}

// ── Shield ──
function updateShield(status) {
  const container = document.getElementById('shieldContainer');
  const text = document.getElementById('shieldText');
  const sub = document.getElementById('shieldSub');
  const check = document.getElementById('shieldCheck');
  const alert = document.getElementById('shieldAlert');

  if (!status.backendOnline) {
    container.className = 'shield-container offline';
    text.textContent = 'Backend Offline';
    sub.textContent = 'Start the backend server to begin monitoring';
    check.classList.add('hidden');
    alert.classList.remove('hidden');
  } else if (status.threatCount > 0) {
    container.className = 'shield-container threat';
    text.textContent = `${status.threatCount} threat${status.threatCount > 1 ? 's' : ''} detected`;
    sub.textContent = 'Suspicious encrypted traffic identified';
    check.classList.add('hidden');
    alert.classList.remove('hidden');
  } else {
    container.className = 'shield-container safe';
    text.textContent = 'All traffic looks safe';
    sub.textContent = 'Monitoring encrypted connections';
    check.classList.remove('hidden');
    alert.classList.add('hidden');
  }
}

// ── Stats ──
function updateStats(status) {
  const sessions = status.sessions || [];
  document.getElementById('activeSessionCount').textContent = sessions.length;
  document.getElementById('threatDetected').textContent = status.threatCount || 0;

  // Count analyzed (sessions with verdicts)
  const analyzed = sessions.filter(s => s.verdict).length;
  document.getElementById('totalAnalyzed').textContent = analyzed;
}

// ── Domain List ──
function updateDomainList(sessions) {
  const container = document.getElementById('domainList');

  if (sessions.length === 0) {
    container.innerHTML = '<div class="empty-state">No domains being monitored yet</div>';
    return;
  }

  container.innerHTML = sessions.map((s, i) => {
    const verdict = s.verdict;
    let pillClass = 'pending';
    let pillText = 'Scanning...';

    if (verdict) {
      if (verdict.verdict === 'malicious') {
        pillClass = 'malicious';
        pillText = `${(verdict.malicious_prob * 100).toFixed(0)}% MAL`;
      } else {
        pillClass = 'safe';
        pillText = 'Safe';
      }
    }

    // Branch scores bar (shown if verdict exists)
    let branchHTML = '';
    if (verdict && verdict.branches) {
      const b = verdict.branches;
      branchHTML = `
        <div class="branch-scores visible">
          ${branchBar('LSTM', b.lstm || 0)}
          ${branchBar('ResNet', b.resnet || 0)}
          ${branchBar('XGBoost', b.xgboost || 0)}
        </div>`;
    }

    const clickable = verdict ? 'clickable' : '';
    return `
      <div class="domain-item ${clickable}" title="${s.domain}" data-domain-index="${i}">
        <div class="domain-info">
          <div>
            <div class="domain-name">${s.domain}</div>
            <div class="domain-packets">${s.packetCount} packets</div>
          </div>
        </div>
        <span class="verdict-pill ${pillClass}">${pillText}</span>
      </div>
      ${branchHTML}`;
  }).join('');

  container.querySelectorAll('.domain-item.clickable').forEach(el => {
    el.addEventListener('click', () => {
      const s = sessions[Number(el.dataset.domainIndex)];
      openDetailPanel({
        domain: s.domain,
        packetCount: s.packetCount,
        sessionId: s.key,
        verdict: s.verdict.verdict,
        confidence: s.verdict.confidence,
        malicious_prob: s.verdict.malicious_prob,
        branches: s.verdict.branches,
        timestamp: s.verdict.timestamp,
      });
    });
  });
}

function branchBar(name, score) {
  const pct = (score * 100).toFixed(1);
  const width = Math.min(100, Math.max(2, score * 100));
  return `
    <div class="branch-row">
      <span class="branch-name">${name}</span>
      <div class="branch-bar-bg">
        <div class="branch-bar" style="width: ${width}%"></div>
      </div>
      <span class="branch-value">${pct}%</span>
    </div>`;
}

// ── History ──
async function loadHistory() {
  try {
    const history = await sendMessage({ type: 'GET_HISTORY' });
    renderHistory(history || []);
  } catch (_) {
    renderHistory([]);
  }
}

function renderHistory(items) {
  const container = document.getElementById('historyList');

  if (items.length === 0) {
    container.innerHTML = '<div class="empty-state">No threats detected yet</div>';
    return;
  }

  container.innerHTML = items.map((item, i) => {
    const time = new Date(item.timestamp).toLocaleString();
    return `
      <div class="history-item" data-index="${i}">
        <div>
          <div class="history-domain">${item.domain}</div>
          <div class="history-meta">${time}</div>
        </div>
        <span class="history-conf">${(item.malicious_prob * 100).toFixed(1)}%</span>
      </div>`;
  }).join('');

  container.querySelectorAll('.history-item').forEach(el => {
    el.addEventListener('click', () => openDetailPanel(items[Number(el.dataset.index)]));
  });
}

// Clear history button
document.getElementById('clearHistory').addEventListener('click', async () => {
  await sendMessage({ type: 'CLEAR_HISTORY' });
  renderHistory([]);
});

// ── Sessions (every scanned session, benign + malicious) ──
async function loadSessions() {
  try {
    const sessions = await sendMessage({ type: 'GET_SESSION_LOG' });
    renderSessions(sessions || []);
  } catch (_) {
    renderSessions([]);
  }
}

function renderSessions(items) {
  const container = document.getElementById('sessionsList');

  if (items.length === 0) {
    container.innerHTML = '<div class="empty-state">No sessions scanned yet</div>';
    return;
  }

  container.innerHTML = items.map((item, i) => {
    const time = new Date(item.timestamp).toLocaleString();
    const isMalicious = item.verdict === 'malicious';
    const pillClass = isMalicious ? 'malicious' : 'safe';
    const pillText = isMalicious
      ? `${(item.malicious_prob * 100).toFixed(0)}% MAL`
      : 'Safe';
    return `
      <div class="history-item ${isMalicious ? '' : 'is-safe'}" data-index="${i}">
        <div>
          <div class="history-domain">${item.domain}</div>
          <div class="history-meta">${time} &middot; ${item.packetCount} packets</div>
        </div>
        <span class="verdict-pill ${pillClass}">${pillText}</span>
      </div>`;
  }).join('');

  container.querySelectorAll('.history-item').forEach(el => {
    el.addEventListener('click', () => openDetailPanel(items[Number(el.dataset.index)]));
  });
}

// Clear sessions button
document.getElementById('clearSessions').addEventListener('click', async () => {
  await sendMessage({ type: 'CLEAR_SESSIONS' });
  renderSessions([]);
});

// ── PCAP Upload ──
// Runs directly in the popup (not relayed through background.js messaging,
// which has payload size limits unsuitable for a multi-MB pcap file).
const DEFAULT_BACKEND_URL = 'http://127.0.0.1:8642';

function getBackendUrl() {
  return new Promise((resolve) => {
    chrome.storage.local.get('backendUrl', (data) => {
      resolve(data.backendUrl || DEFAULT_BACKEND_URL);
    });
  });
}

function setUploadStatus(text, kind) {
  const el = document.getElementById('uploadStatus');
  el.textContent = text;
  el.className = `upload-status ${kind || ''}`;
  el.classList.remove('hidden');
}

document.getElementById('uploadPcapBtn').addEventListener('click', () => {
  document.getElementById('pcapFileInput').click();
});

document.getElementById('pcapFileInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  e.target.value = ''; // allow re-selecting the same file later
  if (!file) return;

  const btn = document.getElementById('uploadPcapBtn');
  btn.disabled = true;
  btn.textContent = 'Analyzing…';
  setUploadStatus(`Analyzing ${file.name}…`, '');

  try {
    const backendUrl = await getBackendUrl();
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${backendUrl}/predict_pcap`, { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || `Backend returned ${res.status}`);
    }

    await importPcapFlows(data.flows);

    const maliciousCount = data.flows.filter(f => f.verdict === 'malicious').length;
    setUploadStatus(
      `Analyzed ${data.flows_analyzed} of ${data.flows_found} flow(s) — ` +
      `${maliciousCount} malicious, ${data.flows_analyzed - maliciousCount} benign.`,
      maliciousCount > 0 ? 'error' : 'success'
    );

    if (maliciousCount > 0) {
      await sendMessage({ type: 'PCAP_IMPORTED', maliciousCount });
    }

    await loadSessions();
    await refresh();
  } catch (err) {
    setUploadStatus(`Upload failed: ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = '📁 Upload PCAP';
  }
});

// Write each analyzed flow into the same storage shape background.js uses
// for live-captured sessions, so they render through the existing list +
// detail panel with no extra rendering code.
function importPcapFlows(flows) {
  return new Promise((resolve) => {
    chrome.storage.local.get(['sessionLog', 'threatLog'], (data) => {
      const sessionLog = data.sessionLog || [];
      const threatLog = data.threatLog || [];
      const now = Date.now();

      for (const f of flows) {
        if (f.verdict === 'error') continue; // skip flows the transform failed on
        const entry = {
          sessionId: f.flow_id,
          domain: `${f.label} (pcap)`,
          packetCount: f.packet_count,
          verdict: f.verdict,
          confidence: f.confidence,
          malicious_prob: f.malicious_prob,
          branches: f.branches,
          timestamp: now,
        };
        sessionLog.unshift(entry);
        if (f.verdict === 'malicious') threatLog.unshift(entry);
      }

      chrome.storage.local.set({
        sessionLog: sessionLog.slice(0, 300),
        threatLog: threatLog.slice(0, 200),
      }, resolve);
    });
  });
}

// ── Session Detail Panel ──
function openDetailPanel(item) {
  const pill = document.getElementById('detailVerdictPill');
  pill.textContent = item.verdict === 'malicious' ? 'Malicious' : 'Benign';
  pill.className = `detail-verdict-pill ${item.verdict === 'malicious' ? 'malicious' : 'benign'}`;

  document.getElementById('detailDomain').textContent = item.domain;
  document.getElementById('detailTime').textContent = item.timestamp
    ? new Date(item.timestamp).toLocaleString() : '—';
  document.getElementById('detailPackets').textContent = item.packetCount ?? '—';
  document.getElementById('detailProb').textContent =
    item.malicious_prob != null ? `${(item.malicious_prob * 100).toFixed(1)}%` : '—';
  document.getElementById('detailConfidence').textContent =
    item.confidence != null ? `${(item.confidence * 100).toFixed(1)}%` : '—';
  document.getElementById('detailSessionId').textContent = item.sessionId || '—';

  const branchesEl = document.getElementById('detailBranches');
  const b = item.branches || {};
  branchesEl.innerHTML = branchBar('LSTM', b.lstm || 0) +
    branchBar('ResNet', b.resnet || 0) + branchBar('XGBoost', b.xgboost || 0);

  document.getElementById('detailPanel').classList.add('open');
}

document.getElementById('detailBack').addEventListener('click', () => {
  document.getElementById('detailPanel').classList.remove('open');
});

// ── Settings ──
function setupSettings() {
  // Load saved settings
  chrome.storage.local.get(['backendUrl', 'threshold', 'notifications', 'autoAnalyze'], (data) => {
    if (data.backendUrl) document.getElementById('backendUrl').value = data.backendUrl;
    if (data.threshold) {
      document.getElementById('threshold').value = data.threshold;
      document.getElementById('thresholdValue').textContent = `${data.threshold}%`;
    }
    if (data.notifications !== undefined) document.getElementById('notifications').checked = data.notifications;
    if (data.autoAnalyze !== undefined) document.getElementById('autoAnalyze').checked = data.autoAnalyze;
  });

  // Threshold slider live update
  document.getElementById('threshold').addEventListener('input', (e) => {
    document.getElementById('thresholdValue').textContent = `${e.target.value}%`;
  });

  // Save button
  document.getElementById('saveSettings').addEventListener('click', () => {
    chrome.storage.local.set({
      backendUrl: document.getElementById('backendUrl').value,
      threshold: parseInt(document.getElementById('threshold').value),
      notifications: document.getElementById('notifications').checked,
      autoAnalyze: document.getElementById('autoAnalyze').checked,
    });

    const btn = document.getElementById('saveSettings');
    btn.textContent = '✓ Saved';
    btn.style.background = 'linear-gradient(135deg, #10b981, #059669)';
    setTimeout(() => {
      btn.textContent = 'Save Settings';
      btn.style.background = '';
    }, 1500);
  });

  // Test alert button (UI demo, does not call the backend)
  document.getElementById('sendTestAlert').addEventListener('click', async () => {
    await sendMessage({ type: 'TEST_ALERT' });
    await refresh();
  });
}

// ── Messaging Helper ──
function sendMessage(msg) {
  return new Promise((resolve) => {
    try {
      chrome.runtime.sendMessage(msg, (response) => {
        resolve(response);
      });
    } catch (_) {
      resolve(null);
    }
  });
}
