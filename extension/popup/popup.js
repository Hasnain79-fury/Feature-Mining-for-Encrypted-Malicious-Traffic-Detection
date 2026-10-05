/**
 * Popup Script — Traffic Guardian
 * Renders dashboard, history, and settings from background state.
 */

document.addEventListener('DOMContentLoaded', init);

// ── State ──
let currentTab = 'monitor';
let refreshTimer = null;

let sessionsCache = [];
let sessionsSearchQuery = '';
let sessionsFilterVerdict = 'all';
let historyCache = [];
let historySearchQuery = '';

async function init() {
  setupTabs();
  setupSettings();
  setupFilters();
  await refresh();
  // Auto-refresh every 3 seconds
  refreshTimer = setInterval(refresh, 3000);
}

// ── Severity bands ──
// Bands the *absolute* malicious_prob (independent of the alert threshold,
// which only decides whether an alert fires, not how bad it is).
function getSeverityInfo(item) {
  if (item.whitelisted) return { cls: 'whitelisted', label: 'Whitelisted' };
  if (item.verdict !== 'malicious') return { cls: 'safe', label: 'Safe' };
  const prob = item.malicious_prob || 0;
  const label = `${(prob * 100).toFixed(0)}% MAL`;
  if (prob >= 0.85) return { cls: 'high', label };
  if (prob >= 0.7) return { cls: 'medium', label };
  return { cls: 'low', label };
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
      if (tabId === 'settings') loadWhitelist();
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
      const sev = getSeverityInfo({
        verdict: verdict.verdict,
        malicious_prob: verdict.malicious_prob,
        whitelisted: verdict.whitelisted,
      });
      pillClass = sev.cls;
      pillText = sev.label;
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
        whitelisted: s.verdict.whitelisted,
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
    historyCache = history || [];
  } catch (_) {
    historyCache = [];
  }
  applyHistoryFilters();
}

function getFilteredHistory() {
  if (!historySearchQuery) return historyCache;
  const q = historySearchQuery.toLowerCase();
  return historyCache.filter(i => (i.domain || '').toLowerCase().includes(q));
}

function applyHistoryFilters() {
  renderHistory(getFilteredHistory());
}

function renderHistory(items) {
  const container = document.getElementById('historyList');

  if (items.length === 0) {
    container.innerHTML = '<div class="empty-state">No threats detected yet</div>';
    return;
  }

  container.innerHTML = items.map((item, i) => {
    const time = new Date(item.timestamp).toLocaleString();
    const sev = getSeverityInfo(item);
    return `
      <div class="history-item" data-index="${i}">
        <div>
          <div class="history-domain">${item.domain}</div>
          <div class="history-meta">${time}</div>
        </div>
        <span class="verdict-pill ${sev.cls}">${sev.label}</span>
      </div>`;
  }).join('');

  container.querySelectorAll('.history-item').forEach(el => {
    el.addEventListener('click', () => openDetailPanel(items[Number(el.dataset.index)]));
  });
}

// Clear history button
document.getElementById('clearHistory').addEventListener('click', async () => {
  await sendMessage({ type: 'CLEAR_HISTORY' });
  historyCache = [];
  applyHistoryFilters();
});

// Export history as CSV
document.getElementById('exportHistory').addEventListener('click', () => {
  downloadCsv(toCsv(getFilteredHistory()), `traffic-guardian-history-${Date.now()}.csv`);
});

// ── Sessions (every scanned session, benign + malicious) ──
async function loadSessions() {
  try {
    const sessions = await sendMessage({ type: 'GET_SESSION_LOG' });
    sessionsCache = sessions || [];
  } catch (_) {
    sessionsCache = [];
  }
  applySessionsFilters();
}

function getFilteredSessions() {
  let items = sessionsCache;
  if (sessionsSearchQuery) {
    const q = sessionsSearchQuery.toLowerCase();
    items = items.filter(i => (i.domain || '').toLowerCase().includes(q));
  }
  if (sessionsFilterVerdict !== 'all') {
    items = items.filter(i => i.verdict === sessionsFilterVerdict);
  }
  return items;
}

function applySessionsFilters() {
  renderSessions(getFilteredSessions());
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
    const sev = getSeverityInfo(item);
    return `
      <div class="history-item ${isMalicious ? '' : 'is-safe'}" data-index="${i}">
        <div>
          <div class="history-domain">${item.domain}</div>
          <div class="history-meta">${time} &middot; ${item.packetCount} packets</div>
        </div>
        <span class="verdict-pill ${sev.cls}">${sev.label}</span>
      </div>`;
  }).join('');

  container.querySelectorAll('.history-item').forEach(el => {
    el.addEventListener('click', () => openDetailPanel(items[Number(el.dataset.index)]));
  });
}

// Clear sessions button
document.getElementById('clearSessions').addEventListener('click', async () => {
  await sendMessage({ type: 'CLEAR_SESSIONS' });
  sessionsCache = [];
  applySessionsFilters();
});

// Export sessions as CSV
document.getElementById('exportSessions').addEventListener('click', () => {
  downloadCsv(toCsv(getFilteredSessions()), `traffic-guardian-sessions-${Date.now()}.csv`);
});

// ── CSV Export helpers (shared by Sessions + History) ──
function toCsv(items) {
  const headers = ['Domain', 'Timestamp', 'Verdict', 'Malicious Prob %', 'Confidence %',
    'LSTM %', 'ResNet %', 'XGBoost %', 'Whitelisted', 'Session ID'];
  const pct = (v) => v != null ? (v * 100).toFixed(1) : '';
  const rows = items.map(item => [
    item.domain,
    item.timestamp ? new Date(item.timestamp).toISOString() : '',
    item.verdict,
    pct(item.malicious_prob),
    pct(item.confidence),
    pct(item.branches && item.branches.lstm),
    pct(item.branches && item.branches.resnet),
    pct(item.branches && item.branches.xgboost),
    item.whitelisted ? 'yes' : 'no',
    item.sessionId || '',
  ]);
  const escape = (v) => `"${String(v == null ? '' : v).replace(/"/g, '""')}"`;
  return [headers, ...rows].map(r => r.map(escape).join(',')).join('\r\n');
}

function downloadCsv(csv, filename) {
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ── Search + verdict filter wiring (Sessions & History tabs) ──
function setupFilters() {
  document.getElementById('sessionsSearch').addEventListener('input', (e) => {
    sessionsSearchQuery = e.target.value.trim();
    applySessionsFilters();
  });

  document.getElementById('sessionsFilterToggle').addEventListener('click', (e) => {
    const btn = e.target.closest('.filter-pill');
    if (!btn) return;
    document.querySelectorAll('#sessionsFilterToggle .filter-pill').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    sessionsFilterVerdict = btn.dataset.filter;
    applySessionsFilters();
  });

  document.getElementById('historySearch').addEventListener('input', (e) => {
    historySearchQuery = e.target.value.trim();
    applyHistoryFilters();
  });
}

// ── PCAP Upload ──
// Runs directly in the popup (not relayed through background.js messaging,
// which has payload size limits unsuitable for a multi-MB pcap file).
const DEFAULT_BACKEND_URL = 'http://127.0.0.1:8642';

function getBackendUrl() {
  return new Promise((resolve) => {
    chrome.storage.sync.get('backendUrl', (data) => {
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

    // Best-effort host identification (MAC/hostname/username) — a separate
    // endpoint and a separate capability from the ML classifier. Never
    // blocks or fails the main upload if this doesn't work out.
    let hosts = null;
    try {
      const hostFormData = new FormData();
      hostFormData.append('file', file);
      const hostRes = await fetch(`${backendUrl}/identify_hosts`, { method: 'POST', body: hostFormData });
      if (hostRes.ok) {
        hosts = (await hostRes.json()).hosts;
      }
    } catch (_) {
      // host identification is a bonus, not required
    }

    await importPcapFlows(data.flows, hosts);

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
function importPcapFlows(flows, hosts) {
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
        if (f.verdict === 'malicious') {
          const hostInfo = pickHostInfo(f.flow_id, hosts);
          if (hostInfo) entry.hostInfo = hostInfo;
        }
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

// A flow_id is "ip:port -> ip:port" — either side could be the local
// (infected) host or the remote server. We don't know which for certain
// (see pcap_transform.py), so try both and use whichever one actually has
// identifying data — a remote/external server IP is very unlikely to show
// up in local ARP/DHCP/NTLM chatter, so this self-selects correctly in
// practice.
function pickHostInfo(flowId, hosts) {
  if (!hosts) return null;
  const parts = flowId.split(' -> ');
  if (parts.length !== 2) return null;
  for (const part of parts) {
    const ip = part.substring(0, part.lastIndexOf(':'));
    const h = hosts[ip];
    if (h && (h.mac || h.hostname || h.username)) {
      return { ip, ...h };
    }
  }
  return null;
}

// ── Session Detail Panel ──
function openDetailPanel(item) {
  const pill = document.getElementById('detailVerdictPill');
  const sev = getSeverityInfo(item);
  pill.textContent = item.whitelisted ? 'Whitelisted' : (item.verdict === 'malicious' ? 'Malicious' : 'Benign');
  pill.className = `detail-verdict-pill ${sev.cls}`;

  document.getElementById('detailDomain').textContent = item.domain;
  document.getElementById('detailTime').textContent = item.timestamp
    ? new Date(item.timestamp).toLocaleString() : '—';
  document.getElementById('detailPackets').textContent = item.packetCount ?? '—';
  document.getElementById('detailProb').textContent =
    item.malicious_prob != null ? `${(item.malicious_prob * 100).toFixed(1)}%` : '—';
  document.getElementById('detailConfidence').textContent =
    item.confidence != null ? `${(item.confidence * 100).toFixed(1)}%` : '—';
  document.getElementById('detailSessionId').textContent = item.sessionId || '—';

  // Confidence breakdown: collapsed by default each time a session opens,
  // populated from data already on the item — no extra backend call.
  document.getElementById('confidenceExplain').classList.add('hidden');
  if (item.malicious_prob != null) {
    document.getElementById('confidenceExplainMal').textContent =
      `${(item.malicious_prob * 100).toFixed(1)}%`;
    document.getElementById('confidenceExplainBenign').textContent =
      `${((1 - item.malicious_prob) * 100).toFixed(1)}%`;
  }

  const branchesEl = document.getElementById('detailBranches');
  const b = item.branches || {};
  branchesEl.innerHTML = branchBar('LSTM', b.lstm || 0) +
    branchBar('ResNet', b.resnet || 0) + branchBar('XGBoost', b.xgboost || 0);

  // Mark-as-safe: only for live-traffic sessions that are currently
  // flagged malicious and not already whitelisted. Pcap flows are excluded
  // — a flow_id-based label never recurs, so whitelisting one is meaningless.
  const markSafeBtn = document.getElementById('detailMarkSafe');
  const isPcapFlow = (item.domain || '').endsWith(' (pcap)');
  if (item.verdict === 'malicious' && !item.whitelisted && !isPcapFlow) {
    markSafeBtn.classList.remove('hidden');
    markSafeBtn.onclick = () => markDomainSafe(item.domain);
  } else {
    markSafeBtn.classList.add('hidden');
    markSafeBtn.onclick = null;
  }

  const hostSection = document.getElementById('detailHostSection');
  if (item.hostInfo) {
    hostSection.classList.remove('hidden');
    document.getElementById('detailHostIp').textContent = item.hostInfo.ip || '—';
    document.getElementById('detailHostMac').textContent = item.hostInfo.mac || 'Not found';
    document.getElementById('detailHostName').textContent = item.hostInfo.hostname || 'Not found';
    const user = item.hostInfo.username
      ? (item.hostInfo.domain ? `${item.hostInfo.domain}\\${item.hostInfo.username}` : item.hostInfo.username)
      : 'Not found';
    document.getElementById('detailHostUser').textContent = user;
    document.getElementById('detailHostNote').textContent =
      'From ARP/DHCP/NTLM parsing of the pcap — separate from the ML model, ' +
      'best-effort only. Full name isn\'t typically present in network traffic.';
  } else {
    hostSection.classList.add('hidden');
  }

  document.getElementById('detailPanel').classList.add('open');
}

document.getElementById('detailBack').addEventListener('click', () => {
  document.getElementById('detailPanel').classList.remove('open');
});

document.getElementById('detailConfidenceItem').addEventListener('click', () => {
  document.getElementById('confidenceExplain').classList.toggle('hidden');
});

// ── Whitelist ──
function markDomainSafe(domain) {
  chrome.storage.sync.get('whitelist', (data) => {
    const list = data.whitelist || [];
    if (!list.includes(domain)) list.push(domain);
    chrome.storage.sync.set({ whitelist: list.slice(-200) }, () => {
      document.getElementById('detailPanel').classList.remove('open');
      loadWhitelist();
      refresh();
    });
  });
}

function loadWhitelist() {
  chrome.storage.sync.get('whitelist', (data) => {
    renderWhitelist(data.whitelist || []);
  });
}

function renderWhitelist(list) {
  const container = document.getElementById('whitelistList');

  if (list.length === 0) {
    container.innerHTML = '<div class="empty-state">No domains whitelisted</div>';
    return;
  }

  container.innerHTML = list.map(domain => `
    <div class="whitelist-item">
      <span class="whitelist-domain" title="${domain}">${domain}</span>
      <button class="whitelist-remove" data-domain="${domain}">&times;</button>
    </div>`).join('');

  container.querySelectorAll('.whitelist-remove').forEach(btn => {
    btn.addEventListener('click', () => {
      chrome.storage.sync.get('whitelist', (data) => {
        const updated = (data.whitelist || []).filter(d => d !== btn.dataset.domain);
        chrome.storage.sync.set({ whitelist: updated }, loadWhitelist);
      });
    });
  });
}

// ── Settings ──
function setupSettings() {
  // Settings (+ whitelist) live in chrome.storage.sync so they follow the
  // user across devices; sessionLog/threatLog stay in local (too large for
  // sync's per-item quota).
  chrome.storage.sync.get(['backendUrl', 'threshold', 'notifications', 'autoAnalyze'], (data) => {
    if (data.backendUrl) document.getElementById('backendUrl').value = data.backendUrl;
    if (data.threshold) {
      document.getElementById('threshold').value = data.threshold;
      document.getElementById('thresholdValue').textContent = `${data.threshold}%`;
    }
    if (data.notifications !== undefined) document.getElementById('notifications').checked = data.notifications;
    if (data.autoAnalyze !== undefined) document.getElementById('autoAnalyze').checked = data.autoAnalyze;
  });
  loadWhitelist();

  // Threshold slider live update
  document.getElementById('threshold').addEventListener('input', (e) => {
    document.getElementById('thresholdValue').textContent = `${e.target.value}%`;
  });

  // Save button
  document.getElementById('saveSettings').addEventListener('click', () => {
    chrome.storage.sync.set({
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
