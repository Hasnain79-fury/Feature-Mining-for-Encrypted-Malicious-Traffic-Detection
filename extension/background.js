/**
 * Background Service Worker — Traffic Guardian
 *
 * Wires together: webRequest listener → SessionManager → Backend API
 * Updates badge, stores history, shows notifications.
 */

import { SessionManager } from './utils/session_manager.js';
import { buildPacketFromWebRequest } from './utils/feature_engineer.js';

const BACKEND_URL = 'http://127.0.0.1:8642';
let backendOnline = false;
let threatCount = 0;

// ── Initialize session manager ──
const manager = new SessionManager(handleVerdict);

// ── 1. Listen to completed requests ──
chrome.webRequest.onCompleted.addListener(
  (details) => {
    if (details.tabId < 0) return; // Skip non-tab requests

    try {
      const url = new URL(details.url);
      if (url.protocol !== 'https:' && url.protocol !== 'http:') return;
      const domain = url.hostname;
      if (!domain || domain === 'localhost' || domain === '127.0.0.1') return;

      const packet = buildPacketFromWebRequest(details, 0);
      manager.addPacket(details.tabId, domain, packet);
    } catch (_) { /* ignore invalid URLs */ }
  },
  { urls: ['<all_urls>'] },
  ['responseHeaders']
);

// ── 2. Receive Resource Timing data from content scripts ──
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'RESOURCE_TIMINGS' && sender.tab) {
    for (const entry of msg.data) {
      try {
        const domain = new URL(entry.name).hostname;
        manager.enrichPacket(sender.tab.id, domain, entry);
      } catch (_) { /* ignore */ }
    }
  }

  if (msg.type === 'GET_STATUS') {
    sendResponse({
      backendOnline,
      threatCount,
      sessions: manager.getActiveSessions(),
    });
    return true;
  }

  if (msg.type === 'GET_HISTORY') {
    chrome.storage.local.get('threatLog', (data) => {
      sendResponse(data.threatLog || []);
    });
    return true;
  }

  if (msg.type === 'CLEAR_HISTORY') {
    chrome.storage.local.set({ threatLog: [] });
    threatCount = 0;
    updateBadge('safe');
    sendResponse({ ok: true });
    return true;
  }

  if (msg.type === 'GET_SESSION_LOG') {
    chrome.storage.local.get('sessionLog', (data) => {
      sendResponse(data.sessionLog || []);
    });
    return true;
  }

  if (msg.type === 'CLEAR_SESSIONS') {
    chrome.storage.local.set({ sessionLog: [] });
    sendResponse({ ok: true });
    return true;
  }

  if (msg.type === 'TEST_ALERT') {
    sendTestAlert();
    sendResponse({ ok: true });
    return true;
  }
});

// ── UI demo helper: simulates a threat through the real badge/notification/
// storage code path, WITHOUT calling the backend or the ML models. Useful
// for seeing what a detection looks like when you don't have real malicious
// traffic to test with. Never claims this reflects an actual model verdict.
function sendTestAlert() {
  const domain = 'simulated-threat.example';
  const result = {
    verdict: 'malicious',
    confidence: 0.91,
    malicious_prob: 0.91,
    branches: { lstm: 0.88, resnet: 0.95, xgboost: 0.83 },
  };
  const payload = { session_id: `test:${domain}`, domain, packet_count: 15 };

  threatCount++;
  updateBadge('threat');
  showNotification(domain, result);
  storeToHistory(payload, result);
  storeToSessionLog(payload, result);
  manager.verdicts.set(domain, { ...result, timestamp: Date.now(), domain });
}

// ── 3. Handle verdicts ──
async function handleVerdict(payload) {
  try {
    const res = await fetch(`${BACKEND_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`Backend ${res.status}`);
    const result = await res.json();

    storeToSessionLog(payload, result);

    if (result.verdict === 'malicious') {
      threatCount++;
      updateBadge('threat');
      showNotification(payload.domain, result);
      storeToHistory(payload, result);
    } else {
      updateBadge('safe');
    }

    // Store latest verdict for this domain
    manager.verdicts.set(payload.domain, {
      ...result,
      timestamp: Date.now(),
      domain: payload.domain,
    });

    return result;
  } catch (err) {
    console.warn('[TrafficGuardian] Backend error:', err.message);
    backendOnline = false;
    updateBadge('offline');
    throw err;
  }
}

// ── Badge management ──
function updateBadge(state) {
  if (state === 'threat') {
    chrome.action.setBadgeText({ text: String(threatCount) });
    chrome.action.setBadgeBackgroundColor({ color: '#EF4444' });
  } else if (state === 'safe') {
    chrome.action.setBadgeText({ text: '' });
    chrome.action.setBadgeBackgroundColor({ color: '#10B981' });
  } else if (state === 'offline') {
    chrome.action.setBadgeText({ text: '!' });
    chrome.action.setBadgeBackgroundColor({ color: '#6B7280' });
  }
}

// ── Notifications ──
function showNotification(domain, result) {
  chrome.notifications.create(`threat-${Date.now()}`, {
    type: 'basic',
    iconUrl: 'icons/icon128.png',
    title: '⚠️ Malicious Traffic Detected',
    message: `${domain} — ${(result.malicious_prob * 100).toFixed(1)}% confidence`,
    priority: 2,
  });
}

// ── History storage (malicious verdicts only) ──
function storeToHistory(payload, result) {
  chrome.storage.local.get('threatLog', (data) => {
    const log = data.threatLog || [];
    log.unshift({
      sessionId: payload.session_id,
      domain: payload.domain,
      packetCount: payload.packet_count || (payload.packets ? payload.packets.length : 0),
      verdict: result.verdict,
      confidence: result.confidence,
      malicious_prob: result.malicious_prob,
      branches: result.branches,
      timestamp: Date.now(),
    });
    // Keep last 200 entries
    chrome.storage.local.set({ threatLog: log.slice(0, 200) });
  });
}

// ── Full session log (every scanned session, benign + malicious) ──
function storeToSessionLog(payload, result) {
  chrome.storage.local.get('sessionLog', (data) => {
    const log = data.sessionLog || [];
    log.unshift({
      sessionId: payload.session_id,
      domain: payload.domain,
      packetCount: payload.packet_count || (payload.packets ? payload.packets.length : 0),
      verdict: result.verdict,
      confidence: result.confidence,
      malicious_prob: result.malicious_prob,
      branches: result.branches,
      timestamp: Date.now(),
    });
    // Keep last 300 entries
    chrome.storage.local.set({ sessionLog: log.slice(0, 300) });
  });
}

// ── Health check (every 30s) ──
async function checkHealth() {
  try {
    const res = await fetch(`${BACKEND_URL}/health`, { signal: AbortSignal.timeout(3000) });
    const data = await res.json();
    backendOnline = data.status === 'ok' && data.models_loaded;
    if (!backendOnline) updateBadge('offline');
  } catch (_) {
    backendOnline = false;
    updateBadge('offline');
  }
}

// Initial check + periodic
checkHealth();
setInterval(checkHealth, 30000);

// Clean up when tabs close
chrome.tabs.onRemoved.addListener((tabId) => {
  for (const key of manager.sessions.keys()) {
    if (key.startsWith(`${tabId}:`)) {
      manager.sessions.delete(key);
    }
  }
});

console.log('[TrafficGuardian] Background service worker started.');
