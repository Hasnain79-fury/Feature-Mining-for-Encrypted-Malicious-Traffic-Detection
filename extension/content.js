/**
 * Content Script — Resource Timing API collector
 * Injected into every page to harvest detailed request timing data
 * that's not available from the webRequest API.
 */

(function () {
  'use strict';

  const HARVEST_INTERVAL_MS = 2000;

  function harvestTimings() {
    const entries = performance.getEntriesByType('resource');
    if (entries.length === 0) return;

    const data = entries.map(e => ({
      name: e.name,
      duration: e.duration,
      startTime: e.startTime,
      requestStart: e.requestStart,
      responseStart: e.responseStart,
      responseEnd: e.responseEnd,
      transferSize: e.transferSize || 0,
      encodedBodySize: e.encodedBodySize || 0,
      decodedBodySize: e.decodedBodySize || 0,
      protocol: e.nextHopProtocol || 'unknown',
    }));

    try {
      chrome.runtime.sendMessage({ type: 'RESOURCE_TIMINGS', data });
    } catch (_) {
      // Extension context invalidated — stop polling
      clearInterval(intervalId);
    }

    performance.clearResourceTimings();
  }

  const intervalId = setInterval(harvestTimings, HARVEST_INTERVAL_MS);

  // Also harvest once on page load
  if (document.readyState === 'complete') {
    setTimeout(harvestTimings, 500);
  } else {
    window.addEventListener('load', () => setTimeout(harvestTimings, 500));
  }
})();
