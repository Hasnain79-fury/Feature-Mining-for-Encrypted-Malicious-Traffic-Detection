/**
 * Feature Engineer — prepares session payload for the backend.
 * Runs in the extension before sending data to the Python backend.
 */

/**
 * Extract content-length from response headers array.
 * @param {Array} headers - chrome.webRequest response headers
 * @returns {number} content length in bytes
 */
export function getContentLength(headers) {
  if (!headers) return 0;
  for (const h of headers) {
    if (h.name.toLowerCase() === 'content-length') {
      return parseInt(h.value, 10) || 0;
    }
  }
  return 0;
}

/**
 * Build a packet data object from a webRequest.onCompleted event.
 * @param {Object} details - chrome.webRequest details
 * @param {number} prevTimestamp - timestamp of previous packet in session
 * @returns {Object} packet data
 */
export function buildPacketFromWebRequest(details, prevTimestamp) {
  const timestamp = details.timeStamp || Date.now();
  const responseSize = getContentLength(details.responseHeaders);

  return {
    timestamp,
    duration: 0,  // Will be enriched by Resource Timing data
    requestSize: 0,  // Not directly available from webRequest
    responseSize,
    transferSize: responseSize,
    encodedBodySize: responseSize,
    decodedBodySize: 0,
    headerSize: 0,
    protocol: 'unknown',
    statusCode: details.statusCode || 0,
    url: details.url,
    ip: details.ip || '',
    type: details.type || '',
    _enriched: false,
  };
}
