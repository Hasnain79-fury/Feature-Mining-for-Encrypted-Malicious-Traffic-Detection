/**
 * Session Manager — groups requests by (tabId, domain) into sessions.
 * Flushes sessions to the backend when enough packets accumulate or on timeout.
 */

export class SessionManager {
  constructor(onFlush) {
    this.sessions = new Map();
    this.verdicts = new Map();
    this.onFlush = onFlush;
    this.MIN_PACKETS = 15;
    this.IDLE_TIMEOUT_MS = 30000;

    // Periodic idle check
    setInterval(() => this._checkIdle(), 10000);
  }

  /**
   * Add a packet to the appropriate session.
   */
  addPacket(tabId, domain, packetData) {
    if (!domain || domain === 'localhost' || domain === '127.0.0.1') return;

    const key = `${tabId}:${domain}`;

    if (!this.sessions.has(key)) {
      this.sessions.set(key, {
        key,
        domain,
        tabId,
        packets: [],
        firstSeen: Date.now(),
        lastActivity: Date.now(),
        flushed: false,
      });
    }

    const session = this.sessions.get(key);
    session.packets.push(packetData);
    session.lastActivity = Date.now();

    if (session.packets.length >= this.MIN_PACKETS && !session.flushed) {
      this._flush(key);
    }
  }

  /**
   * Enrich an existing packet with Resource Timing data.
   */
  enrichPacket(tabId, domain, timingEntry) {
    const key = `${tabId}:${domain}`;
    const session = this.sessions.get(key);
    if (!session) {
      // Create new session from timing data
      this.addPacket(tabId, domain, {
        timestamp: performance.timeOrigin + timingEntry.startTime,
        duration: timingEntry.duration,
        requestSize: 0,
        responseSize: timingEntry.encodedBodySize || 0,
        transferSize: timingEntry.transferSize || 0,
        encodedBodySize: timingEntry.encodedBodySize || 0,
        decodedBodySize: timingEntry.decodedBodySize || 0,
        headerSize: Math.max(0, (timingEntry.transferSize || 0) - (timingEntry.encodedBodySize || 0)),
        protocol: timingEntry.protocol || 'unknown',
        statusCode: 200,
      });
      return;
    }

    // Try to match and enrich the most recent unmatched packet for this session
    const url = timingEntry.name;
    for (let i = session.packets.length - 1; i >= 0; i--) {
      const pkt = session.packets[i];
      if (!pkt._enriched && (!pkt.url || pkt.url === url)) {
        pkt.duration = timingEntry.duration || pkt.duration;
        pkt.transferSize = timingEntry.transferSize || pkt.transferSize;
        pkt.encodedBodySize = timingEntry.encodedBodySize || pkt.encodedBodySize;
        pkt.decodedBodySize = timingEntry.decodedBodySize || pkt.decodedBodySize;
        pkt.headerSize = Math.max(0, (pkt.transferSize || 0) - (pkt.encodedBodySize || 0));
        pkt.protocol = timingEntry.protocol || pkt.protocol;
        pkt._enriched = true;
        break;
      }
    }
  }

  /**
   * Get verdict for a domain.
   */
  getVerdict(domain) {
    return this.verdicts.get(domain) || null;
  }

  /**
   * Get all active sessions.
   */
  getActiveSessions() {
    const result = [];
    for (const [key, session] of this.sessions) {
      result.push({
        key,
        domain: session.domain,
        packetCount: session.packets.length,
        verdict: this.verdicts.get(session.domain) || null,
      });
    }
    return result;
  }

  /**
   * Flush a session to the backend.
   */
  async _flush(key) {
    const session = this.sessions.get(key);
    if (!session || session.packets.length < 3) return;

    session.flushed = true;

    const payload = {
      session_id: key,
      domain: session.domain,
      packet_count: session.packets.length,
      packets: session.packets.map(p => ({
        timestamp: p.timestamp || 0,
        duration: p.duration || 0,
        requestSize: p.requestSize || 0,
        responseSize: p.responseSize || 0,
        transferSize: p.transferSize || 0,
        encodedBodySize: p.encodedBodySize || 0,
        decodedBodySize: p.decodedBodySize || 0,
        headerSize: p.headerSize || 0,
        protocol: p.protocol || 'unknown',
        statusCode: p.statusCode || 0,
      })),
    };

    try {
      const result = await this.onFlush(payload);
      if (result && result.verdict) {
        this.verdicts.set(session.domain, result);
      }
    } catch (err) {
      console.warn('[TrafficGuardian] Flush failed:', err.message);
    }

    // Reset session packets but keep it alive
    session.packets = [];
    session.flushed = false;
  }

  /**
   * Check for idle sessions and flush them.
   */
  _checkIdle() {
    const now = Date.now();
    for (const [key, session] of this.sessions) {
      if (now - session.lastActivity > this.IDLE_TIMEOUT_MS) {
        if (session.packets.length >= 3) {
          this._flush(key);
        }
        this.sessions.delete(key);
      }
    }
  }
}
