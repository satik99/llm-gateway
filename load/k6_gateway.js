// Load test with a simulated provider outage in the middle.
//
//   k6 run load/k6_gateway.js
//   k6 run -e RATE=200 -e SECONDS=120 load/k6_gateway.js
//
// Timeline (SECONDS=90): 0-30s healthy | 30-60s primary upstream fails 100% | 60-90s healthy
// Use -e OUTAGE=0 for clean baseline / max-throughput runs (no failures injected).
import http from 'k6/http';
import { check } from 'k6';
import { Counter } from 'k6/metrics';

const GATEWAY = __ENV.GATEWAY_URL || 'http://localhost:8000';
const KEY = __ENV.GATEWAY_KEY || 'gw-load-key';
const CONTROL = __ENV.MOCK_CONTROL_URL || 'http://localhost:9000/control';
const RATE = parseInt(__ENV.RATE || '100');
const SECONDS = parseInt(__ENV.SECONDS || '90');
const OUTAGE = (__ENV.OUTAGE || '1') === '1';   // -e OUTAGE=0 for clean throughput/baseline runs
const CACHE = (__ENV.CACHE || '0') === '1';       // -e CACHE=1 sends temperature 0 => cacheable
const DISTINCT = parseInt(__ENV.DISTINCT || '50');  // number of different questions in the workload
const SAVE_JSON = (__ENV.SAVE_JSON || '0') === '1';  // -e SAVE_JSON=1 writes load/results/summary.json

// dynamic counters must be created in init context
const servedBy = { openai: new Counter('served_by_openai'), ollama: new Counter('served_by_ollama') };
const other = new Counter('served_by_other');
const cacheKinds = { hit: new Counter('cache_hit'), coalesced: new Counter('cache_coalesced'),
                     miss: new Counter('cache_miss') };

export const options = {
  summaryTrendStats: ['avg', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
  scenarios: {
    load: {
      executor: 'constant-arrival-rate',
      rate: RATE, timeUnit: '1s', duration: `${SECONDS}s`,
      preAllocatedVUs: 100, maxVUs: 1000,
    },
    ...(OUTAGE ? {
      outage_start: {
        executor: 'per-vu-iterations', vus: 1, iterations: 1,
        startTime: `${Math.floor(SECONDS / 3)}s`, exec: 'breakUpstream',
      },
      outage_end: {
        executor: 'per-vu-iterations', vus: 1, iterations: 1,
        startTime: `${Math.floor((2 * SECONDS) / 3)}s`, exec: 'fixUpstream',
      },
    } : {}),
  },
  thresholds: {
    'http_req_failed{scenario:load}': ['rate<0.01'],       // <1% errors, even during the outage
    'http_req_duration{scenario:load}': ['p(95)<3000'],
  },
};

const headers = { 'Content-Type': 'application/json', Authorization: `Bearer ${KEY}` };

export default function () {
  // Vary the prompt so this test is honest for a future cache (day 8): 50 distinct prompts.
  const q = `Question number ${Math.floor(Math.random() * DISTINCT)}: what is your refund policy?`;
  const res = http.post(
    `${GATEWAY}/v1/chat/completions`,
    JSON.stringify({ model: 'gpt-4o-mini', messages: [{ role: 'user', content: q }],
                    ...(CACHE ? { temperature: 0 } : {}) }),
    { headers },
  );
  check(res, { 'status 200': (r) => r.status === 200 });
  const p = res.headers['X-Gateway-Provider'];
  (servedBy[p] || other).add(1);
  const ck = res.headers['X-Gateway-Cache'];
  if (ck && cacheKinds[ck]) cacheKinds[ck].add(1);
}

const setMock = (body) =>
  http.post(CONTROL, JSON.stringify(body), { headers: { 'Content-Type': 'application/json' } });
export function breakUpstream() { setMock({ error_rate: 1.0 }); }
export function fixUpstream() { setMock({ error_rate: 0.0 }); }

export function handleSummary(data) {
  const m = (k, s) => (data.metrics[k] && data.metrics[k].values[s]) || 0;
  const d = 'http_req_duration{scenario:load}';
  const total = m('http_reqs', 'count');
  const out = [
    '',
    '================ LOAD TEST SUMMARY ================',
    `target rate        : ${RATE} req/s for ${SECONDS}s (${OUTAGE ? 'outage in the middle third' : 'no outage'})`,
    `requests sent      : ${total}  (achieved ${m('http_reqs', 'rate').toFixed(1)} req/s)`,
    `latency p50 / p95 / p99 : ${m(d, 'med').toFixed(0)} / ${m(d, 'p(95)').toFixed(0)} / ${m(d, 'p(99)').toFixed(0)} ms`,
    `error rate         : ${(m('http_req_failed{scenario:load}', 'rate') * 100).toFixed(2)} %`,
    `served by openai   : ${m('served_by_openai', 'count')}`,
    `served by ollama   : ${m('served_by_ollama', 'count')}   (fallback during outage)`,
    `cache hit/coalesced/miss : ${m('cache_hit', 'count')} / ${m('cache_coalesced', 'count')} / ${m('cache_miss', 'count')}`,
    `served by other    : ${m('served_by_other', 'count')}   (errors / unknown)`,
    '===================================================',
    '',
  ].join('\n');
  return SAVE_JSON ? { stdout: out, 'load/results/summary.json': JSON.stringify(data, null, 2) } : { stdout: out };
}
