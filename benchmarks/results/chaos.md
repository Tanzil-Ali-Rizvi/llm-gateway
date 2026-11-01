# Reliability and cache benchmark

Simulated providers (primary 300±100 ms, backup 450±150 ms), 25 concurrent clients. Reproduce: `python benchmarks/chaos.py`.

## Flaky primary: 20% of calls fail with HTTP 503

| Setup | Success | Served by fallback | Upstream calls | p50 ms | p95 ms |
|---|---|---|---|---|---|
| direct to primary (no gateway) | **81.8%** | 0.0% | 1000 | 300.9 | 391.8 |
| gateway: retries only | **96.4%** | 0.0% | 1193 | 325.9 | 698.5 |
| gateway: retries + fallback | **100.0%** | 4.2% | 1223 | 322.0 | 787.5 |

## Primary completely down

| Setup | Success | Served by fallback | Upstream calls | p50 ms | p95 ms |
|---|---|---|---|---|---|
| fallback, no circuit breaker | **100.0%** | 100.0% | 3000 | 1101.6 | 1293.4 |
| fallback + circuit breaker | **100.0%** | 100.0% | 1029 | 456.3 | 592.5 |

## Support-bot FAQ workload: 2,000 requests over 300 distinct questions

| Setup | Spend (USD) | Cache hit rate | Upstream calls |
|---|---|---|---|
| no cache | $0.03216 | 0.0% | 2000 |
| exact-match cache | $0.00413 | 87.5% | 251 |
