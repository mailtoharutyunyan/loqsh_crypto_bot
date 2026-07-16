# Quantum Pro — Telegram signal bot

Python port of the TradingView strategy's **5-bucket confluence engine** (Trend · Momentum ·
Structure · Flow · Location) with A/B/C tiers. Runs on GitHub Actions cron, evaluates the
**last closed candle** per symbol, and pushes BUY/SELL alerts to Telegram.

> ⚠️ **Not identical to the Pine version.** Different data feed, EMA seeding and rounding mean a
> few marginal signals will diverge. This is a faithful port of the *logic*, not a mirror of the
> chart. **Validate against TradingView and paper-trade before risking capital.** It sends entry +
> SL/TP levels only — it does not track open positions, TP fills, or P&L.

## Files
```
quantum-bot/                 ← make this the repo root
├─ quantum_signals.py        # data fetch + indicators + buckets + signal + telegram
├─ config.json               # symbols / timeframe / thresholds
├─ requirements.txt
├─ state.json                # auto-created; de-dup memory (one alert per bar). Commit it.
└─ .github/workflows/signals.yml
```

## Setup (5 minutes)

1. **Create a Telegram bot** — message [@BotFather](https://t.me/BotFather) → `/newbot` → copy the **token**.
2. **Get your chat id** — message your new bot once, then open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` and read `result[].message.chat.id`
   (for a group, add the bot to the group and use the group's negative id).
3. **Push this folder to a GitHub repo** (public repo = free unlimited Actions minutes; private
   repo uses your 2,000 free min/month — a 15-min cron is well within it).
4. **Add repo secrets** — Settings → Secrets and variables → Actions → *New repository secret*:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
5. **Edit `config.json`** — set your `symbols` and `timeframe`.
6. **Test now** — Actions tab → *quantum-signals* → **Run workflow**. Check the logs and your Telegram.

## Timeframe ↔ cron
Match the cron in `signals.yml` to your `timeframe` so you check roughly once per new bar:

| timeframe | cron |
|---|---|
| `15m` | `*/15 * * * *` |
| `1h`  | `0 * * * *` |
| `4h`  | `0 */4 * * *` |
| `1d`  | `5 0 * * *` |

⚠ **GitHub cron is best-effort** — runs are commonly 5–15 min late and occasionally skipped. The
bot only ever acts on the *last closed* candle, so a late run still alerts the right bar (just
late). **Do not use GitHub Actions for timeframes ≤ 5m** — the drift is larger than the bar.

## Run locally (to test / backfill parity)
```bash
pip install -r requirements.txt
export TELEGRAM_BOT_TOKEN=...   # omit both to just print instead of send
export TELEGRAM_CHAT_ID=...
python quantum_signals.py
```

## How de-dup works
`state.json` stores, per `SYMBOL|TF`, the last alerted bar and direction. The bot mirrors the Pine
strategy: **one alert per bar**, and (with `allowRepeatDirection:false`) it won't fire the same
direction twice in a row until the opposite side fires, plus a `cooldownBars` gap. The workflow
commits `state.json` back to the repo after each run so memory survives between runs.

## Config keys (defaults mirror the Pine strategy)
| key | default | meaning |
|---|---|---|
| `symbols` | `["BTCUSDT"]` | Binance symbols |
| `timeframe` | `"15m"` | signal timeframe |
| `htf` | `["1h","4h","1d"]` | higher-TF trend confirmation |
| `tradeTier` | `"A + B"` | `A only` \| `A + B` \| `A + B + C` |
| `minBuckets` | `3` | min confluence buckets to trigger |
| `usePullback` | `false` | require pullback-to-EMA21 entry |
| `minVolatilityPct` | `0.0` | low-vol floor (skip dead chop) |
| `allowRepeatDirection` | `false` | allow consecutive same-side signals |
| `cooldownBars` | `4` | min bars between signals |

## Known limitations
- Signal-only; no position/TP tracking across runs.
- CVD is a candle approximation (no tick data), so the Flow bucket is directional, not exact.
- Correlation & killzone filters from the Pine version are off by default (crypto 24/7); add if needed.
- Data host is `data-api.binance.vision` (public, unauthenticated, US-reachable). Swap in
  `config.json` → `data_host` if you prefer another Binance endpoint.
