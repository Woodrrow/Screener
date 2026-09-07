# crypto-factor-screener

Screens the crypto universe on quantitative factors and runs a simulated $1,000
portfolio against those signals.

**This is a research and paper-trading tool.** It never places an order, and it
never needs an exchange API key. The only credential it accepts is an optional
read-only CoinGecko demo key. If you point it at a key with trade permissions,
that is a mistake you had to make deliberately.

---

## What it actually does

1. **Snapshots** the top N coins by market cap into a dated, immutable Parquet
   file. The screener reads snapshots and never the live API, which is what
   makes a forward result honest rather than a story about the present.
2. **Ranks** the investable subset by a weighted blend of cross-sectional factor
   percentiles, writing every component percentile alongside the composite so
   any ranking can be taken apart afterwards.
3. **Simulates** a $1,000 book against that ranking, with fees and slippage
   always charged, and persists the state so re-running weekly accumulates a
   real forward track record.
4. **Reports** the result against three benchmarks, with the caveats at the top
   rather than in a footnote.

### Four rules the code enforces, not just documents

| Rule | Where it lives |
| --- | --- |
| A signal from data at `t` may only fill at `t+1` or later | `assert_execution_after_signal`, `FactorContext.__post_init__` — both raise `LookaheadError` |
| Costs are never zero | `CostModel.__post_init__` raises `ZeroCostError` |
| Cash plus positions always equals equity | `Ledger.reconcile` and `_assert_cost_only_leak` after every trade |
| Same snapshots + same config = same output | Explicit sorts everywhere, seeded RNG, no dict-ordering dependence |

Each has a test that fails if the guard is removed.

---

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/woodrrow/crypto-factor-screener
cd crypto-factor-screener
uv sync

# Optional: a free CoinGecko demo key raises the rate limit.
export COINGECKO_API_KEY=...

uv run screener snapshot                 # today's universe -> data/snapshots/YYYY-MM-DD.parquet
uv run screener history --days 365       # daily prices -> data/history/{coin_id}.parquet
uv run screener screen --preset balanced # rank it
```

Without `uv`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install pytest pytest-cov ruff mypy pandas-stubs types-requests types-PyYAML
```

At this point you have one snapshot, so there is nothing to trade yet — a signal
dated today cannot fill at today's price. Take a second snapshot (next week, or
`--force` on another day) and then:

```bash
uv run screener papertrade --dry-run     # what it would do
uv run screener papertrade               # do it, and persist state
uv run screener report                   # markdown + PNGs in reports/
```

### Commands

| Command | Purpose |
| --- | --- |
| `snapshot` | Fetch the universe into a dated, write-once snapshot |
| `history` | Maintain per-coin daily prices, appending incrementally |
| `factors` | List every registered factor, its direction and missing-data policy |
| `screen` | Rank a snapshot's investable universe by a preset |
| `backtest` | Replay the rebalance loop over reconstructed history |
| `papertrade` | One rebalance step against the newest snapshot; persists state |
| `report` | Markdown report and charts for the forward record |
| `dashboard` | One self-contained HTML page with everything on it |

---

## The dashboard

```bash
uv run screener dashboard      # -> reports/dashboard.html
```

One file, no server, no CDN, no network requests at all - open it straight from
disk. It carries the ranking as a sortable, filterable table with every factor
percentile heat-mapped beside the score, the equity curve against its three
benchmarks, current holdings, the recent trade log with the rank that triggered
each fill, what the filters excluded and why, and the active factor weights.

The heat ramp encodes magnitude as distance from the background in both themes -
pale-on-light, bright-on-dark - and every cell also prints its number, so the
colour is a reading aid rather than the only channel. Dark mode follows the
system setting, with a toggle that overrides it.

The weekly job regenerates it and commits it, so `reports/dashboard.html` is
always the current view. GitHub renders it as source rather than as a page; open
it locally, or enable GitHub Pages if you want a URL.

---

## Worked example

Say it is Monday. You have snapshots for the 3rd and the 10th, and a year of
price history.

```console
$ uv run screener screen --preset balanced --top 5
                      balanced @ 2025-03-10 (61 investable)
 rank  coin       sym    score   dil    liq    m30    m200   size   trend  vol
 1     bitcoin    BTC    71.4    0.98   0.42   0.71   0.83   0.02   0.95   0.97
 2     solana     SOL    64.2    0.55   0.78   0.93   0.88   0.31   0.61   0.44
 3     chainlink  LINK   58.9    0.62   0.51   0.44   0.52   0.66   0.38   0.71
 ...
Wrote 61 ranked rows to reports/screen-2025-03-10-balanced.csv
```

Every percentile is in that CSV next to the score, so "why is SOL second" is a
question you can answer from the file rather than from memory.

```console
$ uv run screener papertrade
              signal 2025-03-03 -> fill 2025-03-10
 action  coin       rank  current $  target $  delta $   reason
 buy     bitcoin    1          0.00    122.50  +122.50   entry_top_rank
 buy     solana     2          0.00    122.50  +122.50   entry_top_rank
 ...
Executed 8 trades. Equity $996.09, cash $20.00, 8 positions.
```

Note the header: the signal is the **3rd**, the fill is the **10th**. The
ranking that chose these coins could not see the prices they were bought at.

A week later the same command finds most of the book still inside the exit band
and trades only the difference:

```console
$ uv run screener papertrade
              signal 2025-03-10 -> fill 2025-03-17
 action  coin       rank  current $  target $  delta $   reason
 sell    dogecoin   17       118.22      0.00  -118.22   exit_rank_exceeded
 buy     aave        6         0.00    124.10  +124.10   entry_top_rank
 hold    bitcoin     1       131.40    131.40    +0.00   hold_within_hysteresis
 ...
```

`dogecoin` was sold because it fell past rank 15, not because it slipped out of
the top 8. That gap is the hysteresis band, and it is the single most valuable
setting in the config at this account size.

---

## Configuration

### `config/screen.yaml`

| Key | Default | Meaning |
| --- | --- | --- |
| `source.provider` | `coingecko` | The only provider shipped; the `DataSource` protocol takes others |
| `source.rate_per_minute` | `10.0` | Token-bucket budget. The conservative end of the free tier |
| `source.max_retries` | `5` | Attempts before the run fails loudly |
| `universe.top_n` | `250` | Coins pulled per snapshot |
| `universe.min_market_cap_usd` | `50000000` | Market-cap floor |
| `universe.min_volume_24h_usd` | `5000000` | 24h volume floor |
| `universe.min_turnover` | `0.01` | Volume ÷ market cap floor |
| `universe.extra_stablecoin_symbols` | `[]` | Additions to the built-in stablecoin list |
| `universe.extra_excluded_coin_ids` | `[]` | Manual exclusions by coin id |
| `default_preset` | `balanced` | Used when `--preset` is omitted |
| `presets` | 3 shipped | Named weight sets, validated on load |

Filters are applied when a snapshot is **read**, not when it is written. Change a
floor and every stored snapshot re-screens under the new rule without a single
extra API call.

### Factors

| Factor | Direction | Missing data | Notes |
| --- | --- | --- | --- |
| `momentum_7d` / `momentum_30d` / `momentum_200d` | higher better | worst | Separate horizons, not one blend |
| `liquidity` | higher better | worst | 24h volume ÷ market cap |
| `dilution` | higher better | worst | Circulating ÷ max supply; **uncapped emission scores worst, never dropped** |
| `drawdown_depth` | higher better | worst | Distance below the ATH; for value presets |
| `trend_health` | higher better | worst | Proximity to the ATH; for trend presets |
| `size_tilt` | lower better | worst | Ranked on market cap with the direction inverted |
| `volatility` | lower better | worst | 60d annualised realised vol from the history store |

Two things to know about the weights:

- **Every weight is positive and each preset sums to 1.0.** A "penalty" factor
  carries its sign in its declared *direction*, not in a negative weight. This
  is why `volatility` has a positive weight and still lowers the score of
  volatile coins. A negative weight in config is rejected with a message saying
  exactly this.
- **Scoring is by percentile, not z-score.** Crypto cross-sections have tails fat
  enough that one 400% mover would dominate a z-blend, and the composite would
  become "whatever pumped hardest this week" wearing a factor model's clothes. A
  percentile caps any one coin's contribution to any one factor at 1.0.

Missing values are ranked separately and then assigned the worst score, so a gap
never shifts anybody else's percentile — and never earns a coin a free pass.

Adding a factor is a function plus a decorator; the screener does not change.

```python
@register("my_factor", direction=Direction.HIGHER_IS_BETTER, description="...")
def my_factor(context: FactorContext) -> pd.Series[float]:
    return context.column("total_volume") / context.column("market_cap")
```

### `config/portfolio.yaml`

| Key | Default | Meaning |
| --- | --- | --- |
| `initial_capital` | `1000.0` | Starting cash |
| `max_positions` | `8` | Hard cap on concurrent holdings |
| `min_position_usd` | `50.0` | Below this the slot stays in cash; fees would eat it |
| `sizing` | `equal_weight` | Also `score_weighted`, `inverse_vol` |
| `rebalance` | `weekly` | Also `monthly`, `daily` |
| `max_weight_per_position` | `0.20` | Excess is redistributed, not forced |
| `cash_buffer` | `0.02` | Fraction held back from the target book |
| `entry_rank_threshold` | `8` | Buy at or above this rank |
| `exit_rank_threshold` | `15` | Sell only once it falls past this one |
| `rebalance_band` | `0.0` | No-trade band on existing positions, as a fraction of target |
| `fee_bps` | `10` | Per side. Cannot be zero |
| `slippage_bps` | `30` | Per side. Cannot be zero |
| `no_shorts` / `no_leverage` | `true` | Cannot be disabled; there is no machinery for either |

**The entry/exit gap is the point.** Buying and selling at the same cutoff churns
the book every time a coin oscillates around the boundary, and at $1,000 with a
40bps round trip that churn is the dominant cost — larger, over a year, than most
plausible factor edges. Widen the gap to trade less. `report` prints turnover so
the effect is visible rather than theoretical.

`rebalance_band` defaults to `0.0`, meaning a **full** rebalance to target every
period. That is the expensive choice, taken on purpose: it makes the simulated
result look worse than a tuned implementation would. Raising it is the obvious
first optimisation, and the turnover figure tells you what it bought you.

---

## The scheduled job

`.github/workflows/weekly-screen.yml` runs Mondays at 08:00 UTC (and on
`workflow_dispatch`): snapshot → history → papertrade → report → commit the
result back to the repo.

Every data step runs before anything is committed. If the CoinGecko call fails,
the job fails there and the repo keeps its last good state rather than gaining a
half-written snapshot. Set `COINGECKO_API_KEY` in repository secrets if you have
a demo key.

`data/snapshots/` and `data/portfolio/` are deliberately **not** gitignored. The
dated snapshots are the point-in-time record and `state.json` is the forward
track record; both belong in version control where they can be diffed. The HTTP
cache and the price history are ignored — they are regenerable and large.

---

## Limitations

Read this part.

**Survivorship bias makes the backtest close to worthless as evidence.** The
history store contains the coins that were in a recent snapshot — that is, the
ones that survived to today. Anything that was delisted, abandoned or collapsed
before then is simply absent, so a backtest is scored against a universe that
already knows which projects made it. This inflates returns in a way no amount
of careful cost modelling repairs. `screener backtest` refuses to run without
`--acknowledge-survivorship` and stamps the warning into every report it
produces. Treat it as a test that the machinery works, not as a result.

**The backtest has no point-in-time supply data.** CoinGecko's free tier serves
today's circulating and max supply, and using those figures for a 2024 rebalance
would understate past dilution — flattering exactly the coins that were emitting
hardest. Both fields are left null in a reconstructed universe, which means the
`dilution` factor scores every coin as the worst case and contributes nothing to
the ranking there. The factor is live in forward paper trading and inert in
backtest, and the report says so.

**CoinGecko volume includes wash trading on some venues.** The `liquidity`
factor and the turnover floor are both computed from reported 24h volume, and
some exchanges report volume that did not happen. This systematically flatters
coins listed primarily on the worst offenders. There is no free fix; the
mitigation is that liquidity is one factor among several rather than a gate.

**Slippage is a flat assumption, not a modelled order book.** 30bps per side is
applied identically to BTC and to a $60m-cap altcoin. It is punitive for the
former and generous for the latter, so the simulation flatters small caps — which
is precisely where the `size_tilt` factor is pointing. A real $125 order in a
thin book can cost several times this. Treat small-cap results with more
suspicion than large-cap ones.

**A forward record under about six months cannot distinguish edge from noise.**
At weekly rebalances that is roughly 26 observations. With crypto's volatility,
the confidence interval around any Sharpe computed on that sample comfortably
contains zero. The report prints a banner when the sample is under 30 daily
observations, but the banner disappearing at 30 days does not mean the result
became meaningful — it means the loudest warning stopped being the most useful
thing on the page.

**Other things worth knowing.** The stablecoin heuristic (within 3% of $1.00 and
under 2% absolute 30-day return) will miss a depegging stablecoin, which is
arguably correct — a depegged stable *is* making a price move — but means one can
briefly enter the universe. The derivative exclusion is list-plus-pattern based
and will miss a wrapped asset whose name and symbol both disguise it. Daily bars
are UTC closes resampled from CoinGecko's hourly series under 90 days, so a
"close" is the last print in the UTC day and not an exchange settlement.

---

## Development

```bash
uv run pytest              # 235 tests, no network — the suite blocks sockets
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

The test suite fails any test that opens a socket, so a change that reaches for
the live API is a test failure rather than a slow, rate-limited, flaky run. All
fixtures are committed under `tests/fixtures/`.
