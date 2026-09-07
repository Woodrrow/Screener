"""Command line entry points."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from screener import factors
from screener import universe as universe_filters
from screener.backtest import SURVIVORSHIP_WARNING, UniverseBuilder, metadata_from_snapshot
from screener.benchmarks import buy_and_hold, equal_weight_basket
from screener.config import (
    PortfolioConfig,
    ScreenConfig,
    load_portfolio_config,
    load_screen_config,
)
from screener.data.coingecko import CoinGeckoClient
from screener.data.history import HistoryStore
from screener.data.snapshots import SnapshotStore, WriteStatus
from screener.data.source import DataSource
from screener.paths import Layout, default_layout
from screener.portfolio.costs import CostModel
from screener.portfolio.engine import prices_on, run_simulation, step_once
from screener.portfolio.schedule import Cadence, is_due
from screener.portfolio.simulator import RebalancePlan
from screener.portfolio.state import PortfolioState, StateStore, config_fingerprint
from screener.report import ReportInputs, write_report
from screener.screen import load_price_panel, run_screen, write_ranking_csv

app = typer.Typer(
    help="Crypto factor screener and $1,000 paper-trading simulator.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

ConfigOption = Annotated[
    Path, typer.Option("--config", "-c", help="Path to screen.yaml.", show_default=True)
]
DataDirOption = Annotated[
    Path, typer.Option("--data-dir", help="Root of the data store.", show_default=True)
]
ReportsDirOption = Annotated[
    Path, typer.Option("--reports-dir", help="Where reports and rankings are written.")
]


def _build_source(config: ScreenConfig, layout: Layout) -> DataSource:
    if config.source.provider == "coingecko":
        return CoinGeckoClient(
            cache_dir=layout.cache,
            rate_per_minute=config.source.rate_per_minute,
            max_retries=config.source.max_retries,
        )
    raise typer.BadParameter(f"unknown provider: {config.source.provider}")


@app.command()
def snapshot(
    config_path: ConfigOption = Path("config/screen.yaml"),
    data_dir: DataDirOption = Path("data"),
    top_n: Annotated[int | None, typer.Option("--top-n", help="Override universe.top_n.")] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing snapshot for today.")
    ] = False,
) -> None:
    """Fetch today's universe and persist it as an immutable dated snapshot."""
    config = load_screen_config(config_path)
    layout = default_layout(data_dir).ensure()
    store = SnapshotStore(layout.snapshots)
    today = datetime.now(UTC).date()

    if store.exists(today) and not force:
        result = store.write(store.read(today), today, source=config.source.provider, force=False)
        console.print(
            f"[yellow]Snapshot for {today} already exists[/] "
            f"({result.rows} rows at {result.path}). Pass --force to refetch."
        )
        raise typer.Exit(code=0)

    source = _build_source(config, layout)
    size = top_n if top_n is not None else config.universe.top_n
    console.print(f"Fetching top {size} coins from [bold]{source.name}[/]...")
    frame = source.fetch_universe(top_n=size, vs_currency=config.source.vs_currency)
    if frame.empty:
        console.print("[red]The provider returned no rows; refusing to write an empty snapshot.[/]")
        raise typer.Exit(code=1)

    result = store.write(frame, today, source=source.name, force=force)
    verb = "Overwrote" if result.status is WriteStatus.OVERWRITTEN else "Wrote"
    console.print(f"[green]{verb}[/] {result.rows} rows to {result.path}")

    classified = universe_filters.classify(
        store.read(today),
        min_market_cap_usd=config.universe.min_market_cap_usd,
        min_volume_24h_usd=config.universe.min_volume_24h_usd,
        min_turnover=config.universe.min_turnover,
        extra_stablecoin_symbols=frozenset(config.universe.extra_stablecoin_symbols),
        extra_excluded_coin_ids=frozenset(config.universe.extra_excluded_coin_ids),
    )
    kept = int((~classified["excluded"]).sum())
    console.print(
        f"Under the current filters this snapshot yields "
        f"[bold]{kept}[/] investable coins of {len(classified)}."
    )
    summary = universe_filters.exclusion_summary(classified)
    if not summary.empty:
        table = Table("reason", "coins", title="Exclusions (recomputed at read time)")
        for row in summary.itertuples(index=False):
            table.add_row(str(row.reason), str(row.coins))
        console.print(table)


@app.command()
def history(
    config_path: ConfigOption = Path("config/screen.yaml"),
    data_dir: DataDirOption = Path("data"),
    days: Annotated[int, typer.Option("--days", help="Trailing window to maintain.")] = 365,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Only the top N investable coins from the latest snapshot."),
    ] = None,
    coin: Annotated[
        list[str] | None, typer.Option("--coin", help="Explicit coin id; repeatable.")
    ] = None,
    force_full: Annotated[
        bool, typer.Option("--force-full", help="Refetch the whole window instead of appending.")
    ] = False,
) -> None:
    """Maintain the per-coin daily price history, appending incrementally."""
    config = load_screen_config(config_path)
    layout = default_layout(data_dir).ensure()
    store = HistoryStore(layout.history)
    today = datetime.now(UTC).date()

    if coin:
        coin_ids = sorted(set(coin))
    else:
        snapshots = SnapshotStore(layout.snapshots)
        latest = snapshots.latest_date()
        if latest is None:
            console.print("[red]No snapshots yet. Run `screener snapshot` first.[/]")
            raise typer.Exit(code=1)
        investable = universe_filters.apply_filters(
            snapshots.read(latest),
            min_market_cap_usd=config.universe.min_market_cap_usd,
            min_volume_24h_usd=config.universe.min_volume_24h_usd,
            min_turnover=config.universe.min_turnover,
            extra_stablecoin_symbols=frozenset(config.universe.extra_stablecoin_symbols),
            extra_excluded_coin_ids=frozenset(config.universe.extra_excluded_coin_ids),
        )
        ranked = investable.sort_values(
            ["market_cap", "coin_id"], ascending=[False, True], kind="mergesort"
        )
        if limit is not None:
            ranked = ranked.head(limit)
        coin_ids = sorted(ranked["coin_id"].astype(str).tolist())

    # BTC and ETH are always maintained: the report benchmarks against both, so
    # their absence would silently drop a benchmark rather than fail.
    coin_ids = sorted(set(coin_ids) | {"bitcoin", "ethereum"})

    source = _build_source(config, layout)
    console.print(f"Updating history for {len(coin_ids)} coins ({days}d window)...")
    added = 0
    skipped = 0
    for coin_id in coin_ids:
        update = store.update(source, coin_id, days=days, today=today, force_full=force_full)
        if update.skipped:
            skipped += 1
        else:
            added += update.rows_added
    console.print(
        f"[green]Done.[/] {added} new daily rows, {skipped} coins already current, "
        f"store at {layout.history}"
    )


@app.command("factors")
def factors_list() -> None:
    """List every registered factor, its direction and its missing-data policy."""
    table = Table("factor", "direction", "missing data", "description")
    for spec in factors.all_specs():
        table.add_row(spec.name, str(spec.direction), str(spec.na_policy), spec.description)
    console.print(table)


@app.command()
def screen(
    config_path: ConfigOption = Path("config/screen.yaml"),
    data_dir: DataDirOption = Path("data"),
    reports_dir: ReportsDirOption = Path("reports"),
    preset: Annotated[
        str | None, typer.Option("--preset", help="Named weight set from screen.yaml.")
    ] = None,
    as_of: Annotated[
        str | None,
        typer.Option("--as-of", help="Snapshot date (YYYY-MM-DD). Defaults to the newest."),
    ] = None,
    top: Annotated[int, typer.Option("--top", help="Rows to print.")] = 20,
    output: Annotated[
        Path | None, typer.Option("--output", help="Write the full ranking here as CSV.")
    ] = None,
) -> None:
    """Rank the investable universe of a snapshot by a preset's composite score."""
    config = load_screen_config(config_path)
    layout = default_layout(data_dir, reports_dir).ensure()
    store = SnapshotStore(layout.snapshots)

    if as_of is None:
        day = store.latest_date()
        if day is None:
            console.print("[red]No snapshots yet. Run `screener snapshot` first.[/]")
            raise typer.Exit(code=1)
    else:
        day = date.fromisoformat(as_of)
        if not store.exists(day):
            console.print(f"[red]No snapshot for {day}.[/] Available: {store.available_dates()}")
            raise typer.Exit(code=1)

    snapshot_frame = store.read(day)
    preset_name, weights = config.weights_for(preset)

    needs_history = any(factors.get(name).needs_history for name in weights)
    price_panel = None
    if needs_history:
        history = HistoryStore(layout.history)
        price_panel = load_price_panel(history, snapshot_frame["coin_id"].astype(str).tolist(), day)
        if price_panel.empty:
            console.print(
                "[yellow]No price history found.[/] History-backed factors will score every "
                "coin as the worst case, which is the conservative default but not informative. "
                "Run `screener history` first."
            )

    result = run_screen(
        snapshot_frame,
        as_of=day,
        weights=weights,
        preset=preset_name,
        universe_config=config.universe,
        price_panel=price_panel,
    )

    table = Table(
        "rank",
        "coin",
        "symbol",
        "score",
        *[f"pct {name}" for name in sorted(weights)],
        title=f"{preset_name} @ {day} ({len(result.ranking)} investable)",
    )
    for row in result.top(top).itertuples(index=False):
        table.add_row(
            str(row.rank),
            str(row.coin_id),
            str(row.symbol),
            f"{row.composite_score:.1f}",
            *[f"{getattr(row, f'pct_{name}'):.2f}" for name in sorted(weights)],
        )
    console.print(table)

    destination = (
        output
        if output is not None
        else (layout.reports_dir / f"screen-{day.isoformat()}-{preset_name}.csv")
    )
    write_ranking_csv(result, destination)
    console.print(f"[green]Wrote[/] {len(result.ranking)} ranked rows to {destination}")


PortfolioConfigOption = Annotated[
    Path, typer.Option("--portfolio", "-p", help="Path to portfolio.yaml.")
]


def _investable(frame: pd.DataFrame, config: ScreenConfig) -> pd.DataFrame:
    return universe_filters.apply_filters(
        frame,
        min_market_cap_usd=config.universe.min_market_cap_usd,
        min_volume_24h_usd=config.universe.min_volume_24h_usd,
        min_turnover=config.universe.min_turnover,
        extra_stablecoin_symbols=frozenset(config.universe.extra_stablecoin_symbols),
        extra_excluded_coin_ids=frozenset(config.universe.extra_excluded_coin_ids),
    )


def _build_benchmarks(
    *,
    price_panel: pd.DataFrame,
    universe_for: Callable[[date], pd.DataFrame | None],
    portfolio: PortfolioConfig,
    capital: float,
    start: date,
    end: date,
) -> dict[str, pd.Series[float]]:
    """BTC, ETH and the honest one: equal-weight top 8, same cadence, same costs."""
    costs = CostModel(fee_bps=portfolio.fee_bps, slippage_bps=portfolio.slippage_bps)
    curves: dict[str, pd.Series[float]] = {}
    for label, coin_id in (("Buy and hold BTC", "bitcoin"), ("Buy and hold ETH", "ethereum")):
        if coin_id in price_panel.columns:
            curves[label] = buy_and_hold(
                price_panel, coin_id, capital=capital, costs=costs, start=start, end=end
            )
    curves["Equal-weight top 8"] = equal_weight_basket(
        price_panel,
        universe_for=universe_for,
        config=portfolio,
        capital=capital,
        start=start,
        end=end,
    )
    return curves


def _rank_snapshot(
    frame: pd.DataFrame,
    *,
    as_of: date,
    config: ScreenConfig,
    preset: str | None,
    price_panel: pd.DataFrame | None,
) -> pd.DataFrame:
    preset_name, weights = config.weights_for(preset)
    return run_screen(
        frame,
        as_of=as_of,
        weights=weights,
        preset=preset_name,
        universe_config=config.universe,
        price_panel=price_panel,
    ).ranking


def _print_plan(plan: RebalancePlan, title: str) -> None:
    table = Table(
        "action", "coin", "rank", "current $", "target $", "delta $", "reason", title=title
    )
    for action in plan.actions:
        table.add_row(
            action.action,
            action.coin_id,
            "-" if action.rank is None else str(action.rank),
            f"{action.current_usd:,.2f}",
            f"{action.target_usd:,.2f}",
            f"{action.delta_usd:+,.2f}",
            action.reason,
        )
    console.print(table)


@app.command()
def backtest(
    config_path: ConfigOption = Path("config/screen.yaml"),
    portfolio_path: PortfolioConfigOption = Path("config/portfolio.yaml"),
    data_dir: DataDirOption = Path("data"),
    reports_dir: ReportsDirOption = Path("reports"),
    from_date: Annotated[str, typer.Option("--from", help="Start date (YYYY-MM-DD).")] = "",
    to_date: Annotated[str, typer.Option("--to", help="End date (YYYY-MM-DD).")] = "",
    preset: Annotated[str | None, typer.Option("--preset")] = None,
    acknowledge_survivorship: Annotated[
        bool,
        typer.Option(
            "--acknowledge-survivorship",
            help="Confirm you understand the backtest universe contains only survivors.",
        ),
    ] = False,
    report: Annotated[
        bool, typer.Option("--report", help="Also write the markdown report and charts.")
    ] = False,
) -> None:
    """Replay the rebalance loop over reconstructed history."""
    if not acknowledge_survivorship:
        console.print(f"[yellow]{SURVIVORSHIP_WARNING}[/]")
        console.print(
            "[red]Refusing to run.[/] Pass --acknowledge-survivorship once you have read that."
        )
        raise typer.Exit(code=2)

    config = load_screen_config(config_path)
    portfolio = load_portfolio_config(portfolio_path)
    layout = default_layout(data_dir, reports_dir).ensure()

    snapshots = SnapshotStore(layout.snapshots)
    latest = snapshots.latest_date()
    if latest is None:
        console.print("[red]No snapshots yet. Run `screener snapshot` first.[/]")
        raise typer.Exit(code=1)
    metadata = metadata_from_snapshot(snapshots.read(latest))

    history = HistoryStore(layout.history)
    coin_ids = history.available_coins()
    if not coin_ids:
        console.print("[red]No price history. Run `screener history` first.[/]")
        raise typer.Exit(code=1)

    builder = UniverseBuilder.from_history(history, coin_ids, metadata)
    covered = builder.dates
    if not covered:
        console.print("[red]The history store has no usable dates.[/]")
        raise typer.Exit(code=1)

    start = date.fromisoformat(from_date) if from_date else covered[0]
    end = date.fromisoformat(to_date) if to_date else covered[-1]
    if end <= start:
        console.print("[red]--to must be after --from.[/]")
        raise typer.Exit(code=1)

    def ranking_for(signal_date: date) -> pd.DataFrame | None:
        frame = builder.frame_for(signal_date)
        if frame.empty:
            return None
        panel = builder.prices.loc[[day for day in builder.prices.index if day <= signal_date]]
        return _rank_snapshot(
            frame, as_of=signal_date, config=config, preset=preset, price_panel=panel
        )

    console.print(f"[yellow]{SURVIVORSHIP_WARNING}[/]")
    console.print(f"Replaying {start} to {end} on {len(coin_ids)} coins...")
    result = run_simulation(
        price_panel=builder.prices,
        ranking_for=ranking_for,
        config=portfolio,
        start=start,
        end=end,
    )

    curve_path = reports_dir / f"backtest-equity-{start}-{end}.csv"
    trades_path = reports_dir / f"backtest-trades-{start}-{end}.csv"
    result.equity_curve.to_frame("equity_usd").to_csv(
        curve_path, float_format="%.10g", lineterminator="\n"
    )
    result.trades.to_csv(trades_path, index=False, float_format="%.10g", lineterminator="\n")

    fees = float(result.trades["fee_usd"].sum()) if not result.trades.empty else 0.0
    slippage = float(result.trades["slippage_usd"].sum()) if not result.trades.empty else 0.0
    total_return = result.final_equity / portfolio.initial_capital - 1.0
    console.print(
        f"[green]Done.[/] {len(result.plans)} rebalances, {len(result.trades)} trades. "
        f"Final equity ${result.final_equity:,.2f} ({total_return:+.1%}). "
        f"Costs ${fees + slippage:,.2f} (${fees:,.2f} fees, ${slippage:,.2f} slippage)."
    )
    if result.skipped_rebalances:
        console.print(
            f"[yellow]{len(result.skipped_rebalances)} rebalance dates had no rankable "
            "universe and were skipped.[/]"
        )
    console.print(f"Wrote {curve_path} and {trades_path}")

    if report:

        def universe_for(signal_date: date) -> pd.DataFrame | None:
            frame = builder.frame_for(signal_date)
            return None if frame.empty else _investable(frame, config)

        benchmarks = _build_benchmarks(
            price_panel=builder.prices,
            universe_for=universe_for,
            portfolio=portfolio,
            capital=portfolio.initial_capital,
            start=start,
            end=end,
        )
        path = write_report(
            ReportInputs(
                label=f"Backtest {start} to {end}",
                equity=result.equity_curve,
                trades=result.trades,
                benchmarks=benchmarks,
                starting_capital=portfolio.initial_capital,
                generated_on=datetime.now(UTC).date(),
                survivorship_warning=SURVIVORSHIP_WARNING,
                notes=[
                    "Supply data is not point-in-time on the free tier, so the dilution "
                    "factor is inert in this backtest and contributes nothing to the ranking.",
                    "All-time highs are running maxima over history to each rebalance date, "
                    "never the present-day figure.",
                ],
            ),
            reports_dir,
            stem=f"backtest-{start}-{end}",
        )
        console.print(f"[green]Report[/] {path}")


@app.command("report")
def report_command(
    config_path: ConfigOption = Path("config/screen.yaml"),
    portfolio_path: PortfolioConfigOption = Path("config/portfolio.yaml"),
    data_dir: DataDirOption = Path("data"),
    reports_dir: ReportsDirOption = Path("reports"),
) -> None:
    """Build the markdown report and charts for the forward paper record."""
    config = load_screen_config(config_path)
    portfolio = load_portfolio_config(portfolio_path)
    layout = default_layout(data_dir, reports_dir).ensure()

    store = StateStore(layout.portfolio)
    state = store.load()
    if state is None or not state.equity_curve:
        # Not an error: there is simply nothing to report yet. `papertrade`
        # treats "too few snapshots" the same way, and a scheduled pipeline
        # should not go red because the record has not started.
        console.print(
            "[yellow]No paper-trading record yet[/], so there is nothing to report. "
            "Run `screener papertrade` once two snapshots exist."
        )
        raise typer.Exit(code=0)

    equity = pd.Series(
        {day: state.equity_curve[day] for day in sorted(state.equity_curve)}, dtype="float64"
    )
    equity.index.name = "date"
    start, end = equity.index[0], equity.index[-1]

    snapshots = SnapshotStore(layout.snapshots)
    history = HistoryStore(layout.history)
    price_panel = history.price_panel(history.available_coins())

    def universe_for(signal_date: date) -> pd.DataFrame | None:
        found = snapshots.read_on_or_before(signal_date)
        if found is None:
            return None
        return _investable(found[1], config)

    benchmarks = _build_benchmarks(
        price_panel=price_panel,
        universe_for=universe_for,
        portfolio=portfolio,
        capital=state.initial_capital,
        start=start,
        end=end,
    )

    path = write_report(
        ReportInputs(
            label="Paper trading record",
            equity=equity,
            trades=store.read_trades(),
            benchmarks=benchmarks,
            starting_capital=state.initial_capital,
            generated_on=datetime.now(UTC).date(),
            notes=[
                "This is a forward record: every trade was priced from a snapshot taken "
                "before the fill, so none of it is fitted to data it could not have seen.",
                "CoinGecko volume figures include wash trading on some venues, which "
                "flatters the liquidity factor for coins listed on the worst offenders.",
            ],
        ),
        reports_dir,
        stem="papertrade-report",
    )
    console.print(f"[green]Report[/] {path}")


@app.command()
def papertrade(
    config_path: ConfigOption = Path("config/screen.yaml"),
    portfolio_path: PortfolioConfigOption = Path("config/portfolio.yaml"),
    data_dir: DataDirOption = Path("data"),
    preset: Annotated[str | None, typer.Option("--preset")] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Compute and print the trades without touching state."),
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Rebalance even if one is not due yet.")
    ] = False,
) -> None:
    """Run one rebalance step against the newest snapshot and persist the result."""
    config = load_screen_config(config_path)
    portfolio = load_portfolio_config(portfolio_path)
    layout = default_layout(data_dir).ensure()

    snapshots = SnapshotStore(layout.snapshots)
    available = snapshots.available_dates()
    if len(available) < 2:
        console.print(
            "[yellow]Need at least two snapshots before trading.[/] A signal from a snapshot "
            "dated t may only be filled at a price dated t+1 or later, so the first run after "
            "the first snapshot is always a no-op."
        )
        raise typer.Exit(code=0)

    execution_date, execution_snapshot = available[-1], snapshots.read(available[-1])
    signal_date, signal_snapshot = available[-2], snapshots.read(available[-2])

    store = StateStore(layout.portfolio)
    state = store.load() or PortfolioState.fresh(portfolio)
    fingerprint = config_fingerprint(portfolio)
    if state.config_fingerprint and state.config_fingerprint != fingerprint:
        console.print(
            "[yellow]Portfolio config has changed since this track record started.[/] "
            "The curve before and after this point is not one strategy."
        )
        state.config_fingerprint = fingerprint

    if not force and not is_due(execution_date, state.last_rebalance, Cadence(portfolio.rebalance)):
        console.print(
            f"No rebalance due: last was {state.last_rebalance}, cadence is "
            f"{portfolio.rebalance}. Pass --force to override."
        )
        raise typer.Exit(code=0)

    history = HistoryStore(layout.history)
    panel = load_price_panel(history, signal_snapshot["coin_id"].astype(str).tolist(), signal_date)
    ranking = _rank_snapshot(
        signal_snapshot,
        as_of=signal_date,
        config=config,
        preset=preset,
        price_panel=panel if not panel.empty else None,
    )

    prices = {
        str(coin_id): float(price)
        for coin_id, price in zip(
            execution_snapshot["coin_id"].astype(str),
            execution_snapshot["current_price"].astype("float64"),
            strict=True,
        )
        if price > 0
    }

    costs = CostModel(fee_bps=portfolio.fee_bps, slippage_bps=portfolio.slippage_bps)
    ledger = state.to_ledger(costs)

    # Mark the book every day since the last run before trading. Holdings were
    # constant over that window, so this is the true daily curve, not a sample.
    full_panel = history.price_panel(sorted(set(ledger.positions) | set(prices)))
    if not full_panel.empty:
        known = sorted(state.equity_curve)
        cursor = (known[-1] if known else signal_date) + timedelta(days=1)
        while cursor < execution_date:
            day_prices = prices_on(full_panel, cursor)
            if day_prices:
                ledger.observe_prices(day_prices)
                state.equity_curve[cursor] = ledger.equity(day_prices)
            cursor += timedelta(days=1)

    plan, executed = step_once(
        ledger,
        config=portfolio,
        ranking=ranking,
        prices=prices,
        signal_date=signal_date,
        execution_date=execution_date,
        costs=costs,
        dry_run=dry_run,
    )

    _print_plan(plan, f"signal {signal_date} -> fill {execution_date}")

    if dry_run:
        console.print("[yellow]Dry run: state and trade log were not written.[/]")
        raise typer.Exit(code=0)

    state.absorb(ledger)
    state.last_rebalance = execution_date
    state.last_signal_date = signal_date
    state.equity_curve[execution_date] = ledger.equity(prices)
    store.append_trades(executed)
    store.save(state)

    console.print(
        f"[green]Executed[/] {len(executed)} trades. Equity "
        f"${state.equity_curve[execution_date]:,.2f}, cash ${state.cash:,.2f}, "
        f"{len(state.positions)} positions."
    )
    console.print(f"State at {store.state_path}, trade log at {store.trades_path}")


@app.command()
def version() -> None:
    """Print the installed version."""
    from screener import __version__

    console.print(__version__)


if __name__ == "__main__":
    app()
