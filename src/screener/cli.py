"""Command line entry points."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from screener import universe as universe_filters
from screener.config import ScreenConfig, load_screen_config
from screener.data.coingecko import CoinGeckoClient
from screener.data.history import HistoryStore
from screener.data.snapshots import SnapshotStore, WriteStatus
from screener.data.source import DataSource
from screener.paths import Layout, default_layout

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


@app.command()
def version() -> None:
    """Print the installed version."""
    from screener import __version__

    console.print(__version__)


if __name__ == "__main__":
    app()
