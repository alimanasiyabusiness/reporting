import csv
import glob
import math
import os
import statistics
from datetime import date, datetime, timedelta

import yfinance as yf


# ============================================================
# CONFIGURATION
# ============================================================

STARTING_CAPITAL = 100_000

CSV_PATTERN = "export-*.csv"
BACKTEST_CSV = "AM-Live-Test.csv"
OUTPUT_HTML = "index.html"

# ------------------------------------------------------------
# Sharpe configuration
# ------------------------------------------------------------
# Trading-day Sharpe:
#   - 252 trading days/year
#   - 3.5% annual risk-free rate
#   - only weekdays included
#
# Daily RF is calculated as annual RF / 252.
# ------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.035

# ------------------------------------------------------------
# SGOV simulation
# ------------------------------------------------------------
# Percentage of total portfolio notionally held in SGOV at the
# beginning of each trading day. The SGOV position is rebalanced
# to this target weight daily as portfolio equity changes.
SGOV_TICKER = "SGOV"
SGOV_ALLOCATION = 0.85

CVAR_LEVEL = 0.95

EXCLUDED_STRATEGIES = {
    "$2 Put Credit Spread"
}


# ============================================================
# FILE LOADING
# ============================================================

def newest_csv():
    files = glob.glob(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            CSV_PATTERN
        )
    )

    if not files:
        raise SystemExit(
            f"No CSV matching {CSV_PATTERN}"
        )

    return max(files, key=os.path.getmtime)


def load_trades(path):
    trades = []
    excluded = 0

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                close_date = datetime.strptime(
                    row["CloseDate"],
                    "%Y-%m-%d"
                ).date()

                open_date = datetime.strptime(
                    row["OpenDate"],
                    "%Y-%m-%d"
                ).date()

                pnl = float(row["ProfitLoss"])

                bp = (
                    float(row["BuyingPower"])
                    if row.get("BuyingPower")
                    else 0.0
                )

                strategy = row.get("Strategy", "")

                if strategy in EXCLUDED_STRATEGIES:
                    excluded += 1
                    continue

                trades.append(
                    {
                        "close": close_date,
                        "open": open_date,
                        "pnl": pnl,
                        "bp": bp,
                        "strategy": strategy,
                    }
                )

            except (ValueError, KeyError):
                continue

    if not trades:
        raise SystemExit(
            f"No parseable trades in {path}"
        )

    return trades, excluded


def load_trades_am(path):
    trades = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        raw_names = reader.fieldnames or []

        keys = [
            k.lstrip("\ufeff").strip().strip('"')
            for k in raw_names
        ]

        for raw in reader:
            row = {
                keys[i]: raw[raw_names[i]]
                for i in range(len(raw_names))
            }

            try:
                close_date = datetime.strptime(
                    row["Date Closed"],
                    "%Y-%m-%d"
                ).date()

                open_date = datetime.strptime(
                    row["Date Opened"],
                    "%Y-%m-%d"
                ).date()

                pnl = float(row["P/L"])

                strategy = row.get("Strategy", "")

                trades.append(
                    {
                        "close": close_date,
                        "open": open_date,
                        "pnl": pnl,
                        "strategy": strategy
                    }
                )

            except (ValueError, KeyError):
                continue

    if not trades:
        raise SystemExit(
            f"No parseable trades in {path}"
        )

    return trades


# ============================================================
# DATE / DAILY DATA
# ============================================================

def load_sgov_returns(start_date, end_date):
    """
    Download daily SGOV total-return series from Yahoo Finance via
    yfinance.

    auto_adjust=True makes the returned Close series dividend-adjusted,
    which is the appropriate series for a reinvested-total-return
    simulation.

    end_date is inclusive to the caller, but yfinance's end parameter
    is exclusive, so we request one extra calendar day.
    """
    end_exclusive = end_date + timedelta(days=1)

    hist = yf.Ticker(SGOV_TICKER).history(
        start=start_date.isoformat(),
        end=end_exclusive.isoformat(),
        interval="1d",
        auto_adjust=True,
        actions=False,
        raise_errors=True,
    )

    if hist.empty:
        raise SystemExit(
            f"No historical data returned for {SGOV_TICKER}. "
            "Check your internet connection or Yahoo Finance availability."
        )

    returns = {}
    previous_close = None

    for ts, row in hist.iterrows():
        d = ts.date()

        if d < start_date or d > end_date:
            continue

        close = float(row["Close"])

        if previous_close is not None and previous_close > 0:
            returns[d] = close / previous_close - 1.0
        else:
            returns[d] = 0.0

        previous_close = close

    if not returns:
        raise SystemExit(
            f"No usable daily returns returned for {SGOV_TICKER}."
        )

    return returns


def daily_pnl(trades, sgov_returns):
    """
    Aggregate strategy P&L by closing date and align the portfolio
    simulation to actual SGOV trading dates. This naturally excludes
    weekends and exchange holidays.

    Returns tuples:
        (date, strategy_pnl, sgov_return)
    """
    start = min(t["open"] for t in trades)
    end = max(t["close"] for t in trades)

    by_day = {}

    for t in trades:
        d = t["close"]
        by_day[d] = by_day.get(d, 0.0) + t["pnl"]

    days = []

    for d in sorted(sgov_returns):
        if start <= d <= end:
            days.append(
                (
                    d,
                    by_day.get(d, 0.0),
                    sgov_returns[d],
                )
            )

    if not days:
        raise SystemExit(
            "No overlapping SGOV trading days were found for the "
            "strategy date range."
        )

    return days


def build_payload(trades, sgov_returns):
    daily = daily_pnl(trades, sgov_returns)

    equity = STARTING_CAPITAL
    returns = []
    eq_curve = []
    total_daily = []
    strategy_daily = []
    sgov_daily = []
    equity_at_start = {}

    for d, strategy_pnl, sgov_return in daily:
        equity_at_start[d] = equity

        sgov_value = equity * SGOV_ALLOCATION
        sgov_pnl = sgov_value * sgov_return
        total_pnl = strategy_pnl + sgov_pnl

        daily_return = (
            total_pnl / equity
            if equity != 0
            else 0.0
        )

        returns.append(daily_return)
        total_daily.append((d, total_pnl))
        strategy_daily.append((d, strategy_pnl))
        sgov_daily.append((d, sgov_pnl))

        equity += total_pnl
        eq_curve.append((d, equity))

    points = [
        (eq_curve[0][0], STARTING_CAPITAL)
    ] + eq_curve

    return {
        "daily": total_daily,
        "strategy_daily": strategy_daily,
        "sgov_daily": sgov_daily,
        "equity": eq_curve,
        "equity_at_start": equity_at_start,
        "returns": returns,
        "points": points,
        "n_days": len(total_daily),
        "trades": trades,
    }


# ============================================================
# RISK METRICS
# ============================================================

def sharpe(
    returns,
    ann=TRADING_DAYS_PER_YEAR,
    rf=RISK_FREE_RATE
):
    """
    Trading-day Sharpe ratio.

    Formula:

        Sharpe =
            mean(daily excess return)
            --------------------------------
            population std(daily excess return)
            *
            sqrt(252)

    Annual risk-free rate:
        3.5%

    Daily risk-free rate:
        3.5% / 252

    Only trading days are present in `returns`.
    """

    if not returns:
        return 0.0

    daily_rf = rf / ann

    excess = [
        r - daily_rf
        for r in returns
    ]

    mu = statistics.fmean(excess)

    sd = statistics.pstdev(excess)

    if sd <= 0:
        return 0.0

    return (
        mu
        / sd
        * math.sqrt(ann)
    )


def sortino(
    returns,
    ann=TRADING_DAYS_PER_YEAR,
    rf=RISK_FREE_RATE
):
    """
    Trading-day Sortino ratio.

    Uses the same 252-day / 3.5% RF convention
    as Sharpe.
    """

    if not returns:
        return 0.0

    daily_rf = rf / ann

    excess = [
        r - daily_rf
        for r in returns
    ]

    mu = statistics.fmean(excess)

    downside = statistics.fmean(
        min(v, 0.0) ** 2
        for v in excess
    )

    dd = math.sqrt(downside)

    if dd <= 0:
        return 0.0

    return (
        mu
        / dd
        * math.sqrt(ann)
    )


def cvar(returns, level=CVAR_LEVEL):
    if not returns:
        return 0.0

    vals = sorted(returns)

    n_worst = max(
        1,
        math.ceil(
            (1 - level)
            * len(vals)
        )
    )

    return statistics.fmean(
        vals[:n_worst]
    )


# ============================================================
# DRAWDOWN
# ============================================================

def drawdown_stats(points):
    if not points:
        return 0.0, 0.0

    peak = points[0][1]

    max_dd = 0.0
    max_dd_val = 0.0

    peak_val = peak

    for _, v in points:

        if v > peak_val:
            peak_val = v
            peak = v

        dd = peak - v

        if dd > max_dd:
            max_dd = dd

    return (
        max_dd / peak
        if peak > 0
        else 0.0,
        max_dd
    )


# ============================================================
# MARGIN UTILIZATION
# ============================================================

def is_trading_day(d):
    """Return True for weekdays (Mon-Fri)."""
    return d.weekday() < 5

def margin_utilization(
    trades,
    daily_days,
    equity_at_day_start
):
    per_day = {}

    for t in trades:

        d = t["open"]

        while d <= t["close"]:

            if is_trading_day(d):

                per_day[d] = (
                    per_day.get(d, 0.0)
                    + t["bp"]
                )

            d += timedelta(days=1)

    day_utils = []

    for d, _ in daily_days:

        equity = equity_at_day_start.get(d)

        if equity and equity != 0:

            day_utils.append(
                per_day.get(d, 0.0)
                / equity
            )

        else:
            day_utils.append(0.0)

    invested_days = sum(
        1
        for u in day_utils
        if u > 0
    )

    if not day_utils:
        return 0.0, 0.0, 0

    return (
        statistics.fmean(day_utils),
        max(day_utils),
        invested_days
    )


# ============================================================
# DATASET STATISTICS
# ============================================================

def dataset_stats(p, live):
    n_days = p["n_days"]

    max_dd_frac, max_dd_val = (
        drawdown_stats(p["points"])
    )

    end_eq = p["points"][-1][1]

    if n_days > 0:

        cagr_ = (
            end_eq
            / STARTING_CAPITAL
        ) ** (
            TRADING_DAYS_PER_YEAR
            / n_days
        ) - 1

    else:
        cagr_ = 0.0

    mar = (
        cagr_
        / max_dd_frac
        if max_dd_frac > 0
        else float("inf")
    )

    returns = p["returns"]

    pnls = [
        t["pnl"]
        for t in p["trades"]
    ]

    months = {}
    strategy_months = {}
    sgov_months = {}

    for d, pnl in p["daily"]:
        months[(d.year, d.month)] = (
            months.get((d.year, d.month), 0.0) + pnl
        )

    for d, pnl in p["strategy_daily"]:
        strategy_months[(d.year, d.month)] = (
            strategy_months.get((d.year, d.month), 0.0) + pnl
        )

    for d, pnl in p["sgov_daily"]:
        sgov_months[(d.year, d.month)] = (
            sgov_months.get((d.year, d.month), 0.0) + pnl
        )

    util = {}

    if live:

        equity_at_start = p["equity_at_start"]

        avg, peak, invested = (
            margin_utilization(
                p["trades"],
                p["daily"],
                equity_at_start
            )
        )

        util = {
            "avg": avg,
            "peak": peak,
            "invested": (
                invested / n_days
                if n_days
                else 0.0
            )
        }

    return {
        "sharpe_raw": sharpe(returns),

        "sortino_raw": sortino(returns),

        "cagr": cagr_,

        "mar": mar,

        "cvar": cvar(returns),

        "end_eq": end_eq,

        "net_pnl": sum(
            pnl
            for _, pnl in p["daily"]
        ),

        "strategy_pnl": sum(
            pnl
            for _, pnl in p["strategy_daily"]
        ),

        "sgov_pnl": sum(
            pnl
            for _, pnl in p["sgov_daily"]
        ),

        "n_trades": len(
            p["trades"]
        ),

        "win_rate": (
            sum(
                1
                for x in pnls
                if x > 0
            )
            / len(pnls)
            if pnls
            else 0.0
        ),

        "avg_trade": (
            statistics.fmean(pnls)
            if pnls
            else 0.0
        ),

        "best_trade": (
            max(pnls)
            if pnls
            else 0.0
        ),

        "worst_trade": (
            min(pnls)
            if pnls
            else 0.0
        ),

        "max_dd_frac": max_dd_frac,

        "max_dd_val": max_dd_val,

        "months": months,

        "strategy_months": strategy_months,

        "sgov_months": sgov_months,

        "util": util,
    }


# ============================================================
# FORMATTING
# ============================================================

def fmt_money(v):
    return f"${v:,.0f}"


def fmt_pct(v):
    return f"{v * 100:,.2f}%"


def fmt_num(v):
    return f"{v:,.2f}"


# ============================================================
# EQUITY CURVE
# ============================================================

def align_series(points_a, points_b):

    d0 = min(
        points_a[0][0],
        points_b[0][0]
    )

    d1 = max(
        points_a[-1][0],
        points_b[-1][0]
    )

    out = []

    for pts in (
        points_a,
        points_b
    ):

        by_day = {
            d: v
            for d, v in pts
        }

        seq = []

        last = None

        d = d0

        while d <= d1:

            if d in by_day:
                last = by_day[d]

            if last is not None:
                seq.append(
                    (
                        d,
                        last
                    )
                )

            d += timedelta(days=1)

        out.append(seq)

    return out


def svg_equity_overlay(
    series_a,
    series_b,
    width=900,
    height=300
):

    vals = (
        [v for _, v in series_a]
        +
        [v for _, v in series_b]
    )

    lo = min(vals)
    hi = max(vals)

    rng = (
        hi - lo
    ) or 1.0

    pad = rng * 0.08

    lo -= pad
    hi += pad

    step = (
        width
        / max(
            1,
            len(series_a) - 1
        )
    )

    def line(series):

        return " ".join(
            f"{i * step:.1f},"
            f"{height - (v - lo) / (hi - lo) * height:.1f}"
            for i, (_, v)
            in enumerate(series)
        )

    a_pts = line(series_a)
    b_pts = line(series_b)

    area = (
        f"0,{height} "
        + a_pts
        + f" {width},{height}"
    )

    grid_y = [
        0,
        height / 2,
        height
    ]

    grid = "".join(
        f'''
        <line
            x1="0"
            y1="{y:.1f}"
            x2="{width}"
            y2="{y:.1f}"
            stroke="#1f2733"
            stroke-width="1"
        />
        '''
        for y in grid_y
    )

    up = (
        series_a[-1][1]
        >= series_a[0][1]
    )

    color = (
        "#22c55e"
        if up
        else "#f87171"
    )

    return f"""
    <svg
        viewBox="0 0 {width} {height}"
        preserveAspectRatio="none"
        style="width:100%;height:auto"
    >

        {grid}

        <polygon
            points="{area}"
            fill="{color}"
            opacity="0.08"
        />

        <polyline
            points="{a_pts}"
            fill="none"
            stroke="{color}"
            stroke-width="2"
        />

        <polyline
            points="{b_pts}"
            fill="none"
            stroke="#f59e0b"
            stroke-width="2"
            stroke-dasharray="6 4"
        />

    </svg>
    """


# ============================================================
# HTML HELPERS
# ============================================================

def color_of(raw, threshold=1.0):

    return (
        "green"
        if raw >= threshold
        else "red"
    )


def card_html(
    label,
    live_val,
    bt_val,
    color,
    bt_color=None
):

    if bt_val is None:

        bt_html = """
        <div class="card-bt">
            <span class="bt-badge">BT</span>
            <span>—</span>
        </div>
        """

    elif bt_color is not None:

        bt_html = f"""
        <div class="card-bt">
            <span class="bt-badge">BT</span>
            <span class="{bt_color}">
                {bt_val}
            </span>
        </div>
        """

    else:

        bt_html = f"""
        <div class="card-bt">
            <span class="bt-badge">BT</span>
            <span>{bt_val}</span>
        </div>
        """

    return f"""
    <div class="card">

        <div class="card-label">
            {label}
        </div>

        <div class="card-value {color}">
            {live_val}
        </div>

        {bt_html}

    </div>
    """


def chip_html(
    label,
    value,
    bt=None
):

    bt_html = (
        f'<span class="chip-bt">BT {bt}</span>'
        if bt is not None
        else ""
    )

    return f"""
    <div class="chip">

        <span class="chip-label">
            {label}
        </span>

        <span class="chip-value">
            {value}
        </span>

        {bt_html}

    </div>
    """


def monthly_html(
    live_months,
    bt_months
):

    keys = sorted(
        set(live_months)
        |
        set(bt_months)
    )

    rows = []

    for k in keys:

        lv = live_months.get(
            k,
            0.0
        )

        bv = bt_months.get(
            k,
            0.0
        )

        rows.append(
            f"""
            <div class="month-row">

                <span>
                    {date(
                        k[0],
                        k[1],
                        1
                    ).strftime("%B %Y")}
                </span>

                <span class="{
                    'pos'
                    if lv >= 0
                    else 'neg'
                }">
                    {fmt_money(lv)}
                </span>

                <span class="{
                    'pos'
                    if bv >= 0
                    else 'neg'
                }">
                    {fmt_money(bv)}
                </span>

            </div>
            """
        )

    return "".join(rows)


# ============================================================
# HTML
# ============================================================

def build_html(stats, meta):

    l = stats["live"]
    b = stats["bt"]

    ytd = lambda s: (
        s["end_eq"]
        / STARTING_CAPITAL
        - 1
    )

    cvar_label = lambda s: (
        f"${abs(s['cvar']) * s['end_eq']:,.0f}"
        f" ({fmt_pct(abs(s['cvar']))})"
    )

    mar_label = lambda s: (
        "∞"
        if s["mar"] == float("inf")
        else fmt_num(s["mar"])
    )

    cards = "".join(
        [

            card_html(
                "Sharpe (252d, 3.5% RF)",
                fmt_num(
                    l["sharpe_raw"]
                ),
                fmt_num(
                    b["sharpe_raw"]
                ),
                color_of(
                    l["sharpe_raw"]
                ),
                color_of(
                    b["sharpe_raw"]
                )
            ),

            card_html(
                "Sortino (252d, 3.5% RF)",
                fmt_num(
                    l["sortino_raw"]
                ),
                fmt_num(
                    b["sortino_raw"]
                ),
                color_of(
                    l["sortino_raw"]
                ),
                color_of(
                    b["sortino_raw"]
                )
            ),

            card_html(
                "MAR",
                mar_label(l),
                mar_label(b),
                (
                    color_of(l["mar"])
                    if l["mar"] != float("inf")
                    else "green"
                ),
                (
                    color_of(b["mar"])
                    if b["mar"] != float("inf")
                    else "green"
                )
            ),

            card_html(
                "cVaR 95% (daily)",
                cvar_label(l),
                cvar_label(b),
                "red",
                "red"
            ),

            card_html(
                "Avg Margin Utilization",
                fmt_pct(
                    l["util"]["avg"]
                ),
                None,
                ""
            ),

            card_html(
                "YTD Return",
                fmt_pct(
                    ytd(l)
                ),
                fmt_pct(
                    ytd(b)
                ),
                color_of(
                    ytd(l),
                    0.0
                ),
                color_of(
                    ytd(b),
                    0.0
                )
            ),

        ]
    )

    chips = "".join(
        [

            chip_html(
                "Net PnL",
                fmt_money(
                    l["net_pnl"]
                ),
                fmt_money(
                    b["net_pnl"]
                )
            ),
            chip_html(
                "0DTE Strategy P&L",
                fmt_money(l["strategy_pnl"]),
                fmt_money(b["strategy_pnl"])
            ),

            chip_html(
                "SGOV P&L (85%)",
                fmt_money(l["sgov_pnl"]),
                fmt_money(b["sgov_pnl"])
            ),

            chip_html(
                "SGOV Allocation",
                fmt_pct(SGOV_ALLOCATION),
                fmt_pct(SGOV_ALLOCATION)
            ),


            chip_html(
                "Trades",
                str(
                    l["n_trades"]
                ),
                str(
                    b["n_trades"]
                )
            ),

            chip_html(
                "Win Rate",
                fmt_pct(
                    l["win_rate"]
                ),
                fmt_pct(
                    b["win_rate"]
                )
            ),

            chip_html(
                "Max Drawdown",
                fmt_pct(
                    l["max_dd_frac"]
                ),
                fmt_pct(
                    b["max_dd_frac"]
                )
            ),

            chip_html(
                "CAGR (ann.)",
                fmt_pct(
                    l["cagr"]
                ),
                fmt_pct(
                    b["cagr"]
                )
            ),

            chip_html(
                "Avg Trade",
                fmt_money(
                    l["avg_trade"]
                )
            ),

            chip_html(
                "Best Trade",
                fmt_money(
                    l["best_trade"]
                )
            ),

            chip_html(
                "Worst Trade",
                fmt_money(
                    l["worst_trade"]
                )
            ),

            chip_html(
                "Peak Utilization",
                fmt_pct(
                    l["util"]["peak"]
                )
            ),

            chip_html(
                "Days in Market",
                fmt_pct(
                    l["util"]["invested"]
                )
            ),

        ]
    )

    months = monthly_html(
        l["months"],
        b["months"]
    )

    return f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
    Performance Dashboard
</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{

    margin: 0;

    background: #0b0f14;

    color: #e5e7eb;

    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

}}

.container {{

    max-width: 1200px;

    margin: auto;

    padding: 32px;

}}

h1 {{

    margin: 0 0 6px 0;

    font-size: 28px;

}}

.subtitle {{

    color: #8b95a5;

    margin-bottom: 28px;

}}

.grid {{

    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(230px, 1fr)
        );

    gap: 14px;

}}

.card {{

    background: #111821;

    border:
        1px solid #1f2937;

    border-radius: 12px;

    padding: 18px;

}}

.card-label {{

    font-size: 13px;

    color: #8b95a5;

    margin-bottom: 8px;

}}

.card-value {{

    font-size: 28px;

    font-weight: 700;

}}

.card-bt {{

    margin-top: 8px;

    font-size: 13px;

    color: #9ca3af;

}}

.bt-badge {{

    background: #252d38;

    padding:
        2px 5px;

    border-radius: 4px;

    margin-right: 5px;

}}

.green {{
    color: #4ade80;
}}

.red {{
    color: #f87171;
}}

.pos {{
    color: #4ade80;
}}

.neg {{
    color: #f87171;
}}

.section {{

    margin-top: 30px;

    background: #111821;

    border:
        1px solid #1f2937;

    border-radius: 12px;

    padding: 20px;

}}

.section-title {{

    font-size: 16px;

    font-weight: 600;

    margin-bottom: 18px;

}}

.chips {{

    display: flex;

    flex-wrap: wrap;

    gap: 10px;

}}

.chip {{

    background: #161e28;

    border:
        1px solid #242e3b;

    border-radius: 8px;

    padding:
        10px 14px;

    min-width: 130px;

}}

.chip-label {{

    display: block;

    font-size: 11px;

    color: #7f8a99;

    margin-bottom: 4px;

}}

.chip-value {{

    font-size: 15px;

    font-weight: 600;

}}

.chip-bt {{

    display: block;

    color: #9ca3af;

    font-size: 11px;

    margin-top: 3px;

}}

.month-row {{

    display: grid;

    grid-template-columns:
        1fr
        150px
        150px;

    padding:
        10px 0;

    border-bottom:
        1px solid #202832;

}}

.month-header {{

    color: #7f8a99;

    font-size: 12px;

}}

.legend {{

    display: flex;

    gap: 20px;

    margin-top: 12px;

    font-size: 12px;

    color: #9ca3af;

}}

.sw {{

    display: inline-block;

    width: 10px;

    height: 10px;

    border-radius: 50%;

    margin-right: 6px;

}}

.sw-live {{

    background: #22c55e;

}}

.sw-bt {{

    background: #f59e0b;

}}

.info {{

    margin-top: 20px;

    padding: 12px 14px;

    background: #0e141c;

    border-radius: 8px;

    color: #8b95a5;

    font-size: 12px;

    line-height: 1.5;

}}

</style>

</head>

<body>

<div class="container">

    <h1>
        Performance Dashboard
    </h1>

    <div class="subtitle">

        {meta["period"]}
        &nbsp;·&nbsp;
        {meta["csv"]}

    </div>


    <!-- ============================================ -->
    <!-- PERFORMANCE CARDS                            -->
    <!-- ============================================ -->

    <div class="grid">

        {cards}

    </div>


    <!-- ============================================ -->
    <!-- SHARPE METHODOLOGY                           -->
    <!-- ============================================ -->

    <div class="info">

        <strong>Sharpe methodology:</strong>

        Trading days only ·
        252-day annualization ·
        3.5% annual risk-free rate ·
        daily RF = 3.5% / 252 ·
        population standard deviation.

        <br><br>

        <strong>SGOV simulation:</strong>
        85% of beginning-of-day portfolio equity is allocated to SGOV,
        using Yahoo Finance dividend-adjusted daily prices. SGOV gains
        are reinvested and the target allocation scales automatically as
        portfolio equity changes.

    </div>


    <!-- ============================================ -->
    <!-- SUMMARY                                      -->
    <!-- ============================================ -->

    <div class="section">

        <div class="section-title">
            Summary
        </div>

        <div class="chips">

            {chips}

        </div>

    </div>


    <!-- ============================================ -->
    <!-- EQUITY CURVE                                 -->
    <!-- ============================================ -->

    <div class="section">

        <div class="section-title">
            Equity Curve
        </div>

        {stats["equity_svg"]}

        <div class="legend">

            <span>
                <span class="sw sw-live"></span>
                Live
            </span>

            <span>
                <span class="sw sw-bt"></span>
                Backtest ({meta["btcsv"]})
            </span>

        </div>

    </div>


    <!-- ============================================ -->
    <!-- MONTHLY P&L                                  -->
    <!-- ============================================ -->

    <div class="section">

        <div class="section-title">
            Monthly P&L
        </div>

        <div class="month-row month-header">

            <span>Month</span>

            <span>Live</span>

            <span>Backtest</span>

        </div>

        {months}

    </div>


    <!-- ============================================ -->
    <!-- EXCLUDED STRATEGIES                          -->
    <!-- ============================================ -->

    {
        f'''
        <div class="info">
            {meta["excluded"]}
        </div>
        '''
        if meta.get("excluded")
        else ""
    }

</div>

</body>

</html>
"""


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Locate files
    # --------------------------------------------------------

    csv_path = newest_csv()

    live_trades, excluded = (
        load_trades(csv_path)
    )

    bt_trades = load_trades_am(
        BACKTEST_CSV
    )


    # --------------------------------------------------------
    # Build datasets
    # --------------------------------------------------------

    all_trades = live_trades + bt_trades

    sgov_start = min(t["open"] for t in all_trades)
    sgov_end = max(t["close"] for t in all_trades)

    sgov_returns = load_sgov_returns(
        sgov_start,
        sgov_end
    )

    live = build_payload(
        live_trades,
        sgov_returns
    )

    bt = build_payload(
        bt_trades,
        sgov_returns
    )


    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    live_stats = dataset_stats(
        live,
        live=True
    )

    bt_stats = dataset_stats(
        bt,
        live=False
    )


    # --------------------------------------------------------
    # Align equity curves
    # --------------------------------------------------------

    live_series, bt_series = (
        align_series(
            live["points"],
            bt["points"]
        )
    )


    # --------------------------------------------------------
    # Build output
    # --------------------------------------------------------

    stats = {

        "live": live_stats,

        "bt": bt_stats,

        "equity_svg":
            svg_equity_overlay(
                live_series,
                bt_series
            ),

    }


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    meta = {
    "csv_path": csv_path,
    "start": live["daily"][0][0].isoformat(),
    "end": live["daily"][-1][0].isoformat(),
    "csv": os.path.basename(csv_path),
    "btcsv": os.path.basename(BACKTEST_CSV),
    "period": (
        f"{live['daily'][0][0].isoformat()} — "
        f"{live['daily'][-1][0].isoformat()}"
    ),
}


    if excluded:

        meta["excluded"] = (
            "Excluded: "
            + ", ".join(
                sorted(
                    EXCLUDED_STRATEGIES
                )
            )
            + f" ({excluded} trades)"
        )


    # --------------------------------------------------------
    # Generate HTML
    # --------------------------------------------------------

    html = build_html(
        stats,
        meta
    )

    output_path = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        OUTPUT_HTML
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(html)


    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print(
        f"Dashboard written to "
        f"{OUTPUT_HTML}"
    )

    print()

    print(
        "Sharpe configuration:"
    )

    print(
        f"  Annualization: "
        f"{TRADING_DAYS_PER_YEAR}"
    )

    print(
        f"  Risk-free rate: "
        f"{RISK_FREE_RATE:.2%}"
    )

    print(
        f"  Daily RF: "
        f"{RISK_FREE_RATE / TRADING_DAYS_PER_YEAR:.8%}"
    )

    print()

    print(
        f"Live Sharpe: "
        f"{live_stats['sharpe_raw']:.4f}"
    )

    print(
        f"Backtest Sharpe: "
        f"{bt_stats['sharpe_raw']:.4f}"
    )

    print()
    print(f"SGOV allocation: {SGOV_ALLOCATION:.0%}")
    print(f"Live strategy P&L: {live_stats['strategy_pnl']:,.2f}")
    print(f"Live SGOV P&L: {live_stats['sgov_pnl']:,.2f}")


if __name__ == "__main__":
    main()
