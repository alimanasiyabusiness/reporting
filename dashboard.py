import csv
import glob
import math
import os
import statistics
from datetime import date, datetime, timedelta
from string import Template

STARTING_CAPITAL = 100_000
CSV_PATTERN = "export-*.csv"
OUTPUT_HTML = "index.html"
TRADING_DAYS_PER_YEAR = 252
CVAR_LEVEL = 0.95
EXCLUDED_STRATEGIES = {"$2 Put Credit Spread"}


def newest_csv():
    files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), CSV_PATTERN))
    if not files:
        raise SystemExit(f"No CSV matching {CSV_PATTERN} found")
    return max(files, key=os.path.getmtime)


def load_trades(path):
    trades = []
    excluded = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                close_date = datetime.strptime(row["CloseDate"], "%Y-%m-%d").date()
                open_date = datetime.strptime(row["OpenDate"], "%Y-%m-%d").date()
                pnl = float(row["ProfitLoss"])
                bp = float(row["BuyingPower"]) if row.get("BuyingPower") else 0.0
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
        raise SystemExit(f"No parseable trades in {path}")
    return trades, excluded


def daily_pnl(trades):
    start = min(t["open"] for t in trades)
    end = max(t["close"] for t in trades)
    by_day = {}
    for t in trades:
        by_day[t["close"]] = by_day.get(t["close"], 0.0) + t["pnl"]
    days = []
    d = start
    while d <= end:
        days.append((d, by_day.get(d, 0.0)))
        d += timedelta(days=1)
    return days


def daily_returns(daily, start_capital):
    returns = []
    eq = start_capital
    for _, pnl in daily:
        returns.append(pnl / eq if eq != 0 else 0.0)
        eq += pnl
    return returns


def sharpe(returns):
    mu = statistics.fmean(returns)
    sd = statistics.pstdev(returns)
    return (mu / sd * math.sqrt(TRADING_DAYS_PER_YEAR)) if sd > 0 else 0.0


def sortino(returns):
    mu = statistics.fmean(returns)
    downside = statistics.fmean(min(v, 0.0) ** 2 for v in returns)
    dd = math.sqrt(downside)
    return (mu / dd * math.sqrt(TRADING_DAYS_PER_YEAR)) if dd > 0 else 0.0


def cvar(returns, level=CVAR_LEVEL):
    vals = sorted(returns)
    n_worst = max(1, math.ceil((1 - level) * len(vals)))
    return statistics.fmean(vals[:n_worst])


def equity_curve(daily, start_capital):
    points = []
    eq = start_capital
    for d, pnl in daily:
        eq += pnl
        points.append((d, eq))
    return points


def drawdown_stats(points):
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
    return max_dd / peak if peak > 0 else 0.0, max_dd


def margin_utilization(trades, daily_days, equity_at_day_start):
    per_day = {}
    for t in trades:
        d = t["open"]
        while d <= t["close"]:
            per_day[d] = per_day.get(d, 0.0) + t["bp"]
            d += timedelta(days=1)
    day_utils = [per_day.get(d, 0.0) / equity_at_day_start[d] for d, _ in daily_days]
    invested_days = sum(1 for u in day_utils if u > 0)
    return statistics.fmean(day_utils), max(day_utils), invested_days


def monthly_pnl(daily):
    months = {}
    for d, pnl in daily:
        months.setdefault((d.year, d.month), 0.0)
        months[(d.year, d.month)] += pnl
    return sorted(months.items())


def fmt_money(v):
    return f"${v:,.0f}"


def fmt_pct(v):
    return f"{v * 100:,.2f}%"


def fmt_num(v):
    return f"{v:,.2f}"


def svg_equity(points, width=900, height=300):
    vals = [v for _, v in points]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    pad = rng * 0.08
    lo, hi = lo - pad, hi + pad
    step = width / max(1, len(points) - 1)
    coords = [(i * step, height - (v - lo) / (hi - lo) * height) for i, (_, v) in enumerate(points)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    area = f"0,{height} " + line + f" {width},{height}"
    grid_y = [0, height / 2, height]
    grid = "".join(
        f'<line x1="0" y1="{y:.1f}" x2="{width}" y2="{y:.1f}" stroke="#1f2733" stroke-width="1"/>'
        for y in grid_y
    )
    up = vals[-1] >= vals[0]
    color = "#22c55e" if up else "#f87171"
    return f"""
    <svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" style="width:100%;height:auto">
      {grid}
      <polygon points="{area}" fill="{color}" opacity="0.12"/>
      <polyline points="{line}" fill="none" stroke="{color}" stroke-width="2"/>
    </svg>"""


def build_html(stats, meta):
    m = {
        "title": "Performance Dashboard",
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "csv": os.path.basename(meta["csv_path"]),
        "period": f"{meta['start']} — {meta['end']}",
        "excluded": meta.get("excluded_note", ""),
    }
    cards = [
        ("Sharpe (252d)", stats["sharpe"], "green" if stats["sharpe_raw"] >= 1 else "red"),
        ("Sortino (252d)", stats["sortino"], "green" if stats["sortino_raw"] >= 1 else "red"),
        ("MAR", stats["mar"], "green" if stats["mar_raw"] >= 1 else "red"),
        ("cVaR 95% (daily)", stats["cvar_label"], "red"),
        ("Avg Margin Utilization", stats["util_avg"], "neutral"),
        ("YTD Return", stats["ytd_pct"], "green" if stats["ytd"] >= 0 else "red"),
    ]
    cards_html = "".join(
        f"""
        <div class="card">
          <div class="card-label">{label}</div>
          <div class="card-value {color}">{value}</div>
        </div>"""
        for label, value, color in cards
    )
    chips = [
        ("Net PnL", fmt_money(stats["net_pnl"])),
        ("Trades", str(stats["n_trades"])),
        ("Win Rate", fmt_pct(stats["win_rate"])),
        ("Avg Trade", fmt_money(stats["avg_trade"])),
        ("Best Trade", fmt_money(stats["best_trade"])),
        ("Worst Trade", fmt_money(stats["worst_trade"])),
        ("Max Drawdown", stats["max_dd_pct"]),
        ("CAGR (ann.)", stats["cagr"]),
        ("Peak Utilization", stats["util_peak"]),
        ("Days in Market", stats["invested_pct"]),
    ]
    chips_html = "".join(
        f"""
        <div class="chip">
          <span class="chip-label">{label}</span>
          <span class="chip-value">{value}</span>
        </div>"""
        for label, value in chips
    )
    months_html = "".join(
        f"""
        <div class="month-row">
          <span>{date(year, month, 1).strftime('%B %Y')}</span>
          <span class="{'pos' if pnl >= 0 else 'neg'}">{fmt_money(pnl)}</span>
        </div>"""
        for (year, month), pnl in stats["monthly"]
    )
    m["cards"], m["chips"], m["months"], m["equity_svg"] = cards_html, chips_html, months_html, stats["equity_svg"]
    template = Template(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_template.html")).read())
    return template.substitute(m)


def main():
    csv_path = newest_csv()
    trades, excluded = load_trades(csv_path)
    daily = daily_pnl(trades)
    points = equity_curve(daily, STARTING_CAPITAL)
    n_days = len(daily)
    max_dd_frac, max_dd_val = drawdown_stats(points)
    end_eq = points[-1][1]
    cagr = (end_eq / STARTING_CAPITAL) ** (TRADING_DAYS_PER_YEAR / n_days) - 1 if n_days > 0 else 0.0
    mar = (cagr / max_dd_frac) if max_dd_frac > 0 else float("inf")
    returns = daily_returns(daily, STARTING_CAPITAL)
    equity_at_start = {}
    eq = STARTING_CAPITAL
    for d, pnl in daily:
        equity_at_start[d] = eq
        eq += pnl
    cvar_fraction = cvar(returns)
    util_avg, util_peak, invested_days = margin_utilization(trades, daily, equity_at_start)
    pnls = [t["pnl"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    monthly = monthly_pnl(daily)
    total_pnl = sum(p for _, p in daily)

    sh, so = sharpe(returns), sortino(returns)
    ytd_fraction = end_eq / STARTING_CAPITAL - 1
    cvar_dlr = abs(cvar_fraction) * end_eq
    stats = {
        "sharpe_raw": sh,
        "sortino_raw": so,
        "sharpe": fmt_num(sh),
        "sortino": fmt_num(so),
        "mar_raw": mar,
        "mar": "∞" if mar == float("inf") else fmt_num(mar),
        "cvar_label": f"${cvar_dlr:,.0f} ({fmt_pct(abs(cvar_fraction))})",
        "cvar": cvar_fraction,
        "cvar_pct": fmt_pct(abs(cvar_fraction)),
        "util_avg": fmt_pct(util_avg),
        "util_peak": fmt_pct(util_peak),
        "invested_pct": fmt_pct(invested_days / n_days),
        "ytd_pct": fmt_pct(ytd_fraction),
        "ytd": ytd_fraction,
        "net_pnl": total_pnl,
        "n_trades": len(trades),
        "win_rate": wins / len(pnls),
        "avg_trade": statistics.fmean(pnls),
        "best_trade": max(pnls),
        "worst_trade": min(pnls),
        "max_dd_pct": fmt_pct(max_dd_frac),
        "max_dd_val": fmt_money(max_dd_val),
        "cagr": fmt_pct(cagr),
        "monthly": monthly,
        "equity_svg": svg_equity(points),
    }
    meta = {"csv_path": csv_path, "start": daily[0][0].isoformat(), "end": daily[-1][0].isoformat()}
    if excluded:
        meta["excluded_note"] = " · Excluded: " + ", ".join(sorted(EXCLUDED_STRATEGIES)) + f" ({excluded} trades)"
    html = build_html(stats, meta)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_HTML), "w") as f:
        f.write(html)
    print(f"Dashboard written to {OUTPUT_HTML} from {meta['csv_path']}")


if __name__ == "__main__":
    main()