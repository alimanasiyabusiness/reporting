import csv
import glob
import math
import os
import statistics
from datetime import date, datetime, timedelta
from string import Template

STARTING_CAPITAL = 100_000
CSV_PATTERN = "export-*.csv"
BACKTEST_CSV = "AM-Live-Test.csv"
OUTPUT_HTML = "index.html"
DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.00
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


def load_trades_am(path):
    trades = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        raw_names = reader.fieldnames or []
        keys = [k.lstrip("\ufeff").strip().strip('"') for k in raw_names]
        for raw in reader:
            row = {keys[i]: raw[raw_names[i]] for i in range(len(raw_names))}
            try:
                close_date = datetime.strptime(row["Date Closed"], "%Y-%m-%d").date()
                open_date = datetime.strptime(row["Date Opened"], "%Y-%m-%d").date()
                pnl = float(row["P/L"])
                strategy = row.get("Strategy", "")
                trades.append({"close": close_date, "open": open_date, "pnl": pnl, "strategy": strategy})
            except (ValueError, KeyError):
                continue
    if not trades:
        raise SystemExit(f"No parseable trades in {path}")
    return trades


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


def equity_curve(daily, start_capital):
    points = []
    eq = start_capital
    for d, pnl in daily:
        eq += pnl
        points.append((d, eq))
    return points


def build_payload(trades):
    daily = daily_pnl(trades)
    eq_curve = equity_curve(daily, STARTING_CAPITAL)
    returns = []
    eq = STARTING_CAPITAL
    for _, pnl in daily:
        returns.append(pnl / eq if eq != 0 else 0.0)
        eq += pnl
    points = [(eq_curve[0][0], STARTING_CAPITAL)] + eq_curve
    return {"daily": daily, "equity": eq_curve, "returns": returns, "points": points, "n_days": len(daily), "trades": trades}


def sharpe(returns, ann=DAYS_PER_YEAR, rf=RISK_FREE_RATE):
    excess = [v - rf / ann for v in returns]
    mu = statistics.fmean(excess)
    sd = statistics.pstdev(excess)
    return (mu / sd * math.sqrt(ann)) if sd > 0 else 0.0


def sortino(returns, ann=DAYS_PER_YEAR, rf=RISK_FREE_RATE):
    excess = [v - rf / ann for v in returns]
    mu = statistics.fmean(excess)
    downside = statistics.fmean(min(v, 0.0) ** 2 for v in excess)
    dd = math.sqrt(downside)
    return (mu / dd * math.sqrt(ann)) if dd > 0 else 0.0


def cvar(returns, level=CVAR_LEVEL):
    vals = sorted(returns)
    n_worst = max(1, math.ceil((1 - level) * len(vals)))
    return statistics.fmean(vals[:n_worst])


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


def dataset_stats(p, live):
    n_days = p["n_days"]
    max_dd_frac, max_dd_val = drawdown_stats(p["points"])
    end_eq = p["points"][-1][1]
    cagr_ = (end_eq / STARTING_CAPITAL) ** (DAYS_PER_YEAR / n_days) - 1 if n_days > 0 else 0.0
    mar = (cagr_ / max_dd_frac) if max_dd_frac > 0 else float("inf")
    returns = p["returns"]
    pnls = [t["pnl"] for t in p["trades"]]
    months = {}
    for d, pnl in p["daily"]:
        months[(d.year, d.month)] = months.get((d.year, d.month), 0.0) + pnl
    util = {}
    if live:
        equity_at_start = {}
        eq = STARTING_CAPITAL
        for d, pnl in p["daily"]:
            equity_at_start[d] = eq
            eq += pnl
        avg, peak, invested = margin_utilization(p["trades"], p["daily"], equity_at_start)
        util = {"avg": avg, "peak": peak, "invested": invested / n_days}
    return {
        "sharpe_raw": sharpe(returns),
        "sortino_raw": sortino(returns),
        "cagr": cagr_,
        "mar": mar,
        "cvar": cvar(returns),
        "end_eq": end_eq,
        "net_pnl": sum(pnl for _, pnl in p["daily"]),
        "n_trades": len(p["trades"]),
        "win_rate": sum(1 for x in pnls if x > 0) / len(pnls),
        "avg_trade": statistics.fmean(pnls),
        "best_trade": max(pnls),
        "worst_trade": min(pnls),
        "max_dd_frac": max_dd_frac,
        "max_dd_val": max_dd_val,
        "months": months,
        "util": util,
    }


def fmt_money(v):
    return f"${v:,.0f}"


def fmt_pct(v):
    return f"{v * 100:,.2f}%"


def fmt_num(v):
    return f"{v:,.2f}"


def align_series(points_a, points_b):
    d0 = min(points_a[0][0], points_b[0][0])
    d1 = max(points_a[-1][0], points_b[-1][0])
    out = []
    for pts in (points_a, points_b):
        by_day = {d: v for d, v in pts}
        seq = []
        last = None
        d = d0
        while d <= d1:
            if d in by_day:
                last = by_day[d]
            seq.append((d, last))
            d += timedelta(days=1)
        out.append(seq)
    return out


def svg_equity_overlay(series_a, series_b, width=900, height=300):
    vals = [v for _, v in series_a] + [v for _, v in series_b]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    pad = rng * 0.08
    lo, hi = lo - pad, hi + pad
    step = width / max(1, len(series_a) - 1)

    def line(series):
        return " ".join(f"{i * step:.1f},{height - (v - lo) / (hi - lo) * height:.1f}" for i, (_, v) in enumerate(series))

    a_pts, b_pts = line(series_a), line(series_b)
    area = f"0,{height} " + a_pts + f" {width},{height}"
    grid_y = [0, height / 2, height]
    grid = "".join(
        f'<line x1="0" y1="{y:.1f}" x2="{width}" y2="{y:.1f}" stroke="#1f2733" stroke-width="1"/>'
        for y in grid_y
    )
    up = series_a[-1][1] >= series_a[0][1]
    color = "#22c55e" if up else "#f87171"
    return f"""
    <svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" style="width:100%;height:auto">
      {grid}
      <polygon points="{area}" fill="{color}" opacity="0.08"/>
      <polyline points="{a_pts}" fill="none" stroke="{color}" stroke-width="2"/>
      <polyline points="{b_pts}" fill="none" stroke="#f59e0b" stroke-width="2" stroke-dasharray="6 4"/>
    </svg>"""


def color_of(raw, threshold=1.0):
    return "green" if raw >= threshold else "red"


def card_html(label, live_val, bt_val, color, bt_color=None):
    if bt_val is None:
        bt_html = f'<div class="card-bt"><span class="bt-badge">BT</span><span>—</span></div>'
    elif bt_color is not None:
        bt_html = f'<div class="card-bt"><span class="bt-badge">BT</span><span class="{bt_color}">{bt_val}</span></div>'
    else:
        bt_html = f'<div class="card-bt"><span class="bt-badge">BT</span><span>{bt_val}</span></div>'
    return f"""
        <div class="card">
          <div class="card-label">{label}</div>
          <div class="card-value {color}">{live_val}</div>
          {bt_html}
        </div>"""


def chip_html(label, value, bt=None):
    bt_html = f'<span class="chip-bt">BT {bt}</span>' if bt is not None else ""
    return f"""
        <div class="chip">
          <span class="chip-label">{label}</span>
          <span class="chip-value">{value}</span>
          {bt_html}
        </div>"""


def monthly_html(live_months, bt_months):
    keys = sorted(set(live_months) | set(bt_months))
    rows = []
    for k in keys:
        lv, bv = live_months.get(k, 0.0), bt_months.get(k, 0.0)
        rows.append(
            f"""
        <div class="month-row">
          <span>{date(k[0], k[1], 1).strftime('%B %Y')}</span>
          <span class="{'pos' if lv >= 0 else 'neg'}">{fmt_money(lv)}</span>
          <span class="{'pos' if bv >= 0 else 'neg'}">{fmt_money(bv)}</span>
        </div>"""
        )
    return "".join(rows)


def build_html(stats, meta):
    l, b = stats["live"], stats["bt"]
    ytd = lambda s: s["end_eq"] / STARTING_CAPITAL - 1
    cvar_label = lambda s: f"${abs(s['cvar']) * s['end_eq']:,.0f} ({fmt_pct(abs(s['cvar']))})"
    mar_label = lambda s: "∞" if s["mar"] == float("inf") else fmt_num(s["mar"])

    cards = "".join(
        [
            card_html("Sharpe (365d, 4% rf)", fmt_num(l["sharpe_raw"]), fmt_num(b["sharpe_raw"]),
                      color_of(l["sharpe_raw"]), color_of(b["sharpe_raw"])),
            card_html("Sortino (365d, 4% rf)", fmt_num(l["sortino_raw"]), fmt_num(b["sortino_raw"]),
                      color_of(l["sortino_raw"]), color_of(b["sortino_raw"])),
            card_html("MAR", mar_label(l), mar_label(b),
                      color_of(l["mar"]) if l["mar"] != float("inf") else "green",
                      color_of(b["mar"]) if b["mar"] != float("inf") else "green"),
            card_html("cVaR 95% (daily)", cvar_label(l), cvar_label(b), "red", "red"),
            card_html("Avg Margin Utilization", fmt_pct(l["util"]["avg"]), None, ""),
            card_html("YTD Return", fmt_pct(ytd(l)), fmt_pct(ytd(b)), color_of(ytd(l), 0.0), color_of(ytd(b), 0.0)),
        ]
    )
    chips = "".join(
        [
            chip_html("Net PnL", fmt_money(l["net_pnl"]), fmt_money(b["net_pnl"])),
            chip_html("Trades", str(l["n_trades"]), str(b["n_trades"])),
            chip_html("Win Rate", fmt_pct(l["win_rate"]), fmt_pct(b["win_rate"])),
            chip_html("Max Drawdown", fmt_pct(l["max_dd_frac"]), fmt_pct(b["max_dd_frac"])),
            chip_html("CAGR (ann.)", fmt_pct(l["cagr"]), fmt_pct(b["cagr"])),
            chip_html("Avg Trade", fmt_money(l["avg_trade"])),
            chip_html("Best Trade", fmt_money(l["best_trade"])),
            chip_html("Worst Trade", fmt_money(l["worst_trade"])),
            chip_html("Peak Utilization", fmt_pct(l["util"]["peak"])),
            chip_html("Days in Market", fmt_pct(l["util"]["invested"])),
        ]
    )
    m = {
        "title": "Performance Dashboard",
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "csv": os.path.basename(meta["csv_path"]),
        "btcsv": os.path.basename(BACKTEST_CSV),
        "period": f"{meta['start']} — {meta['end']}",
        "excluded": meta.get("excluded_note", ""),
        "cards": cards,
        "chips": chips,
        "months": monthly_html(l["months"], b["months"]),
        "equity_svg": stats["equity_svg"],
        "equity_legend": stats["equity_legend"],
    }
    template = Template(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_template.html")).read())
    return template.substitute(m)


def main():
    csv_path = newest_csv()
    live_trades, excluded = load_trades(csv_path)
    bt_trades = load_trades_am(BACKTEST_CSV)

    live = build_payload(live_trades)
    bt = build_payload(bt_trades)

    live_stats = dataset_stats(live, live=True)
    bt_stats = dataset_stats(bt, live=False)

    live_series, bt_series = align_series(live["points"], bt["points"])
    stats = {
        "live": live_stats,
        "bt": bt_stats,
        "equity_svg": svg_equity_overlay(live_series, bt_series),
        "equity_legend": (
            '<div class="legend">'
            '<span><span class="sw sw-live"></span>Live</span>'
            f'<span><span class="sw sw-bt"></span>Backtest ({os.path.basename(BACKTEST_CSV)})</span>'
            "</div>"
        ),
    }
    meta = {"csv_path": csv_path, "start": live["daily"][0][0].isoformat(), "end": live["daily"][-1][0].isoformat()}
    if excluded:
        meta["excluded_note"] = " · Excluded: " + ", ".join(sorted(EXCLUDED_STRATEGIES)) + f" ({excluded} trades)"
    html = build_html(stats, meta)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_HTML), "w") as f:
        f.write(html)
    print(f"Dashboard written to {OUTPUT_HTML} from {meta['csv_path']} + {BACKTEST_CSV}")


if __name__ == "__main__":
    main()
