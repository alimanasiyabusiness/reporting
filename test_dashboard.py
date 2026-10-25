import subprocess
import sys
import unittest
from datetime import date
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
from xml.etree import ElementTree


with patch.dict(sys.modules, {"yfinance": ModuleType("yfinance")}):
    import dashboard


class DailyReturnsChartTests(unittest.TestCase):
    def test_missing_yfinance_explains_how_to_install_dependencies(self):
        dashboard_path = Path(__file__).with_name("dashboard.py")
        result = subprocess.run(
            [sys.executable, "-S", str(dashboard_path)],
            cwd=dashboard_path.parent,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requirements.txt", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_cumulative_returns_compound_each_daily_return(self):
        compound = getattr(dashboard, "cumulative_returns", None)
        self.assertTrue(callable(compound), "cumulative_returns should be available")

        actual = compound([0.2, -0.2, 0.3])

        for value, expected in zip(actual, [0.2, -0.04, 0.248]):
            self.assertAlmostEqual(value, expected)

    def test_progressive_ema_uses_first_observation_as_seed(self):
        ema = getattr(dashboard, "progressive_ema", None)
        self.assertTrue(callable(ema), "progressive_ema should be available")

        actual = ema([0.01, -0.02, 0.01], period=3)

        for value, expected in zip(actual, [0.01, -0.005, 0.0025]):
            self.assertAlmostEqual(value, expected)

    def test_chart_renders_cumulative_line_and_signed_daily_bars(self):
        render = getattr(dashboard, "svg_cumulative_daily_returns", None)
        self.assertTrue(
            callable(render),
            "svg_cumulative_daily_returns should be available",
        )

        svg = render(
            [
                (date(2026, 1, 30), 0.02),
                (date(2026, 2, 2), -0.01),
                (date(2026, 2, 3), 0.03),
            ],
            width=600,
            height=280,
        )
        root = ElementTree.fromstring(svg)

        bars = root.findall(".//rect[@class='daily-bar']")
        self.assertEqual(len(bars), 3)
        self.assertEqual(
            [bar.attrib["data-direction"] for bar in bars],
            ["positive", "negative", "positive"],
        )

        line_classes = {
            node.attrib.get("class")
            for node in root.findall(".//polyline")
        }
        self.assertIn("cumulative-line", line_classes)
        self.assertIn("ema-line", line_classes)

        labels = " ".join(node.text or "" for node in root.findall(".//text"))
        self.assertIn("Jan", labels)
        self.assertIn("Feb", labels)

    def test_empty_chart_shows_a_safe_empty_state(self):
        render = getattr(dashboard, "svg_cumulative_daily_returns", None)
        self.assertTrue(
            callable(render),
            "svg_cumulative_daily_returns should be available",
        )

        root = ElementTree.fromstring(render([]))
        labels = " ".join(node.text or "" for node in root.iter())

        self.assertIn("No daily return data", labels)

    def test_zero_return_is_not_rendered_as_a_positive_bar(self):
        svg = dashboard.svg_cumulative_daily_returns(
            [(date(2026, 2, 2), 0.0)],
        )
        root = ElementTree.fromstring(svg)

        self.assertEqual(root.findall(".//rect[@class='daily-bar']"), [])

    def test_single_observation_has_visible_line_markers(self):
        svg = dashboard.svg_cumulative_daily_returns(
            [(date(2026, 2, 2), 0.01)],
        )
        root = ElementTree.fromstring(svg)

        marker_classes = [
            marker.attrib.get("class")
            for marker in root.findall(".//circle")
        ]
        self.assertCountEqual(
            marker_classes,
            ["cumulative-point", "ema-point"],
        )

    def test_long_history_uses_sparse_month_labels(self):
        monthly_returns = [
            (date(2020 + index // 12, index % 12 + 1, 1), 0.001)
            for index in range(60)
        ]

        root = ElementTree.fromstring(
            dashboard.svg_cumulative_daily_returns(monthly_returns)
        )
        labels = root.findall(".//text[@class='month-label']")

        self.assertGreater(len(labels), 0)
        self.assertLess(len(labels), len(monthly_returns))
        self.assertLessEqual(len(labels), 18)

    def test_html_adds_daily_chart_before_existing_equity_chart(self):
        live = {
            "end_eq": 510_000.0,
            "cvar": -0.01,
            "mar": 2.0,
            "sharpe_raw": 1.0,
            "sortino_raw": 1.5,
            "util": {"avg": 0.2, "peak": 0.3, "invested": 0.5},
            "portfolio_vol": 0.1,
            "spx_vol": 0.15,
            "vol_ratio": 0.67,
            "correlation": 0.3,
            "net_pnl": 10_000.0,
            "n_trades": 10,
            "win_rate": 0.6,
            "max_dd_frac": 0.05,
            "cagr": 0.12,
            "avg_trade": 1_000.0,
            "best_trade": 3_000.0,
            "worst_trade": -2_000.0,
            "months_pct": {(2026, 1): 0.1},
        }
        stats = {
            "live": live,
            "bt": dict(live),
            "daily_returns_svg": '<svg id="daily-returns"></svg>',
            "equity_svg": '<svg id="existing-equity"></svg>',
        }
        meta = {"period": "2026-01", "btcsv": "backtest.csv"}

        html = dashboard.build_html(stats, meta)

        self.assertIn('id="daily-returns"', html)
        daily_chart = html.index('id="daily-returns"')
        equity_chart = html.index('id="existing-equity"')
        self.assertLess(daily_chart, equity_chart)
        self.assertIn("Monthly Return", html)


if __name__ == "__main__":
    unittest.main()
