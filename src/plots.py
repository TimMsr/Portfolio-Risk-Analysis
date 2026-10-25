"""Charts. They show only rebased indices and percentages, never money amounts.

Labels are generic ("Portfolio", "Index ETF sleeve", ...) so that no ticker
or account information can end up in an image.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; works without a display
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Fixed colour per series (categorical palette, validated for colour-blind
# separation); the benchmark is a neutral dashed line so it reads as reference.
STYLE = {
    "Portfolio": {"color": "#2a78d6", "lw": 2.0},
    "Benchmark": {"color": "#52514e", "lw": 1.6, "ls": "--"},
    "Index ETF sleeve": {"color": "#1baf7a", "lw": 1.6},
    "Individual stock sleeve": {"color": "#eb6834", "lw": 1.6},
}


def _finish(ax, title, ylabel, path):
    ax.set_title(title, loc="left", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    ax.figure.autofmt_xdate()
    ax.figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)


def plot_rebased(rebased, path, title="Cumulative time-weighted return (start = 100)"):
    """Cumulative return, each series starting at 100 (DataFrame: date x series name)."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for name in rebased.columns:
        ax.plot(rebased.index, rebased[name], label=name, **STYLE.get(name, {}))
    ax.axhline(100, color="#a3a29c", lw=0.8)
    _finish(ax, title, "Index level", path)


def plot_drawdown(drawdowns, path):
    """Drawdown from the running peak, in percent (DataFrame: date x series name)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for name in drawdowns.columns:
        ax.plot(drawdowns.index, drawdowns[name], label=name, **STYLE.get(name, {}))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    _finish(ax, "Drawdown from previous peak", "Drawdown", path)


def plot_rolling_vol(rolling_vol, path):
    """Rolling 6-month annualised volatility (DataFrame: date x series name)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for name in rolling_vol.columns:
        ax.plot(rolling_vol.index, rolling_vol[name], label=name, **STYLE.get(name, {}))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(bottom=0)
    _finish(ax, "Rolling 6-month volatility (annualised)", "Volatility", path)
