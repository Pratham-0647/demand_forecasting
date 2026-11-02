"""Phase 8: reorder points from quantile forecasts; service level vs. holding cost.

Compares two ways of setting the reorder point for all 30,490 item x store series
at the final origin:
  lightgbm  mu from the LightGBM point forecast, sigma from its P10-P90 spread
  naive     mu from the seasonal naive, sigma from the std of the last 8 weekly totals
            (the "average plus safety stock from history" rule of thumb)

Costs (assumptions, stated in the README): holding cost 25% of sell price per
year; a lost sale costs a 30% gross margin on the sell price. The recommended
operating point minimises holding + lost-margin cost on the forecast curve.

    python scripts/run_decision.py
Writes results/decision.md, results/decision_curve.csv, results/service_level_vs_cost.png.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import load_forecasts, to_markdown  # noqa: E402
from m5fc import inventory as inv  # noqa: E402
from m5fc.baselines import seasonal_naive  # noqa: E402
from m5fc.data import HORIZON, build_hierarchy  # noqa: E402
from m5fc.features import load_panel  # noqa: E402

RES_DIR = ROOT / "results"
HOLDING_RATE = 0.25  # per year, share of sell price
MARGIN = 0.30  # lost-sale cost, share of sell price
LEAD = 7

# Reference palette (dataviz skill), light mode.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"lightgbm": "#2a78d6", "naive": "#eb6834"}
LABELS = {"lightgbm": "LightGBM quantiles", "naive": "Seasonal naive + historical σ"}


def plot(curves: pd.DataFrame, rec: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for policy, g in curves.groupby("policy", sort=False):
        g = g.sort_values("target_csl")
        ax.plot(g["holding_cost_per_week"], g["fill_rate"] * 100, color=SERIES[policy], lw=2,
                marker="o", ms=5, mec=SURFACE, mew=1.5, label=LABELS[policy], zorder=3)
        last = g.iloc[-1]
        ax.annotate(LABELS[policy], (last["holding_cost_per_week"], last["fill_rate"] * 100),
                    xytext=(6, -2), textcoords="offset points", color=INK2, fontsize=8, va="center")
        p90 = g.iloc[(g["target_csl"] - 0.9).abs().argmin()]
        ax.scatter([p90["holding_cost_per_week"]], [p90["fill_rate"] * 100], s=60, facecolor=SURFACE,
                   edgecolor=SERIES[policy], lw=2, zorder=4)
    ax.scatter([rec["holding_cost_per_week"]], [rec["fill_rate"] * 100], s=140, marker="*",
               color=INK, zorder=5)
    ax.annotate(f"Recommended: target CSL {rec['target_csl']:.1%}\nfill rate {rec['fill_rate']:.1%}, "
                f"${rec['holding_cost_per_week']:,.0f}/week",
                (rec["holding_cost_per_week"], rec["fill_rate"] * 100), xytext=(-10, -38),
                textcoords="offset points", ha="right", fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlabel("Safety-stock holding cost per week, all 30,490 item×store series ($)", color=INK2, fontsize=9)
    ax.set_ylabel("Fill rate (%)", color=INK2, fontsize=9)
    ax.set_title("Service level vs. inventory cost — 7-day lead time, d_1914–d_1941\n"
                 "hollow circle = P90 policy (90% target cycle service level)",
                 color=INK, fontsize=10, loc="left")
    ax.grid(color=GRID, lw=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    xmax = curves["holding_cost_per_week"].max()
    ax.set_xlim(0, xmax * 1.3)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--origin", type=int, default=1913)
    args = p.parse_args()
    o = args.origin

    h = build_hierarchy(levels=[12])
    ids = h.ids["unique_id"].to_numpy()
    fcs = load_forecasts(o, 12, ids, models=["lightgbm", "lightgbm_q10", "lightgbm_q90"])
    if len(fcs) < 3:
        raise SystemExit("need lightgbm point and q10/q90 forecasts at this origin")
    y_train, y_true = h.Y[:, :o], h.Y[:, o: o + HORIZON]
    demand_L = inv.lead_time_windows(y_true, LEAD)

    # Sell price per series (hierarchy L12 order) at the test weeks, for costing.
    panel = load_panel()
    key = "L12/" + panel.meta["item_id"] + "_" + panel.meta["store_id"]
    order = pd.Index(key).get_indexer(ids)
    price_d = np.nan_to_num(panel.price[order, o: o + HORIZON], nan=np.nanmedian(panel.price[:, o]))
    price_L = inv.lead_time_windows(price_d, LEAD) / LEAD  # mean price per window
    unit_holding = price_L * HOLDING_RATE / 52 * (LEAD / 7)
    unit_short = price_L * MARGIN

    mu = {"lightgbm": inv.lead_time_windows(fcs["lightgbm"], LEAD),
          "naive": inv.lead_time_windows(seasonal_naive(h.slice_days(o), HORIZON), LEAD)}
    weekly_hist = y_train[:, -56:].reshape(len(y_train), 8, 7).sum(axis=2)
    sigma = {"lightgbm": inv.lead_time_sigma(fcs["lightgbm_q10"], fcs["lightgbm_q90"], LEAD),
             "naive": np.repeat(weekly_hist.std(axis=1, ddof=1)[:, None], demand_L.shape[1], axis=1)}

    rows = []
    for policy in ["lightgbm", "naive"]:
        for r in inv.sweep(mu[policy], sigma[policy], demand_L, unit_holding):
            rop = np.ceil(np.clip(mu[policy] + r["z"] * sigma[policy], 0, None))
            lost = (np.clip(demand_L - rop, 0, None) * unit_short).sum() / demand_L.shape[1]
            rows.append({"policy": policy, **r, "lost_margin_per_week": float(lost),
                         "total_cost_per_week": r["holding_cost_per_week"] + float(lost)})
    curves = pd.DataFrame(rows)
    lg = curves[curves["policy"] == "lightgbm"]
    rec = lg.loc[lg["total_cost_per_week"].idxmin()].to_dict()

    # Inventory needed by each policy to reach the same fill rate (interpolated).
    def cost_at(policy, fill):
        g = curves[curves["policy"] == policy].sort_values("fill_rate")
        return float(np.interp(fill, g["fill_rate"], g["holding_cost_per_week"]))
    compare = pd.DataFrame([{"fill_rate_target": f, "lightgbm_cost": cost_at("lightgbm", f),
                             "naive_cost": cost_at("naive", f)} for f in [0.90, 0.95, 0.97, 0.98]])
    compare["saving_pct"] = 1 - compare["lightgbm_cost"] / compare["naive_cost"]

    RES_DIR.mkdir(exist_ok=True)
    curves.to_csv(RES_DIR / "decision_curve.csv", index=False, float_format="%.5f")
    plot(curves, rec, RES_DIR / "service_level_vs_cost.png")
    cols = ["target_csl", "cycle_service_level", "fill_rate", "safety_stock_units",
            "holding_cost_per_week", "lost_margin_per_week", "total_cost_per_week"]
    md = [
        f"# Decision layer — reorder points at origin d_{o} (7-day lead time)\n",
        f"Assumptions: holding cost {HOLDING_RATE:.0%} of sell price per year; lost sale costs "
        f"{MARGIN:.0%} margin; costs in $ per week summed over all 30,490 series.\n",
        f"**Recommended operating point (min holding + lost margin, LightGBM):** target CSL "
        f"{rec['target_csl']:.1%} -> achieved CSL {rec['cycle_service_level']:.1%}, fill rate "
        f"{rec['fill_rate']:.1%}, holding ${rec['holding_cost_per_week']:,.0f}/week, lost margin "
        f"${rec['lost_margin_per_week']:,.0f}/week.\n",
        "## LightGBM quantile policy\n", to_markdown(lg[cols], 3),
        "\n## Seasonal naive + historical sigma policy\n", to_markdown(curves[curves["policy"] == "naive"][cols], 3),
        "\n## Holding cost to reach the same fill rate\n", to_markdown(compare, 3),
        "\n![Service level vs cost](service_level_vs_cost.png)\n",
    ]
    (RES_DIR / "decision.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
