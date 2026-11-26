"""Run everything: walk-forward test, staffing impact, results.json and chart.png."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from model import add_features, bias, walk_forward, wape
from staffing import summarise

METHODS = [("naive", "Same time last week"), ("seasonal", "Average of last 4 weeks"),
           ("ml", "LightGBM")]


def main(path="volume.csv"):
    df = add_features(pd.read_csv(path, parse_dates=["date"]))
    res = walk_forward(df)
    daily = res.groupby(["date", "special"], as_index=False)[[m for m, _ in METHODS] + ["offered"]].sum()

    out = {"intervals_tested": int(len(res)), "days_tested": int(res["date"].nunique()),
           "folds": int(res["fold"].nunique()), "methods": {}}
    print(f"Walk-forward test: {out['folds']} folds, {out['days_tested']} days, "
          f"{out['intervals_tested']} intervals\n")
    print(f"{'method':26s}{'interval WAPE':>14s}{'daily WAPE':>12s}{'special days':>14s}{'bias':>8s}")
    for m, label in METHODS:
        sp = res[res.special == 1]
        row = {"interval_wape": wape(res.offered, res[m]), "daily_wape": wape(daily.offered, daily[m]),
               "special_wape": wape(sp.offered, sp[m]), "bias": bias(res.offered, res[m])}
        out["methods"][m] = row
        print(f"{label:26s}{row['interval_wape']:>13.1%}{row['daily_wape']:>12.1%}"
              f"{row['special_wape']:>14.1%}{row['bias']:>8.1%}")
    s = summarise(res)
    out["staffing"] = s.to_dict("records")
    print("\nStaffing impact (agents needed from forecast vs from actual volume):")
    print(s.to_string(index=False))
    best = min(out["methods"], key=lambda k: out["methods"][k]["interval_wape"])
    out["best_method"] = best
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)

    # chart: one test week with a campaign or holiday, forecast vs actual
    daily_day = res.groupby("date")[["offered", "naive", "seasonal", "ml"]].sum()
    flagged = res.groupby("date")["special"].max()
    pick = flagged[flagged == 1].index[0]
    week = res[(res.date >= pick - pd.Timedelta(days=2)) & (res.date <= pick + pd.Timedelta(days=4))].reset_index(drop=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
    ax[0].plot(week.index, week.offered, color="#555", lw=1.6, label="Actual")
    ax[0].plot(week.index, week.seasonal, color="#e07b39", lw=1.4, label="Average of last 4 weeks")
    ax[0].plot(week.index, week.ml, color="#2b6cb0", lw=1.6, label="LightGBM")
    ax[0].set_title("One week around a campaign or holiday: contacts per 30-min interval", fontsize=11)
    ax[0].set_xlabel("Interval number in the week")
    ax[0].legend(frameon=False)
    names = ["Last week", "4-week avg", "LightGBM"]
    vals = [out["methods"][m]["interval_wape"] * 100 for m, _ in METHODS]
    bars = ax[1].bar(names, vals, color=["#9aa0a6", "#e07b39", "#2b6cb0"])
    ax[1].set_title("Forecast error (WAPE %, lower is better)", fontsize=11)
    for b, v in zip(bars, vals):
        ax[1].text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}%", ha="center")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "synthetic data", fontsize=8, color="#888")
    fig.tight_layout()
    fig.savefig("chart.png", dpi=130)
    print("\nWrote results.json and chart.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "volume.csv")
