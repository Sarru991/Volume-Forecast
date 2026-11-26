"""Turn a volume forecast into agents needed, and compare with what was really needed.

Uses Erlang C (erlang.py): 80% of contacts answered in 20 seconds, 90% max occupancy,
380 second average handle time. Forecast error is only useful if it shows up here.
"""
import numpy as np
import pandas as pd

from erlang import required_agents

AHT = 380
BIG_GAP = 3


def staffing_table(res: pd.DataFrame, models=("naive", "seasonal", "ml")) -> pd.DataFrame:
    cache = {}

    def need(v):
        v = int(round(v))
        if v not in cache:
            cache[v] = required_agents(v, AHT)
        return cache[v]

    out = pd.DataFrame({"needed": res["offered"].map(need)})
    for m in models:
        out[m] = res[m].map(need) - out["needed"]        # + means over-staffed, - under
    return out


def summarise(res: pd.DataFrame, models=("naive", "seasonal", "ml")) -> pd.DataFrame:
    gap = staffing_table(res, models)
    rows = []
    for m in models:
        g = gap[m]
        rows.append({
            "method": m,
            "within_2_agents_pct": round(100 * (g.abs() <= 2).mean(), 1),
            "short_3plus_intervals": int((g <= -BIG_GAP).sum()),
            "over_3plus_intervals": int((g >= BIG_GAP).sum()),
            "agent_intervals_short": int(-g[g < 0].sum()),
            "agent_intervals_over": int(g[g > 0].sum()),
        })
    return pd.DataFrame(rows)
