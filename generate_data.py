"""Create a SYNTHETIC contact-centre volume history (no real company data).

20 months of 30-minute intervals. Volume is built from a trend, weekday and
month patterns, an hourly shape, holidays and marketing campaigns, plus random
noise and a few unannounced incidents nobody could have forecast.
Fixed seed, so results are repeatable.
"""
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

SEED = 7
START, END = date(2025, 1, 1), date(2026, 8, 31)
OPEN_HOUR, CLOSE_HOUR = 7, 23                       # 32 intervals a day

HOUR_SHAPE = np.array([0.02, 0.04, 0.07, 0.09, 0.10, 0.09, 0.07, 0.06,
                       0.07, 0.08, 0.08, 0.06, 0.05, 0.04, 0.04, 0.04])
DOW_FACTOR = [1.10, 1.00, 0.97, 0.98, 1.05, 0.80, 0.65]      # Mon..Sun
MONTH_FACTOR = {1: 1.00, 2: 0.95, 3: 0.98, 4: 1.00, 5: 1.02, 6: 0.97,
                7: 0.96, 8: 1.00, 9: 1.04, 10: 1.12, 11: 1.15, 12: 1.08}
BASE_DAILY = 1500
TREND_PER_DAY = 0.0003                              # volume grows about 11% a year

# Public holidays (customers call less) and the day after (catch-up calls)
HOLIDAYS = [date(2025, 1, 26), date(2025, 3, 14), date(2025, 8, 15), date(2025, 10, 2),
            date(2025, 10, 20), date(2025, 12, 25), date(2026, 1, 26), date(2026, 3, 4),
            date(2026, 8, 15)]
# Marketing campaigns that lift volume for a few days (known in advance)
CAMPAIGNS = [(date(2025, 2, 10), 3), (date(2025, 5, 12), 4), (date(2025, 7, 7), 3),
             (date(2025, 9, 15), 5), (date(2025, 11, 24), 4), (date(2026, 2, 9), 3),
             (date(2026, 4, 13), 4), (date(2026, 6, 15), 3)]
# Unannounced incidents (a product fault, an outage): impossible to forecast
INCIDENT_DAYS = [date(2025, 4, 8), date(2025, 6, 23), date(2025, 10, 30),
                 date(2026, 1, 14), date(2026, 5, 6), date(2026, 7, 21)]


def build() -> pd.DataFrame:
    rnd = np.random.default_rng(SEED)
    hol = set(HOLIDAYS)
    hol_after = {h + timedelta(days=1) for h in HOLIDAYS}
    camp = {c + timedelta(days=i) for c, n in CAMPAIGNS for i in range(n)}
    inc = set(INCIDENT_DAYS)
    rows, n_days = [], (END - START).days + 1
    for d in range(n_days):
        day = START + timedelta(days=d)
        daily = BASE_DAILY * DOW_FACTOR[day.weekday()] * MONTH_FACTOR[day.month]
        daily *= 1 + TREND_PER_DAY * d
        if day in hol:
            daily *= 0.60
        if day in hol_after:
            daily *= 1.20
        if day in camp:
            daily *= 1.28
        daily *= rnd.normal(1.0, 0.04)              # day-to-day randomness
        for i in range(32):
            hour = OPEN_HOUR + i // 2
            mean = daily * HOUR_SHAPE[i // 2] / 2
            if day in inc and 10 <= hour < 15:
                mean *= 1.5
            offered = max(0, int(rnd.normal(mean, np.sqrt(max(mean, 1)))))
            rows.append((day, i, hour, offered, int(day in hol), int(day in hol_after),
                         int(day in camp)))
    df = pd.DataFrame(rows, columns=["date", "slot", "hour", "offered",
                                     "is_holiday", "is_after_holiday", "is_campaign"])
    df["date"] = pd.to_datetime(df["date"])
    return df


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "volume.csv"
    data = build()
    data.to_csv(out, index=False)
    print(f"Wrote {len(data)} intervals over {data['date'].nunique()} days to {out}")
