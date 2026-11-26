# Contact-Centre Volume Forecaster (ML)

Forecasts contacts per 30-minute interval with LightGBM, compares it against two simple
methods a WFM team really uses, and shows what the forecast error means in **agents**.

> **All data is synthetic.** It is generated from a trend, weekday and month patterns,
> holidays, marketing campaigns and random noise (`generate_data.py`, fixed seed).
> It was built to practise forecasting work, not taken from any company.

## Run it
```
pip install -r requirements.txt
python generate_data.py      # writes volume.csv (608 days x 32 intervals)
python run.py                # walk-forward test, staffing impact, results.json, chart.png
python -m pytest -q          # 7 tests
```

## How it is tested
Walk-forward: the model is trained only on days before each test block, then forecasts
the next 28 days. 8 blocks, 224 test days, 7,168 intervals. Retrained for every block.
Features use only what is known at forecast time: calendar, holidays, campaigns, and
actuals 7, 14, 21 and 28 days earlier. A test checks that no lag peeks at the future.

## Results (synthetic data, 8 test blocks)
| Method | Interval error (WAPE) | Daily error | Holidays and campaigns |
|---|---|---|---|
| Same time last week | 18.5% | 9.4% | 25.6% |
| Average of last 4 weeks | 14.6% | 7.2% | 25.8% |
| **LightGBM** | **12.9%** | **5.5%** | **13.6%** |

Turned into staffing (Erlang C, 80% of contacts in 20 s, 380 s handle time):

| Method | Intervals within 2 agents | Short by 3+ agents | Over by 3+ agents |
|---|---|---|---|
| Same time last week | 65.7% | 1,243 | 1,218 |
| Average of last 4 weeks | 76.4% | 858 | 835 |
| **LightGBM** | **80.4%** | 846 | **558** |

## What this does and does not show
- The biggest win is on **holidays and campaigns**. Simple methods look back at ordinary
  weeks and cannot see them coming. Those events are in the data on purpose, so this shows
  the method works when such patterns exist, not that it will on any company's data.
- LightGBM mostly cuts **over-staffing** (fewer 3+ agent overs). The number of badly
  under-staffed intervals is about the same as the 4-week average (846 vs 858).
- LightGBM slightly under-forecasts overall (bias -1.9%), worth watching in real use.
- Part of the interval error is random noise nobody can forecast. Six unannounced
  incident days are in the data and no method sees them.
- Real data would need real holiday and campaign calendars and cleaning.

## Files
`generate_data.py` synthetic history, `model.py` features, three methods and
walk-forward test, `staffing.py` forecast to agents, `erlang.py` Erlang C maths,
`run.py` the full run, `tests/` seven tests, `chart.png` and `results.json` outputs.
