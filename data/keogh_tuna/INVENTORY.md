# Bluefin Tuna data — local layout

This directory is the local download cache for the Atlantic Bluefin Tuna biologging dataset used by `examples/keogh_tuna_demo.ipynb`. The data file itself is gitignored; only this short layout note is committed.

## What the notebook expects

A CSV with one timestamp column followed by **6 sensor channels** sampled at ~15 s spacing for ~1.8 years (~3.81 M rows). The notebook reads the columns in this order:

| Index | Column        | Unit          | Role                          |
|-------|---------------|---------------|-------------------------------|
| 0     | Date          | Excel serial  | Timestamp (not a sensor)      |
| 1     | Depth         | meters        | Sensor                        |
| 2     | Temperature   | °C            | Sensor                        |
| 3     | Light Level   | tag-internal  | Sensor (used in headline motif) |
| 4     | Ax            | g             | Accelerometer X               |
| 5     | Ay            | g             | Accelerometer Y               |
| 6     | Az            | g             | Accelerometer Z               |

Lines beginning with `;` at the top of the CSV are metadata and are skipped by `pandas.read_csv(..., comment=";")`. Subsequence length used in the showcase is `S = 1280` (~5.3 hours).

## Getting the data

The dataset accompanies a public talk by Eamonn Keogh; obtain it from his shared materials and unzip the CSV into this directory. The notebook will read it as `data/keogh_tuna/<filename>.csv` — adjust the path in the loader cell if your filename differs.
