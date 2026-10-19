Local runs (laptop, res 10, tri-graded, 1000 fs), `dt_control.py`:

| run | pol | dt [as] | steps | RMS Q_sca error |
|---|---|---|---|---|
| global (Batch) | TM | 70.4 | 14208 | 0.0688 |
| LTS, coarse dt = global dt | TM | 70.4 | 14208 | 0.0688 |
| global, cfl 0.475 | TM | 35.2 | 28416 | 0.0762 |
| LTS (Batch) | TM | 136.9 | 7305 | 0.0318 |
| global (Batch) | TE | 69.7 | 14337 | 0.0915 |
| LTS, coarse dt = global dt | TE | 69.7 | 14337 | 0.0915 |
| global, cfl 0.475 | TE | 34.9 | 28675 | 0.0990 |
| LTS (Batch) | TE | 136.9 | 7305 | 0.0511 |
