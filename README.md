# ODIN — Outlier Detection using Indegree Number

Unsupervised anomaly detection with a **k-Nearest-Neighbour graph** (ODIN,
Hautamäki et al., 2004), applied to **system-monitoring data** (CPU, memory,
disk, network) to flag injected anomalies **without using labels at training
time**.

## Idea

ODIN builds a kNN graph over the data points. Each point's **indegree** — how
many other points keep it as a neighbour — measures how central it is. Points
with a low indegree are isolated in the graph and are flagged as outliers. The
detection threshold is chosen on a validation split from the score
distribution, so no labels are used during training (fully unsupervised).

## Pipeline

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `monitorNORM.py` / `monitorINJ.py` | collect system metrics during a normal run and a fault-injected run |
| 2 | `injector.py` / `LoadInjector.py` | inject controlled load / faults (the anomalies) |
| 3 | `merge_dataset.py` | merge runs into a labelled dataset (`dataset.csv`; label 1 = injection, 0 = normal) |
| 4 | `split_dataset.py` | 70/15/15 train / validation / test split into `split/` |
| 5 | `odin.py` | run ODIN, pick the threshold on validation scores, evaluate on test (accuracy, precision, recall, F1, MCC, confusion matrix) |
| 6 | `esperimenti_odin.py` | experiments varying K, threshold, training data and distance metric; reports AUC and train/test times; saves the graphs |

## Experiments

`esperimenti_odin.py` changes one parameter at a time, starting from the
`odin.py` configuration (K = 10, threshold = 20%):

1. **K** — number of neighbours in the graph
2. **Threshold** — fraction of rows flagged as anomalous
3. **Training data** — the injected training set vs a separate normal run
   (`normal_run.csv`), again with no labels used in training
4. **Distance** — euclidean, Manhattan, cosine, Chebyshev, for every K

## Results

- `grafico_K.png` — metrics vs the number of neighbours K
- `grafico_distanze.png` — metrics vs the distance metric
- `grafico_confronto.png` — training on injected data vs a separate normal run
- `risultati_esperimenti.csv` — the full experiment table

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python split_dataset.py      # build the train/validation/test split
python odin.py               # run ODIN and evaluate
python esperimenti_odin.py   # run the parameter experiments + plots
```

## Tech

Python, NumPy, pandas, scikit-learn, matplotlib, seaborn.

## Reference

Hautamäki, Kärkkäinen, Fränti — *Outlier Detection Using k-Nearest Neighbour
Graph*, ICPR 2004. The evaluation scaffold (70/15/15 split, scaler, threshold
chosen on validation scores without labels) follows the course framework, with
ODIN in place of K-Means.
