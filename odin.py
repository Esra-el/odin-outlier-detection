"""
ODIN - Outlier Detection using Indegree Number (Hautamaki et al., 2004)
applicato al dataset creato con monitor.py + injector.py + merge_dataset.py.

Struttura ripresa dal codice del prof (split 70/15/15, scaler, soglia
scelta sui punteggi di validation senza usare le etichette, stessa
funzione di valutazione), ma con ODIN al posto di K-Means.

La divisione in training, validation e test la fa split_dataset.py, che la
salva nella cartella split/: questo script legge quei tre file.

Le prove con parametri diversi sono in esperimenti_odin.py.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

# ------------------------------------------------------------------
# PARAMETRI
# ------------------------------------------------------------------
TRAIN_FILE = "split/train.csv"            # creati da split_dataset.py
VAL_FILE = "split/validation.csv"
TEST_FILE = "split/test.csv"
# Le colonne che ODIN vede. Quelle che mancano nel file letto vengono saltate,
# cosi' lo script funziona sia con il dataset vecchio (solo CPU e memoria)
# sia con quello nuovo (anche disco e rete).
FEATURES = ["cpu_percent", "mem_percent",
            "disk_read_mb_s", "disk_write_mb_s",
            "net_recv_kb_s", "net_sent_kb_s"]
TARGET_COLUMN = "label"                     # 1 = injection, 0 = normale
RANDOM_STATE = 42
K = 10                  # numero di vicini nel grafo kNN
METRIC = "euclidean"    # distanza tra i punti: "euclidean", "manhattan", "cosine", "chebyshev"
CHUNK = 1000            # righe elaborate per volta nel calcolo delle distanze (limita la memoria usata)
ALERT_FRACTION = 0.20   # frazione di righe di validation da segnalare come anomale
                        # (circa la quota attesa di injection con inj_probability = 0.1)


def select_features(df):
    """Restituisce le colonne di FEATURES presenti nel file letto."""
    return [c for c in FEATURES if c in df.columns]


def load_split():
    """Legge i tre file creati da split_dataset.py: training, validation e test."""
    for f in (TRAIN_FILE, VAL_FILE, TEST_FILE):
        if not os.path.exists(f):
            raise SystemExit("Manca il file %s: lanciate prima python split_dataset.py" % f)
    return pd.read_csv(TRAIN_FILE), pd.read_csv(VAL_FILE), pd.read_csv(TEST_FILE)


# ------------------------------------------------------------------
# ODIN
# ------------------------------------------------------------------
class ODIN:
    """
    Si costruisce il grafo kNN dei dati di training: ogni punto ha una freccia
    verso i suoi K vicini piu' prossimi. L'indegree di un punto e' quante
    frecce riceve, cioe' per quanti altri punti lui e' "uno dei K piu' vicini".
    Un punto con indegree basso non e' vicino a nessuno: e' un outlier.

    Per un punto nuovo x (validation o test) l'indegree e' il numero di punti
    di training che metterebbero x tra i loro K vicini, cioe' i punti p per cui
    distanza(p, x) <= distanza di p dal suo K-esimo vicino.
    """

    def __init__(self, k=10, metric="euclidean"):
        self.k = k
        self.metric = metric    # come si misura la distanza tra due punti

    def fit(self, X_train):
        self.X_train = X_train
        # Distanza di ogni punto di training dal suo K-esimo vicino (il "raggio").
        # Le distanze si calcolano a blocchi di righe, cosi' anche con decine di
        # migliaia di righe non serve tenere in memoria la tabella di tutte le coppie.
        radius = np.empty(len(X_train))
        for start in range(0, len(X_train), CHUNK):
            stop = min(start + CHUNK, len(X_train))
            d = pairwise_distances(X_train[start:stop], X_train, metric=self.metric)
            d[np.arange(stop - start), np.arange(start, stop)] = np.inf   # un punto non e' vicino di se stesso
            radius[start:stop] = np.partition(d, self.k - 1, axis=1)[:, self.k - 1]
        self.radius = radius
        return self

    def indegree(self, X):
        # Per ogni punto nuovo: quanti punti di training lo hanno dentro il loro raggio.
        # Anche qui si lavora a blocchi di punti nuovi.
        result = np.empty(len(X), dtype=int)
        for start in range(0, len(X), CHUNK):
            stop = min(start + CHUNK, len(X))
            d = pairwise_distances(self.X_train, X[start:stop], metric=self.metric)   # righe = training, colonne = punti nuovi
            result[start:stop] = (d <= self.radius[:, None]).sum(axis=0)
        return result

    def score(self, X):
        """Punteggio di anomalia: piu' e' alto, piu' il punto e' sospetto."""
        return -self.indegree(X)


# ------------------------------------------------------------------
# VALUTAZIONE (stessa funzione del prof)
# ------------------------------------------------------------------
results = []


def evaluate_model(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    scores = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "FPR": fp / (fp + tn) if fp + tn else 0,
        "FNR": fn / (fn + tp) if fn + tp else 0,
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

    results.append(scores)
    print(pd.Series(scores))

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Normal", "Injection"],
                yticklabels=["Normal", "Injection"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(name)
    plt.tight_layout()
    plt.show()
    return scores


if __name__ == "__main__":
    # --------------------------------------------------------------
    # CARICAMENTO DEI TRE FILE (training, validation, test) E SCALING
    # --------------------------------------------------------------
    train, val, test = load_split()
    features = select_features(train)

    X_train_raw, y_train = train[features], train[TARGET_COLUMN].astype(int).values
    X_val_raw, y_val = val[features], val[TARGET_COLUMN].astype(int).values
    X_test_raw, y_test = test[features], test[TARGET_COLUMN].astype(int).values

    print("Features:", features)
    n_tot = len(train) + len(val) + len(test)
    n_inj = int(y_train.sum() + y_val.sum() + y_test.sum())
    print("Righe totali:", n_tot, "| percentuale di injection:", round(100 * n_inj / n_tot, 1), "%")

    # lo scaler impara media e deviazione solo dal training
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    print("Train:", X_train.shape, np.bincount(y_train))
    print("Validation:", X_val.shape, np.bincount(y_val))
    print("Test:", X_test.shape, np.bincount(y_test))

    # --------------------------------------------------------------
    # ODIN unsupervised: il grafo kNN si costruisce su tutto il training
    # (injection comprese), senza usare le etichette
    # --------------------------------------------------------------
    odin = ODIN(k=K, metric=METRIC).fit(X_train)   # nessuna etichetta usata

    odin_val_scores = odin.score(X_val)
    odin_test_scores = odin.score(X_test)

    # Soglia scelta sui punteggi di validation, senza usare y_val
    odin_threshold = np.quantile(odin_val_scores, 1 - ALERT_FRACTION)
    y_pred_odin = (odin_test_scores >= odin_threshold).astype(int)

    name = "ODIN unsupervised (K=%d, %s)" % (K, METRIC)
    print("\n==========", name, "==========")
    print("Soglia: indegree <=", int(-odin_threshold), "-> anomalia")
    evaluate_model(y_test, y_pred_odin, name)