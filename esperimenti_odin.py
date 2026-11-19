"""
Prove di modifica su ODIN unsupervised (dataset con le injection).

Si parte dalla configurazione di odin.py (K = 10, soglia = 20%) e si
cambia UN parametro alla volta, per vedere come cambiano le metriche:
  1) K, il numero di vicini del grafo
  2) la soglia, cioe' la frazione di righe segnalate come anomale
  3) i dati usati per costruire il grafo: invece del training con le injection,
     un periodo di funzionamento normale raccolto a parte (normal_run.csv).
     Anche qui nessuna etichetta viene usata nel training.
  4) la distanza usata per decidere quali punti sono vicini
     (euclidea, Manhattan, coseno, Chebyshev), per ogni valore di K.

Per ogni prova si misurano anche i tempi di addestramento e di test
e l'AUC (qualita' del punteggio indipendentemente dalla soglia).

Output: tabelle nel terminale, risultati_esperimenti.csv, grafico_K.png,
grafico_distanze.png e, se c'e' normal_run.csv, grafico_confronto.png
"""
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef, roc_auc_score)

from odin import ODIN, RANDOM_STATE, load_split, select_features   # stessi file e parametri di odin.py

BASE_K = 10
BASE_ALERT_FRACTION = 0.20
K_VALUES = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
METRICS = ["euclidean", "manhattan", "cosine", "chebyshev"]   # distanze provate nella prova 4
NORMAL_FILE = "normal_run.csv"   # raccolta fatta solo con monitor.py, senza injection (prova 3)


def run_odin(X_train, X_val, X_test, y_test, k, alert_fraction, metric="euclidean"):
    """Esegue ODIN con K, soglia e distanza indicati e restituisce le metriche sul test."""
    t0 = time.perf_counter()
    odin = ODIN(k=k, metric=metric).fit(X_train)
    train_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    test_scores = odin.score(X_test)
    test_ms = (time.perf_counter() - t0) * 1000

    # soglia scelta sui punteggi di validation, senza usare le etichette
    threshold = np.quantile(odin.score(X_val), 1 - alert_fraction)
    y_pred = (test_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "Soglia (indegree <=)": int(-threshold),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "AUC": roc_auc_score(y_test, test_scores),
        "Train ms": round(train_ms, 1),
        "Test ms": round(test_ms, 1),
    }


def plot_k(table, filename="grafico_K.png"):
    """Grafico di MCC e F1 al variare di K."""
    k_values = table["K"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(k_values, table["MCC"], color="#2a78d6", linewidth=2, marker="o", label="MCC")
    ax.plot(k_values, table["F1"], color="#eb6834", linewidth=2, marker="o", label="F1")
    ax.set_xscale("log")
    ax.set_xticks(k_values)
    ax.set_xticklabels(k_values)
    ax.set_xlabel("K (numero di vicini)")
    ax.set_ylabel("Valore della metrica")
    ax.set_title("ODIN unsupervised Dati Misti: effetto di K", loc="left")
    ax.grid(axis="y", color="#e5e5e5")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.show()


def plot_confronto(table_misti, table_normale, filename="grafico_confronto.png"):
    """MCC al variare di K: training su dati misti (prova 1) contro training su periodo normale (prova 3)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(table_misti["K"], table_misti["MCC"], color="#2a78d6", linewidth=2, marker="o",
            label="Training su dati misti (con injection)")
    ax.plot(table_normale["K"], table_normale["MCC"], color="#1f2937", linewidth=2, marker="o",
            label="Training su periodo normale")
    ax.set_xscale("log")
    k_all = sorted(set(table_misti["K"]) | set(table_normale["K"]))   # tutti i K usati da almeno una linea
    ax.set_xticks(k_all)
    ax.set_xticklabels(k_all)
    ax.set_xlabel("K (numero di vicini)")
    ax.set_ylabel("MCC")
    ax.set_title("ODIN: effetto dei dati usati per il training", loc="left")
    ax.grid(axis="y", color="#e5e5e5")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.show()


def plot_distanze(table, filename="grafico_distanze.png"):
    """MCC al variare di K, una linea per ogni distanza (prova 4)."""
    colors = {"euclidean": "#2a78d6", "manhattan": "#eb6834", "cosine": "#1baa7d", "chebyshev": "#1f2937"}
    names = {"euclidean": "Euclidea", "manhattan": "Manhattan", "cosine": "Coseno", "chebyshev": "Chebyshev"}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for metric in METRICS:
        t = table[table["Distanza"] == metric]
        ax.plot(t["K"], t["MCC"], color=colors[metric], linewidth=2, marker="o", label=names[metric])
    k_all = sorted(set(table["K"]))
    ax.set_xscale("log")
    ax.set_xticks(k_all)
    ax.set_xticklabels(k_all)
    ax.set_xlabel("K (numero di vicini)")
    ax.set_ylabel("MCC")
    ax.set_title("ODIN unsupervised: effetto della distanza", loc="left")
    ax.grid(axis="y", color="#e5e5e5")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.show()


if __name__ == "__main__":
    # Stessi tre file di odin.py (creati da split_dataset.py) e stesso scaling
    train, val, test = load_split()
    features = select_features(train)
    print("Feature usate:", features)

    X_train_raw, y_train = train[features].values, train["label"].astype(int).values
    X_val_raw, y_val = val[features].values, val["label"].astype(int).values
    X_test_raw, y_test = test[features].values, test["label"].astype(int).values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # Prova 1: cambio K, soglia fissa al 20%
    rows = []
    for k in K_VALUES:
        if k >= len(X_train):
            print("K =", k, "saltato: il training ha solo", len(X_train), "righe")
            continue
        rows.append({"K": k, **run_odin(X_train, X_val, X_test, y_test, k, BASE_ALERT_FRACTION)})
    table_k = pd.DataFrame(rows)
    print("\n===== PROVA 1: variazione di K (soglia = 20%) =====")
    print(table_k.set_index("K").round(3).to_string())

    # Prova 2: cambio la soglia, K fisso a 10
    rows = []
    for af in [0.05, 0.10, 0.20, 0.30]:
        rows.append({"Segnalate": "%d%%" % (af * 100),
                     **run_odin(X_train, X_val, X_test, y_test, BASE_K, af)})
    table_soglia = pd.DataFrame(rows)
    print("\n===== PROVA 2: variazione della soglia (K = 10) =====")
    print(table_soglia.set_index("Segnalate").round(3).to_string())

    # Prova 3: il grafo si costruisce su un periodo di funzionamento normale
    # (raccolta fatta solo con monitor.py, senza injection), senza usare etichette.
    # Cambia solo il training: validation.csv e test.csv sono gli stessi delle altre prove,
    # quindi i risultati sono confrontabili.
    tables = [table_k.assign(Prova="K"), table_soglia.assign(Prova="Soglia")]
    table_normale = None
    if os.path.exists(NORMAL_FILE):
        normal = pd.read_csv(NORMAL_FILE)
        X_normal_raw = normal[features].values

        # lo scaler impara media e deviazione dal nuovo training, cioe' dal periodo normale
        scaler_n = StandardScaler()
        Xn_train = scaler_n.fit_transform(X_normal_raw)
        Xn_val = scaler_n.transform(X_val_raw)
        Xn_test = scaler_n.transform(X_test_raw)

        rows = []
        for k in K_VALUES:
            if k >= len(Xn_train):
                print("K =", k, "saltato: il periodo normale ha solo", len(Xn_train), "righe")
                continue
            rows.append({"K": k, **run_odin(Xn_train, Xn_val, Xn_test, y_test, k, BASE_ALERT_FRACTION)})
        table_normale = pd.DataFrame(rows)
        print("\n===== PROVA 3: training su un periodo normale (%d righe, soglia = 20%%) ====="
              % len(Xn_train))
        print(table_normale.set_index("K").round(3).to_string())
        tables.append(table_normale.assign(Prova="Training su periodo normale"))
    else:
        print("\nPROVA 3 saltata: manca il file %s." % NORMAL_FILE)
        print("Fate una raccolta solo con monitor.py (senza injector) e rinominate "
              "Monitored_data.csv in %s." % NORMAL_FILE)

    # Prova 4: cambio la distanza, per ogni K (soglia fissa al 20%, stessi dati della prova 1)
    rows = []
    for metric in METRICS:
        for k in K_VALUES:
            if k >= len(X_train):
                continue
            rows.append({"Distanza": metric, "K": k,
                         **run_odin(X_train, X_val, X_test, y_test, k, BASE_ALERT_FRACTION, metric)})
    table_dist = pd.DataFrame(rows)
    print("\n===== PROVA 4: variazione della distanza (soglia = 20%) =====")
    print(table_dist.set_index(["Distanza", "K"]).round(3).to_string())
    tables.append(table_dist.assign(Prova="Distanza"))

    # Salvataggio risultati e grafici
    pd.concat(tables).round(4).to_csv("risultati_esperimenti.csv", index=False)
    print("\nRisultati salvati in risultati_esperimenti.csv")
    if table_normale is not None:
        plot_confronto(table_k, table_normale)
    plot_distanze(table_dist)
    plot_k(table_k)