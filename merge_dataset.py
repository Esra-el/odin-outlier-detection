"""
Unisce i dati del monitor (Monitored_data.csv) con i tempi delle injection
(output_folder/inj_info.csv) e crea il dataset etichettato (dataset.csv).

Ogni riga del monitor riceve:
  - label    : 1 se in quel momento era attiva un'injection, 0 altrimenti
  - inj_type : il tag dell'injection attiva (es. 'cpu', 'mem') oppure 'normal'
"""
import csv
import re

MONITOR_FILE = "Monitored_data.csv"
INJ_FILE = "output_folder/inj_info.csv"
OUT_FILE = "dataset.csv"
SAMPLING_INTERVAL_MS = 1000  # deve essere uguale a SAMPLING_INTERVAL di monitor.py (in ms)


def read_injections(filename):
    """Legge inj_info.csv e restituisce una lista di (start, end, tag)."""
    injections = []
    with open(filename, newline="") as f:
        for row in csv.DictReader(f):
            # inj_name e' tipo "[cpu]CPUStressInjection(d3000)": prendo il tag tra parentesi quadre
            match = re.match(r"\[(.*?)\]", row["inj_name"])
            tag = match.group(1) if match and match.group(1) else row["inj_name"]
            injections.append((int(row["start"]), int(row["end"]), tag))
    return injections


def label_row(t, injections):
    """
    Una riga del monitor con tempo t descrive l'ultimo secondo (da t-1000 a t),
    perche' cpu_percent misura l'uso della CPU dall'ultima lettura.
    La riga e' etichettata come injection solo se l'injection copre almeno
    meta' di questo secondo (cosi' non si marcano righe toccate per pochi ms).
    """
    window_start = t - SAMPLING_INTERVAL_MS
    active = []
    for (start, end, tag) in injections:
        overlap = min(t, end) - max(window_start, start)
        if overlap >= SAMPLING_INTERVAL_MS / 2:
            active.append(tag)
    if active:
        return 1, "+".join(sorted(set(active)))
    return 0, "normal"


if __name__ == "__main__":
    injections = read_injections(INJ_FILE)

    with open(MONITOR_FILE, newline="") as f:
        rows = list(csv.DictReader(f))

    n_inj = 0
    for row in rows:
        row["label"], row["inj_type"] = label_row(int(row["time"]), injections)
        n_inj += row["label"]

    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("Injection lette:", len(injections))
    print("Righe totali:", len(rows), "| con injection:", n_inj, "| normali:", len(rows) - n_inj)
    print("Dataset salvato in", OUT_FILE)
