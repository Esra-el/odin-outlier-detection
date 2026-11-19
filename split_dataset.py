"""
Divide dataset.csv in training, validation e test (70% / 15% / 15%) e salva
le tre parti in tre file separati nella cartella split/:

    split/train.csv        -> su questo ODIN costruisce il grafo (prove 1, 2 e 4)
    split/validation.csv   -> serve a scegliere la soglia
    split/test.csv         -> serve a calcolare le metriche finali

Si lancia UNA volta, dopo merge_dataset.py. Da quel momento odin.py ed
esperimenti_odin.py leggono questi tre file, quindi la divisione e' sempre la
stessa in tutte le prove. Nella prova 3 (training su normal_run.csv) cambia solo
il training: validation e test restano questi.

La divisione e' casuale ma stratificata: ogni parte ha la stessa percentuale di
righe con injection. random_state fisso = stessa divisione a ogni esecuzione.
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_FILE = "dataset.csv"
SPLIT_FOLDER = "split"
RANDOM_STATE = 42


def describe(name, df):
    """Stampa quante righe e quante injection ci sono in una parte."""
    n_inj = int(df["label"].sum())
    print("%-11s %6d righe | %5d con injection (%.1f%%)"
          % (name, len(df), n_inj, 100 * n_inj / len(df)))
    if "inj_type" in df.columns:
        counts = df["inj_type"].value_counts()
        print("            " + " | ".join("%s: %d" % (t, c) for t, c in counts.items()))


if __name__ == "__main__":
    data = pd.read_csv(DATASET_FILE)

    # 70% training, 30% resto; poi il resto diviso a meta': 15% validation, 15% test
    train, temp = train_test_split(data, test_size=0.3, random_state=RANDOM_STATE,
                                   stratify=data["label"])
    validation, test = train_test_split(temp, test_size=0.5, random_state=RANDOM_STATE,
                                        stratify=temp["label"])

    os.makedirs(SPLIT_FOLDER, exist_ok=True)
    train.to_csv(os.path.join(SPLIT_FOLDER, "train.csv"), index=False)
    validation.to_csv(os.path.join(SPLIT_FOLDER, "validation.csv"), index=False)
    test.to_csv(os.path.join(SPLIT_FOLDER, "test.csv"), index=False)

    print("Dataset diviso e salvato nella cartella '%s':\n" % SPLIT_FOLDER)
    describe("Training", train)
    describe("Validation", validation)
    describe("Test", test)