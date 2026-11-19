# my_normal_run.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("normal_run.csv")
df["minutes"] = (df["time"] - df["time"].iloc[0]) / 60000  # minutes since the start

cols = {
    "cpu_percent": "CPU usage (%)",
    "mem_percent": "Memory usage (%)",
    "disk_read_mb_s": "Disk read (MB/s)",
    "net_recv_kb_s": "Network received (KB/s)",
}

fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
for ax, (col, title) in zip(axes, cols.items()):
    ax.plot(df["minutes"], df[col], linewidth=0.7)
    ax.set_title(title)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("Minutes since recording started")

fig.suptitle("My laptop: normal activity (no injections)")
fig.tight_layout()
fig.savefig("my_normal_run.png", dpi=150)
print("Saved my_normal_run.png")