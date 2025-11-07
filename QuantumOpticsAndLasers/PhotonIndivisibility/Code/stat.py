import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# --- PARAMETERS ---
FILE = r"C:\Users\Ismaele\Desktop\Elaborato2\data\TimeTags_3.txt"
DIR="data"
DIR_MODE = False
BIN = 1e-3         # bin width in seconds (10 µs)
CLOCK = 81e-12       # 81 ps clock period
CHANNELS = [1, 2, 3] # channels to analyze
BIN_CLOCK = int(BIN / CLOCK)


# --- LOAD EVENTS ---
def load_events() -> list[tuple[int, int]]:
    """Load all events (timestamp, channel)."""
    if DIR_MODE:
        lines = []
        for entry in os.listdir(DIR):
            with open(os.path.join(DIR, entry), mode="r", encoding="utf-8") as f:
                lines += f.readlines()[6:]
    else:
        with open(FILE, mode="r", encoding="utf-8") as f:
            lines = f.readlines()[6:]

    events = []
    for line in lines:
        timestamp, ch = line.split(";")
        events.append((int(timestamp.strip()), int(ch.strip())))
    return events


# --- COUNT PHOTONS PER BIN ---
def count_per_bin(events, ch):
    """Return array of photon counts per bin for one channel."""
    ts = np.array([t for t, c in events if c == ch], dtype=np.int64)
    ts.sort()
    ts -= ts[0]
    n_bins = int(ts[-1] // BIN_CLOCK) + 1
    counts = np.zeros(n_bins, dtype=int)
    for t in ts:
        idx = int(t // BIN_CLOCK)
        if idx < n_bins:
            counts[idx] += 1
    return counts


# --- MAIN ---
def main():
    events = load_events()

    plt.figure(figsize=(8, 5))
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for i, ch in enumerate(CHANNELS):
        counts = count_per_bin(events, ch)
        mean_c = np.mean(counts)
        var_c = np.var(counts)
        fano = var_c / mean_c if mean_c != 0 else np.nan

        # histogram of observed counts per bin
        k_values = np.arange(0, counts.max() + 1)
        hist, _ = np.histogram(counts, bins=np.arange(-0.5, counts.max() + 1.5))

        # Fit Poisson curve using mean
        fit = poisson.pmf(k_values, mean_c) * np.sum(hist)

        # plot
        plt.bar(
            k_values + 0.2 * i,
            hist,
            width=0.2,
            color=colors[i],
            alpha=0.6,
            label=f"Ch {ch}",
        )
        plt.plot(k_values, fit, color=colors[i], lw=2, ls="--")

        print(f"Channel {ch}: mean={mean_c:.3f}, var={var_c:.3f}, Fano={fano:.3f}")

    plt.xlabel("Counts per bin")
    plt.ylabel("Occurrences")
    plt.title("Photon Count Statistics (All Channels)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
