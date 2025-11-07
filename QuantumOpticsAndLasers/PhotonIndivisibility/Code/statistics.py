import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# --- PARAMETERS ---
FILE = r"C:\Users\Ismaele\Desktop\Elaborato2\data\TimeTags_5.txt"
DIR_MODE = False
BIN = 1e-3         # 10 µs bin width
CLOCK = 81e-12       # 81 ps clock period
CH = 1               # channel to analyze

BIN_CLOCK = int(BIN / CLOCK)  # bin width in clock ticks

# --- LOAD EVENTS ---
def load_events() -> np.ndarray:
    with open(FILE, mode="r", encoding="utf-8") as f:
        lines = f.readlines()[6:]

    timestamps = []
    for line in lines:
        timestamp, ch = line.split(";")
        timestamp = int(timestamp.strip())
        ch = int(ch.strip())
        if ch == CH:
            timestamps.append(timestamp)

    arr = np.array(timestamps, dtype=np.int64)
    arr.sort()
    arr = arr - arr[0]  # shift to zero
    return arr


# --- MAIN ---
def main() -> None:
    events = load_events()
    total_time_ticks = events[-1]
    n_bins = int(total_time_ticks // BIN_CLOCK) + 1

    # Count photons per bin
    counts = np.zeros(n_bins, dtype=int)
    for event in events:
        idx = event // BIN_CLOCK
        if idx < n_bins:
            counts[int(idx)] += 1

    # Histogram of counts per bin
    hist, edges = np.histogram(counts, bins=range(int(counts.max()) + 2))

    # Expected Poisson distribution
    mean_counts = np.mean(counts)
    k_values = np.arange(0, int(counts.max()) + 1)
    poisson_dist = poisson.pmf(k_values, mean_counts) * np.sum(hist)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(k_values, hist, width=0.8, alpha=0.6, label="Measured distribution")
    plt.plot(k_values, poisson_dist, "r--", lw=2, label=f"Poisson (μ={mean_counts:.2f})")
    plt.xlabel("Counts per bin")
    plt.ylabel("Occurrences")
    plt.title(f"Photon Count Statistics (Channel {CH})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Mean counts per bin = {mean_counts:.3f}")
    print(f"Variance = {np.var(counts):.3f}")
    print(f"Fano factor = {np.var(counts)/mean_counts:.3f} (≈1 → Poissonian)")


if __name__ == "__main__":
    main()
