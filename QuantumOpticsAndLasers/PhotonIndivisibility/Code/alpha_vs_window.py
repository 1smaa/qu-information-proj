import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

# ======== CONFIG ========
FILE1 = "data/TimeTags.txt"     # <-- your first file
FILE2 = "data/TimeTags_6.txt"   # <-- your second file
CLOCK = 82e-12                  # 82 ps per tick

DELAY_1 = -0.129e-9
DELAY_2 = 0.369e-9
DELAY_1_TICKS = int(DELAY_1 / CLOCK)
DELAY_2_TICKS = int(DELAY_2 / CLOCK)

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
# =========================


class Event:
    def __init__(self, ch: int, timestamp: int):
        self._ch = ch
        self._ts = timestamp

    def get_ch(self): return self._ch
    def get_timestamp(self): return self._ts
    def offset(self, ticks: int): self._ts += ticks


def load_events_from_file(file_path: str):
    """Load events from a single file and apply per-channel delay correction."""
    events = []
    print(f"Reading {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[6:]  # skip header lines if present

    for line in lines:
        timestamp, ch = line.split(";")
        timestamp = int(timestamp.strip())
        ch = int(ch.strip())
        e = Event(ch, timestamp)

        if ch == 1:
            e.offset(DELAY_1_TICKS)
        elif ch == 2:
            e.offset(DELAY_2_TICKS)

        events.append(e)

    events.sort(key=lambda e: e.get_timestamp())
    return events


def events_by_channel(events, ch):
    return [e for e in events if e.get_ch() == ch]


def find_triple_coincidences(events, window_ticks):
    """Count coincidences between herald (ch3) and ch1, ch2 within ±window_ticks."""
    her = events_by_channel(events, 3)
    ch1_list = events_by_channel(events, 1)
    ch2_list = events_by_channel(events, 2)

    her_ts = np.array([h.get_timestamp() for h in her], dtype=np.int64)
    ch1_ts = np.array([e.get_timestamp() for e in ch1_list], dtype=np.int64)
    ch2_ts = np.array([e.get_timestamp() for e in ch2_list], dtype=np.int64)

    n3 = len(her_ts)
    n13 = n23 = n123 = 0
    i1 = i2 = 0
    N1, N2 = len(ch1_ts), len(ch2_ts)

    for h_ts in her_ts:
        while i1 < N1 and ch1_ts[i1] < h_ts - window_ticks:
            i1 += 1
        j1 = i1
        ch1_hits = []
        while j1 < N1 and ch1_ts[j1] <= h_ts + window_ticks:
            ch1_hits.append(ch1_ts[j1])
            j1 += 1

        while i2 < N2 and ch2_ts[i2] < h_ts - window_ticks:
            i2 += 1
        j2 = i2
        ch2_hits = []
        while j2 < N2 and ch2_ts[j2] <= h_ts + window_ticks:
            ch2_hits.append(ch2_ts[j2])
            j2 += 1

        if ch1_hits:
            n13 += 1
        if ch2_hits:
            n23 += 1
        if ch1_hits and ch2_hits:
            n123 += 1

    return {"n3": n3, "n13": n13, "n23": n23, "n123": n123}


def compute_alpha(result):
    n3, n13, n23, n123 = result["n3"], result["n13"], result["n23"], result["n123"]
    alpha = (n123 * n3) / (n23 * n13)
    sigma = alpha * np.sqrt((1 / n123) + (1 / n3) + (1 / n13) + (1 / n23))
    return alpha, sigma


def scan_alpha(file_path, max_ns=10):
    """Scan α vs coincidence window and cache results."""
    base = os.path.splitext(os.path.basename(file_path))[0]
    result_file = os.path.join(RESULT_DIR, f"{base}_alpha_vs_window.csv")

    # If results already exist, just load them
    if os.path.exists(result_file):
        print(f"Loading cached results from {result_file}")
        data = np.loadtxt(result_file, delimiter=",", skiprows=1)
        return data[:, 0], data[:, 1], data[:, 2]

    print(f"\n=== Processing {file_path} ===")
    events = load_events_from_file(file_path)
    windows = np.arange(1, max_ns + 1)
    alphas, sigmas = [], []

    for ns in tqdm(windows, desc="Scanning coincidence windows", unit="ns"):
        window_ticks = int((ns * 1e-9) / CLOCK)
        result = find_triple_coincidences(events, window_ticks)
        alpha, sigma = compute_alpha(result)
        alphas.append(alpha)
        sigmas.append(sigma)

    # Save results to file
    with open(result_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["window_ns", "alpha", "sigma"])
        for w, a, s in zip(windows, alphas, sigmas):
            writer.writerow([w, a, s])

    print(f"Results saved to {result_file}")
    return windows, np.array(alphas), np.array(sigmas)


def main():
    windows1, alphas1, sigmas1 = scan_alpha(FILE1)
    windows2, alphas2, sigmas2 = scan_alpha(FILE2)

    plt.figure(figsize=(8, 5))
    plt.errorbar(windows1, alphas1, yerr=sigmas1, fmt='o-', capsize=3, label='Without Light')
    plt.errorbar(windows2, alphas2, yerr=sigmas2, fmt='s-', capsize=3, label='With Light')
    #plt.axhline(1, color='gray', linestyle='--', label='Classical limit α=1')
    plt.xlabel("Coincidence window (ns)")
    plt.ylabel("α parameter")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
