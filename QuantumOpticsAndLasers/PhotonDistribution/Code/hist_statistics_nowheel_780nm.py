import os
import numpy as np
import matplotlib.pyplot as plt

DATA_FOLDERS = ['data/laser 1550 nm with SNSPD/Spinning wheel']
NAMES = ['1550nm_static_remake']
MACHINE_UNIT = 81e-12
BIN = 1e-3
MACHINE_BIN = BIN / MACHINE_UNIT  # float ratio

def hey(folder: str,name: str) -> None:
    entries = os.listdir(folder)
    bins = []

    for entry in entries:
        #print(f"Parsing entry: {entry}")
        file_name = os.path.join(folder, entry)
        with open(file_name, mode="r", encoding="utf-8") as f:
            lines = f.readlines()[1:]

        timestamps = [int(line.strip().split(",")[0]) for line in lines]
        origin = timestamps[0]
        timestamps = [t - origin for t in timestamps]

        i = 1
        start = 0
        bin_count = 1

        while i < len(timestamps):
            # Ignore pulses that are too close together (SPAD afterpulsing / dead time)
            if timestamps[i] - timestamps[i - 1] < 3900:
                i += 1
                continue

            if timestamps[i] < (start + MACHINE_BIN):
                bin_count += 1
            else:
                bins.append(bin_count)
                bin_count = 1
                start += MACHINE_BIN
                while timestamps[i] > (start + MACHINE_BIN):
                    bins.append(0)
                    start += MACHINE_BIN
            i += 1

        bins.append(bin_count)

    # Convert to numpy array
    bins = np.array(bins)
    print(f"Mean: {np.mean(bins)}, Variance: {np.var(bins)}")
    # Histogram of bin counts
    plt.figure(figsize=(8, 5))
    plt.hist(bins, bins=60, color='steelblue', edgecolor='black', alpha=0.8)
    plt.yscale('log')  # log scale helps show the distribution tail
    plt.xlabel(r"Photon counts per bin ($10ms$ window)")
    plt.ylabel("Frequency (log scale)")
    plt.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"latex/Graphs/{name}_hist_afterpulse.png")
    
def main() -> None:
    for i,path in enumerate(DATA_FOLDERS):
        print(f"Elaborating: {NAMES[i]}")
        hey(path,NAMES[i])

if __name__ == "__main__":
    main()
