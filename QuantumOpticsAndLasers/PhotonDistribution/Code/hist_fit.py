import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import gammaln  # stable log-factorial


# === Configuration ===
DATA_FOLDERS = [
    'data/laser 633 nm with SPAD/Spinning wheel'
]
NAMES = [
    '633nm_spinning'
]

MACHINE_UNIT = 81e-12          # seconds per machine unit
BIN = 10e-6                     # 1 ms window
MACHINE_BIN = BIN / MACHINE_UNIT  # window in machine units


# === Distribution Models ===
def poisson_model(k, lamb, scale):
    """Numerically stable Poisson model with adjustable λ and scale."""
    log_p = k * np.log(lamb) - lamb - gammaln(k + 1)
    return scale * np.exp(log_p)


def thermal_model(k, mean, scale):
    """Thermal (Bose–Einstein) model with adjustable mean and scale."""
    return scale * (mean ** k) / ((1 + mean) ** (k + 1))


# === Main fitting routine ===
def hey(folder: str, name: str) -> None:
    entries = os.listdir(folder)
    counts_all = []

    for entry in entries:
        print(f"Parsing entry: {entry}")
        file_path = os.path.join(folder, entry)

        with open(file_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()[1:]  # skip header

        timestamps = [int(line.strip().split(",")[0]) for line in lines]
        origin = timestamps[0]
        timestamps = [t - origin for t in timestamps]

        # Bin photon counts
        i = 1
        start = 0
        bin_count = 1

        while i < len(timestamps):
            # Skip short separations (detector dead time)
            if timestamps[i] - timestamps[i - 1] < 3900:
                i += 1
                continue

            if timestamps[i] < (start + MACHINE_BIN):
                bin_count += 1
            else:
                counts_all.append(bin_count)
                bin_count = 1
                start += MACHINE_BIN
                while timestamps[i] > (start + MACHINE_BIN):
                    counts_all.append(0)
                    start += MACHINE_BIN
            i += 1

        counts_all.append(bin_count)

    # === Statistics ===
    counts_all = np.array(counts_all)
    mean = np.mean(counts_all)
    var = np.var(counts_all)
    fano = var / mean
    print(f"[{name}] Mean={mean:.3f}, Var={var:.3f}, Fano={fano:.3f}")

    # === Histogram ===
    values, bin_edges = np.histogram(counts_all, bins=np.arange(np.max(counts_all) + 2))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # === Fit using curve_fit ===
    if "static" in name.lower():
        popt, _ = curve_fit(poisson_model, bin_centers, values, p0=[mean, np.sum(values)], maxfev=10000)
        lamb_fit, scale_fit = popt
        model_fit = poisson_model(bin_centers, lamb_fit, scale_fit)
        label = f"Poisson fit"
        color = "crimson"
        use_log_scale = False
    else:
        popt, _ = curve_fit(thermal_model, bin_centers, values, p0=[mean, np.sum(values)], maxfev=10000)
        mean_fit, scale_fit = popt
        model_fit = thermal_model(bin_centers, mean_fit, scale_fit)
        label = f"Thermal fit"
        color = "darkorange"
        use_log_scale = True

    # === Normalize for probability comparison ===
    values_norm = values / np.sum(values)
    model_norm = model_fit / np.sum(model_fit)

    # === Plot ===
    plt.figure(figsize=(8, 5))
    plt.bar(bin_centers, values_norm, width=0.9, color='steelblue', alpha=0.7, label="Experimental data")
    plt.plot(bin_centers, model_norm, color=color, lw=2.5, label=label)
    plt.xlabel(r"Photon counts per bin ($10 \ \mu s$ window)")
    plt.ylabel("Probability")

    if use_log_scale:
        plt.yscale('log')
        plt.ylim(1e-6, 1)  # keeps log scale readable
    else:
        plt.ylim(bottom=0)

    plt.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()

    os.makedirs("latex/Graphs", exist_ok=True)
    plt.savefig(f"latex/Graphs/{name}_hist_fit.png", dpi=300)
    plt.close()


# === Main ===
def main() -> None:
    for path, name in zip(DATA_FOLDERS, NAMES):
        print(f"Elaborating: {name}")
        hey(path, name)


if __name__ == "__main__":
    main()
