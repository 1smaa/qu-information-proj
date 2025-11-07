import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DATA_FOLDER = 'data/laser 633 nm with SPAD/Spinning wheel'
MACHINE_UNIT = 81e-12  # 81 ps
SKIP_SHORT_DIFF = 3900  # filter close timestamps if needed

def extract_bits_from_timestamps(timestamps):
    """Generate random bits based on difference comparisons."""
    timestamps = np.array(timestamps, dtype=np.int64)
    diffs = np.diff(timestamps)

    # Optionally filter out unrealistically short intervals (dead-time, afterpulsing)
    diffs = diffs[diffs > SKIP_SHORT_DIFF]

    bits = (diffs[:-1] >= diffs[1:]).astype(np.uint8)  # 0 if D_i < D_i+1 else 1
    return bits

def bits_to_bytes(bits):
    """Group bits into bytes (8 bits per value)."""
    n_bytes = len(bits) // 8
    bits = bits[:n_bytes * 8]
    byte_values = np.packbits(bits)
    return byte_values

def main():
    all_bytes = []

    for entry in os.listdir(DATA_FOLDER):
        print(f"Processing file: {entry}")
        file_name = os.path.join(DATA_FOLDER, entry)
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()[1:]

        timestamps = [int(line.strip().split(",")[0]) for line in lines]
        bits = extract_bits_from_timestamps(timestamps)
        byte_values = bits_to_bytes(bits)
        all_bytes.append(byte_values)

    # Combine all byte sequences
    all_bytes = np.concatenate(all_bytes)
    print(f"Generated {len(all_bytes)} bytes of random data.")

    # Group every 3 bytes into (x, y, z)
    n_points = len(all_bytes) // 3
    coords = all_bytes[:n_points * 3].reshape(-1, 3)

    # Normalize to [0, 1] for plotting
    coords = coords / 255.0

    # Plot random fog
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c=coords, s=1.5, alpha=0.6, linewidths=0)
    ax.set_xlabel("Byte 1 / 255")
    ax.set_ylabel("Byte 2 / 255")
    ax.set_zlabel("Byte 3 / 255")
    plt.tight_layout()
    plt.show()
    plt.savefig("QRNG_633nm_Thermal.png")

if __name__ == "__main__":
    main()
