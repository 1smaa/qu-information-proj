import os
import numpy as np
from tqdm import tqdm
# ======== CONFIG ========
DIR = "data"
CLOCK = 82e-12   # 82 ps per tick

# measured channel delays (in seconds)
DELAY_1 = -0.129e-9
DELAY_2 = 0.369e-9

# convert delays to integer ticks
DELAY_1_TICKS = int(DELAY_1 / CLOCK)
DELAY_2_TICKS = int(DELAY_2 / CLOCK)

# coincidence half-window (±)
WINDOW_NS = 1.0
WINDOW_TICKS = int((WINDOW_NS * 1e-9) / CLOCK)
# ========================


class Event:
    def __init__(self, ch: int, timestamp: int) -> None:
        self._ch = ch
        self._ts = timestamp

    def get_ch(self) -> int:
        return self._ch

    def get_timestamp(self) -> int:
        return self._ts

    def offset(self, ticks: int) -> None:
        self._ts += ticks


def load_events(dir: str) -> list["Event"]:
    """Load all events from directory and apply per-channel delay correction."""
    events = []
    for entry in os.listdir(dir):
        path = os.path.join(dir, entry)
        print(f"Examining file: {path}")
        with open(path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()[6:]
        for line in lines:
            timestamp, ch = line.split(";")
            timestamp = int(timestamp.strip())
            ch = int(ch.strip())
            e = Event(ch, timestamp)

            # --- apply delay compensation ---
            if ch == 1:
                e.offset(DELAY_1_TICKS)
            elif ch == 2:
                e.offset(DELAY_2_TICKS)
            # channel 3 (herald) unchanged
            events.append(e)

    # sort once globally
    events.sort(key=lambda e: e.get_timestamp())
    return events


def events_by_channel(events: list["Event"], ch: int) -> list["Event"]:
    return [e for e in events if e.get_ch() == ch]


def find_triple_coincidences(events, ch1=1, ch2=2, chh=3, window_ticks=WINDOW_TICKS):
    """Count coincidences between herald (chh) and ch1, ch2 within ±window_ticks."""
    her = events_by_channel(events, chh)
    ch1_list = events_by_channel(events, ch1)
    ch2_list = events_by_channel(events, ch2)

    her_ts = np.array([h.get_timestamp() for h in her], dtype=np.int64)
    ch1_ts = np.array([e.get_timestamp() for e in ch1_list], dtype=np.int64)
    ch2_ts = np.array([e.get_timestamp() for e in ch2_list], dtype=np.int64)

    n3 = len(her_ts)
    n13 = 0
    n23 = 0
    n123 = 0
    triples = []

    i1 = i2 = 0
    N1, N2 = len(ch1_ts), len(ch2_ts)
    iterator = tqdm(her_ts, desc="Finding triple coincidences", unit="heralds")
    for h_ts in iterator:
        # search for ch1 within window
        while i1 < N1 and ch1_ts[i1] < h_ts - window_ticks:
            i1 += 1
        j1 = i1
        ch1_hits = []
        while j1 < N1 and ch1_ts[j1] <= h_ts + window_ticks:
            ch1_hits.append(int(ch1_ts[j1]))
            j1 += 1

        # search for ch2 within window
        while i2 < N2 and ch2_ts[i2] < h_ts - window_ticks:
            i2 += 1
        j2 = i2
        ch2_hits = []
        while j2 < N2 and ch2_ts[j2] <= h_ts + window_ticks:
            ch2_hits.append(int(ch2_ts[j2]))
            j2 += 1

        if ch1_hits:
            n13 += 1
        if ch2_hits:
            n23 += 1
        if ch1_hits and ch2_hits:
            n123 += 1
            triples.append((int(h_ts), ch1_hits, ch2_hits))

    return {
        "n3": n3,
        "n13": n13,
        "n23": n23,
        "n123": n123,
        "triples": triples,
    }


def main():
    events = load_events(DIR)
    result = find_triple_coincidences(events, ch1=1, ch2=2, chh=3)

    print("\n--- Heralded Coincidence Summary ---")
    print(f"Heralds (n3):        {result['n3']}")
    print(f"Herald–Ch1 (n13):    {result['n13']}")
    print(f"Herald–Ch2 (n23):    {result['n23']}")
    print(f"Triple (n123):       {result['n123']}")
    alpha=(result["n123"]*result["n3"])/(result["n13"]*result["n23"])
    sigma=alpha*np.sqrt(sum([1/result[key] for key in result.keys() if key!="triples"]))
    print(f"Alpha:       {alpha}+-{sigma}")
    if result["triples"]:
        print("\nFirst 3 triple coincidences (herald_ts, ch1_hits, ch2_hits):")
        for t in result["triples"][:3]:
            print(t)


if __name__ == "__main__":
    main()
