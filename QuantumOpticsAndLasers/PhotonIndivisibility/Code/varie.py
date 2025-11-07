import os
import numpy as np
import matplotlib.pyplot as plt

DIR="data"
BIN=1e-9
CLOCK=82e-12

DIFF_BIN=BIN/CLOCK

import numpy as np
import matplotlib.pyplot as plt

def time_differences(events, ch_signal, ch_herald, window_ticks=5000):
    sig = [e.get_timestamp() for e in events if e.get_ch() == ch_signal]
    her = [e.get_timestamp() for e in events if e.get_ch() == ch_herald]

    diffs = []
    j = 0
    for t in sig:
        # advance j until we reach timestamps close to t
        while j < len(her) and her[j] < t - window_ticks:
            j += 1
        k = j
        # collect all heralds within the large search window
        while k < len(her) and her[k] <= t + window_ticks:
            diffs.append(her[k] - t)
            k += 1
    return np.array(diffs)


def plot_time_correlation(events, pairs, bins=800, window_ticks=5000):
    plt.figure(figsize=(8,5))
    for (ch_sig, ch_her) in pairs:
        diffs = time_differences(events, ch_sig, ch_her, window_ticks)
        if len(diffs) == 0:
            print(f"No data for channels {ch_sig}-{ch_her}")
            continue
        plt.hist(diffs, bins=bins, histtype="step", label=f"({ch_sig},{ch_her})", density=False)

    plt.title("Time-Correlation Histogram")
    plt.xlabel("Δt (clock ticks)  →  convert using CLOCK = 82 ps/tick")
    plt.ylabel("Counts")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()




class Event(object):
    def __init__(self,ch: int,timestamp: int) -> None:
        self._ch=ch
        self._ts=timestamp
        
    def get_ch(self) -> int:
        return self._ch
    
    def get_timestamp(self) -> int:
        return self._ts
    
    def set_ch(self,ch: int) -> int:
        old=self._ch
        self._ch=ch
        return old
    
    def set_timestamp(self,timestamp: int) -> int:
        old=self._ts
        self._ts=timestamp
        return old
    
    def __sub__(self, E2: any) -> int:
        return abs(self._ts-E2.get_timestamp())
    
def match(E1: Event,E2: Event) -> bool:
    return (E1-E2)<DIFF_BIN

def load_events(dir: str) -> list[Event]:
    events=[]
    for entry in os.listdir(dir):
        path=os.path.join(dir,entry)
        print(f"Examining file: {path}")
        with open(path,mode="r",encoding="utf-8") as f:
            lines=f.readlines()
        lines=lines[6:]
        for line in lines:
            timestamp,ch=line.split(";")
            timestamp=int(timestamp.strip())
            ch=int(ch.strip())
            events.append(Event(ch,timestamp))
    return events

def coincidences(events: list["Event"], ch_1: int, ch_2: int) -> list[tuple["Event", "Event"]]:
    ch1_events = [e for e in events if e.get_ch() == ch_1]
    ch2_events = [e for e in events if e.get_ch() == ch_2]

    coinc = []
    j=0
    for i,event in enumerate(ch1_events):
        while j<len(ch2_events) and (ch2_events[j].get_timestamp()<event.get_timestamp() or match(ch2_events[j],event)):
            if match(ch2_events[j],event):
                coinc.append((event,ch2_events[j],))
            j+=1
    return ch1_events,ch2_events,coinc
        

def main() -> None:
    events=load_events(DIR)
    events=sorted(events,key=lambda x:x.get_timestamp())
    c1,c3,c_ht=coincidences(events,1,3)
    #c1,c2,c_hr=coincidences(events,1,2)
    print(len(c1),len(c3),len(c_ht))
    c3,c2,c_ht=coincidences(events,3,2)
    print(len(c3),len(c2),len(c_ht))
    c1,c2,c_ht=coincidences(events,1,2)
    print(len(c1),len(c2),len(c_ht))


import numpy as np
import matplotlib.pyplot as plt

CLOCK = 82e-12   # 82 ps per tick
MAX_DELAY_NS = 20  # ±200 ns window
BIN_NS = 0.1         # histogram bin width in ns

def correlation(events, ch_a, ch_b, max_delay_ns=MAX_DELAY_NS, bin_ns=BIN_NS):
    """Compute time-correlation histogram between two channels."""
    ta = np.array([e.get_timestamp() for e in events if e.get_ch() == ch_a])
    tb = np.array([e.get_timestamp() for e in events if e.get_ch() == ch_b])
    ta.sort(); tb.sort()

    max_delay_ticks = int(max_delay_ns * 1e-9 / CLOCK)
    bin_ticks = int(bin_ns * 1e-9 / CLOCK)
    bins = np.arange(-max_delay_ticks, max_delay_ticks + bin_ticks, bin_ticks)
    diffs = []

    j = 0
    for t in ta:
        while j < len(tb) and tb[j] < t - max_delay_ticks:
            j += 1
        k = j
        while k < len(tb) and tb[k] <= t + max_delay_ticks:
            diffs.append(tb[k] - t)
            k += 1

    hist, edges = np.histogram(diffs, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    times_ns = centers * CLOCK * 1e9
    return times_ns, hist


def plot_correlations(events, pairs=[(2,3), (1,3)], logy=False):
    plt.figure(figsize=(9,6))
    colors = ['tab:blue', 'tab:orange']
    linestyles = ['-', '--']

    for i, (a, b) in enumerate(pairs):
        t, h = correlation(events, a, b)
        #h = h / np.max(h)  # normalize for visual comparison

        # --- Find FWHM ---
        peak_idx = np.argmax(h)
        half_max = 0.5
        left_idx = np.where(h[:peak_idx] < half_max)[0]
        right_idx = np.where(h[peak_idx:] < half_max)[0]

        if len(left_idx) > 0 and len(right_idx) > 0:
            left_half = t[left_idx[-1]]
            right_half = t[peak_idx + right_idx[0]]
            fwhm = right_half - left_half
        else:
            fwhm = np.nan  # in case no clear drop found

        # --- Plotting ---
        plt.plot(t, h, label=f"G({a},{b})(τ)", 
                 color=colors[i % len(colors)], 
                 linestyle=linestyles[i % len(linestyles)], linewidth=1.3)
        plt.axvline(t[peak_idx], color=colors[i % len(colors)], linestyle=':', alpha=0.6)

        plt.text(t[peak_idx], h[peak_idx]*1.05, 
                 f"{t[peak_idx]:.2f} ns", color=colors[i % len(colors)],
                 fontsize=9, ha='center')
        
        if not np.isnan(fwhm):
            plt.text(t[peak_idx], h[peak_idx]*0.5, 
                     f"Bell Width = {fwhm:.2f} ns", color=colors[i % len(colors)],
                     fontsize=9, ha='center', va='bottom')

        print(f"Channels ({a},{b}) → Peak at {t[peak_idx]:.3f} ns, FWHM ≈ {fwhm:.3f} ns")

    plt.title("Time-Correlation Functions G(τ)", fontsize=14)
    plt.xlabel("Delay τ (ns)", fontsize=12)
    plt.ylabel("Normalized coincidence counts", fontsize=12)
    if logy:
        plt.yscale('log')
        plt.ylabel("Normalized coincidence counts (log scale)", fontsize=12)

    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    
def heralded_anticorrelation(events, ch1, ch2, chh, window_ns=5):
    window_ticks = int((window_ns * 1e-9) / CLOCK)
    heralds = [e.get_timestamp() for e in events if e.get_ch() == chh]
    ch1_times = [e.get_timestamp() for e in events if e.get_ch() == ch1]
    ch2_times = [e.get_timestamp() for e in events if e.get_ch() == ch2]

    n13 = n23 = n123 = 0
    j1 = j2 = 0
    for h in heralds:
        # Find if signal 1 or 2 is within window
        has1 = False
        has2 = False

        while j1 < len(ch1_times) and ch1_times[j1] < h - window_ticks:
            j1 += 1
        k1 = j1
        while k1 < len(ch1_times) and ch1_times[k1] <= h + window_ticks:
            has1 = True
            k1 += 1

        while j2 < len(ch2_times) and ch2_times[j2] < h - window_ticks:
            j2 += 1
        k2 = j2
        while k2 < len(ch2_times) and ch2_times[k2] <= h + window_ticks:
            has2 = True
            k2 += 1

        if has1: n13 += 1
        if has2: n23 += 1
        if has1 and has2: n123 += 1

    n3 = len(heralds)
    g2 = (n123 * n3) / (n13 * n23) if n13 and n23 else 0
    print(f"n3={n3}, n13={n13}, n23={n23}, n123={n123}")
    print(f"Heralded g2(0) = {g2:.4f}")
    return g2



# Example usage in your main:
if __name__ == "__main__":
    events = load_events(DIR)
    events = sorted(events, key=lambda e: e.get_timestamp())
    plot_correlations(events)
    heralded_anticorrelation(events,1,2,3)
