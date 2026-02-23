# dtleaks.py
# Methods for working with leak events

import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from .dtgen import str2ts, inside

class Leak:
    def __init__(self, pipe: str, start: str, fixed: str, 
                 abrupt: bool, latent: bool, area: str):
        self.pipe = pipe                  # Pipe Id
        self.start = pd.Timestamp(start)  # Starting time
        self.fixed = pd.Timestamp(fixed)  # Time when the leak was fixed
        self.abrupt = abrupt              # True for abrupt leaks, False for incipient leaks
        self.latent = latent              # True if the leak was not fixed until the end of the timeframe considered
        self.area = area                  # Area (DMA)

    def to_dict(self):
        # convert to dictionary
        return {
            "pipe": self.pipe,
            "start": self.start.isoformat(),
            "fixed": self.fixed.isoformat(),
            "abrupt": self.abrupt,
            "latent": self.latent,
            "area": self.area
        }

    @staticmethod
    def from_dict(data):
        # build from dictionary
        return Leak(
            pipe=data["pipe"],
            start=pd.Timestamp(data["start"]),
            fixed=pd.Timestamp(data["fixed"]),
            abrupt=data["abrupt"],
            latent=data["latent"],
            area=data["area"]
        )

    def __str__(self):
        # convert to string for printing
        return f"{self.pipe}: {self.start} — {self.fixed}, abrupt={self.abrupt},latent={self.latent}, area={self.area}"


class LeakSet:
    # List ofleaks, most of the fuctions are self-explanatory
    def __init__(self):
        self.leaks: List[Leak] = []

    def add(self, leak: Leak):
        self.leaks.append(leak)

    def remove(self, pipe: str):
        self.leaks = [l for l in self.leaks if l.pipe != pipe]

    def __iter__(self):
        for leak in self.leaks:
            yield (leak.start, leak.fixed)

    def __len__(self):
        return len(self.leaks)

    def filter(self, area: Optional[str] = None, latent: Optional[bool] = None,
               start_after: Optional[str] = None, start_before: Optional[str] = None,
               end_after: Optional[str] = None, end_before: Optional[str] = None) -> 'LeakSet':
        start_after = str2ts(start_after)
        start_before = str2ts(start_before)
        end_after = str2ts(end_after)
        end_before = str2ts(end_before)
        filtered = LeakSet()
        for leak in self.leaks:
            if area and leak.area != area:
                continue
            if latent is not None and leak.latent != latent:
                continue
            if start_after and leak.start < pd.Timestamp(start_after):
                continue
            if start_before and leak.start > pd.Timestamp(start_before):
                continue
            if end_after and leak.fixed < pd.Timestamp(end_after):
                continue
            if end_before and leak.fixed > pd.Timestamp(end_before):
                continue
            filtered.add(leak)
        return filtered

    def exists_leak(self, t: str) -> bool:
        t = str2ts(t)
        return any(inside(t, l.start, l.fixed) for l in self.leaks)

    def to_dict(self):
        return [leak.to_dict() for leak in self.leaks]

    def to_dataframe(self):
        return pd.DataFrame(self.to_dict())

    def save_csv(self, filename: str):
        self.to_dataframe().to_csv(filename, index=False)

    @staticmethod
    def from_dict(data):
        ls = LeakSet()
        for leak_dict in data:
            ls.add(Leak.from_dict(leak_dict))
        return ls

    @staticmethod
    def load_csv(filename: str) -> 'LeakSet':
        df = pd.read_csv(filename,parse_dates=["start", "fixed"])
        return LeakSet.from_dict(df.to_dict(orient='records'))

    def __str__(self):
        return "\n".join(str(leak) for leak in self.leaks)

    # Visualization methods

    def plot_bands(self, ax=None, color='lightgrey', alpha=0.4, label=None, **kwargs):
        # plot as shaded areas; useful when there are not many overlapping leaks
        for i, leak in enumerate(self.leaks):
            ax.axvspan(leak.start, leak.fixed, color=color, alpha=alpha, 
                       label=label if i == 0 and label else None,  # Only first band gets label
                       **kwargs)
        return ax

    def plot_steps(self, ax=None, color='lightgrey', alpha=0.4, label=None, **kwargs):
        # plot as shaded areas of increasing height
        num_leaks = len(self.leaks)
        ystep = 1/num_leaks
        for i, leak in enumerate(self.leaks):
            ax.axvspan(leak.start, leak.fixed, ymin=0, ymax=ystep*(i+1), color=color, alpha=alpha,
                       label=label if i == 0 and label else None,  # Only first band gets label
                       **kwargs)
        return ax

    def plot_count(self, ax=None, labels=True, color='green', ystep=1, **kwargs):
        # plot as the step function representing the count of active leaks

        events = []
        for i, leak in enumerate(self.leaks):
            events.append((leak.start, 1, leak.pipe))   # leak start
            events.append((leak.fixed, -1, leak.pipe))  # leak end

        events.sort()
        level = 0
        xs, ys = [], []

        for time, change, pipe in events:
            xs.append(time)
            ys.append(level)

            level += ystep * change
            xs.append(time)
            ys.append(level)

            if labels and change == 1:
               # inscription at the beginning of a horizontal segment
               ax.text(time, level + 0.1 * ystep, ' ' + pipe,
                       fontsize=8, verticalalignment='bottom', horizontalalignment='left')

        ax.step(xs, ys, where='post', color=color, **kwargs)
        return ax

    def plot_gantt(self, ax=None, ystep=0.3, labels=True, colormap={False: 'brown', True: 'red'}, **kwargs):
        # plot as the Gantt chart. 
        # By default incipient leaks are shown as brown stripes, and abrupt leaks as red stripes
        
        _, ymax = ax.get_ylim()
  
        y_base = ymax   # indentation from the main drawing

        for i, leak in enumerate(self.leaks):
            start, end = leak.start, leak.fixed
            y_pos = y_base + i * ystep
            color = colormap.get(leak.abrupt, 'gray')

            ax.plot([start, end], [y_pos, y_pos], color=color, linewidth=3, **kwargs)
            ax.vlines([start, end], ymin=0, ymax=y_pos, color=color, linewidth=1, alpha=0.4)
            if labels:
               ax.text(start, y_pos + 0.05, leak.pipe, fontsize=8, verticalalignment='bottom')
        return ax

    def plot(self, method='bands', ax=None, **kwargs):
        # the wrapper for plotting methods
        if ax is None:
            fig, ax = plt.subplots()
        if method == 'bands':
            return self.plot_bands(ax=ax, **kwargs)
        elif method == 'steps':
            return self.plot_steps(ax=ax, **kwargs)
        elif method == 'gantt':
            return self.plot_gantt(ax=ax, **kwargs)
        elif method == 'count':
            return self.plot_count(ax=ax, **kwargs)
        else:
            raise ValueError(f"Unknown plot method: {method}")
