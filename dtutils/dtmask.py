# dtmask.py
# Methods for selecting training periods

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as ml
from typing import List, Tuple, Optional
from .dtgen import str2ts, inside
from .dtleaks import LeakSet
import pdb

ts = pd.Timestamp
Interval = Tuple[ts, ts]

class TrainSet:
    def __init__(self, intervals: List[Interval] = None):
        if intervals is None:
            self.intervals = []
        else:
            self.intervals = intervals

    @classmethod
    def all_year(cls, year: int) -> 'TrainSet':
        start = pd.Timestamp(f"{year}-01-01 00:00:00")
        end = pd.Timestamp(f"{year}-12-31 23:59:59")
        return cls([(start, end)])

    @staticmethod
    def set_dates(intervals: List[Tuple[str, str]]) -> "TrainSet":
    # construct from a list of time intervals
        parsed_intervals = []
        for start_str, end_str in intervals:
            start = str2ts(start_str)
            end = str2ts(end_str)
            if start > end:
                raise ValueError(f"Start after end in interval: ({start_str}, {end_str})")
            parsed_intervals.append((start, end))
        return TrainSet(parsed_intervals)

    def append_interval(self, start: str, end: str):
    # Append to the training set interval [start, end]
        start_ts = str2ts(start)
        end_ts = str2ts(end)
        self.intervals.append((start_ts, end_ts))

    def exclude_interval(self, start: str, end: str):
    # Exclude from the training set interval [start, end]
        start_ts = str2ts(start)
        end_ts = str2ts(end)
        new_intervals = []
        for s, e in self.intervals:
            if end_ts <= s or e <= start_ts:
                # No overlap
                new_intervals.append((s, e))
            else:
                # Partial overlap
                if s < start_ts:
                    new_intervals.append((s, min(e, start_ts)))
                if end_ts < e:
                    new_intervals.append((max(s, end_ts), e))
        self.intervals = new_intervals

    def exclude_leaks(self, leakset: LeakSet):
    # Excludes from the training set all time periods corresponding to leaks in leakset
        for start, end in leakset:
            self.exclude_interval(start, end)

    @classmethod
    def from_series(cls, ts: pd.Series, low: float = None, high: float = None):
    # Creates intervals where time series ts values are within range [low, high]
        mask = pd.Series(True, index=ts.index)
        if low is not None:
            mask &= ts >= low
        if high is not None:
            mask &= ts <= high
        return cls.from_mask(mask)

    @classmethod
    def from_mask(cls, mask: pd.Series):
    # Converts boolean mask into a list of intervals
        intervals = []
        in_interval = False
        start = None
        for t, val in mask.items():
            if val and not in_interval:
                start = t
                in_interval = True
            elif not val and in_interval:
                intervals.append((start, t))
                in_interval = False
        if in_interval:
            intervals.append((start, mask.index[-1]))
        return cls(intervals)

    def to_mask(self, index: pd.DatetimeIndex) -> pd.Series:
    # Converts list of intervals into a boolean mask for a given time index
        mask = pd.Series(False, index=index)
        for start, end in self.intervals:
            mask[(mask.index >= start) & (mask.index < end)] = True
        return mask

    def invert(self):
    # Converts training periods into non-training and vice versa    
        return TrainSet._invert_intervals(self.intervals)

    @staticmethod
    def _invert_intervals(intervals: List[Interval], full_start: str = None, full_end: str = None):
        full_start = str2ts(full_start)
        full_end = str2ts(full_end)
        intervals = sorted(intervals)
        result = []
        prev_end = full_start
        for start, end in intervals:
            if prev_end is not None and start > prev_end:
                result.append((prev_end, start))
            prev_end = max(prev_end, end) if prev_end else end
        if full_end and prev_end < full_end:
            result.append((prev_end, full_end))
        return TrainSet(result)

    def save_csv(self, path: str):
        df = pd.DataFrame(self.intervals, columns=["start", "end"])
        df.to_csv(path, index=False)

    @classmethod
    def load_csv(cls, path: str):
        df = pd.read_csv(path, parse_dates=["start", "end"])
        return cls(list(zip(df["start"], df["end"])))

    def __str__(self):
        sorted_intervals = sorted(self.intervals)
        interval_lines = [f"  {start} - {end}" for start, end in sorted_intervals]
        joined = "\n".join(interval_lines)
        return f"TrainSet({len(self.intervals)} intervals):\n{joined}"

    def plot_as_bands(self, ax):
        for start, end in self.intervals:
            ax.axvspan(start, end, color='green', alpha=0.3)

    def plot_as_arrows(self, ax: plt.Axes, y: float = 3, color: str = 'green', 
                       label: Optional[str] = None,
                       linewidth: float = 2, arrowprops=None):
        """
        Draws intervals as arrows <-----> on the transmitted axis ax.

        Parameters:
        ax         -- axis matplotlib (for example, obtained from plt.subplots)
        y          -- vertical position of the arrow
        color      -- line color
        label      -- label for legend
        linewidth  -- line thickness
        arrowprops -- additional parameters for arrows (dict)
        """

        if arrowprops is None:
           arrowprops = dict(arrowstyle='<->', color=color, linewidth=linewidth)

        for start, end in self.intervals:
            ax.annotate('', xy=(end, y), xytext=(start, y), 
            arrowprops=arrowprops, annotation_clip=False)

        if label:
           dummy_line = ax.plot([], [], color=color, linewidth=linewidth, label=label)[0]
           ax.legend()