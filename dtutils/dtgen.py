# dtgen.py
# generic utilities

import pandas as pd
from typing import Tuple, Optional

ts = pd.Timestamp
Interval = Tuple[ts, ts]

def str2ts(s: Optional[str]) -> Optional[pd.Timestamp]:
# Converts string to a timestamp. Raises exception for invalid time format 
    if s is None:
        return None
    try:
        return pd.Timestamp(s)
    except Exception as e:
        raise ValueError(f"String can not be converted to timestamp") from e

def inside(t: pd.Timestamp, start: str, end: str) -> bool:
# Checks if t belongs to the interval [start, end]
    start = str2ts(start)
    end = str2ts(end)
    return start <= t < end

def overlap(a: Interval, b: Interval) -> bool:
# Checks if two intervals overlap
    return a[0] < b[1] and b[0] < a[1]