from __future__ import annotations
import calendar
from pathlib import Path
from typing import List

def extract_year_from_path(path: Path) -> int:
    for part in path.parts[::-1]:
        if len(part) == 4 and part.isdigit():
            y = int(part)
            if 1900 <= y <= 2100:
                return y
    raise ValueError(f"Could not find a 4-digit year in path: {path}")

def month_from_doy(doy: int, year_dummy: int = 2001) -> int:
    if doy < 1 or doy > 366:
        raise ValueError(f"DOY out of range: {doy}")
    cum = 0
    for m in range(1, 13):
        nd = calendar.monthrange(year_dummy, m)[1]
        cum += nd
        if doy <= cum:
            return m
    raise ValueError(f"Unable to map DOY to month: {doy}")


def list_doy_subfolders(doy_root: Path) -> List[int]:
    """Return DOY integers found under doy_root.
    Behavior:
      - If doy_root is a DOY folder (name '001'..'366'), return [that DOY].
      - Otherwise, list child folders that look like DOY and return sorted list.
    """
    p = doy_root.expanduser().resolve()

    # If the path itself is a DOY folder, return it.
    if p.name.isdigit() and len(p.name) == 3:
        val = int(p.name)
        if 1 <= val <= 366:
            return [val]

    # Otherwise, list children that are DOY folders.
    doys: List[int] = []
    if not p.is_dir():
        return doys

    for pchild in p.iterdir():
        if not pchild.is_dir():
            continue
        name = pchild.name.strip()
        if len(name) == 3 and name.isdigit():
            val = int(name)
            if 1 <= val <= 366:
                doys.append(val)
    return sorted(doys)


def julian_day_for_month(month: int, year_dummy: int = 2001) -> int:
    return sum(calendar.monthrange(year_dummy, m)[1] for m in range(1, month)) + 1

def out_folder_for_month(month_str: str) -> str:
    doy = julian_day_for_month(int(month_str), year_dummy=2001)
    return f"{doy:03d}"
