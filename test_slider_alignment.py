"""Diagnostic: test slider value alignment with step."""

from datetime import timedelta

import pandas as pd

# Row limit
max_rows = 4518
default_rows = min(5000, max_rows)
step = 200

print(f"Row slider: default_rows={default_rows}, step={step}")
print(f"  default_rows % step = {default_rows % step}")
print(f"  Aligned to step: {default_rows // step * step}")
print(
    f"  → Streamlit snaps {default_rows} to {default_rows // step * step} → CONFLICT on rerun"
)

# Time slider
mn = pd.Timestamp("2026-03-02 12:26:12")
mx = pd.Timestamp("2026-03-02 15:53:38")
step_td = timedelta(minutes=1)

mn_dt = mn.to_pydatetime()
mx_dt = mx.to_pydatetime()

# Seconds that don't align with minute step
print(f"\nTime slider: min={mn_dt}, max={mx_dt}")
print(f"  step={step_td}")
print(f"  min seconds={mn_dt.second} (not 0 → CONFLICT on rerun)")
print(f"  max seconds={mx_dt.second} (not 0 → CONFLICT on rerun)")

# Fixed
mn_floor = mn_dt.replace(second=0, microsecond=0)
mx_ceil = (
    (mx + pd.Timedelta(minutes=1)).to_pydatetime().replace(second=0, microsecond=0)
)
print(f"\nFixed: min={mn_floor}, max={mx_ceil} (both aligned to minutes)")
