import pandas as pd


date_series = pd.Series(["2026-10-01 10:00:00+05:30", "2026-10-03 14:30:00+05:30"])
print(date_series)
date_series_utc = pd.to_datetime(date_series).dt.tz_convert("UTC")

print(date_series_utc)

dates = pd.Series(["2023-10-29 02:30:00"])

localized = pd.to_datetime(dates).dt.tz_localize(
    "Europe/Paris",
    nonexistent="shift_forward",  # Options: 'shift_forward', 'shift_backward', 'NaT'
)

print(localized)


dates = pd.Series(
    [
        "2019-10-27 02:30:00",  # Ambiguous (clock goes back, 2:30 happens twice)
        "2019-03-31 02:30:00",  # Nonexistent (clock goes forward, 2:30 never happened)
    ]
)

# Convert to datetime first
dt_series = pd.to_datetime(dates)

# Apply tz_localize with handling parameters
localized = dt_series.dt.tz_localize(
    "CET",
    ambiguous="NaT",  # Turn ambiguous times into NaT
    nonexistent="shift_forward",  # Shift missing times forward to the next valid hour
)

print(localized)