"""
Author: Manling Tan, Yining Jin, Xiao Luo
NetID: manling4, yining34, xiaoluo6

Paper Title:
Temporal Pointwise Convolutional Networks for
Length of Stay Prediction in the Intensive Care Unit

Paper Authors:
Emma Rocheteau, Pietro Liò, and Stephanie Hyland (2021)

Paper URL:
https://arxiv.org/abs/2007.09483
"""


"""Test hourly binning, forward filling, and ICU LOS label generation.

Uses synthetic heart rate measurements to demonstrate hourly
aggregation, missing-value handling, time since last observation,
and remaining ICU length-of-stay label generation.
"""

import pandas as pd


def main() -> None:
    """Run synthetic hourly preprocessing and LOS label checks.

    Creates sample heart rate measurements, aggregates them into
    hourly bins, and forward-fills missing values.

    Calculates the elapsed time since the most recent observation
    and generates remaining ICU LOS labels starting at hour 5
    for a simulated 30-hour ICU stay.
    """
    # Create sample heart rate measurements for an ICU patient.
    data = pd.DataFrame(
        {
            "hour": [0.17, 0.67, 1.33],
            "heart_rate": [80, 90, 100],
        }
    )

    # Assign each measurement to an hourly time bin.
    data["hour_bin"] = data["hour"].astype(int)

    # Calculate the average heart rate for each hour.
    hourly = data.groupby("hour_bin")["heart_rate"].mean()

    # Include hours with no recorded measurements.
    hourly = hourly.reindex(range(5))

    # Display the hourly heart rate data.
    print(hourly)

    # Forward-fill missing values using the most recent observation.
    hourly_filled = hourly.ffill()

    print("\nAfter forward fill:")
    print(hourly_filled)

    # Track the time elapsed since the most recent measurement.
    last_observed_hour = None
    hours_since_observed = []

    for hour, value in hourly.items():
        if pd.notna(value):
            last_observed_hour = hour

        if last_observed_hour is None:
            hours_since_observed.append(float("nan"))
        else:
            hours_since_observed.append(hour - last_observed_hour)

    # Combine the results into one table.
    result = pd.DataFrame(
        {
            "heart_rate_original": hourly,
            "heart_rate_filled": hourly_filled,
            "hours_since_observed": hours_since_observed,
        }
    )

    print("\nHourly data with time information:")
    print(result)

    # Define the total ICU length of stay in hours.
    total_los_hours = 30

    # Generate remaining LOS labels for each prediction hour.
    prediction_hours = range(5, total_los_hours)

    los_labels = pd.DataFrame(
        {
            "hour": list(prediction_hours),
            "remaining_los_hours": [
                total_los_hours - hour for hour in prediction_hours
            ],
        }
    )

    print("\nRemaining ICU LOS labels:")
    print(los_labels.head(10))


if __name__ == "__main__":
    main()

