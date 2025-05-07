"""
File: generate_ahi.py

Authors: Camden Wall (cnwall2), Francis Alobba (falobba2)
Contribution: PyHealth - Pediatric Sleep Study Task Module
Title: Bringing At-Home Pediatric Sleep Apnea Testing Closer to Reality: A Multi-Modal Transformer Approach
Paper: https://pubmed.ncbi.nlm.nih.gov/38344396/

Description:
------------
This module implements a function to compute the Apnea-Hypopnea Index (AHI) for a set of pediatric sleep studies.
AHI is defined as the number of apnea events per hour of total study time, and is used as a quantitative marker
for the severity of sleep apnea.

Typical Use Case:
-----------------
This function is used as a preprocessing step to generate ground truth AHI values that are used in 
supervised learning tasks or for stratifying patients by severity in research on pediatric sleep apnea.
"""

import os
import pandas as pd
from typing import List, Optional


def generate_ahi_csv(
    in_study_list: List[str],
    out_csv_path: str = "",
    filter_value: Optional[float] = None
) -> None:
    """
    Computes AHI (Apnea-Hypopnea Index) for each study in the input list and saves the result to a CSV.

    Parameters
    ----------
    in_study_list : List[str]
        A list of paths to `.annot` files, each representing a sleep study's annotation data.

    out_csv_path : str, optional
        The output path for the generated CSV file containing AHI values and related metrics.

    filter_value : Optional[float], optional
        If provided, only studies with AHI greater than or equal to this value will be included in the output.

    Returns
    -------
    None
        Saves a CSV file to `out_csv_path` containing the following columns:
            - Study: Name of the study file
            - TotalStudyTime: Duration of the study in hours
            - ApneaEventCount: Number of apnea events
            - HypopneaEventCount: Number of hypopnea events
            - AHI: Apnea-Hypopnea Index

    Notes
    -----
    - AHI is computed as:
        AHI = (ApneaEventCount + HypopneaEventCount)/ TotalStudyTime (in hours)


    Example
    -------
    >>> generate_ahi_csv(
    ...     in_study_list=["/path/to/study1.annot", "/path/to/study2.annot"],
    ...     out_csv_path="ahi_results.csv",
    ...     filter_value=5.0
    ... )
    """

    ahi_data = []

    for annotation_path in in_study_list:
        df = pd.read_csv(annotation_path, sep='\t', 
                         names=["annotation", "time", "value"])
        
        annotations = df["annotation"]
        total_time = df["time"].max()
        time_elapsed_hours = total_time / 3600.0

        apnea_events = annotations.str.contains("Apnea").sum()
        hypopnea_events = annotations.str.contains("Hypopnea").sum()

        ahi = (apnea_events + hypopnea_events) / time_elapsed_hours
        study_name = os.path.basename(annotation_path)

        print(f"{study_name} - AHI: {ahi}")

        if filter_value is None or ahi >= filter_value:
            ahi_data.append({
                "Study": study_name,
                "TotalStudyTime": time_elapsed_hours,
                "ApneaEventCount": apnea_events,
                "HypopneaEventCount": hypopnea_events,
                "AHI": ahi
            })
        else:
            print(f"{study_name} filtered out due to AHI < {filter_value}")

    ahi_df = pd.DataFrame(ahi_data)
    ahi_df.to_csv(out_csv_path, index=False)
    print(f"\nAHI values saved to: {out_csv_path}")


def test_generate_ahi_csv():
    # Create temporary test directory and file
    test_dir = "test_data"
    os.makedirs(test_dir, exist_ok=True)
    test_file_path = os.path.join(test_dir, "test.annot")

    # Simulated annotation data: 3 apneas, 2 hypopneas, over 2 hours (7200 seconds)
    with open(test_file_path, "w") as f:
        f.write(
            "Apnea\t100\t1\n"
            "Apnea\t500\t1\n"
            "Hypopnea\t1000\t1\n"
            "Apnea\t2000\t1\n"
            "Hypopnea\t3000\t1\n"
            "Nonevent\t7200\t0\n"  # Marks end of study
        )

    # Output path
    output_path = os.path.join(test_dir, "ahi_output.csv")

    # Run AHI generation
    generate_ahi_csv(in_study_list=[test_file_path], out_csv_path=output_path)

    # Load and check results
    df = pd.read_csv(output_path)
    assert len(df) == 1, "Output should have 1 row"
    assert df["ApneaEventCount"].iloc[0] == 3
    assert df["HypopneaEventCount"].iloc[0] == 2
    assert abs(df["AHI"].iloc[0] - 2.5) < 1e-6, "AHI should be (3+2)/2 = 2.5"

    print("Test passed.")

    # Cleanup
    os.remove(test_file_path)
    os.remove(output_path)
    os.rmdir(test_dir)

if __name__ == "__main__":
    test_generate_ahi_csv()