# examples/synthetic_dataset.py

"""
Generates synthetic EHR-like patient trajectories for testing.

Each patient contains:
- visits: list of timestamped events
- outcome_time OR censor_time
"""

import random


def generate_synthetic_dataset(num_patients=50, seed=None):
    """
    Generates a stochastic synthetic dataset for experiments/ablations.

    Characteristics:
    - Randomized visit times
    - Random event vs censoring
    - More realistic variability than test dataset

    Args:
        num_patients (int): number of patients
        seed (int, optional): random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)

    patients = []

    for i in range(num_patients):
        num_events = random.randint(5, 15)
        times = sorted(random.sample(range(1, 100), num_events))

        visits = [{"time": t} for t in times]

        if random.random() > 0.5:
            outcome_time = times[-1] + random.randint(5, 20)
            censor_time = None
        else:
            outcome_time = None
            censor_time = times[-1] + random.randint(5, 20)

        patients.append({
            "patient_id": f"p{i}",
            "visits": visits,
            "outcome_time": outcome_time,
            "censor_time": censor_time,
        })

    return patients
