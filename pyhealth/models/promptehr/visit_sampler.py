"""
Sample realistic visit structures from real MIMIC-III data distributions.

This module provides functionality to sample the number of visits per patient
and the number of diagnosis codes per visit, matching the empirical distributions
observed in real EHR data.
"""
import numpy as np
from typing import List


class VisitStructureSampler:
    """Sample realistic visit and code count structures from training data."""

    def __init__(self, patient_records: List, seed: int = 42):
        """Initialize sampler with empirical distributions from training data.

        Args:
            patient_records: List of patient records from training set.
                Each record should have a 'visits' attribute (list of visit codes).
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.RandomState(seed)

        # Extract empirical distributions
        self.num_visits_per_patient = []
        self.codes_per_visit_all = []

        for patient in patient_records:
            # Handle both dict-like and object-like patient records
            if hasattr(patient, 'visits'):
                visits = patient.visits
            elif isinstance(patient, dict) and 'visits' in patient:
                visits = patient['visits']
            else:
                continue

            num_visits = len(visits)
            self.num_visits_per_patient.append(num_visits)

            for visit in visits:
                num_codes = len(visit)
                if num_codes > 0:  # Only include non-empty visits
                    self.codes_per_visit_all.append(num_codes)

        # Convert to numpy arrays
        self.num_visits_per_patient = np.array(self.num_visits_per_patient)
        self.codes_per_visit_all = np.array(self.codes_per_visit_all)

        # Compute statistics for logging
        self.stats = {
            'visits_mean': np.mean(self.num_visits_per_patient),
            'visits_median': np.median(self.num_visits_per_patient),
            'visits_90th': np.percentile(self.num_visits_per_patient, 90),
            'codes_mean': np.mean(self.codes_per_visit_all),
            'codes_median': np.median(self.codes_per_visit_all),
            'codes_90th': np.percentile(self.codes_per_visit_all, 90),
            'codes_95th': np.percentile(self.codes_per_visit_all, 95),
        }

    def sample_num_visits(self) -> int:
        """Sample number of visits from empirical distribution.

        Returns:
            Number of visits (>= 0).
        """
        return int(self.rng.choice(self.num_visits_per_patient))

    def sample_codes_per_visit(self, n_visits: int) -> List[int]:
        """Sample number of codes for each visit from empirical distribution.

        Args:
            n_visits: Number of visits to sample code counts for.

        Returns:
            List of integers representing codes per visit.
        """
        if n_visits == 0:
            return []

        # Sample with replacement from empirical distribution
        codes_counts = self.rng.choice(self.codes_per_visit_all, size=n_visits, replace=True)
        return codes_counts.tolist()

    def sample_structure(self) -> dict:
        """Sample complete visit structure (visits + codes per visit).

        Returns:
            Dictionary with:
                - 'num_visits': int (number of visits)
                - 'codes_per_visit': List[int] (codes for each visit)
        """
        num_visits = self.sample_num_visits()
        codes_per_visit = self.sample_codes_per_visit(num_visits)

        return {
            'num_visits': num_visits,
            'codes_per_visit': codes_per_visit
        }

    def get_statistics(self) -> dict:
        """Get statistics about the underlying distributions.

        Returns:
            Dictionary with mean/median/percentile statistics.
        """
        return self.stats.copy()

    def __repr__(self) -> str:
        """String representation showing distribution statistics."""
        return (
            f"VisitStructureSampler(\n"
            f"  Visits/patient: mean={self.stats['visits_mean']:.2f}, "
            f"median={self.stats['visits_median']:.0f}, "
            f"90th%={self.stats['visits_90th']:.0f}\n"
            f"  Codes/visit: mean={self.stats['codes_mean']:.2f}, "
            f"median={self.stats['codes_median']:.0f}, "
            f"90th%={self.stats['codes_90th']:.0f}, "
            f"95th%={self.stats['codes_95th']:.0f}\n"
            f")"
        )
