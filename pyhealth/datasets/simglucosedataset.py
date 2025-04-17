import logging
from pathlib import Path
import random
import numpy as np
from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)

class SimGlucoseDataset(BaseDataset):
    def __init__(
        self,
        env_fn,
        controller_fn=None,
        dataset_name: str = "simglucose",
        dataset: str = "simglucose",
        num_samples: int = 100,
        rollout_steps: int = 50,
        root: str = "./data",
        config_path: str = None,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "simglucose.yaml"

        super().__init__(
            root=root,
            tables=[],  # no real tables, synthetic
            dataset_name=dataset_name,
            config_path=config_path,
        )

        self.env_fn = env_fn
        self.controller_fn = controller_fn
        self.num_samples = num_samples
        self.rollout_steps = rollout_steps
        self.data = self.simulate_data()
   
    def simulate_data(self):
        """Actually simulate synthetic glucose data."""
        data = []
        for _ in range(self.num_samples):
            env = self.env_fn()
            controller = self.controller_fn() if self.controller_fn else None

            obs, _ = env.reset()
            trajectory = []
            terminated = False
            t = 0

            while not terminated and t < self.rollout_steps:
                if controller:
                    action_value = controller.compute_action(obs)
                    action = np.array([action_value])
                else:
                    action = env.action_space.sample()

                next_obs, reward, terminated, truncated, info = env.step(action)

                glucose = info.get("CGM", next_obs[0]) if isinstance(next_obs, np.ndarray) else None

                trajectory.append({
                    "glucose": float(glucose) if glucose is not None else None,
                    "action": action.tolist(),
                    "reward": float(reward),
                    "timestep": t,
                })

                obs = next_obs
                t += 1

            data.append(trajectory)
        return data

    def load_data(self):
        return None
    
    def get_data(self):
        """Format simulated data into samples."""
        samples = []
        for i, traj in enumerate(self.data):
            sample = {
                "patient_id": str(i),
                "visit_id": "v0",
                "records": traj,
            }
            samples.append(sample)
        return samples