import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import random

import gymnasium as gym
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.base import Action
from simglucose.analysis.risk import risk_index

from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)

class SimGlucoseDataset(BaseDataset):
    """
    SimGlucoseDataset: A synthetic dataset for blood glucose control simulation,
    built on top of the SimGlucose simulator and Magni risk calculations.

    This dataset simulates patient blood glucose trajectories using an internal 
    Gym-compatible wrapper around the SimGlucose environment. It generates CGM readings, 
    insulin actions, meal events, and computes a risk-based reward signal at each timestep.

    Rewards are computed as the negative absolute value of the Magni risk index, encouraging
    blood glucose values to stay within a safe range.

    Attributions:
    - **SimGlucose**: Open-source simulator for type 1 diabetes management, developed by 
      Jingxuan Xie. See [SimGlucose GitHub](https://github.com/jxx123/simglucose) and  
      *Xie, Lei, et al. "SimGlucose: An Open-Source Simulator for Blood Glucose Control." 
      arXiv preprint arXiv:1810.09301 (2018)*.
    - **Deep RL for Blood Glucose Control**: Dataset design inspired by the reinforcement learning 
      pipeline proposed in *Ian Fox, Jaebum Lee, Rodica Pop-Busui, Jenna Wiens. 
      "Deep Reinforcement Learning for Closed-Loop Blood Glucose Control." MLHC 2020.*
    - **Magni Risk Function**: Risk computations based on *Magni, Laura, et al. 
      "A stochastic model to assess the risk of hypoglycemia and hyperglycemia in type 1 diabetes." 
      Journal of diabetes science and technology, 2007.*

    Example usage:

    ```python
    from pyhealth.datasets import SimGlucoseDataset
    import numpy as np

    # Create the dataset
    dataset = SimGlucoseDataset(num_samples=5, rollout_steps=30)

    # Step manually
    obs, reward, done, info = dataset.step(patient_idx=0, action=np.array([0.2]))

    # Rollout all remaining steps
    dataset.rollout_all()

    # Fetch the synthetic dataset
    samples = dataset.get_data()
    print(samples[0])
    ```

    Each record contains:
    - `glucose`: Continuous glucose monitor (CGM) reading (mg/dL)
    - `action`: Insulin dose action applied (U)
    - `reward`: Reward based on risk minimization
    - `meal`: Meal intake (grams of carbohydrates)
    - `insulin`: Insulin delivered (U)
    - `risk`: Magni risk index value

    Note:
    - This dataset is self-contained and does not require manual setup of the SimGlucose environment.
    - Suitable for offline RL training or simulation-based evaluation of blood glucose control policies.
    """

    class BloodGlucoseEnv(gym.Env):
        """Internal gym.Env wrapper around simglucose."""
        def __init__(self):
            super().__init__()
            now = datetime.now()
            start_time = datetime.combine(now.date(), datetime.min.time())        
            patient = T1DPatient.withName('adult#001')
            scenario = RandomScenario(start_time=start_time)
            sensor = CGMSensor.withName('Dexcom', seed=1)
            pump = InsulinPump.withName('Insulet')

            self.env = T1DSimEnv(
                patient=patient,
                sensor=sensor,
                pump=pump,
                scenario=scenario,
            )

            self.last_insulin = 0.0
            self.last_meal = 0.0

            self.observation_space = gym.spaces.Box(
                low=np.array([10.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([600.0, 0.5, 100.0], dtype=np.float32),
                dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=np.array([0.0]),
                high=np.array([0.5]),
                dtype=np.float32
            )

        def reset(self):
            obs = self.env.reset()
            self.last_insulin = 0.0
            self.last_meal = 0.0
            return self._process_obs(obs.observation), {}

        def step(self, action):
            action_value = float(np.clip(action[0], 0.0, 0.5))
            action_obj = Action(basal=0.0, bolus=action_value)
            step = self.env.step(action_obj)

            obs = step.observation
            glucose = obs.CGM
            reward = self.reward_from_risk(glucose)

            self.last_insulin = getattr(step, "insulin", action_value)
            self.last_meal = getattr(step, "meal", 0.0)

            info = {
                "CGM": glucose,
                "insulin": self.last_insulin,
                "meal": self.last_meal,
                "bg": getattr(step, "bg", None),
                "risk": self.magni_risk(glucose),
            }

            terminated = step.done
            truncated = False

            return self._process_obs(obs), reward, terminated, truncated, info

        def _process_obs(self, obs):
            return np.array([
                obs.CGM,
                self.last_insulin,
                self.last_meal
            ], dtype=np.float32)
        
        def magni_risk(self, glucose):
            """
            Compute the Magni risk index based on blood glucose value.
            :param glucose: blood glucose level (mg/dL)
            :return: risk index (higher is worse)
            """
            glucose = np.clip(glucose, 1e-3, None)  # avoid log(0)
            f = 1.509 * ((np.log(glucose)) ** 1.084 - 5.381)
            risk = np.where(f < 0, -1, 1) * 10 * (f ** 2)
            return risk

        def reward_from_risk(self, glucose):
            """
            Reward is negative risk. Safer glucose = higher reward.
            """
            return -np.abs(self.magni_risk(glucose))  # Penalize both high and low
        
    # ========== Dataset Implementation ==========

    def __init__(
        self,
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
            tables=[], 
            dataset_name=dataset_name,
            config_path=config_path,
        )

        self.num_samples = num_samples
        self.rollout_steps = rollout_steps
        self.envs = []
        self.data = []

        self._init_envs()
        self.reset_all()

    def _init_envs(self):
        for _ in range(self.num_samples):
            env = self.BloodGlucoseEnv()
            self.envs.append(env)

    def reset_all(self):
        self.data = []
        for env in self.envs:
            obs, _ = env.reset()
            self.data.append({
                "observations": [obs],
                "actions": [],
                "rewards": [],
                "infos": [],
                "terminated": False,
                "timestep": 0,
            })

    def step(self, patient_idx, action=None):
        patient = self.data[patient_idx]
        env = self.envs[patient_idx]

        if patient["terminated"] or patient["timestep"] >= self.rollout_steps:
            logger.warning(f"Patient {patient_idx} already terminated or max steps reached.")
            return None

        if action is None:
            action = env.action_space.sample()

        next_obs, reward, terminated, truncated, info = env.step(action)

        patient["observations"].append(next_obs)
        patient["actions"].append(action)
        patient["rewards"].append(reward)
        patient["infos"].append(info)
        patient["terminated"] = terminated
        patient["timestep"] += 1

        return next_obs, reward, terminated, info

    def rollout_all(self):
        for patient_idx in range(self.num_samples):
            while not self.data[patient_idx]["terminated"] and self.data[patient_idx]["timestep"] < self.rollout_steps:
                self.step(patient_idx)

    def get_data(self):
        samples = []
        for i, traj in enumerate(self.data):
            records = []
            for t in range(traj["timestep"]):
                obs = traj["observations"][t]
                action = traj["actions"][t]
                reward = traj["rewards"][t]
                info = traj["infos"][t]

                record = {
                    "glucose": float(info.get("CGM", obs[0])),
                    "action": action.tolist(),
                    "reward": float(reward),
                    "timestep": t,
                    "meal": info.get("meal", 0.0),
                    "insulin": info.get("insulin", 0.0),
                    "risk": info.get("risk", None),
                }
                records.append(record)

            sample = {
                "patient_id": str(i),
                "visit_id": "v0",
                "records": records,
            }
            samples.append(sample)

        return samples

    def load_data(self):
        return None
    

