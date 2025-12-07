# Author(s): Esteban Benitez
# NetID(s): benitez5
# Paper Title: Generative ODE Modeling with Known Unknowns
# Paper Link: https://arxiv.org/abs/2003.10775
# Description: 
#   Implements a Gymnasium-based Pendulum dataset generator for sequence modeling,
#   reproducing the environment, parameterization, and data layout described in the paper.

import numpy as np
import gymnasium as gym
import skimage.transform
from tqdm import trange
from .base_dataset import BaseDataset
import logging

logger = logging.getLogger(__name__)


class PendulumData(BaseDataset):
    """
    Synthetic data generator for the OpenAI Gymnasium Pendulum environment.

    This class uses the Pendulum environment from OpenAI's Gymnasium to generate
    sequences of pendulum states and rendered images. The resulting dataset is suitable
    for learning and evaluating dynamical system models, such as generative ODEs.

    Parameters
    ----------
    env_name : str
        The name of the gymnasium environment to use (default: 'Pendulum-v1').
    render_mode : str
        The render mode for the environment (e.g., "rgb_array").
    seed : int
        Random seed for reproducibility.
    data_size : int
        Number of trajectories to simulate (number of pendulum sequences).
    seq_len : int
        Number of time steps per trajectory.
    side : int
        Size (pixels) of each rendered image (output shape: [side, side]).
    friction : float, optional
        Friction coefficient used during simulation (if supported by environment).

    References
    ----------
    .. [1] Linial, Ori, Neta Ravid, Danny Eytan, and Uri Shalit.
       "Generative ODE Modeling with Known Unknowns."
       arXiv preprint arXiv:2003.10775 [cs.LG], 2020. https://arxiv.org/abs/2003.10775
    .. [2] GOKU GitHub repository: https://github.com/orilinial/GOKU

    Example
    -------
    >>> args = {
    ...     "env_name": "Pendulum-v1",
    ...     "render_mode": "rgb_array",
    ...     "seed": 42,
    ...     "data_size": 100,
    ...     "seq_len": 50,
    ...     "side": 64,
    ...     "friction": 0.0,
    ... }
    >>> dataset = PendulumData(**args)
    """

    def __init__(self, **args):
        self.env = gym.make(args.get("Pendulum-v1"), render_mode=args["render_mode"]).unwrapped
        self.env.reset(seed=args.get("seed", 2))
        self.data = np.zeros(
            (args["data_size"], args["seq_len"], args["side"], args["side"])
        )
        self.latent_data = np.zeros((args["data_size"], args["seq_len"], 2))
        self.params_data = []
        self.args = args
        self.data_size = args["data_size"]
        self.seq_len = args["seq_len"]
        self.side = args["side"]

    def create_pendulum_data(self):
        for trial in trange(self.data_size):
            reset_env(self.env, self.args)
        params = get_params()
        unlearned_params = get_unlearned_params()

        for step in range(self.seq_len):
            processed_frame = preproc(self.env.render(), self.side)
            self.data[trial, step] = processed_frame
            obs = step_env(self.args, self.env, [0.0], params, unlearned_params)

            self.latent_data[trial, step, 0] = get_theta(obs)
            self.latent_data[trial, step, 1] = obs[-1]

        self.params_data.append(params)

        self.env.close()
        return self.data, self.latent_data, self.params_data


def get_theta(obs):
    """Transforms coordinate basis from the defaults of the gym pendulum env."""
    theta = np.arctan2(obs[0], -obs[1])
    theta = theta + np.pi / 2
    theta = theta + 2 * np.pi if theta < -np.pi else theta
    theta = theta - 2 * np.pi if theta > np.pi else theta
    return theta


def preproc(X, side):
    """Crops, downsamples, desaturates, etc. the rgb pendulum observation."""
    X = X[..., 0][220:-110, 165:-165] - X[..., 1][220:-110, 165:-165]
    return skimage.transform.resize(X, [int(side), side]) / 255.0


def step_env(args, env, u, params, additional_params):
    th, thdot = env.state

    g = 10.0
    m = 1.0
    b = additional_params["b"]
    l = params["l"]
    dt = env.dt

    if args["friction"]:
        newthdot = thdot + ((-g / l) * np.sin(th + np.pi) - (b / m) * thdot) * dt
    else:
        newthdot = thdot + ((-g / l) * np.sin(th + np.pi)) * dt

    newth = th + newthdot * dt
    newthdot = np.clip(newthdot, -env.max_speed, env.max_speed)

    env.state = np.array([newth, newthdot])
    return env._get_obs()


def get_params():
    l = np.random.uniform(1.0, 2.0)
    params = {"l": l}
    return params


def get_unlearned_params():
    b = 0.7
    params = {"b": b}
    return params


def reset_env(env, args, min_angle=0.0, max_angle=np.pi / 6):
    angle_ok = False
    while not angle_ok:
        obs, info = env.reset()
        theta_init = np.abs(get_theta(obs))
        if min_angle < theta_init < max_angle:
            angle_ok = True
