import unittest
from pyhealth.datasets import PendulumData


class TestPendulumData(unittest.TestCase):
    """Test cases for OpenAI Pendulum Dataset."""

    def setUp(self):
        self.args = {
            "env_name": "Pendulum-v1",
            "render_mode": "rgb_array",
            "seed": 1,
            "data_size": 10,
            "seq_len": 2,
            "side": 1,
            "friction": 0,
        }

    def test_args(self):
        self.assertDictEqual(
            self.args,
            {
                "env_name": "Pendulum-v1",
                "render_mode": "rgb_array",
                "seed": 1,
                "data_size": 10,
                "seq_len": 2,
                "side": 1,
                "friction": 0,
            },
        )

    def test_get_data(self):
        pd = PendulumData(**self.args)
        data, latent_data, params_data = pd.create_pendulum_data()
        self.assertListEqual(
            data.tolist(),
            [
                [[[0.0]], [[0.0]]],
                [[[0.0]], [[0.0]]],
                [[[0.0]], [[0.0]]],
                [[[0.0]], [[0.0]]],
                [[[0.0]], [[0.0]]],
                [[[0.0]], [[0.0]]],
                [[[0.0]], [[0.0]]],
                [[[0.0]], [[0.0]]],
                [[[0.0]], [[0.0]]],
                [[[0.00021478441553081915]], [[0.00021086596582142346]]],
            ],
        )
        self.assertEqual(latent_data.shape, (10, 2, 2))
        self.assertGreater(params_data[0]["l"], 0)
