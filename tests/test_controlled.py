# Author: abdkhan
"""Independent checks of analytic labels and bounds used by the controlled experiments."""
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from kic_controlled import CubicPlant, DELTA, break_job, sensor_job, bank_for


class ExactScoringTests(unittest.TestCase):
    def test_control_variate_clipping_is_opt_in_and_predictable(self):
        bank, _ = bank_for(2, 2., 3.)
        centre = {"a0": .5, "a1": .5}
        cell = bank.cell(centre)
        bank._sw[("a0", "cost", cell)] = (100., 1)
        self.assertEqual(bank._beta_hat("a0", "cost", cell), 2.)
        bank.bound_control_variate = False
        self.assertEqual(bank._beta_hat("a0", "cost", cell), 100./DELTA)

    def test_dither_expectation_against_all_corners(self):
        rng = np.random.default_rng(12)
        for p in (2, 4, 8):
            plant = CubicPlant(rng.uniform(-.9, .9, p))
            c = rng.uniform(.2, .8, p)
            enumeration = np.mean([plant.cost(c+DELTA*np.array(s))
                                   for s in itertools.product((-1, 1), repeat=p)])
            self.assertAlmostEqual(enumeration, plant.cost(c, DELTA), places=12)

    def test_derivative_and_declared_envelopes(self):
        rng = np.random.default_rng(55)
        for p in (2, 8, 32):
            plant = CubicPlant(rng.uniform(-.9, .9, p))
            for _ in range(30):
                u = rng.uniform(.02, .98, p)
                self.assertLessEqual(np.abs(plant.grad(u)).max(), plant.G)
                H = np.diag(plant.h+plant.a*(u-.5))+plant.rho/p*np.ones((p,p))
                self.assertLessEqual(np.linalg.norm(H, 2), plant.L)
                for delta in (0., DELTA):
                    eps = 1e-5
                    diff = np.eye(p)[0]*eps
                    numerical = (plant.cost(u+diff,delta)-plant.cost(u-diff,delta))/(2*eps)
                    self.assertAlmostEqual(numerical, plant.grad(u,delta)[0], places=8)

    def test_pre_actuation_reset_blocks_first_stale_license(self):
        unguarded = break_job(("unmonitored", 0, 1600))
        guarded = break_job(("interlock-0", 0, 1600))
        delayed = break_job(("interlock-1", 0, 1600))
        self.assertGreater(unguarded["prebreak_authorized"], 0)
        self.assertEqual(unguarded["first_harmful"], unguarded["switch"])
        self.assertEqual(guarded["harmful"], 0)
        self.assertEqual(delayed["harmful"], 1)

    def test_sensor_quantization_bound_and_packet_accounting(self):
        r = sensor_job((6, .9, True, 0, 400))
        self.assertLessEqual(r["max_error"], r["error_bound"]+1e-12)
        self.assertEqual(r["delivered"]+r["dropped"]+r["expired"]+r["pending"], r["ticks"])


if __name__ == "__main__":
    unittest.main()
