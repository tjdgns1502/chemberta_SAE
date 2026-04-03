import random
import unittest

import numpy as np
import torch

from chem_sae.utils.randomness import capture_rng_state, restore_rng_state, set_seed


class TestRandomnessState(unittest.TestCase):
    def test_capture_restore_roundtrip(self) -> None:
        set_seed(123)
        state = capture_rng_state()

        first = (
            random.random(),
            float(np.random.rand()),
            float(torch.rand(1).item()),
        )

        # Advance RNG streams and then restore to the captured point.
        _ = random.random()
        _ = np.random.rand()
        _ = torch.rand(1)
        restore_rng_state(state)

        second = (
            random.random(),
            float(np.random.rand()),
            float(torch.rand(1).item()),
        )

        self.assertAlmostEqual(first[0], second[0], places=12)
        self.assertAlmostEqual(first[1], second[1], places=12)
        self.assertTrue(torch.allclose(torch.tensor(first[2]), torch.tensor(second[2])))


if __name__ == "__main__":
    unittest.main()
