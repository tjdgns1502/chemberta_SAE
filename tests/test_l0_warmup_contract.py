import unittest

from chem_sae.train.sae_training import _resolve_warmup_coefficient as resolve_train
from chem_sae.train.sae_training_probe import _resolve_warmup_coefficient as resolve_probe


class TestL0WarmupContract(unittest.TestCase):
    def test_no_warmup_returns_final(self) -> None:
        self.assertAlmostEqual(resolve_train(0.2, 0, 0), 0.2)
        self.assertAlmostEqual(resolve_probe(0.2, 0, 0), 0.2)

    def test_warmup_starts_from_zero(self) -> None:
        self.assertAlmostEqual(resolve_train(0.2, 10, 0), 0.0)
        self.assertAlmostEqual(resolve_probe(0.2, 10, 0), 0.0)

    def test_warmup_scales_linearly(self) -> None:
        self.assertAlmostEqual(resolve_train(0.2, 10, 5), 0.1)
        self.assertAlmostEqual(resolve_probe(0.2, 10, 5), 0.1)

    def test_warmup_caps_at_final(self) -> None:
        self.assertAlmostEqual(resolve_train(0.2, 10, 10), 0.2)
        self.assertAlmostEqual(resolve_train(0.2, 10, 20), 0.2)
        self.assertAlmostEqual(resolve_probe(0.2, 10, 10), 0.2)
        self.assertAlmostEqual(resolve_probe(0.2, 10, 20), 0.2)


if __name__ == "__main__":
    unittest.main()
