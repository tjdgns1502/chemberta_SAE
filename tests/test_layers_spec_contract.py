import unittest

from chem_sae.config import SaeExperimentConfig


class TestLayersSpecContract(unittest.TestCase):
    def test_all_layers(self) -> None:
        cfg = SaeExperimentConfig()
        cfg.layers_spec = "all"
        self.assertEqual(cfg.resolve_layers(4), (0, 1, 2, 3))

    def test_comma_separated_layers(self) -> None:
        cfg = SaeExperimentConfig()
        cfg.layers_spec = "0,2"
        self.assertEqual(cfg.resolve_layers(4), (0, 2))

    def test_empty_layers_spec_raises(self) -> None:
        cfg = SaeExperimentConfig()
        cfg.layers_spec = ""
        with self.assertRaises(ValueError):
            cfg.resolve_layers(4)

    def test_negative_layer_raises(self) -> None:
        cfg = SaeExperimentConfig()
        cfg.layers_spec = "0,-1"
        with self.assertRaises(ValueError):
            cfg.resolve_layers(4)

    def test_out_of_range_layer_raises(self) -> None:
        cfg = SaeExperimentConfig()
        cfg.layers_spec = "0,4"
        with self.assertRaises(ValueError):
            cfg.resolve_layers(4)


if __name__ == "__main__":
    unittest.main()
