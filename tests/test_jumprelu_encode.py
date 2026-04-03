import unittest

import torch

from chem_sae.vendor.jumprelu import JumpReLUAutoencoder


class TestJumpReLUEncode(unittest.TestCase):
    def test_encode_matches_forward_latents(self) -> None:
        torch.manual_seed(0)
        model = JumpReLUAutoencoder(
            n_latents=8,
            n_inputs=4,
            threshold_init=0.2,
            bandwidth=0.05,
            normalize=False,
        )
        x = torch.randn(16, 4, dtype=torch.float32)

        encoded, _ = model.encode(x)
        _, latents, _ = model(x)

        self.assertTrue(torch.allclose(encoded, latents))


if __name__ == "__main__":
    unittest.main()
