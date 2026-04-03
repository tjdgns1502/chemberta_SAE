import unittest

import torch

from chem_sae.vendor.sae_core import Autoencoder


class TestApplyBDecContract(unittest.TestCase):
    def test_encode_pre_act_respects_apply_b_dec_flag(self) -> None:
        x = torch.ones(2, 3)

        model_apply = Autoencoder(
            n_latents=4,
            n_inputs=3,
            normalize=False,
            apply_b_dec_to_input=True,
        )
        model_no_apply = Autoencoder(
            n_latents=4,
            n_inputs=3,
            normalize=False,
            apply_b_dec_to_input=False,
        )

        with torch.no_grad():
            model_apply.pre_bias.fill_(1.0)
            model_no_apply.pre_bias.copy_(model_apply.pre_bias)
            model_no_apply.encoder.weight.copy_(model_apply.encoder.weight)
            model_no_apply.latent_bias.copy_(model_apply.latent_bias)

        out_apply = model_apply.encode_pre_act(x)
        out_no_apply = model_no_apply.encode_pre_act(x)

        # With apply=True and pre_bias=1, centered input becomes zero so output is just latent bias.
        expected_apply = model_apply.latent_bias.unsqueeze(0).expand_as(out_apply)
        self.assertTrue(torch.allclose(out_apply, expected_apply, atol=1e-6))
        self.assertFalse(torch.allclose(out_apply, out_no_apply))


if __name__ == "__main__":
    unittest.main()
