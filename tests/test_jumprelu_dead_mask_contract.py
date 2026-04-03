import unittest

import torch

from chem_sae.vendor.jumprelu import JumpReLUAutoencoder, jumprelu_loss


class TestJumpReLUDeadMaskContract(unittest.TestCase):
    def test_pre_act_loss_applies_only_to_dead_neuron_mask(self) -> None:
        model = JumpReLUAutoencoder(
            n_latents=2,
            n_inputs=2,
            threshold_init=0.5,
            bandwidth=0.05,
            normalize=False,
            pre_act_loss_coefficient=1.0,
        )
        reconstruction = torch.zeros(1, 2)
        original = torch.zeros(1, 2)
        latents = torch.zeros(1, 2)
        hidden_pre = torch.zeros(1, 2)

        loss_alive = jumprelu_loss(
            reconstruction=reconstruction,
            original_input=original,
            latent_activations=latents,
            hidden_pre=hidden_pre,
            l0_coefficient=0.0,
            l1_weight=0.0,
            model=model,
            dead_neuron_mask=torch.tensor([False, False]),
            pre_act_loss_coefficient=1.0,
        )
        loss_dead = jumprelu_loss(
            reconstruction=reconstruction,
            original_input=original,
            latent_activations=latents,
            hidden_pre=hidden_pre,
            l0_coefficient=0.0,
            l1_weight=0.0,
            model=model,
            dead_neuron_mask=torch.tensor([True, True]),
            pre_act_loss_coefficient=1.0,
        )

        self.assertAlmostEqual(float(loss_alive.item()), 0.0, places=7)
        self.assertGreater(float(loss_dead.item()), float(loss_alive.item()))


if __name__ == "__main__":
    unittest.main()
