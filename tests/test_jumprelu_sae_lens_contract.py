import unittest

import torch

from chem_sae.vendor.jumprelu import JumpReLUAutoencoder, _StepFn


class TestJumpReLUSaeLensContract(unittest.TestCase):
    def test_step_fn_has_no_x_gradient(self) -> None:
        hidden_pre = torch.tensor([[0.2, 0.7, 0.1]], requires_grad=True)
        threshold = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)

        l0 = _StepFn.apply(hidden_pre, threshold, 0.05).sum()
        l0.backward()

        self.assertIsNone(hidden_pre.grad)
        self.assertIsNotNone(threshold.grad)

    def test_state_dict_threshold_conversion_roundtrip(self) -> None:
        model = JumpReLUAutoencoder(
            n_latents=4,
            n_inputs=3,
            threshold_init=0.2,
            bandwidth=0.05,
            normalize=False,
        )
        state = model.state_dict()
        model.process_state_dict_for_saving(state)

        self.assertIn("threshold", state)
        self.assertNotIn("log_threshold", state)

        model.process_state_dict_for_loading(state)
        self.assertIn("log_threshold", state)

    def test_fold_w_dec_norm_scales_threshold(self) -> None:
        model = JumpReLUAutoencoder(
            n_latents=4,
            n_inputs=3,
            threshold_init=0.2,
            bandwidth=0.05,
            normalize=False,
        )
        before = model.threshold.detach().clone()
        model.fold_W_dec_norm()
        after = model.threshold.detach().clone()
        self.assertEqual(before.shape, after.shape)


if __name__ == "__main__":
    unittest.main()
