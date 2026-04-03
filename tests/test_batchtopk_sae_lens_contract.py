import unittest

import torch

from chem_sae.vendor.batchtopk_ext import BatchTopK


class TestBatchTopKSaeLensContract(unittest.TestCase):
    def test_forward_matches_saelens_formula(self) -> None:
        x = torch.tensor(
            [
                [[-1.0, 0.1, 0.8, 0.3], [0.2, -0.4, 0.9, 0.5]],
                [[0.7, 0.6, -0.2, 0.0], [0.4, 0.3, 0.2, -0.1]],
            ],
            dtype=torch.float32,
        )
        k = 1.5
        mod = BatchTopK(k=k)

        out = mod(x)
        acts = x.relu()
        flat = acts.flatten()
        num_samples = acts.shape[:-1].numel()
        topk = torch.topk(flat, int(k * num_samples), dim=-1)
        expected = torch.zeros_like(flat).scatter(-1, topk.indices, topk.values).reshape(acts.shape)

        self.assertTrue(torch.equal(out, expected))

    def test_state_dict_contains_only_k(self) -> None:
        mod = BatchTopK(k=2.0)
        state = mod.state_dict()
        self.assertIn("k", state)
        self.assertEqual(set(state.keys()), {"k"})


if __name__ == "__main__":
    unittest.main()
