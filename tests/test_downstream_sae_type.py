import unittest

from chem_sae.eval.downstream import _resolve_sae_type


class DummyAE:
    pass


class JumpReLUAutoencoderDummy:
    pass


class TestDownstreamSaeType(unittest.TestCase):
    def test_arch_override_jumprelu(self) -> None:
        self.assertEqual(_resolve_sae_type(DummyAE(), {"arch": "jumprelu"}), "JumpReLU")

    def test_arch_override_topk(self) -> None:
        self.assertEqual(_resolve_sae_type(DummyAE(), {"arch": "batchtopk"}), "TopK")

    def test_classname_fallback(self) -> None:
        self.assertEqual(_resolve_sae_type(JumpReLUAutoencoderDummy(), None), "JumpReLU")


if __name__ == "__main__":
    unittest.main()
