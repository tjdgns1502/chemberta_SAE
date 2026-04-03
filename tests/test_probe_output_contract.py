import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chem_sae.config import SaeExperimentConfig
from chem_sae.train.sae_training_probe import ProbeLayerResult, run_probe


class _DummyModelConfig:
    num_hidden_layers = 1


class _DummyModel:
    config = _DummyModelConfig()


class _CapturingWandbRunLogger:
    instances = []

    def __init__(
        self,
        *,
        logger_cfg,
        run_root,
        run_id,
        config_payload,
        job_type,
        tags=None,
        group=None,
    ):
        self.logger_cfg = logger_cfg
        self.run_root = run_root
        self.run_id = run_id
        self.config_payload = config_payload
        self.job_type = job_type
        self.tags = list(tags or [])
        self.group = group
        self.enabled = False
        self.run_url = None
        _CapturingWandbRunLogger.instances.append(self)

    def start(self):
        return None

    def finish(self):
        return None

    def should_log(self, step):
        return False

    def should_eval_log(self, step):
        return False

    def log(self, values, *, step=None):
        return None

    def update_summary(self, values):
        return None

    def histogram(self, values):
        return values

    def log_artifact(self, **kwargs):
        return None


def _fake_train_probe_for_layer(
    cfg,
    layer,
    device,
    *,
    epochs,
    base_l0,
    schedule_mode,
    warmup_epochs,
    decay_ratio,
    checkpoint_root,
    resume,
    trial_seed,
    dashboard_mirror_dir,
    wandb_logger,
):
    result = ProbeLayerResult(
        layer=layer,
        nmse=0.123,
        mean_l0=1.5,
        dead_ratio=0.1,
        max_node_share=0.2,
        active_cosine_mean=0.3,
        decoder_cosine_max=0.4,
        global_step=7,
        checkpoint_path=checkpoint_root / f"layer_{layer}" / "best.pt",
    )
    trace = [
        {
            "epoch": 1,
            "base_l0": base_l0,
            "effective_l0": base_l0,
            "schedule_mode": schedule_mode,
            "warmup_epochs": warmup_epochs,
            "decay_ratio": decay_ratio,
        }
    ]
    return result, trace


class TestProbeOutputContract(unittest.TestCase):
    @patch("chem_sae.train.sae_training_probe.train_probe_for_layer", side_effect=_fake_train_probe_for_layer)
    @patch("chem_sae.train.sae_training_probe.prepare_activation_cache", return_value=None)
    @patch("chem_sae.train.sae_training_probe.resolve_layers_from_model", return_value=(0,))
    @patch("chem_sae.train.sae_training_probe.build_mlm_model", return_value=(_DummyModel(), None))
    @patch("chem_sae.train.sae_training_probe.RobertaTokenizerFast.from_pretrained", return_value=object())
    def test_probe_writes_metrics_and_schedule(
        self,
        _mock_tokenizer,
        _mock_build,
        _mock_resolve_layers,
        _mock_prepare_cache,
        _mock_train_probe,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            cfg = SaeExperimentConfig(run_id="probe_output_contract")
            cfg.runs_dir = tmp_root / "runs"
            cfg.acts_dir = cfg.runs_dir / "acts"
            cfg.ckpt_dir = cfg.runs_dir / "checkpoints"
            cfg.log_path = cfg.runs_dir / "downstream_records.csv"
            cfg.layers_spec = "0"
            cfg.ensure_dirs()

            result = run_probe(
                cfg,
                layers=(0,),
                epochs=1,
                base_l0=0.001,
                schedule_mode="none",
                warmup_epochs=1,
                decay_ratio=0.1,
                resume=False,
            )

            schedule_path = Path(result["schedule_trace_path"])
            metrics_path = Path(result["metrics_path"])
            dashboard_index_path = Path(result["dashboard_index_path"])
            self.assertTrue(schedule_path.exists())
            self.assertTrue(metrics_path.exists())
            self.assertTrue(dashboard_index_path.exists())

            schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

            self.assertEqual(schedule["run_id"], "probe_output_contract")
            self.assertIn("0", schedule["per_layer"])
            self.assertEqual(metrics["run_id"], "probe_output_contract")
            self.assertIn("aggregate", metrics)
            self.assertIn("layer_results", metrics)
            self.assertEqual(metrics["layer_results"][0]["layer"], 0)
            dashboard_html = dashboard_index_path.read_text(encoding="utf-8")
            self.assertIn("Probe Dashboard Index", dashboard_html)

    @patch("chem_sae.train.sae_training_probe.WandbRunLogger", new=_CapturingWandbRunLogger)
    @patch("chem_sae.train.sae_training_probe.train_probe_for_layer", side_effect=_fake_train_probe_for_layer)
    @patch("chem_sae.train.sae_training_probe.prepare_activation_cache", return_value=None)
    @patch("chem_sae.train.sae_training_probe.resolve_layers_from_model", return_value=(0,))
    @patch("chem_sae.train.sae_training_probe.build_mlm_model", return_value=(_DummyModel(), None))
    @patch("chem_sae.train.sae_training_probe.RobertaTokenizerFast.from_pretrained", return_value=object())
    def test_probe_builds_layer_grouped_wandb_metadata(
        self,
        _mock_tokenizer,
        _mock_build,
        _mock_resolve_layers,
        _mock_prepare_cache,
        _mock_train_probe,
    ) -> None:
        _CapturingWandbRunLogger.instances.clear()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            cfg = SaeExperimentConfig(run_id="probe_group_contract")
            cfg.runs_dir = tmp_root / "runs"
            cfg.acts_dir = cfg.runs_dir / "acts"
            cfg.ckpt_dir = cfg.runs_dir / "checkpoints"
            cfg.log_path = cfg.runs_dir / "downstream_records.csv"
            cfg.layers_spec = "0"
            cfg.n_latents = 1536
            cfg.logger.log_to_wandb = True
            cfg.logger.run_name = "probe-manual"
            cfg.ensure_dirs()

            run_probe(
                cfg,
                layers=(0,),
                epochs=1,
                base_l0=0.12,
                schedule_mode="none",
                warmup_epochs=1,
                decay_ratio=0.1,
                resume=False,
            )

        logger = _CapturingWandbRunLogger.instances[0]
        self.assertEqual(logger.group, "probe_layer_00")
        self.assertIn("layer:00", logger.tags)
        self.assertIn("l0:0.1200", logger.tags)
        self.assertIn("n_latents:1536", logger.tags)
        self.assertTrue(cfg.logger.run_name.startswith("L00 | l0=0.1200 | n=1536 |"))


if __name__ == "__main__":
    unittest.main()
