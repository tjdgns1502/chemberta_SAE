from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, RobertaTokenizerFast

from chem_sae.config.experiment import SaeExperimentConfig

from .datasets import MLMSmilesDataset


def prepare_mlm_loader(cfg: SaeExperimentConfig, tokenizer: RobertaTokenizerFast):
    dataset = MLMSmilesDataset(cfg.mlm_data_path, tokenizer, max_len=cfg.max_len)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    return DataLoader(
        dataset,
        batch_size=cfg.mlm_batch_size,
        shuffle=True,
        collate_fn=collator,
    )

