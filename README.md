# ChemBERTa SAE Experiments

Sparse Autoencoder (SAE) interpretability experiments on ChemBERTa models.

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd chemberta_repro_final
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained model (if needed):
The scripts will automatically download the ChemBERTa model from HuggingFace.

## Project Structure

```
chemberta_repro_final/
├── data/                      # Dataset files (BBBP, BACE, etc.)
├── docs/                      # Experiment scripts
│   ├── run_baseline_only.py           # Baseline evaluation
│   ├── run_final_hidden_state.py      # Final layer evaluation
│   ├── sae_experiment.py              # Main SAE training
│   └── sae_intervention_experiment.py # SAE intervention experiments
├── sparse_autoencoder/        # SAE implementation
├── code/                      # ChemBERTa code
└── runs/                      # Experiment outputs
```

## Usage

### Run Baseline Evaluation
```bash
python docs/run_baseline_only.py
```

### Run SAE Training
```bash
python docs/sae_experiment.py
```

### Run Intervention Experiments
```bash
bash docs/run_interventions.sh
```

## Data

The `data/` folder contains:
- `BBBP.csv` - Blood-Brain Barrier Penetration dataset
- `bace.csv` - BACE dataset

## Outputs

Results are saved to:
- `runs/sae/` - SAE training checkpoints and activations
- `runs/sae_intervention/` - Intervention experiment results
