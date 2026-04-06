# FINAL_SUBMISSION — Physics-Informed Neural Networks for Soil Moisture Estimation

Self-contained folder for the Nature submission. Everything needed to reproduce the
paper's results is here — no external dependencies on scattered workspace files.

## Folder Structure

```
FINAL_SUBMISSION/
├── code/
│   ├── pinn_training.py              # Training script (4000 epochs: 3000 Stage 1 + 1000 Stage 2)
│   ├── paper_figures.ipynb           # Generates all figures and Table 1
│   ├── tfsolverMIconstrb_float64.py  # Custom RK4 solver (dependency)
│   └── requirements.txt             # Python packages
├── data/
│   ├── campbell_1h.json              # Irrigation (training)
│   ├── davis_1h.json                 # Rainfall + ET0 (training)
│   ├── irrimeter_1h.json            # Soil moisture (training)
│   ├── campbellv_1h.json            # Irrigation (validation)
│   ├── davisv_1h.json               # Rainfall + ET0 (validation)
│   └── irrimeterv_1h.json           # Soil moisture (validation)
├── paper/
│   ├── 03_24_2026_paper.tex          # LaTeX source
│   ├── PINN.bib                      # Bibliography
│   ├── sn-jnl.cls                    # Nature journal class
│   ├── sn-nature.bst                 # Bibliography style
│   ├── estimation_vs_validation_with_plant_uptake.pdf  # Fig. 2
│   ├── error_analysis_heatmap_boxplot.pdf              # Fig. 3
│   ├── pinn_functions_vertical_by_layer.pdf            # Fig. 4
│   ├── PINNsmb.pdf                   # Fig. 1a
│   └── sensorsmillenniumb.pdf        # Fig. 1b
├── weights/
│   └── pinn_weights_H56_lr7.97e-05.weights.h5   # Trained weights (H=56, lr=7.97e-5)
└── README.md
```

## Verified Hyperparameters

These are the **actual** hyperparameters that produced the published results:

| Parameter | Value | Paper Symbol |
|-----------|-------|-------------|
| Neurons per layer | **56** | H |
| Learning rate (Stage 1) | **7.97e-5** | η₁ |
| Learning rate (Stage 2) | **5e-6** | η₂ |
| λ\_c1 (non-negativity) | **113,982,766** | λ\_c1 |
| λ\_c2 (monotonicity) | **1,773,295** | λ\_c2 |
| λ\_c3 (zero-flow) | **126,704,656** | λ\_c3 |
| Epochs Stage 1 | **3000** | — |
| Epochs Stage 2 | **1000** | — |
| Decay steps | **1000** | — |
| Decay rate | **0.9** | — |

## How to Reproduce

### 1. Train the model
```bash
cd FINAL_SUBMISSION/code
python pinn_training.py
```
Weights are saved to `../weights/pinn_weights_H56_lr7.97e-05.weights.h5`.

### 2. Generate figures and table
Open `code/paper_figures.ipynb` in Jupyter and run all cells.
The notebook loads weights from `../weights/` and data from `../data/`.

### 3. Compile the paper
```bash
cd FINAL_SUBMISSION/paper
pdflatex 03_24_2026_paper.tex
bibtex 03_24_2026_paper
pdflatex 03_24_2026_paper.tex
pdflatex 03_24_2026_paper.tex
```

## Source Code Lineage

- **Training script**: derived from `Pinparameb.py` (neurons=56, no fl6 branch)
- **NOT** from `Pinparame.py` (neurons=79, has fl6 — older version, never used for final results)
- Hyperparameters from Optuna Stage 1 (`optuna_stage1b.py`, 200 trials) + Stage 2 (`PINN_optuna_stage2.py`, 30 trials)