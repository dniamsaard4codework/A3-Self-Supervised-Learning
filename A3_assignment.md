---
# Exercises

1. Train the following DINO variants and fill in the table:

   | Setting | Linear Eval Accuracy |
   |---|---|
   | Default (2 global + 4 local, with centering) | ? |
   | No centering (`- self.center` removed) | ? |
   | No local crops (`n_local=0`) | ? |

   a) Plot `dino_loss_fn.center.norm()` across training epochs. Does it grow, shrink, or stabilize?

   b) Explain why removing centering causes collapse, and why removing local crops hurts representation quality.

2. Train three MAE versions with `mask_ratio` ∈ {0.25, 0.50, 0.75} for 50 epochs each and fill in the table:

   | Mask Ratio | Recon Loss | Linear Eval Acc |
   |---|---|---|
   | 0.25 | ? | ? |
   | 0.50 | ? | ? |
   | 0.75 | ? | ? |

   Explain why very low masking (e.g. 0.25) produces worse representations even though reconstruction loss is lower.

3. Fill in the three-way comparison table using results from Parts 1–3:

   | Metric | SimCLR | DINO | MAE |
   |---|---|---|---|
   | Backbone | ResNet-18 | ViT-Tiny | ViT |
   | Needs negative pairs? | Yes | No | No |
   | Needs EMA teacher? | No | Yes | No |
   | Linear Eval Accuracy | ? | ? | ? |
   | Training time/epoch | ? | ? | ? |
   | t-SNE cluster quality (1–5) | ? | ? | ? |
   | Has interpretable attention maps? | No | Yes | No |

   a) Give two reasons why MAE won out over DINO for large-scale general pre-training, and one reason DINO is still preferred for CV-only tasks like segmentation.

   b) You are building a medical image segmentation system with 500 labeled scans. Which pre-training approach would you choose and why?

   ---
## Submission

Submit your work to GitHub. Your repository should contain:

### 1. Training Script (`run.py`)

```bash
# Train
python3 run.py --model dino   --epochs 50 --train
python3 run.py --model mae    --epochs 50 --train

# Linear evaluation
python3 run.py --model dino   --weights saved/dino.pt        --evaluate --linear
python3 run.py --model mae    --weights saved/mae_encoder.pt --evaluate --linear

# Ablations
python3 run.py --model dino --no-centering --epochs 50 --train
python3 run.py --model dino --n-local 0 --epochs 50 --train
python3 run.py --model mae  --mask-ratio 0.25 --epochs 50 --train
python3 run.py --model mae  --mask-ratio 0.50 --epochs 50 --train
```

### 2. `README.md`

**Results table:**

| Model | Linear Eval Acc | Time/epoch | Notes |
|---|---|---|---|
| DINO (Default) | ? | ? | self-distillation |
| DINO (no centering) | ? | ? | collapse ablation |
| DINO (no local crops) | ? | ? | multi-crop ablation |
| MAE mask=0.75 | ? | ? | reconstruction |
| MAE mask=0.50 | ? | ? | masking ablation |
| MAE mask=0.25 | ? | ? | masking ablation |

**Visualizations:**
- Loss curves for DINO, and MAE
- MAE reconstruction grid (original / masked / reconstructed)
- DINO attention map grid (10 images × all heads, from original exercises)
- t-SNE comparison: DINO vs MAE

**Discussion** (3–5 sentences): For a medical image segmentation project with limited labels, which approach would you choose and why?