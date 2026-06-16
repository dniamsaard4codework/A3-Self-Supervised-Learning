"""A3 Self-Supervised Learning package.

Reusable implementations of SimCLR, DINO, and MAE for CIFAR-10, refactored from
the class lab note (``lab_note/A3-Self-Supervised-Learning.ipynb``). Architectures
and hyperparameters are kept identical to the lab; the only additions are ablation
toggles (DINO ``use_centering`` / ``n_local``, MAE ``mask_ratio``) and AMP support.
"""
