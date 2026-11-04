"""
Bayesian hierarchical model for studio film ROI.

Target: sqrt(roi + 1)
Scope:  major studios only (studio_group != 'Independent')
Window: 1990-2017

Fixed effects:
  - 17 binary genre indicators (additive)
  - budget tier (5 levels, Micro baseline)
  - IP type (4 levels, Original baseline)
  - release season (5 levels, Summer baseline)
  - decade (3 levels, 1990s baseline)

Random effects:
  - studio_group (partial pooling)

Models:
  M1 - genre + budget + IP (baseline)
  M2 - M1 + season + decade + studio random effects (full)
"""

import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import pickle
import os

DATA_PATH = "/Users/theoauyeung/Documents/Projects/425 Final Project/data/cleaned/movies_clean.csv"
OUT_DIR   = "/Users/theoauyeung/Documents/Projects/425 Final Project/data/model_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load & prepare ─────────────────────────────────────────────────────────


# ── 2. M1: genre + budget + IP (baseline) ────────────────────────────────────


# ── 3. M2: + season + decade + studio random effects (full) ──────────────────


# ── 4. WAIC comparison ────────────────────────────────────────────────────────


# ── 5. Posterior summaries ────────────────────────────────────────────────────


# ── 6. Save traces ────────────────────────────────────────────────────────────
