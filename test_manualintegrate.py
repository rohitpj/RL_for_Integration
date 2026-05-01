"""
Unified Testing Script for:
 - RL Integration
 - Transformer Integration
 - Manual Integration

Outputs:
 - Data/rl_test_results.csv
 - Data/transformer_test_results.csv
 - Data/manual_test_results.csv
 - Data/all_integration_results.csv (combined comparison)
"""

import torch
import sympy
import pandas as pd
import numpy as np
import tqdm
import gc
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
import sympy
import ast
import copy

sys.path.append("SymbolicMathematics")
from SymbolicMathematics.src.envs import char_sp
from SymbolicMathematics.main import get_parser
from sympy import sympify
from sympy import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
#from sympy.integrals import manualintegrate
import SymbolicMathematics
from transformers import DataCollatorWithPadding
import numexpr
import tqdm

from SymbolicMathematics.src.envs import char_sp
from SymbolicMathematics.main import get_parser



from manualintegrate import manual_test, manualintegrate

from sklearn.preprocessing import LabelEncoder

with open("Utils/vocab.txt", "r") as f:
    vocab = [line.strip() for line in f if line.strip()]

label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("Utils/classes.npy", allow_pickle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


test_df = pd.read_csv("Data/test.csv").sample(frac=1, random_state=44).iloc[:500]
test_expressions = test_df["function"].tolist()
var = sympy.Symbol("x")

manual_results = []
combined_results = []

SAVE_INTERVAL = 1000  


for i, expr in enumerate(tqdm.tqdm(test_expressions, desc="Testing ALL Integrators")):

    try:
        m_result, m_status,  m_steps,m_branches, m_err, m_length = manual_test(expr, var)
        m_output = str(m_result)
    except Exception as e:
        print(f"Manual ERROR: {expr} -> {e}")
        m_output, m_status, m_branches, m_steps, m_length = None, "Error", None, None, None

    manual_results.append({
        "expression": expr,
        "manual_output": m_output,
        "manual_status": m_status,
        "manual_length": m_length,
        "manual_steps": m_steps,
        "manual_branches": m_branches
    })

    if (i + 1) % SAVE_INTERVAL == 0:
        print(f"\n Saving intermediate results at {i+1} expressions…")
        
        pd.DataFrame(manual_results).to_csv("Results/manual_test_results.csv", index=False)



print("\n=== FINAL SAVE OF ALL RESULTS ===")

pd.DataFrame(manual_results).to_csv("Results/only_manual_test_results.csv", index=False)

print("\n=== COMPLETE ===")

manual_results = pd.DataFrame(manual_results)

manual_total = len(manual_results)

manual_correct = (manual_results["manual_status"] == "Correct").sum()
manual_incorrect = (manual_results["manual_status"] == "Incorrect").sum()
manual_none = (manual_results["manual_status"] == "NoneResult").sum()

avg_manual_steps = manual_results['manual_branches'].mean()
avg_manual_rules = manual_results['manual_steps'].mean()

print("\n--- Manual Integration ---")
print(f"Total expressions tested: {manual_total}")
print(f"Correct:     {manual_correct} ({manual_correct/manual_total*100:.2f}%)")

print(f"Avg steps takens:      {avg_manual_rules:.3f}")
print(f"Avg branches: {avg_manual_steps:.3f}")

print("\n" + "="*80)
print("END OF SUMMARY")
print("="*80)

gc.collect()
torch.cuda.empty_cache()
