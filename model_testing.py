
import sys
import time
import gc
import argparse

import torch
import sympy
import pandas as pd
import tqdm

sys.path.append("SymbolicMathematics")
from sympy import sympify
from sympy.functions.elementary.hyperbolic import (
    sinh, cosh, tanh, coth, sech, csch,
    asinh, acosh, atanh, acoth, asech, acsch,
)
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from rl_integrate import rl_integrate, initialize_runtime
from transformer_integrate import transformer_integrate
from manualintegrate import manual_test

TIMEOUT = 30.0
STATUS_MAP = {0: "Correct", 1: "Incorrect", 2: "DontKnowRule"}
HYPERBOLIC_FUNCS = (
    sinh, cosh, tanh, coth, sech, csch,
    asinh, acosh, atanh, acoth, asech, acsch,
)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=384):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Shape (1, max_len, d_model)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)
    
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, num_classes=24, max_len=384):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x, attention_mask):
        x = self.embedding(x) 
        x = self.pos_encoding(x)

        if attention_mask is not None:
            attention_mask = attention_mask == 0 

        x = self.transformer_encoder(x, src_key_padding_mask=attention_mask)
        x = x.mean(dim=1) 
        x = self.fc_out(x)

        return x

class CLS_TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, num_classes=24, max_len=384):
        super(CLS_TransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x, attention_mask):
        x = self.embedding(x)  
        x = self.pos_encoding(x) 

        if attention_mask is not None:
            attention_mask = attention_mask == 0  

        x = self.transformer_encoder(x, src_key_padding_mask=attention_mask)  

        cls_representation = x[:, 0, :]

        logits = self.fc_out(cls_representation)

        return logits
    
def has_hyperbolic_trig(expr) -> bool:
    try:
        return expr.has(*HYPERBOLIC_FUNCS)
    except Exception:
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate integration methods on a dataset")
    parser.add_argument(
        "--methods", nargs="+",
        choices=["rl", "transformer", "manual", "all"],
        default=["all"],
    )
    parser.add_argument("--dataset", default="Data/csv_out/BWD_test.csv")
    parser.add_argument("--expr-col", default="function")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--save-interval", type=int, default=10)
    return parser.parse_args()


def build_selected_methods(method_args):
    return {"rl", "transformer", "manual"} if "all" in method_args else set(method_args)


def _run_one(prefix, fn, expr, var):
    """Call fn(expr, var) → (output, status, steps, branches, length), with error handling."""
    start = time.perf_counter()
    try:
        output, status, steps, branches, length = fn(expr, var)
        elapsed = time.perf_counter() - start
        if elapsed > TIMEOUT:
            return _timeout_row(prefix)
        return {
            f"{prefix}_output":       str(output),
            f"{prefix}_status":       STATUS_MAP.get(status, status),
            f"{prefix}_length":       length,
            f"{prefix}_steps":        steps,
            f"{prefix}_branches":     branches,
            f"{prefix}_time_seconds": elapsed,
        }
    except KeyboardInterrupt:
        print(f"\n[{prefix}] Interrupted: {expr}")
        return _timeout_row(prefix)
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"[{prefix}] Error on {expr!r}: {e}")
        return {
            f"{prefix}_output":       None,
            f"{prefix}_status":       "Error",
            f"{prefix}_length":       None,
            f"{prefix}_steps":        None,
            f"{prefix}_branches":     None,
            f"{prefix}_time_seconds": min(elapsed, TIMEOUT),
        }


def _timeout_row(prefix):
    return {
        f"{prefix}_output":       "DontKnow",
        f"{prefix}_status":       "DontKnowRule",
        f"{prefix}_length":       None,
        f"{prefix}_steps":        None,
        f"{prefix}_branches":     None,
        f"{prefix}_time_seconds": TIMEOUT,
    }


def run_method(prefix, fn, expressions, var, results_list, combined_map, save_fn, save_interval):
    for i, expr in enumerate(tqdm.tqdm(expressions, desc=f"Testing {prefix}")):
        row = _run_one(prefix, fn, expr, var)
        results_list.append({"expression": expr, **row})
        combined_map[expr].update(row)
        if (i + 1) % save_interval == 0:
            save_fn()


def save_results(selected_methods, rl_results, transformer_results, manual_results, combined_map):
    if "rl" in selected_methods and rl_results:
        pd.DataFrame(rl_results).to_csv("results/rl_test_results.csv", index=False)
    if "transformer" in selected_methods and transformer_results:
        pd.DataFrame(transformer_results).to_csv("results/transformer_test_results.csv", index=False)
    if "manual" in selected_methods and manual_results:
        pd.DataFrame(manual_results).to_csv("results/manual_test_results.csv", index=False)
    if len(selected_methods) > 1:
        pd.DataFrame(list(combined_map.values())).to_csv("results/all_integration_results.csv", index=False)


def print_summary(selected_methods, rl_results, transformer_results, manual_results):
    method_results = {
        "manual":      (manual_results,      "manual"),
        "transformer": (transformer_results, "transformer"),
        "rl":          (rl_results,          "rl"),
    }
    for method in ["manual", "transformer", "rl"]:
        if method not in selected_methods:
            continue
        results, prefix = method_results[method]
        if not results:
            continue
        df = pd.DataFrame(results)
        total = len(df)
        correct = (df[f"{prefix}_status"] == "Correct").sum()
        print(f"\n--- {method.title()} Integration ---")
        print(f"Total: {total}  |  Correct: {correct} ({correct / total * 100:.2f}%)")
        for label, col in [
            ("Avg length",   f"{prefix}_length"),
            ("Avg steps",    f"{prefix}_steps"),
            ("Avg branches", f"{prefix}_branches"),
        ]:
            val = df[col].mean() if col in df and df[col].notna().any() else None
            print(f"{label}: {val:.3f}" if val is not None else f"{label}: N/A")
        t_col = f"{prefix}_time_seconds"
        if t_col in df and df[t_col].notna().any():
            print(f"Avg time (s): {df[t_col].dropna().mean():.4f}")
        else:
            print("Avg time (s): N/A")
    print("\n" + "=" * 80)
    print("END OF SUMMARY")
    print("=" * 80)


def main():
    args = parse_args()
    selected_methods = build_selected_methods(args.methods)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = transformer_model = None

    if "rl" in selected_methods:
        actor_path='Models/rl_model/updated_rl_actor.pt',
        critic_path='Models/rl_model/updated_rl_critic.pt',
        agent, _, _ = initialize_runtime(actor_path=actor_path, critic_path=critic_path)
        agent.eval()

    if "transformer" in selected_methods:
        transformer_model = torch.load("Models/full_15_epochs.pt", map_location=device, weights_only=False)
        transformer_model.eval()

    test_df = pd.read_csv(args.dataset).iloc[:args.sample_size].reset_index(drop=True)
    test_df["expr_len"] = test_df[args.expr_col].astype(str).str.len().iloc[:args.sample_size]
    """
    test_df = (
        test_df.sort_values("expr_len")
               .iloc[500:args.sample_size + 500]
               .reset_index(drop=True)
    )
    """
    print(test_df.head())

    var = sympy.Symbol("x")
    expressions = [
        expr for expr in test_df[args.expr_col].tolist()
        if not has_hyperbolic_trig(sympify(expr))
    ]

    rl_results, transformer_results, manual_results = [], [], []
    combined_map = {expr: {"expression": expr} for expr in expressions}

    def do_save():
        save_results(selected_methods, rl_results, transformer_results, manual_results, combined_map)

    if "transformer" in selected_methods:
        # normalise return order to (output, status, steps, branches, length)
        def t_fn(expr, var):
            result, status, steps, branches, length = transformer_integrate(transformer_model, expr, var)
            return result, status, steps, branches, length

        run_method("transformer", t_fn, expressions, var,
                   transformer_results, combined_map, do_save, args.save_interval)
    
    if "manual" in selected_methods:
        # manual_test already returns (result, status, steps, branches, length)
        run_method("manual", manual_test, expressions, var,
                   manual_results, combined_map, do_save, args.save_interval)

    if "rl" in selected_methods:
        # rl_integrate returns (out, status, length, steps, branches) — reorder
        def rl_fn(expr, var):
            out, status, length, steps, branches = rl_integrate(agent, expr, var)
            return out, status, steps, branches, length

        run_method("rl", rl_fn, expressions, var,
                   rl_results, combined_map, do_save, args.save_interval)

    print("\n=== FINAL SAVE ===")
    do_save()

    print_summary(selected_methods, rl_results, transformer_results, manual_results)

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
