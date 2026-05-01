"""
Shared preprocessing utilities used by all training and inference pipelines.

Canonical behaviours:
  - Prefix converter built with parse_args([]) — never reads sys.argv.
  - _u and _theta are substituted with x before prefix conversion.
  - No .isdigit() filtering — digit tokens are kept (consistent with filtered_converted.csv).
  - safe_eval always returns list[str].
"""

import ast
import sys

sys.path.append("SymbolicMathematics")

from sympy import Symbol, sympify
from SymbolicMathematics.src.envs import char_sp
from SymbolicMathematics.main import get_parser


_U = Symbol("_u")
_THETA = Symbol("_theta")
_X = Symbol("x")


def build_prefix_converter():
    """Return a CharSPEnvironment built from default SymbolicMathematics params."""
    parser = get_parser()
    params = parser.parse_args([])
    return char_sp.CharSPEnvironment(params)


def string_to_sympy(raw_expr):
    """Convert a string or SymPy expression to a normalised SymPy expression.

    Substitutes _u and _theta with x. Accepts an already-parsed SymPy object
    so it is safe to call during RL inference where the integrand is not a string.
    """
    sympy_expr = sympify(raw_expr) if isinstance(raw_expr, str) else raw_expr
    return sympy_expr.subs(_U, _X).subs(_THETA, _X)


def expr_to_tokens(prefix_converter, expr) -> list[str]:
    """Convert an expression (str or SymPy) to a list of prefix-notation tokens.

    Uses the supplied prefix_converter so the converter can be reused across
    many calls without being rebuilt each time.
    """
    sympy_expr = string_to_sympy(expr)
    return prefix_converter.sympy_to_prefix(sympy_expr)


def preprocess_expr(raw_expr) -> list[str] | None:
    """One-shot convenience wrapper: build converter, convert, return tokens or None."""
    try:
        return expr_to_tokens(build_prefix_converter(), raw_expr)
    except Exception:
        return None


def safe_eval(x) -> list[str]:
    """Parse a stringified Python list from a CSV cell into a list of strings."""
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
            items = val if isinstance(val, list) else [val]
            return [str(t) for t in items]
        except Exception:
            return []
    if isinstance(x, (int, float)):
        return [str(x)]
    return []


def build_vocab(token_lists: list[list[str]]) -> list[str]:
    """Return a sorted list of unique tokens across all token lists."""
    return sorted({tok for toks in token_lists for tok in toks})


def preprocess_dataset(df, expr_col: str, rule_col: str, prefix_converter=None):
    """Tokenize df[expr_col] to prefix tokens; return (token_lists, labels).

    Rows that fail conversion are silently skipped. Pass an existing
    prefix_converter to avoid rebuilding it for every dataset.
    """
    from tqdm.auto import tqdm

    if prefix_converter is None:
        prefix_converter = build_prefix_converter()

    token_lists, labels = [], []
    rows = df[[expr_col, rule_col]]
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="Tokenizing", unit="expr"):
        try:
            token_lists.append(expr_to_tokens(prefix_converter, str(row[expr_col])))
            labels.append(str(row[rule_col]))
        except Exception:
            continue

    if not token_lists:
        raise RuntimeError("No valid rows after tokenization.")
    return token_lists, labels
