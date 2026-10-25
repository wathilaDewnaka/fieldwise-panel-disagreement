"""Robustness, underfit fold-0 unit: the Chákṣu margin over Soft-KL with the
Soft-KL unit that failed the floor check (seed 44, fold 0) left out."""
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2].as_posix()  # repository root
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import ms_print_w03 as P
for r, v in P.seed_pool("prim", "Chaksu", drop={("B_kl", 44, 0)}).items():
    if r in ("B_kl", "P_emd"): print(r, v)
