"""The framework must not know what the images are.

This is the framework's "dataset-agnostic by construction" expressed as a test
rather than an assertion in a document.

The check is on *executable code* — identifiers, label constants, string values —
not on prose: comments and docstrings may use the cohorts as examples.
"""
import io
import pathlib
import re
import tokenize

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOMAIN_WORDS = re.compile(r"\b(nodule|lidc|lung|radiologist|malignan\w*|dicom|pylidc)\b",
                          re.IGNORECASE)

# Empty, and it must stay that way: every file under `field_disagreement/` is now free of
# domain vocabulary in its executable code. `splits.py` was the last entry — its
# BENIGN / MALIGNANT constants are now NEGATIVE / POSITIVE and its subject column
# is resolved through `splits.subjects`, which accepts the canonical
# `subject_id` as well as the `patient_id` a cached manifest still carries.
KNOWN_IMPURE: set[str] = set()


def _code_only(path: pathlib.Path) -> str:
    """Source with comments and string literals (incl. docstrings) removed."""
    src = path.read_text(encoding="utf-8")
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    except tokenize.TokenError:                      # pragma: no cover
        return src
    return " ".join(out)


def _framework_files():
    return sorted((ROOT / "field_disagreement").rglob("*.py"))


def test_no_domain_vocabulary_in_framework_code():
    offenders = {p.name for p in _framework_files()
                 if DOMAIN_WORDS.search(_code_only(p))}
    leaked = offenders - KNOWN_IMPURE
    assert not leaked, f"domain vocabulary leaked into field_disagreement/ code: {sorted(leaked)}"
    assert offenders == KNOWN_IMPURE, (
        "these are now clean and should be removed from KNOWN_IMPURE: "
        f"{sorted(KNOWN_IMPURE - offenders)}"
    )


def test_framework_never_imports_the_project():
    for p in _framework_files():
        src = p.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+backend\b", src, re.M), p
