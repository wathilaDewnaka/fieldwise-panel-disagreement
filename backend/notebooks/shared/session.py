"""Which hosted runtime this is, and stopping a run cleanly inside its time limit.

A free GPU session has a hard wall-clock limit (Kaggle: 12 h per session). A fold
that is still training when the limit hits is killed, and on Kaggle a version that
is killed by the limit does not keep its output - so every result finished in that
session would go with it. The fix is to never START a unit that cannot finish:

    start_budget(hours)      set the session's deadline (once per process)
    check_budget(store, tag) raise SessionBudgetReached before a NEW unit when the
                             time left is shorter than the longest unit so far
    record_unit_time(...)    remember how long each unit took, in the run folder,
                             so the next session's first check already knows

With no deadline set - a local run, Colab - `check_budget` returns immediately and
reads nothing, so those runs behave exactly as before.

Raising stops the cell, but a raised cell would end a Kaggle "Save & Run All" as an
ERROR. `install_budget_stop` turns the stop into a clean finish instead: from then on
every remaining cell is skipped with a note, and the version completes successfully
with everything finished so far in its output. Nothing about how a unit trains is
touched; this only decides whether the next one starts.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import time
from pathlib import Path

__all__ = ["SessionBudgetReached", "platform", "start_budget", "deadline",
           "check_budget", "record_unit_time", "load_unit_times",
           "install_budget_stop", "release_runtime", "DEADLINE_ENV",
           "TIMES_FILE"]

DEADLINE_ENV = "FDP_SESSION_DEADLINE"
TIMES_FILE = "_unit_seconds.json"
_GUARD = "__field_disagreement_session_guard__"


class SessionBudgetReached(RuntimeError):
    """The next unit would not finish inside this session's time budget."""


# --------------------------------------------------------------------------- #
# the runtime
# --------------------------------------------------------------------------- #
def _has_colab_module() -> bool:
    """`google.colab` is importable. find_spec raises, rather than returning None,
    when the parent `google` package is missing, so the lookup is guarded."""
    import importlib.util
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError):
        return False


def platform() -> str:
    """'kaggle', 'colab' or 'local'. The notebook's setup flags set
    `FDP_PLATFORM`, which wins; otherwise the platform is detected."""
    forced = os.environ.get("FDP_PLATFORM", "").strip().lower()
    if forced in ("kaggle", "colab", "local"):
        return forced
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/input").is_dir():
        return "kaggle"
    if ("COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ
            or "google.colab" in sys.modules or _has_colab_module()):
        return "colab"
    return "local"


def release_runtime(delay: float = 5.0) -> None:            # pragma: no cover
    """Hand the GPU back once every result is saved.

    Colab: disconnect the VM. Kaggle: nothing to do - a "Save & Run All" version
    stops on its own when the last cell ends, and an interactive session is
    stopped from the editor. Local: nothing to do."""
    where = platform()
    if where == "colab":
        print("All experiments finished; results and checkpoints are on Drive.")
        print("Releasing the Colab runtime...")
        sys.stdout.flush()
        time.sleep(delay)                 # let the final log lines flush
        try:
            from google.colab import runtime
            runtime.unassign()
        except Exception as e:            # an older runtime without `unassign`
            print(f"Could not release automatically ({type(e).__name__}: {e}).\n"
                  "Release it by hand: Runtime > Disconnect and delete runtime.")
    elif where == "kaggle":
        print("All experiments finished; results are under /kaggle/working/runs.")
        print("Kaggle: a Save & Run All version stops by itself now. In an "
              "interactive session, click the power icon (Stop session).")
    else:
        print("All experiments finished (local run - nothing to release).")


# --------------------------------------------------------------------------- #
# the budget
# --------------------------------------------------------------------------- #
def start_budget(hours: float, *, now: float | None = None) -> float:
    """Set this session's deadline `hours` from now and return it.

    Set ONCE per process: re-running the setup cell must not buy the session
    time it does not have."""
    existing = deadline()
    if existing is not None:
        return existing
    value = (time.time() if now is None else now) + float(hours) * 3600.0
    os.environ[DEADLINE_ENV] = repr(value)
    return value


def deadline() -> float | None:
    """The deadline as a UNIX time, or None when no budget is in force."""
    raw = os.environ.get(DEADLINE_ENV)
    return float(raw) if raw else None


def _times_path(store) -> Path:
    return Path(store.result_dir) / TIMES_FILE


def load_unit_times(store) -> dict:
    """{tag: seconds} for every unit this run folder has trained."""
    path = _times_path(store)
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def record_unit_time(store, tag: str, seconds: float) -> None:
    """Remember how long `tag` took. Never allowed to fail a finished unit."""
    try:
        times = load_unit_times(store)
        times[str(tag)] = round(float(seconds), 1)
        _times_path(store).write_text(json.dumps(times, indent=1, sort_keys=True))
    except OSError:                                          # pragma: no cover
        pass


def check_budget(store, tag: str, *, now: float | None = None) -> None:
    """Raise SessionBudgetReached if `tag` should not start in this session.

    A unit starts when time is left AND the longest unit this run folder has ever
    recorded fits in it. Before any unit has been timed only the first condition
    applies - the first unit of a fresh study is the one that measures the cost."""
    end = deadline()
    if end is None:
        return
    left = end - (time.time() if now is None else now)
    times = list(load_unit_times(store).values())
    need = max(times) if times else 0.0
    if left > 0 and need <= left:
        return
    raise SessionBudgetReached(
        f"[session] {tag} not started: {max(left, 0) / 3600:.1f} h left in this "
        f"session's budget and the longest unit so far took {need / 3600:.1f} h. "
        f"Every finished unit is saved in the run folder. Start a new session to "
        f"continue - on Kaggle: attach this version's output as an input and "
        f"Save & Run All again; finished units load from cache.")


# --------------------------------------------------------------------------- #
# turning the stop into a clean finish (IPython only)
# --------------------------------------------------------------------------- #
class _Guard:
    """What the transformed cells call. One per shell, held in its namespace."""

    def __init__(self):
        self.stopped: str | None = None

    def active(self) -> bool:
        if self.stopped is None:
            return True
        print("[session] cell skipped - the session budget was reached above.")
        return False

    def absorb(self, exc) -> bool:
        """True (and stop every later cell) if `exc` is the budget stop. Matched
        by NAME: the staged repo is re-imported mid-session, so the class raised
        need not be this module's."""
        if type(exc).__name__ != "SessionBudgetReached":
            return False
        self.stopped = str(exc)
        print(str(exc))
        print("[session] every remaining cell will be skipped; this run ends cleanly.")
        return True

    def value(self, fn):
        if self.stopped is not None:
            return None
        try:
            return fn()
        except BaseException as exc:                          # noqa: BLE001
            if self.absorb(exc):
                return None
            raise


class _StopRemainingCells(ast.NodeTransformer):
    """cell  ->  if guard.active(): try: <cell> except budget stop: absorb

    A trailing expression stays a top-level expression (evaluated through the
    guard) so the notebook still displays it."""

    _UNSAFE_TAIL = (ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)

    def visit_Module(self, node):
        body = list(node.body)
        if not body:
            return node
        tail = None
        if isinstance(body[-1], ast.Expr) and not any(
                isinstance(n, self._UNSAFE_TAIL) for n in ast.walk(body[-1])):
            tail = body.pop()

        def guard(attr):
            return ast.Attribute(value=ast.Name(id=_GUARD, ctx=ast.Load()),
                                 attr=attr, ctx=ast.Load())

        new = []
        if body:
            exc = "__field_disagreement_exc__"
            handler = ast.ExceptHandler(
                type=ast.Name(id="BaseException", ctx=ast.Load()), name=exc,
                body=[ast.If(test=ast.Call(func=guard("absorb"),
                                           args=[ast.Name(id=exc, ctx=ast.Load())],
                                           keywords=[]),
                             body=[ast.Pass()], orelse=[ast.Raise()])])
            wrapped = ast.If(
                test=ast.Call(func=guard("active"), args=[], keywords=[]),
                body=[ast.Try(body=body, handlers=[handler], orelse=[],
                              finalbody=[])],
                orelse=[])
            new.append(ast.copy_location(wrapped, body[0]))
        if tail is not None:
            lam = ast.Lambda(args=ast.arguments(posonlyargs=[], args=[], vararg=None,
                                                kwonlyargs=[], kw_defaults=[],
                                                kwarg=None, defaults=[]),
                             body=tail.value)
            call = ast.Expr(value=ast.Call(func=guard("value"), args=[lam], keywords=[]))
            if not body:
                # a lone expression still has to be skipped once stopped
                call = ast.Expr(value=ast.IfExp(
                    test=ast.Call(func=guard("active"), args=[], keywords=[]),
                    body=call.value, orelse=ast.Constant(value=None)))
            new.append(ast.copy_location(call, tail))
        node.body = new
        return ast.fix_missing_locations(node)


def install_budget_stop(shell=None) -> bool:
    """Make a budget stop end the notebook cleanly. Returns False outside IPython.

    Idempotent: re-running the setup cell does not stack transformers."""
    if shell is None:
        try:
            from IPython import get_ipython
        except ImportError:                                   # pragma: no cover
            return False
        shell = get_ipython()
    if shell is None:
        return False
    if any(type(t).__name__ == "_StopRemainingCells" for t in shell.ast_transformers):
        return True
    shell.user_ns[_GUARD] = _Guard()
    shell.ast_transformers.append(_StopRemainingCells())
    return True
