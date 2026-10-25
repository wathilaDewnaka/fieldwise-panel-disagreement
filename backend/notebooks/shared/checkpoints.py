"""Resumable checkpoints for free-GPU runs (Colab / Kaggle).

Those runtimes time out and disconnect, so training has to survive being killed
mid-run. Each epoch the FULL training state (model + optimiser + schedule + AMP
scaler + RNG) is written atomically, so an interrupted fold resumes from the
exact epoch it stopped on; best-so-far weights live in a separate file, and a
finished unit of work writes a cached result so re-running skips it.

`CheckpointStore` holds the three things the old per-notebook copies read off
notebook globals - the directories, the model name, and the device - so the file
layout is unchanged:

    <ckpt_dir>/<model>_<tag>.ckpt        mid-run resume state
    <ckpt_dir>/<model>_<tag>.best.pt     best-so-far weights
    <result_dir>/<model>_<tag>.result.json   the finished result

`tag` identifies the unit of work (a fold, or mode/arm + fold). `result_dir`
defaults to `ckpt_dir`.

Checkpoints live on DRIVE, not on the Colab VM's local disk. Local disk is faster
and never touches Trash, but it is gone when the session hits its limit - which
is precisely when the resume is needed - so the whole point of writing them would
be lost.

DRIVE TRASH: overwriting a file on a Drive mount does not overwrite it. The
displaced copy goes to Trash and keeps counting against the quota for ~30 days,
so saving state every epoch leaks one full checkpoint per epoch. Deleting a
file does the same thing.

So every write and every delete here is followed by a permanent-delete through
the Drive API. That immediate attempt is not enough on its own, because the mount
syncs ASYNCHRONOUSLY and the displaced copy usually reaches Trash a few seconds
after the local write returns - a purge fired straight away finds nothing. Names
are therefore QUEUED as well, and `sweep()` retries the whole queue when a fold
finishes, reporting how many copies it actually removed. A purge that silently
does nothing is how ~20 GB accumulated unnoticed.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

import random

import numpy as np
import torch


class CheckpointStore:
    """File-backed checkpoint/result store for one run.

    `keep_best=True` retains each unit's best weights after it finishes, which
    makes a later re-scoring an inference-only job instead of a retrain. It is
    off by default because on a Drive mount those are the files that eat the
    quota - which is the whole reason `rm` exists.

    `keep_ckpt=True` retains the RESUME STATE of a finished unit as well, and
    only the two together buy the re-score. `Runner.fit` returns early - without
    training a single epoch - on a unit whose checkpoint says it already reached
    the epoch cap or exhausted patience; with the checkpoint dropped there is
    nothing for that test to read, so `fit` starts at epoch 0 and RETRAINS, and
    `load_best` then has nothing to restore either. Re-running to pick up a new
    metric is a pure re-score only with this flag on. It is off by default for the same quota reason: a kept checkpoint
    carries the optimiser state, so it is the largest file per unit.
    """

    def __init__(self, ckpt_dir, model: str, device, *, result_dir=None,
                 keep_best: bool = False, keep_ckpt: bool = False,
                 use_drive: bool = True):
        self.ckpt_dir = Path(ckpt_dir)
        self.result_dir = Path(result_dir) if result_dir is not None else self.ckpt_dir
        self.model = model
        self.device = device
        self.keep_best = keep_best
        self.keep_ckpt = keep_ckpt
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        # Every filename this run has written to Drive. `sweep()` retries the
        # whole set, which is what catches the copies the immediate purge raced
        # past; it is bounded by the number of distinct tags a run touches.
        self._pending: set[str] = set()
        self._drive = self._connect_drive() if use_drive else None

    # -- paths -------------------------------------------------------------- #
    def path(self, tag: str, kind: str) -> Path:
        home = self.result_dir if kind == "result.json" else self.ckpt_dir
        return home / f"{self.model}_{tag}.{kind}"

    # -- permanent delete ---------------------------------------------------- #
    @staticmethod
    def _connect_drive():
        """Authorise the Drive API up front, so the access prompt (the same one
        `drive.mount` shows) can never interrupt a training run."""
        if not Path("/content/drive").exists():
            return None
        try:
            from google.colab import auth
            from googleapiclient.discovery import build
            auth.authenticate_user()
            # Purging runs after every epoch's save, and each call logs the same
            # "httplib2 transport does not support per-request timeout" warning.
            # One copy per epoch buries the training log, so mute it.
            logging.getLogger("google_auth_httplib2").setLevel(logging.ERROR)
            return build("drive", "v3")
        except Exception as e:                                # pragma: no cover
            print(f"[trash] Drive API unavailable ({type(e).__name__}) - deleted "
                  f"checkpoints will sit in Trash; empty it at "
                  f"drive.google.com/drive/trash")
            return None

    def purge(self, p):                                       # pragma: no cover
        """Queue `p`'s displaced Trash copy for permanent deletion, then sweep.

        The delete is NOT done for `p` alone. The Drive mount syncs
        asynchronously, so the copy an overwrite displaces reaches Trash a few
        seconds AFTER the write returns - a delete fired for it right now finds
        nothing, which is how a finished run still had ~20 GB in Trash. Names
        stay queued instead, and every sweep retries all of them, so each write
        collects what the writes before it left behind."""
        p = Path(p)
        if self._drive is None or not str(p.resolve()).startswith("/content/drive/"):
            return
        self._pending.update((p.name, p.name + ".tmp"))
        self.sweep()

    def sweep(self) -> int:                                   # pragma: no cover
        """Permanently delete every queued name that has reached Drive Trash.

        Returns how many were removed, and says so - a purge that silently does
        nothing is how the leak went unnoticed. Deleting only after the listing
        is finished, because deleting mid-page skips half the matches. A failure
        here costs quota; raising would kill a training run, so it is caught."""
        if self._drive is None or not self._pending:
            return 0
        names, removed = sorted(self._pending), 0
        try:
            for i in range(0, len(names), 40):          # one request per 40 names
                q = " or ".join(f"name = '{n}'" for n in names[i:i + 40])
                found = self._drive.files().list(
                    q=f"({q}) and trashed = true", pageSize=1000,
                    fields="files(id)").execute().get("files", [])
                for f in found:
                    self._drive.files().delete(fileId=f["id"]).execute()
                    removed += 1
        except Exception as e:
            print(f"[trash] purge failed ({type(e).__name__}) - empty Trash "
                  f"manually at drive.google.com/drive/trash")
        if removed:
            print(f"[trash] permanently deleted {removed} displaced copy/copies")
        return removed

    def rm(self, p):
        """Delete `p` for good, rather than parking it in Drive Trash."""
        p = Path(p)
        if not p.exists():
            return
        p.unlink()
        self.purge(p)

    # -- writing ------------------------------------------------------------- #
    WRITE_ATTEMPTS = 4

    def _write(self, obj, dest) -> None:
        """`torch.save(obj, dest)`, atomic and robust to a flaky Drive mount.

        Observed on Colab: `best.pt` wrote fine and the
        `ckpt.tmp` written seconds later into the SAME folder died with
        `inline_container.cc: open file failed ... No such file or directory`.
        torch's zip writer opens and seeks on the target directly, and on the Drive
        FUSE mount a ~1 GB write can find its folder briefly missing while the
        previous file syncs. So:
          * serialise to LOCAL disk first (no seeks on the mount),
          * copy the finished file into `<dest>.tmp`, then `os.replace` it over
            `dest` (a killed write still cannot corrupt the resume state),
          * re-create the folder and retry with backoff before giving up, and say
            which path failed - the old error named no file at all."""
        dest = Path(dest)
        tmp = dest.with_name(dest.name + ".tmp")
        err = None
        for attempt in range(self.WRITE_ATTEMPTS):
            local = None
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                fd, local = tempfile.mkstemp(prefix="field_disagreement_", suffix=".pt")
                os.close(fd)
                torch.save(obj, local)
                shutil.copyfile(local, tmp)
                os.replace(tmp, dest)
                return
            except (OSError, RuntimeError) as e:
                err = e
                if attempt + 1 < self.WRITE_ATTEMPTS:
                    wait = 5 * 2 ** attempt
                    print(f"[ckpt] write to {dest} failed ({type(e).__name__}: "
                          f"{str(e).strip()[:120]}) - retry {attempt + 1}/"
                          f"{self.WRITE_ATTEMPTS - 1} in {wait}s")
                    time.sleep(wait)
            finally:
                if local is not None and os.path.exists(local):
                    os.remove(local)
        raise RuntimeError(
            f"could not write {dest} after {self.WRITE_ATTEMPTS} attempts. Check the "
            f"Drive mount is still up (`!ls '{dest.parent}'`) and that Drive is not "
            f"full.") from err

    # -- training state ------------------------------------------------------ #
    def save_ckpt(self, tag, model, opt, sched, scaler, next_epoch, best_auc,
                  since, extra=None):
        """Atomically persist the full training state (so a killed write cannot
        corrupt it)."""
        dest = self.path(tag, "ckpt")
        self._write({"epoch": next_epoch, "best_auc": best_auc, "since": since,
                    "extra": extra, "model": model.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": (scaler.state_dict() if scaler is not None else None),
                    "rng_torch": torch.get_rng_state(),
                    "rng_numpy": np.random.get_state(),
                    # the GPU and Python generators too, so a fold resumed
                    # mid-training continues the SAME random sequence (dropout
                    # masks are drawn on the GPU)
                    "rng_cuda": (torch.cuda.get_rng_state_all()
                                 if torch.cuda.is_available() else None),
                    "rng_python": random.getstate()}, dest)
        self.purge(dest)            # bin the epoch this one just superseded

    def save_best(self, tag, model, kind: str = "best.pt"):
        bp = self.path(tag, kind)
        self._write(model.state_dict(), bp)
        self.purge(bp)

    def load_best(self, tag, model, kind: str = "best.pt"):
        bp = self.path(tag, kind)
        if bp.exists():
            model.load_state_dict(torch.load(bp, map_location=self.device))

    def resume(self, tag, model, opt, sched, scaler):
        """Load a mid-run checkpoint in place. Returns
        (start_epoch, best_auc, since, extra); (0, -1.0, 0, None) when there is
        nothing to resume."""
        cp = self.path(tag, "ckpt")
        if not cp.exists():
            return 0, -1.0, 0, None
        # weights_only=False: this is our own checkpoint and it carries the
        # numpy/torch RNG state (not just tensors). torch>=2.6 defaults to True,
        # which would reject it - safe to disable for a file we wrote ourselves.
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        for st in opt.state.values():      # optimiser state onto the model's device
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(self.device)
        sched.load_state_dict(ck["sched"])
        if scaler is not None and ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        torch.set_rng_state(ck["rng_torch"])
        np.random.set_state(ck["rng_numpy"])
        if ck.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
        if ck.get("rng_python") is not None:
            random.setstate(ck["rng_python"])
        print(f"  [resume {tag}] from epoch {ck['epoch']}  "
              f"best_auc {ck['best_auc']:.4f}")
        return ck["epoch"], ck["best_auc"], ck["since"], ck.get("extra")

    # -- finished units of work ---------------------------------------------- #
    def cache_result(self, tag, result):
        """Mark a unit of work done: write its result and drop the resume files."""
        with open(self.path(tag, "result.json"), "w") as fh:
            json.dump(result, fh, indent=2)
        drop = tuple(kind for kind, keep in (("ckpt", self.keep_ckpt),
                                             ("best.pt", self.keep_best),
                                             ("band.pt", self.keep_best))
                     if not keep)
        for kind in drop:
            self.rm(self.path(tag, kind))
        # The deletes above, and every epoch's overwrite before them, may still
        # be in flight to Drive. This is the retry that actually collects them.
        self.sweep()

    def cached_result(self, tag, schema=None):
        """A finished unit's result, or None when it is missing or out of date.

        `schema` guards against a stale cache: bump it whenever the evaluation
        starts producing metrics an older result.json cannot contain. Without
        that check a re-run silently reloads the OLD result and every new column
        prints as "-", which reads exactly like "this arm could not produce that
        metric" rather than "you are looking at stale numbers"."""
        rp = self.path(tag, "result.json")
        if not rp.exists():
            return None
        with open(rp) as fh:
            res = json.load(fh)
        if schema is not None and res.get("schema") != schema:
            print(f"  [{tag}] cached result predates the current metric set "
                  f"(schema {res.get('schema')} != {schema}) -> re-running")
            return None
        return res
