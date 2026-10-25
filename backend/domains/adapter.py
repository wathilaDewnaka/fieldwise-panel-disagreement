"""torch Dataset and DataLoaders over the stored agreement-spectrum samples.

The ONE loader for every cohort — LIDC-IDRI, Chákṣu and LNDb read through this
file. It is not part of `field_disagreement/` because it hard-codes a storage
format (a flat directory of .npz files with fixed key names, two of them LIDC's
own vocabulary); `field_disagreement.spectrum.AgreementDataset` is the
format-neutral contract, and this class is one concrete implementation of it.

Each item is a dict:

    image          (3, S, S)   float32   the sample; S = the backbone's input size
                                         (224 + ImageNet-normalised for ResNet-50).
                                         Maps are resized to match.
    label          ()          long      binary class y_i  (negative 0 / positive 1)
    agreement_map  (S, S)      float32   A(x,y) raw annotator count in {0..R}
    agreement_frac (S, S)      float32   normalised agreement A(x,y)/R in [0,1]
    binary_mask    (S, S)      float32   union-of-annotators lesion mask
    opinion_dist   (K,)        float32   normalised histogram over the K rating
                                         levels. K is a property of the
                                         annotation protocol and varies by cohort
                                         - it must match the model's `n_ratings`.
    disagreement   ()          float32   spread of the panel (d_i). Its SCALE is
                                         cohort-specific, so any threshold on it
                                         (`contested_d`) is set per cohort.
    n_readers      ()          long      annotators who scored this sample

plus, on the report-level cohorts, the per-field targets (`d_profile`,
`opinion_fields`, ...).

Augmentation (train only) is a single spatial transform (90-deg rotations + axial
flips) applied IDENTICALLY to the image and every spatial map, so they stay in
one coordinate frame.

DECENTRING (`shift_frac`; 0.0 = off). The preprocessors crop every patch CENTRED on
the lesion, and rot90 + flips are all symmetries of the square about its centre, so
without it the image centre is the lesion centre in every sample. `shift_frac`
translates the lesion off-centre by a random integer offset of up to `shift_frac`
x the patch width. Three properties this implementation guarantees:

  * INTEGER-pixel translation at the native size BEFORE the resize - no
    interpolation, so the mask's pixel count is exactly preserved (asserted). The
    vacated band is REFLECTED tissue on the image and ZERO on the maps (see
    `_shift`): a zero-filled image band would encode the offset direction.
  * CLAMPED per sample so the full mask bounding box stays inside the crop.
  * DETERMINISTIC on val/test/score3 (offset hashed from the file name +
    `shift_seed`), random and fresh on train. Every arm therefore sees
    byte-identical test inputs, which keeps the per-item paired comparison valid.

Old-format files (no agreement_frac / opinion_dist / disagreement / n_readers) are
handled transparently by deriving the missing fields from the per-annotator scores
stored under `malignancy` — the original LIDC field name, kept only for those files.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from field_disagreement import splits as S


# ImageNet statistics for the pretrained ResNet-50 stem.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# Each reader's own call is padded with -1 to this many readers so a batch
# collates (LIDC panels have 3-4 readers, Chakshu 5).
CALLS_PAD = 8


def _resize(t: torch.Tensor, size: int, mode: str) -> torch.Tensor:
    """Resize a (C, H, W) or (H, W) tensor to (size, size). `mode` is
    'bilinear' for continuous fields (image, agreement_frac) or 'nearest' for
    label-like fields (raw agreement counts, binary mask) to keep them crisp."""
    flat = t.ndim == 2
    t = t.unsqueeze(0).unsqueeze(0) if flat else t.unsqueeze(0)
    kw = {"align_corners": False} if mode == "bilinear" else {}
    t = F.interpolate(t, size=(size, size), mode=mode, **kw)
    return t.squeeze(0).squeeze(0) if flat else t.squeeze(0)


# The report field each cohort's diagnosis is read from: LIDC-IDRI / LNDb
# malignancy, Chakshu VCDR.
DIAGNOSTIC_FIELDS = ("malignancy", "VCDR")


class PatchDataset(Dataset):
    def __init__(self, processed_dir, files, train: bool = False,
                 img_size: int = 64, normalize: bool = False,
                 shift_frac: float = 0.0, shift_seed: int = 0):
        self.dir = Path(processed_dir)
        self.files = list(files)
        self.train = train
        self.img_size = int(img_size)                # resize target (native 64 -> 224 for ImageNet nets)
        self.normalize = bool(normalize)             # apply ImageNet mean/std
        self.shift_frac = float(shift_frac)          # decentring: max |offset| as a fraction of the
        self.shift_seed = int(shift_seed)            # native patch width. 0.0 = OFF (historical).

    def __len__(self) -> int:
        return len(self.files)

    # ----- consistent spatial augmentation -------------------------------- #
    @staticmethod
    def _sample_tf() -> tuple[int, bool, bool]:
        """One transform shared by image + all spatial maps."""
        return (int(np.random.randint(0, 4)),
                bool(np.random.rand() < 0.5),
                bool(np.random.rand() < 0.5))

    @staticmethod
    def _apply_tf(arr: np.ndarray, tf, axes: tuple[int, int]) -> np.ndarray:
        """Apply (k*90 rot, flip-axis0, flip-axis1) on the two spatial `axes`."""
        k, f0, f1 = tf
        if k:
            arr = np.rot90(arr, k, axes=axes)
        if f0:
            arr = np.flip(arr, axis=axes[0])
        if f1:
            arr = np.flip(arr, axis=axes[1])
        return np.ascontiguousarray(arr)

    # ----- decentring ------------------------------------------------------ #
    @staticmethod
    def _shift(arr: np.ndarray, dy: int, dx: int, fill: str = "zero") -> np.ndarray:
        """Integer-pixel translation on the LAST TWO axes (so it serves both the
        (3, H, W) image and the (H, W) maps). No interpolation -> the in-mask pixel
        count is preserved exactly whenever the mask does not clip.

        `fill` is what enters the band the translation vacates, and it differs by
        WHAT is being shifted:

          "zero"    - for the MAPS. Outside the translated frame there genuinely is
                      no lesion, so agreement/mask must read 0. Reflecting them would
                      paint a phantom lesion.
          "reflect" - for the IMAGE. Zero-filling it would paint a black band over
                      ~19-20% of every frame at shift_frac 0.20 - a hard synthetic edge
                      whose position encodes which way the lesion moved. Reflected
                      tissue keeps the texture statistics real.

        The clamp in `_sample_shift` keeps the whole mask bbox inside the frame, so
        in-mask pixels are never filled either way.
        """
        h, w = arr.shape[-2], arr.shape[-1]
        if fill == "zero":
            out = np.zeros_like(arr)
            y0, y1 = max(0, dy), min(h, h + dy)
            x0, x1 = max(0, dx), min(w, w + dx)
            out[..., y0:y1, x0:x1] = arr[..., y0 - dy:y1 - dy, x0 - dx:x1 - dx]
            return out
        if fill != "reflect":
            raise ValueError(f"_shift fill must be 'zero' or 'reflect', got {fill!r}")
        # pad by the offset so the vacated band is real (mirrored) tissue, then window
        # the original-size frame back out at the translated position.
        p = min(max(abs(dy), abs(dx)), h - 1, w - 1)
        pad = [(0, 0)] * (arr.ndim - 2) + [(p, p), (p, p)]
        padded = np.pad(arr, pad, mode="reflect")
        return np.ascontiguousarray(
            padded[..., p - dy:p - dy + h, p - dx:p - dx + w])

    def _sample_shift(self, fname: str, mask: np.ndarray) -> tuple[int, int]:
        """Offset (dy, dx), CLAMPED per nodule so the whole mask bbox stays inside the
        crop.
        Random on train; DETERMINISTIC from the file name on val/test/score3 so every
        arm is evaluated on byte-identical geometry."""
        max_px = int(round(self.shift_frac * mask.shape[-1]))
        if max_px <= 0:
            return 0, 0
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            return 0, 0
        lo_y, hi_y = -min(max_px, int(ys.min())), min(max_px, int(mask.shape[-2] - 1 - ys.max()))
        lo_x, hi_x = -min(max_px, int(xs.min())), min(max_px, int(mask.shape[-1] - 1 - xs.max()))
        if self.train:
            # global numpy RNG, exactly like _sample_tf -> governed by set_seed(cfg.seed)
            dy = int(np.random.randint(lo_y, hi_y + 1)) if hi_y >= lo_y else 0
            dx = int(np.random.randint(lo_x, hi_x + 1)) if hi_x >= lo_x else 0
        else:
            # hashlib, NOT hash(): the builtin is salted per process, so it would give a
            # different test geometry on every run and break cross-arm pairing.
            dig = hashlib.md5(f"{fname}|{self.shift_seed}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(dig[:8], "little"))
            dy = int(rng.integers(lo_y, hi_y + 1)) if hi_y >= lo_y else 0
            dx = int(rng.integers(lo_x, hi_x + 1)) if hi_x >= lo_x else 0
        return dy, dx

    # ----- field access with old-format fallbacks ------------------------- #
    @staticmethod
    def _n_readers(d) -> int:
        if "n_readers" in d.files:
            return int(d["n_readers"])
        return int(len(d["malignancy"]))

    @staticmethod
    def _opinion(d) -> np.ndarray:
        if "opinion_dist" in d.files:
            return d["opinion_dist"].astype(np.float32)
        scores = np.asarray(d["malignancy"]).astype(np.int64)
        counts = np.bincount(scores, minlength=6)[1:6].astype(np.float32)
        return counts / max(counts.sum(), 1.0)

    @staticmethod
    def _disagreement(d) -> float:
        if "disagreement" in d.files:
            return float(d["disagreement"])
        return float(np.std(np.asarray(d["malignancy"], dtype=np.float64)))

    @staticmethod
    def _reader_targets(d) -> np.ndarray:
        """Each reader's own diagnostic call as a binary target, padded with -1 to
        CALLS_PAD: LIDC's malignancy rating by the cohort's median rule (> 3 -> 1,
        < 3 -> 0, the neutral 3 -> 0.5), Chákṣu's glaucoma vote as recorded. The
        random-reader arm R_rr draws one per item per step;
        no other arm reads it. A deterministic read, so no RNG stream moves."""
        out = np.full(CALLS_PAD, -1.0, dtype=np.float32)
        if "malignancy" in d.files:
            r = np.asarray(d["malignancy"], dtype=np.float64).ravel()
            calls = np.where(r > 3, 1.0, np.where(r < 3, 0.0, 0.5))
        elif "votes" in d.files:
            calls = np.asarray(d["votes"], dtype=np.float64).ravel()
        else:
            return out
        calls = calls[:CALLS_PAD]
        out[:len(calls)] = calls
        return out

    def __getitem__(self, i: int) -> dict:
        d = np.load(self.dir / self.files[i], allow_pickle=True)
        img = d["image"].astype(np.float32)                  # (3, 64, 64)
        nr = self._n_readers(d)

        amap = d["agreement_map"].astype(np.float32)         # (64, 64) raw count
        if "agreement_frac" in d.files:
            frac = d["agreement_frac"].astype(np.float32)
        else:
            frac = amap / max(nr, 1)
        mask = (d["binary_mask"] > 0).astype(np.float32)

        if self.train:
            tf = self._sample_tf()
            img = self._apply_tf(img, tf, (1, 2))            # channels-first
            amap = self._apply_tf(amap, tf, (0, 1))
            frac = self._apply_tf(frac, tf, (0, 1))
            mask = self._apply_tf(mask, tf, (0, 1))

        # DECENTRING - after rot/flip, BEFORE the resize so the offset is a whole number
        # of native pixels. One offset for the image and every map, so they stay in a
        # single coordinate frame.
        if self.shift_frac > 0.0:
            dy, dx = self._sample_shift(self.files[i], mask)
            if dy or dx:
                n_before = int(mask.sum())
                # image: reflect, so no black band. maps: zero, so no phantom lesion.
                img = self._shift(img, dy, dx, fill="reflect")
                amap = self._shift(amap, dy, dx)
                frac = self._shift(frac, dy, dx)
                mask = self._shift(mask, dy, dx)
                assert int(mask.sum()) == n_before, (
                    f"decentring clipped the mask on {self.files[i]}: {n_before} -> "
                    f"{int(mask.sum())} px at (dy={dy}, dx={dx}). The clamp in _sample_shift "
                    f"is supposed to make this impossible.")

        image = torch.from_numpy(img)                    # (3, 64, 64) in [0, 1]
        amap_t = torch.from_numpy(amap)
        frac_t = torch.from_numpy(frac)
        mask_t = torch.from_numpy(mask)

        # resize image + all spatial maps to the backbone's input grid, so they
        # stay in one coordinate frame.
        if self.img_size != image.shape[-1]:
            image = _resize(image, self.img_size, "bilinear")
            amap_t = _resize(amap_t, self.img_size, "nearest")   # keep integer levels
            frac_t = _resize(frac_t, self.img_size, "bilinear")  # continuous in [0, 1]
            mask_t = _resize(mask_t, self.img_size, "nearest")
        if self.normalize:                               # ImageNet stem expects normalised input
            image = (image - _IMAGENET_MEAN) / _IMAGENET_STD

        # The study's report-level cohorts carry the single-attribute d
        # beside the report-level target. Other cohorts have no such key and
        # their batches are unchanged.
        extra = {}
        if "d_single" in d.files:
            extra["d_single"] = torch.tensor(float(d["d_single"]), dtype=torch.float32)
        if "d_profile" in d.files:
            extra["d_profile"] = torch.from_numpy(np.asarray(d["d_profile"], np.float32))
            # the diagnostic field's own column, unrounded (the study's rank-normal DUP arm)
            if "d_profile_attributes" in d.files:
                names = [str(a) for a in d["d_profile_attributes"]]
                hit = [i for i, a in enumerate(names) if a in DIAGNOSTIC_FIELDS]
                if hit:
                    extra["d_diag"] = torch.tensor(float(d["d_profile"][hit[0]]),
                                                   dtype=torch.float32)
        if "opinion_fields" in d.files:
            extra["opinion_fields"] = torch.from_numpy(
                np.asarray(d["opinion_fields"], np.float32))
        return {
            **extra,
            "image": image,
            "label": torch.tensor(int(d["label"]), dtype=torch.long),
            "agreement_map": amap_t,
            "agreement_frac": frac_t,
            "binary_mask": mask_t,
            "opinion_dist": torch.from_numpy(self._opinion(d)),
            "disagreement": torch.tensor(self._disagreement(d), dtype=torch.float32),
            "n_readers": torch.tensor(nr, dtype=torch.long),
            "reader_targets": torch.from_numpy(self._reader_targets(d)),
            "file": self.files[i],
        }



def balanced_sampler(df, files) -> WeightedRandomSampler:
    """Draw the TRAINING files so each class is equally likely per draw: weight
    1 / (count of its class among `files`). Built from the training split's own
    labels only - validation and test are never resampled and never consulted -
    so it cannot leak. Same number of draws per epoch as the split has items,
    with replacement; the torch RNG (seeded per fold) decides the draw."""
    lab = df.set_index("file").loc[list(files), "label"].to_numpy()
    counts = {c: int((lab == c).sum()) for c in np.unique(lab)}
    w = torch.as_tensor([1.0 / counts[c] for c in lab], dtype=torch.double)
    return WeightedRandomSampler(w, num_samples=len(files), replacement=True)


def make_loaders(processed_dir, df, patient_fold, fold_id: int,
                 batch_size: int = 32, num_workers: int = 0,
                 n_splits: int = 5, img_size: int = 64,
                 normalize: bool = False, shift_frac: float = 0.0,
                 shift_seed: int = 0, balanced: bool = False) -> dict[str, DataLoader]:
    """Patient-disjoint train/val/test (+ score-3) loaders for one CV fold.
    Every loader emits the full target tuple; only the train loader augments.
    `img_size`/`normalize` adapt the patch to the backbone (64 raw for the
    from-scratch nets, 224 + ImageNet-normalised for the ResNet-50 stem).

    `shift_frac` > 0 turns on DECENTRING on EVERY split - train randomly,
    val/test/score3 deterministically. It must be on for train AND test together:
    decentring only the test set would confound the result with a train/test
    distribution shift. 0.0 leaves every split centred."""
    sp = S.make_fold(df, patient_fold, fold_id, n_splits)
    S.assert_no_leakage(df, sp)

    def _dataset(files, train):
        return PatchDataset(processed_dir, files, train=train,
                            img_size=img_size, normalize=normalize,
                            shift_frac=shift_frac, shift_seed=shift_seed)

    def dl(files, train, shuffle):
        if train and balanced:
            return DataLoader(_dataset(files, train), batch_size=batch_size,
                              sampler=balanced_sampler(df, files),
                              num_workers=num_workers, drop_last=False)
        ds = PatchDataset(processed_dir, files, train=train,
                           img_size=img_size, normalize=normalize,
                           shift_frac=shift_frac, shift_seed=shift_seed)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, drop_last=False)

    return {
        "splits": sp,
        "train":  dl(sp["train"],  True,  True),
        "val":    dl(sp["val"],    False, False),
        "test":   dl(sp["test"],   False, False),
        "score3": dl(sp["score3"], False, False),
    }

