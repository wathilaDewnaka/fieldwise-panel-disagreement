"""Frozen pretrained embeddings as a per-field disagreement baseline (CPU, no
training of any network).

Each test-fold image is embedded once by a frozen ImageNet ResNet-50 (the
initialisation every arm starts from, global-average-pooled, 2048 features) and
by a frozen DINOv2 ViT-B/14 (class token, 768 features). Then exactly the
protocol of rev_handcrafted.py: one ridge regression per field, fitted per fold
on the TRAIN + VALIDATION items against the rank-normal target, penalty by 5-fold
inner CV, scored on the fold's centred test images; H_fld minus the baseline with
the seed-pooled, patient-clustered bootstrap.

    python rev_embed.py            (both cohorts, both backbones)
"""
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2].as_posix()  # repository root
import sys, os, json, pickle, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""                       # CPU only
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pathlib import Path
import torch, torch.nn.functional as Fn
import io, contextlib
HERE = os.path.dirname(os.path.abspath(__file__))
with contextlib.redirect_stdout(io.StringIO()):                # helpers only, not its main loop
    src = open(os.path.join(HERE, "rev_handcrafted.py"), encoding="utf8").read()
    g = {"__file__": os.path.join(HERE, "rev_handcrafted.py"), "__name__": "hc"}
    exec(compile(src[:src.index('for coh in ("LIDC", "Chaksu"):')], "hc", "exec"), g)
R, COH, PROC, ARMS, CLAIM, NB, fields, frho, S, FOLDS = (g[k] for k in
    ("R", "COH", "PROC", "ARMS", "CLAIM", "NB", "fields", "frho", "S", "FOLDS"))
rank_normal, ridge_cv = g["rank_normal"], g["ridge_cv"]
os.chdir(ROOT)
torch.set_num_threads(8)
t0 = time.time()


def backbone(name):
    if name == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2); m.fc = torch.nn.Identity()
    else:
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", source="github", skip_validation=True)
    return m.eval()


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def embed(model, imgs):
    out = []
    for i in range(0, len(imgs), 32):
        x = torch.as_tensor(np.stack(imgs[i:i + 32]), dtype=torch.float32)
        x = Fn.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        out.append(model((x - MEAN) / STD).numpy())
    return np.concatenate(out)


summary = {}
for coh in ("LIDC", "Chaksu"):
    proc = Path(PROC[coh])
    df = S.build_manifest(proc)
    assign = json.loads(Path(FOLDS[coh]).read_text())["assignment"]
    files, imgs, D = [], [], {}
    for f in df["file"]:
        z = np.load(proc / f, allow_pickle=True)
        if int(z["label"]) >= 0:
            files.append(f); imgs.append(np.asarray(z["image"], np.float32))
            D[f] = np.asarray(z["d_profile"], float)
    runs = {s: R.load_run(d, s) for s, d in COH[coh].items()}
    for bb in ("resnet50", "dinov2"):
        cache = Path(HERE) / f"emb_{bb}_{coh}.pkl"
        if cache.exists():
            E = pickle.load(open(cache, "rb"))
        else:
            E = dict(zip(files, embed(backbone(bb), imgs)))
            pickle.dump(E, open(cache, "wb"))
        print(f"{coh} {bb}: {len(E)} images, {next(iter(E.values())).size} features, {time.time() - t0:.0f}s", flush=True)
        rng = np.random.default_rng(0)
        point = {"embed": [], CLAIM: []}
        draws = np.zeros(NB)
        for k in range(5):
            sp = S.make_fold(df, assign, k, 5)
            fit = [f for f in sp["train"] + sp["val"] if f in E]
            te = list(runs[42][k][CLAIM]["test_files"])
            Xf = np.stack([E[f] for f in fit]); Xt = np.stack([E[f] for f in te])
            mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
            Xf = np.c_[(Xf - mu) / sd, np.ones(len(Xf))]; Xt = np.c_[(Xt - mu) / sd, np.ones(len(Xt))]
            Df = np.stack([D[f] for f in fit]); Dt = np.stack([D[f] for f in te])
            assert np.allclose(Dt, np.asarray(runs[42][k][CLAIM]["test_d_profile"], float), equal_nan=True)
            Ptest = np.zeros_like(Dt)
            for j in range(Dt.shape[1]):
                ok = np.isfinite(Df[:, j])
                w, _ = ridge_cv(Xf[ok], rank_normal(Df[ok, j], Df[ok, j]), rng,
                                lams=10.0 ** np.arange(0, 7))
                Ptest[:, j] = Xt @ w
            cl = np.asarray(runs[42][k][CLAIM]["test_clusters"])
            groups = [np.flatnonzero(cl == u) for u in np.unique(cl)]
            boots = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(NB)]
            hb = np.array([frho(Ptest[b], Dt[b]) for b in boots]); hp = frho(Ptest, Dt)
            for s in runs:
                Fc = fields(runs[s][k][CLAIM], "test", Dt.shape[1])
                cb = np.array([frho(Fc[b], Dt[b]) for b in boots])
                point[CLAIM].append(frho(Fc, Dt)); point["embed"].append(hp)
                draws += (cb - hb) / (5 * len(runs))
            print(f"  {coh} {bb} fold {k}: embed {hp:.3f}  {time.time() - t0:.0f}s", flush=True)
        d = np.array(point[CLAIM]) - np.array(point["embed"])
        lo, hi = np.percentile(draws, [2.5, 97.5])
        summary[(coh, bb)] = (np.mean(point["embed"]), np.mean(point[CLAIM]), d.mean(), lo, hi, int((d > 0).sum()))
        print(f"== {coh} {bb}: per-field rho {np.mean(point['embed']):.3f}; H_fld {np.mean(point[CLAIM]):.3f}; "
              f"H_fld - embed {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] {int((d > 0).sum())}/{len(d)}", flush=True)
pickle.dump(summary, open(os.path.join(HERE, "rev_embed.pkl"), "wb"))
print("DONE", time.time() - t0)
