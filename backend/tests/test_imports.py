"""The package imports and its public surface is present."""
import field_disagreement as fd


def test_public_api():
    for name in ("build_model", "register_backbone", "adapter_spec",
                 "Prediction", "Decision", "Calibrator", "decide_or_defer", "decide_batch"):
        assert hasattr(fd, name), name


def test_builtin_backbones_registered():
    for name in ("cnn", "resnet50"):
        assert name in fd.BACKBONES, name


def test_subpackages():
    assert callable(fd.metrics.roc_auc)
    assert callable(fd.triage.deferral_ce_loss)
    assert callable(fd.triage.conformal.lac_calibrate)
    assert fd.engine.decide_or_defer is fd.decide_or_defer
