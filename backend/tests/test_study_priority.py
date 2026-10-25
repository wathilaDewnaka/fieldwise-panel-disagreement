"""The field-priority veto (`served_v5`)."""
import numpy as np

from backend.notebooks.shared import study_results as R


def _run(n=160, P=7, seed=0):
    """Five folds where field 6's predicted disagreement flags the positives the
    classifier calls negative, and every other field is noise."""
    rng = np.random.default_rng(seed)
    res = {}
    for k in range(5):
        y = rng.integers(0, 2, n)
        p = np.clip(0.3 * y + rng.uniform(0, 0.6, n), 0, 1)
        F = rng.normal(size=(n, P))
        F[:, 6] += 2.0 * y
        res[k] = {"A_cls": {"test_probs": p.tolist(), "test_labels": y.tolist()},
                  "H_fld": {"test_panel_disp_fields": F.tolist(),
                            "test_panel_disp": F.mean(1).tolist()}}
    return res


def test_the_weights_find_the_field_that_flags_missed_positives():
    w = R.priority_lanes(_run())
    assert int(np.argmax(w)) == 6 and w[6] > 3 * np.abs(np.delete(w, 6)).max()


def test_each_fold_is_scored_out_of_fold_and_never_by_itself():
    res = _run()
    R.priority_lanes(res)
    a = np.asarray(res[0]["H_fld"]["test_priority_lanes"])
    res2 = _run()
    res2[0]["A_cls"]["test_labels"] = (1 - np.asarray(res2[0]["A_cls"]["test_labels"])).tolist()
    R.priority_lanes(res2)
    # fold 0's own labels never enter fold 0's weights, so its signal is unchanged
    assert np.allclose(a, res2[0]["H_fld"]["test_priority_lanes"])


def test_the_signal_is_the_larger_of_the_two_lanes():
    res = _run()
    R.priority_lanes(res)
    lanes = np.asarray(res[1]["H_fld"]["test_priority_lanes"])
    overall = R._within_rank(res[1]["H_fld"]["test_panel_disp"])
    assert (lanes >= overall - 1e-12).all()


def test_no_per_field_outputs_means_no_v5():
    res = _run()
    for k in res:
        res[k]["H_fld"].pop("test_panel_disp_fields")
    assert R.priority_lanes(res) is None
    assert R.study_rule().veto == 0.50
