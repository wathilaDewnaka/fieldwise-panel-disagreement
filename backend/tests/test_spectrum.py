"""The primitives every domain builds its agreement spectrum with.

What is pinned here is what the five per-domain copies had already started to
disagree about: the normalisation that makes a 3-annotator item comparable with a
4-annotator one, a mask threshold that is a real parameter, a histogram over
DECLARED levels, and an ambiguous class code that survives.
"""
import numpy as np
import pytest

from field_disagreement import spectrum


def _box(h=8, w=8, size=4):
    m = np.zeros((h, w), dtype=np.uint8)
    m[:size, :size] = 1
    return m


def test_frac_normalises_by_the_panel_that_read_this_item():
    """A raw count of 2 is unanimous on a 2-annotator item and half a panel on a
    4-annotator one, so raw counts must never be compared across items."""
    assert spectrum.agreement_surface([_box()] * 2)[1].max() == pytest.approx(1.0)
    assert spectrum.agreement_surface([_box()] * 4)[1].max() == pytest.approx(1.0)

    count, frac, binary = spectrum.agreement_surface(
        [_box(size=4), _box(size=6), _box(size=8)])
    assert count.max() == 3                       # the nested core
    assert binary.sum() == 64                     # the widest mask is the union
    assert np.allclose(frac, count / 3)


def test_the_mask_threshold_is_a_real_parameter():
    """Masks arrive boolean, as 0/255, and as float or uint16 from medical-image
    readers, where `> 0` and `> 0.5` are not the same question."""
    faint = np.full((4, 4), 1e-6)
    assert spectrum.agreement_surface([faint], threshold=0.5)[0].sum() == 0
    assert spectrum.agreement_surface([faint], threshold=0.0)[0].sum() == 16


def test_agreement_surface_rejects_an_empty_or_ragged_panel():
    with pytest.raises(ValueError, match="at least one reader"):
        spectrum.agreement_surface([])
    with pytest.raises(ValueError, match="disagree on shape"):
        spectrum.agreement_surface([_box(), _box(h=16, w=16)])


def test_mean_pairwise_dice_scores_two_empty_masks_as_agreement():
    """They agree that there is nothing there; 0 would report the opposite."""
    empty = np.zeros((4, 4), dtype=np.uint8)
    assert spectrum.mean_pairwise_dice([empty, empty]) == 1.0
    assert spectrum.mean_pairwise_dice([_box(size=8), _box(size=4)]) < 1.0


def test_opinion_histogram_keeps_a_bin_for_a_level_nobody_picked():
    """K is a property of the annotation protocol, not of this item's panel."""
    q = spectrum.opinion_histogram([3, 3, 4], levels=(1, 2, 3, 4, 5))
    assert q.tolist() == pytest.approx([0.0, 0.0, 2 / 3, 1 / 3, 0.0])


def test_an_off_scale_value_raises_instead_of_vanishing():
    """Dropping it turns a panel of 4 into a panel of 3 with no trace, while
    `disagreement` and the label still count all four."""
    with pytest.raises(ValueError, match="not one of the declared levels"):
        spectrum.opinion_histogram([1, 2, 9], levels=(1, 2, 3, 4, 5))


def test_rating_entropy_is_exactly_zero_for_a_unanimous_panel():
    """Zero bins are skipped, not clipped to an epsilon whose size would leak in."""
    levels = (1, 2, 3, 4, 5)
    assert spectrum.rating_entropy(spectrum.opinion_histogram([4] * 4, levels)) == 0.0
    assert spectrum.rating_entropy(
        spectrum.opinion_histogram([1, 2, 4, 5], levels)) == pytest.approx(np.log(4))


def test_median_label_keeps_the_undecided_case_as_its_own_code():
    """Collapsing a median that lands on the ambiguous level into either class
    invents a decision the panel declined to make."""
    assert spectrum.median_label([4, 4, 5], ambiguous=3) == 1
    assert spectrum.median_label([1, 2, 2], ambiguous=3) == 0
    assert spectrum.median_label([3, 3, 3], ambiguous=3) == -1
    assert spectrum.median_label([1, 2, 4, 5], ambiguous=3) == -1   # median 3.0


def test_every_domain_reduces_its_panel_through_the_framework():
    """The point of the move: a domain that reintroduces a private copy can drift
    from every other cohort without anything failing."""
    from backend.domains.chaksu import masks as chaksu

    for mod in (chaksu,):
        assert mod.agreement_surface is spectrum.agreement_surface, mod.__name__
