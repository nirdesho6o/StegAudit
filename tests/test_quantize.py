"""quantize.py: integer intervals at beta bits.

Outside judge: at beta = 8 every x in [0, 256) is enumerated and looked up by
a plain linear scan; the counts per token must equal the widths, and find()
must agree with the scan for every x.
"""

import numpy as np
import pytest

from stegaudit.quantize import quantize, tv_distance

rng = np.random.default_rng(0)


def random_dists():
    """Mix of flat, peaked and very skewed (tiny-probability) distributions."""
    out = []
    for k in (1, 2, 5, 50):
        for alpha in (5.0, 1.0, 0.05, 0.005):
            for _ in range(5):
                out.append(rng.dirichlet(np.full(k, alpha)))
    return out


def linear_find(qd, x):
    for i in range(len(qd.widths)):
        lo, hi = qd.interval(i)
        if lo <= x < hi:
            return i
    raise AssertionError("x not covered")


@pytest.mark.parametrize("beta", [8, 12, 16, 32, 62])
def test_intervals_tile_the_range_exactly(beta):
    """CLAUDE.md required: disjoint, cover [0, 2**beta), sum == 2**beta."""
    for p in random_dists():
        qd = quantize(p, beta)
        w = [int(v) for v in qd.widths]              # Python ints: exact sums
        assert sum(w) == 2**beta
        assert all(v >= 0 for v in w)
        assert int(qd.lo[0]) == 0
        for i in range(len(w) - 1):                   # contiguous => disjoint, no gaps
            assert int(qd.lo[i + 1]) == int(qd.lo[i]) + w[i]
        assert int(qd.lo[-1]) + w[-1] == 2**beta


@pytest.mark.parametrize("beta", [8, 16, 32])
def test_widths_track_probabilities(beta):
    for p in random_dists():
        qd = quantize(p, beta)
        top = int(np.argmax(p))
        err = np.abs(qd.widths / 2.0**beta - p)
        assert np.all(np.delete(err, top) < 2.0**-beta)
        assert tv_distance(p, qd) < len(p) * 2.0**-beta
        assert 0 <= qd.remainder < len(p)


def test_exhaustive_lookup_at_beta_8():
    """Every x hits exactly one non-empty interval; hits per token == width.
    So a uniform x picks token i with probability exactly width_i / 2**beta."""
    for p in random_dists():
        qd = quantize(p, 8)
        hits = np.zeros(len(p), dtype=int)
        for x in range(256):
            i = qd.find(x)
            assert i == linear_find(qd, x)
            assert qd.widths[i] > 0
            hits[i] += 1
        assert np.array_equal(hits, qd.widths)


@pytest.mark.parametrize("beta", [16, 32])
def test_find_at_interval_edges(beta):
    for p in random_dists():
        qd = quantize(p, beta)
        for i in np.flatnonzero(qd.widths):
            lo, hi = qd.interval(int(i))
            assert qd.find(lo) == i and qd.find(hi - 1) == i
        assert qd.find(0) >= 0 and qd.find(2**beta - 1) < len(p)


def test_vanishing_counted():
    p = np.array([1 - 3e-10, 1e-10, 1e-10, 1e-10])   # 1e-10 * 2**32 = 0.43 -> width 0
    qd = quantize(p, 32)
    assert qd.n_vanished == 3
    assert list(qd.widths[1:]) == [0, 0, 0]
    # an empty interval can never be selected
    assert all(qd.find(x) == 0 for x in (0, 2**31, 2**32 - 1))


def test_vanishing_boundary_is_p_below_2_to_minus_beta():
    beta = 32
    at = quantize([1 - 2.0**-beta, 2.0**-beta], beta)            # exactly 2**-beta -> width 1
    below = quantize([1 - 2.0**-(beta + 1), 2.0**-(beta + 1)], beta)  # half that -> width 0
    assert at.n_vanished == 0 and int(at.widths[1]) == 1
    assert below.n_vanished == 1 and int(below.widths[1]) == 0


def test_vanishing_more_common_at_small_beta():
    dists = random_dists()
    counts = {b: sum(quantize(p, b).n_vanished for p in dists) for b in (8, 12, 16, 32)}
    assert counts[8] >= counts[12] >= counts[16] >= counts[32]
    assert counts[8] > counts[32]


def test_zero_probability_token_is_not_called_vanished():
    qd = quantize([0.5, 0.0, 0.5], 16)
    assert qd.n_vanished == 0 and int(qd.widths[1]) == 0


def test_single_token_gets_everything():
    qd = quantize([1.0], 32)
    assert int(qd.widths[0]) == 2**32 and qd.find(2**32 - 1) == 0


def test_argmax_tie_goes_to_lowest_position():
    qd = quantize([1 / 3, 1 / 3, 1 / 3], 8)          # floors 85,85,85 -> remainder 1
    assert [int(w) for w in qd.widths] == [86, 85, 85]


@pytest.mark.parametrize("bad", [[], [[0.5, 0.5]], [0.5, np.nan], [0.5, -0.1], [0.0, 0.0], [np.inf, 1]])
def test_rejects_bad_probs(bad):
    with pytest.raises(ValueError):
        quantize(bad, 16)


@pytest.mark.parametrize("beta", [0, 63])
def test_rejects_bad_beta(beta):
    with pytest.raises(ValueError):
        quantize([0.5, 0.5], beta)


def test_find_rejects_out_of_range():
    qd = quantize([0.5, 0.5], 8)
    for x in (-1, 256):
        with pytest.raises(ValueError):
            qd.find(x)
