import numpy as np
import pytest

from sonic.editing import global_remove, masked_remove


@pytest.fixture
def signals():
    rng = np.random.default_rng(0)
    b = rng.standard_normal(1000).astype(np.float32) * 0.1
    s = np.zeros(1000, dtype=np.float32)
    s[400:600] = rng.standard_normal(200) * 0.5
    return b + s, b, s


def test_zero_mask_returns_input_bit_identical(signals):
    x, _, s = signals
    noisy_estimate = s + 0.01
    y = masked_remove(x, noisy_estimate, np.zeros_like(x))
    assert np.array_equal(y, x)


def test_samples_outside_mask_untouched(signals):
    x, _, s = signals
    mask = np.zeros_like(x)
    mask[350:650] = 1.0
    y = masked_remove(x, s + 0.01, mask)
    assert np.array_equal(y[:350], x[:350])
    assert np.array_equal(y[650:], x[650:])


def test_unit_mask_equals_global(signals):
    x, _, s = signals
    est = s * 0.9 + 0.001
    assert np.allclose(masked_remove(x, est, np.ones_like(x)), global_remove(x, est))


def test_perfect_estimate_recovers_background(signals):
    x, b, s = signals
    assert np.allclose(global_remove(x, s), b, atol=1e-6)


def test_shape_mismatch_raises(signals):
    x, _, s = signals
    with pytest.raises(ValueError):
        global_remove(x, s[:-1])
    with pytest.raises(ValueError):
        masked_remove(x, s, np.ones(10))
