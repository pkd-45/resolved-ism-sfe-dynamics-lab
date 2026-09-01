import numpy as np

from resolved_ism.dynamics import orbital_time_myr, specific_angular_momentum, v_over_sigma


def test_v_over_sigma():
    out = v_over_sigma(np.array([200.0]), np.array([20.0]))
    assert np.isclose(out[0], 10.0)


def test_orbital_time_is_positive():
    out = orbital_time_myr(np.array([8.0]), np.array([220.0]))
    assert 150.0 < out[0] < 300.0


def test_exponential_flat_curve_j():
    rd = 3.0
    velocity = 200.0
    radius = np.linspace(0.001, 60.0, 20000)
    sigma = np.exp(-radius / rd)
    measured = specific_angular_momentum(
        radius,
        sigma,
        np.full_like(radius, velocity),
    )
    assert np.isclose(measured, 2 * rd * velocity, rtol=2e-3)
