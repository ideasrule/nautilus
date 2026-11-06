import numpy as np
import pytest

from nautilus import neural


def test_neural_network_emulator():
    # Test that the emulator works and produces decent estimates.

    np.random.seed(0)
    n_dim, n_points = 5, 1000
    x = np.random.random((n_points, n_dim))
    y = np.linalg.norm(x - 0.5, axis=1)
    y = np.argsort(np.argsort(y)) / float(len(y))
    emu = neural.NeuralNetworkEmulator.train(x, y, n_networks=1, pool=None)
    assert np.sqrt(np.mean((y - emu.predict(x))**2)) < 0.3 * np.std(y)


def test_neural_network_emulator_jax():
    # Test that the JAX backend produces estimates as good as scikit-learn.

    pytest.importorskip('jax')
    np.random.seed(0)
    n_dim, n_points = 5, 1000
    x = np.random.random((n_points, n_dim))
    y = np.linalg.norm(x - 0.5, axis=1)
    y = np.argsort(np.argsort(y)) / float(len(y))
    emu = neural.NeuralNetworkEmulator.train(
        x, y, n_networks=4, pool=None, backend='jax')
    assert len(emu.neural_networks) == 4
    assert np.sqrt(np.mean((y - emu.predict(x))**2)) < 0.3 * np.std(y)


def test_neural_network_emulator_jax_fallback():
    # Test that unsupported settings fall back to scikit-learn.

    pytest.importorskip('jax')
    x = np.random.random((100, 2))
    y = np.random.random(100)
    with pytest.warns(Warning, match='does not support'):
        neural.NeuralNetworkEmulator.train(
            x, y, n_networks=1, backend='jax',
            neural_network_kwargs={'activation': 'tanh', 'max_iter': 5})
    with pytest.raises(ValueError):
        neural.NeuralNetworkEmulator.train(x, y, backend='torch')
