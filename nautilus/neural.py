"""Module implementing neural network emulators."""

import os
import warnings
from functools import partial

import numpy as np
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits


@threadpool_limits.wrap(limits=1)
def train_network(x, y, neural_network_kwargs, random_state):
    """Train a network.

    Parameters
    ----------
    x : numpy.ndarray
        Input coordinates.
    y : numpy.ndarray
        Target values.
    neural_network_kwargs : dict
        Keyword arguments passed to the constructor of MLPRegressor.
    random_state : int
        Determines random number generation.

    Returns
    -------
    network : MLPRegressor
        The trained network.

    """
    return MLPRegressor(
        random_state=random_state, **neural_network_kwargs).fit(x, y)


class NeuralNetworkEmulator:
    """Likelihood neural network emulator.

    Attributes
    ----------
    mean : numpy.ndarray
        Mean of input coordinates used for normalizing coordinates.
    scale : numpy.ndarray
        Standard deviation of input coordinates used for normalizing
        coordinates.
    network : sklearn.neural_network.MLPRegressor
        Artifical neural network used for emulation.

    """

    # Minimum number of points for which to predict on the GPU, if the
    # emulator was trained there. For fewer points, numpy is faster.
    gpu_predict_min_points = 64

    @classmethod
    def train(cls, x, y, n_networks=4, neural_network_kwargs=None, pool=None,
              backend='sklearn'):
        """Initialize and train the likelihood neural network emulator.

        Parameters
        ----------
        x : numpy.ndarray
            Input coordinates.
        y : numpy.ndarray
            Target values.
        n_networks : int, optional
            Number of networks used in the emulator. Default is 4.
        neural_network_kwargs : dict or None, optional
            Non-default keyword arguments passed to the constructor of
            MLPRegressor. Default is `None`.
        pool : multiprocessing.Pool, optional
            Pool used for parallel processing. Ignored by the 'jax' and
            'cuda' backends.
        backend : str, optional
            Library used for training: 'sklearn', 'jax' or 'cuda'. The 'jax'
            backend trains all networks simultaneously in a single compiled
            call, which is much faster, especially on a GPU. The 'cuda'
            backend trains them in a single fused CUDA kernel, which is
            faster still but requires CuPy and an NVIDIA GPU. Both fall back
            to 'sklearn' for settings they do not support. Default is
            'sklearn'.

        Returns
        -------
        emulator : NeuralNetworkEmulator
            The likelihood neural network emulator.

        """
        emulator = cls()

        emulator.mean = np.mean(x, axis=0)
        emulator.scale = np.std(x, axis=0)

        if neural_network_kwargs is None:
            neural_network_kwargs = {}

        neural_network_kwargs = dict(  # noqa: C408
            hidden_layer_sizes=(100, 50, 20), alpha=0, learning_rate_init=1e-2,
            max_iter=10000, tol=0, n_iter_no_change=10) | neural_network_kwargs

        if 'random_state' in neural_network_kwargs:
            warnings.warn("The 'random_state' keyword argument passed to the" +
                          " neural network is ignored.", Warning, stacklevel=2)
            neural_network_kwargs = neural_network_kwargs.copy()
            del neural_network_kwargs['random_state']

        x = (x - emulator.mean) / emulator.scale

        if backend not in ['sklearn', 'jax', 'cuda']:
            raise ValueError(f"Unknown neural network backend '{backend}'.")

        if backend in ['jax', 'cuda']:
            if backend == 'jax':
                from . import _jax_mlp as module
            else:
                from . import _cuda_mlp as module
            if module.is_supported(neural_network_kwargs, x.shape[1],
                                   n_networks):
                emulator.neural_networks = module.train_networks(
                    x, y, n_networks, neural_network_kwargs)
                # Also predict on the GPU, but only in this process. Copies
                # of the emulator sent to worker processes use numpy.
                emulator.gpu_backend = backend
                emulator.gpu_pid = os.getpid()
                return emulator
            warnings.warn(f"The '{backend}' backend does not support these" +
                          " neural network settings or is not available." +
                          " Using 'sklearn' instead.", Warning, stacklevel=2)

        f = partial(train_network, x, y, neural_network_kwargs)

        if pool is None:
            emulator.neural_networks = list(map(f, range(n_networks)))
        else:
            emulator.neural_networks = list(pool.map(f, range(n_networks)))

        return emulator

    def predict(self, x):
        """Calculate the emulator likelihood prediction for a group of points.

        Parameters
        ----------
        x : numpy.ndarray
            Normalized coordinates of the training points.

        Returns
        -------
        y_emu : numpy.ndarray
            Emulated normalized likelihood value of the training points.

        """
        weights = self._stacked_weights()
        if weights is None:
            return np.mean(
                [network.predict((x - self.mean) / self.scale) for network in
                 self.neural_networks], axis=0)

        # Evaluate all networks at once. The first layer of every network
        # is a single matrix product; later layers are batched products.
        coefs, intercepts = weights
        n_networks = len(self.neural_networks)
        x = np.atleast_2d(x)
        if len(x) == 0:
            return np.zeros(0)
        x = ((x - self.mean) / self.scale).astype(np.float32)
        if (getattr(self, 'gpu_pid', None) == os.getpid() and
                len(x) >= self.gpu_predict_min_points):
            if self.gpu_backend == 'jax':
                from . import _jax_mlp as module
            else:
                from . import _cuda_mlp as module
            return module.predict(self, x)
        a = x @ coefs[0]
        a += intercepts[0]
        np.maximum(a, 0, out=a)
        a = a.reshape(len(x), n_networks, -1).transpose(1, 0, 2)
        for i in range(1, len(coefs)):
            a = np.matmul(a, coefs[i])
            a += intercepts[i]
            if i < len(coefs) - 1:
                np.maximum(a, 0, out=a)
        return np.mean(a[:, :, 0], axis=0, dtype=np.float64)

    def _stacked_weights(self):
        """Return the weights of all networks stacked for fast prediction.

        Returns
        -------
        weights : tuple or None
            Coefficients and intercepts of all layers in single precision.
            The first layer of all networks is concatenated into one matrix,
            subsequent layers are stacked along a new first axis. None if the
            networks cannot be evaluated this way, i.e., if they do not use
            ReLU activations or have differing architectures.

        """
        if not hasattr(self, '_weights'):
            networks = self.neural_networks
            shapes = [[c.shape for c in network.coefs_] for network in
                      networks]
            if (any(network.activation != 'relu' or
                    network.out_activation_ != 'identity' or
                    network.n_outputs_ != 1 for network in networks) or
                    any(s != shapes[0] for s in shapes)):
                self._weights = None
            else:
                coefs = [np.concatenate(
                    [network.coefs_[0] for network in networks], axis=1)]
                intercepts = [np.concatenate(
                    [network.intercepts_[0] for network in networks])]
                for i in range(1, len(shapes[0])):
                    coefs.append(np.stack(
                        [network.coefs_[i] for network in networks]))
                    intercepts.append(np.stack(
                        [network.intercepts_[i] for network in networks])[
                            :, np.newaxis, :])
                self._weights = (
                    [c.astype(np.float32) for c in coefs],
                    [i.astype(np.float32) for i in intercepts])
        return self._weights

    def write(self, group):
        """Write the emulator to an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.

        """
        group.attrs['n_networks'] = len(self.neural_networks)

        for i, network in enumerate(self.neural_networks):

            for key in network.__dict__:
                if key in ['coefs_', 'intercepts_']:
                    continue
                try:
                    group.attrs[key + f'_{i}'] = getattr(network, key)
                except (TypeError, ValueError):
                    pass

            for k in range(network.n_layers_ - 1):
                group.create_dataset(f'coefs_{k}_{i}', data=network.coefs_[k])
                group.create_dataset(
                    f'intercepts_{k}_{i}', data=network.intercepts_[k])

        group.create_dataset('mean', data=self.mean)
        group.create_dataset('scale', data=self.scale)

    @classmethod
    def read(cls, group):
        """Read the emulator from an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.

        Returns
        -------
        emulator : NeuralNetworkEmulator
            The likelihood neural network emulator.

        """
        emulator = cls()

        emulator.mean = np.array(group['mean'])
        emulator.scale = np.array(group['scale'])

        emulator.neural_networks = []

        for i in range(group.attrs['n_networks']):

            network = MLPRegressor()

            for key in group.attrs:
                if key.rsplit('_', 1)[1] == f'{i}':
                    setattr(network, key.rsplit('_', 1)[0], group.attrs[key])

            network.coefs_ = [
                np.array(group[f'coefs_{k}_{i}']) for k in
                range(network.n_layers_ - 1)]
            network.intercepts_ = [
                np.array(group[f'intercepts_{k}_{i}']) for k in
                range(network.n_layers_ - 1)]

            emulator.neural_networks.append(network)

        return emulator
