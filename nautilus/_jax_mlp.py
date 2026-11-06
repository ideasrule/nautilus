"""Train many neural networks at once with JAX.

This reproduces what `sklearn.neural_network.MLPRegressor.fit` does for the
settings Nautilus uses (ReLU activations, squared loss, Adam, shuffled
minibatches and stopping once the training loss stops improving), but trains
all networks of an emulator simultaneously and runs the entire training loop
inside a single compiled XLA call. On a GPU this is typically two orders of
magnitude faster than training the networks one by one with scikit-learn.
The trained weights are returned as ordinary `MLPRegressor` objects so that
prediction and I/O work exactly as before.
"""

import weakref
from functools import lru_cache

import numpy as np
from sklearn.neural_network import MLPRegressor

# Weights of emulators copied to the JAX device, kept separately so that
# emulators can still be pickled and sent to worker processes.
_DEVICE_WEIGHTS = weakref.WeakKeyDictionary()

# Keyword arguments of MLPRegressor this backend implements, with defaults.
SUPPORTED_KWARGS = dict(  # noqa: C408
    hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=1e-4,
    batch_size='auto', learning_rate='constant', learning_rate_init=1e-3,
    max_iter=200, shuffle=True, tol=1e-4, early_stopping=False,
    beta_1=0.9, beta_2=0.999, epsilon=1e-8, n_iter_no_change=10)


def is_supported(neural_network_kwargs, n_dim=None, n_networks=None):
    """Check whether the JAX backend can emulate the requested network.

    Parameters
    ----------
    neural_network_kwargs : dict
        Keyword arguments for the constructor of MLPRegressor.
    n_dim : int or None, optional
        Number of input dimensions. Not used by this backend.
    n_networks : int or None, optional
        Number of networks. Not used by this backend.

    Returns
    -------
    supported : bool
        Whether the requested settings are supported.

    """
    kwargs = SUPPORTED_KWARGS | neural_network_kwargs
    return (set(neural_network_kwargs) <= set(SUPPORTED_KWARGS) and
            kwargs['activation'] == 'relu' and kwargs['solver'] == 'adam' and
            kwargs['learning_rate'] == 'constant' and kwargs['shuffle'] and
            not kwargs['early_stopping'])


@lru_cache
def _make_trainer(layer_sizes, batch_size):
    """Compile a function training a stack of networks."""
    import jax
    import jax.numpy as jnp

    # Training does not need full single precision. Ask for the fastest
    # precision explicitly, i.e., TF32 on recent NVIDIA GPUs, in case
    # 'jax_default_matmul_precision' has been raised globally. This makes
    # training about twice as fast in that case.
    fast = jax.lax.Precision.DEFAULT

    def forward(params, x):
        for w, b in params[:-1]:
            x = jax.nn.relu(jnp.matmul(x, w, precision=fast) + b)
        w, b = params[-1]
        return (jnp.matmul(x, w, precision=fast) + b)[:, 0]

    def batch_loss(params, x, y, weight, alpha):
        n = jnp.sum(weight)
        r = forward(params, x) - y
        penalty = sum(jnp.sum(w * w) for w, _ in params)
        return 0.5 * (jnp.sum(weight * r * r) + alpha * penalty) / n

    loss_and_grad = jax.value_and_grad(batch_loss)

    def train_one(key, x, y, n, hyper):
        n_pad, n_dim = x.shape
        sizes = (n_dim, ) + layer_sizes + (1, )

        # Glorot-uniform initialization, as in scikit-learn.
        params = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            key, key_w, key_b = jax.random.split(key, 3)
            bound = np.sqrt(6.0 / (fan_in + fan_out))
            params.append((
                jax.random.uniform(key_w, (fan_in, fan_out), jnp.float32,
                                   -bound, bound),
                jax.random.uniform(key_b, (fan_out, ), jnp.float32,
                                   -bound, bound)))
        zeros = jax.tree.map(jnp.zeros_like, params)

        n_batches = (n + batch_size - 1) // batch_size
        offsets = jnp.arange(batch_size)

        def minibatch(i, carry):
            params, m, v, t, perm, loss_sum = carry
            select = jax.lax.dynamic_slice(perm, (i * batch_size, ),
                                           (batch_size, ))
            # The last batch may be partial; mask the padding.
            weight = (i * batch_size + offsets < n).astype(jnp.float32)
            loss, grad = loss_and_grad(
                params, x[select], y[select], weight, hyper['alpha'])
            t = t + 1
            b1, b2 = hyper['beta_1'], hyper['beta_2']
            m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, m, grad)
            v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, v, grad)
            lr = (hyper['learning_rate_init'] * jnp.sqrt(1 - b2**t) /
                  (1 - b1**t))
            params = jax.tree.map(
                lambda p, m, v: p - lr * m / (jnp.sqrt(v) + hyper['epsilon']),
                params, m, v)
            return params, m, v, t, perm, loss_sum + loss * jnp.sum(weight)

        def epoch(state):
            key, perm_key = jax.random.split(state['key'])
            # Random permutation of the n real points, padding sorted last.
            r = jax.random.uniform(perm_key, (n_pad, ))
            perm = jnp.argsort(jnp.where(jnp.arange(n_pad) < n, r, 2.0))
            params, m, v, t, _, loss_sum = jax.lax.fori_loop(
                0, n_batches, minibatch,
                (state['params'], state['m'], state['v'], state['t'], perm,
                 jnp.float32(0)))
            loss = loss_sum / n
            # Same stopping criterion as scikit-learn without early stopping.
            no_improvement = jnp.where(
                loss > state['best_loss'] - hyper['tol'],
                state['no_improvement'] + 1, 0)
            return dict(  # noqa: C408
                key=key, params=params, m=m, v=v, t=t, loss=loss,
                best_loss=jnp.minimum(state['best_loss'], loss),
                no_improvement=no_improvement, n_iter=state['n_iter'] + 1)

        def keep_going(state):
            return ((state['no_improvement'] <= hyper['n_iter_no_change']) &
                    (state['n_iter'] < hyper['max_iter']))

        state = dict(  # noqa: C408
            key=key, params=params, m=zeros, v=zeros, t=jnp.float32(0),
            loss=jnp.float32(jnp.inf), best_loss=jnp.float32(jnp.inf),
            no_improvement=jnp.int32(0), n_iter=jnp.int32(0))
        # Under vmap, networks that have stopped keep their state unchanged
        # while the others continue training.
        state = jax.lax.while_loop(keep_going, epoch, state)
        return state['params'], state['n_iter'], state['loss'], \
            state['best_loss']

    return jax.jit(jax.vmap(train_one, in_axes=(0, None, None, None, None)))


def train_networks(x, y, n_networks, neural_network_kwargs, seed=0):
    """Train several networks on the same data simultaneously.

    Parameters
    ----------
    x : numpy.ndarray
        Normalized input coordinates.
    y : numpy.ndarray
        Target values.
    n_networks : int
        Number of networks to train.
    neural_network_kwargs : dict
        Keyword arguments for the constructor of MLPRegressor. Must be
        supported according to `is_supported`.
    seed : int, optional
        Seed for weight initialization and shuffling. Default is 0.

    Returns
    -------
    networks : list of sklearn.neural_network.MLPRegressor
        The trained networks.

    """
    import jax

    kwargs = SUPPORTED_KWARGS | neural_network_kwargs
    n, n_dim = x.shape
    if kwargs['batch_size'] == 'auto':
        batch_size = min(200, n)
    else:
        batch_size = int(np.clip(kwargs['batch_size'], 1, n))
    layer_sizes = tuple(np.atleast_1d(kwargs['hidden_layer_sizes']).tolist())

    # Pad to a power of two so that only a few array shapes get compiled.
    n_pad = 1 << int(np.ceil(np.log2(-(-n // batch_size) * batch_size)))
    x_pad = np.zeros((n_pad, n_dim), dtype=np.float32)
    x_pad[:n] = x
    y_pad = np.zeros(n_pad, dtype=np.float32)
    y_pad[:n] = y

    hyper = {key: np.float32(kwargs[key]) for key in [
        'alpha', 'learning_rate_init', 'tol', 'beta_1', 'beta_2', 'epsilon']}
    hyper['max_iter'] = np.int32(kwargs['max_iter'])
    hyper['n_iter_no_change'] = np.int32(kwargs['n_iter_no_change'])

    keys = jax.random.split(jax.random.PRNGKey(seed), n_networks)
    params, n_iter, loss, best_loss = jax.device_get(
        _make_trainer(layer_sizes, batch_size)(
            keys, x_pad, y_pad, np.int32(n), hyper))

    return networks_from_params(params, n_iter, loss, best_loss, n,
                                layer_sizes, neural_network_kwargs, seed)


def networks_from_params(params, n_iter, loss, best_loss, n, layer_sizes,
                         neural_network_kwargs, seed):
    """Create fitted `MLPRegressor` objects from trained weights.

    Parameters
    ----------
    params : list
        Coefficients and intercepts of each layer, stacked for all networks.
    n_iter : numpy.ndarray
        Number of epochs each network was trained for.
    loss : numpy.ndarray
        Final training loss of each network.
    best_loss : numpy.ndarray
        Best training loss of each network.
    n : int
        Number of training points.
    layer_sizes : tuple
        Sizes of the hidden layers.
    neural_network_kwargs : dict
        Keyword arguments for the constructor of MLPRegressor.
    seed : int
        Seed used for training.

    Returns
    -------
    networks : list of sklearn.neural_network.MLPRegressor
        The trained networks.

    """
    networks = []
    for i in range(len(n_iter)):
        network = MLPRegressor(random_state=seed + i, **neural_network_kwargs)
        network.coefs_ = [np.asarray(w[i], dtype=float) for w, _ in params]
        network.intercepts_ = [np.asarray(b[i], dtype=float)
                               for _, b in params]
        network.n_features_in_ = network.coefs_[0].shape[0]
        network.n_outputs_ = 1
        network.n_layers_ = len(layer_sizes) + 2
        network.out_activation_ = 'identity'
        network.n_iter_ = int(n_iter[i])
        network.t_ = int(n_iter[i]) * n
        network.loss_ = float(loss[i])
        network.best_loss_ = float(best_loss[i])
        networks.append(network)

    return networks


@lru_cache
def _make_predict():
    """Compile a function evaluating stacked networks."""
    import jax
    import jax.numpy as jnp

    # Use full single precision, not TF32, to match predictions with numpy.
    highest = jax.lax.Precision.HIGHEST

    def predict(coefs, intercepts, x):
        n_networks = coefs[1].shape[0]
        a = jax.nn.relu(jnp.matmul(x, coefs[0], precision=highest) +
                        intercepts[0])
        a = a.reshape(x.shape[0], n_networks, -1).transpose(1, 0, 2)
        for i in range(1, len(coefs)):
            a = jnp.matmul(a, coefs[i], precision=highest) + intercepts[i]
            if i < len(coefs) - 1:
                a = jax.nn.relu(a)
        return jnp.mean(a[:, :, 0], axis=0)

    return jax.jit(predict)


def predict(emulator, x):
    """Evaluate an emulator with JAX.

    Parameters
    ----------
    emulator : nautilus.neural.NeuralNetworkEmulator
        The emulator. Its networks must support fast stacked evaluation.
    x : numpy.ndarray
        Normalized input coordinates in single precision.

    Returns
    -------
    y_emu : numpy.ndarray
        Emulated normalized likelihood value of the points.

    """
    import jax

    if emulator not in _DEVICE_WEIGHTS:
        _DEVICE_WEIGHTS[emulator] = jax.device_put(
            emulator._stacked_weights())
    coefs, intercepts = _DEVICE_WEIGHTS[emulator]
    # Pad to a power of four so that only a few array shapes get compiled.
    n = len(x)
    n_pad = max(64, 4**int(np.ceil(np.log(n) / np.log(4))))
    x_pad = np.zeros((n_pad, x.shape[1]), dtype=np.float32)
    x_pad[:n] = x
    return np.asarray(_make_predict()(coefs, intercepts, x_pad))[:n].astype(
        float)
