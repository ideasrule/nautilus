"""Train many neural networks at once with a fused CUDA kernel.

This trains the same networks as `_jax_mlp`, i.e., it reproduces
`sklearn.neural_network.MLPRegressor.fit` for the settings Nautilus uses, but
runs the entire training of all networks of an emulator in a single
persistent CUDA kernel. Each network is distributed over several thread
blocks that each process part of every minibatch; blocks of the same network
combine their gradients through global memory, separated by grid-wide
synchronizations. This avoids launching dozens of tiny kernels per training
step, which otherwise dominates the run time. Requires CuPy and an NVIDIA
GPU.
"""

from functools import lru_cache

import numpy as np

import weakref

from ._jax_mlp import SUPPORTED_KWARGS, networks_from_params
from ._jax_mlp import is_supported as _settings_supported

# Weights of emulators copied to the GPU, kept separately so that emulators
# can still be pickled and sent to worker processes.
_DEVICE_WEIGHTS = weakref.WeakKeyDictionary()

# Device functions for one layer. Each thread computes several outputs so
# that values loaded from shared memory are reused. Weight matrices are
# stored in shared memory with an odd row stride, which avoids bank
# conflicts both when reading rows and columns.
_DEVICE_FUNCTIONS = r"""
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define BATCH {batch}
#define CHUNK {chunk}
#define CHUNK_PAD {chunk_pad}
#define N_BLOCKS {n_blocks}
#define N_THREADS {n_threads}
#define N_PARAMS {n_params}
#define SLICE {slice}
#define R {rows}
#define K {cols}

// Width of vector loads along a dimension of the given size.
template <int S> constexpr int vec_width() {{
    return S % 4 == 0 ? 4 : (S % 2 == 0 ? 2 : 1);
}}

// Load W consecutive floats from 16-, 8- or 4-byte aligned shared memory.
template <int W>
__device__ __forceinline__ void load(const float* p, float* out) {{
    if constexpr (W == 4) {{
        const float4 v = *reinterpret_cast<const float4*>(p);
        out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
    }} else if constexpr (W == 2) {{
        const float2 v = *reinterpret_cast<const float2*>(p);
        out[0] = v.x; out[1] = v.y;
    }} else {{
        out[0] = p[0];
    }}
}}

// a_out = relu(a_in w + b), or the residual a_in w + b - y for the output.
template <int S_IN, int S_OUT, int STRIDE, bool OUTPUT>
__device__ __forceinline__ void forward_layer(
        const float* __restrict__ a_in, const float* __restrict__ w,
        const float* __restrict__ b, float* __restrict__ a_out,
        const float* __restrict__ y, int cnt) {{
    constexpr int V = vec_width<S_IN>();
    for (int item = threadIdx.x; item < CHUNK_PAD / R * S_OUT;
            item += N_THREADS) {{
        const int j = item % S_OUT;
        const int r0 = item / S_OUT * R;
        float acc[R];
        #pragma unroll
        for (int i = 0; i < R; i++) acc[i] = b[j];
        #pragma unroll
        for (int k = 0; k < S_IN; k += V) {{
            float w_k[V];
            #pragma unroll
            for (int v = 0; v < V; v++) w_k[v] = w[(k + v) * STRIDE + j];
            #pragma unroll
            for (int i = 0; i < R; i++) {{
                float a[V];
                load<V>(a_in + (r0 + i) * S_IN + k, a);
                #pragma unroll
                for (int v = 0; v < V; v++)
                    acc[i] = fmaf(a[v], w_k[v], acc[i]);
            }}
        }}
        #pragma unroll
        for (int i = 0; i < R; i++) {{
            const int r = r0 + i;
            a_out[r * S_OUT + j] = OUTPUT ? (r < cnt ? acc[i] - y[r] : 0.0f) :
                                   fmaxf(acc[i], 0.0f);
        }}
    }}
}}

// Gradient sums of weights (a_in^T d) and biases (sum of d) over the rows.
template <int S_IN, int S_OUT>
__device__ __forceinline__ void gradient_layer(
        const float* __restrict__ a_in, const float* __restrict__ d,
        float* __restrict__ g_w, float* __restrict__ g_b) {{
    constexpr int V = vec_width<S_IN>() < K ? vec_width<S_IN>() : K;
    for (int item = threadIdx.x; item < (S_IN + K - 1) / K * S_OUT;
            item += N_THREADS) {{
        const int j = item % S_OUT;
        const int k0 = item / S_OUT * K;
        float acc[K];
        float acc_b = 0.0f;
        #pragma unroll
        for (int i = 0; i < K; i++) acc[i] = 0.0f;
        #pragma unroll 5
        for (int r = 0; r < CHUNK_PAD; r++) {{
            const float d_rj = d[r * S_OUT + j];
            acc_b += d_rj;
            #pragma unroll
            for (int i = 0; i < K; i += V) {{
                if (S_IN % K == 0 || k0 + i < S_IN) {{
                    float a[V];
                    load<V>(a_in + r * S_IN + k0 + i, a);
                    #pragma unroll
                    for (int v = 0; v < V; v++)
                        acc[i + v] = fmaf(a[v], d_rj, acc[i + v]);
                }}
            }}
        }}
        #pragma unroll
        for (int i = 0; i < K; i++)
            if (S_IN % K == 0 || k0 + i < S_IN)
                g_w[(k0 + i) * S_OUT + j] = acc[i];
        if (k0 == 0) g_b[j] = acc_b;
    }}
}}

// Back-propagate deltas: d_out = (d w^T) * (a_in > 0).
template <int S_IN, int S_OUT, int STRIDE>
__device__ __forceinline__ void delta_layer(
        const float* __restrict__ d, const float* __restrict__ w,
        const float* __restrict__ a_in, float* __restrict__ d_out) {{
    constexpr int V = vec_width<S_OUT>();
    for (int item = threadIdx.x; item < CHUNK_PAD / R * S_IN;
            item += N_THREADS) {{
        const int k = item % S_IN;
        const int r0 = item / S_IN * R;
        float acc[R];
        #pragma unroll
        for (int i = 0; i < R; i++) acc[i] = 0.0f;
        #pragma unroll
        for (int j = 0; j < S_OUT; j += V) {{
            float w_j[V];
            #pragma unroll
            for (int v = 0; v < V; v++) w_j[v] = w[k * STRIDE + j + v];
            #pragma unroll
            for (int i = 0; i < R; i++) {{
                float d_v[V];
                load<V>(d + (r0 + i) * S_OUT + j, d_v);
                #pragma unroll
                for (int v = 0; v < V; v++)
                    acc[i] = fmaf(d_v[v], w_j[v], acc[i]);
            }}
        }}
        #pragma unroll
        for (int i = 0; i < R; i++) {{
            const int r = r0 + i;
            d_out[r * S_IN + k] = a_in[r * S_IN + k] > 0.0f ? acc[i] : 0.0f;
        }}
    }}
}}

__device__ __forceinline__ bool is_weight(int i) {{
    return {is_weight};
}}
"""

_KERNEL = r"""
extern "C" __global__ void __launch_bounds__(N_THREADS, 1) train(
        const float* __restrict__ x, const float* __restrict__ y, int n,
        const int* __restrict__ perm, int n_epochs, float lr, float beta_1,
        float beta_2, float epsilon, float alpha, double tol, int max_iter,
        int n_iter_no_change, int n_networks,
        float* params, float* adam_m, float* adam_v, float* grads,
        double* acc, double* best_loss, double* loss, int* no_improvement,
        int* n_iter, int* active, int* t_step) {{
    cg::grid_group grid = cg::this_grid();
    extern __shared__ float4 shared4[];
    float* p_s = reinterpret_cast<float*>(shared4);
    float* a_s = p_s + {n_params_shared};
    float* d_s = a_s + {n_act};
    float* d2_s = d_s + CHUNK_PAD * {max_width};
    float* y_s = d2_s + CHUNK_PAD * {max_width};
    __shared__ double red[N_THREADS / 32];

    const int net = blockIdx.x / N_BLOCKS;
    const int part = blockIdx.x % N_BLOCKS;
    const int tid = threadIdx.x;
    float* p_g = params + (size_t)net * N_PARAMS;
    float* g_g = grads + (size_t)net * N_BLOCKS * N_PARAMS;
    float* g = g_g + (size_t)part * N_PARAMS;
    const int n_batches = (n + BATCH - 1) / BATCH;
    int t = t_step[net];
    int is_active = active[net];
    // beta^t for Adam's bias correction, updated incrementally.
    double beta_1_t = pow((double)beta_1, t);
    double beta_2_t = pow((double)beta_2, t);

    for (int epoch = 0; epoch < n_epochs; epoch++) {{
        const int* perm_e = perm + ((size_t)net * n_epochs + epoch) * n;
        for (int batch = 0; batch < n_batches; batch++) {{
            const int start = batch * BATCH;
            const int nb = min(BATCH, n - start);
            const int r_start = part * CHUNK;
            const int cnt = max(0, min(CHUNK, nb - r_start));
            if (is_active) {{
                // Load parameters and this block's part of the batch.
{load_params}
                for (int i = tid; i < CHUNK_PAD * {n_dim}; i += N_THREADS) {{
                    const int r = i / {n_dim};
                    a_s[i] = r < cnt ? x[(size_t)perm_e[start + r_start + r] *
                                         {n_dim} + i % {n_dim}] : 0.0f;
                }}
                for (int r = tid; r < CHUNK_PAD; r += N_THREADS)
                    y_s[r] = r < cnt ? y[perm_e[start + r_start + r]] : 0.0f;
                __syncthreads();
{forward}
                // Squared residuals for the loss.
                double sq = 0.0;
                for (int r = tid; r < CHUNK_PAD; r += N_THREADS)
                    sq += (double)d_s[r] * d_s[r];
                for (int o = 16; o > 0; o /= 2)
                    sq += __shfl_down_sync(0xffffffff, sq, o);
                if (tid % 32 == 0) red[tid / 32] = sq;
                __syncthreads();
                if (tid == 0) {{
                    for (int i = 1; i < N_THREADS / 32; i++) sq += red[i];
                    atomicAdd(acc + net, 0.5 * sq);
                }}
{backward}
            }}
            grid.sync();

            // Combine gradients and update this block's slice with Adam.
            if (is_active) {{
                t += 1;
                beta_1_t *= beta_1;
                beta_2_t *= beta_2;
                const float lr_t = lr * sqrtf((float)(1.0 - beta_2_t)) /
                                   (float)(1.0 - beta_1_t);
                const int i_start = part * SLICE;
                const int i_end = min(N_PARAMS, i_start + SLICE);
                double w2 = 0.0;
                #pragma unroll 2
                for (int i = i_start + tid; i < i_end; i += N_THREADS) {{
                    float g_i = 0.0f;
                    #pragma unroll
                    for (int b = 0; b < N_BLOCKS; b++)
                        g_i += g_g[(size_t)b * N_PARAMS + i];
                    const float p = p_g[i];
                    if (alpha != 0.0f && is_weight(i)) {{
                        g_i += alpha * p;
                        w2 += (double)p * p;
                    }}
                    g_i /= nb;
                    const size_t k = (size_t)net * N_PARAMS + i;
                    const float m = beta_1 * adam_m[k] + (1.0f - beta_1) * g_i;
                    const float v = (beta_2 * adam_v[k] +
                                     (1.0f - beta_2) * g_i * g_i);
                    adam_m[k] = m;
                    adam_v[k] = v;
                    p_g[i] = p - lr_t * m / (sqrtf(v) + epsilon);
                }}
                if (alpha != 0.0f) {{
                    for (int o = 16; o > 0; o /= 2)
                        w2 += __shfl_down_sync(0xffffffff, w2, o);
                    if (tid % 32 == 0) atomicAdd(acc + net, 0.5 * alpha * w2);
                }}
            }}
            grid.sync();
        }}

        // Stopping criterion as in scikit-learn without early stopping.
        if (part == 0 && tid == 0 && is_active) {{
            const double l = acc[net] / n;
            acc[net] = 0.0;
            loss[net] = l;
            n_iter[net] += 1;
            no_improvement[net] = l > best_loss[net] - tol ?
                                  no_improvement[net] + 1 : 0;
            if (l < best_loss[net]) best_loss[net] = l;
            if (no_improvement[net] > n_iter_no_change ||
                    n_iter[net] >= max_iter)
                active[net] = 0;
        }}
        grid.sync();
        is_active = active[net];
        // Stop early if all networks have converged.
        int any_active = 0;
        for (int i = 0; i < n_networks; i++) any_active |= active[i];
        if (!any_active) break;
    }}
    if (part == 0 && tid == 0) t_step[net] = t;
}}
"""


def is_available():
    """Check whether CuPy and a cooperative-launch capable GPU are usable.

    Returns
    -------
    available : bool
        Whether the CUDA backend can be used.

    """
    try:
        import cupy
        device = cupy.cuda.Device()
        return bool(device.attributes['CooperativeLaunch'])
    except Exception:
        return False


def is_supported(neural_network_kwargs, n_dim, n_networks):
    """Check whether the CUDA backend can train the requested networks.

    Parameters
    ----------
    neural_network_kwargs : dict
        Keyword arguments for the constructor of MLPRegressor.
    n_dim : int
        Number of input dimensions.
    n_networks : int
        Number of networks to train simultaneously.

    Returns
    -------
    supported : bool
        Whether the settings are supported and a suitable GPU is available.

    """
    if not _settings_supported(neural_network_kwargs) or not is_available():
        return False
    import cupy
    device = cupy.cuda.Device()
    # All thread blocks must fit on the GPU at the same time.
    if n_networks > device.attributes['MultiProcessorCount']:
        return False
    kwargs = SUPPORTED_KWARGS | neural_network_kwargs
    batch_size = 200 if kwargs['batch_size'] == 'auto' else int(
        kwargs['batch_size'])
    sizes = (n_dim, ) + tuple(
        np.atleast_1d(kwargs['hidden_layer_sizes']).tolist()) + (1, )
    n_blocks = _n_blocks(n_networks)
    return (_shared_memory(sizes, batch_size, n_blocks) <=
            device.attributes['MaxSharedMemoryPerBlockOptin'])


def _n_blocks(n_networks):
    """Number of thread blocks each network is distributed over."""
    import cupy
    n_sm = cupy.cuda.Device().attributes['MultiProcessorCount']
    return max(1, min(8, n_sm // n_networks))


def _shared_memory(sizes, batch_size, n_blocks):
    """Bytes of shared memory per block needed by the kernel."""
    return _layout(sizes, batch_size, n_blocks)['n_shared']


def _layout(sizes, batch_size, n_blocks, rows=5):
    """Offsets of all buffers used by the kernel."""
    chunk = -(-batch_size // n_blocks)
    chunk_pad = -(-chunk // rows) * rows

    def align(i):
        # Keep all shared memory buffers aligned to 16 bytes.
        return -(-i // 4) * 4

    # Offsets of weights and biases in global memory (dense) and shared
    # memory (weights with an odd row stride).
    layout = dict(w_offset=[], b_offset=[], w_offset_s=[], b_offset_s=[],
                  stride=[], chunk=chunk, chunk_pad=chunk_pad, rows=rows)
    i = i_s = 0
    for s_in, s_out in zip(sizes[:-1], sizes[1:]):
        stride = s_out if s_out % 2 == 1 else s_out + 1
        layout['stride'].append(stride)
        layout['w_offset'].append(i)
        layout['b_offset'].append(i + s_in * s_out)
        i += s_in * s_out + s_out
        layout['w_offset_s'].append(i_s)
        layout['b_offset_s'].append(i_s + s_in * stride)
        i_s += s_in * stride + s_out
    layout['n_params'] = i
    layout['n_params_shared'] = align(i_s)
    a_offset = [0]
    for s in sizes[:-2]:
        a_offset.append(a_offset[-1] + align(chunk_pad * s))
    layout['a_offset'] = a_offset
    layout['n_act'] = align(a_offset[-1] + chunk_pad * sizes[-2])
    layout['max_width'] = align(max(sizes[1:]))
    layout['n_shared'] = 4 * (
        layout['n_params_shared'] + layout['n_act'] +
        2 * chunk_pad * layout['max_width'] + chunk_pad)
    return layout


@lru_cache
def _make_kernel(sizes, batch_size, n_blocks, n_threads):
    """Compile the training kernel for the given architecture.

    Parameters
    ----------
    sizes : tuple
        Number of neurons in each layer, including input and output.
    batch_size : int
        Number of points per minibatch.
    n_blocks : int
        Number of thread blocks each network is distributed over.
    n_threads : int
        Number of threads per block.

    Returns
    -------
    kernel : cupy.RawKernel
        The compiled kernel.
    n_params : int
        Number of parameters per network.
    n_shared : int
        Bytes of dynamic shared memory per block.

    """
    import cupy

    layout = _layout(sizes, batch_size, n_blocks)
    w_offset, b_offset = layout['w_offset'], layout['b_offset']
    w_offset_s, b_offset_s = layout['w_offset_s'], layout['b_offset_s']
    stride, a_offset = layout['stride'], layout['a_offset']
    n_params = layout['n_params']

    load, forward, backward = [], [], []
    layers = list(zip(sizes[:-1], sizes[1:]))
    for l, (s_in, s_out) in enumerate(layers):
        load.append(
            f"#pragma unroll 4\n"
            f"for (int i = tid; i < {s_in * s_out}; i += N_THREADS) "
            f"p_s[{w_offset_s[l]} + i / {s_out} * {stride[l]} + i % {s_out}]"
            f" = p_g[{w_offset[l]} + i];\n"
            f"for (int i = tid; i < {s_out}; i += N_THREADS) "
            f"p_s[{b_offset_s[l]} + i] = p_g[{b_offset[l]} + i];")
        a_out = (f"a_s + {a_offset[l + 1]}" if l < len(layers) - 1 else
                 "d_s")
        forward.append(
            f"forward_layer<{s_in}, {s_out}, {stride[l]}, "
            f"{'true' if l == len(layers) - 1 else 'false'}>("
            f"a_s + {a_offset[l]}, p_s + {w_offset_s[l]}, "
            f"p_s + {b_offset_s[l]}, {a_out}, y_s, cnt);\n__syncthreads();")
    d_in, d_out = "d_s", "d2_s"
    for l in reversed(range(len(layers))):
        s_in, s_out = layers[l]
        code = (f"gradient_layer<{s_in}, {s_out}>(a_s + {a_offset[l]}, "
                f"{d_in}, g + {w_offset[l]}, g + {b_offset[l]});")
        if l > 0:
            code += (f"\ndelta_layer<{s_in}, {s_out}, {stride[l]}>({d_in}, "
                     f"p_s + {w_offset_s[l]}, a_s + {a_offset[l]}, {d_out});")
            d_in, d_out = d_out, d_in
        backward.append(code + "\n__syncthreads();")

    def indent(lines):
        return "\n".join("                " + line for block in lines
                         for line in block.split("\n"))

    is_weight = " || ".join(f"(i >= {w} && i < {b})" for w, b in
                            zip(w_offset, b_offset))
    source = (_DEVICE_FUNCTIONS + _KERNEL).format(
        batch=batch_size, chunk=layout['chunk'],
        chunk_pad=layout['chunk_pad'], n_blocks=n_blocks,
        n_threads=n_threads, n_params=n_params,
        slice=-(-n_params // n_blocks), rows=layout['rows'], cols=4,
        is_weight=is_weight, n_params_shared=layout['n_params_shared'],
        n_act=layout['n_act'], max_width=layout['max_width'],
        n_dim=sizes[0], load_params=indent(load), forward=indent(forward),
        backward=indent(backward))
    n_shared = layout['n_shared']
    kernel = cupy.RawModule(code=source, options=('--std=c++17', ),
                            enable_cooperative_groups=True
                            ).get_function('train')
    kernel.max_dynamic_shared_size_bytes = n_shared
    return kernel, n_params, n_shared


def train_networks(x, y, n_networks, neural_network_kwargs, seed=0,
                   epochs_per_launch=16):
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
        supported according to `_jax_mlp.is_supported`.
    seed : int, optional
        Seed for weight initialization and shuffling. Default is 0.
    epochs_per_launch : int, optional
        How many epochs to run per kernel launch. Default is 16.

    Returns
    -------
    networks : list of sklearn.neural_network.MLPRegressor
        The trained networks.

    """
    import cupy

    kwargs = SUPPORTED_KWARGS | neural_network_kwargs
    n, n_dim = x.shape
    if kwargs['batch_size'] == 'auto':
        batch_size = min(200, n)
    else:
        batch_size = int(np.clip(kwargs['batch_size'], 1, n))
    layer_sizes = tuple(np.atleast_1d(kwargs['hidden_layer_sizes']).tolist())
    sizes = (n_dim, ) + layer_sizes + (1, )

    n_blocks = _n_blocks(n_networks)
    n_threads = 256
    kernel, n_params, n_shared = _make_kernel(
        sizes, batch_size, n_blocks, n_threads)

    # Glorot-uniform initialization, as in scikit-learn.
    rng = np.random.default_rng(seed)
    params = np.empty((n_networks, n_params), dtype=np.float32)
    for i in range(n_networks):
        p = []
        for s_in, s_out in zip(sizes[:-1], sizes[1:]):
            bound = np.sqrt(6.0 / (s_in + s_out))
            p.append(rng.uniform(-bound, bound, s_in * s_out))
            p.append(rng.uniform(-bound, bound, s_out))
        params[i] = np.concatenate(p)

    hyper = tuple(np.float32(kwargs[key]) for key in [
        'learning_rate_init', 'beta_1', 'beta_2', 'epsilon', 'alpha']) + (
        np.float64(kwargs['tol']), np.int32(kwargs['max_iter']),
        np.int32(kwargs['n_iter_no_change']), np.int32(n_networks))

    x_d = cupy.asarray(x, dtype=cupy.float32)
    y_d = cupy.asarray(y, dtype=cupy.float32)
    params_d = cupy.asarray(params)
    m_d = cupy.zeros_like(params_d)
    v_d = cupy.zeros_like(params_d)
    grads_d = cupy.empty((n_networks, n_blocks, n_params), dtype=cupy.float32)
    acc_d = cupy.zeros(n_networks, dtype=cupy.float64)
    best_d = cupy.full(n_networks, np.inf, dtype=cupy.float64)
    loss_d = cupy.full(n_networks, np.inf, dtype=cupy.float64)
    no_imp_d = cupy.zeros(n_networks, dtype=cupy.int32)
    n_iter_d = cupy.zeros(n_networks, dtype=cupy.int32)
    active_d = cupy.ones(n_networks, dtype=cupy.int32)
    t_d = cupy.zeros(n_networks, dtype=cupy.int32)
    rng_d = cupy.random.default_rng(seed)

    max_iter = int(kwargs['max_iter'])
    epochs_done = 0
    while epochs_done < max_iter:
        n_epochs = min(epochs_per_launch, max_iter - epochs_done)
        # Independent random permutation per network and epoch.
        perm = cupy.argsort(rng_d.random((n_networks, n_epochs, n),
                                         dtype=cupy.float32), axis=-1
                            ).astype(cupy.int32)
        kernel((n_networks * n_blocks, ), (n_threads, ), (
            x_d, y_d, np.int32(n), perm, np.int32(n_epochs)) + hyper + (
            params_d, m_d, v_d, grads_d, acc_d, best_d, loss_d,
            no_imp_d, n_iter_d, active_d, t_d), shared_mem=n_shared)
        epochs_done += n_epochs
        if not cupy.any(active_d):
            break

    params = cupy.asnumpy(params_d)
    weights = []
    for i, (s_in, s_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        start = sum(a * b + b for a, b in zip(sizes[:i], sizes[1:i + 1]))
        weights.append((
            params[:, start:start + s_in * s_out].reshape(-1, s_in, s_out),
            params[:, start + s_in * s_out:start + s_in * s_out + s_out]))
    return networks_from_params(
        weights, cupy.asnumpy(n_iter_d), cupy.asnumpy(loss_d),
        cupy.asnumpy(best_d), n, layer_sizes, neural_network_kwargs, seed)


def predict(emulator, x):
    """Evaluate an emulator on the GPU with CuPy.

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
    import cupy

    if emulator not in _DEVICE_WEIGHTS:
        coefs, intercepts = emulator._stacked_weights()
        _DEVICE_WEIGHTS[emulator] = (
            [cupy.asarray(c) for c in coefs],
            [cupy.asarray(i) for i in intercepts])
    coefs, intercepts = _DEVICE_WEIGHTS[emulator]
    n_networks = coefs[1].shape[0]
    a = cupy.asarray(x) @ coefs[0]
    a += intercepts[0]
    cupy.maximum(a, 0, out=a)
    a = a.reshape(len(x), n_networks, -1).transpose(1, 0, 2)
    for i in range(1, len(coefs)):
        a = cupy.matmul(a, coefs[i])
        a += intercepts[i]
        if i < len(coefs) - 1:
            cupy.maximum(a, 0, out=a)
    return cupy.asnumpy(cupy.mean(a[:, :, 0], axis=0)).astype(float)
