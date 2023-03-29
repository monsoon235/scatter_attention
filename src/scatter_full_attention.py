import math

import torch
from torch.autograd import Function
from torch_scatter import scatter_softmax
from einops import rearrange
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from const import LH_DEFAULT, LD_DEFAULT, MAX_THREADS_PER_BLOCK
from utils import zero_cost_repeat


@cuda.jit
def numba_full_attention_step1_forward(
        Q: DeviceNDArray,
        K: DeviceNDArray,
        K_match: DeviceNDArray,
        K_match_offset: DeviceNDArray,
        attn: DeviceNDArray,
):
    q_idx = cuda.blockIdx.x
    # m_idx is parallelized because attn is accumulated in dim D

    LQ, LH, LD = Q.shape
    LM = K_match.shape[0]

    assert LH == LH_DEFAULT
    assert LD == LD_DEFAULT

    start = cuda.shared.array(1, K_match.dtype)
    end = cuda.shared.array(1, K_match.dtype)
    query_vec = cuda.shared.array((LH_DEFAULT, LD_DEFAULT), Q.dtype)

    if cuda.threadIdx.x == 0:
        start[0] = K_match_offset[q_idx]
        end[0] = K_match_offset[q_idx + 1] if q_idx < LQ - 1 else LM

    for d_idx in range(cuda.threadIdx.x, LD, cuda.blockDim.x):
        for h_idx in range(LH):
            query_vec[h_idx, d_idx] = Q[q_idx, h_idx, d_idx]

    cuda.syncthreads()

    for tid in range(cuda.threadIdx.x, end[0] - start[0], cuda.blockDim.x):
        m_idx = start[0] + tid
        a_idx = m_idx
        k_idx = K_match[m_idx]
        for h_idx in range(LH):
            attn_val = 0
            for d_idx in range(LD):
                q_val = query_vec[h_idx, d_idx]
                k_val = K[k_idx, h_idx, d_idx]
                attn_val += q_val * k_val
            attn[a_idx, h_idx] = attn_val


@cuda.jit
def numba_full_attention_step1_backward_Q(
        attn_grad: DeviceNDArray,
        K: DeviceNDArray,
        K_match: DeviceNDArray,
        K_match_offset: DeviceNDArray,
        Q_grad: DeviceNDArray,
):
    q_idx = cuda.blockIdx.x
    h_idx = cuda.threadIdx.x
    d_idx = cuda.threadIdx.y

    LQ, LH, LD = Q_grad.shape
    LM = K_match.shape[0]

    assert LD == LD_DEFAULT
    assert LH == LH_DEFAULT

    start = cuda.shared.array(1, K_match.dtype)
    end = cuda.shared.array(1, K_match.dtype)
    q_grad_vec = cuda.shared.array((LH_DEFAULT, LD_DEFAULT), Q_grad.dtype)

    if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0:
        start[0] = K_match_offset[q_idx]
        end[0] = K_match_offset[q_idx + 1] if q_idx < LQ - 1 else LM

    q_grad_vec[h_idx, d_idx] = 0

    cuda.syncthreads()

    for m_idx in range(start[0], end[0]):
        k_idx = K_match[m_idx]
        a_idx = m_idx
        k_val = K[k_idx, h_idx, d_idx]
        attn_grad_val = attn_grad[a_idx, h_idx]
        q_grad_vec[h_idx, d_idx] += attn_grad_val * k_val

    cuda.syncthreads()

    Q_grad[q_idx, h_idx, d_idx] = q_grad_vec[h_idx, d_idx]


@cuda.jit
def numba_full_attention_step1_backward_K(
        attn_grad: DeviceNDArray,
        Q: DeviceNDArray,
        Q_match: DeviceNDArray,
        Q_match_offset: DeviceNDArray,
        m_match: DeviceNDArray,
        K_grad: DeviceNDArray,
):
    k_idx = cuda.blockIdx.x
    h_idx = cuda.threadIdx.x
    d_idx = cuda.threadIdx.y

    LK, LH, LD = K_grad.shape
    LM = Q_match.shape[0]

    assert LD == LD_DEFAULT
    assert LH == LH_DEFAULT

    start = cuda.shared.array(1, Q_match.dtype)
    end = cuda.shared.array(1, Q_match.dtype)
    key_grad_vec = cuda.shared.array((LH_DEFAULT, LD_DEFAULT), K_grad.dtype)

    if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0:
        start[0] = Q_match_offset[k_idx]
        end[0] = Q_match_offset[k_idx + 1] if k_idx < LK - 1 else LM

    key_grad_vec[h_idx, d_idx] = 0

    cuda.syncthreads()

    for m_idx in range(start[0], end[0]):
        q_idx = Q_match[m_idx]
        a_idx = m_match[m_idx]
        q_val = Q[q_idx, h_idx, d_idx]
        attn_grad_val = attn_grad[a_idx, h_idx]
        key_grad_vec[h_idx, d_idx] += attn_grad_val * q_val

    cuda.syncthreads()

    K_grad[k_idx, h_idx, d_idx] = key_grad_vec[h_idx, d_idx]


@cuda.jit
def numba_full_attention_step2_forward(
        attn: DeviceNDArray,
        V: DeviceNDArray,
        V_match: DeviceNDArray,
        V_match_offset: DeviceNDArray,
        out: DeviceNDArray,
):
    q_idx = cuda.blockIdx.x
    h_idx = cuda.threadIdx.x
    d_idx = cuda.threadIdx.y
    # h_idx/d_idx is parallelized because attn is accumulated in dim m

    LQ, LH, LD = out.shape
    LM = V_match.shape[0]

    assert LD == LD_DEFAULT
    assert LH == LH_DEFAULT

    start = cuda.shared.array(1, V_match.dtype)
    end = cuda.shared.array(1, V_match.dtype)

    if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0:
        start[0] = V_match_offset[q_idx]
        end[0] = V_match_offset[q_idx + 1] if q_idx < LQ - 1 else LM

    cuda.syncthreads()

    out_val = 0
    for m_idx in range(start[0], end[0]):
        v_idx = V_match[m_idx]
        a_idx = m_idx
        v_val = V[v_idx, h_idx, d_idx]
        attn_val = attn[a_idx, h_idx]
        out_val += v_val * attn_val

    out[q_idx, h_idx, d_idx] = out_val


@cuda.jit
def numba_full_attention_step2_backward(
        out_grad: DeviceNDArray,
        attn: DeviceNDArray,
        V: DeviceNDArray,
        Q_match: DeviceNDArray,
        Q_match_offset: DeviceNDArray,
        m_match: DeviceNDArray,
        attn_grad: DeviceNDArray,
        V_grad: DeviceNDArray,
):
    v_idx = cuda.blockIdx.x
    h_idx = cuda.threadIdx.x

    LV, LH, LD = V.shape
    LM = Q_match.shape[0]

    assert LD == LD_DEFAULT

    start = cuda.shared.array(1, Q_match_offset.dtype)
    end = cuda.shared.array(1, Q_match_offset.dtype)
    value_vec = cuda.shared.array((LH_DEFAULT, LD_DEFAULT), V.dtype)
    value_grad_vec = cuda.shared.array((LH_DEFAULT, LD_DEFAULT), V_grad.dtype)

    if cuda.threadIdx.x == 0:
        start[0] = Q_match_offset[v_idx]
        end[0] = Q_match_offset[v_idx + 1] if v_idx < LV - 1 else LM

    for d_idx in range(LD):
        value_vec[h_idx, d_idx] = V[v_idx, h_idx, d_idx]
        value_grad_vec[h_idx, d_idx] = 0

    cuda.syncthreads()

    for m_idx in range(start[0], end[0]):
        q_idx = Q_match[m_idx]
        a_idx = m_match[m_idx]
        attn_grad_val = 0
        for d_idx in range(LD):
            v_val = value_vec[h_idx, d_idx]
            out_grad_val = out_grad[q_idx, h_idx, d_idx]
            attn_val = attn[a_idx, h_idx]
            value_grad_vec[h_idx, d_idx] += out_grad_val * attn_val
            attn_grad_val += out_grad_val * v_val
        attn_grad[a_idx, h_idx] = attn_grad_val

    cuda.syncthreads()

    for d_idx in range(LD):
        V_grad[v_idx, h_idx, d_idx] = value_grad_vec[h_idx, d_idx]


class FullAttentionStep1(Function):

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, extra) -> torch.Tensor:
        cuda.select_device(Q.device.index)

        K_match = extra['K_match']
        Q_match = extra['Q_match']
        K_match_offset = extra['K_match_offset']
        Q_match_offset = extra['Q_match_offset']
        m_match = extra['m_match']
        max_match_of_q = extra['max_match_of_q']

        LM = K_match.shape[0]
        LQ, LH, LD = Q.shape

        attn = torch.zeros(size=(LM, LH), dtype=Q.dtype, device=Q.device)

        Q_cuda = cuda.as_cuda_array(Q.detach())
        K_cuda = cuda.as_cuda_array(K.detach())
        K_match_cuda = cuda.as_cuda_array(K_match)
        K_match_offset_cuda = cuda.as_cuda_array(K_match_offset)
        attn_cuda = cuda.as_cuda_array(attn)

        blocks = (LQ,)
        threads = (min(max_match_of_q, MAX_THREADS_PER_BLOCK),)
        numba_full_attention_step1_forward[blocks, threads](
            Q_cuda, K_cuda, K_match_cuda, K_match_offset_cuda, attn_cuda,
        )

        ctx.save_for_backward(Q, K, Q_match, K_match, Q_match_offset, K_match_offset, m_match)
        return attn

    @staticmethod
    def backward(ctx, attn_grad):
        cuda.select_device(attn_grad.device.index)

        Q, K, Q_match, K_match, Q_match_offset, K_match_offset, m_match = ctx.saved_tensors

        LQ, LH, LD = Q.shape
        LK = K.shape[0]

        Q_grad = torch.zeros_like(Q)
        K_grad = torch.zeros_like(K)

        Q_cuda = cuda.as_cuda_array(Q.detach())
        K_cuda = cuda.as_cuda_array(K.detach())
        Q_match_cuda = cuda.as_cuda_array(Q_match)
        K_match_cuda = cuda.as_cuda_array(K_match)
        Q_match_offset_cuda = cuda.as_cuda_array(Q_match_offset)
        K_match_offset_cuda = cuda.as_cuda_array(K_match_offset)
        m_match_cuda = cuda.as_cuda_array(m_match)
        attn_grad_cuda = cuda.as_cuda_array(attn_grad)
        Q_grad_cuda = cuda.as_cuda_array(Q_grad)
        K_grad_cuda = cuda.as_cuda_array(K_grad)

        blocks = (LQ,)
        threads = (LH, LD)
        numba_full_attention_step1_backward_Q[blocks, threads](
            attn_grad_cuda, K_cuda, K_match_cuda, K_match_offset_cuda, Q_grad_cuda,
        )

        blocks = (LK,)
        threads = (LH, LD)
        numba_full_attention_step1_backward_K[blocks, threads](
            attn_grad_cuda, Q_cuda, Q_match_cuda, Q_match_offset_cuda, m_match_cuda, K_grad_cuda,
        )

        return Q_grad, K_grad, None


class FullAttentionStep2(Function):

    @staticmethod
    def forward(ctx, attn: torch.Tensor, V: torch.Tensor, extra) -> torch.Tensor:
        cuda.select_device(attn.device.index)

        Q_match = extra['Q_match']
        V_match = extra['V_match']
        Q_match_offset = extra['Q_match_offset']
        V_match_offset = extra['V_match_offset']
        m_match = extra['m_match']

        LV, LH, LD = V.shape
        LQ = V_match_offset.shape[0]

        out = torch.zeros(size=(LQ, LH, LD), dtype=attn.dtype, device=attn.device)

        attn_cuda = cuda.as_cuda_array(attn.detach())
        V_cuda = cuda.as_cuda_array(V.detach())
        V_match_cuda = cuda.as_cuda_array(V_match)
        V_match_offset_cuda = cuda.as_cuda_array(V_match_offset)
        out_cuda = cuda.as_cuda_array(out)

        blocks = (LQ,)
        threads = (LH, LD)
        numba_full_attention_step2_forward[blocks, threads](
            attn_cuda, V_cuda, V_match_cuda, V_match_offset_cuda, out_cuda,
        )

        ctx.save_for_backward(attn, V, Q_match, Q_match_offset, m_match)
        return out

    @staticmethod
    def backward(ctx, out_grad: torch.Tensor):
        cuda.select_device(out_grad.device.index)

        attn, V, Q_match, Q_match_offset, m_match = ctx.saved_tensors

        LV, LH, LD = V.shape

        attn_grad = torch.zeros_like(attn)
        V_grad = torch.zeros_like(V)

        attn_cuda = cuda.as_cuda_array(attn.detach())
        V_cuda = cuda.as_cuda_array(V.detach())
        Q_match_cuda = cuda.as_cuda_array(Q_match)
        Q_match_offset_cuda = cuda.as_cuda_array(Q_match_offset)
        m_match_cuda = cuda.as_cuda_array(m_match)
        out_grad_cuda = cuda.as_cuda_array(out_grad)
        attn_grad_cuda = cuda.as_cuda_array(attn_grad)
        V_grad_cuda = cuda.as_cuda_array(V_grad)

        blocks = (LV,)
        threads = (LH,)
        numba_full_attention_step2_backward[blocks, threads](
            out_grad_cuda, attn_cuda, V_cuda, Q_match_cuda, Q_match_offset_cuda, m_match_cuda,
            attn_grad_cuda, V_grad_cuda,
        )

        return attn_grad, V_grad, None


def full_attention_gateway(
        Q: torch.Tensor,  # [B, LQ, H, D]
        K: torch.Tensor,  # [B, LK, H, D]
        V: torch.Tensor,  # [B, LV, H, D]
        hitmap: torch.Tensor  # [B, LQ, LK]
) -> torch.Tensor:
    B, LQ, LH, LD = Q.shape
    _, LK, _, _ = K.shape
    _, LV, _, _ = V.shape
    device = Q.device
    assert LK == LV
    assert K.device == device and V.device == device and hitmap.device == device

    with torch.no_grad():
        # idx of each Q/K token (batch flattened)
        Q_idx = torch.arange(B * LQ, dtype=torch.long, device=device).view(B, LQ)
        K_idx = torch.arange(B * LK, dtype=torch.long, device=device).view(B, LK)

        # idx of Q of each attention match
        Q_match = zero_cost_repeat(Q_idx, dim=1, repeat=LK)[hitmap.transpose(-2, -1)]  # [LM]
        # num of corresponding Q of each K
        matched_q_num_of_k = hitmap.sum(dim=-2).flatten()  # [B * LK]
        max_match_of_k = int(matched_q_num_of_k.max())
        Q_match_offset = matched_q_num_of_k.cumsum(dim=0) - matched_q_num_of_k  # [B * LK]

        K_match = zero_cost_repeat(K_idx, dim=1, repeat=LQ)[hitmap]
        matched_k_num_of_q = hitmap.sum(dim=-1).flatten()  # [B * LQ]
        max_match_of_q = int(matched_k_num_of_q.max())
        K_match_offset = matched_k_num_of_q.cumsum(dim=0) - matched_k_num_of_q  # [B * LQ]

        LM = Q_match.shape[0]  # num of attention matches
        m_ids_matrix = torch.empty(size=(B, LQ, LK), dtype=torch.long, device=device)
        m_ids_matrix[hitmap] = torch.arange(LM, dtype=torch.long, device=device)
        # m_idx for each matching in view of order of K
        m_match = m_ids_matrix.transpose(-2, -1)[hitmap.transpose(-2, -1)]  # [LM]

        # q_idx of corresponding Q in matching arr
        Q_class = zero_cost_repeat(Q_idx, dim=2, repeat=LK)[hitmap]

    extra = {
        'Q_match': Q_match,
        'K_match': K_match,
        'V_match': K_match,
        'Q_match_offset': Q_match_offset,
        'K_match_offset': K_match_offset,
        'V_match_offset': K_match_offset,
        'm_match': m_match,
        'max_match_of_q': max_match_of_q,
    }

    Q = rearrange(Q, 'b lq h d -> (b lq) h d')
    K = rearrange(K, 'b lk h d -> (b lk) h d')
    V = rearrange(V, 'b lv h d -> (b lv) h d')

    attn = FullAttentionStep1.apply(Q, K, extra)
    attn = attn / math.sqrt(LD)
    attn_softmax = scatter_softmax(attn, Q_class, dim=0)
    out = FullAttentionStep2.apply(attn_softmax, V, extra)  # [B * LQ, LH, LD]
    out = rearrange(out, '(b lq) lh ld -> b lq lh ld', b=B, lq=LQ, lh=LH, ld=LD)

    return out
