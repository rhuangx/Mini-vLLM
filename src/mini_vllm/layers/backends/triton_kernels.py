"""Triton kernels for the triton backend. Every bound is a runtime value and every config is
fixed, so the kernels launch the same way during CUDA-graph capture as outside it."""

from __future__ import annotations

import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def rms_norm_kernel(X, R, W, Y, RES, stride_x, stride_r, n_cols, eps,
                    HAS_RESIDUAL: tl.constexpr, BLOCK: tl.constexpr):
    """One row: y = rms_norm(x [+ r]) * w; with a residual, the sum is also stored to RES.
    The add rounds to the input dtype first, as Hugging Face's does."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    ok = cols < n_cols
    x = tl.load(X + row * stride_x + cols, mask=ok, other=0.0)
    if HAS_RESIDUAL:
        x = x + tl.load(R + row * stride_r + cols, mask=ok, other=0.0)
        tl.store(RES + row * n_cols + cols, x, mask=ok)
    x32 = x.to(tl.float32)
    normed = x32 * tl.rsqrt(tl.sum(x32 * x32, axis=0) / n_cols + eps)
    w = tl.load(W + cols, mask=ok, other=0.0)
    y = normed.to(x.dtype).to(tl.float32) * w.to(tl.float32)
    tl.store(Y + row * n_cols + cols, y.to(Y.dtype.element_ty), mask=ok)


@triton.jit
def silu_and_mul_kernel(X, Y, stride_x, n_cols, BLOCK: tl.constexpr):
    """y = silu(x[:, :n]) * x[:, n:], in fp32."""
    row, block = tl.program_id(0), tl.program_id(1)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    ok = cols < n_cols
    gate = tl.load(X + row * stride_x + cols, mask=ok, other=0.0).to(tl.float32)
    up = tl.load(X + row * stride_x + n_cols + cols, mask=ok, other=0.0).to(tl.float32)
    y = gate / (1.0 + tl.exp(-gate)) * up
    tl.store(Y + row * n_cols + cols, y.to(Y.dtype.element_ty), mask=ok)


@triton.jit
def rope_kernel(X, POS, COS, SIN, Y, num_heads, stride_xt, stride_xh, HALF: tl.constexpr,
                BLOCK: tl.constexpr):
    """One (token, head): [x1, x2] -> [x1 c - x2 s, x2 c + x1 s] at the token's position."""
    token, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BLOCK)
    ok = d < HALF
    pos = tl.load(POS + token).to(tl.int64)
    base = X + token * stride_xt + head * stride_xh
    x1 = tl.load(base + d, mask=ok, other=0.0).to(tl.float32)
    x2 = tl.load(base + HALF + d, mask=ok, other=0.0).to(tl.float32)
    c = tl.load(COS + pos * 2 * HALF + d, mask=ok, other=0.0)
    s = tl.load(SIN + pos * 2 * HALF + d, mask=ok, other=0.0)
    out = Y + (token * num_heads + head) * 2 * HALF
    tl.store(out + d, (x1 * c - x2 * s).to(Y.dtype.element_ty), mask=ok)
    tl.store(out + HALF + d, (x2 * c + x1 * s).to(Y.dtype.element_ty), mask=ok)


@triton.jit
def store_kv_kernel(K, V, KC, VC, SLOTS, k_scale, v_scale, stride_k, stride_v,
                    ROW: tl.constexpr, BLOCK: tl.constexpr, FP8: tl.constexpr):
    """One token's H_k x D keys and values into its slot; slot -1 (a padded row) writes nothing.
    fp8 pages take (x / scale) saturated to e4m3, as the bits of a uint8."""
    token = tl.program_id(0)
    slot = tl.load(SLOTS + token).to(tl.int64)
    if slot < 0:
        return
    cols = tl.arange(0, BLOCK)
    ok = cols < ROW
    k = tl.load(K + token * stride_k + cols, mask=ok, other=0.0)
    v = tl.load(V + token * stride_v + cols, mask=ok, other=0.0)
    if FP8:
        k = tl.clamp(k.to(tl.float32) / k_scale, -448.0, 448.0).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        v = tl.clamp(v.to(tl.float32) / v_scale, -448.0, 448.0).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(KC + slot * ROW + cols, k.to(KC.dtype.element_ty), mask=ok)
    tl.store(VC + slot * ROW + cols, v.to(VC.dtype.element_ty), mask=ok)


@triton.jit
def paged_attention_kernel(
    Q, KC, VC, OUT, TABLES, CU_SEQLENS, CONTEXT_LENS, seq_offset, sm_scale, k_scale, v_scale,
    stride_qt, stride_qh, stride_ot, stride_oh, stride_table, stride_slot, stride_kv_head,
    GROUP: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr, PAGE: tl.constexpr,
    Q_TOKENS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    FP8: tl.constexpr, IEEE: tl.constexpr,
):
    """One program: Q_TOKENS query tokens of one sequence times the GROUP query heads that share
    one KV head, so each K/V tile is read once for the whole group. Keys come from the pages
    through the block table, fresh and cached alike, so one kernel serves decodes, verifies,
    chunked prefills and prefix hits. Query token i sees context positions <= ctx - q_len + i;
    a sequence with no context (a padded row) yields zeros."""
    tile, kv_head, seq = tl.program_id(0), tl.program_id(1), tl.program_id(2) + seq_offset
    q_start = tl.load(CU_SEQLENS + seq)
    q_len = tl.load(CU_SEQLENS + seq + 1) - q_start
    first = tile * Q_TOKENS
    if first >= q_len:
        return
    history = tl.load(CONTEXT_LENS + seq) - q_len

    m = tl.arange(0, BLOCK_M)
    token = first + m // GROUP
    head = kv_head * GROUP + m % GROUP
    row_ok = (m < Q_TOKENS * GROUP) & (token < q_len)
    d = tl.arange(0, BLOCK_D)
    d_ok = d < D
    q_ptrs = Q + (q_start + token)[:, None] * stride_qt + head[:, None] * stride_qh + d[None, :]
    q = tl.load(q_ptrs, mask=row_ok[:, None] & d_ok[None, :], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    end = history + tl.minimum(first + Q_TOKENS, q_len)  # the last row's causal bound
    qk_scale = sm_scale * 1.4426950408889634
    for start in range(0, end, BLOCK_N):
        n = start + tl.arange(0, BLOCK_N)
        n_ok = n < end
        block = tl.load(TABLES + seq * stride_table + n // PAGE, mask=n_ok, other=0).to(tl.int64)
        slot = block * PAGE + n % PAGE
        kv_ptrs = slot[:, None] * stride_slot + kv_head * stride_kv_head + d[None, :]
        kv_ok = n_ok[:, None] & d_ok[None, :]
        k = tl.load(KC + kv_ptrs, mask=kv_ok, other=0)
        v = tl.load(VC + kv_ptrs, mask=kv_ok, other=0)
        if FP8:
            k = (k.to(tl.float8e4nv, bitcast=True).to(tl.float32) * k_scale).to(q.dtype)
            v = (v.to(tl.float8e4nv, bitcast=True).to(tl.float32) * v_scale).to(q.dtype)
        if IEEE:  # noqa: SIM108 (a constexpr branch, resolved at compile time)
            scores = tl.dot(q, tl.trans(k), input_precision="ieee")
        else:
            scores = tl.dot(q, tl.trans(k))
        visible = (n[None, :] <= history + token[:, None]) & n_ok[None, :]
        scores = tl.where(visible, scores * qk_scale, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(scores - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        if IEEE:
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    out = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
    o_ptrs = OUT + (q_start + token)[:, None] * stride_ot + head[:, None] * stride_oh + d[None, :]
    tl.store(o_ptrs, out.to(OUT.dtype.element_ty), mask=row_ok[:, None] & d_ok[None, :])
