//! Thinking Machines Inkling 2 Mini (`model_type: inkling2_loop`) text forward.
//!
//! Reference: ml-explore/mlx-lm PR #2086 (mlx_lm/models/inkling2/{model,layers,checkpoint}.py).
//! 24 physical layers with middle-layer looping (layers 6..17 x 3 passes = 48 visits),
//! BlockAttentionResidual across 6-layer blocks, cross-pass MessageInput/MessageGate,
//! RelativeLogits + 4 short-convs per layer + sinks, and SoftcappedSwiGLU MoE (128 experts, top-8).

const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const log = @import("log.zig");

const A = mlx.mlx_array;
const S = mlx.mlx_stream;
const none: A = .{ .ctx = null };

const Scope = struct {
    a: std.mem.Allocator,
    s: S,
    items: std.ArrayList(A) = .empty,

    fn init(a: std.mem.Allocator, s: S) Scope {
        return .{ .a = a, .s = s };
    }
    fn deinit(self: *Scope) void {
        for (self.items.items) |x| _ = mlx.mlx_array_free(x);
        self.items.deinit(self.a);
    }
    fn keep(self: *Scope, x: A) !A {
        self.items.append(self.a, x) catch |e| {
            _ = mlx.mlx_array_free(x);
            return e;
        };
        return x;
    }
    fn res(self: *Scope, rc: c_int, o: *const A) !A {
        mlx.check(rc) catch |e| {
            _ = mlx.mlx_array_free(o.*);
            return e;
        };
        return self.keep(o.*);
    }
    fn out(_: *Scope, x: A) A {
        var o = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&o, x);
        return o;
    }
    fn op1(sc: *Scope, comptime fun: anytype, x: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(fun(&o, x, sc.s), &o);
    }
    fn op2(sc: *Scope, comptime fun: anytype, x: A, y: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(fun(&o, x, y, sc.s), &o);
    }
    fn add(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_add, x, y);
    }
    fn sub(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_subtract, x, y);
    }
    fn mul(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_multiply, x, y);
    }
    fn div(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_divide, x, y);
    }
    fn matmul(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_matmul, x, y);
    }
    fn f(sc: *Scope, v: f32) !A {
        return sc.keep(mlx.mlx_array_new_float(v));
    }
    fn astype(sc: *Scope, x: A, dt: mlx.mlx_dtype) !A {
        if (mlx.mlx_array_dtype(x) == dt) return x;
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_astype(&o, x, dt, sc.s), &o);
    }
    fn reshape(sc: *Scope, x: A, shape: []const c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, sc.s), &o);
    }
    fn transpose(sc: *Scope, x: A, axes: []const c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, sc.s), &o);
    }
    fn broadcast(sc: *Scope, x: A, shape: []const c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_broadcast_to(&o, x, shape.ptr, shape.len, sc.s), &o);
    }
    fn expand(sc: *Scope, x: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_expand_dims(&o, x, axis, sc.s), &o);
    }
    fn squeeze(sc: *Scope, x: A, axis: c_int) !A {
        const sh = mlx.getShape(x);
        const ax: usize = if (axis < 0) @intCast(@as(isize, @intCast(sh.len)) + axis) else @intCast(axis);
        var nsh: [8]c_int = undefined;
        var ni: usize = 0;
        for (sh, 0..) |d, i| {
            if (i != ax) {
                nsh[ni] = d;
                ni += 1;
            }
        }
        return sc.reshape(x, nsh[0..ni]);
    }
    fn sliceAxis(sc: *Scope, x: A, axis_in: isize, lo: usize, hi: usize) !A {
        const sh = mlx.getShape(x);
        const axis: usize = if (axis_in < 0) @intCast(@as(isize, @intCast(sh.len)) + axis_in) else @intCast(axis_in);
        var start: [8]c_int = @splat(0);
        var stop: [8]c_int = undefined;
        const st: [8]c_int = @splat(1);
        @memcpy(stop[0..sh.len], sh);
        start[axis] = @intCast(lo);
        stop[axis] = @intCast(hi);
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_slice(&o, x, &start, sh.len, &stop, sh.len, &st, sh.len, sc.s), &o);
    }
    fn concat(sc: *Scope, xs: []const A, axis: c_int) !A {
        if (xs.len == 1) return xs[0];
        const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_concatenate_axis(&o, vec, axis, sc.s), &o);
    }
    fn sliceUpdate(sc: *Scope, x: A, upd: A, start: []const c_int, stop: []const c_int) !A {
        const strides: [8]c_int = @splat(1);
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_slice_update(&o, x, upd, start.ptr, start.len, stop.ptr, stop.len, &strides, start.len, sc.s), &o);
    }
    fn argmax(sc: *Scope, x: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_argmax_axis(&o, x, axis, false, sc.s), &o);
    }
    fn neg(sc: *Scope, x: A) !A {
        return sc.op1(mlx.mlx_negative, x);
    }
    fn ints(sc: *Scope, vals: []const i32) !A {
        const sh = [_]c_int{@intCast(vals.len)};
        return sc.keep(mlx.mlx_array_new_data(vals.ptr, &sh, 1, .int32));
    }
    fn take(sc: *Scope, x: A, idx: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_take_axis(&o, x, idx, axis, sc.s), &o);
    }
    fn takeAlong(sc: *Scope, x: A, idx: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_take_along_axis(&o, x, idx, axis, sc.s), &o);
    }
    fn sum(sc: *Scope, x: A, axis: c_int, keep_: bool) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_sum_axis(&o, x, axis, keep_, sc.s), &o);
    }
    fn mean(sc: *Scope, x: A, axis: c_int, keep_: bool) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_mean_axis(&o, x, axis, keep_, sc.s), &o);
    }
    fn softmax(sc: *Scope, x: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_softmax_axis(&o, x, axis, true, sc.s), &o);
    }
    fn where(sc: *Scope, c: A, x: A, y: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_where(&o, c, x, y, sc.s), &o);
    }
    fn sqrt(sc: *Scope, x: A) !A {
        return sc.op1(mlx.mlx_sqrt, x);
    }
    fn sigmoid(sc: *Scope, x: A) !A {
        return sc.op1(mlx.mlx_sigmoid, x);
    }
    fn tanh(sc: *Scope, x: A) !A {
        return sc.op1(mlx.mlx_tanh, x);
    }
    fn minimum(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_minimum, x, y);
    }
    fn maximum(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_maximum, x, y);
    }
    fn zeros(sc: *Scope, shape: []const c_int, dt: mlx.mlx_dtype) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_zeros(&o, shape.ptr, shape.len, dt, sc.s), &o);
    }
    fn rms(sc: *Scope, x: A, w: A, eps: f32) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_fast_rms_norm(&o, x, w, eps, sc.s), &o);
    }
    fn rmsUnweighted(sc: *Scope, x: A, eps: f32) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_fast_rms_norm(&o, try sc.astype(x, .float32), none, eps, sc.s), &o);
    }
};

const QLin = struct {
    w: A = none,
    s: A = none,
    b: A = none,
    bits: u32 = 0,
    gs: u32 = 64,

    fn apply(self: *const QLin, sc: *Scope, x: A) !A {
        if (self.s.ctx == null) {
            const tr = try sc.transpose(self.w, &.{ 1, 0 });
            const xf = if (mlx.mlx_array_dtype(x) != mlx.mlx_array_dtype(tr))
                try sc.astype(x, mlx.mlx_array_dtype(tr))
            else
                x;
            return sc.matmul(xf, tr);
        }
        var o = mlx.mlx_array_new();
        const x_in = try sc.astype(x, mlx.mlx_array_dtype(self.s));
        return sc.res(mlx.mlx_quantized_matmul(
            &o,
            x_in,
            self.w,
            self.s,
            self.b,
            true,
            mlx.mlx_optional_int.some(@intCast(self.gs)),
            mlx.mlx_optional_int.some(@intCast(self.bits)),
            "affine",
            sc.s,
        ), &o);
    }

    fn gatherMm(self: *const QLin, sc: *Scope, x: A, idx: A) !A {
        var o = mlx.mlx_array_new();
        if (self.s.ctx == null) {
            const tr = try sc.transpose(self.w, &.{ 0, 2, 1 });
            return sc.res(mlx.mlx_gather_mm(&o, x, tr, none, idx, false, sc.s), &o);
        }
        const x_in = try sc.astype(x, mlx.mlx_array_dtype(self.s));
        return sc.res(mlx.mlx_gather_qmm(
            &o,
            x_in,
            self.w,
            self.s,
            self.b,
            none,
            idx,
            true,
            mlx.mlx_optional_int.some(@intCast(self.gs)),
            mlx.mlx_optional_int.some(@intCast(self.bits)),
            "affine",
            false,
            sc.s,
        ), &o);
    }
};

const MessageGateWeights = struct {
    down: QLin = .{},
    forget: QLin = .{},
    gain: QLin = .{},
};

const LayerWeights = struct {
    is_global: bool,
    input_norm: A = none,
    q_input_norm: A = none,
    k_input_norm: A = none,
    v_input_norm: A = none,
    post_attn_norm: A,

    q_proj: QLin,
    k_proj: QLin,
    v_proj: QLin,
    r_proj: QLin,
    o_proj: QLin,
    gate_proj: QLin,
    sink_proj: QLin,
    rel_proj: A,
    q_l2_scale: A,
    rel_l2_scale: A,
    k_sconv: A = none,
    v_sconv: A = none,
    k_sconv_static: A = none,
    k_sconv_proj: QLin = .{},
    v_sconv_static: A = none,
    v_sconv_proj: QLin = .{},
    attn_sconv: A,
    mlp_sconv: A,

    router_w: A,
    router_bias: A,
    sw_gate: QLin,
    sw_up: QLin,
    sw_down: QLin,

    attn_res_attn_q: A = none,
    attn_res_mlp_q: A,

    has_controllers: bool = false,
    ctrl_q: MessageGateWeights = .{},
    ctrl_k: MessageGateWeights = .{},
    ctrl_v: MessageGateWeights = .{},
    ctrl_mlp: MessageGateWeights = .{},
};

const VisitCache = struct {
    k: A = none,
    v: A = none,
    conv_k: A = none,
    conv_v: A = none,
    conv_attn: A = none,
    conv_mlp: A = none,
    /// The full `pre-chunk || chunk` conv window of the last forward, kept so a
    /// partial accept can rebuild the state at the accepted row.
    win_k: A = none,
    win_v: A = none,
    win_attn: A = none,
    win_mlp: A = none,

    fn reset(self: *VisitCache) void {
        inline for (&.{ &self.k, &self.v, &self.conv_k, &self.conv_v, &self.conv_attn, &self.conv_mlp, &self.win_k, &self.win_v, &self.win_attn, &self.win_mlp }) |p| {
            if (p.*.ctx != null) {
                _ = mlx.mlx_array_free(p.*);
                p.* = none;
            }
        }
    }
};

// ── DSpark draft (`mtp.safetensors`) ──────────────────────────────────────
//
// The checkpoint's native speculator: the anchor plus `block - 1` mask rows
// through three dense layers whose K/V also read the target's captured
// features (`target_layer_ids`), then the trunk head on the draft rows. The
// block is BIDIRECTIONAL: a row sees the whole block, plus the context rows
// of the tokens before the anchor (the last `sliding_window` of them on a
// local layer).

const CtxKV = struct {
    /// [1, Hkv, len, head_dim]
    k: A = none,
    v: A = none,
    len: usize = 0,

    fn reset(self: *CtxKV) void {
        if (self.k.ctx != null) _ = mlx.mlx_array_free(self.k);
        if (self.v.ctx != null) _ = mlx.mlx_array_free(self.v);
        self.* = .{};
    }
};

const DsparkLayer = struct {
    is_local: bool = false,
    input_norm: A = none,
    post_attn_norm: A = none,
    dflare_norm: A = none,
    q_norm: A = none,
    k_norm: A = none,
    k_norm_ctx: A = none,
    q_proj: QLin = .{},
    k_proj: QLin = .{},
    v_proj: QLin = .{},
    o_proj: QLin = .{},
    k_proj_ctx: QLin = .{},
    v_proj_ctx: QLin = .{},
    gate_proj: QLin = .{},
    up_proj: QLin = .{},
    down_proj: QLin = .{},
};

/// The checkpoint's `markov_head` (`markov_head_type: rnn_hidden`): a GRU-like
/// recurrence over the draft block that restores intra-block dependency. Its
/// gate/candidate/output projection reads
/// `[state, markov_w1[prev_token], draft_hidden, fused_target_feature]`; the
/// output latent is projected back to `hidden_size` by `hidden_out` and folded
/// through the trunk head, i.e. a logit-space bias (SGLang's
/// `Inkling2MTPHead`). `dflare_norm` normalizes the fused target feature of the
/// last committed position.
const Markov = struct {
    /// `[padded_vocab, rank]`, BF16 embedding of the previous token.
    w1: A = none,
    /// `[2*rank + 2*hidden, 3*rank]` — `rnn_proj.weight` pre-transposed.
    rnn_w_t: A = none,
    /// `[3*rank]`.
    rnn_b: A = none,
    /// `[rank, hidden]` — `hidden_out.weight` pre-transposed.
    out_w_t: A = none,
    /// `[hidden]`.
    out_b: A = none,
    /// `[hidden]`.
    dflare_norm: A = none,
    rank: usize = 0,

    fn deinit(self: *Markov, a: std.mem.Allocator) void {
        _ = a;
        if (self.rnn_w_t.ctx != null) _ = mlx.mlx_array_free(self.rnn_w_t);
        if (self.out_w_t.ctx != null) _ = mlx.mlx_array_free(self.out_w_t);
    }
};

const Dspark = struct {
    /// Rows per draft block, the anchor included.
    block: usize,
    layers: []DsparkLayer,
    norm: A = none,
    mask_embed: A = none,
    /// softmax(`fuse_weight`, -1): `[layers + 1, taps]`.
    fuse_alpha: A = none,
    ctx: []CtxKV,
    /// Sliding window a local draft layer reads before the block.
    window: usize,
    rope_theta: f32 = 1000000.0,
    markov: ?Markov = null,

    fn deinit(self: *Dspark, a: std.mem.Allocator) void {
        for (self.ctx) |*c| c.reset();
        a.free(self.ctx);
        a.free(self.layers);
        if (self.fuse_alpha.ctx != null) _ = mlx.mlx_array_free(self.fuse_alpha);
        if (self.markov) |*m| m.deinit(a);
    }
};

/// Rotary table for absolute positions `[p0, p0 + n)` in the duplicated
/// (rotate-half) layout, f32, shaped `[1, n, 1, head_dim]`.
fn ropeTable(sc: *Scope, p0: usize, n: usize, hd: usize, theta: f32, out: *[2]A) !void {
    const half = hd / 2;
    const cos_buf = try sc.a.alloc(f32, n * hd);
    defer sc.a.free(cos_buf);
    const sin_buf = try sc.a.alloc(f32, n * hd);
    defer sc.a.free(sin_buf);
    for (0..n) |i| {
        for (0..half) |j| {
            const expo = -2.0 * @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(hd));
            const ang = @as(f32, @floatFromInt(p0 + i)) * std.math.pow(f32, theta, expo);
            const c = @cos(ang);
            const s = @sin(ang);
            cos_buf[i * hd + j] = c;
            cos_buf[i * hd + half + j] = c;
            sin_buf[i * hd + j] = s;
            sin_buf[i * hd + half + j] = s;
        }
    }
    const shape = [_]c_int{ 1, @intCast(n), 1, @intCast(hd) };
    out[0] = try sc.keep(mlx.mlx_array_new_data(cos_buf.ptr, &shape, 4, .float32));
    out[1] = try sc.keep(mlx.mlx_array_new_data(sin_buf.ptr, &shape, 4, .float32));
}

/// RoPE over a `[1, n, heads, head_dim]` array at absolute positions `[p0, p0 + n)`.
fn ropeApply(sc: *Scope, x: A, p0: usize, n: usize, hd: usize, theta: f32) !A {
    var tab: [2]A = .{ none, none };
    try ropeTable(sc, p0, n, hd, theta, &tab);
    const half = @divExact(hd, 2);
    const x1 = try sc.sliceAxis(x, 3, 0, half);
    const x2 = try sc.sliceAxis(x, 3, half, hd);
    const rot = try sc.concat(&.{ try sc.neg(x2), x1 }, 3);
    const dt = mlx.mlx_array_dtype(x);
    const out = try sc.add(try sc.mul(try sc.astype(x, .float32), tab[0]), try sc.mul(try sc.astype(rot, .float32), tab[1]));
    return sc.astype(out, dt);
}

fn loadDsparkLayer(weights: *const model.Weights, i: usize, local: bool) !DsparkLayer {
    var kb: [192]u8 = undefined;
    const p = std.fmt.bufPrint(&kb, "layers.{d}", .{i}) catch return error.NameTooLong;
    const key = struct {
        fn k(buf: []u8, prefix: []const u8, suffix: []const u8) ![]const u8 {
            return std.fmt.bufPrint(buf, "{s}.{s}", .{ prefix, suffix }) catch error.NameTooLong;
        }
    };
    var nb: [192]u8 = undefined;
    return .{
        .is_local = local,
        .input_norm = try Model.getReq(weights, try key.k(&nb, p, "input_layernorm.weight")),
        .post_attn_norm = try Model.getReq(weights, try key.k(&nb, p, "post_attention_layernorm.weight")),
        .dflare_norm = try Model.getReq(weights, try key.k(&nb, p, "dflare_norm.weight")),
        .q_norm = try Model.getReq(weights, try key.k(&nb, p, "self_attn.q_norm.weight")),
        .k_norm = try Model.getReq(weights, try key.k(&nb, p, "self_attn.k_norm.weight")),
        .k_norm_ctx = try Model.getReq(weights, try key.k(&nb, p, "self_attn.k_norm_ctx.weight")),
        .q_proj = .{ .w = try Model.getReq(weights, try key.k(&nb, p, "self_attn.q_proj.weight")) },
        .k_proj = .{ .w = try Model.getReq(weights, try key.k(&nb, p, "self_attn.k_proj.weight")) },
        .v_proj = .{ .w = try Model.getReq(weights, try key.k(&nb, p, "self_attn.v_proj.weight")) },
        .o_proj = .{ .w = try Model.getReq(weights, try key.k(&nb, p, "self_attn.o_proj.weight")) },
        .k_proj_ctx = .{ .w = try Model.getReq(weights, try key.k(&nb, p, "self_attn.k_proj_ctx.weight")) },
        .v_proj_ctx = .{ .w = try Model.getReq(weights, try key.k(&nb, p, "self_attn.v_proj_ctx.weight")) },
        .gate_proj = .{ .w = try Model.getReq(weights, try key.k(&nb, p, "mlp.gate_proj.weight")) },
        .up_proj = .{ .w = try Model.getReq(weights, try key.k(&nb, p, "mlp.up_proj.weight")) },
        .down_proj = .{ .w = try Model.getReq(weights, try key.k(&nb, p, "mlp.down_proj.weight")) },
    };
}

fn loadMarkov(sc: *Scope, weights: *const model.Weights) !?Markov {
    if (weights.get("markov_head.markov_w1.weight") == null) return null;
    if (std.c.getenv("MLX_SERVE_INKLING_MARKOV")) |v| if (v[0] == '0') return null;
    const rnn_w = try Model.getReq(weights, "markov_head.rnn_proj.weight");
    const out_w = try Model.getReq(weights, "markov_head.hidden_out.weight");
    const w1 = try Model.getReq(weights, "markov_head.markov_w1.weight");
    const rnn_w_t = sc.out(try sc.transpose(try sc.astype(rnn_w, .float32), &.{ 1, 0 }));
    const out_w_t = sc.out(try sc.transpose(try sc.astype(out_w, .float32), &.{ 1, 0 }));
    try mlx.check(mlx.mlx_array_eval(rnn_w_t));
    try mlx.check(mlx.mlx_array_eval(out_w_t));
    return Markov{
        .w1 = w1,
        .rnn_w_t = rnn_w_t,
        .rnn_b = try Model.getReq(weights, "markov_head.rnn_proj.bias"),
        .out_w_t = out_w_t,
        .out_b = try Model.getReq(weights, "markov_head.hidden_out.bias"),
        .dflare_norm = try Model.getReq(weights, "markov_head.dflare_norm.weight"),
        .rank = @intCast(mlx.getShape(w1)[1]),
    };
}

fn loadDspark(a: std.mem.Allocator, cfg: model.ModelConfig, weights: *const model.Weights, s: S) !?*Dspark {
    if (cfg.dspark_block < 2 or cfg.dspark_num_layers == 0) return null;
    if (weights.get("fuse_weight") == null or weights.get("mask_embed") == null) return null;
    const n_layers: usize = @intCast(cfg.dspark_num_layers);
    const layers = try a.alloc(DsparkLayer, n_layers);
    errdefer a.free(layers);
    for (0..n_layers) |i| {
        const local = (cfg.dspark_local_mask & (@as(u32, 1) << @intCast(i))) != 0;
        layers[i] = try loadDsparkLayer(weights, i, local);
    }
    const ctx = try a.alloc(CtxKV, n_layers);
    errdefer a.free(ctx);
    for (ctx) |*c| c.* = .{};

    var sc = Scope.init(a, s);
    defer sc.deinit();
    const alpha = try sc.softmax(try sc.astype(try Model.getReq(weights, "fuse_weight"), .float32), -1);
    try mlx.check(mlx.mlx_array_eval(alpha));

    const dp = try a.create(Dspark);
    dp.* = .{
        .block = @intCast(cfg.dspark_block),
        .layers = layers,
        .ctx = ctx,
        .window = @intCast(cfg.sliding_window),
        .rope_theta = cfg.dspark_rope_theta,
        .norm = try Model.getReq(weights, "norm.weight"),
        .mask_embed = try Model.getReq(weights, "mask_embed"),
        .fuse_alpha = sc.out(alpha),
        .markov = try loadMarkov(&sc, weights),
    };
    if (dp.markov != null) log.info("[inkling2] DSpark markov head loaded (rnn_hidden, rank={d})\n", .{dp.markov.?.rank});
    return dp;
}

fn draftMlp(sc: *Scope, lw: *const DsparkLayer, x: A) !A {
    const g = try lw.gate_proj.apply(sc, x);
    const u = try lw.up_proj.apply(sc, x);
    const gf = try sc.astype(g, .float32);
    const act = try sc.mul(try sc.mul(gf, try sc.sigmoid(gf)), try sc.astype(u, .float32));
    return lw.down_proj.apply(sc, try sc.astype(act, mlx.mlx_array_dtype(g)));
}

pub const DsparkRound = struct {
    /// Committed this round, in order: `[t1, accepted drafts…]`.
    tokens: []u32,
    /// Drafts accepted (the always-committed `t1` excluded).
    accepted: u32,
    /// Next round's trunk token, from the ORIGINAL verify logits.
    next_token: u32,
    /// Absolute position count after the round.
    position: usize,

    pub fn deinit(self: *DsparkRound, allocator: std.mem.Allocator) void {
        allocator.free(self.tokens);
    }
};

pub const Model = struct {
    a: std.mem.Allocator,
    s: S,
    cfg: model.ModelConfig,
    embed: QLin,
    embed_norm: A,
    norm: A,
    unembed: QLin,
    output_query: A,
    message_query: A,
    layers: []LayerWeights,
    caches: []VisitCache,
    offset: usize = 0,
    /// Absolute position / row count of the LAST forward's input chunk.
    fwd_pos0: usize = 0,
    pending_n: usize = 0,
    /// Captured target features of the last forward: `[1, n, taps * hidden]`.
    cap: A = none,
    cap_pos: usize = 0,
    cap_n: usize = 0,
    /// Row of `cap` that belongs to the last COMMITTED token — the Markov
    /// head's fused target feature (`f_rnn`). Prefill: the chunk's last row; a
    /// verify round: the last accepted row.
    cap_commit_idx: usize = 0,
    /// Positions whose DSpark context K/V is committed.
    ctx_len: usize = 0,
    dspark: ?*Dspark = null,
    owned: std.ArrayList(A) = .empty,
    compiled_softcap: mlx.mlx_closure = .{ .ctx = null },
    compiled_l2_q: mlx.mlx_closure = .{ .ctx = null },
    compiled_l2_k: mlx.mlx_closure = .{ .ctx = null },
    compiled_l2_rel: mlx.mlx_closure = .{ .ctx = null },
    ops_compiled: bool = false,

    fn compileOne(comptime cb: anytype, payload: ?*anyopaque) mlx.mlx_closure {
        const raw = mlx.mlx_closure_new_func_payload(&cb, payload, null);
        defer _ = mlx.mlx_closure_free(raw);
        var comp = mlx.mlx_closure{ .ctx = null };
        if (mlx.mlx_compile(&comp, raw, true) == 0 and comp.ctx != null) return comp;
        return .{ .ctx = null };
    }

    fn applyCompiled2(comp: mlx.mlx_closure, sc: *Scope, x: A, y: A) !A {
        const args = [_]A{ x, y };
        const in_vec = mlx.mlx_vector_array_new_data(&args, 2);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var out_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(out_vec);
        try mlx.check(mlx.mlx_closure_apply(&out_vec, comp, in_vec));
        var res = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&res, out_vec, 0));
        return sc.keep(res);
    }

    fn applyCompiled1(comp: mlx.mlx_closure, sc: *Scope, x: A) !A {
        const args = [_]A{x};
        const in_vec = mlx.mlx_vector_array_new_data(&args, 1);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var out_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(out_vec);
        try mlx.check(mlx.mlx_closure_apply(&out_vec, comp, in_vec));
        var res = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&res, out_vec, 0));
        return sc.keep(res);
    }

    fn softcapCb(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
        const self: *const Model = @ptrCast(@alignCast(payload.?));
        var gate = mlx.mlx_array_new();
        var up = mlx.mlx_array_new();
        if (mlx.mlx_vector_array_get(&gate, input, 0) != 0) return -1;
        defer _ = mlx.mlx_array_free(gate);
        if (mlx.mlx_vector_array_get(&up, input, 1) != 0) return -1;
        defer _ = mlx.mlx_array_free(up);
        var sc = Scope.init(self.a, self.s);
        defer sc.deinit();
        const dt = mlx.mlx_array_dtype(gate);
        const g32 = sc.astype(gate, .float32) catch return -1;
        const u32_a = sc.astype(up, .float32) catch return -1;
        const gs_f = sc.f(self.cfg.mlp_gate_softcap) catch return -1;
        const us_f = sc.f(self.cfg.mlp_up_softcap) catch return -1;
        const g_cap = sc.mul(sc.tanh(sc.div(g32, gs_f) catch return -1) catch return -1, gs_f) catch return -1;
        const u_cap = sc.mul(sc.tanh(sc.div(u32_a, us_f) catch return -1) catch return -1, us_f) catch return -1;
        const act = sc.astype(sc.mul(sc.mul(g_cap, sc.sigmoid(g32) catch return -1) catch return -1, u_cap) catch return -1, dt) catch return -1;
        const out_arr = [_]A{act};
        res.* = mlx.mlx_vector_array_new_data(&out_arr, 1);
        return 0;
    }

    fn l2ClipScaledCb(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque, threshold: f32) c_int {
        const self: *const Model = @ptrCast(@alignCast(payload.?));
        var x = mlx.mlx_array_new();
        var scale = mlx.mlx_array_new();
        if (mlx.mlx_vector_array_get(&x, input, 0) != 0) return -1;
        defer _ = mlx.mlx_array_free(x);
        if (mlx.mlx_vector_array_get(&scale, input, 1) != 0) return -1;
        defer _ = mlx.mlx_array_free(scale);
        var sc = Scope.init(self.a, self.s);
        defer sc.deinit();
        const out = l2Clip(&sc, x, scale, threshold) catch return -1;
        const out_arr = [_]A{out};
        res.* = mlx.mlx_vector_array_new_data(&out_arr, 1);
        return 0;
    }

    fn l2ClipQCb(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
        const self: *const Model = @ptrCast(@alignCast(payload.?));
        return l2ClipScaledCb(res, input, payload, self.cfg.qk_l2_clip_threshold);
    }

    fn l2ClipRelCb(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
        const self: *const Model = @ptrCast(@alignCast(payload.?));
        return l2ClipScaledCb(res, input, payload, self.cfg.rel_l2_clip_threshold);
    }

    fn l2ClipKCb(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
        const self: *const Model = @ptrCast(@alignCast(payload.?));
        var x = mlx.mlx_array_new();
        if (mlx.mlx_vector_array_get(&x, input, 0) != 0) return -1;
        defer _ = mlx.mlx_array_free(x);
        var sc = Scope.init(self.a, self.s);
        defer sc.deinit();
        const out = l2Clip(&sc, x, null, self.cfg.qk_l2_clip_threshold) catch return -1;
        const out_arr = [_]A{out};
        res.* = mlx.mlx_vector_array_new_data(&out_arr, 1);
        return 0;
    }

    fn getReq(weights: *const model.Weights, key: []const u8) !A {
        return weights.get(key) orelse {
            log.err("inkling2_loop: missing weight {s}\n", .{key});
            return error.MissingWeight;
        };
    }

    fn loadQLin(weights: *const model.Weights, prefix: []const u8, bits_default: u32, gs_default: u32) !QLin {
        var wb: [192]u8 = undefined;
        var sb: [192]u8 = undefined;
        var bb: [192]u8 = undefined;
        const wk = std.fmt.bufPrint(&wb, "{s}.weight", .{prefix}) catch return error.NameTooLong;
        const w = try getReq(weights, wk);
        const sk = std.fmt.bufPrint(&sb, "{s}.scales", .{prefix}) catch return error.NameTooLong;
        const sc = weights.get(sk) orelse none;
        if (sc.ctx == null) return .{ .w = w };
        const bk = std.fmt.bufPrint(&bb, "{s}.biases", .{prefix}) catch return error.NameTooLong;
        const bi = weights.get(bk) orelse none;
        const bits: u32 = if (bits_default > 0) bits_default else 4;
        const gs: u32 = if (gs_default > 0) gs_default else 64;
        return .{ .w = w, .s = sc, .b = bi, .bits = bits, .gs = gs };
    }

    fn loadController(weights: *const model.Weights, layer_prefix: []const u8, name: []const u8) !MessageGateWeights {
        var b1: [192]u8 = undefined;
        var b2: [192]u8 = undefined;
        var b3: [192]u8 = undefined;
        const dw = try getReq(weights, std.fmt.bufPrint(&b1, "{s}.controllers.{s}.down_weight", .{ layer_prefix, name }) catch return error.NameTooLong);
        const fw = try getReq(weights, std.fmt.bufPrint(&b2, "{s}.controllers.{s}.forget_weight", .{ layer_prefix, name }) catch return error.NameTooLong);
        const gw = try getReq(weights, std.fmt.bufPrint(&b3, "{s}.controllers.{s}.gain_weight", .{ layer_prefix, name }) catch return error.NameTooLong);
        return .{
            .down = .{ .w = dw },
            .forget = .{ .w = fw },
            .gain = .{ .w = gw },
        };
    }

    pub fn init(a: std.mem.Allocator, cfg: model.ModelConfig, weights: *const model.Weights, s: S) !Model {
        const bits = cfg.quant_bits;
        const gs = cfg.quant_group_size;
        var owned: std.ArrayList(A) = .empty;
        errdefer {
            for (owned.items) |x| _ = mlx.mlx_array_free(x);
            owned.deinit(a);
        }

        const embed = try loadQLin(weights, "model.llm.embed", bits, gs);
        const embed_norm = try getReq(weights, "model.llm.embed_norm.weight");
        const norm = try getReq(weights, "model.llm.norm.weight");
        const unembed = try loadQLin(weights, "model.llm.unembed", bits, gs);
        const output_query = try getReq(weights, "model.llm.attn_res.output_query");

        // Build message_query [core_len, 4, hidden_size] from cross_pass_queries and cross_pass_kv_queries
        const cp_q = try getReq(weights, "model.llm.middle_cycle_cross_pass_queries");
        const cp_kv = try getReq(weights, "model.llm.middle_cycle_cross_pass_kv_queries");
        var message_query = none;
        {
            var sc = Scope.init(a, s);
            defer sc.deinit();
            const dims: c_int = @intCast(cfg.hidden_size);
            const q_r = try sc.reshape(try sc.astype(cp_q, .float32), &.{ -1, 2, dims });
            const kv_r = try sc.reshape(try sc.astype(cp_kv, .float32), &.{ -1, 2, dims });
            const q0 = try sc.sliceAxis(q_r, 1, 0, 1);
            const q1 = try sc.sliceAxis(q_r, 1, 1, 2);
            const mq = try sc.concat(&.{ q0, kv_r, q1 }, 1);
            try mlx.check(mlx.mlx_array_eval(mq));
            message_query = sc.out(mq);
            try owned.append(a, message_query);
        }

        const phys_n: usize = @intCast(cfg.physical_num_layers);
        const layers = try a.alloc(LayerWeights, phys_n);
        errdefer a.free(layers);

        const total_visits: usize = @intCast(cfg.num_hidden_layers);
        const caches = try a.alloc(VisitCache, total_visits);
        errdefer a.free(caches);
        for (caches) |*c| c.* = .{};

        // Determine physical layer global flags
        for (0..phys_n) |pi| {
            var lb: [128]u8 = undefined;
            var kb: [192]u8 = undefined;
            const lp = std.fmt.bufPrint(&lb, "model.llm.layers.{d}", .{pi}) catch return error.NameTooLong;
            const is_global = weights.get(std.fmt.bufPrint(&kb, "{s}.q_input_norm.weight", .{lp}) catch return error.NameTooLong) != null;

            var input_norm = none;
            var q_in = none;
            var k_in = none;
            var v_in = none;
            if (is_global) {
                q_in = try getReq(weights, std.fmt.bufPrint(&kb, "{s}.q_input_norm.weight", .{lp}) catch return error.NameTooLong);
                k_in = try getReq(weights, std.fmt.bufPrint(&kb, "{s}.k_input_norm.weight", .{lp}) catch return error.NameTooLong);
                v_in = try getReq(weights, std.fmt.bufPrint(&kb, "{s}.v_input_norm.weight", .{lp}) catch return error.NameTooLong);
            } else {
                input_norm = try getReq(weights, std.fmt.bufPrint(&kb, "{s}.attn_norm.weight", .{lp}) catch return error.NameTooLong);
            }
            const post_attn_norm = try getReq(weights, std.fmt.bufPrint(&kb, "{s}.mlp_norm.weight", .{lp}) catch return error.NameTooLong);

            var attn_res_attn_q = none;
            if (pi > 0) {
                attn_res_attn_q = try getReq(weights, std.fmt.bufPrint(&kb, "model.llm.attn_res.attn_queries.{d}", .{pi - 1}) catch return error.NameTooLong);
            }
            const attn_res_mlp_q = try getReq(weights, std.fmt.bufPrint(&kb, "model.llm.attn_res.mlp_queries.{d}", .{pi}) catch return error.NameTooLong);

            // Cast router_w to float32 once at load
            const rw_raw = try getReq(weights, std.fmt.bufPrint(&kb, "{s}.mlp.gate.weight", .{lp}) catch return error.NameTooLong);
            var rw_f32 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&rw_f32, rw_raw, .float32, s));
            try mlx.check(mlx.mlx_array_eval(rw_f32));
            try owned.append(a, rw_f32);

            const has_ctrl = pi >= cfg.loop_core_start and pi < cfg.loop_core_end;
            layers[pi] = .{
                .is_global = is_global,
                .input_norm = input_norm,
                .q_input_norm = q_in,
                .k_input_norm = k_in,
                .v_input_norm = v_in,
                .post_attn_norm = post_attn_norm,
                .q_proj = try loadQLin(weights, std.fmt.bufPrint(&kb, "{s}.attn.wq_du", .{lp}) catch return error.NameTooLong, bits, gs),
                .k_proj = try loadQLin(weights, std.fmt.bufPrint(&kb, "{s}.attn.wk_dv", .{lp}) catch return error.NameTooLong, bits, gs),
                .v_proj = try loadQLin(weights, std.fmt.bufPrint(&kb, "{s}.attn.wv_dv", .{lp}) catch return error.NameTooLong, bits, gs),
                .r_proj = try loadQLin(weights, std.fmt.bufPrint(&kb, "{s}.attn.wr_du", .{lp}) catch return error.NameTooLong, bits, gs),
                .o_proj = try loadQLin(weights, std.fmt.bufPrint(&kb, "{s}.attn.wo_ud", .{lp}) catch return error.NameTooLong, bits, gs),
                .gate_proj = try loadQLin(weights, std.fmt.bufPrint(&kb, "{s}.attn.gate_attn", .{lp}) catch return error.NameTooLong, bits, gs),
                .sink_proj = try loadQLin(weights, std.fmt.bufPrint(&kb, "{s}.attn.attn_sink_proj", .{lp}) catch return error.NameTooLong, bits, gs),
                .rel_proj = try getReq(weights, std.fmt.bufPrint(&kb, "{s}.attn.rel_logits_proj.proj", .{lp}) catch return error.NameTooLong),
                .q_l2_scale = try getReq(weights, std.fmt.bufPrint(&kb, "{s}.attn.q_l2_clip.scale", .{lp}) catch return error.NameTooLong),
                .rel_l2_scale = try getReq(weights, std.fmt.bufPrint(&kb, "{s}.attn.rel_l2_clip.scale", .{lp}) catch return error.NameTooLong),
                .k_sconv = if (!is_global) try getReq(weights, std.fmt.bufPrint(&kb, "{s}.attn.k_sconv.weight", .{lp}) catch return error.NameTooLong) else none,
                .v_sconv = if (!is_global) try getReq(weights, std.fmt.bufPrint(&kb, "{s}.attn.v_sconv.weight", .{lp}) catch return error.NameTooLong) else none,
                .k_sconv_static = if (is_global) try getReq(weights, std.fmt.bufPrint(&kb, "{s}.attn.k_sconv.static_w", .{lp}) catch return error.NameTooLong) else none,
                .k_sconv_proj = if (is_global) .{ .w = try getReq(weights, std.fmt.bufPrint(&kb, "{s}.attn.k_sconv.weight_proj", .{lp}) catch return error.NameTooLong) } else .{},
                .v_sconv_static = if (is_global) try getReq(weights, std.fmt.bufPrint(&kb, "{s}.attn.v_sconv.static_w", .{lp}) catch return error.NameTooLong) else none,
                .v_sconv_proj = if (is_global) .{ .w = try getReq(weights, std.fmt.bufPrint(&kb, "{s}.attn.v_sconv.weight_proj", .{lp}) catch return error.NameTooLong) } else .{},
                .attn_sconv = try getReq(weights, std.fmt.bufPrint(&kb, "{s}.attn_sconv.weight", .{lp}) catch return error.NameTooLong),
                .mlp_sconv = try getReq(weights, std.fmt.bufPrint(&kb, "{s}.mlp_sconv.weight", .{lp}) catch return error.NameTooLong),
                .router_w = rw_f32,
                .router_bias = try getReq(weights, std.fmt.bufPrint(&kb, "{s}.mlp.gate.bias", .{lp}) catch return error.NameTooLong),
                .sw_gate = try loadQLin(weights, std.fmt.bufPrint(&kb, "{s}.mlp.switch_mlp.gate_proj", .{lp}) catch return error.NameTooLong, bits, gs),
                .sw_up = try loadQLin(weights, std.fmt.bufPrint(&kb, "{s}.mlp.switch_mlp.up_proj", .{lp}) catch return error.NameTooLong, bits, gs),
                .sw_down = try loadQLin(weights, std.fmt.bufPrint(&kb, "{s}.mlp.switch_mlp.down_proj", .{lp}) catch return error.NameTooLong, bits, gs),
                .attn_res_attn_q = attn_res_attn_q,
                .attn_res_mlp_q = attn_res_mlp_q,
                .has_controllers = has_ctrl,
                .ctrl_q = if (has_ctrl) try loadController(weights, lp, "attn") else .{},
                .ctrl_k = if (has_ctrl) try loadController(weights, lp, "k") else .{},
                .ctrl_v = if (has_ctrl) try loadController(weights, lp, "v") else .{},
                .ctrl_mlp = if (has_ctrl) try loadController(weights, lp, "mlp") else .{},
            };
        }

        const dspark = try loadDspark(a, cfg, weights, s);

        return .{
            .a = a,
            .s = s,
            .cfg = cfg,
            .dspark = dspark,
            .embed = embed,
            .embed_norm = embed_norm,
            .norm = norm,
            .unembed = unembed,
            .output_query = output_query,
            .message_query = message_query,
            .layers = layers,
            .caches = caches,
            .owned = owned,
        };
    }

    fn compileOps(self: *Model) void {
        self.ops_compiled = true;
        self.compiled_softcap = compileOne(softcapCb, @ptrCast(self));
        self.compiled_l2_q = compileOne(l2ClipQCb, @ptrCast(self));
        self.compiled_l2_k = compileOne(l2ClipKCb, @ptrCast(self));
        self.compiled_l2_rel = compileOne(l2ClipRelCb, @ptrCast(self));
    }

    pub fn deinit(self: *Model) void {
        if (self.compiled_softcap.ctx != null) _ = mlx.mlx_closure_free(self.compiled_softcap);
        if (self.compiled_l2_q.ctx != null) _ = mlx.mlx_closure_free(self.compiled_l2_q);
        if (self.compiled_l2_k.ctx != null) _ = mlx.mlx_closure_free(self.compiled_l2_k);
        if (self.compiled_l2_rel.ctx != null) _ = mlx.mlx_closure_free(self.compiled_l2_rel);
        self.resetState();
        if (self.cap.ctx != null) _ = mlx.mlx_array_free(self.cap);
        if (self.dspark) |dp| {
            dp.deinit(self.a);
            self.a.destroy(dp);
        }
        self.a.free(self.caches);
        self.a.free(self.layers);
        for (self.owned.items) |x| _ = mlx.mlx_array_free(x);
        self.owned.deinit(self.a);
    }

    pub fn resetState(self: *Model) void {
        for (self.caches) |*c| c.reset();
        self.offset = 0;
        self.ctx_len = 0;
        self.fwd_pos0 = 0;
        self.pending_n = 0;
        self.cap_commit_idx = 0;
        if (self.cap.ctx != null) {
            _ = mlx.mlx_array_free(self.cap);
            self.cap = none;
        }
        if (self.dspark) |dp| for (dp.ctx) |*c| c.reset();
    }

    fn applyEmbed(self: *const Model, sc: *Scope, ids: []const u32) !A {
        const sh = [_]c_int{ 1, @intCast(ids.len) };
        const id_arr = try sc.keep(mlx.mlx_array_new_data(ids.ptr, &sh, 2, .uint32));
        if (self.embed.s.ctx == null) {
            return sc.take(self.embed.w, id_arr, 0);
        }
        const rows_w = try sc.take(self.embed.w, id_arr, 0);
        const rows_s = try sc.take(self.embed.s, id_arr, 0);
        const rows_b = try sc.take(self.embed.b, id_arr, 0);
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_dequantize(
            &o,
            rows_w,
            rows_s,
            rows_b,
            mlx.mlx_optional_int.some(@intCast(self.embed.gs)),
            mlx.mlx_optional_int.some(@intCast(self.embed.bits)),
            "affine",
            none,
            .{ .value = .bfloat16, .has_value = true },
            sc.s,
        ), &o);
    }

    fn l2Clip(sc: *Scope, x: A, scale: ?A, threshold: f32) !A {
        const dt = mlx.mlx_array_dtype(x);
        const xf = try sc.astype(x, .float32);
        const states = if (scale) |s_arr| try sc.mul(xf, try sc.astype(s_arr, .float32)) else xf;
        const sq = try sc.mul(states, states);
        const nrm = try sc.sqrt(try sc.sum(sq, -1, true));
        const ratio = try sc.div(try sc.f(threshold), try sc.add(nrm, try sc.f(1e-6)));
        const factor = try sc.minimum(ratio, try sc.f(1.0));
        return sc.astype(try sc.mul(states, factor), dt);
    }

    /// block_states: slice of [1, S, D] f32 arrays; partial: [1, S, D] f32; query: [Q, D] f32
    /// returns [S, Q, D] f32
    fn depthAttention(self: *const Model, sc: *Scope, block_states: []const A, partial: A, query: A) !A {
        const dims: f32 = @floatFromInt(self.cfg.hidden_size);
        const scale = try sc.f(1.0 / @sqrt(dims));
        const S_len: c_int = mlx.getShape(partial)[1];
        const D_len: c_int = mlx.getShape(partial)[2];

        // Build [N+1, S, D] using stack buffer when N < 16
        var stack_buf: [16]A = undefined;
        const need = block_states.len + 1;
        const list = if (need <= stack_buf.len) stack_buf[0..need] else try sc.a.alloc(A, need);
        defer if (need > stack_buf.len) sc.a.free(list);
        for (block_states, 0..) |bs, i| {
            list[i] = try sc.reshape(bs, &.{ 1, S_len, D_len });
        }
        list[block_states.len] = try sc.reshape(partial, &.{ 1, S_len, D_len });
        const stacked_raw = try sc.concat(list, 0); // [N+1, S, D]
        const stacked = try sc.transpose(stacked_raw, &.{ 1, 0, 2 }); // [S, N+1, D]
        const q_f = try sc.astype(query, .float32);
        const q_2d = try sc.reshape(q_f, &.{ -1, D_len }); // [Q, D]
        const q_t = try sc.transpose(q_2d, &.{ 1, 0 }); // [D, Q]
        const scores = try sc.mul(try sc.matmul(stacked, q_t), scale); // [S, N+1, Q]
        const weights = try sc.softmax(scores, -2); // [S, N+1, Q]
        const w_t = try sc.transpose(weights, &.{ 0, 2, 1 }); // [S, Q, N+1]
        return sc.matmul(w_t, stacked); // [S, Q, D]
    }

    fn shortConv(sc: *Scope, x: A, w: A, prev_state: *A, win_out: *A, K: usize) !A {
        const dt = mlx.mlx_array_dtype(x);
        const sh = mlx.getShape(x); // [1, S, C]
        const S_len: usize = @intCast(sh[1]);
        const C_len: c_int = sh[2];
        const xf = try sc.astype(x, .float32);
        const left = if (prev_state.*.ctx != null)
            try sc.astype(prev_state.*, .float32)
        else
            try sc.zeros(&.{ 1, @intCast(K - 1), C_len }, .float32);
        const win = try sc.concat(&.{ left, xf }, 1); // [1, K-1+S, C]

        const win_w = try sc.astype(win, mlx.mlx_array_dtype(w));
        var conv_o = mlx.mlx_array_new();
        const conv = try sc.res(mlx.mlx_conv1d(&conv_o, win_w, w, 1, 0, 1, C_len, sc.s), &conv_o);
        const inv_sqrt = try sc.f(1.0 / @sqrt(@as(f32, @floatFromInt(K + 1))));
        const out = try sc.mul(try sc.add(try sc.astype(conv, .float32), xf), inv_sqrt);

        // Save new conv state [1, K-1, C] and the window it came from (lazy,
        // evaluated at end of forward); a partial accept re-slices the window.
        const next_st = try sc.sliceAxis(win, 1, S_len, S_len + K - 1);
        if (prev_state.*.ctx != null) _ = mlx.mlx_array_free(prev_state.*);
        prev_state.* = sc.out(next_st);
        if (win_out.*.ctx != null) _ = mlx.mlx_array_free(win_out.*);
        win_out.* = sc.out(win);

        return sc.astype(out, dt);
    }

    fn dynamicShortConv(
        self: *const Model,
        sc: *Scope,
        x: A,
        dyn_in: A,
        static_w: A,
        w_proj: *const QLin,
        prev_state: *A,
        win_out: *A,
        K: usize,
    ) !A {
        const dt = mlx.mlx_array_dtype(x);
        const sh = mlx.getShape(x); // [1, S, C]
        const S_len: usize = @intCast(sh[1]);
        const S_c: c_int = sh[1];
        const C_len: c_int = sh[2];
        const H_kv: c_int = @intCast(self.cfg.num_key_value_heads);
        const D_kv: c_int = @divExact(C_len, H_kv);
        const K_c: c_int = @intCast(K);

        const dyn_w = try sc.reshape(try sc.astype(try w_proj.apply(sc, dyn_in), .float32), &.{ 1, S_c, H_kv, 1, K_c });
        const st_t = try sc.transpose(try sc.astype(static_w, .float32), &.{ 1, 0 }); // [C, K]
        const st_w = try sc.reshape(st_t, &.{ H_kv, D_kv, K_c });
        const weights = try sc.add(st_w, dyn_w); // [1, S, Hkv, Dkv, K]

        const xf = try sc.astype(x, .float32);
        const left = if (prev_state.*.ctx != null)
            try sc.astype(prev_state.*, .float32)
        else
            try sc.zeros(&.{ 1, @intCast(K - 1), C_len }, .float32);
        const win = try sc.concat(&.{ left, xf }, 1); // [1, K-1+S, C]

        var lags = try sc.a.alloc(A, K);
        defer sc.a.free(lags);
        for (0..K) |lag| {
            const off = K - 1 - lag;
            const sl = try sc.sliceAxis(win, 1, off, off + S_len); // [1, S, C]
            lags[lag] = try sc.reshape(sl, &.{ 1, S_c, H_kv, D_kv, 1 });
        }
        const windows = try sc.concat(lags, -1); // [1, S, Hkv, Dkv, K]
        const conv_sum = try sc.reshape(try sc.sum(try sc.mul(windows, weights), -1, false), &.{ 1, S_c, C_len });
        const inv_sqrt = try sc.f(1.0 / @sqrt(@as(f32, @floatFromInt(2 * K + 1))));
        const out = try sc.mul(try sc.add(conv_sum, xf), inv_sqrt);

        const next_st = try sc.sliceAxis(win, 1, S_len, S_len + K - 1);
        if (prev_state.*.ctx != null) _ = mlx.mlx_array_free(prev_state.*);
        prev_state.* = sc.out(next_st);
        if (win_out.*.ctx != null) _ = mlx.mlx_array_free(win_out.*);
        win_out.* = sc.out(win);

        return sc.astype(out, dt);
    }

    fn messageGate(self: *const Model, sc: *Scope, g: *const MessageGateWeights, x: A, msg: A) !A {
        const dt = mlx.mlx_array_dtype(x);
        const eps = self.cfg.rms_norm_eps;
        const nx = try sc.rmsUnweighted(x, eps);
        const nm = try sc.rmsUnweighted(msg, eps);
        const cat = try sc.concat(&.{ nx, nm }, -1);
        const s_proj = try sc.astype(try g.down.apply(sc, cat), .float32);
        const fg = try sc.tanh(try sc.astype(try g.forget.apply(sc, s_proj), .float32));
        const mg = try sc.tanh(try sc.astype(try g.gain.apply(sc, s_proj), .float32));
        const one_minus_fg = try sc.sub(try sc.f(1.0), fg);
        const res = try sc.add(try sc.mul(one_minus_fg, try sc.astype(x, .float32)), try sc.mul(mg, try sc.astype(msg, .float32)));
        return sc.astype(res, dt);
    }

    fn attention(
        self: *const Model,
        sc: *Scope,
        lw: *const LayerWeights,
        vc: *VisitCache,
        hidden: A,
        q_in: A,
        k_in: A,
        v_in: A,
        seq_len: usize,
    ) !A {
        const Hq: c_int = @intCast(self.cfg.num_attention_heads);
        const Hkv: c_int = @intCast(self.cfg.num_key_value_heads);
        const D: c_int = @intCast(self.cfg.head_dim);
        const d_rel: c_int = @intCast(self.cfg.inkling_d_rel);
        const S_len: c_int = @intCast(seq_len);
        const K: usize = @intCast(self.cfg.inkling_sconv_kernel);

        var q = try lw.q_proj.apply(sc, q_in);
        var k = try lw.k_proj.apply(sc, k_in);
        var v = try lw.v_proj.apply(sc, v_in);
        var r = try lw.r_proj.apply(sc, q_in);
        const gate = try lw.gate_proj.apply(sc, hidden);
        const sinks = try lw.sink_proj.apply(sc, hidden);

        if (lw.is_global) {
            k = try self.dynamicShortConv(sc, k, k_in, lw.k_sconv_static, &lw.k_sconv_proj, &vc.conv_k, &vc.win_k, K);
            v = try self.dynamicShortConv(sc, v, v_in, lw.v_sconv_static, &lw.v_sconv_proj, &vc.conv_v, &vc.win_v, K);
        } else {
            k = try shortConv(sc, k, lw.k_sconv, &vc.conv_k, &vc.win_k, K);
            v = try shortConv(sc, v, lw.v_sconv, &vc.conv_v, &vc.win_v, K);
        }

        q = try sc.reshape(q, &.{ 1, S_len, Hq, D });
        k = try sc.reshape(k, &.{ 1, S_len, Hkv, D });
        v = try sc.reshape(v, &.{ 1, S_len, Hkv, D });
        r = try sc.reshape(r, &.{ 1, S_len, Hq, d_rel });

        q = if (self.compiled_l2_q.ctx != null) try applyCompiled2(self.compiled_l2_q, sc, q, lw.q_l2_scale) else try l2Clip(sc, q, lw.q_l2_scale, self.cfg.qk_l2_clip_threshold);
        k = if (self.compiled_l2_k.ctx != null) try applyCompiled1(self.compiled_l2_k, sc, k) else try l2Clip(sc, k, null, self.cfg.qk_l2_clip_threshold);
        const v_dt = mlx.mlx_array_dtype(v);
        v = try sc.astype(try sc.rmsUnweighted(v, self.cfg.rms_norm_eps), v_dt);
        r = if (self.compiled_l2_rel.ctx != null) try applyCompiled2(self.compiled_l2_rel, sc, r, lw.rel_l2_scale) else try l2Clip(sc, r, lw.rel_l2_scale, self.cfg.rel_l2_clip_threshold);

        if (lw.is_global and self.cfg.inkling_log_n_floor > 0) {
            const floor_f: f32 = @floatFromInt(self.cfg.inkling_log_n_floor);
            const alpha = self.cfg.inkling_log_alpha;
            if (self.offset + seq_len > self.cfg.inkling_log_n_floor) {
                const tau_buf = try sc.a.alloc(f32, seq_len);
                defer sc.a.free(tau_buf);
                for (0..seq_len) |ti| {
                    const pos_f: f32 = @floatFromInt(self.offset + ti + 1);
                    const diff = @max(@log(pos_f) - @log(floor_f), 0.0);
                    tau_buf[ti] = 1.0 + alpha * diff;
                }
                const tau = try sc.keep(mlx.mlx_array_new_data(tau_buf.ptr, &[_]c_int{ 1, S_len, 1, 1 }, 4, .float32));
                q = try sc.astype(try sc.mul(try sc.astype(q, .float32), tau), mlx.mlx_array_dtype(q));
                r = try sc.astype(try sc.mul(try sc.astype(r, .float32), tau), mlx.mlx_array_dtype(r));
            }
        }

        const pos_bias = try sc.matmul(r, try sc.astype(lw.rel_proj, mlx.mlx_array_dtype(r))); // [1, S, Hq, R]
        const R: usize = @intCast(mlx.getShape(pos_bias)[3]);

        const q_t = try sc.transpose(q, &.{ 0, 2, 1, 3 }); // [1, Hq, S, D]
        const k_new = try sc.transpose(k, &.{ 0, 2, 1, 3 }); // [1, Hkv, S, D]
        const v_new = try sc.transpose(v, &.{ 0, 2, 1, 3 }); // [1, Hkv, S, D]

        // Update KV cache (lazy, evaluated at end of forward)
        const k_all = if (vc.k.ctx != null) try sc.concat(&.{ vc.k, k_new }, 2) else k_new;
        const v_all = if (vc.v.ctx != null) try sc.concat(&.{ vc.v, v_new }, 2) else v_new;
        if (vc.k.ctx != null) _ = mlx.mlx_array_free(vc.k);
        if (vc.v.ctx != null) _ = mlx.mlx_array_free(vc.v);
        vc.k = sc.out(k_all);
        vc.v = sc.out(v_all);

        const total_kv: usize = @intCast(mlx.getShape(k_all)[2]);
        const win: usize = if (!lw.is_global) self.cfg.sliding_window else 0;
        const kv_start: usize = if (win > 0 and total_kv > win) total_kv - win else 0;
        const kv_len: usize = total_kv - kv_start;
        const k_view = if (kv_start > 0) try sc.sliceAxis(k_all, 2, kv_start, total_kv) else k_all;
        const v_view = if (kv_start > 0) try sc.sliceAxis(v_all, 2, kv_start, total_kv) else v_all;

        // Fast path for decode (seq_len == 1): GPU-only mask + fused SDPA with sinks
        if (seq_len == 1) {
            var k_idx = mlx.mlx_array_new();
            const k_idx_arr = try sc.res(mlx.mlx_arange(&k_idx, @floatFromInt(kv_start), @floatFromInt(total_kv), 1.0, .int32, sc.s), &k_idx);
            const q_pos_arr = try sc.keep(mlx.mlx_array_new_int(@intCast(self.offset)));
            const dist = try sc.sub(q_pos_arr, k_idx_arr); // [kv_len]
            const zero_i = try sc.keep(mlx.mlx_array_new_int(0));
            const max_r_i = try sc.keep(mlx.mlx_array_new_int(@intCast(R - 1)));
            const clipped = try sc.minimum(try sc.maximum(dist, zero_i), max_r_i); // [kv_len]

            const pb_2d = try sc.reshape(pos_bias, &.{ Hq, @intCast(R) });
            const bias_2d = try sc.take(pb_2d, clipped, -1); // [Hq, kv_len]
            const r_i = try sc.keep(mlx.mlx_array_new_int(@intCast(R)));
            var in_r_cmp = mlx.mlx_array_new();
            const in_r = try sc.res(mlx.mlx_less(&in_r_cmp, dist, r_i, sc.s), &in_r_cmp);
            const zero_bf = try sc.astype(try sc.f(0.0), mlx.mlx_array_dtype(pos_bias));
            const mask_2d = try sc.where(in_r, bias_2d, zero_bf);
            const mask_4d = try sc.reshape(mask_2d, &.{ 1, Hq, 1, @intCast(kv_len) });

            const sinks_1d = try sc.astype(try sc.reshape(sinks, &.{Hq}), mlx.mlx_array_dtype(q_t));
            const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
            var sdpa_o = mlx.mlx_array_new();
            const sdpa_arr = try sc.res(mlx.mlx_fast_scaled_dot_product_attention(
                &sdpa_o,
                q_t,
                k_view,
                v_view,
                attn_scale,
                "array",
                mask_4d,
                sinks_1d,
                false,
                sc.s,
            ), &sdpa_o);
            const ctx_h = try sc.transpose(sdpa_arr, &.{ 0, 2, 1, 3 });
            const ctx_flat = try sc.reshape(ctx_h, &.{ 1, S_len, Hq * D });
            const gated = try sc.mul(try sc.astype(ctx_flat, .float32), try sc.sigmoid(try sc.astype(gate, .float32)));
            return lw.o_proj.apply(sc, try sc.astype(gated, mlx.mlx_array_dtype(gate)));
        }

        // Prefill path (seq_len > 1): build relative bias + causal/sliding mask [1, Hq, S, kv_len]
        const dist_n = seq_len * kv_len;
        const clip_buf = try sc.a.alloc(i32, dist_n);
        defer sc.a.free(clip_buf);
        const vis_buf = try sc.a.alloc(f32, dist_n);
        defer sc.a.free(vis_buf);
        const in_r_buf = try sc.a.alloc(f32, dist_n);
        defer sc.a.free(in_r_buf);

        for (0..seq_len) |qi| {
            const q_pos: isize = @intCast(self.offset + qi);
            for (0..kv_len) |ki| {
                const k_pos: isize = @intCast(kv_start + ki);
                const d = q_pos - k_pos;
                const idx = qi * kv_len + ki;
                const vis = d >= 0 and (win == 0 or d < @as(isize, @intCast(win)));
                vis_buf[idx] = if (vis) 1.0 else 0.0;
                in_r_buf[idx] = if (d >= 0 and d < @as(isize, @intCast(R))) 1.0 else 0.0;
                const clamped: isize = if (d < 0) 0 else if (d >= @as(isize, @intCast(R))) @as(isize, @intCast(R - 1)) else d;
                clip_buf[idx] = @intCast(clamped);
            }
        }
        const clip_arr = try sc.keep(mlx.mlx_array_new_data(clip_buf.ptr, &[_]c_int{ 1, 1, S_len, @intCast(kv_len) }, 4, .int32));
        const vis_arr = try sc.keep(mlx.mlx_array_new_data(vis_buf.ptr, &[_]c_int{ 1, 1, S_len, @intCast(kv_len) }, 4, .float32));
        const in_r_arr = try sc.keep(mlx.mlx_array_new_data(in_r_buf.ptr, &[_]c_int{ 1, 1, S_len, @intCast(kv_len) }, 4, .float32));

        const pb_t = try sc.transpose(pos_bias, &.{ 0, 2, 1, 3 }); // [1, Hq, S, R]
        const gathered_bias = try sc.takeAlong(
            try sc.astype(pb_t, .float32),
            try sc.broadcast(clip_arr, &.{ 1, Hq, S_len, @intCast(kv_len) }),
            -1,
        );
        const masked_bias = try sc.mul(gathered_bias, in_r_arr);
        const neg_inf = try sc.f(-1e9);
        const one_minus_vis = try sc.sub(try sc.f(1.0), vis_arr);
        const attn_mask = try sc.add(try sc.mul(masked_bias, vis_arr), try sc.mul(one_minus_vis, neg_inf));

        const gqa_grp: c_int = @divExact(Hq, Hkv);
        const q_gqa = try sc.reshape(try sc.astype(q_t, .float32), &.{ 1, Hkv, gqa_grp, S_len, D });
        const k_gqa = try sc.reshape(try sc.astype(k_view, .float32), &.{ 1, Hkv, 1, @intCast(kv_len), D });
        const v_gqa = try sc.reshape(v_view, &.{ 1, Hkv, 1, @intCast(kv_len), D });
        const mask_gqa = try sc.reshape(attn_mask, &.{ 1, Hkv, gqa_grp, S_len, @intCast(kv_len) });

        const attn_scale = try sc.f(1.0 / @sqrt(@as(f32, @floatFromInt(D))));
        const qk = try sc.add(try sc.mul(try sc.matmul(q_gqa, try sc.transpose(k_gqa, &.{ 0, 1, 2, 4, 3 })), attn_scale), mask_gqa);
        const sinks_gqa = try sc.reshape(try sc.transpose(try sc.astype(sinks, .float32), &.{ 0, 2, 1 }), &.{ 1, Hkv, gqa_grp, S_len, 1 });
        const combined = try sc.concat(&.{ qk, sinks_gqa }, -1);
        const probs_full = try sc.softmax(combined, -1);
        const probs = try sc.sliceAxis(probs_full, -1, 0, kv_len);
        const ctx_v = try sc.matmul(try sc.astype(probs, mlx.mlx_array_dtype(v_gqa)), v_gqa); // [1, Hkv, gqa_grp, S, D]
        const ctx_h = try sc.transpose(try sc.reshape(ctx_v, &.{ 1, Hq, S_len, D }), &.{ 0, 2, 1, 3 }); // [1, S, Hq, D]
        const ctx_flat = try sc.reshape(ctx_h, &.{ 1, S_len, Hq * D });

        const gated = try sc.mul(try sc.astype(ctx_flat, .float32), try sc.sigmoid(try sc.astype(gate, .float32)));
        return lw.o_proj.apply(sc, try sc.astype(gated, mlx.mlx_array_dtype(gate)));
    }

    fn moe(self: *const Model, sc: *Scope, lw: *const LayerWeights, x: A, seq_len: usize) !A {
        const dt = mlx.mlx_array_dtype(x);
        const T_len: c_int = @intCast(seq_len);
        const D_len: c_int = @intCast(self.cfg.hidden_size);
        const top_k: c_int = @intCast(self.cfg.num_experts_per_tok);
        const x_flat = try sc.reshape(x, &.{ T_len, D_len });

        const logits = try sc.matmul(try sc.astype(x_flat, .float32), try sc.transpose(lw.router_w, &.{ 1, 0 }));
        const scores = try sc.sigmoid(logits);
        const biased = try sc.add(scores, try sc.astype(lw.router_bias, .float32));

        // Top-k via argpartition(-biased, k-1)
        var neg_biased = mlx.mlx_array_new();
        const neg_b = try sc.res(mlx.mlx_negative(&neg_biased, biased, sc.s), &neg_biased);
        var part = mlx.mlx_array_new();
        const part_arr = try sc.res(mlx.mlx_argpartition_axis(&part, neg_b, top_k - 1, -1, sc.s), &part);
        const indices = try sc.sliceAxis(part_arr, -1, 0, @intCast(top_k));

        const weights_k = try sc.takeAlong(scores, indices, -1);
        const denom = try sc.add(try sc.sum(weights_k, -1, true), try sc.f(1e-20));
        const rw = try sc.mul(try sc.div(weights_k, denom), try sc.f(self.cfg.router_scaling_factor));

        // Experts: x_exp [T, 1, 1, D]
        const x_exp = try sc.reshape(x_flat, &.{ T_len, 1, 1, D_len });
        const gate_out = try lw.sw_gate.gatherMm(sc, x_exp, indices);
        const up_out = try lw.sw_up.gatherMm(sc, x_exp, indices);

        const act = if (self.compiled_softcap.ctx != null)
            try applyCompiled2(self.compiled_softcap, sc, gate_out, up_out)
        else blk: {
            const g32 = try sc.astype(gate_out, .float32);
            const u32_a = try sc.astype(up_out, .float32);
            const gs_f = try sc.f(self.cfg.mlp_gate_softcap);
            const us_f = try sc.f(self.cfg.mlp_up_softcap);
            const g_cap = try sc.mul(try sc.tanh(try sc.div(g32, gs_f)), gs_f);
            const u_cap = try sc.mul(try sc.tanh(try sc.div(u32_a, us_f)), us_f);
            break :blk try sc.astype(try sc.mul(try sc.mul(g_cap, try sc.sigmoid(g32)), u_cap), dt);
        };

        const down_out = try sc.squeeze(try lw.sw_down.gatherMm(sc, act, indices), -2); // [T, k, D]
        const weighted = try sc.mul(try sc.astype(down_out, .float32), try sc.expand(rw, -1));
        const summed = try sc.sum(weighted, -2, false); // [T, D]
        return sc.reshape(try sc.astype(summed, dt), &.{ 1, T_len, D_len });
    }

    fn runLayer(
        self: *const Model,
        sc: *Scope,
        phys_idx: usize,
        visit_idx: usize,
        h_in: A,
        block_states: *std.ArrayList(A),
        messages: ?A,
        seq_len: usize,
    ) !A {
        const lw = &self.layers[phys_idx];
        const vc = &self.caches[visit_idx];
        const dt = mlx.mlx_array_dtype(h_in);
        const eps = self.cfg.rms_norm_eps;
        const K: usize = @intCast(self.cfg.inkling_sconv_kernel);

        const normed_res = try sc.rmsUnweighted(h_in, eps);
        var q_in = h_in;
        var k_in = h_in;
        var v_in = h_in;
        if (phys_idx > 0) {
            const attn_inputs = try self.depthAttention(sc, block_states.items, normed_res, lw.attn_res_attn_q); // [S, Q, D]
            if (lw.is_global) {
                q_in = try sc.astype(try sc.reshape(try sc.sliceAxis(attn_inputs, 1, 0, 1), &.{ 1, @intCast(seq_len), -1 }), dt);
                k_in = try sc.astype(try sc.reshape(try sc.sliceAxis(attn_inputs, 1, 1, 2), &.{ 1, @intCast(seq_len), -1 }), dt);
                v_in = try sc.astype(try sc.reshape(try sc.sliceAxis(attn_inputs, 1, 2, 3), &.{ 1, @intCast(seq_len), -1 }), dt);
            } else {
                const s0 = try sc.astype(try sc.reshape(try sc.sliceAxis(attn_inputs, 1, 0, 1), &.{ 1, @intCast(seq_len), -1 }), dt);
                q_in = s0;
                k_in = s0;
                v_in = s0;
            }
        }

        if (messages) |msg| {
            const m0 = try sc.astype(try sc.reshape(try sc.sliceAxis(msg, 1, 0, 1), &.{ 1, @intCast(seq_len), -1 }), dt);
            const m1 = try sc.astype(try sc.reshape(try sc.sliceAxis(msg, 1, 1, 2), &.{ 1, @intCast(seq_len), -1 }), dt);
            const m2 = try sc.astype(try sc.reshape(try sc.sliceAxis(msg, 1, 2, 3), &.{ 1, @intCast(seq_len), -1 }), dt);
            q_in = try self.messageGate(sc, &lw.ctrl_q, q_in, m0);
            k_in = try self.messageGate(sc, &lw.ctrl_k, k_in, m1);
            v_in = try self.messageGate(sc, &lw.ctrl_v, v_in, m2);
        }

        if (lw.is_global) {
            q_in = try sc.rms(q_in, lw.q_input_norm, eps);
            k_in = try sc.rms(k_in, lw.k_input_norm, eps);
            v_in = try sc.rms(v_in, lw.v_input_norm, eps);
        } else {
            q_in = try sc.rms(q_in, lw.input_norm, eps);
            k_in = try sc.rms(k_in, lw.input_norm, eps);
            v_in = try sc.rms(v_in, lw.input_norm, eps);
        }

        var attn_out = try self.attention(sc, lw, vc, q_in, q_in, k_in, v_in, seq_len);
        attn_out = try shortConv(sc, attn_out, lw.attn_sconv, &vc.conv_attn, &vc.win_attn, K);

        var residual = try sc.astype(attn_out, .float32);
        if (phys_idx % self.cfg.n_layers_per_attn_res_block == 0) {
            try block_states.append(sc.a, normed_res);
        } else {
            residual = try sc.add(try sc.astype(h_in, .float32), residual);
        }

        const normed_states = try sc.rmsUnweighted(residual, eps);
        const mlp_in_raw = try self.depthAttention(sc, block_states.items, normed_states, lw.attn_res_mlp_q); // [S, 1, D]
        var mlp_in = try sc.astype(try sc.reshape(mlp_in_raw, &.{ 1, @intCast(seq_len), -1 }), dt);
        if (messages) |msg| {
            const m3 = try sc.astype(try sc.reshape(try sc.sliceAxis(msg, 1, 3, 4), &.{ 1, @intCast(seq_len), -1 }), dt);
            mlp_in = try self.messageGate(sc, &lw.ctrl_mlp, mlp_in, m3);
        }

        const ff_normed = try sc.rms(mlp_in, lw.post_attn_norm, eps);
        var mlp_out = try self.moe(sc, lw, ff_normed, seq_len);
        mlp_out = try shortConv(sc, mlp_out, lw.mlp_sconv, &vc.conv_mlp, &vc.win_mlp, K);

        const h_out = try sc.add(residual, try sc.astype(mlp_out, .float32));
        return sc.astype(h_out, dt);
    }

    pub const ForwardOpts = struct {
        /// Logits for EVERY row (a spec verify), not just the last one.
        all_rows: bool = false,
        /// Record the DSpark target-feature taps of this chunk.
        capture: bool = false,
    };

    fn tapSlot(self: *const Model, visit: usize) ?usize {
        if (self.cfg.dspark_ntaps == 0) return null;
        for (self.cfg.dspark_taps[0..self.cfg.dspark_ntaps], 0..) |t, i| {
            if (t == visit) return i;
        }
        return null;
    }

    pub fn forward(self: *Model, ids: []const u32) !A {
        return self.forwardWith(ids, .{});
    }

    pub fn forwardWith(self: *Model, ids: []const u32, opts: ForwardOpts) !A {
        if (!self.ops_compiled) self.compileOps();
        var sc = Scope.init(self.a, self.s);
        defer sc.deinit();

        const seq_len = ids.len;
        const eps = self.cfg.rms_norm_eps;
        var h = try self.applyEmbed(&sc, ids);
        h = try sc.rms(h, self.embed_norm, eps);
        const dt = mlx.mlx_array_dtype(h);

        self.fwd_pos0 = self.offset;
        self.pending_n = seq_len;
        var taps: [8]A = @splat(none);
        // A draft needs the taps of EVERY forward (the prefill included).
        const capture = self.dspark != null;

        var block_states: std.ArrayList(A) = .empty;
        defer block_states.deinit(self.a);

        var visit_idx: usize = 0;
        const core_start: usize = @intCast(self.cfg.loop_core_start);
        const core_end: usize = @intCast(self.cfg.loop_core_end);
        const phys_n: usize = @intCast(self.cfg.physical_num_layers);

        // Prelude (0..core_start)
        for (0..core_start) |pi| {
            if (capture) if (self.tapSlot(visit_idx)) |ti| {
                taps[ti] = h;
            };
            h = try self.runLayer(&sc, pi, visit_idx, h, &block_states, null, seq_len);
            visit_idx += 1;
        }

        const prelude_h = h;
        const prelude_blocks_len = block_states.items.len;

        // Looped core (core_start..core_end) x num_passes
        for (0..self.cfg.loop_num_passes) |pass_idx| {
            var bank_list: std.ArrayList(A) = .empty;
            defer bank_list.deinit(self.a);
            var has_bank = false;
            var bank_partial: A = none;
            if (pass_idx > 0) {
                has_bank = true;
                for (block_states.items[prelude_blocks_len + 1 ..]) |bs| {
                    try bank_list.append(self.a, bs);
                }
                bank_partial = try sc.rmsUnweighted(h, eps);
                // Reset to prelude state
                h = prelude_h;
                block_states.items.len = prelude_blocks_len;
            }

            for (core_start..core_end, 0..) |pi, loop_i| {
                if (capture) if (self.tapSlot(visit_idx)) |ti| {
                    taps[ti] = h;
                };
                var msg: ?A = null;
                if (has_bank) {
                    const mq_i = try sc.squeeze(try sc.sliceAxis(self.message_query, 0, loop_i, loop_i + 1), 0); // [4, D]
                    msg = try self.depthAttention(&sc, bank_list.items, bank_partial, mq_i); // [S, 4, D]
                }
                h = try self.runLayer(&sc, pi, visit_idx, h, &block_states, msg, seq_len);
                visit_idx += 1;
            }
        }

        // Epilogue (core_end..phys_n)
        for (core_end..phys_n) |pi| {
            if (capture) if (self.tapSlot(visit_idx)) |ti| {
                taps[ti] = h;
            };
            h = try self.runLayer(&sc, pi, visit_idx, h, &block_states, null, seq_len);
            visit_idx += 1;
        }

        // Final output_query + norm + unembed
        const final_normed = try sc.rmsUnweighted(h, eps);
        const final_attn = try self.depthAttention(&sc, block_states.items, final_normed, self.output_query); // [S, 1, D]
        const h_final = try sc.astype(try sc.reshape(final_attn, &.{ 1, @intCast(seq_len), -1 }), dt);
        const h_normed = try sc.rms(h_final, self.norm, eps);

        // The last tap is the final norm's own output (`target_layer_ids`
        // ends with the trunk's layer count).
        if (capture) if (self.tapSlot(visit_idx)) |ti| {
            taps[ti] = h_normed;
        };

        // Slice last token only to save unembed work (a verify wants them all).
        const head_in = if (opts.all_rows) h_normed else try sc.sliceAxis(h_normed, 1, seq_len - 1, seq_len);
        var logits = try self.unembed.apply(&sc, head_in);
        if (self.cfg.unpadded_vocab_size > 0 and self.cfg.unpadded_vocab_size < self.cfg.vocab_size) {
            logits = try sc.sliceAxis(logits, -1, 0, @intCast(self.cfg.unpadded_vocab_size));
        }
        if (self.cfg.final_logit_softcapping > 0) {
            const cap_f = try sc.f(self.cfg.final_logit_softcapping);
            logits = try sc.mul(try sc.tanh(try sc.div(try sc.astype(logits, .float32), cap_f)), cap_f);
        }

        const vocab_dim: c_int = @intCast(if (self.cfg.unpadded_vocab_size > 0) self.cfg.unpadded_vocab_size else self.cfg.vocab_size);
        const out = try sc.reshape(logits, if (opts.all_rows) &.{ 1, @intCast(seq_len), vocab_dim } else &.{ 1, 1, vocab_dim });

        if (capture) {
            var parts: [8]A = @splat(none);
            var np: usize = 0;
            for (0..self.cfg.dspark_ntaps) |ti| {
                if (taps[ti].ctx == null) break;
                parts[np] = taps[ti];
                np += 1;
            }
            if (np == self.cfg.dspark_ntaps and np > 0) {
                const feat = try sc.concat(parts[0..np], -1);
                if (self.cap.ctx != null) _ = mlx.mlx_array_free(self.cap);
                self.cap = sc.out(feat);
                self.cap_pos = self.fwd_pos0;
                self.cap_n = seq_len;
                // Default commit row: the chunk's last. A verify round overrides
                // this after its partial-accept decision (dsparkRound).
                self.cap_commit_idx = seq_len - 1;
                // Commit this chunk's context K/V NOW: prefill runs in chunks,
                // and a deferred capture would leave a gap the draft cannot fill.
                try self.dsparkCtxAppend(&sc, self.fwd_pos0, seq_len);
                self.ctx_len = self.fwd_pos0 + seq_len;
            }
        }

        self.offset += seq_len;
        const t_eval0 = std.Io.Clock.awake.now(std.Io.Threaded.global_single_threaded.io()).nanoseconds;
        try mlx.check(mlx.mlx_array_eval(out));
        // Batch-eval all updated KV and conv states in one sync so they don't accumulate a multi-step lazy graph.
        var state_evals: [288]A = undefined;
        var se_n: usize = 0;
        for (self.caches) |vc| {
            if (vc.k.ctx != null) { state_evals[se_n] = vc.k; se_n += 1; }
            if (vc.v.ctx != null) { state_evals[se_n] = vc.v; se_n += 1; }
            if (vc.conv_k.ctx != null) { state_evals[se_n] = vc.conv_k; se_n += 1; }
            if (vc.conv_v.ctx != null) { state_evals[se_n] = vc.conv_v; se_n += 1; }
            if (vc.conv_attn.ctx != null) { state_evals[se_n] = vc.conv_attn; se_n += 1; }
            if (vc.conv_mlp.ctx != null) { state_evals[se_n] = vc.conv_mlp; se_n += 1; }
        }
        if (se_n > 0) {
            const vec = mlx.mlx_vector_array_new_data(&state_evals, se_n);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_eval(vec));
        }
        const t_eval1 = std.Io.Clock.awake.now(std.Io.Threaded.global_single_threaded.io()).nanoseconds;
        if (seq_len == 1 and self.offset == 30) {
            log.info("inkling2 decode step30 eval={d:.2}ms\n", .{@as(f64, @floatFromInt(t_eval1 - t_eval0)) / 1e6});
        }
        return sc.out(out);
    }
    // ── DSpark draft ──────────────────────────────────────────────────────

    /// Drafts per block (0 = no draft loaded).
    pub fn dsparkDrafts(self: *const Model) usize {
        if (self.dspark) |dp| return dp.block - 1;
        return 0;
    }

    fn dsparkAttn(self: *const Model, sc: *Scope, dp: *const Dspark, lw: *const DsparkLayer, li: usize, x_in: A) !A {
        const B = dp.block;
        const Hq: c_int = @intCast(self.cfg.num_attention_heads);
        const Hkv: c_int = @intCast(self.cfg.num_key_value_heads);
        const D: c_int = @intCast(self.cfg.head_dim);
        const eps = self.cfg.rms_norm_eps;
        const Bc: c_int = @intCast(B);
        const pos0 = self.offset;

        var q = try lw.q_proj.apply(sc, x_in);
        var k = try lw.k_proj.apply(sc, x_in);
        var v = try lw.v_proj.apply(sc, x_in);
        q = try sc.reshape(q, &.{ 1, Bc, Hq, D });
        k = try sc.reshape(k, &.{ 1, Bc, Hkv, D });
        v = try sc.reshape(v, &.{ 1, Bc, Hkv, D });
        q = try sc.rms(q, lw.q_norm, eps);
        k = try sc.rms(k, lw.k_norm, eps);
        q = try ropeApply(sc, q, pos0, B, @intCast(D), dp.rope_theta);
        k = try ropeApply(sc, k, pos0, B, @intCast(D), dp.rope_theta);

        const q_t = try sc.transpose(q, &.{ 0, 2, 1, 3 }); // [1, Hq, B, D]
        const k_t = try sc.transpose(k, &.{ 0, 2, 1, 3 }); // [1, Hkv, B, D]
        const v_t = try sc.transpose(v, &.{ 0, 2, 1, 3 });

        const cv = dp.ctx[li];
        var keys = k_t;
        var vals = v_t;
        if (cv.len > 0) {
            var ck = cv.k;
            var cvv = cv.v;
            if (lw.is_local and dp.window > 0 and cv.len > dp.window) {
                ck = try sc.sliceAxis(cv.k, 2, cv.len - dp.window, cv.len);
                cvv = try sc.sliceAxis(cv.v, 2, cv.len - dp.window, cv.len);
            }
            keys = try sc.concat(&.{ ck, k_t }, 2);
            vals = try sc.concat(&.{ cvv, v_t }, 2);
        }
        var sdpa_o = mlx.mlx_array_new();
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
        const o = try sc.res(mlx.mlx_fast_scaled_dot_product_attention(
            &sdpa_o,
            q_t,
            keys,
            vals,
            scale,
            "", // no mask: the draft block is bidirectional
            none,
            none,
            false,
            sc.s,
        ), &sdpa_o);
        const ctx_h = try sc.transpose(o, &.{ 0, 2, 1, 3 }); // [1, B, Hq, D]
        const flat = try sc.reshape(ctx_h, &.{ 1, Bc, Hq * D });
        return lw.o_proj.apply(sc, flat);
    }

    /// The anchor's normalized embedding plus the block's mask rows, through
    /// the draft layers, then the trunk head. With the checkpoint's Markov head
    /// (`rnn_hidden`) the block's rows are still computed in ONE parallel pass;
    /// only the logits go sequential: row `k`'s bias reads the token at
    /// position `p+k-1` (the anchor for the first draft row), exactly SGLang's
    /// `Inkling2MTPHead.step_bias` loop.
    fn dsparkDraft(self: *const Model, allocator: std.mem.Allocator, t1: u32) ![]u32 {
        const dp = self.dspark orelse return error.NoDspark;
        const B = dp.block;
        const D: c_int = @intCast(self.cfg.hidden_size);
        const eps = self.cfg.rms_norm_eps;
        const nd = B - 1;
        var sc = Scope.init(self.a, self.s);
        defer sc.deinit();

        const anchor = try sc.rms(try self.applyEmbed(&sc, &.{t1}), self.embed_norm, eps);
        const mask = try sc.broadcast(dp.mask_embed, &.{ 1, @intCast(nd), D });
        var res = try sc.concat(&.{ anchor, mask }, 1);

        for (dp.layers, 0..) |*lw, li| {
            const xn = try sc.rms(res, lw.input_norm, eps);
            res = try sc.add(res, try self.dsparkAttn(&sc, dp, lw, li, xn));
            const mn = try sc.rms(res, lw.post_attn_norm, eps);
            res = try sc.add(res, try draftMlp(&sc, lw, mn));
        }
        const h = try sc.rms(res, dp.norm, eps);
        const hidden_rows = try sc.sliceAxis(h, 1, 1, B);
        var logits = try self.unembed.apply(&sc, hidden_rows);
        if (self.cfg.unpadded_vocab_size > 0 and self.cfg.unpadded_vocab_size < self.cfg.vocab_size) {
            logits = try sc.sliceAxis(logits, -1, 0, @intCast(self.cfg.unpadded_vocab_size));
        }

        const out = try allocator.alloc(u32, nd);
        errdefer allocator.free(out);

        if (dp.markov) |*md| {
            // `f_rnn`: the fused target feature of the last committed position,
            // fixed for the whole block (SGLang `compute_f_rnn`).
            var f_rnn = try sc.zeros(&.{ 1, D }, .float32);
            if (self.cap.ctx != null and self.cap_n > 0 and self.cfg.dspark_ntaps > 0) {
                const row = @min(self.cap_commit_idx, self.cap_n - 1);
                const feats = try sc.sliceAxis(self.cap, 1, row, row + 1);
                // `mixFeats` keeps the row axis for the context-K/V arm; the head
                // wants one [1, hidden] row beside the state.
                const mixed = try mixFeats(&sc, dp, feats, dp.layers.len, md.dflare_norm, 1, eps);
                f_rnn = try sc.reshape(mixed, &.{ 1, D });
            }
            const R: usize = md.rank;
            const Rc: c_int = @intCast(R);
            var state = try sc.zeros(&.{ 1, Rc }, .float32);
            var prev: u32 = t1;
            for (1..B) |k| {
                const hk = try sc.reshape(try sc.sliceAxis(hidden_rows, 1, k - 1, k), &.{ 1, D });
                const prev_emb = try sc.take(md.w1, try sc.ints(&.{@intCast(prev)}), 0);
                const z = try sc.concat(&.{
                    state,
                    try sc.astype(prev_emb, .float32),
                    try sc.astype(hk, .float32),
                    f_rnn,
                }, 1);
                const proj = try sc.add(try sc.matmul(z, md.rnn_w_t), try sc.astype(md.rnn_b, .float32));
                const gate_raw = try sc.sliceAxis(proj, 1, 0, R);
                const cand_raw = try sc.sliceAxis(proj, 1, R, 2 * R);
                const out_raw = try sc.sliceAxis(proj, 1, 2 * R, 3 * R);
                // state = sigmoid(gate)*state + sigmoid(-gate)*tanh(candidate)
                state = try sc.add(
                    try sc.mul(try sc.sigmoid(gate_raw), state),
                    try sc.mul(try sc.sigmoid(try sc.neg(gate_raw)), try sc.tanh(cand_raw)),
                );
                const latent = try sc.tanh(out_raw);
                const corr = try sc.add(try sc.matmul(latent, md.out_w_t), try sc.astype(md.out_b, .float32));
                var pcorr = try self.unembed.apply(&sc, try sc.reshape(corr, &.{ 1, 1, D }));
                if (self.cfg.unpadded_vocab_size > 0 and self.cfg.unpadded_vocab_size < self.cfg.vocab_size) {
                    pcorr = try sc.sliceAxis(pcorr, -1, 0, @intCast(self.cfg.unpadded_vocab_size));
                }
                var row_logits = try sc.add(try sc.sliceAxis(logits, 1, k - 1, k), pcorr);
                if (self.cfg.final_logit_softcapping > 0) {
                    const cap_f = try sc.f(self.cfg.final_logit_softcapping);
                    row_logits = try sc.mul(try sc.tanh(try sc.div(try sc.astype(row_logits, .float32), cap_f)), cap_f);
                }
                const am = try sc.argmax(row_logits, -1);
                try mlx.check(mlx.mlx_array_eval(am));
                const p = mlx.mlx_array_data_int32(am) orelse return error.NoData;
                out[k - 1] = @intCast(p[0]);
                prev = out[k - 1];
            }
            return out;
        }

        if (self.cfg.final_logit_softcapping > 0) {
            const cap_f = try sc.f(self.cfg.final_logit_softcapping);
            logits = try sc.mul(try sc.tanh(try sc.div(try sc.astype(logits, .float32), cap_f)), cap_f);
        }
        const am = try sc.argmax(logits, -1);
        try mlx.check(mlx.mlx_array_eval(am));
        const p = mlx.mlx_array_data_int32(am) orelse return error.NoData;
        for (0..nd) |i| out[i] = @intCast(p[i]);
        return out;
    }

    fn mixFeats(sc: *Scope, dp: *const Dspark, feats: A, li: usize, norm_w: A, rows: usize, eps: f32) !A {
        const taps_n: c_int = @intCast(mlx.getShape(feats)[2]);
        const hidden: c_int = @intCast(mlx.getShape(norm_w)[0]);
        const nt = @divExact(taps_n, hidden);
        const f4 = try sc.reshape(feats, &.{ 1, @intCast(rows), nt, hidden });
        const alpha = try sc.sliceAxis(dp.fuse_alpha, 0, li, li + 1); // [1, taps]
        const a4 = try sc.reshape(try sc.astype(alpha, .float32), &.{ 1, 1, nt, 1 });
        const wsum = try sc.sum(try sc.mul(try sc.astype(f4, .float32), a4), 2, false);
        return sc.rms(wsum, norm_w, eps);
    }

    /// Projects `rows` captured rows into every draft layer's context K/V,
    /// starting at absolute position `pos0`.
    fn dsparkCtxAppend(self: *Model, sc: *Scope, pos0: usize, rows: usize) !void {
        const dp = self.dspark orelse return;
        if (rows == 0 or self.cap.ctx == null) return;
        const feats = try sc.sliceAxis(self.cap, 1, 0, rows);
        const Hkv: c_int = @intCast(self.cfg.num_key_value_heads);
        const D: c_int = @intCast(self.cfg.head_dim);
        const eps = self.cfg.rms_norm_eps;
        for (dp.layers, 0..) |*lw, li| {
            const mixed = try mixFeats(sc, dp, feats, li, lw.dflare_norm, rows, eps);
            var k = try lw.k_proj_ctx.apply(sc, mixed);
            const v = try lw.v_proj_ctx.apply(sc, mixed);
            k = try sc.rms(try sc.reshape(k, &.{ 1, @intCast(rows), Hkv, D }), lw.k_norm_ctx, eps);
            k = try ropeApply(sc, k, pos0, rows, @intCast(D), dp.rope_theta);
            const k_t = try sc.transpose(k, &.{ 0, 2, 1, 3 });
            const v_r = try sc.reshape(v, &.{ 1, @intCast(rows), Hkv, D });
            const v_t = try sc.transpose(v_r, &.{ 0, 2, 1, 3 });
            const cv = &dp.ctx[li];
            var nk = k_t;
            var nv = v_t;
            if (cv.len > 0) {
                nk = try sc.concat(&.{ cv.k, k_t }, 2);
                nv = try sc.concat(&.{ cv.v, v_t }, 2);
            }
            if (cv.k.ctx != null) _ = mlx.mlx_array_free(cv.k);
            if (cv.v.ctx != null) _ = mlx.mlx_array_free(cv.v);
            cv.k = sc.out(nk);
            cv.v = sc.out(nv);
            cv.len += rows;
        }
        // `dsparkCtxAppend` runs outside `forwardWith`'s eval; force the new rows.
        var ev: [64]A = undefined;
        var n: usize = 0;
        for (dp.ctx) |cv| {
            if (n + 2 > ev.len) break;
            if (cv.k.ctx != null) { ev[n] = cv.k; n += 1; }
            if (cv.v.ctx != null) { ev[n] = cv.v; n += 1; }
        }
        if (n > 0) {
            const vec = mlx.mlx_vector_array_new_data(&ev, n);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_eval(vec));
        }
    }

    /// Rolls the trunk state back to `keep` absolute rows after a verify. The
    /// K/V caches slice, the conv states are re-read from the window the
    /// forward kept.
    fn truncateState(self: *Model, keep: usize, sc: *Scope) !void {
        const K: usize = @intCast(self.cfg.inkling_sconv_kernel);
        const full = self.fwd_pos0 + self.pending_n;
        const kc = keep - self.fwd_pos0;
        var ev: [320]A = undefined;
        var n: usize = 0;
        for (self.caches) |*vc| {
            if (vc.k.ctx != null) {
                const cur: usize = @intCast(mlx.getShape(vc.k)[2]);
                if (keep < cur) {
                    const nk = sc.out(try sc.sliceAxis(vc.k, 2, 0, keep));
                    const nv = sc.out(try sc.sliceAxis(vc.v, 2, 0, keep));
                    _ = mlx.mlx_array_free(vc.k);
                    _ = mlx.mlx_array_free(vc.v);
                    vc.k = nk;
                    vc.v = nv;
                }
            }
            inline for (.{ .{ &vc.win_k, &vc.conv_k }, .{ &vc.win_v, &vc.conv_v }, .{ &vc.win_attn, &vc.conv_attn }, .{ &vc.win_mlp, &vc.conv_mlp } }) |pair| {
                if (pair[0].*.ctx != null) {
                    if (keep < full) {
                        const st = try sc.sliceAxis(pair[0].*, 1, kc, kc + K - 1);
                        if (pair[1].*.ctx != null) _ = mlx.mlx_array_free(pair[1].*);
                        pair[1].* = sc.out(st);
                    }
                    _ = mlx.mlx_array_free(pair[0].*);
                    pair[0].* = none;
                }
            }
            _ = &vc;
            inline for (.{ &vc.k, &vc.v, &vc.conv_k, &vc.conv_v, &vc.conv_attn, &vc.conv_mlp }) |ap| {
                if (ap.*.ctx != null and n < ev.len) {
                    ev[n] = ap.*;
                    n += 1;
                }
            }
        }
        if (self.dspark) |dp| {
            for (dp.ctx) |*cv| {
                if (cv.len > keep and cv.k.ctx != null) {
                    const nk = sc.out(try sc.sliceAxis(cv.k, 2, 0, keep));
                    const nv = sc.out(try sc.sliceAxis(cv.v, 2, 0, keep));
                    _ = mlx.mlx_array_free(cv.k);
                    _ = mlx.mlx_array_free(cv.v);
                    cv.k = nk;
                    cv.v = nv;
                    cv.len = keep;
                    if (n < ev.len) {
                        ev[n] = cv.k;
                        n += 1;
                    }
                }
            }
        }
        if (n > 0) {
            const vec = mlx.mlx_vector_array_new_data(&ev, n);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_eval(vec));
        }
        self.offset = keep;
    }

    /// One draft + verify round: `t1` plus up to `cap_drafts` drafted tokens.
    pub fn dsparkRound(self: *Model, allocator: std.mem.Allocator, t1: u32, cap_drafts: usize) !DsparkRound {
        const dp = self.dspark orelse return error.NoDspark;
        if (self.ctx_len != self.offset) return error.NoDspark;
        const ndraft = @min(dp.block - 1, cap_drafts);
        if (ndraft == 0) return error.NoDspark;
        const drafts = try self.dsparkDraft(allocator, t1);
        defer allocator.free(drafts);

        const ids = try allocator.alloc(u32, ndraft + 1);
        defer allocator.free(ids);
        ids[0] = t1;
        @memcpy(ids[1..], drafts[0..ndraft]);

        const pos0 = self.offset;
        const logits = try self.forwardWith(ids, .{ .all_rows = true, .capture = true });
        defer _ = mlx.mlx_array_free(logits);

        var sc = Scope.init(self.a, self.s);
        defer sc.deinit();
        const am = try sc.argmax(logits, -1); // [1, n]
        try mlx.check(mlx.mlx_array_eval(am));
        const p = mlx.mlx_array_data_int32(am) orelse return error.NoData;

        var accepted: usize = 0;
        while (accepted < ndraft and @as(u32, @intCast(p[accepted])) == drafts[accepted]) accepted += 1;
        const next_token: u32 = @intCast(p[accepted]);

        const keep = pos0 + accepted + 1;
        try self.truncateState(keep, &sc);
        self.ctx_len = keep;
        // The Markov head's fused feature comes from the LAST committed row of
        // this verify chunk (SGLang `store_verify_commit`).
        self.cap_commit_idx = accepted;

        const toks = try allocator.alloc(u32, accepted + 1);
        toks[0] = t1;
        @memcpy(toks[1..], drafts[0..accepted]);
        return .{ .tokens = toks, .accepted = @intCast(accepted), .next_token = next_token, .position = keep };
    }
};
