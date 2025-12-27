# i need to implement an SGLang llm inference simulator like [https://github.com/llm-d/llm-d-inference-sim](https://github.com/llm-d/llm-d-inference-sim) and [https://github.com/microsoft/vidur](https://github.com/microsoft/vidur) in python and with strong ability to express any metric, most importantly a throughput rate, as a mathematical function, preferably in sympy (it should account for branching, loops, choice of kernels on device and all the pecularities) as later i would use that function in optimization procedure/solver.

A good way to get an “SGLang-like” simulator *and* a solver-friendly throughput function is to (1) build a discrete-event inference simulator (Vidur-style) and (2) make every latency/metric computation produce a **SymPy expression** alongside numeric evaluation, using `Piecewise/Min/Max/Sum` to capture branches, kernel choices, and loops.[^1_1]

## Core architecture

- Use an **event-driven simulator** with explicit state (queues, KV cache occupancy, active batches, network links) so routing/scheduling decisions are modeled the same way real systems behave under load.
- Structure it similarly to Vidur: a model specification → operator latency model → hierarchical scheduler (global/router + replica scheduler + within-replica microbatching) → metrics tracker.
- Keep the simulator “pluggable” at two seams:
    - **Scheduler plug-ins** (continuous batching policies, chunked prefill, preempt/recompute, admission control).
    - **Operator/kernel plug-ins** (attention backend choices, communication collectives, KV movement), each returning both numeric latency and a SymPy expression for that latency.


## Symbolic metrics (SymPy) design

- Represent *every* timing primitive as a SymPy expression, then build metrics as compositions:
    - Per-iteration time: `t_iter = Max(t_compute, t_mem, t_comm)` (common when one bottleneck dominates).
    - Decode loop time for a request: `t_decode = Sum(t_step(i), (i, 1, n_out))`, with `t_step` depending on batch composition and KV state.
- Encode branching/choices with SymPy:
    - **Branching** (e.g., cache hit vs miss): `Piecewise((t_hit, hit), (t_miss, True))`.
    - **Kernel selection** (pick fastest feasible kernel): `Min(t_flash(n, b), t_paged(n, b), t_xformers(n, b))` plus feasibility as `Piecewise` guards (e.g., head-dim constraints, alignment, workspace).
    - **Chunked prefill loops**: `Sum(t_prefill_chunk(c), (c, 1, n_chunks))`, where `n_chunks = ceiling(n_prompt / chunk_size)` can be modeled with `ceiling` / `floor`.
- Throughput as a symbolic function:
    - Tokens/sec over a horizon: `TP = (Sum(tokens_done(k), (k,1,K)))/T_total`, with `T_total = Sum(t_iter(j), (j,1,J))`.
- Practical tip: don’t try to make the whole queueing dynamics perfectly symbolic; instead keep *state evolution* numeric, while making *latency laws and metric formulas* symbolic in terms of decision variables (batch caps, chunk size, routing policy knobs, etc.), then “freeze” observed/state-dependent quantities into parameters for the solver.


## SGLang-specific pieces to model

- **RadixAttention / prefix reuse**: model cache hits as reduced prefill cost (cached tokens don’t re-run compute) and track a hit-rate or matched-prefix length distribution; SGLang refers to KV cache reuse as “RadixAttention” and indicates it is enabled by default.[^1_2]
- **Cache-aware routing**: include a router that chooses a worker to maximize expected cache hits subject to load; SGLang v0.4 reports large throughput gains with a cache-aware router (and much higher cache hit rate in their example).[^1_1]
- **Router deployment modes / disaggregation**: if you need prefill/decode split across different pools, model KV transfer as an explicit communication+serialization term; SGLang router docs list “Prefill-Decode Disaggregation” as a supported mode.[^1_3]
- **Speculative decoding**: treat it as a different decode kernel graph (draft model work + verification work + acceptance branch); SGLang documents an EAGLE-based speculative decoding option and notes compatibility with radix cache and chunked prefill.[^1_4]


## Implementation plan (concrete)

- Define a small IR for “latency expressions”:
    - `Latency(value: sympy.Expr, numeric: float, metadata: dict)` and overload `+`, `Max`, `Min`, etc., so your simulator code stays clean.
- Implement operator models:
    - Start from a simple decomposition aligned with Vidur’s operator triaging (token-level vs sequence-level/attention vs communication).
    - For decode attention, allow a memory-bound approximation driven by total KV reads (Vidur explicitly motivates decode attention as largely memory-bound and modeled via total KV-cache reads).
- Borrow the **simple end-to-end timing law** used by llm-d-inference-sim as your initial “outer” metric, then refine internally:
    - For non-streaming, llm-d-inference-sim models response delay as `TTFT + ITL*(n_out-1)` (or KV-transfer latency instead of TTFT in a P/D case).
- Add schedulers incrementally:

1. Continuous batching (vLLM-like) baseline.
2. Chunked prefill.
3. Cache-aware routing + prefix cache model.[^1_2][^1_1]
4. Disaggregated prefill/decode with KV transfer.[^1_3]
5. Speculative decoding graph.[^1_4]

What’s the primary optimization problem you want to solve with the SymPy throughput function (e.g., choose `chunk_size`, `max_batch_tokens`, `tp/pp`, router policy knobs, or something else), and is the target scenario single-node or multi-node with a separate router?
<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_5][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_6][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_7][^1_8][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: https://lmsys.org/blog/2024-12-04-sglang-v0-4/

[^1_2]: https://github.com/sgl-project/sglang/issues/906

[^1_3]: https://docs.sglang.ai/advanced_features/router.html

[^1_4]: https://docs.sglang.ai/backend/speculative_decoding.html

[^1_5]: https://github.com/llm-d/llm-d-inference-sim

[^1_6]: https://github.com/microsoft/vidur

[^1_7]: https://arxiv.org/abs/2410.00161

[^1_8]: https://www.ijraset.com/best-journal/openai-model-efficient-memory-reduce-management-for-the-large-language-models-llms-serving-with-paged-attention-of-sharing-the-kv-cashes-205

[^1_9]: https://arxiv.org/abs/2411.02886

[^1_10]: https://arxiv.org/abs/2412.10319

[^1_11]: https://arxiv.org/abs/2509.04377

[^1_12]: https://dl.acm.org/doi/10.1145/3748273.3749196

[^1_13]: https://arxiv.org/abs/2506.07311

[^1_14]: https://arxiv.org/abs/2412.03131

[^1_15]: https://arxiv.org/abs/2511.00868

[^1_16]: https://arxiv.org/abs/2510.11292

[^1_17]: https://dl.acm.org/doi/pdf/10.1145/3600006.3613165

[^1_18]: https://arxiv.org/pdf/2309.06180.pdf

[^1_19]: http://arxiv.org/pdf/2405.10637.pdf

[^1_20]: https://arxiv.org/pdf/2402.06082.pdf

[^1_21]: https://arxiv.org/pdf/2408.01890v1.pdf

[^1_22]: http://arxiv.org/pdf/2503.08879.pdf

[^1_23]: https://arxiv.org/html/2407.12866v1

[^1_24]: https://arxiv.org/html/2411.05787

[^1_25]: https://lmsys.org/blog/2025-09-10-sglang-hicache/

[^1_26]: https://cloud.tencent.com/developer/article/2424704

[^1_27]: https://blog.csdn.net/weixin_41544125/article/details/149672704

[^1_28]: https://blogs.novita.ai/sglang/

[^1_29]: https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/SpecForge_SGlang.html

[^1_30]: https://hebiao064.github.io/fa3-attn-backend-basic

[^1_31]: https://sgl-project.github.io

[^1_32]: https://docs.sglang.io/advanced_features/router.html

[^1_33]: https://lmsys.org/blog/2025-07-17-mtp/

[^1_34]: https://github.com/sgl-project/sglang/discussions/652

[^1_35]: https://docs.sglang.ai/_sources/router/router.md

[^1_36]: https://lmsys.org/blog/2025-07-25-spec-forge/

[^1_37]: https://huggingface.co/docs/text-generation-inference/en/conceptual/paged_attention

[^1_38]: https://github.com/sgl-project/sglang/issues/2687

[^1_39]: https://docs.sglang.io/advanced_features/speculative_decoding.html

[^1_40]: https://faradawnyang.substack.com/p/sglang-code-walkthrough

[^1_41]: https://arxiv.org/abs/2306.10998

[^1_42]: https://arxiv.org/abs/2409.05923

[^1_43]: http://arxiv.org/pdf/2405.05465.pdf

[^1_44]: https://arxiv.org/pdf/2408.05499.pdf

[^1_45]: https://arxiv.org/html/2406.01967v1

[^1_46]: https://arxiv.org/html/2503.08415v1

[^1_47]: https://arxiv.org/pdf/2501.00032.pdf

[^1_48]: https://arxiv.org/pdf/2306.07629.pdf

[^1_49]: https://arxiv.org/html/2309.08168

[^1_50]: http://arxiv.org/pdf/2411.17337.pdf

[^1_51]: https://github.com/llm-d/llm-d

[^1_52]: https://llm-d.ai/docs/architecture/Components/inference-sim

[^1_53]: https://github.com/llm-d/llm-d-inference-sim/actions

[^1_54]: https://github.com/orgs/llm-d/packages/container/llm-d-inference-sim/453031926

[^1_55]: https://github.com/llm-d

[^1_56]: https://apanwariisc.github.io/publications/mlsys-2024-vidur/vidur_mlsys24.pdf

[^1_57]: https://www.reddit.com/r/LocalLLaMA/comments/1d5dlzh/how_do_you_measure_throughput_of_your_llm_model/

[^1_58]: https://github.com/Muhtasham/simulator

[^1_59]: https://bentoml.com/llm/inference-optimization/llm-inference-metrics

[^1_60]: https://symbl.ai/developers/blog/a-guide-to-llm-inference-performance-monitoring/

[^1_61]: http://llm-d.ai/docs/architecture/Components/inference-sim

[^1_62]: https://arxiv.org/html/2405.05465v1

[^1_63]: https://apxml.com/courses/quantized-llm-deployment/chapter-3-performance-evaluation-quantized-llms/measuring-inference-latency-throughput

[^1_64]: https://www.microsoft.com/en-us/research/publication/vidur-a-large-scale-simulation-framework-for-llm-inference/

[^1_65]: https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/

[^1_66]: https://github.com/orgs/llm-d/packages

[^1_67]: https://arxiv.org/abs/2405.05465

[^1_68]: https://www.baseten.co/blog/llm-transformer-inference-guide/

[^1_69]: https://github.com/llm-d/llm-d/releases


---

# what are the limitations and obstacles for having a purely symbolic queueing dynamics and end-to-end formulas for metrics in terms of configuration variables?

Now I'll generate the final comprehensive answer:

## Limitations \& Obstacles: Purely Symbolic Queueing Dynamics for LLM Inference Simulators

The pursuit of **purely symbolic (closed-form) queueing dynamics and end-to-end metrics** for LLM inference simulators confronts **fundamental mathematical barriers** that make comprehensive symbolic solutions infeasible. These obstacles span theoretical foundations, structural peculiarities of LLM workloads, and practical solver challenges.

***

### 1. **Loss of Markovian Assumptions \& Classical Closed-Form Solutions**

Classical queueing theory (M/M/1, M/M/c, M/D/1) yields closed-form steady-state distributions because **Markovian assumptions**—exponential inter-arrivals (Poisson process) and exponential service times—decouple the queue's past from its future. In LLM inference, **neither assumption holds**:[^2_1]

**Non-Poisson Arrivals**: Real workloads are bursty with correlated arrivals (time-of-day patterns, cascading API calls in compound systems). The G (general) arrival process breaks the renewal assumption. No closed-form exists for G/G/1 systems in the general case.[^2_1]

**Non-Exponential Service**: Prefill time scales as O(seq_in²) / compute (compute-bound GEMM), while decode time scales as O(seq_out) / bandwidth (memory-bound, linear). These are deterministic or heavy-tailed (Weibull-like), not exponential. Even for M/G/1 (Poisson arrivals with general service), exact solutions require solving **Lindley integral equations** numerically. The Pollaczek-Khinchin formula:

```
L_q = (λ² * E[S²]) / (2(1 - ρ))
```

appears closed-form, but **E[S²] is state-dependent and empirical** in LLM systems—it depends on unknown input/output lengths and batch composition, making it unmeasurable at submission time.

***

### 2. **State Dependence Breaks Markovian Structure**

In real systems, both arrival rate λ(t) and service rate μ(t) depend on system state (queue depth, cache occupancy, batch size). This **state-dependent service** μ(n) destroys the Markov chain analysis. For even the simplest state-dependent system (M^n/M^n/1), the steady-state solution is **implicit and requires iteration**—a system of coupled nonlinear equations with no closed form.[^2_2]

**For LLM**: State dependence is severe because:

- Batch size affects GPU utilization curves (nonlinear, non-convex).
- KV cache occupancy affects attention latency (cache hits reduce flops).
- Position in decode loop affects attention recomputation cost.

The state-dependent queue equations involve **convolution-like integrals**  that cannot be solved symbolically. Solutions are found numerically via fixed-point iteration, not closed-form algebra.[^2_2]

***

### 3. **Two-Phase Processing with Phase-Dependent Latencies**

Each request transitions through two **coupled, distinct phases**:

- **Prefill** (0 → n_in): Compute-bound, latency ~ O(seq_in²) / compute.
- **Decode** (n_in → n_in + n_out): Memory-bound, latency ~ O(seq_out) / memory_bw.

A single-phase M/G/1 model treats the entire request as one "job," but LLM requests have:

- **Unknown service time at arrival**: n_out (output length) is only determined during generation—it's a **stochastic variable**.
- **Coupled scheduling**: In continuous batching, prefill and decode requests compete for the same GPU in **iteration-level interleaving**.

Expressing throughput requires modeling **batch composition** (distribution over request phases), not a scalar service time. The scheduler must decide at each iteration:

- Which prefill requests complete?
- Which decode requests advance?
- Which new requests admit?

**There is no closed form** for the optimal decision because it depends on **predicted output lengths**, **cache hit probabilities**, and **hardware utilization curves**—all state-dependent and learned from data. The TTFT/TPOT trade-off creates **non-convex Pareto frontier** that is empirically explored, not symbolically derived.

***

### 4. **Branching \& Conditional Kernel Selection**

Different attention kernels (FlashAttention, xformers paged, CUTLASS) are feasible in different regions of config space:

- FlashAttention: Feasible for seq_len ≤ 16K, head_dim ∈ {64, 128}, with workspace constraints.
- Paged attention: Always feasible, but 2–3× slower for small batches.

Kernel choice creates a **piecewise-defined regime** where feasibility conditions depend on config parameters. When a solver explores the parameter space, it may:[^2_3]

1. **Move between regimes**: The optimal kernel switches, causing discontinuities in the latency function. Gradients are undefined, and solvers get stuck or oscillate.
2. **Face incompatible performance laws**: FlashAttention latency ∝ seq²/bw; paged attention ∝ tokens/iter + page_misses. Each regime has **fundamentally different scaling**, making a unified symbolic formula unstable numerically.

Nested Piecewise expressions (kernel choice × batch size regime × cache state) create **intractable symbolic simplifications** that SymPy cannot resolve efficiently. The resulting throughput formula is **correct mathematically but unsolvable numerically** because the solver cannot handle discontinuities.

***

### 5. **KV Cache \& Prefix Reuse (RadixAttention)**

SGLang's RadixAttention enables cache reuse when prompts share prefixes. The **cache hit rate π** is a function of:

- Request routing strategy.
- Arrival timing (must overlap to share cache).
- Eviction policy (LRU, policy-based).
- Cache capacity.

**There is no closed-form for π(config)** because:

1. **Data-dependent**: Hit rate depends on empirical request patterns ("What % of requests share prefixes?"), which is workload-specific and not derivable from config.
2. **Stochastic cache dynamics**: Cache occupancy evolves as a Markov chain (random eviction) or deterministic system (policy-based). Either way, the occupancy distribution requires solving the **full state space numerically**.
3. **Adaptive routing**: If the router optimizes for cache hits, it solves an **online optimization problem** at runtime—the solution is dynamic and not pre-computable.

The throughput gain:

```
TP_cached = base_TP / (1 - π * gain_from_hit + (1 - π) * cost_of_miss)
```

becomes parameterized by **measured π**, not a symbolic quantity. Trying to include cache dynamics in a symbolic formula would require embedding **fixed-point equations** or **Markov chain analysis**, introducing transcendental functions (Bessel functions appear in queueing solutions ) with no closed form.[^2_4]

***

### 6. **Continuous Batching \& Non-Stationary Scheduling**

In continuous batching, the scheduler makes per-iteration decisions (batch cap, preemption threshold, admission control). These decisions affect latencies, which in turn affect optimal decisions—a **feedback loop**.[^2_5]

Optimal scheduling in LLM systems is **non-stationary**: The optimal preemption threshold depends on:

- Current **KV cache size** (output-length-dependent, stochastic).
- Queue **depth** (affects preemption pain and admission risk).
- **Arrival predictions** (if new request is short, preemption pays off; if long, not).

Even simple throughput models reduce to **implicit fixed-point equations**:

```
TP = (tokens_per_cycle) / cycle_time(TP, batch_cap, ...)
```

where throughput TP appears on both sides. Solving for TP requires **numerical fixed-point iteration** or root-finding, not closed-form algebra.

Practical schedulers optimize **multi-objective functions** (minimize p50 latency, maximize throughput, minimize cost, ensure fairness). The Pareto frontier is **empirically explored** (via RL or Bayesian optimization), not symbolically derived. **SymPy cannot represent the optimal policy** as a closed-form function of state.

***

### 7. **Coupling Across Multiple Scales**

The system involves **15+ coupled dimensions**:

- Hardware specs (FLOP/s, memory bandwidth, capacity).
- Model specs (size, head dimension, hidden dimension).
- Scheduling knobs (batch cap, chunk size, tensor parallelism, pipeline parallelism).
- Workload (arrival rate, prompt/output length distribution).
- Policies (routing, cache eviction, preemption strategy).

Each dimension introduces **conditional logic, nonlinearity, and feedback**. The composed throughput function is:

- **Highly nonlinear** (batch utilization curves, saturation effects).
- **Piecewise-defined** (kernel switches, phase transitions).
- **Implicit** (throughput feeds back into cycle time).

SymPy can represent this, but:

1. **Symbolic simplification fails** (too many terms, conditionals, interdependencies).
2. **Numerical evaluation is slow** (nested Piecewise requires many condition checks).
3. **Solvers get lost** (high-dimensional, non-convex, discontinuous landscape with undefined gradients).

***

### 8. **Transient Dynamics \& Cache Warmup**

Classical queueing formulas assume **steady state** (t → ∞). But LLM workloads exhibit:

- **Warmup phase**: Cache starts empty, hit rate ramps from 0% to steady-state over minutes to hours.
- **Burst-transient**: Traffic spikes cause queue filling → latency spikes → queue drainage.
- **Phase transitions**: System switches from cache-hit-dominated (high throughput) to eviction-dominated (low throughput) at certain loads.

The transient response is governed by **coupled nonlinear ODEs** (under fluid-limit approximation), which have **no closed-form solution** for general state-dependent service rates. The steady-state formula **does not apply** during early operation, making it unsuitable for characterizing real deployment dynamics.

***

### 9. **Measurement Feedback \& Adaptive Policies**

Real schedulers use runtime measurements to adapt:

```
if KV_growth_rate > threshold:
    preempt_request()
if cache_hit_rate < target:
    flush_cache()
```

This creates a **dynamical system with feedback**:

```
state(t+1) = F(state(t), decisions(t))
decisions(t) = scheduler(state(t))
```

The closed-loop throughput depends on the **fixed-point state** where scheduler decisions are optimal *given those same decisions*. There is **no closed-form solution** unless the system simplifies (e.g., fluid limit), but then you lose the adaptive feedback that makes real systems robust.

***

### 10. **What IS Feasible Symbolically**

Despite these obstacles, some components work well:

**Single-Operator Latency Kernels**: For an isolated attention kernel:

```python
t_attn = Max(
    n * b * d² / compute,      # compute-bound GEMM
    n * b * d * 4 / bandwidth   # memory-bound KV reads
)
```

This works because you're modeling **one hardware operation** with clear compute/memory trade-offs, without queue-state coupling.

**Fluid-Limit Metrics** (high-load regime):

```
TP_steady_state ≈ [min(compute_capacity, bw / flops_per_byte)] * (1 - overhead)
```

Useful for quick estimates and parameter sweeps, but breaks when load drops below saturation or cache hit rate varies.

**Sensitivity Analysis**: Given a fixed-point formula, you can compute symbolic derivatives:

```python
dTP / dbatch_cap = ...  [symbolic expression]
```

This informs optimization without solving the formula exactly.

**Piecewise-Defined Regimes with Enumeration**: If there are ~5–100 kernel/policy combinations, enumerate all; optimize within each; return the best. Works if transitions are rare and the number of regimes is small.

***

### 11. **The Pragmatic Hybrid Architecture**

Given these limitations, the recommendation is **not** to pursue purely symbolic queueing dynamics, but a **hybrid system**:

**Symbolic Layer**: Operator kernels, bottleneck laws, sensitivity derivatives.

- Single-operator latency: SymPy Max/Min with local constraints.
- Bottleneck analysis: Assume one phase dominant; derive closed-form.
- Sensitivity analysis: Compute ∂TP/∂config symbolically.

**Numeric Layer**: Event-driven simulator for queue evolution, cache dynamics, adaptive scheduling.

- Discrete-event simulation for state transitions.
- Numeric optimization of scheduling decisions.
- Measurement of empirical effects (cache hit rate, overhead).

**Bridge**: Measure empirical quantities; substitute into symbolic formulas.

- Simulate a config to measure cache_hit_rate, preempt_overhead.
- Plug into: `TP_formula(cache_hit_rate=0.85, preempt_cost=0.1, ...)`
- Optimize over config space (batch_cap, chunk_size, tp_size).

This **avoids symbolically solving for cache hit rate** (impossible) while retaining **interpretability and optimization benefits** of symbolic metrics for the aspects where they work.

***

### **Key

<span style="display:none">[^2_10][^2_11][^2_12][^2_13][^2_14][^2_15][^2_16][^2_17][^2_18][^2_19][^2_20][^2_21][^2_22][^2_23][^2_24][^2_25][^2_26][^2_27][^2_28][^2_29][^2_30][^2_31][^2_32][^2_33][^2_34][^2_35][^2_36][^2_37][^2_38][^2_39][^2_40][^2_41][^2_42][^2_43][^2_44][^2_45][^2_46][^2_47][^2_48][^2_49][^2_50][^2_51][^2_52][^2_53][^2_54][^2_55][^2_56][^2_57][^2_6][^2_7][^2_8][^2_9]</span>

<div align="center">⁂</div>

[^2_1]: https://ric.zp.edu.ua/article/view/241738

[^2_2]: https://www-2.rotman.utoronto.ca/opher.baron/files/A21_state-dependent%20MG1%20queueing%20systems.pdf

[^2_3]: https://lmsys.org/blog/2024-12-04-sglang-v0-4/

[^2_4]: https://symbl.ai/developers/blog/a-guide-to-llm-inference-performance-monitoring/

[^2_5]: https://www.youtube.com/watch?v=YwSD5hJfyAU

[^2_6]: https://link.springer.com/10.1007/s11249-022-01633-z

[^2_7]: https://link.springer.com/10.1007/s00170-022-10607-3

[^2_8]: https://link.springer.com/10.1007/s11134-022-09767-6

[^2_9]: https://www.mdpi.com/1999-4893/15/5/151

[^2_10]: http://www.iieta.org/journals/ti-ijes/paper/10.18280/ti-ijes.630105

[^2_11]: https://linkinghub.elsevier.com/retrieve/pii/S0378475423005384

[^2_12]: http://link.springer.com/10.1007/978-3-319-31317-7_20

[^2_13]: https://www.preprints.org/manuscript/202111.0008/v1

[^2_14]: https://linkinghub.elsevier.com/retrieve/pii/S0141029616300852

[^2_15]: http://macs.journals.semnan.ac.ir/article_371.html

[^2_16]: https://arxiv.org/pdf/2409.08075.pdf

[^2_17]: https://arxiv.org/pdf/1207.0382.pdf

[^2_18]: https://arxiv.org/pdf/2305.07229.pdf

[^2_19]: https://www.techrxiv.org/articles/preprint/Explicit_Results_for_the_Distributions_of_Queue_Lengths_for_a_Non-Preemptive_Two-Level_Priority_Queue/24153564/1/files/42377121.pdf

[^2_20]: https://arxiv.org/pdf/2307.07746.pdf

[^2_21]: https://www.mdpi.com/2073-4336/15/3/19/pdf?version=1716971751

[^2_22]: https://ccsenet.org/journal/index.php/ijsp/article/download/0/0/50268/54411

[^2_23]: https://arxiv.org/html/2503.16633v1

[^2_24]: https://disco.ethz.ch/courses/hs15/des/lectures/des_chapter6-4on1.pdf

[^2_25]: https://homepages.ecs.vuw.ac.nz/~schukova/SCIE201/Lectures/Lecture9_final2018.html

[^2_26]: https://people.orie.cornell.edu/jdai/publications/daiMeyn95.pdf

[^2_27]: https://textbook.simio.com/SASMAA7/ch-queueing.html

[^2_28]: https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=1110\&context=cstech

[^2_29]: https://onlinelibrary.wiley.com/doi/abs/10.1002/9780470400531.eorms0329

[^2_30]: https://docenti.ing.unipi.it/~a080368/Teaching/PEVA/pdf/Notes on Queueing Theory - Students 20220727.pdf

[^2_31]: https://www2.isye.gatech.edu/~sman/courses/6644/Module05Q-QueueingSlides_171025.pdf

[^2_32]: https://www.columbia.edu/~ww2040/MAMA_LiuWhitt_081210.pdf

[^2_33]: https://www.le.infn.it/~chiodini/allow_listing/slideMSFNSN/qing.pdf

[^2_34]: https://www.semanticscholar.org/paper/224d400a77223794ae075e89935dbf25eed8dc7a

[^2_35]: https://www.semanticscholar.org/paper/f6bab5736430b5ec01bc5b22f31f89b6db3d2786

[^2_36]: https://www.tandfonline.com/doi/full/10.1080/02533839.2010.9671610

[^2_37]: http://link.springer.com/10.1007/s11134-016-9503-0

[^2_38]: https://www.semanticscholar.org/paper/f696c1aa92e7ba7c6173cb0deebcca81198fe70c

[^2_39]: https://pubs.aip.org/aip/acp/article/1026023

[^2_40]: https://ieeexplore.ieee.org/document/10891778/

[^2_41]: https://onlinelibrary.wiley.com/doi/10.1155/2011/401365

[^2_42]: https://www.semanticscholar.org/paper/bf83a6d0562ad13a982d2c6ee50877b29240196b

[^2_43]: https://downloads.hindawi.com/archive/2003/579034.pdf

[^2_44]: http://arxiv.org/pdf/2004.01475.pdf

[^2_45]: https://downloads.hindawi.com/journals/mpe/2014/519579.pdf

[^2_46]: http://arxiv.org/pdf/1907.13019.pdf

[^2_47]: https://arxiv.org/pdf/2001.00946.pdf

[^2_48]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11637148/

[^2_49]: https://arxiv.org/pdf/1106.0138.pdf

[^2_50]: https://dl.acm.org/doi/pdf/10.1145/3565287.3610275

[^2_51]: https://pure.tue.nl/ws/portalfiles/portal/1755140/200513578.pdf

[^2_52]: https://www.malayajournal.org/articles/paper10.pdf

[^2_53]: https://arxiv.org/pdf/2503.07545.pdf

[^2_54]: https://www.sciencedirect.com/science/article/abs/pii/S0378375801001616

[^2_55]: https://www.sciencedirect.com/science/article/abs/pii/S0377042723006222

[^2_56]: https://arxiv.org/html/2512.16134v1

[^2_57]: https://cyberleninka.ru/article/n/survey-on-non-markovian-queue-with-phase-service-using-supplementary-variable-technique/pdf


---

# i feel like the above output is incomplete, please finish outlining the detailed plan of implementation _/\_

Below is a concrete, implementation-oriented plan that finishes the outline and turns it into a buildable Python project, while preserving your requirement: “every metric (esp. throughput) must be expressible as a symbolic function (SymPy), including branching/loops/kernel choices.”

## Target deliverable

Build two front-ends on the same core: (A) a **trace-driven discrete-event simulator** (Vidur-like), and (B) an optional lightweight **OpenAI/vLLM-compatible mock server** (llm-d-inference-sim-like) for black-box workload replay and metrics scraping.[^3_1]
The mock server is useful because llm-d-inference-sim already defines a clean metric surface (TTFT, TPOT, inter-token latency, queue time, prefill/decode time, etc.) and a simple end-to-end timing model for non-streaming responses: `TTFT + ITL*(n_out-1)` (with KV-transfer latency substituted in P/D cases).

## Package layout \& data model

Use a “small core + plugins” structure so SGLang-specific features (router, radix cache, speculative decoding) are just additional modules rather than rewriting the simulator.[^3_2][^3_1]
A workable layout:

- `sim/`
    - `core/`
        - `events.py`: `Event` base class, `Arrival`, `Dispatch`, `IterationStart`, `KernelStart/End`, `TokenEmitted`, `RequestDone`, `KvTransferStart/End`.
        - `state.py`: `ClusterState`, `ReplicaState`, `GpuState`, `CacheState`, `NetworkState`.
        - `workload.py`: request objects (`prompt_len`, `max_new_tokens`, `streaming`, sampling params), traces, synthetic generators (Vidur supports synthetic + trace-based length and poisson interval generators).[^3_1]
    - `models/`
        - `model_spec.py`: transformer dimensions, num_layers, kv bytes/token, flops/token, attention scaling.
        - `operator_graph.py`: “per token per layer” operator DAG templates for prefill/decode.
    - `kernels/`
        - `latency_models.py`: kernel families + feasibility + piecewise selection (`flash`, `paged`, `cutlass`, comm kernels).
    - `schedulers/`
        - `global_router.py`: round-robin vs cache-aware routing; include disaggregation mode (prefill/decode separation).[^3_2]
        - `replica_scheduler.py`: vLLM-v1-like continuous batching, chunked prefill knobs (Vidur exposes scheduler config incl. `chunk_size` and `batch_size_cap`).[^3_1]
    - `metrics/`
        - `definitions.py`: TTFT, TPOT, throughput, E2E, queue time histograms consistent with the llm-d/vLLM metric names if you want drop-in Prometheus compatibility.
        - `symbolic.py`: SymPy metric builder + simplifier + exporters (SymPy -> lambdify -> numeric).
    - `server/` (optional)
        - `openai_server.py`: minimal `/v1/chat/completions`, `/v1/completions`, `/metrics` like llm-d-inference-sim.


## Symbolic expression system (the “secret sauce”)

Implement a dual evaluation path: **(1) numeric simulation** for state evolution, and **(2) symbolic construction** for latency/metric formulas. The critical trick is to make symbolic expressions depend on *configuration variables* and a small set of *frozen state parameters* (measured or sampled during simulation), because fully symbolic queue evolution is generally intractable (as discussed).

Concrete design:

- Define a typed wrapper:
    - `Expr = sympy.Expr`
    - `Val = float`
    - `SymVal(expr: Expr, val: Val, meta: dict)`
- Provide combinators that keep expr+val aligned:
    - `sym_add(a,b)`, `sym_max(a,b)`, `sym_min(a,b)`, `sym_piecewise([(cond, x), (True, y)])`.
- Represent “kernel choice” as:
    - `t_attn = Piecewise((t_flash, flash_feasible), (t_paged, True))` or `Min(t_flash, t_paged)` when both feasible.
- Represent loops as:
    - `Sum(t_step(i), (i, 1, n_out))` for decode, and `Sum(t_chunk(j), (j, 1, ceil(n_in/chunk)))` for chunked prefill.
- Represent routing / cache hit branches explicitly:
    - cache hit/miss: `Piecewise((t_hit, hit), (t_miss, True))`
    - P/D KV transfer substitution: llm-d-inference-sim explicitly uses `kv-cache-transfer-latency` in place of TTFT in P/D cases.

This lets you later export a single symbolic throughput function $TP(config)$ and feed it into an optimizer—while still computing “hit”, “feasible”, “queue nonempty”, etc. numerically.

## Simulation engine \& SGLang features

Implement an event loop (priority queue by time), but structure replica execution as **iteration ticks** (like modern continuous batching), because many key decisions happen per-iteration (batch composition, chunk selection, token emission).[^3_1]
Roadmap (build in this order):

1. **Baseline single-replica continuous batching**
    - Replica has `waiting_queue`, `running_set`, `iteration_batch`.
    - Each tick: select batch under `batch_size_cap` and `chunk_size` parameters (Vidur exposes these knobs in `replica_scheduler_config`).[^3_1]
2. **Prefill vs decode separation**
    - Add request phase machine: WAITING → PREFILL → DECODE → DONE.
    - Support disaggregation mode by introducing a KV-transfer edge (a network+serialization latency term); llm-d-inference-sim explicitly calls out P/D disaggregation and KV transfer latency behavior.[^3_2]
3. **Prefix/radix caching + cache-aware routing**
    - Add `CacheState` with capacity, block size, and a prefix index; track “prefix length matched” per request and compute saved prefill cost when hit.
    - Add a router policy that tries to send requests to replicas with best expected cache reuse; SGLang v0.4 specifically highlights a cache-aware router with higher cache hit rate and throughput improvements.[^3_3]
4. **Speculative decoding (optional, later)**
    - Model as a decode subgraph with acceptance probability parameter and a branch (accept → commit tokens, reject → verify fewer tokens / redo work); SGLang documents speculative decoding as a supported backend feature.[^3_4]

## Calibration, validation, and solver integration

Start with a profiling interface that can ingest (a) microbench tables per kernel (latency vs shapes), (b) bandwidth ceilings, or (c) measured TPOT/TTFT curves from real SGLang runs; then fit the symbolic kernel coefficients to match.

For validation, emit the same kind of outputs Vidur emphasizes (aggregated metrics + traces); Vidur’s CLI-driven design and rich config surface are a good template for running many scenarios reproducibly.[^3_1]
For the solver loop:

- Decide which variables are “decision vars” vs “frozen params”:
    - Decision vars: `chunk_size`, `batch_size_cap`, TP/PP degrees, router policy knobs.
    - Frozen params (measured per workload/config): cache hit rate, prefix length distribution, acceptance rate for speculative decoding.
- Produce:

1. `TP_expr(config_symbols, frozen_params)` (SymPy).
2. A fast numeric evaluator via `sympy.lambdify`.
3. Constraints as symbolic predicates (memory limit, KV capacity, feasibility guards).
- Optimize using either:
    - Mixed-integer (if kernel choices are discrete), or
    - Enumerate discrete regimes (kernel/router modes) and run continuous optimization per regime.

If you answer these, the plan can be pinned down to concrete class signatures and the first runnable milestone:

1) Is the primary target **single node** (8×GPU) or **multi-node + router + P/D disaggregation**?[^3_2]
2) Which SGLang features must be “day-1”: radix/prefix cache, chunked prefill, speculative decoding, or router/disaggregation?[^3_3][^3_4]
<span style="display:none">[^3_10][^3_11][^3_12][^3_13][^3_14][^3_15][^3_16][^3_17][^3_18][^3_19][^3_20][^3_21][^3_22][^3_23][^3_24][^3_25][^3_26][^3_27][^3_28][^3_29][^3_30][^3_31][^3_5][^3_6][^3_7][^3_8][^3_9]</span>

<div align="center">⁂</div>

[^3_1]: https://github.com/microsoft/vidur

[^3_2]: https://docs.sglang.ai/advanced_features/router.html

[^3_3]: https://lmsys.org/blog/2024-12-04-sglang-v0-4/

[^3_4]: https://docs.sglang.ai/backend/speculative_decoding.html

[^3_5]: http://arxiv.org/pdf/2405.05465.pdf

[^3_6]: https://arxiv.org/abs/2310.19773

[^3_7]: https://arxiv.org/abs/2408.09408

[^3_8]: https://arxiv.org/html/2405.04233

[^3_9]: https://www.mdpi.com/2306-5354/10/3/380/pdf?version=1679325075

[^3_10]: http://arxiv.org/pdf/2403.15377v4.pdf

[^3_11]: https://arxiv.org/html/2503.02063v1

[^3_12]: https://arxiv.org/html/2501.01427v1

[^3_13]: https://github.com/microsoft/vidur/issues/27

[^3_14]: https://www.microsoft.com/en-us/research/publication/vidur-a-large-scale-simulation-framework-for-llm-inference/

[^3_15]: https://stackoverflow.com/questions/2234204/find-nearest-latitude-longitude-with-an-sql-query

[^3_16]: https://www.facebook.com/groups/dataengineeringpilipinas/posts/1458566498112746/

[^3_17]: https://github.com/ozcanmiraay/vidur-energy

[^3_18]: https://ai92blog.blogspot.com/2024/08/vllm-parameters-performance.html

[^3_19]: https://www.microsoft.com/en-us/research/publication/vidur-a-large-scale-simulation-framework-for-llm-inference/?lang=zh-cn

[^3_20]: https://github.com/allenai/kb

[^3_21]: https://blog.csdn.net/javastart/article/details/137691490

[^3_22]: https://edoc.ub.uni-muenchen.de/29401/7/Berrendorf_Max.pdf

[^3_23]: https://www.facebook.com/groups/developerkaki/posts/2546623179016913/

[^3_24]: https://discuss.vllm.ai/t/should-vllm-consider-prefix-caching-when-chunked-prefill-is-enabled/903

[^3_25]: https://github.com/microsoft/vidur/issues/9

[^3_26]: https://github.com/whoisdsmith/project-awesome

[^3_27]: https://gateway-api-inference-extension.sigs.k8s.io/guides/epp-configuration/prefix-aware/

[^3_28]: https://etalpykla.vilniustech.lt/bitstream/handle/123456789/157325/O_Pozdniakovos_disertacija.pdf?sequence=1\&isAllowed=y

[^3_29]: https://dspace.mit.edu/bitstream/handle/1721.1/152792/sinha-anusinha-phd-hst-2023-thesis.pdf?sequence=1\&isAllowed=y

[^3_30]: https://arxiv.org/abs/2405.05465

[^3_31]: http://scyr.kpi.fei.tuke.sk/wp-content/scyr-files/history/SCYR_2021_Proceedings.pdf


---

# primary target is multi-node with router and p/d disaggregation. we absolutely have to implement prefix cache, chunked prefill, disagregation, P/D instance ratio, Prefix-caching, Tensor/Data/Expert Parallelism and Batching.

Below is a concrete implementation plan specialized for **multi-node + router + prefill/decode (P/D) disaggregation**, with **prefix cache, chunked prefill, P/D instance ratio, prefix-caching-aware routing, Tensor/Data/Expert Parallelism, and batching** as first-class knobs.[^4_1]

## System topology \& config surface

Model the deployment as 3 layers: **router → worker pools (prefill/decode) → intra-worker parallel groups (TP/DP/EP)**.[^4_1]

**Top-level objects**

- `Router`: routes requests to workers, supports explicit PD disaggregation (separate prefill and decode pools) and separate policies for prefill vs decode pools.
- `PrefillWorkerPool`: `N_prefill` worker instances (your P/D ratio numerator), each with its own prefix cache state and scheduling loop.
- `DecodeWorkerPool`: `N_decode` worker instances (your P/D ratio denominator), each with decode KV residency + token loop scheduling.

**Router modes/knobs you must represent**

- PD mode must accept two worker lists (prefill servers and decode servers) and allow distinct prefill-policy/decode-policy routing.
- Cache-aware routing policy needs an approximate per-worker radix tree, a `cache_threshold` (minimum prefix match ratio), and “balance thresholds” that switch to shortest-queue when imbalanced.
- Data-parallelism-aware routing should exist as a separate mode (`dp-aware`) that coordinates with DP ranks for distribution across replicas.

**Config table (core decision variables)**


| Knob | Symbol(s) for solver | Where it acts | What it changes |
| :-- | :-- | :-- | :-- |
| P/D ratio | $N_p, N_d$ | Router + pools | Prefill capacity vs decode capacity, KV-transfer pressure. |
| Chunked prefill | $c$ | Prefill worker scheduler | Prefill broken into chunks; impacts TTFT and overlap behavior. [^4_1] |
| Prefix caching | $C_{kv}, B_{blk}, \pi_{evict}$ | Prefill cache + router | Cache hit rates, effective prefill work, routing decisions. [^4_1] |
| Batching caps | $B_{tok}, B_{req}$ | Worker scheduler | Iteration batch composition and GPU utilization. [^4_1] |
| Tensor parallelism | $TP$ | Within a worker | Per-op compute split + extra comm. (modeled as collectives). [^4_1] |
| Data parallelism | $DP$ | Cross-workers / within node | Replication + DP-aware routing + DP-attention patterns. [^4_1] |
| Expert parallelism | $EP$ | MoE layers | All-to-all traffic + expert capacity constraints. [^4_1] |

## Event-driven simulator core (what to code first)

Use a discrete-event engine, but make GPU progress happen in **iteration “ticks”** so batching decisions are explicit and you can reproduce SGLang-style “batch scheduler runs ahead / overlaps CPU+GPU” as an overhead term.[^4_1]

**Core data structures**

- `Request`: `{id, prompt_tokens, max_new_tokens, arrival_time, sampling_params, stream, prefix_group_id(optional)}`.
- `PrefillTask`: `{req_id, remaining_prompt_tokens, chunk_size=c, produced_kv_bytes, prefix_match_len}`.[^4_1]
- `DecodeTask`: `{req_id, remaining_new_tokens, kv_handle, decode_state}`.
- `KVHandle`: logical KV object with `{bytes, location(prefill_pool_id, decode_pool_id), transfer_status}` so P/D transfer is modeled explicitly.

**Minimal event set**

- `RequestArrives(router)` → router enqueues.
- `RouterDispatchToPrefill(worker)` → creates/attaches `PrefillTask`.
- `PrefillIterationStart(worker)` → scheduler selects batch of chunks.[^4_1]
- `PrefillChunkDone(worker, req)` → update prefix cache + KV produced.[^4_1]
- `KVTransferStart(prefill→decode)` / `KVTransferDone` → network+serialization modeled; decode can’t start until KV arrives.
- `RouterDispatchToDecode(worker)` → creates/attaches `DecodeTask`.
- `DecodeIterationStart(worker)` → scheduler selects decode batch (plus any interleaving you choose).[^4_1]
- `TokenEmitted(req)` / `RequestDone(req)` → metrics.

This lets P/D ratio show up naturally: if $N_d$ is small, decode queues build; if $N_p$ is small, TTFT grows; if network is slow, KV transfer dominates.

## Prefix cache + cache-aware routing (must be “real”)

Implement prefix caching as two coupled models: (1) per-prefill-worker **prefix cache state**, and (2) router’s **approximate radix tree per worker** used for cache-aware decisions.[^4_1]

**Prefix cache state (worker-side)**

- `RadixCache`: store prefix nodes with `{token_span, kv_bytes, last_access_time}` and enforce capacity + eviction.
- For each arriving request, compute `prefix_match_ratio = matched_tokens / prompt_tokens` and `matched_tokens` to decide “hit path” vs “miss path”.
- Eviction policy: approximate LRU with an eviction interval and max tree size (mirror router’s idea of periodic eviction).

**Router cache-aware routing (router-side)**

- Maintain an approximate radix tree per worker, as described by SGLang’s cache-aware load balancing design (approximate tree, lazily updated).[^4_1]
- Implement the documented decision logic:
    - If balanced: route to highest prefix match if match exceeds `cache_threshold`, else route to most available cache capacity.
    - If imbalanced: route to least busy (shortest queue) worker.

In PD mode, do this twice: prefill policy chooses prefill worker based on prefix match; decode policy can be round-robin or load-based as configured.

## Parallelism modeling (TP/DP/EP) inside workers

Treat “parallelism” as a **group topology + extra collective ops** added to the operator graph for each iteration.[^4_1]

**Tensor Parallelism (TP)**

- Inside each worker instance, represent a `TPGroup` of size $TP$ with a per-layer cost model:
    - compute cost scaled down by $TP$,
    - plus communication cost for required collectives (e.g., all-reduce / all-gather) inserted at the right points.[^4_1]
- In the simulator, those collectives become explicit “comm kernels” whose latency depends on message size and interconnect type (NVLink vs network).[^4_1]

**Data Parallelism (DP)**

- At the router level, DP is “many replicas” and is supported explicitly (router replaces/coordinates DP-size style deployments and even has DP-aware routing).
- Additionally, model SGLang’s DP-attention pattern (notably for DeepSeek) where attention is done per-DP worker and then results are all-gathered before MoE and redistributed after MoE.[^4_1]
- Implementation: in your per-iteration operator graph, add:
    - `AllGather(attn_output)` before MoE,
    - `Scatter/Redistribute(after MoE)`, with message sizes derived from hidden dims and batch tokens.[^4_1]

**Expert Parallelism (EP)**

- For MoE layers, model EP as:
    - gating dispatch → **all-to-all** of token activations to expert owners,
    - expert compute,
    - **all-to-all** return.[^4_1]
- Even if EP details evolve, this all-to-all skeleton matches the “communication-heavy MoE” reality and aligns with SGLang’s direction of integrating expert parallelism for MoE throughput.[^4_1]


## Symbolic throughput \& constraints (SymPy-ready)

Make every latency primitive return both `(numeric_value, sympy_expr)` so you can build end-to-end formulas that include loops (decode tokens), chunk loops (prefill), and piecewise branches (kernel choice, cache hit/miss, balance vs cache-aware routing).[^4_1]

**What becomes symbolic vs frozen**

- Symbolic decision vars: $N_p, N_d, c, B_{tok}, TP, DP, EP$ (and any router thresholds you want to optimize).
- Frozen (measured from simulation or workload): prefix-match distribution per request group, cache eviction “effective hit rate,” network RTT/bw, etc., because router/queue evolution is stateful.[^4_1]

**Throughput construction (practical form)**

- Define throughput over horizon $T$: `TP = total_emitted_tokens / T` as a SymPy expression parameterized by the above vars, where `total_emitted_tokens` comes from loop counts and `T` is sum/max of iteration costs depending on overlap assumptions.[^4_1]
- Router-induced branching is expressed with `Piecewise` (balanced vs imbalanced) and cache hit/miss branching is expressed with `Piecewise` keyed by prefix match ratio thresholds.

If you confirm (1) target interconnect assumptions (intra-node NVLink vs inter-node Ethernet/IB) and (2) whether decode workers ever do prefill fallback (mixed mode), the next step is to pin down the exact operator DAG templates and the minimal set of collectives for TP/DP/EP so the first runnable simulator already produces meaningful TTFT/TPOT/throughput curves under PD mode.[^4_1]

<div align="center">⁂</div>

[^4_1]: https://github.com/llm-d/llm-d-inference-sim

