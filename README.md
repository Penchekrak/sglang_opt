with 

```python
    N_p: Symbol = Symbol("N_p", positive=True, integer=True)  # num prefill workers
    N_d: Symbol = Symbol("N_d", positive=True, integer=True)  # num decode workers

    # Scheduler configuration
    chunk_size: Symbol = Symbol("c", positive=True, integer=True)
    batch_cap_tokens: Symbol = Symbol("B_tok", positive=True, integer=True)
    batch_cap_requests: Symbol = Symbol("B_req", positive=True, integer=True)

    # Parallelism
    tp_size: Symbol = Symbol("TP", positive=True, integer=True)
    dp_size: Symbol = Symbol("DP", positive=True, integer=True)
    ep_size: Symbol = Symbol("EP", positive=True, integer=True)

    # Hardware
    gpu_flops: Symbol = Symbol("F_gpu", positive=True)  # FLOP/s
    gpu_mem_bw: Symbol = Symbol("B_mem", positive=True)  # bytes/s
    gpu_mem_cap: Symbol = Symbol("M_gpu", positive=True)  # bytes
    nvlink_bw: Symbol = Symbol("B_nvlink", positive=True)  # bytes/s
    network_bw: Symbol = Symbol("B_net", positive=True)  # bytes/s
    network_latency: Symbol = Symbol("L_net", positive=True)  # seconds

    # Model
    num_layers: Symbol = Symbol("L", positive=True, integer=True)
    hidden_dim: Symbol = Symbol("H", positive=True, integer=True)
    head_dim: Symbol = Symbol("d", positive=True, integer=True)
    num_heads: Symbol = Symbol("n_h", positive=True, integer=True)
    num_experts: Symbol = Symbol("E", positive=True, integer=True)
    top_k_experts: Symbol = Symbol("k", positive=True, integer=True)
    kv_bytes_per_token: Symbol = Symbol("kv_b", positive=True)

    # Cache
    cache_capacity: Symbol = Symbol("C_kv", positive=True)
    block_size: Symbol = Symbol("B_blk", positive=True, integer=True)

    # Frozen params (measured during simulation)
    cache_hit_rate: Symbol = Symbol("pi_hit", positive=True)
    avg_prompt_len: Symbol = Symbol("n_in_avg", positive=True)
    avg_output_len: Symbol = Symbol("n_out_avg", positive=True)
    avg_prefix_match: Symbol = Symbol("prefix_match", positive=True)
```

being defined and 

```python
    constraints.add_memory_constraint(80 * 1024**3)  # 80GB H100
    constraints.add_ttft_slo(0.5)  # 500ms TTFT
    constraints.add_tpot_slo(0.05)  # 50ms TPOT
    constraints.add_total_gpu_constraint(16)
```

one has sympy expressions as follows:

```
metrics.throughput:
 Min(B_mem*B_req*N_d/(2*L*d*n_h*(2*n_in_avg + n_out_avg)), F_gpu*N_p*TP*n_out_avg/(L*n_in_avg*(8*H**2 + 4*d*n_h*n_in_avg)))
memory_limit:
 -4*B_req*L*d*n_h*(n_in_avg + n_out_avg)/TP - 8*H*c + 77309411328.0 - (128000*H + 2*L*(12*E*H**2 + 4*H**2))/TP > 0
ttft_slo:
 0.5 - Max(4*L*d*n_h*Min(c, n_in_avg*(-pi_hit*prefix_match + 1))/B_mem, L*(8*H**2 + 4*d*n_h*n_in_avg)*Min(c, n_in_avg*(-pi_hit*prefix_match + 1))/(F_gpu*TP)) > 0
tpot_slo:
 0.05 - Max(L*(8*H**2 + 4*d*n_h*(n_in_avg + n_out_avg/2))/(F_gpu*TP), 2*L*d*n_h*(2*n_in_avg + n_out_avg)/B_mem)/B_req > 0
max_gpus:
 -TP*(N_d + N_p) + 16 > 0
```
