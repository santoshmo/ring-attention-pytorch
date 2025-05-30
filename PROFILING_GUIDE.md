# Tree Attention Profiling Guide

This guide explains how to use and interpret the profiling capabilities added to the tree attention implementation.

## Quick Start

```bash
# Basic profiling test
python assert_tree_attn.py --world-size 4 --seq-len 1024 --enable-profiling

# CUDA profiling with more iterations for better statistics
python assert_tree_attn.py --world-size 4 --seq-len 2048 --use-cuda --enable-profiling --profile-iterations 20

# Run multiple profiling scenarios
python profile_examples.py
```

## Understanding the Output

### Sample Output Breakdown

```
[Rank 0] Profiling Results (all times in ms):
============================================================
COMPUTATION:
  comp_shard_kv            :    0.034 ±  0.029 (total:    0.102)
  comp_local_attention     :    7.950 ±  0.844 (total:   23.851)
  comp_num_den             :    0.025 ±  0.003 (total:    0.074)
  comp_final_output        :    0.034 ±  0.031 (total:    0.101)

COMMUNICATION:
  comm_max_lse             :    4.754 ±  2.607 (total:   14.262)
  comm_sum_den             :    1.162 ±  1.122 (total:    3.486)
  comm_sum_num             :    0.546 ±  0.271 (total:    1.637)

SUMMARY:
  Total Computation  :   67.801 ms ( 77.8%)
  Total Communication:   19.386 ms ( 22.2%)
  Total Time         :   87.186 ms
```

### Computation Components

- **`comp_shard_kv`**: Time to shard key/value tensors across processes
- **`comp_local_attention`**: Core attention computation (most expensive)
  - Uses FlashAttention on CUDA or standard PyTorch operations on CPU
- **`comp_num_den`**: Computing numerator and denominator for final aggregation
- **`comp_final_output`**: Final division and type conversion

### Communication Components

- **`comm_max_lse`**: All-reduce to find maximum log-sum-exp across all processes
  - Small data transfer (just the LSE values)
- **`comm_sum_den`**: All-reduce to sum denominators across processes
- **`comm_sum_num`**: All-reduce to sum numerators across processes
  - Largest data transfer (full attention outputs)

### Performance Metrics

- **Mean ± Variance**: Average time and variability across iterations
- **Total**: Sum of all iterations
- **Computation %**: Percentage of time spent on local computation
- **Communication %**: Percentage of time spent on inter-process communication

## Optimization Insights

### Compute-Bound vs Communication-Bound

**Good scaling potential (Compute-bound):**
```
Total Computation  : 85.2% 
Total Communication: 14.8%
```
- Most time spent on local computation
- Adding more processes will likely improve performance
- Communication overhead is manageable

**Limited scaling potential (Communication-bound):**
```
Total Computation  : 35.1%
Total Communication: 64.9%
```
- Most time spent on communication
- Adding more processes may hurt performance
- Consider larger sequences per process or optimization of communication

### Scaling Analysis

Compare results across different configurations:

1. **Increase world size** (more processes):
   - Computation time should decrease (less work per process)
   - Communication time should increase (more processes to coordinate)

2. **Increase sequence length**:
   - Computation time should increase (more work per process)
   - Communication time should remain relatively stable

3. **CUDA vs CPU**:
   - CUDA should show higher computation percentage (faster local compute)
   - CPU may be more communication-bound

## Common Patterns

### Optimal Configuration
- Computation: 70-85%
- Communication: 15-30%
- Low variance in timings

### Communication Bottleneck
- Computation: <50%
- Communication: >50%
- High variance in `comm_*` timings
- **Solution**: Increase sequence length per process

### Memory/Bandwidth Limitation
- High `comm_sum_num` relative to other communication
- **Solution**: Consider gradient checkpointing or different tensor layouts

## Troubleshooting

### High Variance in Timings
- Run more iterations: `--profile-iterations 20`
- Check for background processes
- Ensure consistent hardware setup

### Unexpected Communication Costs
- Verify network topology and bandwidth
- Check if processes are on same node vs distributed
- Consider NCCL optimizations for CUDA

### Low Computation Percentage
- Increase `--seq-len` to give more work per process
- Reduce `--world-size` if communication dominates
- Check if hardware is underutilized

## Advanced Usage

### Custom Profiling in Code

```python
from assert_tree_attn import Timer, profiled_tree_attn_decode

# Create timer
timer = Timer(use_cuda=torch.cuda.is_available())

# Profile your attention call
with timer.time_block("my_attention"):
    output = profiled_tree_attn_decode(q, k, v, timer)

# Print results
timer.print_stats()
```

### Profiling Different Scenarios

```python
# Compare different sequence lengths
for seq_len in [512, 1024, 2048, 4096]:
    # Run profiling and compare computation/communication ratios
    
# Compare different world sizes  
for world_size in [2, 4, 8, 16]:
    # Analyze scaling characteristics
```

## References

- [Tree Attention Paper](https://arxiv.org/abs/2408.04093)
- [Ring Attention Paper](https://arxiv.org/abs/2310.01889)
- [Flash Attention](https://arxiv.org/abs/2205.14135) 