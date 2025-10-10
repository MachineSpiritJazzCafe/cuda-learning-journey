# Vector Reduction

## Concepts Mastered
- Parallel Reduction Patterns
- Tree-Based Algorithm with Stride Doubling
- Thread Synchronization (`__syncthreads()`)
- Multi-Stage Reduction for Large Arrays
- GPU/CPU Hybrid Strategy Selection
- Global Memory Performance Bottlenecks
- Thread Divergence in Reduction Patterns

## Versions

### v1-naive-reduction
**Current Implementation**: Basic tree-based reduction with global memory

**Key Features:**
- Uses increasing stride pattern (1, 2, 4, 8...)
- Operates directly on global memory (slow: 400-800 cycle latency)
- Requires `__syncthreads()` to prevent race conditions
- Each block reduces to single value at `input[blockIdx.x * blockDim.x]`

**Multi-Stage Strategies:**

**GPU-Only Approach** (`naiveReduceGPU`):
- Launches multiple kernels until result fits in one block
- Example: 1M elements → 3,907 → 16 → 1 (3 stages)
- Minimal data transfer (4 bytes final result)
- Best for: Many blocks (>128)

**Hybrid GPU+CPU** (`naiveReduceHybrid`):
- Single GPU kernel produces partial sums
- CPU finishes remaining reduction
- Transfers partial results to host
- Best for: Few blocks (≤128)

**Smart Selection** (`smartReduce`):
- Automatically chooses strategy based on problem size
- Threshold: 128 blocks
- Balances kernel launch overhead vs data transfer cost

**Performance Characteristics:**
- Thread utilization decreases each stride (50% → 25% → 12.5%...)
- Thread divergence: `tid % (2 * stride)` causes warp inefficiency
- Memory bandwidth limited by global memory access
- Time complexity: O(log N) steps, but slow per step

**Limitations:**
- No shared memory usage (10-20x slower than optimized version)
- Poor thread divergence pattern
- All operations hit DRAM instead of on-chip cache
- Sequential addressing not optimized for memory coalescing

**Next Optimizations:**
- Use shared memory for intermediate values
- Implement sequential addressing to reduce divergence
- Unroll final loop iterations
- Use warp shuffle primitives for last 32 elements

## Build & Run
```bash
make help (for build, test and clean)
```
