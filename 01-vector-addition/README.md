# Vector Addition - CUDA Learning

## Concepts Mastered
- Thread indexing and global thread ID calculation
- Stride loops for handling large datasets
- WSL2 compatibility and environment detection
- Unified memory management
- Robust error checking patterns

## Versions

### v1-basic.cu ✅
**Current Implementation**: Basic stride pattern with unified memory
- Uses `blockIdx.x * blockDim.x + threadIdx.x` for thread indexing
- Stride loop: `for (int i = index; i < N; i += stride)`
- WSL2 safe prefetching
- Comprehensive error checking

**Compile & Run:**
```bash
make help (for build, test and clean)
```
