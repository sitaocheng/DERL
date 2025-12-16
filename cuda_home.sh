nvcc_path=$(which nvcc 2>/dev/null) || \
  nvcc_path=$(find /usr/local /opt /cpfs 2>/dev/null -path "*/bin/nvcc" -type f -executable | head -1)

if [[ -n "$nvcc_path" ]]; then
    CUDA_HOME=$(dirname "$(dirname "$nvcc_path")")
    if [[ -f "$CUDA_HOME/include/cuda.h" ]]; then
        echo "$CUDA_HOME"
    fi
fi