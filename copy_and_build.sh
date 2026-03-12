@ -0,0 +1,52 @@
#!/bin/bash

# Script to copy llama.cpp to remote PC and build with ET backend

REMOTE_USER="root"
REMOTE_HOST="aifoundry1"
REMOTE_PATH="/home/saqib/llamaCpp/llama.cpp"
LOCAL_PATH="/home/saqib/Documents/Prj/P1/ET_platform/llama_original/llama.cpp"

echo "Copying llama.cpp to remote PC..."

# Create remote directory if it doesn't exist
ssh ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_PATH}"

# Copy files using rsync for efficiency
rsync -avz --progress \
    --exclude='.git' \
    --exclude='build' \
    --exclude='*.o' \
    --exclude='*.so' \
    --exclude='*.a' \
    --exclude='bin/' \
    --exclude='__pycache__/' \
    ${LOCAL_PATH}/ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/

echo "Files copied successfully!"
echo "Building on remote PC..."

# # Run the build commands on remote PC
# ssh ${REMOTE_USER}@${REMOTE_HOST} << 'EOF'
# cd /home/saqib/llamaCpp/llama.cpp

# echo "Configuring with CMake..."
# cmake -B build \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DGGML_ET=ON \
#     -DGGML_ET_SYSEMU=OFF \
#     -DGGML_CUDA=OFF \
#     -DGGML_VULKAN=OFF \
#     -DGGML_RPC=OFF \
#     -DGGML_BLAS=OFF \
#     -DLLAMA_CURL=OFF \
#     -DGGML_SCHED_MAX_COPIES=1 \
#     -DGGML_CCACHE=OFF

# echo "Building llama-cli..."
# cmake --build build --config Release -j $(nproc) -t llama-cli

# echo "Build completed!"
# EOF

# echo "Process completed. Check the output above for any errors."