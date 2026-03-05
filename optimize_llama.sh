if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS \
            -DGGML_NATIVE=ON -DGGML_NEON=ON -DGGML_ARM_FMA=ON \
            -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF" \
.venv/bin/pip install llama-cpp-python --force-reinstall --no-cache-dir