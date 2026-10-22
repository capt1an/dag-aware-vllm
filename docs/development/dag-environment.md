# Development environment for the DAG/Responses fork

Use the upstream vLLM installation workflow: `uv venv` followed by
`uv pip install -e .`. Dependencies are declared by `setup.py`,
`pyproject.toml` and `requirements/`. This fork does not require `uv sync`
or provide a complete environment lockfile.

## Tested profile

The previous local inference tests used Linux x86-64, Python 3.12.13,
vLLM 0.19.1, PyTorch 2.10.0+cu129, CUDA 12.9, Transformers 5.5.3 and
Hugging Face Hub 1.10.2. GPU inference requires a compatible NVIDIA driver
and separately obtained model weights. The wheel below requires glibc 2.31
or newer. These observations are not a cross-platform compatibility claim.

## Install for Python-only development

From this repository's root, with uv installed:

```bash
uv venv --python 3.12 .venv

VLLM_USE_PRECOMPILED=1 \
VLLM_VERSION_OVERRIDE=0.19.1 \
VLLM_PRECOMPILED_WHEEL_LOCATION=https://files.pythonhosted.org/packages/78/20/f41216b79c87372a9d03175f36fa1411ee61059ce8c557d2691722ea4aae/vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl \
uv pip install --python .venv/bin/python --torch-backend=cu129 \
  --constraint requirements/dag-dev-constraints.txt \
  --build-constraint requirements/dag-dev-constraints.txt --editable .

uv pip install --python .venv/bin/python -r requirements/lint.txt
.venv/bin/pre-commit install
uv pip check --python .venv/bin/python
```

This uses local editable Python source with the official 0.19.1 native
extensions. Pinning the wheel location avoids inadvertently using a newer
nightly binary. PyTorch versions remain those specified by upstream
`requirements/cuda.txt`; `--torch-backend=cu129` selects their CUDA variant.
The additional constraints retain the Transformers/Hub compatibility used
by the Codex integration. Other transitive dependencies are not fully locked.

The official native wheel's SHA256 is
`71a87f46cafab4489c69a5c5c83b870d0235e5694d8222303d460576293dc719`.
If downloading it separately, verify with `sha256sum` and pass its local
path as `VLLM_PRECOMPILED_WHEEL_LOCATION`. Upstream `setup.py` does not
automatically verify that separately supplied digest.

Build isolation can download PyTorch and CUDA dependencies before editable
installation. Allow sufficient download time and disk space. Proxy and
package-index overrides belong in the caller's environment; do not commit
credentials or machine-specific network settings.

## Checks without starting model inference

```bash
.venv/bin/python -c 'import torch, vllm, vllm._C; print(vllm.__version__, torch.__version__, torch.version.cuda, vllm.__file__)'
.venv/bin/python -m unittest discover -s tests/v1/core -p test_dynamic_dag.py -v
```

The source path must point into this checkout. These checks load the Python
package/native extension and exercise DAG queue regressions without serving
a model. The earlier environment used this installation approach; a fresh
installation of the commands above has not been rerun after repository
migration. Dependency downloading is currently paused.

## C/CUDA changes and other targets

Precompiled extensions do not incorporate native source edits. For C/CUDA
development, omit the precompiled-wheel variables and follow the upstream
source-build/toolchain instructions. Consult the existing installation docs
and `requirements/` for other devices and platforms.

Commit source, requirements and these instructions. Keep `.venv`, downloaded
wheels, compiled libraries, model weights and run logs outside version control.
