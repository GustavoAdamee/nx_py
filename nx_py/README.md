# NxPy

Plug-and-play interoperability between [Nx](https://github.com/elixir-nx/nx) tensors and Python.

NxPy lets an Elixir application share tensor data with an embedded Python runtime (via [Pythonx](https://github.com/livebook-dev/pythonx)). Tensors are passed directly as `Pythonx.eval` globals — a `Pythonx.Encoder` implementation for `Nx.Tensor` handles the per-backend conversion automatically. No runtime configuration step.

> This project is part of an undergraduate capstone thesis (TCC) at UTFPR investigating Nx ↔ Python interoperability. The focus is **interoperability** — how tensor data crosses the Elixir/Python boundary — not raw performance.

## Overview

```
Nx tensor (any supported backend)
        │
        ▼  (passed as a Pythonx.eval global)
Pythonx.Encoder for Nx.Tensor  ──►  NxPy.Encoder.encode_pointer/1 (zero-copy)
                                  ──║  NxPy.Encoder.encode_binary/1 (copy-based, Binary)
                                    ║
                                    ▼
                          nx_py.NxTensor (Python object wrapping a numpy array)
                                    │
                                    ▼  (returned from Pythonx.eval)
                       NxPy.from_python/2 ──►  Nx tensor
```

Backends are dispatched by inspecting `tensor.data`:

- **Pointer-capable backends, including `EXLA.Backend`** — zero-copy. `Nx.to_pointer(tensor, mode: :local)` yields a raw address; a NumPy view is built over it via `ctypes`/`frombuffer` and wrapped in `NxTensor`. Mutations in Python are visible on the backend buffer.
- **`Nx.BinaryBackend`** — copy-based. `Nx.to_binary/1` produces bytes; `frombuffer(...).copy().reshape(shape)` builds a Python-owned NumPy array wrapped in `NxTensor`.

Backends without pointer support raise a `Protocol.UndefinedError` naming the offending struct.

A small Python package bundled with `nx_py` (installed as a local editable `uv` dependency) defines the `NxTensor` class, so the Elixir side always knows what shape of object Python hands back on `from_python/2`.

## Quick start

```elixir
# Initialize the Python environment (installs numpy, ml-dtypes, and the
# bundled nx_py package that defines NxTensor)
NxPy.init_python()

# Elixir -> Python: any Nx.Tensor works directly as a Pythonx.eval global
tensor = Nx.iota({5}, type: {:s, 32}, backend: EXLA.Backend)

{result, _} = Pythonx.eval(
  """
  arr = tensor.array          # tensor is an nx_py.NxTensor; arr is a real numpy array
  arr *= 10                   # in-place: writes straight into the EXLA DeviceBuffer
  tensor
  """,
  %{"tensor" => tensor}
)

updated = NxPy.from_python(result, backend: EXLA.Backend)
values = Nx.to_flat_list(updated)

# Returning or otherwise referencing `tensor` after the last use of `updated`
# keeps its DeviceBuffer alive throughout the operation.
{values, tensor}
```

No `NxPy.configure/1` call, no `:type`/`:shape` opts — reconstruction reads shape/dtype from the returned `NxTensor`. `backend:` selects the reconstruction path and defaults to `Nx.default_backend/0`.

`NxPy.to_python/1` is kept only as a thin, documented wrapper around `Pythonx.encode!/1`, since `Pythonx.Encoder` dispatch makes it unnecessary in practice — tensors just work as `Pythonx.eval` globals.

### Python environment setup

`init_python/1` calls `Pythonx.uv_init/1` with NumPy, ml-dtypes, and the bundled `nx_py` package (referenced as a local editable path dependency, resolved via `Application.app_dir(:nx_py, "priv/python")` at runtime):

```elixir
# Defaults: numpy==2.2.2 + ml-dtypes==0.5.0 + bundled nx_py, Python 3.13
NxPy.init_python()

# Add extra dependencies
NxPy.init_python(extra_deps: ["scipy==1.12.0"])

# Override Python version
NxPy.init_python(python_version: "==3.12.*")
```

The generated `pyproject.toml` includes a `tool.uv.sources` table and `reinstall-package = ["nx_py"]`, so edits to the bundled package are picked up on the next `init_python/1`.

## Compatibility matrix

| Backend | Pythonx.Encoder for Nx.Tensor | Reconstruction |
|---|:---:|:---:|
| `EXLA.Backend` | X (zero-copy) | X (`Nx.from_pointer/4`) |
| `Nx.BinaryBackend` | X (copy-based) | X (`Nx.from_binary/2` + `Nx.reshape/2`) |
| Other pointer-capable backend | X (zero-copy) | X (`Nx.from_pointer/4`) |
| Backend without pointer support | raises `Protocol.UndefinedError` | raises `ArgumentError` |

## Type coverage

`NxPy.Dtype` maps every Nx type to a NumPy dtype string (and back):

- **Native NumPy** — `{:f, 64|32|16}`, `{:s, 64|32|16|8}`, `{:u, 64|32|16|8}`, `{:c, 64|128}`
- **Via `ml_dtypes`** — `{:bf, 16}`, `{:f, 8}`, `{:f8_e4m3fn, 8}`, `{:s, 4|2}`, `{:u, 4|2}`

Sub-byte integers (`{:s, 2}`, `{:s, 4}`, `{:u, 2}`, `{:u, 4}`) are widened to 8-bit on the EXLA encode path before being passed to Python, because EXLA packs sub-byte values while NumPy/`ml_dtypes` stores them unpacked (1 byte per element). On the return path they arrive as `int8`/`uint8`; the caller can `Nx.as_type/2` manually if the original sub-byte type matters (documented non-goal of this refactor — may expand in the future).

## GC safety (zero-copy path)

- The original Elixir `tensor` variable must stay in scope for the duration of the `Pythonx.eval` call that encodes it (normal Elixir variable scoping across the call achieves this — no extra bookkeeping needed).
- The reconstructed tensor aliases the memory of the original buffer. The original tensor must outlive not just the `eval` but also any use of the reconstructed tensor — otherwise the reconstructed tensor reads freed memory. This is inherent to zero-copy aliasing and is called out in the `from_python/2` docs.
- **Merely being lexically in scope is not enough.** Once the original tensor is no longer referenced, the BEAM may collect its `DeviceBuffer` before the reconstructed tensor is read.
- Reference the original tensor after the last use of the reconstructed tensor. For example, return it together with the consumed result from a regular function, or retain it in `GenServer` state.
- **Known limitation (sub-byte EXLA widening):** for `{:s,4}`/`{:s,2}`/`{:u,4}`/`{:u,2}` on `EXLA.Backend`, the encoder allocates a *transient* widened `DeviceBuffer` and the NumPy view wraps that transient buffer. Because the encoder returns only a `Pythonx.Object` (no Elixir-side handle survives), that widened buffer has no BEAM reference and may be collected before the caller's `eval` body runs — a use-after-free window. This refactor **does not fix this**; non-sub-byte EXLA types are unaffected. A future revision could copy sub-byte types like the Binary path, or thread a retained handle through.

## Installation

`nx_py` is not published yet.

```elixir
def deps do
  [
    {:nx, github: "elixir-nx/nx", sparse: "nx", override: true},
    {:exla, github: "elixir-nx/nx", sparse: "exla"},
    {:pythonx, "~> 0.4.7"},
    {:nx_py, path: "../nx_py"}
  ]
end
```

Requires Elixir 1.17+ and a working `uv`/Python installation visible to Pythonx.

## Development

```bash
mix deps.get
mix compile --warnings-as-errors
mix format --check-formatted

# Pure-Elixir tests (no network):
mix test test/nx_py_test.exs test/nx_py/dtype_test.exs test/nx_py/encoder_test.exs

# Full suite (integration test needs network for uv + numpy + ml-dtypes + bundled nx_py):
mix test

# Standalone integration script (also needs network):
elixir nx_py/test_integration.exs
```

Test files mirror `lib/`:

```
lib/nx_py.ex                →  test/nx_py_test.exs           (build_toml/1 — pure Elixir)
lib/nx_py/encoder.ex        →  test/nx_py/encoder_test.exs   (structural preconditions — pure Elixir)
lib/nx_py/dtype.ex          →  test/nx_py/dtype_test.exs     (to_numpy/from_numpy — pure Elixir)
                             test/nx_py/integration_test.exs (real Pythonx — encoder + from_python)
```

## Extending

### Add a new backend

Backends that implement Nx's pointer callbacks use the zero-copy path automatically. `Nx.BinaryBackend` uses the copy-based path. A custom backend therefore only needs to implement the appropriate Nx backend callbacks; no nx_py-specific dispatch clause is required.

### Add a new Python library

This refactor removed the `NxPy.PythonLib` behaviour (NumPy is now the only, hardcoded target). If a different Python library becomes relevant, `NxPy.Dtype` is the single place to adjust (or fork into a separate module with its own `to_numpy/1`/`from_numpy/1` equivalents).

## License

TODO
