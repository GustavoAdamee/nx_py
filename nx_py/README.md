# NxPy

Plug-and-play interoperability between [Nx](https://github.com/elixir-nx/nx) tensors and Python.

NxPy lets an Elixir application share tensor data with an embedded Python runtime (via [Pythonx](https://github.com/livebook-dev/pythonx)) without forcing a particular Nx backend or Python library on you. Backends and Python library adapters are pluggable through two behaviours, so the same `NxPy.to_python/1` / `NxPy.from_python/2` API works whether you're using the copy-based `Nx.BinaryBackend` path or the zero-copy `EXLA` pointer path.

> This project is part of an undergraduate capstone thesis (TCC) at UTFPR investigating Nx ↔ Python interoperability. The focus is **interoperability** — how tensor data crosses the Elixir/Python boundary — not raw performance.

## Overview

```
Nx tensor (any backend)
        │
        ▼
NxPy.to_python/1  ──►  backend.to_python_ref/1  ──►  Python reference map
        │                                                   │
        │                                                   ▼
        │                                   Pythonx.eval reads pointer / data
        │
        ▼
NxPy.from_python/2  ──►  backend.from_python_ref/3  ──►  Nx tensor
```

Two behaviours define the contract:

- **`NxPy.Backend`** — how an Nx backend exposes tensor memory to Python and reconstructs a tensor afterwards.
- **`NxPy.PythonLib`** — how a Python library (e.g. NumPy) maps Nx types to Python dtype strings and whether it needs `ml_dtypes` for exotic types.

### Built-in backends

| Module | Mode | Mechanism |
|---|---|---|
| `NxPy.Backend.Binary` | copy-based | `Nx.to_binary/1` → Erlang binary → `Nx.from_binary/2` |
| `NxPy.Backend.EXLA` | zero-copy | `Nx.to_pointer(tensor, mode: :local)` → raw address accessed from Python via `ctypes.from_address` |

### Built-in Python library adapters

| Module | Target | Notes |
|---|---|---|
| `NxPy.PythonLib.Numpy` | NumPy | Standard types map natively; bf16, fp8, and sub-byte integers go through `ml_dtypes` |

## Quick start

```elixir
# Initialize the Python environment (numpy + ml-dtypes)
NxPy.init_python()

# Configure at startup — pick a backend and a Python library adapter
NxPy.configure(backend: NxPy.Backend.EXLA, python_lib: NxPy.PythonLib.Numpy)

# Convert a tensor into a Python reference map
ref = NxPy.to_python(tensor)

# ref fields available to Pythonx.eval:
#   :pointer    — raw memory address (zero-copy) or nil (copy-based)
#   :data       — binary (copy-based) or nil (zero-copy)
#   :data_size  — byte size of the data
#   :dtype      — Python dtype string, e.g. "float32"
#   :shape      — shape as a list
#   :tensor_ref — keeps the backing buffer alive while Python holds the ref
#   :mode       — :zero_copy | :copy_based

# Reconstruct an Nx tensor from a Python reference map
tensor = NxPy.from_python(ref, type: {:s, 32}, shape: {5})
```

### Python environment setup

`init_python/1` calls `Pythonx.uv_init/1` with NumPy and ml-dtypes as base dependencies:

```elixir
# Defaults: numpy==2.2.2 + ml-dtypes==0.5.0, Python 3.13
NxPy.init_python()

# Add extra dependencies
NxPy.init_python(extra_deps: ["scipy==1.12.0"])

# Override Python version
NxPy.init_python(python_version: "==3.12.*")
```

### Example: zero-copy with EXLA + ctypes

```python
# Inside Pythonx.eval — ref is the map returned by NxPy.to_python/1
import ctypes
arr = (ctypes.c_float * ref["data_size"]).from_address(ref["pointer"])
numpy_view = np.frombuffer(arr, dtype=np.dtype(ref["dtype"])) \
               .reshape(ref["shape"])
# numpy_view shares memory with the EXLA DeviceBuffer — no copy.
```

## Compatibility matrix

Which backend/Python-library combinations are currently supported. New rows/columns can be added as more backends and Python libraries are implemented.

| Backend \ Python library | `NxPy.PythonLib.Numpy` | ... |
|---|:---:|:---:|
| `NxPy.Backend.Binary` | X | |
| `NxPy.Backend.EXLA` | X | |
| ... | | |

## Type coverage

`NxPy.PythonLib.Numpy` maps every Nx type to a NumPy dtype:

- **Native NumPy** — `{:f, 64|32|16}`, `{:s, 64|32|16|8}`, `{:u, 64|32|16|8}`, `{:c, 64|128}`
- **Via `ml_dtypes`** — `{:bf, 16}`, `{:f, 8}`, `{:f8_e4m3fn, 8}`, `{:s, 4|2}`, `{:u, 4|2}`

Sub-byte integers (`{:s, 2}`, `{:s, 4}`, `{:u, 2}`, `{:u, 4}`) are widened to 8-bit on the EXLA backend before being passed to Python, because EXLA packs sub-byte values while NumPy/`ml_dtypes` stores them unpacked (1 byte per element). The original type is restored on the way back.

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

Requires Elixir 1.17+ and a working Python installation visible to Pythonx.

## Development

```bash
mix deps.get
mix compile --warnings-as-errors
mix format --check-formatted
mix test                                   # all tests
mix test test/nx_py/backend/exla_test.exs  # single file
mix test test/nx_py/backend/exla_test.exs:42  # single test at line
```

Test files mirror `lib/`:

```
lib/nx_py/backend/binary.ex   →  test/nx_py/backend/binary_test.exs
lib/nx_py/backend/exla.ex     →  test/nx_py/backend/exla_test.exs
lib/nx_py/python_lib/numpy.ex →  test/nx_py/python_lib/numpy_test.exs
```

An integration test (`test/nx_py/integration_test.exs`) exercises the full `to_python` → `from_python` round-trip across backends.

## Extending

### Add a new backend

Implement `NxPy.Backend`:

```elixir
defmodule MyBackend do
  @behaviour NxPy.Backend

  @impl true
  def to_python_ref(tensor) do
    # Return a map with :pointer | :data, :data_size, :shape, :tensor_ref, :mode
  end

  @impl true
  def from_python_ref(ref, type, shape) do
    # Reconstruct an Nx tensor from the ref
  end
end
```

Then configure it:

```elixir
NxPy.configure(backend: MyBackend, python_lib: NxPy.PythonLib.Numpy)
```

### Add a new Python library adapter

Implement `NxPy.PythonLib`:

```elixir
defmodule NxPy.PythonLib.MyLib do
  @behaviour NxPy.PythonLib

  @impl true
  def nx_to_dtype(%{type: {:f, 32}}), do: "float32"
  # ...other type clauses

  @impl true
  def requires_ml_dtypes?(%{type: {:bf, 16}}), do: true
  def requires_ml_dtypes?(_), do: false
end
```

## License

TODO
