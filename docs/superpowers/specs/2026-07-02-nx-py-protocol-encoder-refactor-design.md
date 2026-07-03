# Design: nx_py — Protocol/Encoder-Based Refactor

**Date:** 2026-07-02
**Scope:** Replace `nx_py`'s config + behaviour-based architecture (`NxPy.Config`, `NxPy.Backend`, `NxPy.PythonLib`) with a protocol/encoder-based design, per advisor feedback from the 2026-07-02 supervisor call.

---

## Context

The current `nx_py` (see `docs/superpowers/specs/2026-06-07-nx-py-mix-library-design.md`) requires an explicit `NxPy.configure(backend: ..., python_lib: ...)` step, and exposes `to_python/1` / `from_python/2` that operate on a hand-rolled Elixir "reference map" (`%{pointer:, data:, dtype:, shape:, tensor_ref:, mode:}`). Users must manually reconstruct a NumPy array from that map inside `Pythonx.eval` using `ctypes`.

The advisor's feedback pushes toward using Pythonx's own extension points instead of nx_py's own config/behaviour layer:

- **`Pythonx.Encoder`** is a protocol Pythonx already dispatches on for every value passed through `Pythonx.eval`/`Pythonx.encode!`. Implementing it `for: Nx.Tensor` means any tensor can be passed directly into Python code — no manual ref map, no `ctypes` boilerplate for the caller.
- Nx already exposes the concrete backend as a struct on `tensor.data` (`%Nx.BinaryBackend{}`, `%EXLA.Backend{}`), so backend-specific logic can live as a `case` inside a single encoder implementation instead of a separately configured `NxPy.Backend` module.
- A small Python package bundled with `nx_py` (installed as a local editable `uv` dependency, following the advisor's `pyproject.toml`/`tool.uv.sources` snippet) can define a known `NxTensor` class, so the Elixir side always knows what shape of object Python hands back on `from_python`.
- With both of the above in place, there's no runtime state left to configure — `NxPy.Config` can go away.

This refactor keeps the thesis focus on **interoperability**, not performance: zero-copy remains available for `EXLA.Backend`, but the point of this change is a cleaner, more idiomatic Elixir↔Python boundary.

---

## Public API (after refactor)

```elixir
# Initialize the bundled Python environment (unchanged entry point, new internals)
NxPy.init_python()

# Elixir -> Python: any Nx.Tensor can be passed directly as a Pythonx.eval global
tensor = Nx.iota({5}, type: {:s, 32}) |> Nx.backend_transfer(EXLA.Backend)

{result, _} = Pythonx.eval(
  """
  arr = tensor.array          # tensor is an nx_py.NxTensor, arr is a real numpy array
  arr *= 10                   # in-place: writes straight into the EXLA DeviceBuffer
  tensor
  """,
  %{"tensor" => tensor}
)

# Python -> Elixir: reconstruct from whatever NxTensor Python hands back
updated = NxPy.from_python(result, backend: EXLA.Backend)
```

No `NxPy.configure/1` call, no `:type`/`:shape` opts — `from_python/2` reads shape/dtype off the returned `NxTensor` itself. `backend:` selects the reconstruction path (defaults to `Nx.BinaryBackend`, i.e. a copy).

`NxPy.to_python/1` is kept only as a thin, documented convenience (`defdelegate to_python(tensor), to: Pythonx, as: :encode!`), since `Pythonx.Encoder` dispatch makes it unnecessary in practice — tensors just work as `Pythonx.eval` globals.

---

## Architecture

```
nx_py/
  lib/
    nx_py.ex                    # init_python/1, to_python/1 (thin), from_python/2
    nx_py/encoder.ex            # defimpl Pythonx.Encoder, for: Nx.Tensor
    nx_py/dtype.ex              # Nx type <-> numpy dtype string, both directions
  priv/
    python/
      pyproject.toml            # the bundled package's own project file (numpy, ml-dtypes deps)
      nx_py/
        __init__.py
        nx_tensor.py             # class NxTensor
```

`NxPy.Config`, `NxPy.Backend` (+ `Binary`/`EXLA` impls), and `NxPy.PythonLib` (+ `Numpy` impl) are deleted. `NxPy.Application` was only ever starting the `Config` Agent, so it's deleted too, along with the `mod: {NxPy.Application, []}` entry in `mix.exs` — `nx_py` no longer needs to be a registered OTP application.

---

## Bundled Python package

```python
# priv/python/nx_py/nx_tensor.py
class NxTensor:
    def __init__(self, array):
        self.array = array

    @property
    def pointer(self):
        return self.array.ctypes.data

    @property
    def dtype(self):
        return str(self.array.dtype)

    @property
    def shape(self):
        return self.array.shape
```

```toml
# priv/python/pyproject.toml
[project]
name = "nx_py"
version = "0.0.0"
requires-python = ">= 3.13"
dependencies = ["numpy==2.2.2", "ml-dtypes==0.5.0"]

[tool.setuptools]
packages = ["nx_py"]
```

`NxPy.init_python/1` builds the *consumer-side* `pyproject.toml` passed to `Pythonx.uv_init/1`, referencing the bundled package as a local editable path dependency (resolved via `Application.app_dir(:nx_py, "priv/python")` at runtime, so it works whether `nx_py` is a path, git, or Hex dependency):

```elixir
def init_python(opts \\ []) do
  py_src_path = Application.app_dir(:nx_py, "priv/python")

  Pythonx.uv_init("""
  [project]
  name = "nx_py_project"
  version = "0.0.0"
  requires-python = "#{python_version(opts)}"
  dependencies = ["nx_py @ #{py_src_path}"]

  [tool.uv.sources]
  nx_py = { path = "#{py_src_path}", editable = true }

  [tool.uv]
  reinstall-package = ["nx_py"]
  """)
end
```

`:extra_deps` (e.g. `scipy`) stay supported by appending to `dependencies`, same as today.

---

## `to_python` — `Pythonx.Encoder` for `Nx.Tensor`

```elixir
defimpl Pythonx.Encoder, for: Nx.Tensor do
  def encode(tensor, _encoder) do
    case tensor.data do
      %EXLA.Backend{} ->
        NxPy.Encoder.encode_exla(tensor)

      %Nx.BinaryBackend{} ->
        NxPy.Encoder.encode_binary(tensor)

      other ->
        raise Protocol.UndefinedError,
          protocol: @protocol,
          value: tensor,
          description: "nx_py does not support the #{inspect(other.__struct__)} backend yet"
    end
  end
end
```

- **`EXLA.Backend`** — sub-byte integers (`{:s,4}`, `{:s,2}`, `{:u,4}`, `{:u,2}`) are widened to 8-bit first (same `maybe_widen` logic as today), then `Nx.to_pointer(tensor, mode: :local)` gives an address; a `Pythonx.eval` call builds a NumPy array view over that address via `ctypes`/`frombuffer` and wraps it in `nx_py.NxTensor` — zero-copy, mutations in Python are visible on the DeviceBuffer.
- **`Nx.BinaryBackend`** — `Nx.to_binary/1` produces bytes; `Pythonx.eval` builds a NumPy array via `frombuffer(...).copy()` and wraps it in `NxTensor` — copy-based.
- **other backends** — raise clearly, naming the offending struct.

`NxPy.Dtype` replaces `NxPy.PythonLib.Numpy`, keeping the same 20-type Nx↔NumPy mapping table, but as a plain module (no behaviour) since NumPy is now the only, hardcoded target.

---

## `from_python/2`

```elixir
def from_python(%Pythonx.Object{} = obj, opts \\ []) do
  backend = Keyword.get(opts, :backend, Nx.BinaryBackend)

  case backend do
    EXLA.Backend ->
      {pointer, data_size, dtype_str, shape} = read_pointer_fields(obj)
      type = NxPy.Dtype.from_numpy(dtype_str)
      ptr = %Nx.Pointer{kind: :local, address: pointer, data_size: data_size}
      Nx.from_pointer(EXLA.Backend, ptr, type, List.to_tuple(shape))

    Nx.BinaryBackend ->
      {binary, dtype_str, shape} = read_binary_fields(obj)
      type = NxPy.Dtype.from_numpy(dtype_str)
      binary |> Nx.from_binary(type) |> Nx.reshape(List.to_tuple(shape))
  end
end
```

`read_pointer_fields/1` and `read_binary_fields/1` each do a single `Pythonx.eval` call against `obj` (e.g. `"(obj.pointer, obj.array.nbytes, obj.dtype, tuple(obj.shape))"`), then `Pythonx.decode/1` the natively-decodable tuple result (ints/str/tuple — all supported by `Pythonx.decode/1` today). No custom Pythonx decoder is required.

`obj` does **not** need to be the same object originally sent to Python — any `NxTensor` instance works, since all needed metadata lives on the object itself. This is a behavior change from today's implementation (which assumed the same DeviceBuffer was reused) and is intentionally more general.

---

## GC safety

Unchanged constraint, just no longer carried via an explicit `tensor_ref` field: for the zero-copy `EXLA.Backend` path, the original Elixir `tensor` variable must stay in scope for the duration of the `Pythonx.eval` call that encodes it (normal Elixir variable scoping across the call achieves this — no extra bookkeeping needed, since `Pythonx.eval`'s globals-encoding step runs synchronously against the tensor value the caller passed in). This is documented as a usage contract, same as before.

---

## Non-goals / follow-ups

- Narrowing sub-byte types (`{:s,4}`/`{:s,2}`/etc.) back on `from_python` when the returned array is still 8-bit-widened is not addressed in this refactor — the caller can `Nx.as_type/2` manually. Matches the advisor's "can expand in the future" framing for the `NxTensor` class.
- Torchx/EMLX backends are not implemented now — same `Protocol.UndefinedError` fallback as today's behaviour-based dispatch, extendable later by adding another `case` clause.
- IPC mode, remote/FLAME evaluation are out of scope, as before.

---

## Migration impact

- `README.md`'s quick-start and "Extending" sections need rewriting around the new API (no more `NxPy.configure/1`, no ref-map/`ctypes` example).
- `test/nx_py/config_test.exs`, `backend/*_test.exs`, `python_lib/numpy_test.exs` are replaced by tests for `NxPy.Encoder`/`Nx.Tensor` Pythonx.Encoder impl, `NxPy.Dtype`, and `NxPy.from_python/2`.
- `test/nx_py/integration_test.exs` and `test_integration.exs` (round-trip test) are updated to the new API shape.
- Existing Livebook notebooks under `livebook_files/` that use the old ref-map approach are historical POC artifacts (per project memory) and are **not** in scope to update as part of this refactor — they predate `nx_py` as a library.
