defmodule NxPy do
  @moduledoc """
  Plug-and-play interoperability between Nx tensors and Python.

  Tensors can be passed directly as `Pythonx.eval` globals — the
  `Pythonx.Encoder` implementation for `Nx.Tensor` (see `NxPy.Encoder`)
  handles the per-backend conversion automatically. `from_python/2`
  reconstructs an `Nx.Tensor` from an `nx_py.NxTensor` Python object
  returned by Python code.

  ## Quick start

      # Initialize the bundled Python environment (installs numpy, ml-dtypes,
      # and the nx_py package that defines NxTensor)
      NxPy.init_python()

      # Elixir -> Python: any Nx.Tensor works directly as a Pythonx.eval global
      tensor = Nx.iota({5}, type: {:s, 32}) |> Nx.backend_transfer(EXLA.Backend)

      {result, _} = Pythonx.eval(
        \"\"\"
        arr = tensor.array          # tensor is an nx_py.NxTensor; arr is a real numpy array
        arr *= 10                   # in-place: writes straight into the EXLA DeviceBuffer
        tensor
        \"\"\",
        %{\"tensor\" => tensor}
      )

      updated = NxPy.from_python(result, backend: EXLA.Backend)
      consume(updated)

      # Referencing or returning `tensor` after the last use of `updated`
      # keeps its DeviceBuffer alive for the whole operation.
      {updated, tensor}
  """

  @default_python_version "==3.13.*"

  @doc """
  Initializes the Python environment via `Pythonx.uv_init/1`.

  Installs NumPy, ml-dtypes, and the bundled `nx_py` Python package (which
  defines the `NxTensor` class the encoder imports). The bundled package is
  referenced as a local editable path dependency, resolved at runtime via
  `Application.app_dir(:nx_py, "priv/python")` — works whether `nx_py` is a
  path, git, or Hex dependency.

  ## Options

    * `:extra_deps` — additional pip dependency strings (default `[]`)
    * `:python_version` — Python version constraint (default `"==3.13.*"`)

  ## Examples

      NxPy.init_python()
      NxPy.init_python(extra_deps: ["scipy==1.12.0"])
      NxPy.init_python(python_version: "==3.12.*")
  """
  def init_python(opts \\ []) do
    Pythonx.uv_init(build_toml(opts))
  end

  @doc false
  def build_toml(opts \\ []) do
    py_src_path = Application.app_dir(:nx_py, "priv/python")
    extra_deps = Keyword.get(opts, :extra_deps, [])
    python_version = Keyword.get(opts, :python_version, @default_python_version)

    dependencies = ["nx_py @ #{py_src_path}" | extra_deps]

    deps_str = Enum.map_join(dependencies, ",\n", &~s(  "#{&1}"))

    """
    [project]
    name = "nx_py_project"
    version = "0.0.0"
    requires-python = "#{python_version}"
    dependencies = [
    #{deps_str}
    ]

    [tool.uv.sources]
    nx_py = { path = "#{py_src_path}", editable = true }

    [tool.uv]
    reinstall-package = ["nx_py"]
    """
  end

  @doc """
  Encodes a tensor for Python. Thin convenience over `Pythonx.encode!/1`.

  In practice you don't need this — passing the tensor directly as a
  `Pythonx.eval` global dispatches through the `Pythonx.Encoder` impl
  automatically. Kept for explicitness and symmetry with `from_python/2`.
  """
  def to_python(tensor) do
    Pythonx.encode!(tensor)
  end

  @doc """
  Reconstructs an `Nx.Tensor` from an `nx_py.NxTensor` Python object.

  Calls `NxTensor.to_nx/1` via a single `Pythonx.eval` + `Pythonx.decode/1`,
  then rebuilds the tensor through the chosen backend.

  ## Options

    * `:backend` — reconstruction backend (defaults to `Nx.default_backend/0`).
      `Nx.BinaryBackend` is copy-based; pointer-capable backends are zero-copy.

  `obj` does **not** need to be the same object originally sent to Python —
  any `NxTensor` instance works, since all needed metadata lives on the
  object itself.

  ## GC safety (zero-copy path)

  The reconstructed tensor aliases the memory of the original buffer.
  The original tensor must outlive not just the `eval` but also any use of
  the reconstructed tensor — otherwise the reconstructed tensor reads freed
  memory. This is inherent to zero-copy aliasing.

  Merely keeping the original tensor lexically in scope is **not enough**.
  Once it is no longer referenced, the BEAM may collect its `DeviceBuffer`:

      def gc_example(original) do
        reconstructed = round_trip_through_python(original)
        # `original` is no longer referenced and may be collected from here.
        consume(reconstructed)
      end

  Reference the original after the last use of the reconstructed tensor. For
  example, returning both values keeps the original reachable for the whole
  operation:

      def gc_safe_example(original) do
        reconstructed = round_trip_through_python(original)
        result = consume(reconstructed)
        {result, original}
      end

  """
  def from_python(%Pythonx.Object{} = obj, opts \\ []) do
    backend = Keyword.get(opts, :backend, Nx.default_backend())
    backend_module = backend_module(backend)

    cond do
      backend_module == Nx.BinaryBackend ->
        {binary, dtype_str, shape} = read_fields(obj, "binary")
        type = NxPy.Dtype.from_numpy(dtype_str)
        binary |> Nx.from_binary(type) |> Nx.reshape(shape)

      function_exported?(backend_module, :from_pointer, 5) ->
        {pointer, data_size, dtype_str, shape} = read_fields(obj, "pointer")
        type = NxPy.Dtype.from_numpy(dtype_str)
        ptr = %Nx.Pointer{kind: :local, address: pointer, data_size: data_size}
        Nx.from_pointer(backend, ptr, type, shape)

      true ->
        raise ArgumentError,
              "backend #{inspect(backend)} cannot reconstruct a tensor from a pointer"
    end
  end

  defp read_fields(obj, mode) do
    {result, %{}} =
      Pythonx.eval(
        "obj.to_nx(mode.decode())",
        %{"obj" => obj, "mode" => mode}
      )

    Pythonx.decode(result)
  end

  defp backend_module({backend, _opts}), do: backend
  defp backend_module(backend), do: backend
end
