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

      # Python -> Elixir: reconstruct from whatever NxTensor Python hands back
      updated = NxPy.from_python(result, backend: EXLA.Backend)

      # `tensor` isn't referenced again above, so without this line the BEAM
      # could free its DeviceBuffer before `updated` (which aliases the same
      # memory) is read. See "GC safety" on `from_python/2` below.
      send(self(), {:keep_alive, tensor})
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

    deps_str =
      dependencies
      |> Enum.map(&~s(  "#{&1}"))
      |> Enum.join(",\n")

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

  Reads `.pointer`, `.dtype`, and `.shape` off the object via a single
  `Pythonx.eval` + `Pythonx.decode/1`, then rebuilds the tensor through
  the chosen backend.

  ## Options

    * `:backend` — `EXLA.Backend` (zero-copy, aliases the original buffer) or
      `Nx.BinaryBackend` (default; copy-based).

  `obj` does **not** need to be the same object originally sent to Python —
  any `NxTensor` instance works, since all needed metadata lives on the
  object itself.

  ## GC safety (EXLA path)

  The reconstructed EXLA tensor aliases the memory of the original buffer.
  The original tensor must outlive not just the `eval` but also any use of
  the reconstructed tensor — otherwise the reconstructed tensor reads freed
  memory. This is inherent to zero-copy aliasing.

  Merely keeping the original tensor lexically in scope is **not enough**:
  if it is never referenced again after being sent to Python, the BEAM's
  liveness analysis can consider it dead and collect its `DeviceBuffer`
  before the reconstructed tensor is read, even while still "in scope" of
  the same function. The caller must add an explicit reference after the
  last use of the reconstructed tensor, for example:

      send(self(), {:keep_alive, original_tensor})

  """
  def from_python(%Pythonx.Object{} = obj, opts \\ []) do
    backend = Keyword.get(opts, :backend, Nx.BinaryBackend)

    case backend do
      EXLA.Backend ->
        {pointer, data_size, dtype_str, shape} = read_pointer_fields(obj)
        type = NxPy.Dtype.from_numpy(dtype_str)
        ptr = %Nx.Pointer{kind: :local, address: pointer, data_size: data_size}
        Nx.from_pointer(EXLA.Backend, ptr, type, shape)

      Nx.BinaryBackend ->
        {binary, dtype_str, shape} = read_binary_fields(obj)
        type = NxPy.Dtype.from_numpy(dtype_str)
        binary |> Nx.from_binary(type) |> Nx.reshape(shape)
    end
  end

  defp read_pointer_fields(obj) do
    {result, %{}} =
      Pythonx.eval(
        "(obj.pointer, obj.array.nbytes, obj.dtype, tuple(obj.shape))",
        %{"obj" => obj}
      )

    Pythonx.decode(result)
  end

  defp read_binary_fields(obj) do
    {result, %{}} =
      Pythonx.eval(
        "(obj.array.tobytes(), obj.dtype, tuple(obj.shape))",
        %{"obj" => obj}
      )

    Pythonx.decode(result)
  end
end
