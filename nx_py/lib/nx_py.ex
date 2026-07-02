defmodule NxPy do
  @moduledoc """
  Plug-and-play interoperability between Nx tensors and Python.

  ## Quick start

      # Configure at startup
      NxPy.configure(backend: NxPy.Backend.Binary, python_lib: NxPy.PythonLib.Numpy)

      # Convert tensor to Python reference
      ref = NxPy.to_python(tensor)

      # Use ref fields (pointer, data, dtype, shape, data_size) in Pythonx.eval
      # ...

      # Reconstruct tensor from Python
      tensor = NxPy.from_python(ref, type: {:s, 32}, shape: {5})
  """

  alias NxPy.Config

  @base_deps ["numpy==2.2.2", "ml-dtypes==0.5.0"]
  @default_python_version "==3.13.*"

  @doc """
  Initializes the Python environment via `Pythonx.uv_init/1`.

  Sets up a Python project with NumPy and ml-dtypes as base dependencies.

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
    extra_deps = Keyword.get(opts, :extra_deps, [])
    python_version = Keyword.get(opts, :python_version, @default_python_version)
    all_deps = @base_deps ++ extra_deps

    deps_str = all_deps |> Enum.map(&~s(  "#{&1}")) |> Enum.join(",\n")

    """
    [project]
    name = "nx_py_project"
    version = "0.0.0"
    requires-python = "#{python_version}"
    dependencies = [
    #{deps_str}
    ]
    """
  end

  @doc """
  Configures the nx_py library at runtime.

  ## Options

    * `:backend` — module implementing `NxPy.Backend` (e.g. `NxPy.Backend.EXLA`, `NxPy.Backend.Binary`)
    * `:python_lib` — module implementing `NxPy.PythonLib` (e.g. `NxPy.PythonLib.Numpy`)

  ## Examples

      NxPy.configure(backend: NxPy.Backend.EXLA, python_lib: NxPy.PythonLib.Numpy)
  """
  def configure(opts) do
    backend = Keyword.fetch!(opts, :backend)
    python_lib = Keyword.fetch!(opts, :python_lib)
    Config.set(%{backend: backend, python_lib: python_lib})
  end

  @doc """
  Converts an Nx tensor into a Python reference map.

  Returns a map containing everything Python needs to access the tensor data:

    * `:pointer` — raw memory address (zero-copy), or `nil` (copy-based)
    * `:data` — binary data (copy-based), or `nil` (zero-copy)
    * `:data_size` — size of the data in bytes
    * `:dtype` — Python dtype string (e.g. `"float32"`, `"int32"`)
    * `:shape` — tensor shape as a list
    * `:tensor_ref` — reference to the original tensor (keeps DeviceBuffer alive)
    * `:mode` — `:zero_copy` or `:copy_based`

  ## Examples

      ref = NxPy.to_python(tensor)
      # Use ref.pointer and ref.dtype with ctypes.from_address in Python
  """
  def to_python(tensor) do
    config = Config.get()
    dtype = config.python_lib.nx_to_dtype(tensor)
    ref = config.backend.to_python_ref(tensor)
    Map.put(ref, :dtype, dtype)
  end

  @doc """
  Reconstructs an Nx tensor from a Python reference map.

  ## Options

    * `:type` — the Nx type of the result tensor (e.g. `{:s, 32}`, `{:f, 64}`)
    * `:shape` — the shape of the result tensor (e.g. `{5}`, `{2, 3}`)

  ## Examples

      tensor = NxPy.from_python(ref, type: {:s, 32}, shape: {5})
  """
  def from_python(ref, opts) do
    config = Config.get()
    type = Keyword.fetch!(opts, :type)
    shape = Keyword.fetch!(opts, :shape)
    config.backend.from_python_ref(ref, type, shape)
  end
end
