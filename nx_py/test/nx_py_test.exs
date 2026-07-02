defmodule NxPyTest do
  use ExUnit.Case

  setup do
    NxPy.Config.set(%{})
    :ok
  end

  describe "build_toml/1" do
    test "generates TOML with default deps and python version" do
      toml = NxPy.build_toml()

      assert toml =~ ~s(requires-python = "==3.13.*")
      assert toml =~ ~s("numpy==2.2.2")
      assert toml =~ ~s("ml-dtypes==0.5.0")
      assert toml =~ ~s(name = "nx_py_project")
    end

    test "appends extra_deps to base deps" do
      toml = NxPy.build_toml(extra_deps: ["scipy==1.12.0", "pandas==2.2.0"])

      assert toml =~ ~s("numpy==2.2.2")
      assert toml =~ ~s("ml-dtypes==0.5.0")
      assert toml =~ ~s("scipy==1.12.0")
      assert toml =~ ~s("pandas==2.2.0")
    end

    test "overrides python version" do
      toml = NxPy.build_toml(python_version: "==3.12.*")

      assert toml =~ ~s(requires-python = "==3.12.*")
      refute toml =~ ~s(requires-python = "==3.13.*")
    end
  end

  describe "configure/1" do
    test "sets backend and python_lib" do
      NxPy.configure(backend: NxPy.Backend.Binary, python_lib: NxPy.PythonLib.Numpy)

      config = NxPy.Config.get()
      assert config.backend == NxPy.Backend.Binary
      assert config.python_lib == NxPy.PythonLib.Numpy
    end
  end

  describe "to_python/1 with Binary backend" do
    setup do
      NxPy.configure(backend: NxPy.Backend.Binary, python_lib: NxPy.PythonLib.Numpy)
      :ok
    end

    test "returns reference map with dtype, data, and copy-based mode" do
      tensor = Nx.iota({5}, type: {:s, 32})
      ref = NxPy.to_python(tensor)

      assert ref.mode == :copy_based
      assert ref.pointer == nil
      assert ref.data != nil
      assert ref.data_size == 5 * 4
      assert ref.dtype == "int32"
      assert ref.shape == [5]
      assert ref.tensor_ref == tensor
    end

    test "maps dtype via PythonLib" do
      tensor = Nx.iota({3}, type: {:f, 64})
      ref = NxPy.to_python(tensor)

      assert ref.dtype == "float64"
    end
  end

  describe "from_python/2 with Binary backend" do
    setup do
      NxPy.configure(backend: NxPy.Backend.Binary, python_lib: NxPy.PythonLib.Numpy)
      :ok
    end

    test "reconstructs tensor from reference" do
      tensor = Nx.iota({5}, type: {:s, 32})
      ref = NxPy.to_python(tensor)

      result = NxPy.from_python(ref, type: {:s, 32}, shape: {5})

      assert Nx.to_flat_list(result) == Nx.to_flat_list(tensor)
      assert result.type == {:s, 32}
      assert result.shape == {5}
    end

    test "reconstructs with modified data" do
      tensor = Nx.tensor([1, 2, 3], type: {:s, 32})
      ref = NxPy.to_python(tensor)

      # Simulate Python returning new data
      new_data = Nx.tensor([10, 20, 30], type: {:s, 32}) |> Nx.to_binary()
      ref = %{ref | data: new_data}

      result = NxPy.from_python(ref, type: {:s, 32}, shape: {3})

      assert Nx.to_flat_list(result) == [10, 20, 30]
    end
  end
end
