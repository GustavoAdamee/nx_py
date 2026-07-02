defmodule NxPy.IntegrationTest do
  use ExUnit.Case

  setup do
    NxPy.Config.set(%{backend: NxPy.Backend.Binary, python_lib: NxPy.PythonLib.Numpy})
    :ok
  end

  describe "full round-trip with Binary backend" do
    test "s32 tensor round-trip produces correct result" do
      tensor = Nx.iota({100}, type: {:s, 32})
      ref = NxPy.to_python(tensor)

      # Verify ref fields are usable for Python interop
      assert is_binary(ref.data)
      assert ref.dtype == "int32"
      assert ref.shape == [100]
      assert ref.data_size == 400

      # Simulate Python computation: multiply by 10
      # In real usage, the user would pass ref.data to Pythonx.eval
      result_tensor = Nx.multiply(tensor, 10)
      result_data = Nx.to_binary(result_tensor)

      # User updates ref.data with Python result
      ref = %{ref | data: result_data}
      result = NxPy.from_python(ref, type: {:s, 32}, shape: {100})

      assert Nx.to_flat_list(result) == Nx.to_flat_list(Nx.multiply(tensor, 10))
    end

    test "f64 2D tensor round-trip" do
      tensor = Nx.iota({10, 10}, type: {:f, 64})
      ref = NxPy.to_python(tensor)

      assert ref.dtype == "float64"
      assert ref.shape == [10, 10]
      assert ref.data_size == 10 * 10 * 8

      result = NxPy.from_python(ref, type: {:f, 64}, shape: {10, 10})
      assert Nx.to_flat_list(result) == Nx.to_flat_list(tensor)
    end

    test "bf16 tensor has correct dtype and ml_dtypes flag" do
      tensor = Nx.iota({5}, type: {:bf, 16})
      ref = NxPy.to_python(tensor)

      assert ref.dtype == "bfloat16"
      assert NxPy.PythonLib.Numpy.requires_ml_dtypes?(tensor) == true
    end
  end
end
