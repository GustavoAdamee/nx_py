defmodule NxPy.PythonLib.NumpyTest do
  use ExUnit.Case

  describe "nx_to_dtype/1" do
    test "maps standard float types" do
      assert NxPy.PythonLib.Numpy.nx_to_dtype(%{type: {:f, 64}}) == "float64"
      assert NxPy.PythonLib.Numpy.nx_to_dtype(%{type: {:f, 32}}) == "float32"
      assert NxPy.PythonLib.Numpy.nx_to_dtype(%{type: {:f, 16}}) == "float16"
    end

    test "maps brain float and fp8 types (ml_dtypes)" do
      assert NxPy.PythonLib.Numpy.nx_to_dtype(%{type: {:bf, 16}}) == "bfloat16"
      assert NxPy.PythonLib.Numpy.nx_to_dtype(%{type: {:f, 8}}) == "float8_e5m2"
      assert NxPy.PythonLib.Numpy.nx_to_dtype(%{type: {:f8_e4m3fn, 8}}) == "float8_e4m3fn"
    end

    test "maps signed integer types" do
      assert NxPy.PythonLib.Numpy.nx_to_dtype(%{type: {:s, 64}}) == "int64"
      assert NxPy.PythonLib.Numpy.nx_to_dtype(%{type: {:s, 32}}) == "int32"
      assert NxPy.PythonLib.Numpy.nx_to_dtype(%{type: {:s, 16}}) == "int16"
      assert NxPy.PythonLib.Numpy.nx_to_dtype(%{type: {:s, 8}}) == "int8"
    end

    test "maps unsigned integer types" do
      assert NxPy.PythonLib.Numpy.nx_to_dtype(%{type: {:u, 64}}) == "uint64"
      assert NxPy.PythonLib.Numpy.nx_to_dtype(%{type: {:u, 32}}) == "uint32"
      assert NxPy.PythonLib.Numpy.nx_to_dtype(%{type: {:u, 16}}) == "uint16"
      assert NxPy.PythonLib.Numpy.nx_to_dtype(%{type: {:u, 8}}) == "uint8"
    end

    test "maps complex types" do
      assert NxPy.PythonLib.Numpy.nx_to_dtype(%{type: {:c, 64}}) == "complex64"
      assert NxPy.PythonLib.Numpy.nx_to_dtype(%{type: {:c, 128}}) == "complex128"
    end

    test "maps sub-byte integer types (ml_dtypes)" do
      assert NxPy.PythonLib.Numpy.nx_to_dtype(%{type: {:s, 4}}) == "int4"
      assert NxPy.PythonLib.Numpy.nx_to_dtype(%{type: {:s, 2}}) == "int2"
      assert NxPy.PythonLib.Numpy.nx_to_dtype(%{type: {:u, 4}}) == "uint4"
      assert NxPy.PythonLib.Numpy.nx_to_dtype(%{type: {:u, 2}}) == "uint2"
    end
  end

  describe "requires_ml_dtypes?/1" do
    test "returns true for ml_dtypes types" do
      assert NxPy.PythonLib.Numpy.requires_ml_dtypes?(%{type: {:bf, 16}}) == true
      assert NxPy.PythonLib.Numpy.requires_ml_dtypes?(%{type: {:f, 8}}) == true
      assert NxPy.PythonLib.Numpy.requires_ml_dtypes?(%{type: {:f8_e4m3fn, 8}}) == true
      assert NxPy.PythonLib.Numpy.requires_ml_dtypes?(%{type: {:s, 4}}) == true
      assert NxPy.PythonLib.Numpy.requires_ml_dtypes?(%{type: {:s, 2}}) == true
      assert NxPy.PythonLib.Numpy.requires_ml_dtypes?(%{type: {:u, 4}}) == true
      assert NxPy.PythonLib.Numpy.requires_ml_dtypes?(%{type: {:u, 2}}) == true
    end

    test "returns false for native NumPy types" do
      assert NxPy.PythonLib.Numpy.requires_ml_dtypes?(%{type: {:f, 64}}) == false
      assert NxPy.PythonLib.Numpy.requires_ml_dtypes?(%{type: {:f, 32}}) == false
      assert NxPy.PythonLib.Numpy.requires_ml_dtypes?(%{type: {:s, 32}}) == false
      assert NxPy.PythonLib.Numpy.requires_ml_dtypes?(%{type: {:u, 8}}) == false
      assert NxPy.PythonLib.Numpy.requires_ml_dtypes?(%{type: {:c, 128}}) == false
    end
  end
end
