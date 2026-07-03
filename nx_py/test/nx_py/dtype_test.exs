defmodule NxPy.DtypeTest do
  use ExUnit.Case

  describe "to_numpy/1" do
    test "maps standard float types" do
      assert NxPy.Dtype.to_numpy({:f, 64}) == "float64"
      assert NxPy.Dtype.to_numpy({:f, 32}) == "float32"
      assert NxPy.Dtype.to_numpy({:f, 16}) == "float16"
    end

    test "maps brain float and fp8 types (ml_dtypes)" do
      assert NxPy.Dtype.to_numpy({:bf, 16}) == "bfloat16"
      assert NxPy.Dtype.to_numpy({:f, 8}) == "float8_e5m2"
      assert NxPy.Dtype.to_numpy({:f8_e4m3fn, 8}) == "float8_e4m3fn"
    end

    test "maps signed integer types" do
      assert NxPy.Dtype.to_numpy({:s, 64}) == "int64"
      assert NxPy.Dtype.to_numpy({:s, 32}) == "int32"
      assert NxPy.Dtype.to_numpy({:s, 16}) == "int16"
      assert NxPy.Dtype.to_numpy({:s, 8}) == "int8"
    end

    test "maps unsigned integer types" do
      assert NxPy.Dtype.to_numpy({:u, 64}) == "uint64"
      assert NxPy.Dtype.to_numpy({:u, 32}) == "uint32"
      assert NxPy.Dtype.to_numpy({:u, 16}) == "uint16"
      assert NxPy.Dtype.to_numpy({:u, 8}) == "uint8"
    end

    test "maps complex types" do
      assert NxPy.Dtype.to_numpy({:c, 64}) == "complex64"
      assert NxPy.Dtype.to_numpy({:c, 128}) == "complex128"
    end

    test "maps sub-byte integer types (ml_dtypes)" do
      assert NxPy.Dtype.to_numpy({:s, 4}) == "int4"
      assert NxPy.Dtype.to_numpy({:s, 2}) == "int2"
      assert NxPy.Dtype.to_numpy({:u, 4}) == "uint4"
      assert NxPy.Dtype.to_numpy({:u, 2}) == "uint2"
    end
  end

  describe "from_numpy/1" do
    test "inverts standard float types" do
      assert NxPy.Dtype.from_numpy("float64") == {:f, 64}
      assert NxPy.Dtype.from_numpy("float32") == {:f, 32}
      assert NxPy.Dtype.from_numpy("float16") == {:f, 16}
    end

    test "inverts brain float and fp8 types" do
      assert NxPy.Dtype.from_numpy("bfloat16") == {:bf, 16}
      assert NxPy.Dtype.from_numpy("float8_e5m2") == {:f, 8}
      assert NxPy.Dtype.from_numpy("float8_e4m3fn") == {:f8_e4m3fn, 8}
    end

    test "inverts signed integer types" do
      assert NxPy.Dtype.from_numpy("int64") == {:s, 64}
      assert NxPy.Dtype.from_numpy("int32") == {:s, 32}
      assert NxPy.Dtype.from_numpy("int16") == {:s, 16}
      assert NxPy.Dtype.from_numpy("int8") == {:s, 8}
    end

    test "inverts unsigned integer types" do
      assert NxPy.Dtype.from_numpy("uint64") == {:u, 64}
      assert NxPy.Dtype.from_numpy("uint32") == {:u, 32}
      assert NxPy.Dtype.from_numpy("uint16") == {:u, 16}
      assert NxPy.Dtype.from_numpy("uint8") == {:u, 8}
    end

    test "inverts complex types" do
      assert NxPy.Dtype.from_numpy("complex64") == {:c, 64}
      assert NxPy.Dtype.from_numpy("complex128") == {:c, 128}
    end

    test "sub-byte types are never returned (they arrive widened as int8/uint8)" do
      # Sub-byte integer types {:s,4}/{:s,2}/{:u,4}/{:u,2} are widened to 8-bit
      # on the encode path, so the return path only ever sees int8/uint8.
      # Caller is responsible for Nx.as_type/2 narrowing if they want the
      # original sub-byte type back (documented non-goal of this refactor).
      assert NxPy.Dtype.from_numpy("int8") == {:s, 8}
      assert NxPy.Dtype.from_numpy("uint8") == {:u, 8}
    end

    test "raises on unknown dtype string" do
      assert_raise FunctionClauseError, fn ->
        NxPy.Dtype.from_numpy("not_a_real_dtype")
      end
    end
  end
end
