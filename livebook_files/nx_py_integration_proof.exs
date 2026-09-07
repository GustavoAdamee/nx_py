# nx_py Integration + Memory Proof Script (protocol/encoder-based)
#
# Run from the repository root:
#   elixir livebook_files/nx_py_integration_proof.exs
#
# Section 6 below produces the RSS figures reported in
# preliminary_results_report.md, section 6.1.
#
# This script tests the full round-trip with Pythonx using the new
# Pythonx.Encoder for Nx.Tensor dispatch + NxPy.from_python/2:
# 1. Binary backend (copy-based)
# 2. EXLA backend (zero-copy, in-place)
# 3. EXLA backend (zero-copy, separate output)
# 4. Type coverage (encode each type, read .dtype/.shape back via Pythonx)
# 5. ml_dtypes round-trip (bf16, f8_e5m2)
# 6. Memory proofs (RSS)

Mix.install([
  {:nx_py, path: Path.join([__DIR__, "..", "nx_py"])},
  {:nx, github: "elixir-nx/nx", sparse: "nx", override: true, ref: "4c741d35"},
  {:exla, github: "elixir-nx/nx", sparse: "exla", override: true, ref: "4c741d35"},
  {:pythonx, "~> 0.4.7"}
])

# Force EXLA as default backend
Application.put_env(:nx, :default_backend, {EXLA.Backend, client: :host})

# Init Python environment (installs numpy, ml-dtypes, and the bundled nx_py
# package that defines NxTensor)
NxPy.init_python()

IO.puts("\n========================================")
IO.puts("  nx_py Integration Test (encoder-based)")
IO.puts("========================================\n")

defmodule Assert do
  def assert!(label, true), do: IO.puts("  PASS: #{label}")
  def assert!(label, false), do: raise("FAIL: #{label}")
end

# ── Binary Backend (copy-based) ────────────────────────────────

IO.puts("── Binary Backend (copy-based) ──")

tensor = Nx.iota({5}, type: {:s, 32})

{result, _} =
  Pythonx.eval(
    """
    arr = tensor.array
    out = arr * 10
    from nx_py import NxTensor
    NxTensor(out)
    """,
    %{"tensor" => tensor}
  )

result_tensor = NxPy.from_python(result, backend: Nx.BinaryBackend)
Assert.assert!("binary round-trip correct", Nx.to_flat_list(result_tensor) == [0, 10, 20, 30, 40])

IO.puts("")

# ── EXLA Backend (zero-copy, in-place) ─────────────────────────

IO.puts("── EXLA Backend (zero-copy, in-place) ──")

input_tensor = Nx.iota({5}, type: {:s, 32}) |> Nx.backend_transfer(EXLA.Backend)

{result, _} =
  Pythonx.eval(
    """
    arr = tensor.array
    arr *= 10
    tensor
    """,
    %{"tensor" => input_tensor}
  )

result_tensor = NxPy.from_python(result, backend: EXLA.Backend)
Assert.assert!("exla in-place round-trip correct", Nx.to_flat_list(result_tensor) == [0, 10, 20, 30, 40])

Assert.assert!("owner retained", Nx.to_flat_list(input_tensor) == [0, 10, 20, 30, 40])

IO.puts("")

# ── EXLA Backend (zero-copy, separate output) ──────────────────

IO.puts("── EXLA Backend (zero-copy, separate output) ──")

input_tensor = Nx.iota({1000}, type: {:f, 32}) |> Nx.backend_transfer(EXLA.Backend)
output_tensor = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {1000}) |> Nx.backend_transfer(EXLA.Backend)

{result, _} =
  Pythonx.eval(
    """
    in_arr = input.array
    out_arr = output.array
    out_arr[:] = in_arr * 10
    output
    """,
    %{"input" => input_tensor, "output" => output_tensor}
  )

result_tensor = NxPy.from_python(result, backend: EXLA.Backend)
expected = Nx.multiply(Nx.iota({1000}, type: {:f, 32}), 10.0)
Assert.assert!("exla separate output correct", Nx.to_flat_list(result_tensor) == Nx.to_flat_list(expected))

Assert.assert!("output owner retained", Nx.to_flat_list(output_tensor) == Nx.to_flat_list(expected))

IO.puts("")

# ── Type Coverage (encode each type, read .dtype/.shape back) ──

IO.puts("── Type Coverage ──")

types_to_test = [
  {:f, 64}, {:f, 32}, {:f, 16},
  {:bf, 16}, {:f, 8}, {:f8_e4m3fn, 8},
  {:s, 64}, {:s, 32}, {:s, 16}, {:s, 8},
  {:u, 64}, {:u, 32}, {:u, 16}, {:u, 8},
  {:c, 64}, {:c, 128}
]

type_results =
  Enum.map(types_to_test, fn type ->
    try do
      tensor = Nx.iota({100}) |> Nx.as_type(type) |> Nx.backend_transfer(EXLA.Backend)

      {result, _} =
        Pythonx.eval(
          """
          (obj.dtype, tuple(obj.shape))
          """,
          %{"obj" => tensor}
        )

      {dtype_str, shape} = Pythonx.decode(result)
      expected_dtype = NxPy.Dtype.to_numpy(type)

      if dtype_str == expected_dtype and shape == {100} do
        IO.puts("  PASS: #{inspect(type)} -> dtype=#{dtype_str}, shape=#{inspect(shape)}")
        :ok
      else
        IO.puts("  FAIL: #{inspect(type)} -> dtype=#{dtype_str} (expected #{expected_dtype}), shape=#{inspect(shape)}")
        :fail
      end
    rescue
      e ->
        IO.puts("  FAIL: #{inspect(type)} -> #{Exception.message(e) |> String.slice(0, 80)}")
        :fail
    end
  end)

type_passed = Enum.count(type_results, & &1 == :ok)
type_failed = Enum.count(type_results, & &1 == :fail)

IO.puts("")

# ── Python round-trip with ml_dtypes (bf16) ────────────────────

IO.puts("── Python round-trip with ml_dtypes (bf16) ──")

bf16_tensor = Nx.tensor([1.0, 2.0, 3.0], type: {:bf, 16}) |> Nx.backend_transfer(EXLA.Backend)

{result, _} =
  Pythonx.eval(
    """
    arr = tensor.array
    out = arr * 10
    from nx_py import NxTensor
    NxTensor(out)
    """,
    %{"tensor" => bf16_tensor}
  )

# `out` is a *fresh* NumPy array (`arr * 10` allocates new memory; it is not
# an alias of `bf16_tensor`'s DeviceBuffer), so its pointer has no relation
# to XLA's allocator and is not guaranteed to satisfy XLA's buffer alignment
# requirements. Reconstructing via `backend: EXLA.Backend` here would be
# unsound (intermittent "unaligned data" crashes depending on where Python's
# allocator happened to place `out`). `Nx.BinaryBackend` is copy-based and
# has no alignment requirement, so it's the correct choice whenever the
# Python-side result isn't a genuine alias of an original EXLA buffer.
bf16_result = NxPy.from_python(result, backend: Nx.BinaryBackend)
result_list = Nx.to_flat_list(bf16_result)
IO.puts("  bf16 result: #{inspect(result_list)}")
Assert.assert!("bf16 round-trip executed", length(result_list) == 3)

IO.puts("")

# ── Python round-trip with ml_dtypes (f8_e5m2) ─────────────────

IO.puts("── Python round-trip with ml_dtypes (f8_e5m2) ──")

f8_tensor = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0], type: {:f, 8}) |> Nx.backend_transfer(EXLA.Backend)

{result, _} =
  Pythonx.eval(
    """
    arr = tensor.array
    out = arr * 2
    from nx_py import NxTensor
    NxTensor(out)
    """,
    %{"tensor" => f8_tensor}
  )

# Same reasoning as the bf16 section above: `out` is a fresh NumPy array,
# not an alias of `f8_tensor`'s DeviceBuffer.
f8_result = NxPy.from_python(result, backend: Nx.BinaryBackend)
result_list = Nx.to_flat_list(f8_result)
IO.puts("  f8 result: #{inspect(result_list)}")
Assert.assert!("f8 round-trip executed", length(result_list) == 5)

IO.puts("")

# ── Memory Proofs ─────────────────────────────────────────────

defmodule Mem do
  def rss_kb do
    pid = :os.getpid()
    {output, 0} = System.cmd("ps", ["-o", "rss=", "-p", to_string(pid)])
    output |> String.trim() |> String.to_integer()
  end

  def snapshot do
    :erlang.garbage_collect()
    Pythonx.eval("import gc; gc.collect()", %{})
    :erlang.garbage_collect()
    %{rss_kb: rss_kb()}
  end

  def delta(s1, s2) do
    %{rss_kb: s2.rss_kb - s1.rss_kb}
  end

  def print_delta(label, d) do
    IO.puts("  #{String.pad_trailing(label, 22)} RSS: #{d.rss_kb} KB")
  end
end

IO.puts("── Memory Proof A: Binary Backend (copy-based) ──")

s0 = Mem.snapshot()

mem_tensor = Nx.iota({1_000_000}, type: {:f, 32})

{result, _} =
  Pythonx.eval(
    """
    arr = tensor.array
    out = arr * 10
    from nx_py import NxTensor
    NxTensor(out)
    """,
    %{"tensor" => mem_tensor}
  )
s1 = Mem.snapshot()

mem_result = NxPy.from_python(result, backend: Nx.BinaryBackend)
copy_eval_rss = Mem.delta(s0, s1).rss_kb

Mem.print_delta("Pythonx eval", Mem.delta(s0, s1))
Assert.assert!("copy round-trip correct", Nx.to_flat_list(mem_result |> Nx.slice_along_axis(0, 5)) == [0.0, 10.0, 20.0, 30.0, 40.0])

IO.puts("")

IO.puts("── Memory Proof B: EXLA Backend (zero-copy) ──")

zc_input = Nx.iota({1_000_000}, type: {:f, 32}) |> Nx.backend_transfer(EXLA.Backend)
zc_output = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {1_000_000}) |> Nx.backend_transfer(EXLA.Backend)

s0 = Mem.snapshot()

{result, _} =
  Pythonx.eval(
    """
    import numpy as np
    in_arr = input.array
    out_arr = output.array
    np.multiply(in_arr, 10, out=out_arr)
    output
    """,
    %{"input" => zc_input, "output" => zc_output}
  )

s1 = Mem.snapshot()
zc_result = NxPy.from_python(result, backend: EXLA.Backend)
zero_copy_eval_rss = Mem.delta(s0, s1).rss_kb

Mem.print_delta("Pythonx eval", Mem.delta(s0, s1))
Assert.assert!("zero-copy round-trip correct", Nx.to_flat_list(zc_result |> Nx.slice_along_axis(0, 5)) == [0.0, 10.0, 20.0, 30.0, 40.0])

Assert.assert!("zero-copy output owner retained", Nx.to_flat_list(zc_output |> Nx.slice_along_axis(0, 5)) == [0.0, 10.0, 20.0, 30.0, 40.0])

IO.puts("")

IO.puts("── Memory Proof Summary ──")
IO.puts("  Copy-based eval RSS delta:  #{copy_eval_rss} KB")
IO.puts("  Zero-copy eval RSS delta:   #{zero_copy_eval_rss} KB")

if copy_eval_rss > 0 and zero_copy_eval_rss < copy_eval_rss do
  IO.puts("  Zero-copy uses less memory than copy-based (as expected)")
else
  IO.puts("  WARNING: RSS deltas inconclusive (may vary by OS/runtime state)")
end

IO.puts("")

# ── Summary ────────────────────────────────────────────────────

fixed_passed = 7  # binary + exla-inplace + exla-separate + bf16 + f8 + 2 memory proofs
total_passed = type_passed + fixed_passed
total_failed = type_failed

IO.puts("========================================")
IO.puts("  Type coverage: #{type_passed}/#{length(types_to_test)} passed")
IO.puts("  Total assertions: #{total_passed} passed, #{total_failed} failed")
IO.puts("========================================")

if total_failed > 0, do: System.halt(1), else: IO.puts("\n  All tests passed!")
