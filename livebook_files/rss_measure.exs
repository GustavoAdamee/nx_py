# nx_py RSS measurement — ONE run, ONE transfer path, in a fresh VM.
#
# Not meant to be run by hand: `rss_repetitions.exs` spawns this script once
# per repetition so every measurement starts from a clean BEAM + Python heap.
#
#   elixir livebook_files/rss_measure.exs <binary|exla> <run_id> <n_elements>
#   elixir livebook_files/rss_measure.exs --env
#
# Both paths execute the SAME Python operation — an in-place multiply on the
# received array — so the only variable is the transfer mechanism. RSS (via
# `ps`) is sampled around three phases:
#
#   encode  Elixir -> Python   (Pythonx.encode!/1 through the Pythonx.Encoder impl)
#   op      np.multiply(arr, 10, out=arr) inside Python
#   decode  Python -> Elixir   (NxPy.from_python/2)
#
# RSS is used instead of :erlang.memory/0 because NIF (EXLA) and Python
# allocations live outside the BEAM allocators and are invisible to it.
#
# Output: one line per phase on stdout, prefixed with "CSV," so the runner can
# filter them out of Mix/Pythonx noise.

Mix.install([
  {:nx_py, path: Path.join([__DIR__, "..", "nx_py"])},
  {:nx, github: "elixir-nx/nx", sparse: "nx", override: true, ref: "4c741d35"},
  {:exla, github: "elixir-nx/nx", sparse: "exla", override: true, ref: "4c741d35"},
  {:pythonx, "~> 0.4.7"}
])

NxPy.init_python()

defmodule Rss do
  def kb do
    {out, 0} = System.cmd("ps", ["-o", "rss=", "-p", System.pid()])
    out |> String.trim() |> String.to_integer()
  end

  # Collect garbage on both sides so transient objects don't pollute the
  # sample, then read RSS.
  def snapshot do
    :erlang.garbage_collect()
    Pythonx.eval("import gc; gc.collect()", %{})
    :erlang.garbage_collect()
    kb()
  end
end

case System.argv() do
  ["--env"] ->
    {result, _} =
      Pythonx.eval(
        """
        import platform, numpy, ml_dtypes
        (platform.python_version(), numpy.__version__, ml_dtypes.__version__)
        """,
        %{}
      )

    {py, np, mld} = Pythonx.decode(result)
    {os, 0} = System.cmd("sw_vers", ["-productVersion"])
    {cpu, 0} = System.cmd("sysctl", ["-n", "machdep.cpu.brand_string"])
    {mem, 0} = System.cmd("sysctl", ["-n", "hw.memsize"])

    IO.puts("""
    ENV,os,macOS #{String.trim(os)}
    ENV,cpu,#{String.trim(cpu)}
    ENV,ram_gb,#{div(String.to_integer(String.trim(mem)), 1024 * 1024 * 1024)}
    ENV,elixir,#{System.version()}
    ENV,otp,#{System.otp_release()} (ERTS #{:erlang.system_info(:version)})
    ENV,nx,#{Application.spec(:nx, :vsn)} (git ref 4c741d35)
    ENV,exla,#{Application.spec(:exla, :vsn)} (git ref 4c741d35)
    ENV,pythonx,#{Application.spec(:pythonx, :vsn)} (Hex)
    ENV,nx_py,#{Application.spec(:nx_py, :vsn)}
    ENV,python,#{py}
    ENV,numpy,#{np}
    ENV,ml_dtypes,#{mld}\
    """)

  [path, run_id, n] ->
    n = String.to_integer(n)

    backend =
      case path do
        "binary" -> Nx.BinaryBackend
        "exla" -> {EXLA.Backend, client: :host}
      end

    multiply = "import numpy as np\nnp.multiply(obj.array, 10, out=obj.array)\nNone"

    # In-VM warm-up: the first encode/eval/decode imports numpy, ml_dtypes and
    # nx_py and grows the Python heap. Do it on a tiny tensor so those one-off
    # costs are not attributed to the measured phases.
    warm = Nx.iota({8}, type: {:f, 32}, backend: backend)
    warm_obj = Pythonx.encode!(warm)
    Pythonx.eval(multiply, %{"obj" => warm_obj})
    warm_back = NxPy.from_python(warm_obj, backend: backend)

    # The tensor is created before the first sample: its allocation is the
    # same in both paths and is not part of the transfer.
    tensor = Nx.iota({n}, type: {:f, 32}, backend: backend)
    _ = Nx.to_flat_list(Nx.slice(tensor, [0], [1]))

    s0 = Rss.snapshot()
    obj = Pythonx.encode!(tensor)
    s1 = Rss.snapshot()
    Pythonx.eval(multiply, %{"obj" => obj})
    s2 = Rss.snapshot()
    back = NxPy.from_python(obj, backend: backend)
    s3 = Rss.snapshot()

    # Correctness + semantics: values came back multiplied; on the zero-copy
    # path Python's in-place write is also visible on the ORIGINAL tensor.
    [+0.0, 10.0, 20.0] = back |> Nx.slice([0], [3]) |> Nx.to_flat_list()
    original_head = tensor |> Nx.slice([0], [3]) |> Nx.to_flat_list()

    expected_original =
      case path do
        "binary" -> [0.0, 1.0, 2.0]
        "exla" -> [0.0, 10.0, 20.0]
      end

    if original_head != expected_original do
      raise "unexpected aliasing on #{path}: original head = #{inspect(original_head)}"
    end

    for {phase, before, after_} <- [{"encode", s0, s1}, {"op", s1, s2}, {"decode", s2, s3}] do
      IO.puts("CSV,#{path},#{run_id},#{n},#{phase},#{before},#{after_},#{after_ - before}")
    end

    # Keep every handle alive until the end. On the zero-copy path the Python
    # array and the decoded tensor are views of the source EXLA buffer: if the
    # source is garbage-collected first, they read freed memory (see the GC
    # safety contract in NxPy.from_python/2).
    _ = {warm, warm_obj, warm_back, tensor, obj, back}
end
