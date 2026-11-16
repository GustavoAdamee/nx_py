# Automated LLDB zero-copy proof (Section 3.8 of the thesis).
#
#   elixir livebook_files/lldb_zero_copy_auto.exs [output.log]
#
# The script attaches LLDB to ITS OWN BEAM process (`lldb --batch -p <pid>`)
# and inspects the EXLA buffer address directly, as an observer that is
# neither Elixir nor Python. No sudo and no DEBUG build of EXLA are needed:
# only raw memory reads/writes at a known address are used, no breakpoints.
#
# Phases (one buffer, three observers):
#   1. Elixir creates s32 [1, 2, 3, 4, 5] on EXLA host and gets its address.
#      LLDB reads the address.
#   2. Python receives the tensor through nx_py (zero-copy view) and writes
#      arr[0] = 99 in place. LLDB reads the SAME address -> sees 99. The
#      address NumPy reports (arr.ctypes.data) is the one Elixir obtained.
#   3. LLDB writes 77 at element [4] from outside both runtimes. Elixir reads
#      the ORIGINAL tensor and NumPy reads its view -> both see 99 and 77.
#
# The whole session (commands + LLDB output) is printed and, when an output
# path is given, saved for the thesis.

Mix.install([
  {:nx_py, path: Path.join([__DIR__, "..", "nx_py"])},
  {:nx, github: "elixir-nx/nx", sparse: "nx", override: true, ref: "4c741d35"},
  {:exla, github: "elixir-nx/nx", sparse: "exla", override: true, ref: "4c741d35"},
  {:pythonx, "~> 0.4.7"}
])

NxPy.init_python()

defmodule Session do
  def start, do: Agent.start_link(fn -> [] end, name: __MODULE__)

  def puts(line \\ "") do
    IO.puts(line)
    Agent.update(__MODULE__, &[line | &1])
  end

  def lines, do: __MODULE__ |> Agent.get(& &1) |> Enum.reverse()

  # Runs LLDB attached to this very process. While LLDB is attached the BEAM
  # is stopped; it resumes on detach and System.cmd/3 returns.
  def lldb(commands) do
    args =
      ["--batch", "-p", System.pid()] ++
        Enum.flat_map(commands, &["-o", &1]) ++ ["-o", "process detach"]

    puts("$ lldb --batch -p #{System.pid()} " <> Enum.map_join(commands, " ", &~s(-o "#{&1}")))
    {out, status} = System.cmd("lldb", args, stderr_to_stdout: true)
    if status != 0, do: raise("lldb exited with #{status}:\n#{out}")

    # Keep only the command echo and the memory dump; drop attach banners and
    # the thread backtrace LLDB prints when it stops the process.
    out
    |> String.split("\n")
    |> Enum.filter(&(String.starts_with?(&1, "(lldb)") or String.starts_with?(&1, "0x")))
    |> Enum.each(&puts("  " <> &1))
  end
end

Session.start()

{:ok, lldb_version} =
  case System.cmd("lldb", ["--version"]) do
    {v, 0} -> {:ok, v |> String.split("\n") |> hd()}
  end

Session.puts("# nx_py zero-copy proof with LLDB as external observer")
Session.puts("# #{lldb_version}; Elixir #{System.version()}; OTP #{System.otp_release()}; " <>
  "EXLA #{Application.spec(:exla, :vsn)}; Pythonx #{Application.spec(:pythonx, :vsn)}; " <>
  "BEAM pid #{System.pid()}")
Session.puts()

# --- Phase 1 ---------------------------------------------------------------
tensor = Nx.tensor([1, 2, 3, 4, 5], type: {:s, 32}, backend: {EXLA.Backend, client: :host})
%{address: address, data_size: size} = Nx.to_pointer(tensor, mode: :local)
hex = "0x" <> Integer.to_string(address, 16)
read = "memory read --format d --size 4 --count 5 #{hex}"

Session.puts("## 1. Elixir: tensor EXLA s32 #{inspect(Nx.to_flat_list(tensor))}")
Session.puts("Nx.to_pointer -> address #{hex}, data_size #{size}")
Session.lldb([read])
Session.puts()

# --- Phase 2 ---------------------------------------------------------------
obj = Pythonx.encode!(tensor)

{result, %{}} =
  Pythonx.eval(
    """
    arr = obj.array
    arr[0] = 99
    (arr.ctypes.data, arr.flags["OWNDATA"], arr.tolist())
    """,
    %{"obj" => obj}
  )

{py_address, owndata, py_values} = Pythonx.decode(result)

Session.puts("## 2. Python: arr = obj.array; arr[0] = 99")
Session.puts("arr.ctypes.data = 0x#{Integer.to_string(py_address, 16)} " <>
  "(same as Elixir: #{py_address == address}); OWNDATA = #{owndata}; arr = #{inspect(py_values)}")
Session.lldb([read])
Session.puts()

# --- Phase 3 ---------------------------------------------------------------
Session.puts("## 3. LLDB writes 77 at element [4] (#{hex} + 16)")
Session.lldb(["memory write --size 4 --format d #{hex}+16 77", read])

elixir_view = Nx.to_flat_list(tensor)
{result, %{}} = Pythonx.eval("obj.array.tolist()", %{"obj" => obj})
python_view = Pythonx.decode(result)

Session.puts("Elixir (original tensor): #{inspect(elixir_view)}")
Session.puts("Python (obj.array):       #{inspect(python_view)}")

expected = [99, 2, 3, 4, 77]
ok = py_address == address and owndata == false and elixir_view == expected and python_view == expected
Session.puts()
Session.puts("RESULT: #{if ok, do: "PASS", else: "FAIL"} - one buffer, three observers")

case System.argv() do
  [out] -> File.write!(out, Enum.join(Session.lines(), "\n") <> "\n")
  [] -> :ok
end

_ = {tensor, obj}
unless ok, do: System.halt(1)
