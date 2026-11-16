# Runs the RSS measurement protocol for the thesis (Section 3.8) and writes a CSV.
#
#   elixir livebook_files/rss_repetitions.exs [output.csv] [n_runs] [n_elements]
#
# Defaults: livebook_files/results/rss.csv, 10 runs, 1_000_000 f32 elements
# (~3.8 MiB). For each path (binary, exla): 1 warm-up run (run 0, flagged in
# the CSV and excluded from the summary) + n_runs measured runs. Every run is a
# separate `elixir rss_measure.exs` process, i.e. a fresh BEAM + Python heap.
# Paths alternate run by run so slow drift in the machine state affects both
# equally.
#
# Also writes <output>.env.csv with the software/hardware versions.

script = Path.join(__DIR__, "rss_measure.exs")

{out, runs, n} =
  case System.argv() do
    [] -> {Path.join([__DIR__, "results", "rss.csv"]), 10, 1_000_000}
    [o] -> {o, 10, 1_000_000}
    [o, r] -> {o, String.to_integer(r), 1_000_000}
    [o, r, e] -> {o, String.to_integer(r), String.to_integer(e)}
  end

File.mkdir_p!(Path.dirname(out))

run = fn args ->
  {output, status} = System.cmd("elixir", [script | args], stderr_to_stdout: true)

  if status != 0 do
    IO.puts(output)
    raise "rss_measure.exs #{Enum.join(args, " ")} exited with #{status}"
  end

  output
  |> String.split("\n")
  |> Enum.filter(&String.starts_with?(&1, ["CSV,", "ENV,"]))
  |> Enum.map(fn line -> line |> String.split(",", parts: 2) |> List.last() end)
end

IO.puts("Collecting environment...")
env = run.(["--env"])
File.write!(Path.rootname(out) <> ".env.csv", Enum.join(["item,valor" | env], "\n") <> "\n")

rows =
  for i <- 0..runs, path <- ["binary", "exla"] do
    label = if i == 0, do: "warm-up", else: "run #{i}/#{runs}"
    IO.puts("#{path} #{label}")

    run.([path, Integer.to_string(i), Integer.to_string(n)])
    |> Enum.map(&(&1 <> "," <> to_string(i == 0)))
  end
  |> List.flatten()

header = "caminho,execucao,elementos,fase,rss_antes_kb,rss_depois_kb,delta_kb,aquecimento"
File.write!(out, Enum.join([header | rows], "\n") <> "\n")
IO.puts("\nWrote #{out}")

# Summary (warm-up excluded): median and min-max of delta per path and phase,
# plus the per-run total over the three phases.
parsed =
  for row <- rows,
      [path, run_id, _n, phase, _b, _a, delta, "false"] <- [String.split(row, ",")] do
    {path, String.to_integer(run_id), phase, String.to_integer(delta)}
  end

median = fn xs ->
  s = Enum.sort(xs)
  len = length(s)
  if rem(len, 2) == 1, do: Enum.at(s, div(len, 2)), else: (Enum.at(s, div(len, 2) - 1) + Enum.at(s, div(len, 2))) / 2
end

tensor_kb = div(n * 4, 1024)
IO.puts("\nTensor size: #{tensor_kb} KB  (#{n} x f32)\n")
IO.puts(String.pad_trailing("path/phase", 16) <> String.pad_leading("median KB", 12) <> String.pad_leading("min..max KB", 20))

for path <- ["binary", "exla"] do
  per_phase =
    for phase <- ["encode", "op", "decode"] do
      {phase, for({^path, _, ^phase, d} <- parsed, do: d)}
    end

  totals =
    parsed
    |> Enum.filter(&(elem(&1, 0) == path))
    |> Enum.group_by(&elem(&1, 1), &elem(&1, 3))
    |> Enum.map(fn {_run, ds} -> Enum.sum(ds) end)

  for {label, ds} <- per_phase ++ [{"TOTAL", totals}] do
    IO.puts(
      String.pad_trailing("#{path}/#{label}", 16) <>
        String.pad_leading(to_string(median.(ds)), 12) <>
        String.pad_leading("#{Enum.min(ds)}..#{Enum.max(ds)}", 20)
    )
  end
end
