# nx_py

Zero-copy tensor interoperability between [Numerical Elixir (Nx)](https://github.com/elixir-nx/nx)
and Python, built on [Pythonx](https://github.com/livebook-dev/pythonx).

This repository holds the library and the verification artefacts of an undergraduate
capstone thesis (TCC) in Computer Engineering at UTFPR, Campus Curitiba.

| Path | Contents |
|------|----------|
| [`nx_py/`](nx_py/) | The library (Mix project) — see its [README](nx_py/README.md) for usage, the backend compatibility matrix, the type table and the memory-lifetime rules |
| [`livebook_files/`](livebook_files/) | Verification scripts and the demo notebook (below) |
| [`livebook_files/results/`](livebook_files/results/) | Data measured for the thesis |
| [`docs/research/`](docs/research/) | Research notes (in Portuguese) |

## Verification scripts

Each script installs its own dependencies with `Mix.install` and is run from the repository root.

| Script | What it does |
|--------|--------------|
| `nx_py_integration_proof.exs` | End-to-end round trips on both transfer paths and type coverage with a real Python interpreter |
| `rss_repetitions.exs` | Memory (RSS) measurement protocol: spawns `rss_measure.exs` once per run in a fresh VM and writes `results/rss.csv` |
| `lldb_zero_copy_auto.exs` | Uses LLDB as an external observer to show that Elixir, Python and the debugger read and write the same buffer; writes `results/lldb_zero_copy.log` when given an output path |
| `nx_py_demo.livemd` | Livebook notebook demonstrating the library |

```bash
elixir livebook_files/nx_py_integration_proof.exs
elixir livebook_files/rss_repetitions.exs
elixir livebook_files/lldb_zero_copy_auto.exs livebook_files/results/lldb_zero_copy.log
```

The library tests run from inside `nx_py/` with `mix test`.

## Acknowledgments

Advisor: **Adolfo Neto**. Co-advisor: **Paulo Valente** ([@polvalente](https://github.com/polvalente)).

## License

[MIT](nx_py/LICENSE).
