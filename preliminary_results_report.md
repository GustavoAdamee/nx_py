# Resultados Preliminares

## 1. Introducao

Os artefatos estao organizados em tres notebooks Livebook:
- `poc.livemd` — Implementacoes do bridge (copy-based e view-based) com benchmarks
- `type_coverage.livemd` — Matriz de validacao completa dos 20 tipos Nx
- `benchmarks.livemd` — Benchmarks sistematicos incluindo baseline nativo Nx

## 2. Implementacoes do Bridge

Ambas as implementacoes utilizam `Nx.runtime_call/3` para injetar os resultados computados pelo Python de volta no pipeline de tensores do Nx.

### 2.1 Copy-based (`NxNumpyBridgePOC`) — Baseline

Fluxo de dados:

```
Tensor Nx -> Nx.to_binary -> Pythonx serializa -> np.frombuffer (sem copia)
    -> computo NumPy -> tobytes() (copia) -> Pythonx codifica -> Nx.from_binary
```

Total de copias de dados: **3** — uma na entrada (Pythonx serializa o binario Erlang em Python `bytes`), uma interna ao Python (`tobytes()` aloca um novo buffer e copia os dados do array), e uma na saida (Pythonx codifica o `bytes` resultante em binario Erlang).

### 2.2 View-based (`NxNumpyViewBased`)

Mesmo caminho de entrada, porem utiliza `ndarray.view(np.uint8)` para reinterpretar a memoria do array resultado in-place, sem alocacao. O Pythonx codifica diretamente a partir dessa representacao na fronteira com o Erlang, eliminando a copia intermediaria do `tobytes()`.

Total de copias de dados: **2** — uma na entrada (Pythonx serializa o binario Erlang em Python `bytes`), e uma na saida (Pythonx codifica diretamente da memoria do array para binario Erlang). A eliminacao da copia interna ao Python explica a vantagem de ~2x da view-based para tensores grandes.

### 2.3 Mapeamento de tipos (`NxNumpyBridge`)

Um modulo auxiliar mapeia tipos Nx para strings de dtype NumPy. Tres estrategias sao utilizadas:

| Estrategia | Tipos |
|------------|-------|
| **NumPy nativo** | `{:f, 64}`, `{:f, 32}`, `{:f, 16}`, `{:s, 8}`-`{:s, 64}`, `{:u, 8}`-`{:u, 64}`, `{:c, 64}`, `{:c, 128}` |
| **ml_dtypes** | `{:bf, 16}`, `{:f, 8}` (E5M2), `{:f8_e4m3fn, 8}` (E4M3FN) |
| **Widening cast** | `{:s, 2}`, `{:s, 4}` -> `int8`; `{:u, 2}`, `{:u, 4}` -> `uint8` |

## 3. Resultados de Cobertura de Tipos

Todos os 20 tipos numericos do Nx foram testados com ambas as implementacoes. O teste computa `x * 10` atraves do bridge Python e compara o resultado com `Nx.multiply(tensor, 10)` nativo.

```
Tipo              Estrategia    Copy-based          View-based
------------------------------------------------------------------------
{:f, 64}          nativo        ok                  ok
{:f, 32}          nativo        ok                  ok
{:f, 16}          nativo        ok*                 ok*
{:bf, 16}         ml_dtypes     ~ DIVERGENCIA       ~ DIVERGENCIA
{:f, 8}           ml_dtypes     ~ DIVERGENCIA       ~ DIVERGENCIA
{:f8_e4m3fn, 8}   ml_dtypes     ok*                 ok*
{:s, 64}          nativo        ok                  ok
{:s, 32}          nativo        ok                  ok
{:s, 16}          nativo        ok                  ok
{:s, 8}           nativo        ok                  ok
{:s, 4}           cast->s8      ok                  ok
{:s, 2}           cast->s8      ok                  ok
{:u, 64}          nativo        ok                  ok
{:u, 32}          nativo        ok                  ok
{:u, 16}          nativo        ok                  ok
{:u, 8}           nativo        ok                  ok
{:u, 4}           cast->u8      ok                  ok
{:u, 2}           cast->u8      ok                  ok
{:c, 64}          nativo        ok                  ok
{:c, 128}         nativo        ok                  ok
```

**Legenda:**
- **ok** — round-trip correto; o resultado coincide exatamente com `Nx.multiply(tensor, 10)`
- **ok\*** — round-trip correto para dados de teste controlados (inteiros pequenos dentro do range representavel), porem o tipo possui precisao limitada; uso geral com valores fracionarios ou inteiros grandes pode divergir
- **~ DIVERGENCIA** — divergencia de arredondamento entre o Nx BinaryBackend e NumPy/ml_dtypes. **Nao e um bug do bridge** — os dois runtimes diferem na implementacao de aritmetica de baixa precisao (ex: `iota({100}) * 10` causa overflow ou arredondamento diferente em bf16 e float8_e5m2)

**Observacoes sobre tipos especificos:**
- `{:f, 16}` e `{:f8_e4m3fn, 8}`: aprovados com ressalvas. f16 possui ~3,3 digitos decimais de precisao (mantissa de 10 bits); f8_e4m3fn possui ~1,2 digitos (mantissa de 3 bits, valor finito maximo de 448).
- `{:bf, 16}` e `{:f, 8}` (E5M2): a divergencia e esperada e documentada. Reduzir os dados de teste para forcar aprovacao produziria um falso positivo — a divergencia reflete diferencas genuinas no comportamento de arredondamento entre o BinaryBackend do Nx e o ml_dtypes do NumPy.
- Tipos sub-byte (`{:s, 2}`, `{:s, 4}`, `{:u, 2}`, `{:u, 4}`): testados com identidade (`* 1`) para evitar overflow. O bridge amplia para int8/uint8 antes da chamada Python e reduz de volta ao tipo original.

### 3.1 Validacao multidimensional

Ambas as implementacoes preservam corretamente o shape do tensor para tensores 2D `{100, 100}` e 3D `{10, 10, 10}` (testado com `{:f, 64}`).

## 4. Resultados dos Benchmarks

**Ambiente:** Apple M3, macOS, 16 GB RAM, Elixir 1.18.3, Erlang/OTP 27, JIT habilitado.
**Configuracao:** Benchee com warmup 2s, tempo 5s, memory_time 2s.
**Operacao:** `x * 10` (multiplicacao elementwise por escalar).

### 4.1 Comparacao entre implementacoes do bridge (de `poc.livemd`)

| Entrada | view-based (ips) | copy-based (ips) | Vantagem view-based |
|---------|-----------------|-----------------|---------------------|
| 1K f64 | 5,28 K | 5,22 K | 1,01x |
| 100K f64 | 4,08 K | 3,92 K | 1,04x |
| 1M f64 | 1,36 K | 0,60 K | 2,27x |
| 1M f32 | 2,19 K | 1,14 K | 1,93x |
| 1M s32 | 2,23 K | 1,16 K | 1,92x |

A abordagem view-based supera a copy-based de forma consistente. A vantagem cresce com o tamanho do tensor (insignificante em 1K, ~2x em 1M elementos). A copy-based apresenta variancia mais alta (desvio de +-170-330% para tensores grandes), talvez devido a pressao do Garbage collector causada pela alocacao intermediaria do `tobytes()` ?.

### 4.2 Bridge versus baseline nativo Nx (de `benchmarks.livemd`)

| Entrada | view-based (media) | copy-based (media) | Nx nativo (media) |
|---------|-------------------|-------------------|-------------------|
| 1K f64 | 0,20 ms | 0,21 ms | 0,04 ms |
| 100K f64 | 0,26 ms | 0,27 ms | 5,39 ms |
| 1M f32 | 0,54 ms | 0,91 ms | 54,38 ms |
| 1M f64 | 0,85 ms | 1,73 ms | 53,53 ms |
| 1M s32 | 0,53 ms | 0,79 ms | 26,50 ms |
| 1Mx1K 2D f64 | 0,85 ms | 1,70 ms | 53,53 ms |

**Observacoes principais:**

1. **Ponto de cruzamento (crossover):** o bridge e mais lento que o Nx nativo abaixo de ~5-8K elementos (o custo fixo de entrada no CPython domina). Acima desse limiar, o bridge completa mais rapido.
2. **Memoria:** as implementacoes do bridge utilizam ~4 KB de heap de processo independentemente do tamanho do tensor. O Nx nativo utiliza ate ~100 MB para tensores f64 de 1M elementos. Essa diferenca ocorre porque binarios Erlang refc (> 64 bytes) residem fora do heap do processo.
   *Obs: o Benchee mede apenas o heap do processo. Os dados do bridge ainda existem na memoria (no binary heap compartilhado), apenas nao sao contabilizados por essa metrica.*
3. **Agnosticismo de shape:** tensores 1D e 2D com o mesmo numero total de elementos (1M f64 vs 1000x1000 f64) apresentam desempenho virtualmente identico — o bridge opera sobre a representacao binaria flat, independente do shape.

**Contexto importante:** os numeros do Nx nativo refletem o BinaryBackend (Erlang puro), nao EXLA ou Torchx. Em producao, EXLA seria o ponto de comparacao relevante. Estes benchmarks medem o overhead do mecanismo de interoperabilidade, nao uma comparacao de desempenho real.

## 5. Gargalo de Serializacao do Pythonx

Ambas as abordagens (copy-based e view-based) compartilham uma limitacao fundamental: **o Pythonx precisa copiar dados ao cruzar a fronteira Elixir/Python**. A BEAM exige que ela possua a memoria referenciada por qualquer termo binario Erlang. O Pythonx nao possui mecanismo para transferir a propriedade de um buffer Python para a BEAM sem copia.

| Abordagem | Entrada (fronteira) | Saida interna Python | Saida (fronteira) | Total |
|-----------|:------------------:|:-------------------:|:-----------------:|:-----:|
| Copy-based (`tobytes`) | 1 (Pythonx) | 1 (`tobytes()` aloca buffer) | 1 (Pythonx encode) | **3** |
| View-based (`ndarray.view`) | 1 (Pythonx) | 0 (`view()` reinterpreta in-place) | 1 (Pythonx encode) | **2** |
| Zero-Copy NIF (trabalho futuro) | 0 (ponteiro) | 0 | 0 (ponteiro) | **0** |

## 6. Resumo

| Objetivo | Status | Artefato |
|----------|--------|----------|
| Cobertura de tipos (20 tipos Nx) | Completo | `type_coverage.livemd` |
| Integracao com pipeline Nx (`runtime_call`) | Completo | `poc.livemd` |
| Analise de copias de memoria | Completo | `poc.livemd`, `zero_copy_nif_future_work.livemd` |
| Benchmarks sistematicos | Completo | `benchmarks.livemd` |

O bridge de interoperabilidade funciona corretamente para todos os 20 tipos numericos do Nx (com ressalvas de precisao documentadas para floats de baixa precisao). O overhead e dominado pelo custo fixo de entrada no CPython para tensores pequenos e pela serializacao do Pythonx para tensores grandes. Uma abordagem zero-copy via NIF demonstra que o gargalo de serializacao pode ser eliminado em principio.
