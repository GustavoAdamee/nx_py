# GC safety no caminho zero-copy: por que o tensor original precisa continuar referenciado

## Objetivo

Documentar por que `NxPy.from_python/2` (caminho zero-copy, backends com suporte a ponteiro)
exige que o chamador mantenha o tensor original alcancavel apos o round-trip por Python, e por
que essa necessidade so surgiu no refactor `7cd887f` ("switch to protocol/encoder-based
architecture"), nao existindo no design original (`629d9db`).

> Nota de versao: a primeira versao deste documento prescrevia
> `send(self(), {:keep_alive, tensor})` como solucao. Esse idioma foi removido na revisao do
> PR #6 — ele nao funciona de dentro de um `GenServer`, onde a mensagem cai na propria mailbox
> do processo em vez de criar uma referencia viva no escopo. A recomendacao atual e referenciar
> o tensor original apos o ultimo uso do tensor reconstruido (por exemplo, devolvendo os dois
> juntos), ou reter o dono em estado explicito (`GenServer` state, `Process.put/2`).

## O problema

Exemplo **incorreto** — o dono deixa de ser referenciado antes da leitura:

```elixir
def gc_unsafe(original) do
  {result, _} =
    Pythonx.eval(
      """
      arr = tensor.array
      arr *= 10
      tensor
      """,
      %{"tensor" => original}
    )

  reconstructed = NxPy.from_python(result, backend: EXLA.Backend)

  # `original` nao e mais referenciado a partir deste ponto. A analise de liveness
  # da BEAM pode considerar o termo morto e coletar seu DeviceBuffer — mesmo que
  # `original` ainda esteja lexicamente "em escopo" na funcao.
  consume(reconstructed)
end
```

Exemplo **correto** — referenciar o dono depois do ultimo uso do reconstruido:

```elixir
def gc_safe(original) do
  {result, _} =
    Pythonx.eval(
      """
      arr = tensor.array
      arr *= 10
      tensor
      """,
      %{"tensor" => original}
    )

  reconstructed = NxPy.from_python(result, backend: EXLA.Backend)
  consumed = consume(reconstructed)

  # Devolver `original` junto do resultado o mantem alcancavel durante toda a operacao.
  {consumed, original}
end
```

Dentro de um `GenServer`, o equivalente e manter o dono no estado do processo (ou em
`Process.put/2`) enquanto o tensor reconstruido estiver em uso — nao enviar uma mensagem para
si mesmo.

## Por que existe: dois handles, uma unica alocacao fisica

A chave para entender o risco esta em `Nx.from_pointer/4`
(`nx/exla/lib/exla/backend.ex:137-181`):

```elixir
def from_pointer(%Nx.Pointer{} = pointer, type, dims, backend_opts, opts) do
  ...
  ref = EXLA.NIF.create_buffer_from_device_pointer(
    client.ref, mode, handle_nif, typespec, device_id
  )
  buffer = EXLA.DeviceBuffer.from_ref(ref, client, device_id, typespec)
  %{template | data: %EXLA.Backend{buffer: buffer}}
end
```

E, um nivel abaixo, na NIF C++ (`nx/exla/c_src/exla/exla.cc:408-420`):

```cpp
} else if (pointer_kind == "local") {
  auto ptr_int = fine::decode<int64_t>(env, pointer_data);
  ptr = reinterpret_cast<void *>(ptr_int);
}
...
auto buffer = unwrap(client->client()->CreateViewOfDeviceBuffer(
    ptr, shape, memory_space, on_delete_callback));
return fine::make_resource<ExlaBuffer>(std::move(buffer));
```

`CreateViewOfDeviceBuffer` cria uma **view** sobre o endereco `ptr` — nenhum byte e copiado.
Isso significa que existem duas camadas distintas, e apenas uma delas e realmente nova:

| Camada | Tensor original | Tensor reconstruido |
|---|---|---|
| Memoria fisica (os bytes do tensor) | mesma | **mesma** — endereco identico, zero-copy |
| Handle/recurso NIF que rastreia essa memoria no lado Elixir (`ExlaBuffer` → `EXLA.DeviceBuffer`) | um recurso | **um segundo recurso, independente** |

Ou seja: "novo" se refere apenas ao *handle* (o objeto leve que carrega o ponteiro e seu
refcount) — nao aos dados do tensor, que continuam sendo os mesmos bytes fisicos. Os dois
handles sao independentes entre si: cada um pode ser coletado pela BEAM sem que isso afete
diretamente o outro. O problema e que o PJRT trata o buffer original como o "dono" responsavel
pelo tempo de vida daquela alocacao; a view criada por `from_pointer` nao estende esse tempo de
vida — ela so e valida enquanto a alocacao original ainda existir. Se o recurso original for
coletado, a memoria pode ser liberada por baixo da view, mesmo que a view em si ainda seja um
termo Elixir/NIF valido e "vivo".

Se fosse uma copia de dados, os dois tensores seriam totalmente independentes e o timing do GC
seria irrelevante. E exatamente por ser zero-copy (uma view, nao uma copia) que o cuidado com o
GC se torna necessario.

## Por que o design anterior nao precisava disso

Antes do refactor `7cd887f`, `NxPy.Backend.EXLA` (`629d9db:nx_py/lib/nx_py/backend/exla.ex`)
funcionava assim:

```elixir
def to_python_ref(tensor) do
  ptr = Nx.to_pointer(tensor, mode: :local)
  %{pointer: ptr.address, ..., tensor_ref: tensor, mode: :zero_copy}
end

def from_python_ref(ref, _type, _shape) do
  # Python escreveu diretamente no DeviceBuffer — so devolve o tensor.
  ref.tensor_ref
end
```

O mapa de referencia (`ref`) carregava o proprio struct `tensor` dentro do campo `:tensor_ref`.
`from_python_ref/3` nunca reconstruia nada a partir de um ponteiro bruto — ele devolvia
literalmente o mesmo termo Elixir (`ref.tensor_ref`). Como o chamador precisava manter `ref` em
uso ate o fim (para poder chamar `from_python_ref` sobre ele), o tensor original permanecia
transitivamente alcancavel o tempo todo. Nao havia nenhuma janela em que a BEAM pudesse
considera-lo morto, porque nao havia um segundo handle/recurso sendo criado — era sempre o
mesmo.

O refactor removeu essa indirecao (`NxPy.Backend`/`NxPy.PythonLib`, campo `tensor_ref`) em favor
de um design mais simples e generico: `from_python/2` reconstroi a partir apenas dos metadados
(`pointer`, `dtype`, `shape`) lidos do objeto Python devolvido, sem exigir que seja o mesmo
objeto originalmente enviado (README: *"`obj` does not need to be the same object originally
sent to Python"*). Essa generalidade — poder reconstruir a partir de qualquer `NxTensor`, nao só
do que foi originalmente encodado — e o que tornou `from_pointer/4` necessario, e e exatamente
isso que introduz o segundo handle independente e, com ele, a janela de use-after-free.

## Conclusao

A exigencia de manter o dono referenciado e o preco de uma API mais generica, nao um descuido.
O design anterior garantia liveness "de graca" porque nunca soltava o termo original; o design
atual troca essa garantia estrutural por uma API que aceita qualquer `NxTensor` Python como
entrada de `from_python/2` — o que exige reconstrucao via ponteiro bruto (`Nx.from_pointer/4`)
e, portanto, cuidado do chamador para manter o tensor original alcancavel enquanto o tensor
reconstruido estiver em uso.

O contrato final ficou documentado na secao "GC safety (zero-copy path)" do `README.md` e nos
docs de `NxPy.from_python/2`, com o par de exemplos incorreto/correto mostrado acima.
