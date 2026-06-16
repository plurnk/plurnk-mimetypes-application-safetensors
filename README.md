# @plurnk/plurnk-mimetypes-application-safetensors

`application/x-safetensors` (safetensors model weights) mimetype handler for the [plurnk](https://github.com/plurnk) ecosystem. Hand-rolled binary reader, no parser dependency.

## install

```
npm i @plurnk/plurnk-mimetypes-application-safetensors
```

## what it does

A safetensors file is `<uint64 header-length><JSON header><raw tensor bytes>`. The JSON header is a tensor inventory: `{ name: { dtype, shape, data_offsets }, …, "__metadata__": {…} }`. This handler reads **only that header** — never the weight bytes — so an agent can answer "what's in this checkpoint" without loading the file.

- `extractRaw(content)` — the tensor names as `field` symbols.
- `deepJson(content)` — the `{ name: { dtype, shape } }` view, a jsonpath target (`$['model.norm.weight'].shape`). `data_offsets` are dropped as byte-layout noise.
- `toText` (regex/glob + embed-source) — `name: dtype [shape]` lines.
- `validate` — throws on a bad header; every other channel degrades to empty.

References are not applicable.

## license

MIT.
