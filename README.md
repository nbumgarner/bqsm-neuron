# bqsm-neuron

A small language model served through a **wave-equilibrium engine** — a saturable-gain
oscillator medium that *relaxes to a fixed point* instead of pushing activations through a
stack of layers. int8, no dense high-precision matmul, **no GPU**. It runs on hardware nobody
would try a modern model on (developed on a 2007 dual-Xeon).

It's more than an engine — it's a small system that **watches and narrates its own state**:

- **Wave-equilibrium inference** — attention as ring proximity, a coupled-oscillator FFN, and a
  closed-form settling to equilibrium (Hopfield/DEQ-style).
- **Live one-shot learning** — teach it a fact or a behavior at runtime (`/learn`), no retrain,
  no gradient step; recalled exactly or by geometric near-match. The base weights stay byte-exact.
- **An always-on inner monologue** — a background stream the model runs to itself, **grounded in
  its own measured state**: it reads its real coupling/order-parameter values and reports them,
  and the telemetry catches it where its self-story outruns the measurement.
- **A Hermes-style tool-calling agent** (`/agent`) — declare tools, it emits `<tool_call>`,
  executes, feeds the result back, and answers. Tool-calls are teachable live via `/learn`.
- **A self-contained showcase** (`showcase.html`) — chat, the live cylinder visualizer, the
  monologue + its measured couplings + criticality regime, and the agent trace, on one page.

## Run

```bash
# 1. build the CPU kernels (x86 SSE4.1; see arm/ for aarch64)
cc -O3 -msse4.1 -fopenmp -fPIC -shared -o libint8.so   kernels_sse.c
cc -O3 -msse4.1 -fopenmp -fPIC -shared -o libint8gemm.so kernels_sse.c
cc -O3 -msse4.1 -fopenmp -fPIC -shared -o libbf16.so    bf16_sse.c
cc -O3 -msse4.1 -fopenmp -fPIC -shared -o libbf16_mt.so bf16_mt.c

# 2. point at a ChatML instruct model (a Hermes-3 / Llama-3.2 checkout) and quantise once
export BQSM_MODEL=/path/to/model  BQSM_BLOB=./model.int8
python3 bqsm_int8.py --build

# 3. serve, then open http://localhost:8781/showcase
python3 bqsm_serve_int8.py --port 8781 --host 0.0.0.0
```

## Endpoints

| route | what |
|-------|------|
| `POST /generate` | chat (`{prompt, n, history, system}`) → job; poll `/jobs/<id>` |
| `POST /agent`    | Hermes tool-calling loop → job with a tool `trace` + `answer` |
| `GET /learn?source=&target=` | teach a fact or a tool-call, live |
| `GET /plastic?fact_assoc=` | tune associative-recall threshold; see last near-match |
| `GET /cyl` | live telemetry: cylinder state + the voice's measured couplings |
| `GET /voice/log?n=` · `/voice/pause?on=` | monologue history / start-stop |
| `GET /showcase` | the interactive page |

## Efficiency

`phox_energy.py` and `phox_profile.py` measure it honestly: tokens/sec, per-token profile, and
tokens/joule where the CPU exposes RAPL. On this class of hardware it's memory-bandwidth-bound —
not faster-per-token than a GPU, but FLOP-and-precision-light in a way that could matter for
energy at scale on modern silicon.

## License

Source-available — see [LICENSE.md](LICENSE.md). Free for individuals and companies under
US$3M gross revenue; commercial license above that.
