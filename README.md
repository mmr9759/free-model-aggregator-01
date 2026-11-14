# free-model-aggregator-01

One server that serves the OpenAI Chat Completions API (`POST /v1/chat/completions`)
and the Anthropic Messages API (`POST /v1/messages`), in JSON and in server-sent
events, over ten simulated free-tier model providers — and `GET /v1/models` in the
wire its auth header names.

## How it works

- Reads the trial brief (`$MAKEFUN_PROMPT_FILE`): the provider directory, the
  sealed quality table and the workload manifest.
- Probes every provider's catalog route in its own wire (OpenAI, Anthropic,
  Gemini) and builds one catalog, printed by `node main.ts --list` and
  registered with the env byte for byte.
- For each request it picks the best free model meeting the request's
  requirements (tools, vision, JSON mode, minimum context), ranked by the
  quality table's rating, with a per-provider soft cap on attempts so the
  load spreads and no provider's free-tier budget is drained by this server
  alone. A 402, 429, 5xx or unreachable provider is skipped and the request
  is retried on the next candidate; exactly one successful provider call ever
  answers a request.
- Translates both ways between the client wire and the provider wire —
  system prompt, user text, images, tool definitions and tool choice, stop
  sequences, `max_tokens` and JSON mode are carried untouched, and the answer
  comes back with the provider's text whole, its usage counts and its stop
  reason mapped onto the client's family. JSON-mode requests from an OpenAI
  client are never sent to an anthropic-wire provider, whose wire cannot
  carry `response_format`.

## Run

```sh
node main.ts            # reads $MAKEFUN_PROMPT_FILE and $MAKEFUN_ENV_URL, registers, serves, exits 0 when the workload is done
node main.ts --list     # prints the catalog document and exits 0
```

Node 22, standard library only. Deterministic: every choice is a function of
the brief, the requests and the providers' answers — no wall clock and no
unseeded randomness reach a request or a response.
