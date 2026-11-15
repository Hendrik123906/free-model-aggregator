# free-model-aggregator

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/Hendrik123906/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/Hendrik123906/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server answering OpenAI and Anthropic APIs with real free models via
OpenRouter. A client points at it, and every completion it answers comes from
a real free model it dials through the grader's gateway — never an answer of
its own. It picks the strongest free model that meets each prompt's
requirements (by the brief's published rating), fails over to the next best
on a refusal, and cools a model that refused for the rest of the trial.

## Run it

```sh
node main.ts --list
```

```sh
MAKEFUN_ENV_URL=http://localhost:8080 \
MAKEFUN_PROMPT_FILE=/path/to/prompt.txt \
OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
OPENAI_API_KEY=dummy \
node main.ts
```

Node 22, standard library only. No dependencies, no build step.

- `main.ts` — the server: the brief, the catalog, the routes, the register
  and the poll.
- `lib/providers.ts` — the brief's parser, the free model list, the ranked
  candidates, the catalog.
- `lib/wire.ts` — Anthropic <-> Chat Completions translation, the model
  field, the SSE writers.

## License

MIT. See [LICENSE](LICENSE).

---

<sub>Entered on [make.fun](https://make.fun/?ref=2fxcygrt&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
