# Release evals

The release evals run the example bots through scripted conversations and
simulated callers, and judge what each bot says. [`manifest.yaml`](manifest.yaml)
lists which scenarios each example runs.

## How it works

Each example is a bot built on the development runner. The suite builds every
example the manifest names once, starts it for each run on its own port, and the
eval harness connects to the runner's `/ws` route as an RTVI client. It plays the
user's side of the conversation, listens to what the bot says, and judges it.

A scenario is one check:

- A **scripted** scenario (`scenarios/scripted/<name>.yaml`) writes out the
  user's turns, with what the bot should do after each one. `capital_question`,
  for example, asks for the capital of Germany and checks the answer says
  Berlin.
- A **simulated** scenario (`scenarios/simulated/<name>.yaml`) lets an LLM play
  a caller with a goal, and a judge then reads the whole conversation.

The judge, user and simulator blocks the scenarios include sit in `scenarios/`.

## What you need

- **The bots' API keys**, in the environment or a `.env` file, as each example's
  doc comment lists them.
- **The judge.** Scenarios judge with Jev
  (`factory: evals.judges.typesafe_classifier`), which needs `TYPESAFE_API_KEY`.
- **A local LLM.** The explainer, and the caller in simulations, run
  `gemma4:12b` on [Ollama](https://ollama.com):

  ```sh
  ollama pull gemma4:12b
  ```

- **Local speech, for audio scenarios.** The user's speech is synthesized with
  Kokoro and the bot's is transcribed with Moonshine. Kokoro needs the ONNX
  Runtime and espeak-ng; Moonshine needs libmoonshine (`make deps-moonshine`,
  then export `JARGO_MOONSHINE_LIB`). See
  [Installation](../../docs/getting-started/installation.md). Text scenarios
  need neither.

## Running

From the repository root:

```sh
go run ./cmd/jargo eval suite evals/release/manifest.yaml
go run ./cmd/jargo eval suite evals/release/manifest.yaml -p voice-cartesia   # one bot
go run ./cmd/jargo eval suite evals/release/manifest.yaml -s capital_question # one scenario
go run ./cmd/jargo eval suite evals/release/manifest.yaml -n nightly          # test-runs/nightly/
```

Each run writes to `test-runs/<name>/`, a timestamp unless `-n` is given: the
bots' and the harness's logs, the recordings when recording is on, and
`results.jsonl` with one record per run.

## Coverage

The manifest covers the examples jargo has. Bots whose example is not in jargo
yet (vision, image generation, video, flows, turn management, multi-worker,
thinking and others) are left out until their example is.
