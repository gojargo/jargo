# RTVI protocol specification

RTVI (Real-Time Voice Interface) is the protocol a client app, such as a web page or a mobile app, and a bot use to talk to each other during a session. Next to the audio and video, the session's transport carries RTVI messages as data messages: the client says it's ready, sends text or files, and answers requests; the bot says when the user and the bot start and stop speaking, what they said, and what the bot is doing.

Every message is a JSON object like this one:

```json
{ "label": "rtvi-ai", "type": "send-text", "id": "msg_1", "data": { "content": "Hi!" } }
```

`label` is always `"rtvi-ai"`, `type` says which message it is, and `data` carries its content. Every message a client sends has an `id`, and the server's answer to it repeats that `id`.

This folder describes every message, for each version of the protocol, in a form a program can read: client libraries can generate their code from it, and anyone can build a client or a server from it without reading jargo's code.

```
spec/
├── spec.go                  # the documents, embedded, and the contract hash
├── released.json            # the protocol versions jargo has released
├── 2.3.0/
│   └── rtvi.asyncapi.json   # the document of protocol 2.3.0
└── <version>/
    └── rtvi.asyncapi.json
```

A Go program reads a document with `spec.Document(version)`, and the versions there are with `spec.Versions()`.

## Reading a document

Each `<version>/rtvi.asyncapi.json` is an [AsyncAPI 3.0](https://www.asyncapi.com/docs/reference/specification/v3.0.0) document. AsyncAPI is a standard format for describing message-based APIs, and its tools can render a document as a web page or generate code from it.

The document is written from the server's point of view. Its `receiveClientMessages` operation lists the messages a client sends, and `sendServerMessages` the messages the server sends. Each message in `components.messages` has:

- `summary`: what the message means and when it is sent.
- `payload`: the schema of the whole message, as [JSON Schema](https://json-schema.org/): `label`, `type`, `id` when the message carries one, and `data`. Each field of `data` has a description.
- `examples`: an example message.
- `x-rtvi-enabled-by`: for a message the server only sends when a setting turns it on, the setting's name. The settings are the RTVI observer's: in jargo, the field of `rtvi.ObserverParams` of the same name, such as `MetricsEnabled` for `metrics_enabled`.
- `x-rtvi-min-client-version`: the lowest client protocol version the server sends the message to. A client tells the server its version in `client-ready`. From protocol 2.4.0, the server sends messages as it does to a client of its own version until then (the `current-version-before-client-ready` behavior), so before `client-ready` a client can receive messages and output types newer than its version.
- `x-rtvi-deprecated`: for a deprecated message, what replaces it.

The schema of a client message says what the server accepts, and the schema of a server message what the server sends. The server leaves out each field that has no value, so in a server message a field that can have none is optional and never `null`, and every other field is always there. A deprecated field is marked `deprecated`.

The document's `x-rtvi-behaviors` lists the rules the server follows across messages, beyond the shape of each one: which message answers a request and with which `id`, which version `bot-ready` reports, which messages the settings turn off, and how the server holds the bot's text until the bot speaks it and drops it when the user interrupts. Each behavior has an `id`, a `description`, and the types of the `messages` it involves.

## What jargo does not implement

The documents describe the whole protocol. The server does not yet implement these messages, so it neither sends nor handles them, and answers a client that sends one with an `error-response`:

- `send-file`, which the server does not take from the client.
- `ui-event`, `ui-snapshot`, `ui-cancel-job-group`, `ui-command` and `ui-job-group`, which need a UI worker the server does not have.
- `system-log`, and its setting `system_logs_enabled`, since the server does not forward its log to the client.
- `llm-function-call`, deprecated in favor of `llm-function-call-in-progress`, which the server sends.

The conformance tests list them, with the same reasons, and fail when one of them is implemented, so the list stays true.

## Changing the protocol

The documents are the protocol's, shipped as they are released: jargo implements them and does not generate them. Its Go types and the code that sends and handles each message are checked against them.

1. A new protocol version comes with its document, in a new `<version>/rtvi.asyncapi.json`. Leave the earlier versions' documents as they are.
2. Change the message types in `../messages.go`, and the code that sends or handles the message: the `Observer` in `../observer.go` sends most server messages, and the `Processor` in `../processor.go` handles client messages. A message that clients older than the new version must not receive is gated on the client's version in the observer.
3. Bump `ProtocolVersion` in `../messages.go` to the new version.
4. Run `go test ./processor/rtvi/...`. A message the document adds fails the conformance tests until the server sends or handles it, or until it is listed as not implemented, with the reason.

The descriptions in a document can mention jargo where the protocol needs a server's name for something, such as its settings. Keep such an edit to the descriptions and examples: the hash below leaves them out, so it doesn't change.

## Releases

Clients implement a protocol version from its document, so once a release has shipped a version, its wire format never changes: what the server accepts and sends, and how. Any change to it is a new protocol version. The document's descriptions and examples can still be improved, and a message can be marked deprecated, since a deprecated message is still sent and accepted.

A released version's schemas can also describe its wire format more precisely, where the server already sends exactly what the more precise schema says, such as `metrics` data typed by kind rather than left as any object. The conformance tests must show the server's messages match it. Since the hash covers the schemas, update the version's `contract_sha256` in `released.json` in the same change.

`released.json` records each protocol version a jargo release has shipped, with a hash of its wire format and the jargo release that first shipped it:

```json
{ "2.3.0": { "contract_sha256": "5e9c…", "jargo": "0.3.0" } }
```

The hash covers the document without its descriptions, summaries, titles, examples, tags and deprecations, so rewording a description or deprecating a message doesn't change it, and a change to the wire format does. `spec.ContractSHA256(version)` computes it. When a release ships a new protocol version, add the version to `released.json` with its hash. A version already recorded stays as it is. `spec_test.go` fails when the wire format of a recorded version has changed, which means it's time to bump `ProtocolVersion`.

## How the tests use the documents

`spec_test.go` checks the documents on their own: the current protocol version has the newest one, each is well formed (unique message types, each in one group, the operations splitting them by direction, a description for every field, behaviors naming known messages, an `id` on every answer), and a released version's wire format is unchanged.

`../conformance_test.go` checks that jargo behaves as each version's document says. It runs jargo's real `Processor` and `Observer`, holds a session with a client of each version that has a document and that the server still serves (the current major version and `LegacySupportedMajor`), and checks every message the server sends:

- It is a message the document lists for the server to send.
- Its payload matches the document's schema. The check reads the JSON document, not jargo's Go types.
- The client's version is at least the message's `x-rtvi-min-client-version`. Sessions with clients just below each minimum check that those messages are left out.

For the messages a client sends, the example of each must match its schema in the JSON document, and the server must accept it without an error, for each version it still serves, also with a field added to its data that the document doesn't have. The server must also answer an `error-response` to each message the current version's document forbids, built from the message's schema: one without `data` when `data` is required, one missing each required field, and one with each field of the wrong type. `lenient` in the test lists the few the server accepts anyway, with the reason. Each behavior in the document of a version the server still serves has a test in `behaviorTests`, by its `id`, and a behavior without a test fails.

The session must make the server send every message of the current version, so a message the document adds fails until `runSession` in the test makes the server send it. The tests also check that every message type `messages.go` declares is in the document, that `Processor.handleMessage` handles every client message the document describes, and that each `x-rtvi-enabled-by` setting is one of the observer's, on by default exactly when the document says so.

Schemas allow fields they don't list, since a client ignores fields it doesn't know. So a field jargo sends beyond what the document describes, such as the `turn` and `text_aggregation` entries of `metrics`, passes the conformance tests, and a client that doesn't know it ignores it.
