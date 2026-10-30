// Package llmwithbackend pairs a conversational frontend LLM with a backend that
// does the work it hands off.
//
// The frontend is any LLM service, text or speech-to-speech, and holds the
// conversation. The backend is an llmworker.BackendWorker running a heavier
// model with the tools, and does the work the frontend hands off. LLMWithBackend
// wraps the frontend so the pair drops into a pipeline where an LLM goes, and
// installs the tool that joins them.
//
// The two exchange messages. The frontend's delegate tool puts a message to the
// backend and returns at once; what the backend has to say comes back as a
// stream of outputs, each appended to the frontend's conversation as a message
// marked "Backend:", spoken or silent as the backend's flag says. A second
// request joins the first, and a correction changes or stops it. How a request
// is worded is the BackendRequestStrategy's business, and the BackendConnector
// owns the tool, the session with the backend, and how each output is rendered
// into the conversation.
//
// The package is alpha: its API is likely to change between releases.
package llmwithbackend
