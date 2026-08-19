// Package responses provides OpenAI's Responses API as an LLM service, in two
// forms.
//
// NewHTTPLLM streams over a POST per turn, the same shape as the
// chat-completions service. NewLLM holds a WebSocket open for the session and
// adds the API's incremental-context optimization: when the conversation so far
// matches what the previous turn sent, only the new items travel and the server
// recalls the rest by response id. On a long conversation that is the difference
// between resending the whole history every turn and sending one message.
//
// Both consume an LLMContextFrame and emit LLM response frames like every other
// jargo LLM service, and both support tool calling.
package responses

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"strings"

	"github.com/gojargo/jargo/adapter"
	responsesadapter "github.com/gojargo/jargo/adapter/responses"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/llm"
	errs "github.com/gojargo/jargo/utils/errors"
)

const (
	// defaultBaseURL is the OpenAI API base for the HTTP service.
	defaultBaseURL = "https://api.openai.com/v1"
	// defaultWSURL is the Responses WebSocket endpoint.
	defaultWSURL = "wss://api.openai.com/v1/responses"
	// defaultModel is a current Responses-capable model.
	defaultModel = "gpt-4.1"
	// readLimit bounds a single inbound WebSocket message.
	readLimit = 1 << 22
)

// Input item types in the Responses API.
const (
	itemMessage    = "message"
	itemFuncCall   = "function_call"
	itemFuncOutput = "function_call_output"
)

// errStatus is returned when the API responds with a non-200 status.
//
//nolint:gochecknoglobals // sentinel error
var errStatus = errors.New("openai responses: unexpected status")

// errServer wraps a failure the API reported on the stream.
//
//nolint:gochecknoglobals // sentinel error
var errServer = errors.New("openai responses: server error")

// errStreamDone stops the event scan once the turn has ended. It never reaches
// the caller.
//
//nolint:gochecknoglobals // sentinel error
var errStreamDone = errors.New("openai responses: stream complete")

// Config configures the Responses LLM services. The sampling controls are
// pointers so a deliberate zero is distinguishable from "unset"; a nil value is
// omitted from the request, leaving the API default.
type Config struct {
	// APIKey is the OpenAI API key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the HTTP API base; empty uses the hosted API. It is
	// ignored by the WebSocket service, which uses WSURL.
	BaseURL string
	// WSURL overrides the Responses WebSocket endpoint; empty uses the hosted
	// one. It is ignored by the HTTP service.
	WSURL string
	// Model is the model id; empty uses a current default.
	Model string
	// Instructions is the system prompt, sent alongside the input rather than as
	// a message. A system message on the context takes precedence.
	Instructions string
	// MaxOutputTokens caps the response length; 0 omits it.
	MaxOutputTokens int
	// Temperature is the sampling temperature; nil omits it.
	Temperature *float64
	// TopP is the nucleus-sampling parameter; nil omits it.
	TopP *float64
	// ServiceTier selects the processing tier ("auto", "flex", "priority");
	// empty omits it.
	ServiceTier string
	// Store asks OpenAI to retain the conversation for 30 days. It is off by
	// default. The WebSocket service's incremental-context optimization does not
	// need it: that uses a connection-local cache rather than stored state.
	Store bool
	// Extra sets arbitrary additional request fields not modeled above, applied
	// to every request. They override the modeled fields on conflict.
	Extra map[string]any
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// The Responses wire types. They live in the adapter, which is what converts a
// conversation into them; these aliases keep them reachable under this
// package's name.
type (
	// inputItem is one entry of a Responses request's input list.
	inputItem = responsesadapter.InputItem
	// responsesTool is a function tool advertised on the request.
	responsesTool = responsesadapter.Tool
)

// request is a Responses API request.
type request struct {
	Model           string          `json:"model"`
	Input           []inputItem     `json:"input"`
	Stream          bool            `json:"stream"`
	Store           bool            `json:"store"`
	Instructions    string          `json:"instructions,omitempty"`
	MaxOutputTokens int             `json:"max_output_tokens,omitempty"`
	Temperature     *float64        `json:"temperature,omitempty"`
	TopP            *float64        `json:"top_p,omitempty"`
	ServiceTier     string          `json:"service_tier,omitempty"`
	Tools           []responsesTool `json:"tools,omitempty"`
	// PreviousResponseID lets the server recall the conversation it already
	// holds, so Input carries only what is new. The HTTP service never sets it:
	// over HTTP the API requires Store, while the WebSocket service's cache is
	// connection-local.
	PreviousResponseID string `json:"previous_response_id,omitempty"`
}

// newRequest builds the request for a conversation, without the streaming or
// continuation fields the transports set themselves.
func (c Config) newRequest(
	convo *frames.LLMContext, opts adapter.Options, withTools bool,
) (request, error) {
	var a responsesadapter.Adapter
	// The configured instructions are this service's own default, standing in
	// when neither the call nor the conversation states one.
	if opts.SystemInstruction == "" && convo.System() == "" {
		opts.SystemInstruction = c.Instructions
	}
	p, err := a.LLMInvocationParams(convo, opts)
	if err != nil {
		return request{}, err
	}
	req := request{
		Model:           c.Model,
		Input:           p.Input,
		Stream:          true,
		Store:           c.Store,
		Instructions:    p.Instructions,
		MaxOutputTokens: c.MaxOutputTokens,
		Temperature:     c.Temperature,
		TopP:            c.TopP,
		ServiceTier:     c.ServiceTier,
	}
	if withTools {
		req.Tools = p.Tools
	}
	return req, nil
}

// runInference answers the conversation once over HTTP and returns the text.
// Both services share it: a one-shot inference is a plain request either way,
// so the connection the WebSocket service holds open for its turns is left to
// them.
func runInference(
	ctx context.Context, cfg Config, client *http.Client,
	convo *frames.LLMContext, opts llm.InferenceOptions,
) (string, error) {
	body, err := cfg.newRequest(
		convo, adapter.Options{SystemInstruction: opts.SystemInstruction}, false,
	)
	if err != nil {
		return "", err
	}
	body.Stream = false
	if opts.MaxTokens > 0 {
		body.MaxOutputTokens = opts.MaxTokens
	}
	if opts.SystemInstruction != "" {
		// The Responses API carries the instruction beside the conversation
		// rather than in it, so the one this inference was given stands in place
		// of the conversation's own.
		body.Instructions = opts.SystemInstruction
	}
	raw, err := encodeBody(body, cfg.Extra)
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(
		ctx, http.MethodPost, cfg.BaseURL+"/responses", bytes.NewReader(raw),
	)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+cfg.APIKey)

	resp, err := client.Do(req)
	if err != nil {
		return "", llm.AsCompletionTimeout(ctx, err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return "", errs.NewHTTPStatusError(resp.StatusCode, fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg))
	}
	var answer struct {
		Output []struct {
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&answer); err != nil {
		return "", err
	}
	var out strings.Builder
	for _, item := range answer.Output {
		for _, c := range item.Content {
			if c.Type == "output_text" {
				out.WriteString(c.Text)
			}
		}
	}
	return out.String(), nil
}

// encodeBody marshals the request, merging any extra fields over the modeled
// ones. The merge cost is paid only when extra is non-empty.
func encodeBody(req request, extra map[string]any) ([]byte, error) {
	if len(extra) == 0 {
		return json.Marshal(req)
	}
	raw, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	maps.Copy(m, extra)
	return json.Marshal(m)
}
