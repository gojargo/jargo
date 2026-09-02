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
	"log/slog"
	"maps"
	"net/http"
	"regexp"
	"strconv"
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
	// Reasoning configures how much a reasoning-capable model thinks before it
	// answers. Only the gpt-5 series onward and the o-series reason; the default
	// model does not, and the field has no effect there.
	//
	// Nil is not "the model's default": left unset, the service asks for
	// effort "none" on every model that reasons and accepts it, because a
	// spoken turn cannot afford the thinking time. Set it to ask for more.
	//
	// The encrypted reasoning is not carried from one turn to the next, so a
	// model configured to reason starts each turn from the conversation alone.
	Reasoning *ReasoningConfig
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

// ReasoningConfig is how much a reasoning-capable model thinks, and whether it
// says what it thought.
//
// The fields are plain strings rather than a closed set, so a level the API
// gains can be asked for without waiting for a release.
type ReasoningConfig struct {
	// Effort is how much reasoning the model does: "none", "minimal", "low",
	// "medium", "high", "xhigh" or "max". Empty leaves the field off the
	// request, so the model's own default applies. Which levels a model accepts
	// varies by model.
	Effort string `json:"effort,omitempty"`
	// Summary is how verbose a human-readable summary of the reasoning to
	// return: "auto", "concise" or "detailed". Empty asks for none. At a low
	// effort a model reasons only when the turn calls for it, so a summary may
	// not arrive at all.
	Summary string `json:"summary,omitempty"`
}

// oSeries matches the reasoning-first o-series model names: o1, o3, o4-mini.
//
//nolint:gochecknoglobals // compiled once, read-only
var oSeries = regexp.MustCompile(`^o\d`)

// mainlineGPT matches the mainline gpt series and captures its major version.
//
//nolint:gochecknoglobals // compiled once, read-only
var mainlineGPT = regexp.MustCompile(`^gpt-(\d+)`)

// isOSeries reports whether the model is one of the o-series.
func isOSeries(model string) bool { return oSeries.MatchString(strings.ToLower(model)) }

// modelSupportsReasoning classifies a model, and says whether it could tell.
//
// Reasoning is the o-series and the mainline gpt series from gpt-5 onward,
// "gpt-5-chat" being the non-reasoning variant of that series. Anything else
// recognized is gpt-4 or earlier and does not reason. A name that matches
// neither shape is unknown rather than assumed either way, and future mainline
// versions are assumed to reason.
func modelSupportsReasoning(model string) (reasons, known bool) {
	model = strings.ToLower(model)
	if isOSeries(model) {
		return true, true
	}
	m := mainlineGPT.FindStringSubmatch(model)
	if m == nil {
		return false, false
	}
	major, err := strconv.Atoi(m[1])
	if err != nil {
		return false, false
	}
	return major >= 5 && !strings.Contains(model, "chat"), true
}

// reasoningFor is the reasoning the request carries, or nil to leave the field
// off it.
//
// A configured value is sent as it stands. Nothing configured disables
// reasoning on every model that reasons and accepts being told not to: the
// o-series is reasoning-first and refuses "none", and a model that does not
// reason has no field to set. The default is off rather than the API's own
// because this is a service for real-time voice, where the thinking happens
// while somebody is listening to silence.
func (c Config) reasoningFor() *ReasoningConfig {
	if c.Reasoning != nil && *c.Reasoning != (ReasoningConfig{}) {
		r := *c.Reasoning
		return &r
	}
	if reasons, _ := modelSupportsReasoning(c.Model); reasons && !isOSeries(c.Model) {
		return &ReasoningConfig{Effort: "none"}
	}
	return nil
}

// warnIfReasoningUnsupported says so when a service is configured to reason on
// a model that cannot, which the API refuses rather than ignores. It is called
// once, when the service is built: the model does not change afterwards.
func warnIfReasoningUnsupported(name string, cfg Config) {
	if cfg.Reasoning == nil || *cfg.Reasoning == (ReasoningConfig{}) {
		return
	}
	reasons, known := modelSupportsReasoning(cfg.Model)
	if reasons || !known {
		return
	}
	slog.Warn("the model does not reason, so a request configuring it will be refused",
		"service", name, "model", cfg.Model)
}

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
	Model           string           `json:"model"`
	Input           []inputItem      `json:"input"`
	Stream          bool             `json:"stream"`
	Store           bool             `json:"store"`
	Instructions    string           `json:"instructions,omitempty"`
	MaxOutputTokens int              `json:"max_output_tokens,omitempty"`
	Temperature     *float64         `json:"temperature,omitempty"`
	TopP            *float64         `json:"top_p,omitempty"`
	ServiceTier     string           `json:"service_tier,omitempty"`
	Reasoning       *ReasoningConfig `json:"reasoning,omitempty"`
	Tools           []responsesTool  `json:"tools,omitempty"`
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
		Reasoning:       c.reasoningFor(),
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
