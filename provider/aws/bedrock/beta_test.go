package bedrock

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/anthropics/anthropic-sdk-go/bedrock"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/anthropic"
	"github.com/gojargo/jargo/service/llm"
)

// sentRequest is what the stub recorded of one request.
type sentRequest struct {
	betaHeader []string
	betaBody   json.RawMessage
}

// bedrockStub stands in for the Bedrock runtime, recording the beta header and
// the anthropic_beta body field of every request. It answers a one-shot invoke
// with a message; a streamed invoke gets the same body, which the client cannot
// read as a stream, and only the request matters here.
func bedrockStub(t *testing.T) (*httptest.Server, func() []sentRequest) {
	t.Helper()
	var mu sync.Mutex
	var sent []sentRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body struct {
			AnthropicBeta json.RawMessage `json:"anthropic_beta"` //nolint:tagliatelle // Bedrock's field name
		}
		_ = json.Unmarshal(raw, &body)
		mu.Lock()
		sent = append(sent, sentRequest{betaHeader: r.Header.Values("anthropic-beta"), betaBody: body.AnthropicBeta})
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"m1","type":"message","role":"assistant","model":"claude",` +
			`"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",` +
			`"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	t.Cleanup(srv.Close)
	return srv, func() []sentRequest {
		mu.Lock()
		defer mu.Unlock()
		return append([]sentRequest(nil), sent...)
	}
}

// staticAWS is an AWS config that signs with fixed credentials, touching
// neither the environment nor instance metadata.
func staticAWS() aws.Config {
	return aws.Config{
		Region:      "us-east-1",
		Credentials: credentials.NewStaticCredentialsProvider("AKIATEST", "secret", ""),
	}
}

type discardSink struct{}

func (discardSink) Text(string) error          { return nil }
func (discardSink) Tool(frames.ToolCall) error { return nil }

// runAll makes one request down each path the service has.
func runAll(t *testing.T, svc *anthropic.LLMService) {
	t.Helper()
	convo := frames.NewLLMContext("sys")
	convo.AddUserMessage("hi")
	if _, err := svc.RunInference(t.Context(), convo, llm.InferenceOptions{}); err != nil {
		t.Fatalf("RunInference: %v", err)
	}
	// The stub cannot speak Bedrock's event stream, so the streamed paths fail
	// once their request is sent; the request is what is under test.
	_ = svc.Generate(t.Context(), convo, func(string) error { return nil })
	_ = svc.GenerateWithTools(t.Context(), convo, discardSink{})
}

// TestBedrockSendsNoInterleavedThinkingBeta checks no request reaching Bedrock
// carries the interleaved-thinking beta, as a header or moved into the body.
func TestBedrockSendsNoInterleavedThinkingBeta(t *testing.T) {
	srv, sent := bedrockStub(t)
	svc := newLLM(Config{Model: "us.anthropic.claude-sonnet-4-5-v1:0"}, staticAWS(),
		option.WithBaseURL(srv.URL), option.WithMaxRetries(0))
	runAll(t, svc.LLMService)

	got := sent()
	if len(got) != 3 {
		t.Fatalf("requests = %d, want one per path", len(got))
	}
	for i, r := range got {
		if len(r.betaHeader) > 0 || len(r.betaBody) > 0 {
			t.Errorf("request %d carries a beta: header %v, body %s", i, r.betaHeader, r.betaBody)
		}
	}
}

// TestBedrockStubSeesABeta checks the stub would catch a beta: the same service
// without the Bedrock opt-out sends one, moved into the body by the SDK.
func TestBedrockStubSeesABeta(t *testing.T) {
	srv, sent := bedrockStub(t)
	cfg := Config{Model: "us.anthropic.claude-sonnet-4-5-v1:0"}.anthropicConfig()
	cfg.OmitInterleavedThinkingBeta = false
	svc := anthropic.NewLLMWithOptions("BedrockLLM", cfg, []option.RequestOption{
		bedrock.WithConfig(staticAWS()), option.WithBaseURL(srv.URL), option.WithMaxRetries(0),
	})
	convo := frames.NewLLMContext("sys")
	convo.AddUserMessage("hi")
	if _, err := svc.RunInference(t.Context(), convo, llm.InferenceOptions{}); err != nil {
		t.Fatalf("RunInference: %v", err)
	}

	got := sent()
	if len(got) != 1 || string(got[0].betaBody) != `["interleaved-thinking-2025-05-14"]` {
		t.Errorf("requests = %+v, want the beta in the body", got)
	}
}
