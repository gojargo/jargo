package decisions

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
)

// usageReply is the usage a stand-in reply reports.
const usageReply = `{"total_input_tokens": 12, "total_output_tokens": 0, "total_tokens": 12}`

// seen is what a stand-in Google was sent.
type seen struct {
	mu       sync.Mutex
	requests []seenRequest
}

type seenRequest struct {
	method, path, key string
	body              map[string]any
}

func (s *seen) all() []seenRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.requests)
}

// last is the body of the last request.
func (s *seen) last(t *testing.T) map[string]any {
	t.Helper()
	all := s.all()
	if len(all) == 0 {
		t.Fatal("no request was sent")
	}
	return all[len(all)-1].body
}

// question is the one question the last request sent.
func (s *seen) question(t *testing.T) map[string]any {
	t.Helper()
	qs, _ := s.last(t)["questions"].(map[string]any)
	if len(qs) != 1 {
		t.Fatalf("questions = %v, want one", qs)
	}
	for _, q := range qs {
		question, _ := q.(map[string]any)
		return question
	}
	return nil
}

// stub runs a stand-in for Google answering every request with handle, and a
// client that sends to it.
func stub(
	t *testing.T, handle func(w http.ResponseWriter, r *http.Request), opts ...func(*ClientConfig),
) (*Client, *seen) {
	t.Helper()
	s := &seen{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(data, &body)
		s.mu.Lock()
		s.requests = append(s.requests, seenRequest{
			method: r.Method, path: r.URL.Path, key: r.Header.Get("X-Goog-Api-Key"), body: body,
		})
		s.mu.Unlock()
		handle(w, r)
	}))
	t.Cleanup(srv.Close)
	cfg := ClientConfig{APIKey: "key", BaseURL: srv.URL, HTTPClient: srv.Client()}
	for _, o := range opts {
		o(&cfg)
	}
	c, err := NewClient(cfg)
	if err != nil {
		t.Fatal(err)
	}
	c.sleep = func(context.Context, time.Duration) error { return nil }
	return c, s
}

// reply answers with one answer, to the question named "answer".
func reply(answer string) func(http.ResponseWriter, *http.Request) {
	return replies(`{"answer": ` + answer + `}`)
}

// replies answers with the answers by question name.
func replies(answers string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprintf(w, `{"answers": %s, "usage": %s}`, answers, usageReply)
	}
}

func noul(probability float64) string {
	return fmt.Sprintf(`{"type": "noul", "noul": %v, "confidence": %v}`,
		probability, math.Max(probability, 1-probability))
}

func failureReply(status int, message string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{"code": status, "message": message, "status": "INVALID_ARGUMENT"},
		})
	}
}

func model(w http.ResponseWriter, _ *http.Request) {
	_, _ = io.WriteString(w, `{"name": "models/gemini-decision-1"}`)
}

func noulQuestion(instructions string) []NamedQuestion {
	return []NamedQuestion{{Name: "answer", Question: Question{Type: "noul", Instructions: instructions}}}
}

func wantClassifierError(t *testing.T, err error, fragment string) {
	t.Helper()
	if !errors.Is(err, classifier.ErrClassifier) || !strings.Contains(err.Error(), fragment) {
		t.Fatalf("err = %v, want a classifier error about %q", err, fragment)
	}
}

func b64(data []byte) string { return base64.StdEncoding.EncodeToString(data) }

// Client

func TestClientSendsQuestionsToTheDecisionsEndpointWithTheKey(t *testing.T) {
	c, s := stub(t, reply(noul(0.9)))
	answers, usage, err := c.Ask(t.Context(), "hello", noulQuestion("a greeting?"))
	if err != nil {
		t.Fatal(err)
	}
	if answers["answer"]["noul"] != 0.9 || usage != (Usage{InputTokens: 12}) {
		t.Fatalf("got %v, %+v", answers, usage)
	}
	r := s.all()[0]
	if r.method != http.MethodPost || r.path != "/v1beta/decisions" || r.key != "key" {
		t.Fatalf("sent %s %s with %q", r.method, r.path, r.key)
	}
	body, _ := json.Marshal(r.body)
	want := `{"input":"hello","model":"gemini-decision-1",` +
		`"questions":{"answer":{"instructions":"a greeting?","type":"noul"}}}`
	if string(body) != want {
		t.Fatalf("body = %s\nwant %s", body, want)
	}
}

func TestClientTheModelPicksWhereQuestionsGo(t *testing.T) {
	c, s := stub(t, reply(noul(0.9)), func(cfg *ClientConfig) { cfg.Model = "gemini-decision-next" })
	if _, _, err := c.Ask(t.Context(), "a", noulQuestion("?")); err != nil {
		t.Fatal(err)
	}
	if got := s.last(t)["model"]; got != "gemini-decision-next" {
		t.Fatalf("model = %v", got)
	}
}

func TestClientAnswersAreMatchedToQuestionsByName(t *testing.T) {
	c, _ := stub(t, replies(`{"b": `+noul(0.2)+`, "a": `+noul(0.8)+`}`))
	answers, _, err := c.Ask(t.Context(), "x", []NamedQuestion{
		{Name: "a", Question: Question{Type: "noul", Instructions: "a?"}},
		{Name: "b", Question: Question{Type: "noul", Instructions: "b?"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if answers["a"]["noul"] != 0.8 || answers["b"]["noul"] != 0.2 {
		t.Fatalf("answers = %v", answers)
	}
}

func TestClientSendsQuestionsInOrder(t *testing.T) {
	var mu sync.Mutex
	var raw []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		raw, _ = io.ReadAll(r.Body)
		mu.Unlock()
		replies(`{"b": `+noul(0.2)+`, "a": `+noul(0.8)+`}`)(w, r)
	}))
	t.Cleanup(srv.Close)
	c, err := NewClient(ClientConfig{APIKey: "key", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := c.Ask(t.Context(), "x", []NamedQuestion{
		{Name: "b", Question: Question{Type: "noul", Instructions: "b?"}},
		{Name: "a", Question: Question{Type: "noul", Instructions: "a?"}},
	}); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	want := `{"model":"gemini-decision-1","input":"x","questions":{` +
		`"b":{"type":"noul","instructions":"b?"},"a":{"type":"noul","instructions":"a?"}}}`
	if string(raw) != want {
		t.Fatalf("body = %s\nwant %s", raw, want)
	}
}

func TestClientCountsTokensOverRequests(t *testing.T) {
	c, _ := stub(t, reply(noul(0.5)))
	for _, state := range []string{"a", "b"} {
		if _, _, err := c.Ask(t.Context(), state, noulQuestion("?")); err != nil {
			t.Fatal(err)
		}
	}
	if u := c.Usage(); u.InputTokens != 24 {
		t.Fatalf("usage = %+v, want 24 in", u)
	}
}

func TestClientRetriesWhenBusy(t *testing.T) {
	var mu sync.Mutex
	statuses := []int{429}
	c, s := stub(t, func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		if len(statuses) > 0 {
			failureReply(statuses[0], "Resource has been exhausted")(w, r)
			statuses = statuses[1:]
			return
		}
		reply(noul(0.7))(w, r)
	})
	answers, _, err := c.Ask(t.Context(), "a", noulQuestion("?"))
	if err != nil {
		t.Fatal(err)
	}
	if answers["answer"]["noul"] != 0.7 || len(s.all()) != 2 {
		t.Fatalf("got %v after %d requests", answers, len(s.all()))
	}
}

func TestClientRetriesWhatMayPass(t *testing.T) {
	for _, status := range []int{408, 409, 429, 500, 503, 529} {
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			zero := 0
			c, s := stub(t, failureReply(status, "busy"), func(cfg *ClientConfig) { cfg.MaxRetries = &zero })
			_, _, err := c.Ask(t.Context(), "a", noulQuestion("?"))
			wantClassifierError(t, err, "busy")
			one := 1
			c2, s2 := stub(t, failureReply(status, "busy"), func(cfg *ClientConfig) { cfg.MaxRetries = &one })
			_, _, err = c2.Ask(t.Context(), "a", noulQuestion("?"))
			wantClassifierError(t, err, "busy")
			if len(s.all()) != 1 || len(s2.all()) != 2 {
				t.Fatalf("sent %d and %d, want 1 then 2", len(s.all()), len(s2.all()))
			}
		})
	}
}

func TestClientDoesNotRetryARejectedRequest(t *testing.T) {
	c, s := stub(t, failureReply(400, "bad"))
	_, _, err := c.Ask(t.Context(), "a", noulQuestion("?"))
	wantClassifierError(t, err, "HTTP 400: bad")
	if len(s.all()) != 1 {
		t.Fatalf("sent %d requests, want 1", len(s.all()))
	}
}

func TestClientBacksOffBetweenRetries(t *testing.T) {
	c, _ := stub(t, failureReply(503, "busy"))
	var waits []time.Duration
	c.sleep = func(_ context.Context, d time.Duration) error {
		waits = append(waits, d)
		return nil
	}
	_, _, err := c.Ask(t.Context(), "a", noulQuestion("?"))
	wantClassifierError(t, err, "HTTP 503: busy")
	if len(waits) != 3 {
		t.Fatalf("waited %v, want three waits", waits)
	}
	for i, base := range []time.Duration{250 * time.Millisecond, 500 * time.Millisecond, time.Second} {
		if waits[i] > base || waits[i] < base*3/4 {
			t.Fatalf("wait %d = %v, want within a quarter under %v", i, waits[i], base)
		}
	}
}

func TestClientWaitsAsLongAsGoogleAsks(t *testing.T) {
	var mu sync.Mutex
	headers := []map[string]string{{"Retry-After": "3"}, {"Retry-After-Ms": "1500"}}
	c, _ := stub(t, func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		if len(headers) > 0 {
			for k, v := range headers[0] {
				w.Header().Set(k, v)
			}
			headers = headers[1:]
			failureReply(429, "slow down")(w, r)
			return
		}
		reply(noul(0.5))(w, r)
	})
	var waits []time.Duration
	c.sleep = func(_ context.Context, d time.Duration) error {
		waits = append(waits, d)
		return nil
	}
	if _, _, err := c.Ask(t.Context(), "a", noulQuestion("?")); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(waits, []time.Duration{3 * time.Second, 1500 * time.Millisecond}) {
		t.Fatalf("waited %v, want 3s then 1.5s", waits)
	}
}

func TestClientGivesUpAfterMaxRetriesWithGooglesReason(t *testing.T) {
	one := 1
	c, s := stub(t, failureReply(429, "Resource has been exhausted"),
		func(cfg *ClientConfig) { cfg.MaxRetries = &one })
	_, _, err := c.Ask(t.Context(), "a", noulQuestion("?"))
	wantClassifierError(t, err, "HTTP 429: Resource has been exhausted")
	if len(s.all()) != 2 {
		t.Fatalf("sent %d requests, want 2", len(s.all()))
	}
}

func TestClientARejectedRequestCarriesGooglesReason(t *testing.T) {
	c, _ := stub(t, failureReply(400, "Question key 'a b' is invalid"))
	_, _, err := c.Ask(t.Context(), "a", noulQuestion("?"))
	wantClassifierError(t, err, "request failed: HTTP 400: Question key 'a b' is invalid")
}

func TestClientGooglesReasonIsStripped(t *testing.T) {
	c, _ := stub(t, failureReply(400, " question IDs must match [A-Za-z0-9_-]+\n"))
	_, _, err := c.Ask(t.Context(), "a", noulQuestion("?"))
	want := "Google Decisions request failed: HTTP 400: question IDs must match [A-Za-z0-9_-]+"
	if err == nil || !strings.HasSuffix(err.Error(), want) {
		t.Fatalf("err = %v, want it to end in %q", err, want)
	}
}

func TestClientARejectedRequestWithoutAReasonHasOnlyTheStatus(t *testing.T) {
	c, _ := stub(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = io.WriteString(w, "<html>not found</html>")
	})
	_, _, err := c.Ask(t.Context(), "a", noulQuestion("?"))
	if err == nil || !strings.HasSuffix(err.Error(), "request failed: HTTP 404") {
		t.Fatalf("err = %v, want only the status", err)
	}
}

func TestClientABlockedInputIsAnErrorWithGooglesReason(t *testing.T) {
	c, _ := stub(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"prompt_feedback": {"block_reason": "safety",
			"block_reason_message": "unsafe input"}, "usage": {"total_input_tokens": 3}}`)
	})
	_, _, err := c.Ask(t.Context(), "a", noulQuestion("?"))
	wantClassifierError(t, err, "blocked the input: safety: unsafe input")
	if c.Usage().InputTokens != 3 {
		t.Fatalf("usage = %+v, want 3 in", c.Usage())
	}
}

func TestClientAReplyWithoutAnswersIsAnError(t *testing.T) {
	c, _ := stub(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprintf(w, `{"usage": %s}`, usageReply)
	})
	_, _, err := c.Ask(t.Context(), "a", noulQuestion("?"))
	wantClassifierError(t, err, "no answers")
}

func TestClientAQuestionLeftUnansweredIsAnErrorNamingIt(t *testing.T) {
	c, _ := stub(t, reply(noul(0.5)))
	_, _, err := c.Ask(t.Context(), "a", []NamedQuestion{
		{Name: "answer", Question: Question{Type: "noul", Instructions: "?"}},
		{Name: "other", Question: Question{Type: "noul", Instructions: "?"}},
	})
	wantClassifierError(t, err, "no answer for other")
}

func TestClientAReplyWithoutUsageCountsNoTokens(t *testing.T) {
	c, _ := stub(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"answers": {"answer": `+noul(0.5)+`}}`)
	})
	if _, _, err := c.Ask(t.Context(), "a", noulQuestion("?")); err != nil {
		t.Fatal(err)
	}
	if c.Usage().InputTokens != 0 {
		t.Fatalf("usage = %+v, want none counted", c.Usage())
	}
}

func TestClientUnreachableIsAnError(t *testing.T) {
	zero := 0
	c, err := NewClient(ClientConfig{APIKey: "key", BaseURL: "http://127.0.0.1:1", MaxRetries: &zero})
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = c.Ask(t.Context(), "a", noulQuestion("?"))
	wantClassifierError(t, err, "request failed")
}

func TestClientRetriesWhenUnreachable(t *testing.T) {
	two := 2
	c, err := NewClient(ClientConfig{APIKey: "key", BaseURL: "http://127.0.0.1:1", MaxRetries: &two})
	if err != nil {
		t.Fatal(err)
	}
	waits := 0
	c.sleep = func(context.Context, time.Duration) error {
		waits++
		return nil
	}
	_, _, err = c.Ask(t.Context(), "a", noulQuestion("?"))
	wantClassifierError(t, err, "request failed")
	if waits != 2 {
		t.Fatalf("retried %d times, want 2", waits)
	}
}

// brokenTLSClient is a client speaking TLS to a server that does not, so every
// request fails in the TLS handshake.
func brokenTLSClient(t *testing.T) *Client {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	t.Cleanup(srv.Close)
	zero := 0
	c, err := NewClient(ClientConfig{
		APIKey: "key", BaseURL: "https://" + srv.Listener.Addr().String(), MaxRetries: &zero,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(c.Close)
	return c
}

func TestClientABrokenTLSConnectionIsAnError(t *testing.T) {
	_, _, err := brokenTLSClient(t).Ask(t.Context(), "a", noulQuestion("?"))
	wantClassifierError(t, err, "request failed")
}

func TestClientNoAnswerInTimeIsAnError(t *testing.T) {
	release := make(chan struct{})
	c, _ := stub(t, func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}, func(cfg *ClientConfig) { cfg.Timeout = 50 * time.Millisecond })
	defer close(release)
	_, _, err := c.Ask(t.Context(), "a", noulQuestion("?"))
	wantClassifierError(t, err, "did not answer within 0.05s")
}

func TestClientConnectLooksUpTheModel(t *testing.T) {
	c, s := stub(t, model, func(cfg *ClientConfig) { cfg.Model = "gemini-decision-next" })
	if err := c.Connect(t.Context()); err != nil {
		t.Fatal(err)
	}
	all := s.all()
	if len(all) != 1 || all[0].method != http.MethodGet || all[0].path != "/v1beta/models/gemini-decision-next" ||
		all[0].key != "key" {
		t.Fatalf("sent %+v, want GET /v1beta/models/gemini-decision-next", all)
	}
}

func TestClientConnectSendsOnlyOnce(t *testing.T) {
	c, s := stub(t, model)
	for range 2 {
		if err := c.Connect(t.Context()); err != nil {
			t.Fatal(err)
		}
	}
	if n := len(s.all()); n != 1 {
		t.Fatalf("connected %d times, want once", n)
	}
}

func TestClientARefusedConnectCarriesGooglesReason(t *testing.T) {
	c, _ := stub(t, failureReply(404, "models/gemini-decision-1 is not found"))
	err := c.Connect(t.Context())
	wantClassifierError(t, err, "could not connect to Google Decisions: HTTP 404: models/gemini-decision-1")
}

func TestClientABrokenTLSConnectionAtConnectIsAnError(t *testing.T) {
	err := brokenTLSClient(t).Connect(t.Context())
	wantClassifierError(t, err, "could not connect")
}

func TestClientKeepsIdleConnectionsOpenBetweenQuestions(t *testing.T) {
	c, err := NewClient(ClientConfig{APIKey: "key"})
	if err != nil {
		t.Fatal(err)
	}
	transport, ok := c.http.Transport.(*http.Transport)
	if !ok || transport.IdleConnTimeout != 240*time.Second || !transport.ForceAttemptHTTP2 {
		t.Fatalf("transport = %+v, want HTTP/2 kept alive for 240s", c.http.Transport)
	}
}

func TestClientCloseClosesTheConnectionPool(t *testing.T) {
	c, err := NewClient(ClientConfig{APIKey: "key"})
	if err != nil {
		t.Fatal(err)
	}
	c.Close()
	if !c.isClosed() {
		t.Fatal("the client is still open")
	}
	_, _, err = c.Ask(t.Context(), "a", noulQuestion("?"))
	wantClassifierError(t, err, "closed")
}

func TestClientNeedsAnAPIKey(t *testing.T) {
	if _, err := NewClient(ClientConfig{}); err == nil {
		t.Fatal("a client with no key was built")
	}
}

// Classifier

func newDecisions(t *testing.T, client *Client, name string) *Classifier {
	t.Helper()
	c, err := New(Config{Client: client, Name: name})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func yesNo(instructions any) []classifier.Named[classifier.YesNoQuestion] {
	return []classifier.Named[classifier.YesNoQuestion]{{
		Name: "answer", Question: classifier.YesNoQuestion{Instructions: instructions},
	}}
}

func TestYesNo(t *testing.T) {
	client, s := stub(t, reply(noul(0.95)))
	results, err := newDecisions(t, client, "").YesNo(t.Context(), "Please leave a message",
		yesNo("is this a voicemail greeting?"))
	if err != nil {
		t.Fatal(err)
	}
	if results["answer"].Probability != 0.95 {
		t.Fatalf("probability = %v, want 0.95", results["answer"].Probability)
	}
	q := s.question(t)
	if len(q) != 2 || q["type"] != "noul" || q["instructions"] != "is this a voicemail greeting?" {
		t.Fatalf("question = %v", q)
	}
}

func TestYesNoSaysWhatCountsAsEachAnswerInItsCriteria(t *testing.T) {
	client, s := stub(t, reply(noul(0.2)))
	c := newDecisions(t, client, "")
	var criteria []any
	for _, q := range []classifier.YesNoQuestion{
		{Instructions: "voicemail?", Yes: "a recording", No: "a person"},
		{Instructions: "voicemail?", Yes: "a recording"},
	} {
		if _, err := c.YesNo(t.Context(), "Hello?", []classifier.Named[classifier.YesNoQuestion]{
			{Name: "answer", Question: q},
		}); err != nil {
			t.Fatal(err)
		}
		criteria = append(criteria, s.question(t)["criteria"])
	}
	got, _ := json.Marshal(criteria)
	want := `[{"false_description":"a person","true_description":"a recording"},{"true_description":"a recording"}]`
	if string(got) != want {
		t.Fatalf("criteria = %s\nwant %s", got, want)
	}
}

func TestStructuredStateAndInstructionsAreSentAsJSON(t *testing.T) {
	client, s := stub(t, reply(noul(0.5)))
	state := []any{map[string]any{"role": "user", "content": "hi"}}
	instructions := map[string]any{"question": "did they greet?", "speaker": "user"}
	if _, err := newDecisions(t, client, "").YesNo(t.Context(), state, yesNo(instructions)); err != nil {
		t.Fatal(err)
	}
	body := s.last(t)
	input, _ := body["input"].(string)
	if input != `[{"content": "hi", "role": "user"}]` {
		t.Fatalf("input = %q", input)
	}
	if got := s.question(t)["instructions"]; got != `{"question": "did they greet?", "speaker": "user"}` {
		t.Fatalf("instructions = %q", got)
	}
}

func TestTextEscapesWhatIsOutsideASCII(t *testing.T) {
	got, err := text(map[string]any{"word": "caf\u00e9 \U0001F600", "quote": `a "b", c: d`})
	if err != nil {
		t.Fatal(err)
	}
	want := `{"quote": "a \"b\", c: d", "word": "caf\u00e9 \ud83d\ude00"}`
	if got != want {
		t.Fatalf("text = %s\nwant %s", got, want)
	}
}

func TestChoice(t *testing.T) {
	client, s := stub(t, reply(`{"type": "choice", "choice": "technical",
		"probabilities": {"billing": 0.16, "technical": 0.83}, "confidence": 0.83}`))
	options := []classifier.Option{
		{Name: "billing", Description: "payments"},
		{Name: "technical", Description: map[string]any{"covers": []any{"outages"}}},
		{Name: "sales"},
	}
	results, err := newDecisions(t, client, "").Choice(t.Context(), "Checkout is down",
		[]classifier.Named[classifier.ChoiceQuestion]{{Name: "answer", Question: classifier.ChoiceQuestion{
			Instructions: "which team?", Options: options,
		}}})
	if err != nil {
		t.Fatal(err)
	}
	r := results["answer"]
	if r.Choice != "technical" || r.Confidence != 0.83 || len(r.Probabilities) != 3 ||
		r.Probabilities["billing"] != 0.16 || r.Probabilities["technical"] != 0.83 || r.Probabilities["sales"] != 0 {
		t.Fatalf("got %+v", r)
	}
	q, _ := toGoogle(classifier.ChoiceQuestion{Instructions: "which team?", Options: options})
	data, _ := json.Marshal(q)
	want := `{"type":"choice","instructions":"which team?","criteria":{"billing":"payments",` +
		`"technical":"{\"covers\": [\"outages\"]}","sales":""}}`
	if string(data) != want {
		t.Fatalf("question = %s\nwant %s", data, want)
	}
	if got := s.question(t)["type"]; got != "choice" {
		t.Fatalf("sent a %v question", got)
	}
}

func TestTooManyOptionsIsAnErrorBeforeAnyRequest(t *testing.T) {
	client, s := stub(t, reply(noul(1.0)))
	options := make([]classifier.Option, MaxChoiceOptions+1)
	for i := range options {
		options[i] = classifier.Option{Name: fmt.Sprintf("o%d", i)}
	}
	_, err := newDecisions(t, client, "").Choice(t.Context(), "hmm", []classifier.Named[classifier.ChoiceQuestion]{{
		Name: "answer", Question: classifier.ChoiceQuestion{Instructions: "?", Options: options},
	}})
	wantClassifierError(t, err, "at most 255 options")
	if len(s.all()) != 0 {
		t.Fatal("a request was sent")
	}
}

func TestImagesGoAfterTheState(t *testing.T) {
	client, s := stub(t, reply(noul(0.99)))
	c := newDecisions(t, client, "")
	if !c.SupportsImages() {
		t.Fatal("the classifier cannot see images")
	}
	png := classifier.Image{Data: []byte("png bytes"), ContentType: classifier.ImagePNG}
	question := yesNo("is there a red circle?")
	if _, err := c.YesNo(t.Context(), "a camera frame", question, png); err != nil {
		t.Fatal(err)
	}
	got, _ := json.Marshal(s.last(t)["input"])
	want := `[{"text":"a camera frame","type":"text"},` +
		`{"data":"` + b64([]byte("png bytes")) + `","mime_type":"image/png","type":"image"}]`
	if string(got) != want {
		t.Fatalf("input = %s\nwant %s", got, want)
	}
	if _, err := c.YesNo(t.Context(), "no frame", question); err != nil {
		t.Fatal(err)
	}
	if got := s.last(t)["input"]; got != "no frame" {
		t.Fatalf("input = %v, want the state as text", got)
	}
}

// tone is a 440 Hz tone as 16-bit mono PCM.
func tone(seconds float64, sampleRate int) classifier.Audio {
	samples := int(seconds * float64(sampleRate))
	pcm := make([]byte, 2*samples)
	for i := range samples {
		v := int16(8000 * math.Sin(2*math.Pi*440*float64(i)/float64(sampleRate)))
		binary.LittleEndian.PutUint16(pcm[2*i:], uint16(v)) //nolint:gosec // two's complement
	}
	return classifier.Audio{Audio: pcm, SampleRate: sampleRate}
}

func TestAClipGoesWhereItIsGivenAs16KHzWAV(t *testing.T) {
	client, s := stub(t, reply(noul(0.1)))
	c := newDecisions(t, client, "")
	if !c.SupportsAudio() {
		t.Fatal("the classifier cannot hear audio")
	}
	png := classifier.Image{Data: []byte("png bytes"), ContentType: classifier.ImagePNG}
	if _, err := c.YesNo(t.Context(), "a caller", yesNo("does the caller sound upset?"),
		tone(1, 8000), png); err != nil {
		t.Fatal(err)
	}
	input, _ := s.last(t)["input"].([]any)
	if len(input) != 3 {
		t.Fatalf("input = %v, want text, audio and image", input)
	}
	text, _ := input[0].(map[string]any)
	clip, _ := input[1].(map[string]any)
	image, _ := input[2].(map[string]any)
	if text["type"] != "text" || text["text"] != "a caller" || image["type"] != "image" {
		t.Fatalf("input = %v", input)
	}
	if clip["type"] != "audio" || clip["mime_type"] != "audio/wav" {
		t.Fatalf("clip = %v", clip)
	}
	encoded, _ := clip["data"].(string)
	wav, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil || len(wav) < 44 || string(wav[:4]) != "RIFF" || string(wav[8:12]) != "WAVE" {
		t.Fatalf("clip is not a WAV: %v", err)
	}
	rate := binary.LittleEndian.Uint32(wav[24:28])
	channels := binary.LittleEndian.Uint16(wav[22:24])
	bits := binary.LittleEndian.Uint16(wav[34:36])
	frames := (len(wav) - 44) / 2
	if rate != 16000 || channels != 1 || bits != 16 || frames < 15200 || frames > 16800 {
		t.Fatalf("WAV is %d Hz, %d channels, %d bits, %d frames", rate, channels, bits, frames)
	}
}

func TestEachFileGoesInTheBlockForItsType(t *testing.T) {
	client, s := stub(t, reply(noul(0.5)))
	files := []classifier.Input{
		classifier.File{ContentType: "application/pdf", Data: []byte("%PDF-1.4")},
		classifier.File{ContentType: "video/mp4", URL: "https://example.com/clip.mp4"},
		classifier.File{ContentType: "audio/mp3", Data: []byte("ID3")},
		classifier.File{ContentType: "image/gif", Data: []byte("GIF89a")},
	}
	if _, err := newDecisions(t, client, "").YesNo(t.Context(), "a ticket", yesNo("?"), files...); err != nil {
		t.Fatal(err)
	}
	input, _ := s.last(t)["input"].([]any)
	got, _ := json.Marshal(input[1:])
	want := `[{"data":"` + b64([]byte("%PDF-1.4")) + `","mime_type":"application/pdf","type":"document"},` +
		`{"mime_type":"video/mp4","type":"video","uri":"https://example.com/clip.mp4"},` +
		`{"data":"` + b64([]byte("ID3")) + `","mime_type":"audio/mp3","type":"audio"},` +
		`{"data":"` + b64([]byte("GIF89a")) + `","mime_type":"image/gif","type":"image"}]`
	if string(got) != want {
		t.Fatalf("files = %s\nwant %s", got, want)
	}
}

func TestTheFileTypesAreTheOnesGoogleReads(t *testing.T) {
	client, _ := stub(t, reply(noul(0.5)))
	types := newDecisions(t, client, "").FileTypes()
	for _, want := range []string{"application/pdf", "video/mp4", "audio/mp3", "image/gif"} {
		if !slices.Contains(types, want) {
			t.Fatalf("file types %v lack %s", types, want)
		}
	}
	if len(types) != len(FileTypes) {
		t.Fatalf("file types = %v, want every one of %v", types, FileTypes)
	}
}

func TestAFileOfAnotherTypeIsRefusedBeforeAnyRequest(t *testing.T) {
	client, s := stub(t, reply(noul(0.5)))
	notes := classifier.File{ContentType: "text/plain", Data: []byte("notes")}
	_, err := newDecisions(t, client, "").YesNo(t.Context(), "notes", yesNo("?"), notes)
	wantClassifierError(t, err, "doesn't take text/plain files")
	if len(s.all()) != 0 {
		t.Fatal("a request was sent")
	}
}

func yesOrNo() []classifier.Named[classifier.ChoiceQuestion] {
	return []classifier.Named[classifier.ChoiceQuestion]{{Name: "answer", Question: classifier.ChoiceQuestion{
		Instructions: "?", Options: []classifier.Option{{Name: "yes"}, {Name: "no"}},
	}}}
}

func TestAChoiceOutsideTheOptionsIsAnError(t *testing.T) {
	client, _ := stub(t, reply(`{"type": "choice", "choice": "maybe", "confidence": 0.8}`))
	_, err := newDecisions(t, client, "").Choice(t.Context(), "hmm", yesOrNo())
	wantClassifierError(t, err, `Google Decisions chose "maybe"`)
}

func TestAnUnusableProbabilityIsAnError(t *testing.T) {
	client, _ := stub(t, reply(`{"type": "choice", "choice": "yes",
		"probabilities": {"yes": "high", "no": 0.1}, "confidence": 0.8}`))
	_, err := newDecisions(t, client, "").Choice(t.Context(), "hmm", yesOrNo())
	wantClassifierError(t, err, "unusable probability")
}

func TestScoreKeysProbabilitiesByLevel(t *testing.T) {
	client, s := stub(t, reply(`{"type": "score", "score": 1.97,
		"legend": {"0": "calm", "1": "impatient", "2": "{\"level\": \"angry\"}"},
		"probabilities": {"0": 0.01, "1": 0.01, "2": 0.98}, "confidence": 0.98}`))
	levels := []any{"calm", "impatient", map[string]any{"level": "angry"}}
	results, err := newDecisions(t, client, "").Score(t.Context(), "This is the third time!",
		[]classifier.Named[classifier.ScoreQuestion]{{Name: "answer", Question: classifier.ScoreQuestion{
			Instructions: "how upset?", Levels: levels,
		}}})
	if err != nil {
		t.Fatal(err)
	}
	r := results["answer"]
	if r.Score != 1.97 || r.Confidence != 0.98 || len(r.Levels) != 3 {
		t.Fatalf("got %+v", r)
	}
	for i, want := range []float64{0.01, 0.01, 0.98} {
		if r.Levels[i].Probability != want {
			t.Fatalf("level %d = %+v, want %v", i, r.Levels[i], want)
		}
	}
	if p, err := r.Probability(map[string]any{"level": "angry"}); err != nil || p != 0.98 {
		t.Fatalf("angry = %v, %v", p, err)
	}
	got, _ := json.Marshal(s.question(t))
	want := `{"criteria":["calm","impatient","{\"level\": \"angry\"}"],"instructions":"how upset?","type":"score"}`
	if string(got) != want {
		t.Fatalf("question = %s\nwant %s", got, want)
	}
}

func TestAMissingAnswerFieldIsAnError(t *testing.T) {
	client, _ := stub(t, reply(`{"type": "noul"}`))
	_, err := newDecisions(t, client, "").YesNo(t.Context(), "a", yesNo("?"))
	wantClassifierError(t, err, "noul")
}

func TestAnAnswerOfAnotherKindIsAnError(t *testing.T) {
	client, _ := stub(t, reply(`{"type": "choice", "choice": "yes", "confidence": 0.9}`))
	_, err := newDecisions(t, client, "").YesNo(t.Context(), "a", yesNo("?"))
	wantClassifierError(t, err, "noul")
}

func TestAskSendsEveryQuestionInOneRequest(t *testing.T) {
	client, s := stub(t, replies(`{"greeting": `+noul(0.9)+`,
		"mood": {"type": "score", "score": 0.2, "probabilities": {"0": 0.8, "1": 0.2}, "confidence": 0.8}}`))
	results, err := newDecisions(t, client, "").Ask(t.Context(), "Hello there!",
		[]classifier.Named[classifier.Question]{
			{Name: "greeting", Question: classifier.YesNoQuestion{Instructions: "is this a greeting?"}},
			{Name: "mood", Question: classifier.ScoreQuestion{Instructions: "how upset?", Levels: []any{"calm", "upset"}}},
		})
	if err != nil {
		t.Fatal(err)
	}
	all := s.all()
	qs, _ := all[0].body["questions"].(map[string]any)
	greetingQ, _ := qs["greeting"].(map[string]any)
	moodQ, _ := qs["mood"].(map[string]any)
	if len(all) != 1 || len(qs) != 2 || greetingQ["type"] != "noul" || moodQ["type"] != "score" {
		t.Fatalf("sent %d requests with %v", len(all), qs)
	}
	greeting, _ := results["greeting"].(classifier.YesNoResult)
	mood, _ := results["mood"].(classifier.ScoreResult)
	calm, _ := mood.Probability("calm")
	if greeting.Probability != 0.9 || calm != 0.8 {
		t.Fatalf("got %+v", results)
	}
}

func TestNeedsAnAPIKeyOrAClient(t *testing.T) {
	if _, err := New(Config{}); err == nil {
		t.Fatal("a classifier with no key and no client was built")
	}
}

func TestAClientOfItsOwnTakesTheModelAndTimeout(t *testing.T) {
	c, err := New(Config{APIKey: "key", Model: "gemini-decision-next", Timeout: 1500 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	if !c.ownsClient || c.Model() != "gemini-decision-next" || c.Client().Timeout() != 1500*time.Millisecond ||
		c.Client().http.Timeout != 1500*time.Millisecond {
		t.Fatalf("client = %+v", c.Client())
	}
	if err := c.Cleanup(t.Context()); err != nil {
		t.Fatal(err)
	}
}

func TestCleanupClosesOnlyAnOwnedClient(t *testing.T) {
	owned, err := New(Config{APIKey: "key"})
	if err != nil {
		t.Fatal(err)
	}
	if err := owned.Cleanup(t.Context()); err != nil {
		t.Fatal(err)
	}
	if !owned.Client().isClosed() {
		t.Error("an owned client was left open")
	}

	shared, err := NewClient(ClientConfig{APIKey: "key"})
	if err != nil {
		t.Fatal(err)
	}
	c := newDecisions(t, shared, "")
	if err := c.Cleanup(t.Context()); err != nil {
		t.Fatal(err)
	}
	if shared.isClosed() {
		t.Error("a shared client was closed")
	}
}

func TestSetupOpensTheConnection(t *testing.T) {
	client, s := stub(t, model)
	if err := newDecisions(t, client, "").Setup(t.Context()); err != nil {
		t.Fatal(err)
	}
	all := s.all()
	if len(all) != 1 || all[0].path != "/v1beta/models/gemini-decision-1" {
		t.Fatalf("sent %+v, want the model looked up", all)
	}
}

func TestAFailedConnectAtSetupIsOnlyAWarning(t *testing.T) {
	zero := 0
	client, err := NewClient(ClientConfig{APIKey: "key", BaseURL: "http://127.0.0.1:1", MaxRetries: &zero})
	if err != nil {
		t.Fatal(err)
	}
	if err := newDecisions(t, client, "").Setup(t.Context()); err != nil {
		t.Fatalf("Setup = %v, want only a warning", err)
	}
}

func TestASharedClientConnectsOnce(t *testing.T) {
	client, s := stub(t, model)
	first, second := newDecisions(t, client, ""), newDecisions(t, client, "")
	var wg sync.WaitGroup
	for _, c := range []*Classifier{first, second} {
		wg.Go(func() { _ = c.Setup(t.Context()) })
	}
	wg.Wait()
	_ = first.Setup(t.Context())
	if n := len(s.all()); n != 1 {
		t.Fatalf("connected %d times, want once", n)
	}
}

func TestEveryCallReportsItsTimeAndTokens(t *testing.T) {
	client, _ := stub(t, reply(noul(0.9)))
	c := newDecisions(t, client, "voicemail")
	got := make(chan []frames.MetricsData, 1)
	event.On(c.Events(), classifier.EventMetrics, func(_ context.Context, data []frames.MetricsData) {
		got <- data
	})
	if _, err := c.YesNo(t.Context(), "hello", yesNo("a greeting?")); err != nil {
		t.Fatal(err)
	}
	var data []frames.MetricsData
	select {
	case data = <-got:
	case <-time.After(time.Second):
		t.Fatal("no metrics reported")
	}
	if len(data) != 2 {
		t.Fatalf("got %d metrics, want time and tokens", len(data))
	}
	processing, ok1 := data[0].(frames.ProcessingMetricsData)
	usage, ok2 := data[1].(frames.LLMUsageMetricsData)
	if !ok1 || !ok2 {
		t.Fatalf("got %T and %T", data[0], data[1])
	}
	if processing.Processor != "voicemail" || processing.Model != "gemini-decision-1" {
		t.Fatalf("processing = %+v", processing)
	}
	if usage.Value.PromptTokens != 12 || usage.Value.TotalTokens != 12 {
		t.Fatalf("usage = %+v", usage.Value)
	}
}

// TestLive asks Google itself, when a key is at hand.
func TestLive(t *testing.T) {
	key := os.Getenv("GOOGLE_API_KEY")
	if key == "" {
		t.Skip("GOOGLE_API_KEY not set")
	}
	c, err := New(Config{APIKey: key})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Cleanup(context.Background()) })
	if err := c.Client().Connect(t.Context()); err != nil {
		t.Fatal(err)
	}
	results, err := c.Ask(t.Context(),
		"Hi, you've reached Sam. I can't take your call right now, leave a message.",
		[]classifier.Named[classifier.Question]{
			{Name: "voicemail", Question: classifier.YesNoQuestion{Instructions: "is this a voicemail greeting?"}},
			{Name: "turn", Question: classifier.ChoiceQuestion{
				Instructions: "is the speaker's turn over?",
				Options: []classifier.Option{
					{Name: "complete", Description: "finished"}, {Name: "short", Description: "a brief pause"},
				},
			}},
			{Name: "mood", Question: classifier.ScoreQuestion{
				Instructions: "how upset is the speaker?", Levels: []any{"calm", "impatient", "angry"},
			}},
		})
	if err != nil {
		t.Fatal(err)
	}
	voicemail, _ := results["voicemail"].(classifier.YesNoResult)
	turn, _ := results["turn"].(classifier.ChoiceResult)
	mood, _ := results["mood"].(classifier.ScoreResult)
	sum := 0.0
	for _, p := range turn.Probabilities {
		sum += p
	}
	if voicemail.Probability <= 0.5 || !slices.Contains([]string{"complete", "short"}, turn.Choice) ||
		math.Abs(sum-1) >= 0.05 || mood.Score < 0 || mood.Score > 2 {
		t.Fatalf("results = %+v", results)
	}
	if c.Client().Usage().InputTokens == 0 {
		t.Fatal("no tokens counted")
	}

	question := []classifier.Named[classifier.YesNoQuestion]{{
		Name: "red", Question: classifier.YesNoQuestion{Instructions: "Is there a red circle in the image?"},
	}}
	red, err := c.YesNo(t.Context(), "A camera frame.", question, circle(t, color.RGBA{R: 255, A: 255}))
	if err != nil || !red["red"].IsYes() {
		t.Fatalf("red circle: %+v, %v", red, err)
	}
	blue, err := c.YesNo(t.Context(), "A camera frame.", question, circle(t, color.RGBA{B: 255, A: 255}))
	if err != nil || blue["red"].IsYes() {
		t.Fatalf("blue circle: %+v, %v", blue, err)
	}

	clip, err := c.YesNo(t.Context(), "A clip of audio.",
		[]classifier.Named[classifier.YesNoQuestion]{{
			Name: "speech", Question: classifier.YesNoQuestion{Instructions: "Is a person speaking in this clip?"},
		}}, tone(2, 16000))
	if err != nil || clip["speech"].IsYes() {
		t.Fatalf("clip: %+v, %v", clip, err)
	}
	video, err := classifier.NewFile("video/mp4", nil,
		"https://storage.googleapis.com/generativeai-downloads/videos/ForBiggerBlazes.mp4")
	if err != nil {
		t.Fatal(err)
	}
	fire, err := c.YesNo(t.Context(), "A video.", []classifier.Named[classifier.YesNoQuestion]{{
		Name: "fire", Question: classifier.YesNoQuestion{Instructions: "Does the video show fire or flames?"},
	}}, video)
	if err != nil || !fire["fire"].IsYes() {
		t.Fatalf("video: %+v, %v", fire, err)
	}
}

// circle is a camera-sized JPEG with a circle of the color in the middle.
func circle(t *testing.T, fill color.Color) classifier.Image {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 640, 480))
	for y := range 480 {
		for x := range 640 {
			dx, dy := float64(x-320), float64(y-240)
			if dx*dx+dy*dy <= 100*100 {
				img.Set(x, y, fill)
			} else {
				img.Set(x, y, color.White)
			}
		}
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, nil); err != nil {
		t.Fatal(err)
	}
	return classifier.Image{Data: buf.Bytes(), ContentType: classifier.ImageJPEG}
}
