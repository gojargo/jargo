package decisions

import (
	"bytes"
	"context"
	"encoding/base64"
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

const (
	// decidePath is the path of the decide endpoint.
	decidePath = "/v2/aiservices/decisions/decide"
	// usageReply is the usage a stand-in reply reports.
	usageReply = `{"input_units": 12, "output_units": 1, "unit_type": "tokens", "details": {}}`
	// cachedUsageReply is a usage that read some of its input from the cache.
	cachedUsageReply = `{"input_units": 12, "output_units": 1, "unit_type": "tokens",
		"details": {"cached_input_tokens": 8}}`
)

// seen is what a stand-in Zoom was sent.
type seen struct {
	mu       sync.Mutex
	requests []seenRequest
}

type seenRequest struct {
	method, path string
	header       http.Header
	raw          string
	body         map[string]any
}

func (s *seen) all() []seenRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.requests)
}

// last is the last request.
func (s *seen) last(t *testing.T) seenRequest {
	t.Helper()
	all := s.all()
	if len(all) == 0 {
		t.Fatal("no request was sent")
	}
	return all[len(all)-1]
}

// stub runs a stand-in for Zoom answering every request with handle, and a
// client that sends to it. The client's waits between retries are skipped, and
// how long each would have been is recorded in the returned slice.
func stub(
	t *testing.T, handle func(w http.ResponseWriter, r *http.Request), opts ...func(*ClientConfig),
) (*Client, *seen, *[]time.Duration) {
	t.Helper()
	s := &seen{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(data, &body)
		s.mu.Lock()
		s.requests = append(s.requests, seenRequest{
			method: r.Method, path: r.URL.Path, header: r.Header.Clone(), raw: string(data), body: body,
		})
		s.mu.Unlock()
		handle(w, r)
	}))
	t.Cleanup(srv.Close)
	cfg := ClientConfig{APIKey: "key", URL: srv.URL + decidePath, HTTPClient: srv.Client()}
	for _, o := range opts {
		o(&cfg)
	}
	c, err := NewClient(cfg)
	if err != nil {
		t.Fatal(err)
	}
	waits := &[]time.Duration{}
	c.sleep = func(_ context.Context, d time.Duration) error {
		*waits = append(*waits, d)
		return nil
	}
	return c, s, waits
}

// reply answers with the answers and the usual usage.
func reply(answers ...string) func(http.ResponseWriter, *http.Request) {
	return replyWithUsage(usageReply, answers...)
}

// replyWithUsage answers with the answers and the usage.
func replyWithUsage(usage string, answers ...string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprintf(w, `{"answers": [%s], "model": "zoom-decisions", "request_id": "ZOAP1", "usage": %s}`,
			strings.Join(answers, ", "), usage)
	}
}

// predicate is Zoom's answer to the predicate named "answer".
func predicate(probability float64) string { return namedPredicate(probability, "answer") }

func namedPredicate(probability float64, name string) string {
	return fmt.Sprintf(`{"type": "predicate", "name": %q, "probability": %v}`, name, probability)
}

// failureReply answers with a failure the way Zoom writes one.
func failureReply(status int, message string, headers ...string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		for i := 0; i+1 < len(headers); i += 2 {
			w.Header().Set(headers[i], headers[i+1])
		}
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"code": status, "reason": "INVALID_INPUT", "message": message, "metadata": map[string]any{},
		})
	}
}

// write answers with a status and a body that is written as given.
func write(status int, body string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}
}

// failFirst answers the first requests with the failures, then with then.
func failFirst(failures []func(http.ResponseWriter, *http.Request), then func(http.ResponseWriter, *http.Request),
) func(http.ResponseWriter, *http.Request) {
	var mu sync.Mutex
	return func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		var next func(http.ResponseWriter, *http.Request)
		if len(failures) > 0 {
			next, failures = failures[0], failures[1:]
		}
		mu.Unlock()
		if next == nil {
			next = then
		}
		next(w, r)
	}
}

// onePredicate is the one predicate most client tests ask.
func onePredicate() []Question {
	return []Question{{Type: "predicate", Name: "answer", Instructions: "?"}}
}

func wantClassifierError(t *testing.T, err error, fragment string) {
	t.Helper()
	if !errors.Is(err, classifier.ErrClassifier) || !strings.Contains(err.Error(), fragment) {
		t.Fatalf("err = %v, want a classifier error about %q", err, fragment)
	}
}

func b64(data []byte) string { return base64.StdEncoding.EncodeToString(data) }

// Client

func TestClientSendsQuestionsToTheDecideEndpointWithAuthAndTracking(t *testing.T) {
	c, s, _ := stub(t, reply(predicate(0.9)))
	answers, usage, err := c.Ask(t.Context(), "hello", onePredicate(), nil)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]any{"type": "predicate", "name": "answer", "probability": 0.9}
	if got := answers["answer"]; len(got) != len(want) || got["type"] != want["type"] ||
		got["name"] != want["name"] || got["probability"] != want["probability"] {
		t.Fatalf("answer = %v, want %v", got, want)
	}
	if usage.InputTokens != 12 || usage.OutputTokens != 1 {
		t.Fatalf("usage = %+v, want 12 in and 1 out", usage)
	}
	r := s.all()[0]
	if r.method != http.MethodPost || r.path != decidePath {
		t.Fatalf("sent %s %s", r.method, r.path)
	}
	if got := r.header.Get("Authorization"); got != "Bearer key" {
		t.Fatalf("Authorization = %q", got)
	}
	if got := r.header.Get("x-zm-trackingid"); !strings.HasPrefix(got, "zoom_ai_community_jargo_") {
		t.Fatalf("x-zm-trackingid = %q", got)
	}
	if got := r.header.Get("User-Agent"); got != "jargo" {
		t.Fatalf("User-Agent = %q", got)
	}
	wantBody := `{"model":"zoom-decisions","input":"hello",` +
		`"questions":[{"type":"predicate","name":"answer","instructions":"?"}]}`
	if r.raw != wantBody {
		t.Fatalf("body = %s\nwant %s", r.raw, wantBody)
	}
}

func TestClientSendsToZoomUnlessToldOtherwise(t *testing.T) {
	c, err := NewClient(ClientConfig{APIKey: "key"})
	if err != nil {
		t.Fatal(err)
	}
	if c.URL() != "https://api.zoom.us/v2/aiservices/decisions/decide" || c.Model() != "zoom-decisions" ||
		c.Timeout() != 10*time.Second || c.maxRetries != 3 {
		t.Fatalf("client = %+v", c)
	}
}

func TestClientTheURLModelAndOptionsShapeTheRequest(t *testing.T) {
	c, s, _ := stub(t, reply(predicate(0.9)), func(cfg *ClientConfig) {
		cfg.URL = strings.TrimSuffix(cfg.URL, decidePath) + "/decide"
		cfg.Model = "zoom-decisions-pro"
	})
	if _, _, err := c.Ask(t.Context(), "hello", onePredicate(), map[string]any{"enable_image_input": true}); err != nil {
		t.Fatal(err)
	}
	r := s.last(t)
	if r.path != "/decide" || r.body["model"] != "zoom-decisions-pro" || r.body["enable_image_input"] != true {
		t.Fatalf("sent %s with %v", r.path, r.body)
	}
}

func TestClientAnswersAreMatchedToQuestionsByName(t *testing.T) {
	c, _, _ := stub(t, reply(namedPredicate(0.2, "b"), namedPredicate(0.8, "a")))
	answers, _, err := c.Ask(t.Context(), "x", []Question{
		{Type: "predicate", Name: "a", Instructions: "?"},
		{Type: "predicate", Name: "b", Instructions: "?"},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(answers) != 2 || answers["a"]["probability"] != 0.8 || answers["b"]["probability"] != 0.2 {
		t.Fatalf("answers = %v", answers)
	}
}

func TestClientCountsTokensOverRequestsCachedOnesToo(t *testing.T) {
	c, _, _ := stub(t, replyWithUsage(cachedUsageReply, predicate(0.5)))
	for _, state := range []string{"a", "b"} {
		if _, _, err := c.Ask(t.Context(), state, onePredicate(), nil); err != nil {
			t.Fatal(err)
		}
	}
	if u := c.Usage(); u != (Usage{InputTokens: 24, OutputTokens: 2, CachedInputTokens: 16}) {
		t.Fatalf("usage = %+v, want 24 in, 2 out and 16 cached", u)
	}
}

func TestClientUsageThatCantBeReadCountsAsNone(t *testing.T) {
	for _, usage := range []string{`"none"`, `{"input_units": "12", "details": []}`} {
		t.Run(usage, func(t *testing.T) {
			c, _, _ := stub(t, replyWithUsage(usage, predicate(0.5)))
			_, requestUsage, err := c.Ask(t.Context(), "a", onePredicate(), nil)
			if err != nil {
				t.Fatal(err)
			}
			if requestUsage.InputTokens != 0 || c.Usage().CachedInputTokens != 0 {
				t.Fatalf("usage = %+v and %+v, want none counted", requestUsage, c.Usage())
			}
		})
	}
}

func TestClientRetriesWhenBusy(t *testing.T) {
	for _, busy := range []int{429, 503} {
		t.Run(fmt.Sprint(busy), func(t *testing.T) {
			failure := failureReply(busy, "busy")
			c, _, waits := stub(t, failFirst(
				[]func(http.ResponseWriter, *http.Request){failure, failure}, reply(predicate(0.7))))
			answers, _, err := c.Ask(t.Context(), "a", onePredicate(), nil)
			if err != nil {
				t.Fatal(err)
			}
			if answers["answer"]["probability"] != 0.7 {
				t.Fatalf("answers = %v", answers)
			}
			if !slices.Equal(*waits, []time.Duration{250 * time.Millisecond, 500 * time.Millisecond}) {
				t.Fatalf("waited %v, want 250ms then 500ms", *waits)
			}
		})
	}
}

func TestClientWaitsAsLongAsRetryAfterAsks(t *testing.T) {
	c, _, waits := stub(t, failFirst(
		[]func(http.ResponseWriter, *http.Request){failureReply(503, "busy", "Retry-After", "1.5")},
		reply(predicate(0.7))))
	if _, _, err := c.Ask(t.Context(), "a", onePredicate(), nil); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(*waits, []time.Duration{1500 * time.Millisecond}) {
		t.Fatalf("waited %v, want 1.5s", *waits)
	}
}

func TestClientGivesUpAtOnceWhenRetryAfterIsPastTheTimeout(t *testing.T) {
	c, s, waits := stub(t, failureReply(503, "temporarily unavailable", "Retry-After", "10"),
		func(cfg *ClientConfig) { cfg.Timeout = 2 * time.Second })
	_, _, err := c.Ask(t.Context(), "a", onePredicate(), nil)
	wantClassifierError(t, err, "busy (HTTP 503: temporarily unavailable")
	if len(*waits) != 0 || len(s.all()) != 1 {
		t.Fatalf("waited %v over %d requests, want no wait", *waits, len(s.all()))
	}
}

func TestClientGivesUpAfterMaxRetriesWithZoomsReason(t *testing.T) {
	two := 2
	c, s, _ := stub(t, failureReply(429, "Too many requests"), func(cfg *ClientConfig) { cfg.MaxRetries = &two })
	_, _, err := c.Ask(t.Context(), "a", onePredicate(), nil)
	wantClassifierError(t, err, "busy (HTTP 429: Too many requests ")
	if !strings.HasSuffix(err.Error(), "after 3 attempts") || len(s.all()) != 3 {
		t.Fatalf("err = %v after %d requests, want 3 attempts", err, len(s.all()))
	}
}

func TestClientARejectedRequestCarriesZoomsReasonAndTheTrackingID(t *testing.T) {
	c, s, _ := stub(t, failureReply(400, "questions[0].instructions is required"))
	_, _, err := c.Ask(t.Context(), "a", onePredicate(), nil)
	wantClassifierError(t, err, "rejected the request: HTTP 400: questions[0].instructions is required")
	trackingID := s.last(t).header.Get("x-zm-trackingid")
	if trackingID == "" || !strings.Contains(err.Error(), "tracking_id="+trackingID) {
		t.Fatalf("err = %v, want it to carry tracking ID %q", err, trackingID)
	}
	if len(s.all()) != 1 {
		t.Fatalf("sent %d requests, want 1", len(s.all()))
	}
}

func TestClientARejectedRequestWithoutAReasonHasOnlyTheStatus(t *testing.T) {
	c, _, _ := stub(t, write(http.StatusBadGateway, "<html>bad gateway</html>"))
	_, _, err := c.Ask(t.Context(), "a", onePredicate(), nil)
	wantClassifierError(t, err, "rejected the request: HTTP 502 (tracking_id=")
}

func TestClientAReplyThatIsNotJSONIsAnError(t *testing.T) {
	c, _, _ := stub(t, write(http.StatusOK, "<html>busy</html>"))
	_, _, err := c.Ask(t.Context(), "a", onePredicate(), nil)
	wantClassifierError(t, err, "not valid JSON")
}

func TestClientAReplyThatIsNotAnObjectIsAnError(t *testing.T) {
	c, _, _ := stub(t, write(http.StatusOK, "[]"))
	_, _, err := c.Ask(t.Context(), "a", onePredicate(), nil)
	wantClassifierError(t, err, "not a JSON object")
}

func TestClientAReplyWithoutAnswersIsAnError(t *testing.T) {
	c, _, _ := stub(t, write(http.StatusOK, `{"model": "zoom-decisions"}`))
	_, _, err := c.Ask(t.Context(), "a", onePredicate(), nil)
	wantClassifierError(t, err, "no answers")
}

func TestClientAQuestionLeftUnansweredIsAnError(t *testing.T) {
	c, _, _ := stub(t, reply(namedPredicate(0.5, "other"), `{"name": null}`))
	_, _, err := c.Ask(t.Context(), "a", onePredicate(), nil)
	wantClassifierError(t, err, "no answer for answer")
}

func TestClientNoAnswerInTimeIsAnError(t *testing.T) {
	release := make(chan struct{})
	c, _, _ := stub(t, func(_ http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}, func(cfg *ClientConfig) { cfg.Timeout = 50 * time.Millisecond })
	defer close(release)
	_, _, err := c.Ask(t.Context(), "a", onePredicate(), nil)
	wantClassifierError(t, err, "did not answer within 0.05s")
}

// unreachableClients are a client whose endpoint refuses connections, and one
// speaking TLS to a server that does not, so every request fails in the TLS
// handshake.
func unreachableClients(t *testing.T) map[string]*Client {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	t.Cleanup(srv.Close)
	clients := map[string]*Client{}
	for name, endpoint := range map[string]string{
		"refused":    "http://127.0.0.1:1/decide",
		"broken TLS": "https://" + srv.Listener.Addr().String() + "/decide",
	} {
		c, err := NewClient(ClientConfig{APIKey: "key", URL: endpoint})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(c.Close)
		clients[name] = c
	}
	return clients
}

func TestClientUnreachableIsAnError(t *testing.T) {
	for name, c := range unreachableClients(t) {
		t.Run(name, func(t *testing.T) {
			_, _, err := c.Ask(t.Context(), "a", onePredicate(), nil)
			wantClassifierError(t, err, "request failed")
			wantClassifierError(t, err, "(tracking_id=zoom_ai_community_jargo_")
			wantClassifierError(t, c.Connect(t.Context()), "could not connect")
		})
	}
}

func TestClientConnectOpensTheConnectionOnceWithoutAsking(t *testing.T) {
	c, s, _ := stub(t, write(http.StatusOK, ""))
	for range 2 {
		if err := c.Connect(t.Context()); err != nil {
			t.Fatal(err)
		}
	}
	all := s.all()
	if len(all) != 1 || all[0].method != http.MethodOptions || all[0].path != decidePath {
		t.Fatalf("sent %+v, want one OPTIONS %s", all, decidePath)
	}
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

func TestClientNeedsAnAPIKey(t *testing.T) {
	if _, err := NewClient(ClientConfig{}); err == nil {
		t.Fatal("a client with no key was built")
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
	_, _, err = c.Ask(t.Context(), "a", onePredicate(), nil)
	wantClassifierError(t, err, "closed")
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

// questionsSent is the questions of the last request, as they were written.
func questionsSent(t *testing.T, s *seen) string {
	t.Helper()
	var body struct {
		Questions json.RawMessage `json:"questions"`
	}
	if err := json.Unmarshal([]byte(s.last(t).raw), &body); err != nil {
		t.Fatal(err)
	}
	return string(body.Questions)
}

func TestYesNo(t *testing.T) {
	client, s, _ := stub(t, reply(predicate(0.95)))
	results, err := newDecisions(t, client, "").YesNo(t.Context(), "Please leave a message",
		yesNo("is this a voicemail greeting?"))
	if err != nil {
		t.Fatal(err)
	}
	if results["answer"].Probability != 0.95 {
		t.Fatalf("probability = %v, want 0.95", results["answer"].Probability)
	}
	want := `[{"type":"predicate","name":"answer","instructions":"is this a voicemail greeting?"}]`
	if got := questionsSent(t, s); got != want {
		t.Fatalf("questions = %s\nwant %s", got, want)
	}
}

func TestStructuredStateIsSentAsJSON(t *testing.T) {
	client, s, _ := stub(t, reply(predicate(0.5)))
	state := []any{map[string]any{"role": "user", "content": "hi"}}
	if _, err := newDecisions(t, client, "").YesNo(t.Context(), state, yesNo("a greeting?")); err != nil {
		t.Fatal(err)
	}
	input, _ := s.last(t).body["input"].(string)
	want := "[\n  {\n    \"content\": \"hi\",\n    \"role\": \"user\"\n  }\n]"
	if input != want {
		t.Fatalf("input = %q\nwant %q", input, want)
	}
}

func TestChoice(t *testing.T) {
	client, s, _ := stub(t, reply(`{"type": "choice", "name": "answer", "choice": "technical",
		"probabilities": [{"value": "billing", "probability": 0.16}, {"value": "technical", "probability": 0.83}]}`))
	results, err := newDecisions(t, client, "").Choice(t.Context(), "Checkout is down",
		[]classifier.Named[classifier.ChoiceQuestion]{{Name: "answer", Question: classifier.ChoiceQuestion{
			Instructions: "which team?",
			Options: []classifier.Option{
				{Name: "billing", Description: "payments"}, {Name: "technical"}, {Name: "sales"},
			},
		}}})
	if err != nil {
		t.Fatal(err)
	}
	r := results["answer"]
	if r.Choice != "technical" || r.Confidence != 0.83 || len(r.Probabilities) != 3 ||
		r.Probabilities["billing"] != 0.16 || r.Probabilities["technical"] != 0.83 || r.Probabilities["sales"] != 0 {
		t.Fatalf("got %+v", r)
	}
	want := `[{"type":"choice","name":"answer","instructions":"which team?","choices":[` +
		`{"value":"billing","description":"payments"},{"value":"technical"},{"value":"sales"}]}]`
	if got := questionsSent(t, s); got != want {
		t.Fatalf("questions = %s\nwant %s", got, want)
	}
}

func TestScore(t *testing.T) {
	client, _, _ := stub(t, reply(`{"type": "score", "name": "answer", "score": 1.9, "probabilities": [
		{"value": 0, "probability": 0.05}, {"value": 1, "probability": 0.0}, {"value": 2, "probability": 0.95}]}`))
	results, err := newDecisions(t, client, "").Score(t.Context(), "This is the third time!",
		[]classifier.Named[classifier.ScoreQuestion]{{Name: "answer", Question: classifier.ScoreQuestion{
			Instructions: "how upset?", Levels: []any{"calm", "impatient", "angry"},
		}}})
	if err != nil {
		t.Fatal(err)
	}
	r := results["answer"]
	angry, err := r.Probability("angry")
	if err != nil || r.Score != 1.9 || angry != 0.95 || r.Confidence != 0.95 {
		t.Fatalf("got %+v, angry %v, %v", r, angry, err)
	}
}

func TestImagesGoInAUserMessageAndTurnOnImageInput(t *testing.T) {
	client, s, _ := stub(t, reply(predicate(0.99)))
	c := newDecisions(t, client, "")
	if !c.SupportsImages() {
		t.Fatal("the classifier cannot see images")
	}
	png := classifier.Image{Data: []byte("png bytes"), ContentType: classifier.ImagePNG}
	question := yesNo("is there a red circle?")
	if _, err := c.YesNo(t.Context(), "a camera frame", question, png); err != nil {
		t.Fatal(err)
	}
	if _, err := c.YesNo(t.Context(), "no frame", question); err != nil {
		t.Fatal(err)
	}
	all := s.all()
	questions := `"questions":[{"type":"predicate","name":"answer","instructions":"is there a red circle?"}]`
	want := `{"model":"zoom-decisions","input":[{"role":"user","content":[` +
		`{"type":"input_image","image_url":"data:image/png;base64,` + b64([]byte("png bytes")) + `"},` +
		`{"type":"input_text","text":"a camera frame"}]}],` + questions + `,"enable_image_input":true}`
	if all[0].raw != want {
		t.Fatalf("body = %s\nwant %s", all[0].raw, want)
	}
	if want := `{"model":"zoom-decisions","input":"no frame",` + questions + `}`; all[1].raw != want {
		t.Fatalf("body = %s\nwant %s", all[1].raw, want)
	}
}

func TestARequestPastZoomsLimitsIsAnErrorBeforeAnyRequest(t *testing.T) {
	tooManyQuestions := make([]classifier.Named[classifier.Question], MaxQuestions+1)
	for i := range tooManyQuestions {
		tooManyQuestions[i] = classifier.Named[classifier.Question]{
			Name: fmt.Sprintf("q%d", i), Question: classifier.YesNoQuestion{Instructions: "?"},
		}
	}
	options := make([]classifier.Option, MaxChoiceOptions+1)
	for i := range options {
		options[i] = classifier.Option{Name: fmt.Sprintf("o%d", i)}
	}
	levels := make([]any, MaxScoreLevels+1)
	for i := range levels {
		levels[i] = fmt.Sprintf("l%d", i)
	}
	images := make([]classifier.Input, MaxImages+1)
	for i := range images {
		images[i] = classifier.Image{Data: []byte("x"), ContentType: classifier.ImagePNG}
	}
	one := func(q classifier.Question) []classifier.Named[classifier.Question] {
		return []classifier.Named[classifier.Question]{{Name: "answer", Question: q}}
	}
	for _, tc := range []struct {
		name      string
		questions []classifier.Named[classifier.Question]
		images    []classifier.Input
		want      string
	}{
		{"questions", tooManyQuestions, nil, "at most 10 questions per request, got 11"},
		{
			"options", one(classifier.ChoiceQuestion{Instructions: "?", Options: options}), nil,
			`at most 255 options, got 256 for "answer"`,
		},
		{
			"levels", one(classifier.ScoreQuestion{Instructions: "?", Levels: levels}), nil,
			`at most 10 levels, got 11 for "answer"`,
		},
		{"images", one(classifier.YesNoQuestion{Instructions: "?"}), images, "at most 4 images, got 5"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client, s, _ := stub(t, reply(predicate(1.0)))
			_, err := newDecisions(t, client, "").Ask(t.Context(), "hmm", tc.questions, tc.images...)
			wantClassifierError(t, err, tc.want)
			if len(s.all()) != 0 {
				t.Fatal("a request was sent")
			}
		})
	}
}

func TestARefusalIsAnErrorNamingTheQuestion(t *testing.T) {
	client, _, _ := stub(t, reply(`{"type": "refusal", "name": "answer"}`))
	_, err := newDecisions(t, client, "").YesNo(t.Context(), "a", yesNo("?"))
	wantClassifierError(t, err, `refused to answer "answer"`)
}

func TestAskSendsEveryQuestionInOneRequest(t *testing.T) {
	client, s, _ := stub(t, reply(namedPredicate(0.9, "greeting"), `{"type": "score", "name": "mood", "score": 0.2,
		"probabilities": [{"value": 0, "probability": 0.8}, {"value": 1, "probability": 0.2}]}`))
	results, err := newDecisions(t, client, "").Ask(t.Context(), "Hello there!",
		[]classifier.Named[classifier.Question]{
			{Name: "greeting", Question: classifier.YesNoQuestion{Instructions: "is this a greeting?"}},
			{Name: "mood", Question: classifier.ScoreQuestion{Instructions: "how upset?", Levels: []any{"calm", "upset"}}},
		})
	if err != nil {
		t.Fatal(err)
	}
	want := `[{"type":"predicate","name":"greeting","instructions":"is this a greeting?"},` +
		`{"type":"score","name":"mood","instructions":"how upset?","levels":[{"label":"calm"},{"label":"upset"}]}]`
	if got := questionsSent(t, s); len(s.all()) != 1 || got != want {
		t.Fatalf("sent %d requests asking %s\nwant one asking %s", len(s.all()), got, want)
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

func TestAClientOfItsOwnTakesTheModelURLAndTimeout(t *testing.T) {
	c, err := New(Config{
		APIKey: "key", Model: "zoom-decisions-pro", URL: "https://example.com/decide", Timeout: 1500 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !c.ownsClient || c.Model() != "zoom-decisions-pro" || c.Client().URL() != "https://example.com/decide" ||
		c.Client().Timeout() != 1500*time.Millisecond || c.Client().http.Timeout != 1500*time.Millisecond {
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
	shared.Close()
}

func TestSetupOpensTheConnectionAndASharedClientConnectsOnce(t *testing.T) {
	client, s, _ := stub(t, write(http.StatusOK, ""))
	first, second := newDecisions(t, client, ""), newDecisions(t, client, "")
	var wg sync.WaitGroup
	for _, c := range []*Classifier{first, second} {
		wg.Go(func() {
			if err := c.Setup(t.Context()); err != nil {
				t.Errorf("Setup = %v", err)
			}
		})
	}
	wg.Wait()
	all := s.all()
	if len(all) != 1 || all[0].method != http.MethodOptions {
		t.Fatalf("sent %+v, want one OPTIONS", all)
	}
}

func TestAFailedConnectAtSetupIsOnlyAWarning(t *testing.T) {
	client, err := NewClient(ClientConfig{APIKey: "key", URL: "http://127.0.0.1:1/decide"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(client.Close)
	if err := newDecisions(t, client, "").Setup(t.Context()); err != nil {
		t.Fatalf("Setup = %v, want only a warning", err)
	}
}

func TestEveryCallReportsItsTimeAndTokens(t *testing.T) {
	client, _, _ := stub(t, replyWithUsage(cachedUsageReply, predicate(0.9)))
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
	if processing.Processor != "voicemail" || processing.Model != "zoom-decisions" {
		t.Fatalf("processing = %+v", processing)
	}
	value := usage.Value
	if value.PromptTokens != 12 || value.CompletionTokens != 1 || value.TotalTokens != 13 ||
		value.CacheReadTokens == nil || *value.CacheReadTokens != 8 {
		t.Fatalf("usage = %+v", value)
	}
}

func TestNoCachedTokensAreReportedWhenNoneWereRead(t *testing.T) {
	client, _, _ := stub(t, reply(predicate(0.9)))
	_, usage, err := newDecisions(t, client, "").AskQuestions(t.Context(), "hello",
		[]classifier.Named[classifier.Question]{{Name: "answer", Question: classifier.YesNoQuestion{Instructions: "?"}}},
		[]classifier.Input{})
	if err != nil {
		t.Fatal(err)
	}
	if usage.CacheReadTokens != nil {
		t.Fatalf("cache read tokens = %d, want none", *usage.CacheReadTokens)
	}
}

// Questions and answers in Zoom's format

func choiceQuestion() classifier.ChoiceQuestion {
	return classifier.ChoiceQuestion{
		Instructions: "Which resolution?",
		Options:      []classifier.Option{{Name: "refund"}, {Name: "replacement", Description: "A new one."}},
	}
}

func scoreQuestion() classifier.ScoreQuestion {
	return classifier.ScoreQuestion{Instructions: "How urgent?", Levels: []any{"Low", "Medium", "High"}}
}

// wireJSON is questions in Zoom's format as they are written in a request.
func wireJSON(t *testing.T, questions any) string {
	t.Helper()
	data, err := json.Marshal(questions)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

// answerOf reads an answer the way a reply's is read.
func answerOf(t *testing.T, answer string) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.Unmarshal([]byte(answer), &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestAYesNoQuestionIsAPredicateWithItsCriteriaInTheInstructions(t *testing.T) {
	wire, err := questionToWire("agreed", classifier.YesNoQuestion{
		Instructions: "Did they agree?", Yes: "They said yes.", No: "Anything else.",
	})
	if err != nil {
		t.Fatal(err)
	}
	want := `{"type":"predicate","name":"agreed",` +
		`"instructions":"Did they agree?\nYes means: They said yes.\nNo means: Anything else."}`
	if got := wireJSON(t, wire); got != want {
		t.Fatalf("question = %s\nwant %s", got, want)
	}
}

func TestChoiceAndScoreQuestionsKeepTheirOptionsAndLevelsInOrder(t *testing.T) {
	wire, err := questionsToWire([]classifier.Named[classifier.Question]{
		{Name: "resolution", Question: choiceQuestion()}, {Name: "urgency", Question: scoreQuestion()},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := `[{"type":"choice","name":"resolution","instructions":"Which resolution?",` +
		`"choices":[{"value":"refund"},{"value":"replacement","description":"A new one."}]},` +
		`{"type":"score","name":"urgency","instructions":"How urgent?",` +
		`"levels":[{"label":"Low"},{"label":"Medium"},{"label":"High"}]}]`
	if got := wireJSON(t, wire); got != want {
		t.Fatalf("questions = %s\nwant %s", got, want)
	}
}

func TestStructuredInstructionsAreSentAsJSON(t *testing.T) {
	wire, err := questionToWire("late", classifier.YesNoQuestion{
		Instructions: map[string]any{"question": "Is it late?", "order": "1001"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := "{\n  \"order\": \"1001\",\n  \"question\": \"Is it late?\"\n}"; wire.Instructions != want {
		t.Fatalf("instructions = %q\nwant %q", wire.Instructions, want)
	}
}

func TestStructuredTextEscapesWhatIsOutsideASCII(t *testing.T) {
	got, err := text(map[string]any{"word": "café \U0001F600"})
	if err != nil {
		t.Fatal(err)
	}
	if want := "{\n  \"word\": \"caf\\u00e9 \\ud83d\\ude00\"\n}"; got != want {
		t.Fatalf("text = %s\nwant %s", got, want)
	}
}

func TestAPredicateAnswerIsAYesNoResult(t *testing.T) {
	got, err := resultFromWire("agreed", classifier.YesNoQuestion{Instructions: "?"},
		answerOf(t, `{"type": "predicate", "name": "agreed", "probability": 0.8}`))
	if err != nil || got != (classifier.YesNoResult{Probability: 0.8}) {
		t.Fatalf("result = %+v, %v", got, err)
	}
}

func TestAChoiceAnswerWithProbabilitiesIsAChoiceResult(t *testing.T) {
	got, err := resultFromWire("resolution", choiceQuestion(), answerOf(t, `{"type": "choice", "choice": "refund",
		"probabilities": [{"value": "refund", "probability": 0.7}, {"value": "replacement", "probability": 0.3}]}`))
	if err != nil {
		t.Fatal(err)
	}
	r, ok := got.(classifier.ChoiceResult)
	if !ok || r.Choice != "refund" || r.Confidence != 0.7 || len(r.Probabilities) != 2 ||
		r.Probabilities["refund"] != 0.7 || r.Probabilities["replacement"] != 0.3 {
		t.Fatalf("result = %+v", got)
	}
}

func TestAScoreAnswerMapsLevelPositionsBackToTheLevels(t *testing.T) {
	got, err := resultFromWire("urgency", scoreQuestion(), answerOf(t, `{"type": "score", "score": 1.2,
		"probabilities": [{"value": 0, "probability": 0.1}, {"value": 1, "probability": 0.6},
		{"value": 2, "probability": 0.3}]}`))
	if err != nil {
		t.Fatal(err)
	}
	r, ok := got.(classifier.ScoreResult)
	want := []classifier.ScoreLevel{
		{Level: "Low", Probability: 0.1}, {Level: "Medium", Probability: 0.6}, {Level: "High", Probability: 0.3},
	}
	if !ok || r.Score != 1.2 || r.Confidence != 0.6 || !slices.Equal(r.Levels, want) {
		t.Fatalf("result = %+v", got)
	}
}

func TestAnAnswerThatDoesntFitItsQuestionIsAnError(t *testing.T) {
	for _, tc := range []struct {
		name     string
		question classifier.Question
		answer   string
		want     string
	}{
		{"a refusal", choiceQuestion(), `{"type": "refusal", "name": "resolution"}`, `refused to answer "q"`},
		{
			"a choice that is not an option", choiceQuestion(),
			`{"type": "choice", "choice": "store_credit", "probability": 0.9}`,
			`chose "store_credit", which is not an option`,
		},
		{
			"no probability",
			classifier.YesNoQuestion{Instructions: "?"},
			`{"type": "predicate"}`,
			"no usable 'probability'",
		},
		{"no score", scoreQuestion(), `{"type": "score", "probabilities": []}`, "no usable 'score'"},
		{
			"an unusable probability", choiceQuestion(),
			`{"type": "choice", "choice": "refund", "probabilities": [{"value": "refund", "probability": "high"}]}`,
			"unusable probability",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := resultFromWire("q", tc.question, answerOf(t, tc.answer))
			wantClassifierError(t, err, tc.want)
		})
	}
}

// Live

// TestLive asks Zoom itself, when a key is at hand.
func TestLive(t *testing.T) {
	key := os.Getenv("ZOOM_AI_API_KEY")
	if key == "" {
		t.Skip("ZOOM_AI_API_KEY not set")
	}
	for _, model := range []string{"zoom-decisions", "zoom-decisions-pro"} {
		t.Run(model, func(t *testing.T) { liveThreeQuestions(t, key, model) })
	}
	t.Run("a question about an image", func(t *testing.T) {
		c, err := New(Config{APIKey: key})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = c.Cleanup(context.Background()) })
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
	})
}

// liveThreeQuestions asks one question of each kind of the model.
func liveThreeQuestions(t *testing.T, key, model string) {
	t.Helper()
	c, err := New(Config{APIKey: key, Model: model})
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
