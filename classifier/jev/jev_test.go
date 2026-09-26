package jev

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/utils/events"
)

// usageReply is the usage a stand-in reply reports.
const usageReply = `{"input_tokens": 12, "output_tokens": 3}`

// seen is what a stand-in Jev was sent.
type seen struct {
	mu       sync.Mutex
	requests []seenRequest
}

type seenRequest struct {
	method, path, auth string
	body               map[string]any
}

func (s *seen) all() []seenRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.requests)
}

// question is the question named "answer" in the last request.
func (s *seen) question(t *testing.T) map[string]any {
	t.Helper()
	all := s.all()
	qs, _ := all[len(all)-1].body["questions"].(map[string]any)
	q, ok := qs["answer"].(map[string]any)
	if !ok {
		t.Fatalf("no question named answer in %v", all[len(all)-1].body)
	}
	return q
}

// stub runs a stand-in for Jev answering every request with handle, and a
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
			method: r.Method, path: r.URL.Path, auth: r.Header.Get("Authorization"), body: body,
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

// replyWith answers with one answer named "answer".
func replyWith(answer string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprintf(w, `{"model": "jev-1.13.0", "answers": {"answer": %s}, "usage": %s}`, answer, usageReply)
	}
}

func status(code int) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(code) }
}

func noul(instructions string) []NamedQuestion {
	return []NamedQuestion{{Name: "answer", Question: Question{Type: "noul", Instructions: instructions}}}
}

func wantClassifierError(t *testing.T, err error, fragment string) {
	t.Helper()
	if !errors.Is(err, classifier.ErrClassifier) || !strings.Contains(err.Error(), fragment) {
		t.Fatalf("err = %v, want a classifier error about %q", err, fragment)
	}
}

// Client

func TestClientSendsOneQuestionWithAuth(t *testing.T) {
	c, s := stub(t, replyWith(`{"type": "noul", "noul": 0.9}`))
	answers, usage, err := c.Ask(t.Context(), "hello", noul("a greeting?"))
	if err != nil {
		t.Fatal(err)
	}
	if answers["answer"]["noul"] != 0.9 || usage != (Usage{InputTokens: 12, OutputTokens: 3}) {
		t.Fatalf("got %v, %+v", answers, usage)
	}
	r := s.all()[0]
	if r.path != "/v1/systemone" || r.auth != "Bearer key" {
		t.Fatalf("sent %s with %q", r.path, r.auth)
	}
	body, _ := json.Marshal(r.body)
	want := `{"model":"jev-1.13.0","questions":{"answer":{"instructions":"a greeting?","type":"noul"}},"state":"hello"}`
	if string(body) != want {
		t.Fatalf("body = %s\nwant %s", body, want)
	}
}

func TestClientCountsTokensOverRequests(t *testing.T) {
	c, _ := stub(t, replyWith(`{"type": "noul", "noul": 0.5}`))
	for _, state := range []string{"a", "b"} {
		if _, _, err := c.Ask(t.Context(), state, noul("?")); err != nil {
			t.Fatal(err)
		}
	}
	if u := c.Usage(); u != (Usage{InputTokens: 24, OutputTokens: 6}) {
		t.Fatalf("usage = %+v, want 24 in and 6 out", u)
	}
}

func TestClientRetriesWhenBusy(t *testing.T) {
	var mu sync.Mutex
	statuses := []int{429, 529}
	c, _ := stub(t, func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		if len(statuses) > 0 {
			w.WriteHeader(statuses[0])
			statuses = statuses[1:]
			return
		}
		replyWith(`{"type": "noul", "noul": 0.7}`)(w, r)
	})
	var waits []time.Duration
	c.sleep = func(_ context.Context, d time.Duration) error {
		waits = append(waits, d)
		return nil
	}
	answers, _, err := c.Ask(t.Context(), "a", noul("?"))
	if err != nil {
		t.Fatal(err)
	}
	if answers["answer"]["noul"] != 0.7 {
		t.Fatalf("got %v", answers)
	}
	if !slices.Equal(waits, []time.Duration{250 * time.Millisecond, 500 * time.Millisecond}) {
		t.Fatalf("waited %v, want 250ms then 500ms", waits)
	}
}

func TestClientGivesUpAfterMaxRetries(t *testing.T) {
	two := 2
	c, _ := stub(t, status(429), func(cfg *ClientConfig) { cfg.MaxRetries = &two })
	_, _, err := c.Ask(t.Context(), "a", noul("?"))
	wantClassifierError(t, err, "busy")
}

func TestClientErrors(t *testing.T) {
	cases := []struct {
		name     string
		handle   func(http.ResponseWriter, *http.Request)
		fragment string
	}{
		{"a rejected request", status(401), "401"},
		{"a reply that is not JSON", func(w http.ResponseWriter, _ *http.Request) {
			_, _ = io.WriteString(w, "<html>busy</html>")
		}, "not valid JSON"},
		{"a reply that is not an object", func(w http.ResponseWriter, _ *http.Request) {
			_, _ = io.WriteString(w, `["answers"]`)
		}, "not a JSON object"},
		{"a reply without answers", func(w http.ResponseWriter, _ *http.Request) {
			_, _ = io.WriteString(w, `{"model": "jev-1.13.0"}`)
		}, "no answers"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c, _ := stub(t, tc.handle)
			_, _, err := c.Ask(t.Context(), "a", noul("?"))
			wantClassifierError(t, err, tc.fragment)
		})
	}
}

func TestClientIgnoresUsageThatIsNotAnObject(t *testing.T) {
	c, _ := stub(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"answers": {"answer": {"type": "noul", "noul": 0.5}}, "usage": "none"}`)
	})
	if _, _, err := c.Ask(t.Context(), "a", noul("?")); err != nil {
		t.Fatal(err)
	}
	if c.Usage().InputTokens != 0 {
		t.Fatalf("usage = %+v, want none counted", c.Usage())
	}
}

func TestClientUnreachableIsAnError(t *testing.T) {
	c, err := NewClient(ClientConfig{APIKey: "key", BaseURL: "http://127.0.0.1:1"})
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = c.Ask(t.Context(), "a", noul("?"))
	wantClassifierError(t, err, "failed")
}

func TestClientNeedsAnAPIKey(t *testing.T) {
	if _, err := NewClient(ClientConfig{}); err == nil {
		t.Fatal("a client with no key was built")
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

// Classifier

func newJev(t *testing.T, client *Client, name string) *Classifier {
	t.Helper()
	c, err := New(Config{Client: client, Name: name})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestYesNo(t *testing.T) {
	client, s := stub(t, replyWith(`{"type": "noul", "noul": 0.95}`))
	results, err := newJev(t, client, "").YesNo(t.Context(), "Please leave a message",
		[]classifier.Named[classifier.YesNoQuestion]{{
			Name: "answer", Question: classifier.YesNoQuestion{Instructions: "is this a voicemail greeting?"},
		}})
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

func TestYesNoWithWhatCountsAsEachAnswer(t *testing.T) {
	client, s := stub(t, replyWith(`{"type": "noul", "noul": 0.2}`))
	_, err := newJev(t, client, "").YesNo(t.Context(), "Hello?", []classifier.Named[classifier.YesNoQuestion]{{
		Name: "answer", Question: classifier.YesNoQuestion{
			Instructions: "is this a voicemail greeting?",
			Yes:          "a recorded greeting or carrier message",
			No:           "a person talking",
		},
	}})
	if err != nil {
		t.Fatal(err)
	}
	criteria, _ := s.question(t)["criteria"].(map[string]any)
	if criteria["true"] != "a recorded greeting or carrier message" || criteria["false"] != "a person talking" {
		t.Fatalf("criteria = %v", criteria)
	}
}

func turnOptions() []classifier.Option {
	return []classifier.Option{
		{Name: "complete", Description: "the turn is over"},
		{Name: "short", Description: "a brief pause"},
		{Name: "long", Description: "asked for time"},
	}
}

func TestChoice(t *testing.T) {
	client, s := stub(t, replyWith(`{"type": "choice", "choice": "short",
		"probabilities": {"complete": 0.1, "short": 0.88}, "confidence": 0.8}`))
	results, err := newJev(t, client, "").Choice(t.Context(), "I think, um",
		[]classifier.Named[classifier.ChoiceQuestion]{{Name: "answer", Question: classifier.ChoiceQuestion{
			Instructions: "is the user's turn over?", Options: turnOptions(),
		}}})
	if err != nil {
		t.Fatal(err)
	}
	r := results["answer"]
	if r.Choice != "short" || r.Confidence != 0.8 ||
		r.Probabilities["complete"] != 0.1 || r.Probabilities["short"] != 0.88 || r.Probabilities["long"] != 0 {
		t.Fatalf("got %+v", r)
	}
	criteria, _ := s.question(t)["criteria"].(map[string]any)
	if criteria["complete"] != "the turn is over" || criteria["long"] != "asked for time" {
		t.Fatalf("criteria = %v", criteria)
	}
}

func TestChoiceCriteriaKeepTheOptionsInOrder(t *testing.T) {
	q, err := toJev(classifier.ChoiceQuestion{Instructions: "?", Options: turnOptions()})
	if err != nil {
		t.Fatal(err)
	}
	data, _ := json.Marshal(q)
	want := `{"type":"choice","instructions":"?","criteria":{"complete":"the turn is over",` +
		`"short":"a brief pause","long":"asked for time"}}`
	if string(data) != want {
		t.Fatalf("question = %s\nwant %s", data, want)
	}
}

func TestTooManyOptionsIsAnErrorBeforeAnyRequest(t *testing.T) {
	client, s := stub(t, replyWith(`{"type": "choice", "choice": "o0", "confidence": 1.0}`))
	options := make([]classifier.Option, MaxChoiceOptions+1)
	for i := range options {
		options[i] = classifier.Option{Name: fmt.Sprintf("o%d", i)}
	}
	_, err := newJev(t, client, "").Choice(t.Context(), "hmm", []classifier.Named[classifier.ChoiceQuestion]{{
		Name: "answer", Question: classifier.ChoiceQuestion{Instructions: "?", Options: options},
	}})
	wantClassifierError(t, err, "at most 255 options")
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
	client, _ := stub(t, replyWith(`{"type": "choice", "choice": "maybe", "confidence": 0.8}`))
	_, err := newJev(t, client, "").Choice(t.Context(), "hmm", yesOrNo())
	wantClassifierError(t, err, "not an option")
}

func TestAnUnusableProbabilityIsAnError(t *testing.T) {
	client, _ := stub(t, replyWith(`{"type": "choice", "choice": "yes",
		"probabilities": {"yes": "high", "no": 0.1}, "confidence": 0.8}`))
	_, err := newJev(t, client, "").Choice(t.Context(), "hmm", yesOrNo())
	wantClassifierError(t, err, "unusable probability")
}

func TestScoreKeysProbabilitiesByLevel(t *testing.T) {
	client, s := stub(t, replyWith(`{"type": "score", "score": 1.05,
		"legend": {"0": "calm", "1": "frustrated", "2": "angry"},
		"probabilities": {"0": 0.0, "1": 0.95, "2": 0.05}, "confidence": 0.92}`))
	levels := []any{"calm", "frustrated", "angry"}
	results, err := newJev(t, client, "").Score(t.Context(), "This is the third time!",
		[]classifier.Named[classifier.ScoreQuestion]{{Name: "answer", Question: classifier.ScoreQuestion{
			Instructions: "how upset is the user?", Levels: levels,
		}}})
	if err != nil {
		t.Fatal(err)
	}
	r := results["answer"]
	want := []classifier.ScoreLevel{
		{Level: "calm", Probability: 0},
		{Level: "frustrated", Probability: 0.95},
		{Level: "angry", Probability: 0.05},
	}
	if r.Score != 1.05 || r.Confidence != 0.92 || !slices.Equal(r.Levels, want) {
		t.Fatalf("got %+v", r)
	}
	if p, err := r.Probability("frustrated"); err != nil || p != 0.95 {
		t.Fatalf("Probability(frustrated) = %v, %v", p, err)
	}
	criteria, _ := s.question(t)["criteria"].([]any)
	if len(criteria) != 3 || criteria[1] != "frustrated" {
		t.Fatalf("criteria = %v", criteria)
	}
}

func TestMissingAnswerFieldIsAnError(t *testing.T) {
	client, _ := stub(t, replyWith(`{"type": "noul"}`))
	_, err := newJev(t, client, "").YesNo(t.Context(), "a", []classifier.Named[classifier.YesNoQuestion]{{
		Name: "answer", Question: classifier.YesNoQuestion{Instructions: "?"},
	}})
	wantClassifierError(t, err, "noul")
}

func TestNeedsAKeyOrAClient(t *testing.T) {
	if _, err := New(Config{}); err == nil {
		t.Fatal("a classifier with no key and no client was built")
	}
}

func TestAClientOfItsOwnTakesTheModelEndpointAndTimeout(t *testing.T) {
	c, err := New(Config{APIKey: "k", Model: "jev-x", BaseURL: "https://jev.test", Timeout: 1500 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	if !c.ownsClient || c.Client().Model() != "jev-x" || c.Client().BaseURL() != "https://jev.test" ||
		c.Client().Timeout() != 1500*time.Millisecond {
		t.Fatalf("client = %+v", c.Client())
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
	c := newJev(t, shared, "")
	if err := c.Cleanup(t.Context()); err != nil {
		t.Fatal(err)
	}
	if shared.isClosed() {
		t.Error("a shared client was closed")
	}
}

func TestSetupOpensTheConnection(t *testing.T) {
	client, s := stub(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, `{"models": []}`) })
	if err := newJev(t, client, "").Setup(t.Context()); err != nil {
		t.Fatal(err)
	}
	all := s.all()
	if len(all) != 1 || all[0].method != http.MethodGet || all[0].path != "/v1/models" {
		t.Fatalf("sent %+v, want GET /v1/models", all)
	}
}

func TestAFailedConnectAtSetupIsOnlyAWarning(t *testing.T) {
	client, err := NewClient(ClientConfig{APIKey: "key", BaseURL: "http://127.0.0.1:1"})
	if err != nil {
		t.Fatal(err)
	}
	if err := newJev(t, client, "").Setup(t.Context()); err != nil {
		t.Fatalf("Setup = %v, want only a warning", err)
	}
}

func TestChoiceWithStructuredAndEmptyDescriptions(t *testing.T) {
	client, s := stub(t, replyWith(`{"type": "choice", "choice": "b", "probabilities": {"b": 1.0},
		"confidence": 1.0}`))
	instructions := map[string]any{"question": "which?", "note": "casual"}
	results, err := newJev(t, client, "").Choice(t.Context(), "hey",
		[]classifier.Named[classifier.ChoiceQuestion]{{Name: "answer", Question: classifier.ChoiceQuestion{
			Instructions: instructions,
			Options: []classifier.Option{
				{Name: "a", Description: map[string]any{"examples": []any{"hi", "hello"}}},
				{Name: "b"},
			},
		}}})
	if err != nil || results["answer"].Choice != "b" {
		t.Fatalf("got %+v, %v", results, err)
	}
	q := s.question(t)
	got, _ := json.Marshal(q)
	want := `{"criteria":{"a":{"examples":["hi","hello"]},"b":null},` +
		`"instructions":{"note":"casual","question":"which?"},"type":"choice"}`
	if string(got) != want {
		t.Fatalf("question = %s\nwant %s", got, want)
	}
}

func TestScoreWithStructuredLevels(t *testing.T) {
	client, _ := stub(t, replyWith(`{"type": "score", "score": 1.0, "probabilities": {"0": 0.1, "1": 0.9},
		"confidence": 0.9}`))
	levels := []any{
		map[string]any{"level": "calm"},
		map[string]any{"level": "angry", "signs": []any{"shouting"}},
	}
	results, err := newJev(t, client, "").Score(t.Context(), "This is the third time!",
		[]classifier.Named[classifier.ScoreQuestion]{{Name: "answer", Question: classifier.ScoreQuestion{
			Instructions: "how upset?", Levels: levels,
		}}})
	if err != nil {
		t.Fatal(err)
	}
	r := results["answer"]
	if p, err := r.Probability(map[string]any{"level": "calm"}); err != nil || p != 0.1 {
		t.Fatalf("calm = %v, %v", p, err)
	}
	if p, err := r.Probability(map[string]any{"signs": []any{"shouting"}, "level": "angry"}); err != nil || p != 0.9 {
		t.Fatalf("angry = %v, %v", p, err)
	}
	if _, err := r.Probability(map[string]any{"level": "elated"}); !errors.Is(err, classifier.ErrNoSuchLevel) {
		t.Fatalf("elated: err = %v, want no such level", err)
	}
}

func TestAskSendsEveryQuestionInOneRequest(t *testing.T) {
	client, s := stub(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprintf(w, `{"model": "jev-1.13.0", "answers": {
			"greeting": {"type": "noul", "noul": 0.9},
			"mood": {"type": "score", "score": 0.2, "probabilities": {"0": 0.8, "1": 0.2}, "confidence": 0.8}
		}, "usage": %s}`, usageReply)
	})
	results, err := newJev(t, client, "").Ask(t.Context(), "Hello there!", []classifier.Named[classifier.Question]{
		{Name: "greeting", Question: classifier.YesNoQuestion{Instructions: "is this a greeting?"}},
		{Name: "mood", Question: classifier.ScoreQuestion{Instructions: "how upset?", Levels: []any{"calm", "upset"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	all := s.all()
	qs, _ := all[0].body["questions"].(map[string]any)
	if len(all) != 1 || len(qs) != 2 {
		t.Fatalf("sent %d requests with %v", len(all), qs)
	}
	greeting, _ := results["greeting"].(classifier.YesNoResult)
	mood, _ := results["mood"].(classifier.ScoreResult)
	calm, _ := mood.Probability("calm")
	if greeting.Probability != 0.9 || mood.Score != 0.2 || calm != 0.8 {
		t.Fatalf("got %+v", results)
	}
}

func TestTypedMethodTakesSeveralQuestionsOfItsKind(t *testing.T) {
	client, _ := stub(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprintf(w, `{"answers": {"a": {"type": "noul", "noul": 0.1}, "b": {"type": "noul", "noul": 0.7}},
			"usage": %s}`, usageReply)
	})
	results, err := newJev(t, client, "").YesNo(t.Context(), "hi", []classifier.Named[classifier.YesNoQuestion]{
		{Name: "a", Question: classifier.YesNoQuestion{Instructions: "a?"}},
		{Name: "b", Question: classifier.YesNoQuestion{Instructions: "b?"}},
	})
	if err != nil || results["a"].Probability != 0.1 || results["b"].Probability != 0.7 {
		t.Fatalf("got %+v, %v", results, err)
	}
}

func TestAMissingAnswerIsAnError(t *testing.T) {
	client, _ := stub(t, replyWith(`{"type": "noul", "noul": 0.5}`))
	_, err := newJev(t, client, "").Ask(t.Context(), "hi", []classifier.Named[classifier.Question]{
		{Name: "answer", Question: classifier.YesNoQuestion{Instructions: "?"}},
		{Name: "other", Question: classifier.YesNoQuestion{Instructions: "?"}},
	})
	wantClassifierError(t, err, "no answer for other")
}

func TestASharedClientConnectsOnce(t *testing.T) {
	client, s := stub(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, `{"models": []}`) })
	first, second := newJev(t, client, ""), newJev(t, client, "")
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

func TestYesWhenYesIsLikelier(t *testing.T) {
	if !(classifier.YesNoResult{Probability: 0.5}).IsYes() || !(classifier.YesNoResult{Probability: 0.9}).IsYes() ||
		(classifier.YesNoResult{Probability: 0.49}).IsYes() {
		t.Fatal("IsYes should be true from 0.5 up")
	}
}

func metricsOf(t *testing.T, c *Classifier) []frames.MetricsData {
	t.Helper()
	got := make(chan []frames.MetricsData, 1)
	events.On(c.Events(), classifier.EventMetrics, func(_ context.Context, data []frames.MetricsData) {
		got <- data
	})
	_, err := c.YesNo(t.Context(), "hello", []classifier.Named[classifier.YesNoQuestion]{{
		Name: "answer", Question: classifier.YesNoQuestion{Instructions: "a greeting?"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	select {
	case data := <-got:
		return data
	case <-time.After(time.Second):
		t.Fatal("no metrics reported")
		return nil
	}
}

func TestEveryCallReportsItsTimeAndTokens(t *testing.T) {
	client, _ := stub(t, replyWith(`{"type": "noul", "noul": 0.9}`))
	c := newJev(t, client, "")
	data := metricsOf(t, c)
	if len(data) != 2 {
		t.Fatalf("got %d metrics, want time and tokens", len(data))
	}
	processing, ok1 := data[0].(frames.ProcessingMetricsData)
	usage, ok2 := data[1].(frames.LLMUsageMetricsData)
	if !ok1 || !ok2 {
		t.Fatalf("got %T and %T", data[0], data[1])
	}
	if processing.Processor != c.Name() || processing.Model != c.Model() || processing.Value < 0 {
		t.Fatalf("processing = %+v", processing)
	}
	if usage.Value.PromptTokens != 12 || usage.Value.CompletionTokens != 3 || usage.Value.TotalTokens != 15 {
		t.Fatalf("usage = %+v", usage.Value)
	}
}

func TestANamedClassifierReportsMetricsUnderItsName(t *testing.T) {
	client, _ := stub(t, replyWith(`{"type": "noul", "noul": 0.9}`))
	c := newJev(t, client, "voicemail")
	for _, d := range metricsOf(t, c) {
		if d.MetricsProcessor() != "voicemail" {
			t.Fatalf("metrics under %q, want voicemail", d.MetricsProcessor())
		}
	}
}

// TestLive asks Jev itself, when a key is at hand.
func TestLive(t *testing.T) {
	key := os.Getenv("TYPESAFE_API_KEY")
	if key == "" {
		t.Skip("TYPESAFE_API_KEY not set")
	}
	c, err := New(Config{APIKey: key})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Cleanup(context.Background()) })

	yesNo, err := c.YesNo(t.Context(), "Hi, you've reached Sam. I can't take your call right now, leave a message.",
		[]classifier.Named[classifier.YesNoQuestion]{{
			Name: "answer", Question: classifier.YesNoQuestion{Instructions: "is this a voicemail greeting?"},
		}})
	if err != nil || yesNo["answer"].Probability <= 0.5 {
		t.Fatalf("voicemail: %+v, %v", yesNo, err)
	}
	choice, err := c.Choice(t.Context(), "I'd like to book a table for, um",
		[]classifier.Named[classifier.ChoiceQuestion]{{Name: "answer", Question: classifier.ChoiceQuestion{
			Instructions: "is the user's turn over?",
			Options: []classifier.Option{
				{Name: "complete", Description: "the user finished"},
				{Name: "short", Description: "a brief pause"},
				{Name: "long", Description: "asked for time"},
			},
		}}})
	if err != nil || !slices.Contains([]string{"complete", "short", "long"}, choice["answer"].Choice) {
		t.Fatalf("turn: %+v, %v", choice, err)
	}
	if c.Client().Usage().InputTokens == 0 {
		t.Fatal("no tokens counted")
	}
}
