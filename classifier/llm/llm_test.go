package llm_test

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/classifier/llm"
	"github.com/gojargo/jargo/frames"
	llmservice "github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/utils/events"
)

// A scripted LLM stands in for the model: RunInference returns the next
// scripted reply, so the tests exercise the question rendering, the JSON parsing
// and the results.

type request struct {
	message     string
	instruction string
	schema      json.RawMessage
}

type scriptedLLM struct {
	mu       sync.Mutex
	replies  []string
	requests []request
}

func (s *scriptedLLM) RunInference(
	_ context.Context, convo *frames.LLMContext, opts llmservice.InferenceOptions,
) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	msgs := convo.Messages()
	s.requests = append(s.requests, request{
		message: msgs[len(msgs)-1].Text, instruction: opts.SystemInstruction, schema: opts.ResponseSchema,
	})
	reply := s.replies[0]
	s.replies = s.replies[1:]
	return reply, nil
}

// failingLLM fails RunInference the way a provider client does.
type failingLLM struct{}

//nolint:err113 // a provider's own failure, as it reports it
func (failingLLM) RunInference(context.Context, *frames.LLMContext, llmservice.InferenceOptions) (string, error) {
	return "", errors.New("connection reset")
}

// silentLLM never answers RunInference.
type silentLLM struct{}

func (silentLLM) RunInference(
	ctx context.Context, _ *frames.LLMContext, _ llmservice.InferenceOptions,
) (string, error) {
	<-ctx.Done()
	return "", ctx.Err()
}

func newClassifier(t *testing.T, cfg llm.Config, replies ...string) (*llm.Classifier, *scriptedLLM) {
	t.Helper()
	scripted := &scriptedLLM{replies: replies}
	if cfg.LLM == nil {
		cfg.LLM = scripted
	}
	c, err := llm.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return c, scripted
}

func yesNo(name, instructions string) []classifier.Named[classifier.YesNoQuestion] {
	return []classifier.Named[classifier.YesNoQuestion]{
		{Name: name, Question: classifier.YesNoQuestion{Instructions: instructions}},
	}
}

func choice(name string, q classifier.ChoiceQuestion) []classifier.Named[classifier.ChoiceQuestion] {
	return []classifier.Named[classifier.ChoiceQuestion]{{Name: name, Question: q}}
}

func score(name string, q classifier.ScoreQuestion) []classifier.Named[classifier.ScoreQuestion] {
	return []classifier.Named[classifier.ScoreQuestion]{{Name: name, Question: q}}
}

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestYesNoAnswersFromTheJSONReply(t *testing.T) {
	c, _ := newClassifier(t, llm.Config{}, `{"greeting": {"probability": 0.93}}`)
	results, err := c.YesNo(t.Context(), "Hello?", yesNo("greeting", "a greeting?"))
	if err != nil {
		t.Fatal(err)
	}
	if results["greeting"].Probability != 0.93 {
		t.Fatalf("probability = %v, want 0.93", results["greeting"].Probability)
	}
}

func TestChoiceAnswersWithAProbabilityPerOption(t *testing.T) {
	c, _ := newClassifier(t, llm.Config{},
		`{"turn": {"choice": "short", "probabilities": {"short": 0.8, "complete": 0.2}}}`)
	q := classifier.ChoiceQuestion{
		Instructions: "is the turn over?",
		Options: []classifier.Option{
			{Name: "complete", Description: "finished"},
			{Name: "short", Description: "cut off"},
			{Name: "long", Description: "needs time"},
		},
	}
	results, err := c.Choice(t.Context(), "I'd like to", choice("turn", q))
	if err != nil {
		t.Fatal(err)
	}
	r := results["turn"]
	if r.Choice != "short" || r.Confidence != 0.8 {
		t.Fatalf("got %+v, want short at 0.8", r)
	}
	want := map[string]float64{"complete": 0.2, "short": 0.8, "long": 0}
	for k, v := range want {
		if r.Probabilities[k] != v {
			t.Fatalf("probabilities = %v, want %v", r.Probabilities, want)
		}
	}
}

func TestScoreAnswersFromAProbabilityPerLevel(t *testing.T) {
	c, _ := newClassifier(t, llm.Config{}, `{"mood": {"probabilities": {"0": 0.0, "1": 0.3, "2": 0.7}}}`)
	q := classifier.ScoreQuestion{Instructions: "how upset?", Levels: []any{"calm", "impatient", "frustrated"}}
	results, err := c.Score(t.Context(), "This is the third time!", score("mood", q))
	if err != nil {
		t.Fatal(err)
	}
	r := results["mood"]
	if !near(r.Score, 1.7) || r.Confidence != 0.7 {
		t.Fatalf("got %+v, want a score of 1.7 at 0.7", r)
	}
	want := []classifier.ScoreLevel{
		{Level: "calm", Probability: 0},
		{Level: "impatient", Probability: 0.3},
		{Level: "frustrated", Probability: 0.7},
	}
	if !slices.Equal(r.Levels, want) {
		t.Fatalf("levels = %v, want %v", r.Levels, want)
	}
	if p, err := r.Probability("frustrated"); err != nil || p != 0.7 {
		t.Fatalf("Probability(frustrated) = %v, %v; want 0.7", p, err)
	}
}

func TestScoreProbabilitiesAreScaledToSumToOne(t *testing.T) {
	c, _ := newClassifier(t, llm.Config{}, `{"mood": {"probabilities": {"0": 0.5, "1": 1.0}}}`)
	q := classifier.ScoreQuestion{Instructions: "how upset?", Levels: []any{"calm", "upset"}}
	results, err := c.Score(t.Context(), "hmm", score("mood", q))
	if err != nil {
		t.Fatal(err)
	}
	r := results["mood"]
	if !near(r.Levels[0].Probability, 1.0/3) || !near(r.Levels[1].Probability, 2.0/3) || !near(r.Score, 2.0/3) {
		t.Fatalf("got %+v, want the levels scaled to 1/3 and 2/3", r)
	}
}

func TestAScoreWithoutAnyProbabilityIsAnError(t *testing.T) {
	c, _ := newClassifier(t, llm.Config{}, `{"mood": {"probabilities": {}}}`)
	q := classifier.ScoreQuestion{Instructions: "how upset?", Levels: []any{"calm", "upset"}}
	if _, err := c.Score(t.Context(), "hmm", score("mood", q)); !errors.Is(err, classifier.ErrClassifier) {
		t.Fatalf("err = %v, want a classifier error", err)
	}
}

func TestSeveralQuestionsGoInOneCall(t *testing.T) {
	c, scripted := newClassifier(t, llm.Config{},
		`{"greeting": {"probability": 0.9}, "mood": {"probabilities": {"0": 1.0, "1": 0.0}}}`)
	results, err := c.Ask(t.Context(), "Hello there!", []classifier.Named[classifier.Question]{
		{Name: "greeting", Question: classifier.YesNoQuestion{Instructions: "a greeting?"}},
		{Name: "mood", Question: classifier.ScoreQuestion{Instructions: "how upset?", Levels: []any{"calm", "upset"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(scripted.requests) != 1 {
		t.Fatalf("made %d calls, want one", len(scripted.requests))
	}
	greeting, ok1 := results["greeting"].(classifier.YesNoResult)
	mood, ok2 := results["mood"].(classifier.ScoreResult)
	if !ok1 || !ok2 || greeting.Probability != 0.9 || mood.Score != 0 {
		t.Fatalf("got %+v", results)
	}
}

func TestTheQuestionIsRenderedWithItsOptionsAndTheReplyShape(t *testing.T) {
	c, scripted := newClassifier(t, llm.Config{}, `{"which": {"choice": "a"}}`)
	_, err := c.Choice(t.Context(), map[string]any{"assistant": "Where to?", "user": "japan"},
		choice("which", classifier.ChoiceQuestion{
			Instructions: "which?",
			Options:      []classifier.Option{{Name: "a", Description: "first"}, {Name: "b"}},
		}))
	if err != nil {
		t.Fatal(err)
	}
	msg, instruction := scripted.requests[0].message, scripted.requests[0].instruction
	for _, want := range []string{
		`Question "which": which?`,
		"- a: first\n- b\n",
		`Reply with one JSON object of this shape: {"which": <answer to "which">}`,
	} {
		if !strings.Contains(msg, want) {
			t.Errorf("message lacks %q:\n%s", want, msg)
		}
	}
	if !strings.HasPrefix(msg, "Input:\n{\n  \"assistant\": \"Where to?\"") {
		t.Errorf("message does not start with the state as JSON:\n%s", msg)
	}
	if !strings.Contains(instruction, "single JSON object") {
		t.Errorf("instruction = %q, want the default one", instruction)
	}
}

func TestYesAndNoMeaningsAreWrittenIntoTheQuestion(t *testing.T) {
	c, scripted := newClassifier(t, llm.Config{}, `{"q": {"probability": 0.1}}`)
	_, err := c.YesNo(t.Context(), "Hello?", []classifier.Named[classifier.YesNoQuestion]{{
		Name:     "q",
		Question: classifier.YesNoQuestion{Instructions: "is this a voicemail?", Yes: "a recording", No: "a person"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	msg := scripted.requests[0].message
	if !strings.Contains(msg, "Yes means: a recording") || !strings.Contains(msg, "No means: a person") {
		t.Fatalf("message lacks the meanings:\n%s", msg)
	}
}

func TestCustomInstructionsReachTheLLM(t *testing.T) {
	c, scripted := newClassifier(t, llm.Config{Instructions: "Decide."}, `{"q": {"probability": 0.5}}`)
	if _, err := c.YesNo(t.Context(), "hi", yesNo("q", "?")); err != nil {
		t.Fatal(err)
	}
	if got := scripted.requests[0].instruction; got != "Decide." {
		t.Fatalf("instruction = %q, want Decide.", got)
	}
}

func TestAFencedReplyIsParsed(t *testing.T) {
	c, _ := newClassifier(t, llm.Config{}, "Sure:\n```json\n{\"q\": {\"probability\": 0.4}}\n```\n")
	results, err := c.YesNo(t.Context(), "hi", yesNo("q", "?"))
	if err != nil || results["q"].Probability != 0.4 {
		t.Fatalf("got %+v, %v; want 0.4", results, err)
	}
}

// schemaOf decodes the reply schema of the first request.
func schemaOf(t *testing.T, s *scriptedLLM) map[string]any {
	t.Helper()
	var schema map[string]any
	if err := json.Unmarshal(s.requests[0].schema, &schema); err != nil {
		t.Fatal(err)
	}
	return schema
}

func asMap(t *testing.T, v any) map[string]any {
	t.Helper()
	m, ok := v.(map[string]any)
	if !ok {
		t.Fatalf("got %T, want an object", v)
	}
	return m
}

func stringsOf(t *testing.T, v any) []string {
	t.Helper()
	list, ok := v.([]any)
	if !ok {
		t.Fatalf("got %T, want an array", v)
	}
	out := make([]string, 0, len(list))
	for _, e := range list {
		s, ok := e.(string)
		if !ok {
			t.Fatalf("got %T, want a string", e)
		}
		out = append(out, s)
	}
	return out
}

func required(t *testing.T, schema any) []string {
	t.Helper()
	return stringsOf(t, asMap(t, schema)["required"])
}

func props(t *testing.T, schema any) map[string]any {
	t.Helper()
	return asMap(t, asMap(t, schema)["properties"])
}

func TestTheReplySchemaHasOneAnswerPerQuestion(t *testing.T) {
	c, scripted := newClassifier(t, llm.Config{},
		`{"q": {"probability": 0.5}, "w": {"choice": "a", "probabilities": {"a": 1, "b": 0}}}`)
	_, err := c.Ask(t.Context(), "hi", []classifier.Named[classifier.Question]{
		{Name: "q", Question: classifier.YesNoQuestion{Instructions: "?"}},
		{Name: "w", Question: classifier.ChoiceQuestion{
			Instructions: "?", Options: []classifier.Option{{Name: "a", Description: ""}, {Name: "b"}},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	schema := schemaOf(t, scripted)
	if got := required(t, schema); !slices.Equal(got, []string{"q", "w"}) {
		t.Errorf("required = %v, want [q w]", got)
	}
	if schema["additionalProperties"] != false {
		t.Errorf("additionalProperties = %v, want false", schema["additionalProperties"])
	}
	if got := required(t, props(t, schema)["q"]); !slices.Equal(got, []string{"probability"}) {
		t.Errorf("q required = %v, want [probability]", got)
	}
	w := props(t, props(t, schema)["w"])
	enum := stringsOf(t, asMap(t, w["choice"])["enum"])
	if !slices.Equal(enum, []string{"a", "b"}) {
		t.Errorf("enum = %v, want [a b]", enum)
	}
	if got := required(t, w["probabilities"]); !slices.Equal(got, []string{"a", "b"}) {
		t.Errorf("probabilities required = %v, want [a b]", got)
	}
}

func TestTheScoreSchemaAsksForAProbabilityPerLevel(t *testing.T) {
	c, scripted := newClassifier(t, llm.Config{}, `{"mood": {"probabilities": {"0": 0.5, "1": 0.5}}}`)
	q := classifier.ScoreQuestion{Instructions: "?", Levels: []any{"a", "b"}}
	if _, err := c.Score(t.Context(), "hi", score("mood", q)); err != nil {
		t.Fatal(err)
	}
	mood := props(t, schemaOf(t, scripted))["mood"]
	if got := required(t, mood); !slices.Equal(got, []string{"probabilities"}) {
		t.Errorf("mood required = %v, want [probabilities]", got)
	}
	if got := required(t, props(t, mood)["probabilities"]); !slices.Equal(got, []string{"0", "1"}) {
		t.Errorf("probabilities required = %v, want [0 1]", got)
	}
}

func twoOptions() classifier.ChoiceQuestion {
	return classifier.ChoiceQuestion{
		Instructions: "?", Options: []classifier.Option{{Name: "a", Description: ""}, {Name: "b", Description: ""}},
	}
}

func TestAChoiceWithoutProbabilitiesHasNoConfidence(t *testing.T) {
	c, _ := newClassifier(t, llm.Config{}, `{"q": {"choice": "b"}}`)
	results, err := c.Choice(t.Context(), "hi", choice("q", twoOptions()))
	if err != nil {
		t.Fatal(err)
	}
	r := results["q"]
	if r.Choice != "b" || r.Confidence != 0 || r.Probabilities["a"] != 0 || r.Probabilities["b"] != 0 {
		t.Fatalf("got %+v, want b with no confidence", r)
	}
}

func TestALoneAnswerWithoutItsNameIsAccepted(t *testing.T) {
	c, _ := newClassifier(t, llm.Config{}, `{"choice": "b", "probabilities": {"a": 0.2, "b": 0.8}}`)
	results, err := c.Choice(t.Context(), "hi", choice("q", twoOptions()))
	if err != nil || results["q"].Choice != "b" {
		t.Fatalf("got %+v, %v; want b", results, err)
	}
}

func TestAFailedLLMCallIsAClassifierError(t *testing.T) {
	c, _ := newClassifier(t, llm.Config{LLM: failingLLM{}})
	_, err := c.YesNo(t.Context(), "hi", yesNo("q", "?"))
	if !errors.Is(err, classifier.ErrClassifier) || !strings.Contains(err.Error(), "connection reset") {
		t.Fatalf("err = %v, want a classifier error carrying the cause", err)
	}
}

func TestATextReplyIsAnError(t *testing.T) {
	c, _ := newClassifier(t, llm.Config{}, "I think yes.")
	_, err := c.YesNo(t.Context(), "hi", yesNo("q", "?"))
	if !errors.Is(err, classifier.ErrClassifier) || !strings.Contains(err.Error(), "JSON object") {
		t.Fatalf("err = %v, want a classifier error about the JSON object", err)
	}
}

func TestAMissingAnswerIsAnError(t *testing.T) {
	c, _ := newClassifier(t, llm.Config{}, `{"other": {"probability": 0.5}}`)
	_, err := c.YesNo(t.Context(), "hi", []classifier.Named[classifier.YesNoQuestion]{
		{Name: "a", Question: classifier.YesNoQuestion{Instructions: "?"}},
		{Name: "b", Question: classifier.YesNoQuestion{Instructions: "?"}},
	})
	if !errors.Is(err, classifier.ErrClassifier) || !strings.Contains(err.Error(), `no answer for "a"`) {
		t.Fatalf("err = %v, want no answer for a", err)
	}
}

func TestAChoiceOutsideTheOptionsIsAnError(t *testing.T) {
	c, _ := newClassifier(t, llm.Config{}, `{"q": {"choice": "c"}}`)
	_, err := c.Choice(t.Context(), "hi", choice("q", twoOptions()))
	if !errors.Is(err, classifier.ErrClassifier) || !strings.Contains(err.Error(), "not an option") {
		t.Fatalf("err = %v, want not an option", err)
	}
}

func TestAReplyThatDoesNotComeInTimeIsAnError(t *testing.T) {
	c, _ := newClassifier(t, llm.Config{LLM: silentLLM{}, Timeout: 50 * time.Millisecond})
	_, err := c.YesNo(t.Context(), "hello", yesNo("answer", "a greeting?"))
	if !errors.Is(err, classifier.ErrClassifier) || !strings.Contains(err.Error(), "did not answer within") {
		t.Fatalf("err = %v, want did not answer within", err)
	}
}

func TestAServiceIsRequired(t *testing.T) {
	if _, err := llm.New(llm.Config{}); err == nil {
		t.Fatal("a classifier with no LLM was built")
	}
}

// metricsOf asks one question and returns what the classifier reported.
func metricsOf(t *testing.T, c *llm.Classifier) []frames.MetricsData {
	t.Helper()
	got := make(chan []frames.MetricsData, 1)
	events.On(c.Events(), classifier.EventMetrics, func(_ context.Context, data []frames.MetricsData) {
		got <- data
	})
	if _, err := c.YesNo(t.Context(), "hello", yesNo("answer", "a greeting?")); err != nil {
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

func TestEveryCallReportsItsTimeWithoutTokens(t *testing.T) {
	c, _ := newClassifier(t, llm.Config{}, `{"answer": {"probability": 0.9}}`)
	data := metricsOf(t, c)
	if len(data) != 1 {
		t.Fatalf("got %d metrics, want the processing time alone", len(data))
	}
	processing, ok := data[0].(frames.ProcessingMetricsData)
	if !ok {
		t.Fatalf("got %T, want processing metrics", data[0])
	}
	if processing.Processor != c.Name() || processing.Model != "" || processing.Value < 0 {
		t.Fatalf("got %+v", processing)
	}
}

func TestANamedClassifierReportsMetricsUnderItsName(t *testing.T) {
	c, _ := newClassifier(t, llm.Config{Name: "voicemail"}, `{"answer": {"probability": 0.9}}`)
	data := metricsOf(t, c)
	if c.Name() != "voicemail" || data[0].MetricsProcessor() != "voicemail" {
		t.Fatalf("name = %q, metrics under %q; want voicemail", c.Name(), data[0].MetricsProcessor())
	}
}
