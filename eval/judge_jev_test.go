package eval_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/classifier/jev"
	"github.com/gojargo/jargo/eval"
	"gopkg.in/yaml.v3"
)

// jevRequest is one request the stand-in Jev was sent.
type jevRequest struct {
	method, path, auth string
	body               map[string]any
}

// jevAPI stands in for Jev's API: it records each request and answers it, from
// a queue or from a function of the request body when the order requests
// arrive in is not fixed. A queued "stall" outlasts the client's timeout.
type jevAPI struct {
	mu        sync.Mutex
	responses []any
	respond   func(body map[string]any) (int, map[string]any)
	requests  []jevRequest
	srv       *httptest.Server
	release   chan struct{}
}

const stall = "stall"

func newJevAPI(t *testing.T, responses ...any) *jevAPI {
	t.Helper()
	api := &jevAPI{responses: responses, release: make(chan struct{})}
	api.srv = httptest.NewServer(http.HandlerFunc(api.handle))
	t.Cleanup(func() {
		close(api.release)
		api.srv.Close()
	})
	return api
}

func (a *jevAPI) handle(w http.ResponseWriter, r *http.Request) {
	data, _ := io.ReadAll(r.Body)
	var body map[string]any
	_ = json.Unmarshal(data, &body)
	a.mu.Lock()
	a.requests = append(a.requests, jevRequest{
		method: r.Method, path: r.URL.Path, auth: r.Header.Get("Authorization"), body: body,
	})
	if r.Method == http.MethodGet {
		a.mu.Unlock()
		_, _ = io.WriteString(w, `{"models": []}`)
		return
	}
	var next any
	if a.respond == nil {
		next = a.responses[0]
		a.responses = a.responses[1:]
	}
	respond := a.respond
	a.mu.Unlock()

	status, payload := http.StatusOK, map[string]any{}
	switch v := next.(type) {
	case string: // stall
		select {
		case <-a.release:
		case <-r.Context().Done():
		}
		return
	case int:
		status = v
	case map[string]any:
		payload = v
	default:
		status, payload = respond(body)
	}
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func (a *jevAPI) all() []jevRequest {
	a.mu.Lock()
	defer a.mu.Unlock()
	return slices.Clone(a.requests)
}

func (a *jevAPI) posts() []jevRequest {
	var out []jevRequest
	for _, r := range a.all() {
		if r.method == http.MethodPost {
			out = append(out, r)
		}
	}
	return out
}

// client is a Jev client sending to the stand-in.
func (a *jevAPI) client(t *testing.T, timeout time.Duration) *jev.Client {
	t.Helper()
	c, err := jev.NewClient(jev.ClientConfig{APIKey: "k", BaseURL: a.srv.URL, Timeout: timeout})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func (a *jevAPI) classifier(t *testing.T) *jev.Classifier {
	t.Helper()
	c, err := jev.New(jev.Config{Client: a.client(t, 500*time.Millisecond)})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func jevChoice(choice string, confidence float64) map[string]any {
	rest := (1 - confidence) / 2
	probabilities := map[string]any{}
	for _, o := range replyOptions {
		probabilities[o] = rest
	}
	probabilities[choice] = confidence
	return map[string]any{"type": "choice", "choice": choice, "confidence": confidence, "probabilities": probabilities}
}

func jevNoul(p float64) map[string]any { return map[string]any{"type": "noul", "noul": p} }

func jevTurn(meets float64) map[string]any {
	probabilities := map[string]any{"meets": meets, "fails": 1 - meets, "not_applicable": 0.0}
	choice, confidence := "meets", meets
	if 1-meets > meets {
		choice, confidence = "fails", 1-meets
	}
	return map[string]any{"type": "choice", "choice": choice, "confidence": confidence, "probabilities": probabilities}
}

func jevOK(answers map[string]any) map[string]any {
	return map[string]any{"model": "jev-1", "answers": answers, "usage": map[string]any{}}
}

func jevJudge(t *testing.T, api *jevAPI) *eval.ClassifierJudge {
	t.Helper()
	return newJudge(t, api.classifier(t), nil)
}

// configJudge is the judge a judge.eval block builds over the release suite's
// Jev factory, sending to api.
func configJudge(t *testing.T, api *jevAPI, block string) (*eval.ClassifierJudge, error) {
	t.Helper()
	t.Setenv("TYPESAFE_API_KEY", "k")
	var cfg eval.JudgeEvalConfig
	if err := yaml.Unmarshal(
		[]byte("factory: evals.judges.typesafe_classifier\nendpoint: "+api.srv.URL+"/\n"+block), &cfg); err != nil {
		t.Fatal(err)
	}
	return eval.NewClassifierJudgeFromConfig(&cfg)
}

func TestJevAReplyIsAskedAsAChoiceOverTheConversation(t *testing.T) {
	api := newJevAPI(t, jevOK(map[string]any{"verdict": jevChoice("yes", 0.95)}))
	j := jevJudge(t, api)
	j.AddUserMessage("What's the weather?")
	j.AddAssistantMessage("Let me check.")
	j.AddAssistantMessage("It's 72 and sunny.")

	v := j.Evaluate(t.Context(), "describes the weather", nil)

	if !v.Passed() || *v.Confidence != 0.95 || !strings.Contains(v.Reason, "P(yes)=0.95") {
		t.Fatalf("verdict = %+v", v)
	}
	r := api.posts()[0]
	if r.path != "/v1/systemone" || r.auth != "Bearer k" {
		t.Fatalf("request = %s %s", r.path, r.auth)
	}
	state, _ := r.body["state"].(map[string]any)
	convo, _ := json.Marshal(state["conversation"])
	if state["latest_bot_reply"] != "Let me check. It's 72 and sunny." ||
		string(convo) != `[{"speaker":"user","text":"What's the weather?"}]` {
		t.Fatalf("state = %v", state)
	}
	questions, _ := r.body["questions"].(map[string]any)
	q, _ := questions["verdict"].(map[string]any)
	criteria, _ := q["criteria"].(map[string]any)
	if q["type"] != "choice" || len(criteria) != 3 || criteria["continue"] == nil {
		t.Fatalf("question = %v", q)
	}
}

func TestJevACallIsAskedAsAYesOrNo(t *testing.T) {
	api := newJevAPI(t, jevOK(map[string]any{"answer": jevNoul(0.2)}))
	j := jevJudge(t, api)
	j.AddUserMessage("Book six o'clock.")
	v := j.EvaluateCall(t.Context(), "book", map[string]any{"time": "7pm"}, "books six o'clock")
	if v.Verdict != eval.VerdictNo || !near(*v.Confidence, 0.8) {
		t.Fatalf("verdict = %+v", v)
	}
	body := api.posts()[0].body
	state, _ := body["state"].(map[string]any)
	call, _ := json.Marshal(state["call"])
	questions, _ := body["questions"].(map[string]any)
	answer, _ := questions["answer"].(map[string]any)
	if string(call) != `{"arguments":{"time":"7pm"},"name":"book"}` || answer["type"] != "noul" {
		t.Fatalf("body = %v", body)
	}
}

func TestJevARunAsksTheGoalAndEveryTurnAtOnce(t *testing.T) {
	api := newJevAPI(t)
	api.respond = func(body map[string]any) (int, map[string]any) {
		state, _ := body["state"].(map[string]any)
		reply, ok := state["latest_bot_reply"].(string)
		if !ok {
			return http.StatusOK, jevOK(map[string]any{"answer": jevNoul(0.9)})
		}
		meets := 0.1
		if reply == "Sure." {
			meets = 0.9
		}
		answers := map[string]any{}
		questions, _ := body["questions"].(map[string]any)
		for name := range questions {
			answers[name] = jevTurn(meets)
		}
		return http.StatusOK, jevOK(answers)
	}
	j := jevJudge(t, api)
	j.AddUserMessage("Book a table at six.")
	j.AddAssistantMessage("Sure.")
	j.AddUserMessage("Thanks.")
	j.AddAssistantMessage("Bye.")

	verdicts := j.EvaluateRun(t.Context(), polite, "a table is booked")

	if !verdicts.Goal.Passed() || !slices.Equal(verdictsOf(verdicts.Turns["polite"]), []string{"yes", "no"}) {
		t.Fatalf("verdicts = %+v", verdicts)
	}
	posts := api.posts()
	if len(posts) != 3 {
		t.Fatalf("%d questions, want 3", len(posts))
	}
	for _, p := range posts {
		state, _ := p.body["state"].(map[string]any)
		if _, ok := state["latest_bot_reply"]; !ok {
			continue
		}
		questions, _ := p.body["questions"].(map[string]any)
		q, _ := questions["polite"].(map[string]any)
		criteria, _ := q["criteria"].(map[string]any)
		if len(criteria) != 3 || criteria["meets"] == nil || criteria["fails"] == nil || criteria["not_applicable"] == nil {
			t.Fatalf("turn question = %v", q)
		}
		return
	}
}

func jevAsk(t *testing.T, j *eval.ClassifierJudge) eval.JudgeVerdict {
	t.Helper()
	j.AddAssistantMessage("It rains.")
	return j.Evaluate(t.Context(), "mentions weather", nil)
}

func TestJevATimeoutIsAskedAgain(t *testing.T) {
	api := newJevAPI(t, stall, jevOK(map[string]any{"verdict": jevChoice("yes", 0.95)}))
	if v := jevAsk(t, jevJudge(t, api)); !v.Passed() || len(api.posts()) != 2 {
		t.Fatalf("verdict = %+v after %d posts", v, len(api.posts()))
	}
}

func TestJevARejectedQuestionIsAskedAgain(t *testing.T) {
	api := newJevAPI(t, http.StatusServiceUnavailable, jevOK(map[string]any{"verdict": jevChoice("yes", 0.95)}))
	if v := jevAsk(t, jevJudge(t, api)); !v.Passed() || len(api.posts()) != 2 {
		t.Fatalf("verdict = %+v after %d posts", v, len(api.posts()))
	}
}

func TestJevASecondFailureFailsTheQuestion(t *testing.T) {
	api := newJevAPI(t, stall, stall)
	v := jevAsk(t, jevJudge(t, api))
	if v.Verdict != eval.VerdictNone || !strings.HasPrefix(v.Reason, "judge call failed: ") || len(api.posts()) != 2 {
		t.Fatalf("verdict = %+v after %d posts", v, len(api.posts()))
	}
}

func TestJevTheConnectionIsOpenedWhenTheJudgeIsCreated(t *testing.T) {
	api := newJevAPI(t, jevOK(map[string]any{"verdict": jevChoice("yes", 0.95)}))
	j, err := configJudge(t, api, "explainer: false\n")
	if err != nil {
		t.Fatal(err)
	}
	jevAsk(t, j)
	all := api.all()
	if len(all) != 2 || all[0].method != http.MethodGet || all[0].path != "/v1/models" ||
		all[1].method != http.MethodPost {
		t.Fatalf("requests = %+v, want the models listed, then the question", all)
	}
	_ = j.Close(t.Context())
}

// closedClient reports whether a Jev client refuses to ask, as a closed one
// does.
func closedClient(t *testing.T, c *jev.Client) bool {
	t.Helper()
	_, _, err := c.Ask(t.Context(), "x", nil)
	return err != nil && strings.Contains(err.Error(), "closed")
}

func TestJevCloseReleasesAConnectionTheJudgeOpened(t *testing.T) {
	api := newJevAPI(t)
	j, err := configJudge(t, api, "explainer: false\n")
	if err != nil {
		t.Fatal(err)
	}
	if err := j.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	c, _ := j.Classifier().(*jev.Classifier)
	if c == nil || !closedClient(t, c.Client()) {
		t.Fatal("the connection the judge opened is still open")
	}
}

func TestJevCloseLeavesAConnectionTheJudgeWasGivenOpen(t *testing.T) {
	api := newJevAPI(t)
	c := api.classifier(t)
	j, err := eval.NewClassifierJudge(eval.ClassifierJudgeConfig{Classifier: c})
	if err != nil {
		t.Fatal(err)
	}
	if err := j.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	// The classifier was given its client; only the one it created is closed.
	if closedClient(t, c.Client()) {
		t.Fatal("a connection the judge was given was closed")
	}
}

func TestJevServiceTypesafeJudgesWithJevAndExplainsWithAnLLM(t *testing.T) {
	j, err := configJudge(t, newJevAPI(t), "model: jev-9\n")
	if err != nil {
		t.Fatal(err)
	}
	c, ok := j.Classifier().(*jev.Classifier)
	if !ok || c.Model() != "jev-9" || c.Client().Timeout() != eval.SuiteClassifierTimeout {
		t.Fatalf("classifier = %T %+v", j.Classifier(), c)
	}
	_ = j.Close(t.Context())
}

func TestJevAMissingAPIKeyIsAnError(t *testing.T) {
	t.Setenv("TYPESAFE_API_KEY", "")
	_, err := eval.NewClassifierJudgeFromConfig(&eval.JudgeEvalConfig{
		Factory: "evals.judges.typesafe_classifier",
	})
	if err == nil || !strings.Contains(err.Error(), "TYPESAFE_API_KEY") {
		t.Fatalf("err = %v", err)
	}
}
