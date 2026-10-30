package responses

import (
	"context"
	"errors"
	"maps"
	"slices"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// The events a response sends on a lane.

func laneCreated(lane, responseID string) string {
	return `{"type":"response.created","stream_id":"` + lane + `","response":{"id":"` + responseID + `"}}`
}

func laneDelta(lane, text string) string {
	return `{"type":"response.output_text.delta","stream_id":"` + lane + `","delta":"` + text + `"}`
}

func laneCompleted(lane, responseID string) string {
	return `{"type":"response.completed","stream_id":"` + lane + `","response":{"id":"` + responseID + `"}}`
}

// newLaneService builds a service talking to a fake endpoint that answers the
// Nth request with the Nth script.
func newLaneService(t *testing.T, scripts ...[]string) (*Service, *fakeResponsesWS) {
	t.Helper()
	srv := newFakeResponsesWS(t, scripts)
	svc := NewLLM(Config{APIKey: "k", WSURL: srv.wsURL()})
	t.Cleanup(svc.disconnect)
	return svc, srv
}

// generate runs one turn over convo and returns the text it produced.
func generate(t *testing.T, svc *Service, convo *frames.LLMContext) string {
	t.Helper()
	var text strings.Builder
	if err := svc.Generate(context.Background(), convo, func(s string) error {
		text.WriteString(s)
		return nil
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	return text.String()
}

// generateInterrupted runs one turn over convo and interrupts it at its first
// text.
func generateInterrupted(t *testing.T, svc *Service, convo *frames.LLMContext) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	err := svc.Generate(ctx, convo, func(string) error {
		cancel()
		return nil
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("interrupted Generate = %v, want context.Canceled", err)
	}
}

// drainingLanes is the lanes the service holds as draining, sorted.
func drainingLanes(svc *Service) []string {
	svc.mu.Lock()
	defer svc.mu.Unlock()
	return slices.Sorted(maps.Keys(svc.draining))
}

// TestWSNextResponseDoesNotWaitForAnAbandonedOne checks an interrupted response
// is abandoned on its lane: the next goes out at once on another, continuing
// from the last completed response, and the abandoned response's events are
// discarded.
func TestWSNextResponseDoesNotWaitForAnAbandonedOne(t *testing.T) {
	svc, srv := newLaneService(t,
		[]string{laneCreated("lane-0", "resp_1"), laneDelta("lane-0", "Hello"), laneCompleted("lane-0", "resp_1")},
		// Started, then left running: only the interruption ends this turn.
		[]string{laneCreated("lane-1", "resp_2"), laneDelta("lane-1", "partial")},
		[]string{
			laneDelta("lane-1", "stale"),
			laneCreated("lane-2", "resp_3"),
			laneCompleted("lane-1", "resp_2"),
			laneDelta("lane-2", "fresh"),
			laneCompleted("lane-2", "resp_3"),
		},
	)

	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")
	generate(t, svc, convo)

	convo.AddAssistantMessage("Hello")
	convo.AddUserMessage("go")
	generateInterrupted(t, svc, convo)

	if got := generate(t, svc, convo); got != "fresh" {
		t.Errorf("text = %q, want only the new response's", got)
	}
	if got, want := srv.streamIDs(), []string{"lane-0", "lane-1", "lane-2"}; !slices.Equal(got, want) {
		t.Errorf("lanes = %v, want %v", got, want)
	}
	if got := srv.turn(2)["previous_response_id"]; got != "resp_1" {
		t.Errorf("previous_response_id = %v, want resp_1, the last completed response", got)
	}
	if got := srv.messageTypes(); !slices.Equal(got, []string{"response.create", "response.create", "response.create"}) {
		t.Errorf("messages = %v, want only the three requests", got)
	}
	svc.mu.Lock()
	responseID := svc.cont.responseID
	svc.mu.Unlock()
	if responseID != "resp_3" {
		t.Errorf("continuing from %q, want resp_3", responseID)
	}
	if lanes := drainingLanes(svc); len(lanes) != 0 {
		t.Errorf("draining lanes = %v, want none once the abandoned response ended", lanes)
	}
}

// TestWSResponsesAlternateLanes checks each response goes on a lane other than
// the previous response's.
func TestWSResponsesAlternateLanes(t *testing.T) {
	svc, srv := newLaneService(t,
		[]string{laneDelta("lane-0", "one"), laneCompleted("lane-0", "resp_1")},
		[]string{laneDelta("lane-1", "two"), laneCompleted("lane-1", "resp_2")},
		[]string{laneDelta("lane-0", "three"), laneCompleted("lane-0", "resp_3")},
	)

	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")
	generate(t, svc, convo)
	convo.AddAssistantMessage("one")
	convo.AddUserMessage("again")
	generate(t, svc, convo)
	convo.AddAssistantMessage("two")
	convo.AddUserMessage("more")
	generate(t, svc, convo)

	if got, want := srv.streamIDs(), []string{"lane-0", "lane-1", "lane-0"}; !slices.Equal(got, want) {
		t.Errorf("lanes = %v, want %v", got, want)
	}
	var previous []any
	for n := range 3 {
		previous = append(previous, srv.turn(n)["previous_response_id"])
	}
	if want := []any{nil, "resp_1", "resp_2"}; !slices.Equal(previous, want) {
		t.Errorf("previous_response_id = %v, want %v", previous, want)
	}
}

// TestWSCancellationKeepsTheConnection checks an interrupted response is
// abandoned on its lane, with no request to the server, and the connection is
// kept.
func TestWSCancellationKeepsTheConnection(t *testing.T) {
	svc, srv := newLaneService(t,
		[]string{laneCreated("lane-0", "resp_1"), laneDelta("lane-0", "partial")},
	)

	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")
	generateInterrupted(t, svc, convo)

	svc.mu.Lock()
	sess := svc.sess
	svc.mu.Unlock()
	if sess == nil {
		t.Error("the connection was dropped")
	}
	if got := drainingLanes(svc); !slices.Equal(got, []string{"lane-0"}) {
		t.Errorf("draining lanes = %v, want [lane-0]", got)
	}
	if got := srv.messageTypes(); !slices.Equal(got, []string{"response.create"}) {
		t.Errorf("messages = %v, want only the request", got)
	}
}

// TestWSAnAbandonedResponsesErrorIsNotReported checks an error on a draining
// lane ends that lane's response and nothing else.
func TestWSAnAbandonedResponsesErrorIsNotReported(t *testing.T) {
	svc, _ := newLaneService(t,
		[]string{
			`{"type":"error","stream_id":"lane-1","message":"stale"}`,
			laneDelta("lane-0", "Hello"),
			laneCompleted("lane-0", "resp_1"),
		},
	)
	svc.mu.Lock()
	svc.draining = map[string]bool{"lane-1": true}
	svc.mu.Unlock()

	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")
	if got := generate(t, svc, convo); got != "Hello" {
		t.Errorf("text = %q, want Hello", got)
	}
	if lanes := drainingLanes(svc); len(lanes) != 0 {
		t.Errorf("draining lanes = %v, want none", lanes)
	}
}

// TestWSEveryLaneTakenReconnects checks that with every lane holding an
// abandoned response or the previous response, the connection is closed, which
// ends the abandoned ones.
func TestWSEveryLaneTakenReconnects(t *testing.T) {
	svc, _ := newLaneService(t,
		[]string{laneDelta("lane-0", "hi"), laneCompleted("lane-0", "resp_1")},
	)
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")
	generate(t, svc, convo)

	svc.mu.Lock()
	if svc.cont.responseID != "resp_1" || svc.cont.lane != "lane-0" {
		t.Fatalf("continuation = %+v, want resp_1 on lane-0", svc.cont)
	}
	svc.draining = map[string]bool{}
	for i := 1; i < laneCount; i++ {
		svc.draining[laneName(i)] = true
	}
	svc.mu.Unlock()

	if got := svc.chooseNextLane(); got != "lane-0" {
		t.Errorf("lane = %q, want lane-0", got)
	}
	svc.mu.Lock()
	defer svc.mu.Unlock()
	if svc.sess != nil {
		t.Error("the connection was kept")
	}
	if svc.cont.responseID != "" {
		t.Errorf("still continuing from %q", svc.cont.responseID)
	}
	if len(svc.draining) != 0 {
		t.Errorf("draining lanes = %v, want none", svc.draining)
	}
}
