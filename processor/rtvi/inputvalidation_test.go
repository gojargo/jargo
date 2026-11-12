package rtvi_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/utils/events"
)

// clientSends runs a raw client message through an RTVI processor and returns
// the error responses it answered with and the keys it pushed.
func clientSends(t *testing.T, raw string) (errorResponses int, keys []frames.KeypadEntry) {
	t.Helper()
	proc := rtvi.NewProcessor()
	var mu sync.Mutex
	task := pipeline.NewWorker(pipeline.New(proc), pipeline.WorkerConfig{ReachedDownstreamFilter: pipeline.AnyFrame})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		defer mu.Unlock()
		switch fr := f.(type) {
		case *frames.OutputTransportMessageUrgentFrame:
			if m, ok := fr.Message.(rtvi.Message); ok && m.Type == rtvi.TypeErrorResponse {
				errorResponses++
			}
		case *frames.InputDTMFFrame:
			keys = append(keys, fr.Button)
		}
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	task.QueueFrame(frames.NewInputTransportMessageFrame([]byte(raw)))
	time.Sleep(300 * time.Millisecond)
	task.StopWhenDone()
	<-done
	mu.Lock()
	defer mu.Unlock()
	return errorResponses, keys
}

// Ported from upstream (TestRTVIDTMF): an invalid key, an empty list, and the
// legacy single-button field are each answered with an error, and a batch is
// validated whole, so one bad key pushes none of them.
func TestDTMFInvalidDataIsAnsweredWithAnErrorAndPushesNothing(t *testing.T) {
	for name, data := range map[string]string{
		"an invalid key":       `{"buttons":["1","Z"]}`,
		"an empty list":        `{"buttons":[]}`,
		"the legacy field":     `{"button":"1"}`,
		"a key outside 0-9*#":  `{"buttons":["A"]}`,
		"two characters a key": `{"buttons":["12"]}`,
	} {
		t.Run(name, func(t *testing.T) {
			errs, keys := clientSends(t, `{"label":"rtvi-ai","type":"dtmf","id":"m1","data":`+data+`}`)
			if errs != 1 {
				t.Errorf("error responses = %d, want 1", errs)
			}
			if len(keys) != 0 {
				t.Errorf("pushed %v, want no key", keys)
			}
		})
	}
}

// TestDTMFValidKeysArePushedInOrder checks a valid batch reaches the pipeline
// key by key.
func TestDTMFValidKeysArePushedInOrder(t *testing.T) {
	errs, keys := clientSends(t, `{"label":"rtvi-ai","type":"dtmf","id":"m1","data":{"buttons":["1","*","#"]}}`)
	if errs != 0 {
		t.Errorf("error responses = %d, want none", errs)
	}
	if len(keys) != 3 || keys[0] != "1" || keys[1] != "*" || keys[2] != "#" {
		t.Errorf("pushed %v, want 1, *, #", keys)
	}
}

// A send-text with no content is answered with an error, rather than dropped
// in silence, which would leave a client waiting on its acknowledgement.
func TestSendTextWithoutContentIsAnsweredWithAnError(t *testing.T) {
	errs, _ := clientSends(t, `{"label":"rtvi-ai","type":"send-text","id":"m1","data":{}}`)
	if errs != 1 {
		t.Errorf("error responses = %d, want 1", errs)
	}
}
