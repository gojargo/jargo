package rtvi_test

import (
	"context"
	"encoding/base64"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/rtvi"
)

// handled is what an RTVI processor did with the messages a client sent: what
// it answered them with, and the keys and audio it pushed.
type handled struct {
	// answers are the type of each message sent with an id, by that id, in the
	// order they were sent.
	answers map[string][]string
	// errors counts the general error messages, which answer no request.
	errors int
	keys   []frames.KeypadEntry
	audio  []*frames.InputAudioRawFrame
	// ended reports that the processor ended the worker.
	ended bool
}

// errorResponses counts the error-responses.
func (h handled) errorResponses() int {
	n := 0
	for _, types := range h.answers {
		for _, msgType := range types {
			if msgType == rtvi.TypeErrorResponse {
				n++
			}
		}
	}
	return n
}

// clientSends runs raw client messages through an RTVI processor and returns
// what it did with them.
func clientSends(t *testing.T, raw ...string) handled {
	t.Helper()
	proc := rtvi.NewProcessor()
	var mu sync.Mutex
	h := handled{answers: map[string][]string{}}
	task := pipeline.NewWorker(pipeline.New(proc), pipeline.WorkerConfig{ReachedDownstreamFilter: pipeline.AnyFrame})
	event.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		defer mu.Unlock()
		switch fr := f.(type) {
		case *frames.OutputTransportMessageUrgentFrame:
			m, ok := fr.Message.(rtvi.Message)
			switch {
			case !ok:
			case m.Type == rtvi.TypeError:
				h.errors++
			case m.ID != "":
				h.answers[m.ID] = append(h.answers[m.ID], m.Type)
			}
		case *frames.InputDTMFFrame:
			h.keys = append(h.keys, fr.Button)
		case *frames.InputAudioRawFrame:
			h.audio = append(h.audio, fr)
		case *frames.EndWorkerFrame:
			h.ended = true
		}
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	for _, r := range raw {
		task.QueueFrame(frames.NewInputTransportMessageFrame([]byte(r)))
	}
	time.Sleep(300 * time.Millisecond)
	task.StopWhenDone()
	<-done
	mu.Lock()
	defer mu.Unlock()
	return h
}

// clientMessage is a client message of msgType with id m1, carrying data when
// it is not empty.
func clientMessage(msgType, data string) string {
	if data == "" {
		return `{"label":"rtvi-ai","type":"` + msgType + `","id":"m1"}`
	}
	return `{"label":"rtvi-ai","type":"` + msgType + `","id":"m1","data":` + data + `}`
}

// An invalid key, an empty list, and the legacy single-button field are each
// answered with an error, and a batch is validated whole, so one bad key pushes
// none of them.
func TestDTMFInvalidDataIsAnsweredWithAnErrorAndPushesNothing(t *testing.T) {
	for name, data := range map[string]string{
		"an invalid key":       `{"buttons":["1","Z"]}`,
		"an empty list":        `{"buttons":[]}`,
		"the legacy field":     `{"button":"1"}`,
		"a key outside 0-9*#":  `{"buttons":["A"]}`,
		"two characters a key": `{"buttons":["12"]}`,
	} {
		t.Run(name, func(t *testing.T) {
			h := clientSends(t, clientMessage(rtvi.TypeDTMF, data))
			if h.errorResponses() != 1 {
				t.Errorf("error responses = %d, want 1", h.errorResponses())
			}
			if len(h.keys) != 0 {
				t.Errorf("pushed %v, want no key", h.keys)
			}
		})
	}
}

// TestDTMFValidKeysArePushedInOrder checks a valid batch reaches the pipeline
// key by key.
func TestDTMFValidKeysArePushedInOrder(t *testing.T) {
	h := clientSends(t, clientMessage(rtvi.TypeDTMF, `{"buttons":["1","*","#"]}`))
	if h.errorResponses() != 0 {
		t.Errorf("error responses = %d, want none", h.errorResponses())
	}
	if !reflect.DeepEqual(h.keys, []frames.KeypadEntry{"1", "*", "#"}) {
		t.Errorf("pushed %v, want 1, *, #", h.keys)
	}
}

// A send-text with no content is answered with an error, rather than dropped
// in silence, which would leave a client waiting on its acknowledgement.
func TestSendTextWithoutContentIsAnsweredWithAnError(t *testing.T) {
	h := clientSends(t, clientMessage(rtvi.TypeSendText, `{}`))
	if h.errorResponses() != 1 {
		t.Errorf("error responses = %d, want 1", h.errorResponses())
	}
}

// TestRawAudioPushesInputAudioDownstream checks a raw-audio message reaches the
// pipeline as the user's audio, and draws no answer.
func TestRawAudioPushesInputAudioDownstream(t *testing.T) {
	h := clientSends(t, clientMessage(rtvi.TypeRawAudio, `{"base64Audio":"AAAA","sampleRate":16000,"numChannels":1}`))
	if h.errorResponses() != 0 {
		t.Errorf("error responses = %d, want none", h.errorResponses())
	}
	if len(h.audio) != 1 || h.audio[0].SampleRate != 16000 {
		t.Fatalf("pushed %v, want one frame of 16000 Hz audio", h.audio)
	}
}

// TestRawAudioBatchPushesOneFramePerChunkInOrder checks a batch arrives as one
// frame per chunk, in the order the client captured them.
func TestRawAudioBatchPushesOneFramePerChunkInOrder(t *testing.T) {
	first := base64.StdEncoding.EncodeToString([]byte{1, 0})
	second := base64.StdEncoding.EncodeToString([]byte{2, 0})
	h := clientSends(t, clientMessage(rtvi.TypeRawAudioBatch,
		`{"base64AudioBatch":["`+first+`","`+second+`"],"sampleRate":16000,"numChannels":1}`))
	if h.errorResponses() != 0 {
		t.Errorf("error responses = %d, want none", h.errorResponses())
	}
	var got [][]byte
	for _, f := range h.audio {
		got = append(got, f.Audio)
	}
	if want := [][]byte{{1, 0}, {2, 0}}; !reflect.DeepEqual(got, want) {
		t.Errorf("pushed %v, want %v", got, want)
	}
}

// TestRawAudioWithoutSampleRateGetsAnErrorResponse checks malformed audio is
// answered, so the client knows its audio was not received, and pushes
// nothing.
func TestRawAudioWithoutSampleRateGetsAnErrorResponse(t *testing.T) {
	h := clientSends(t, clientMessage(rtvi.TypeRawAudio, `{"base64Audio":"AAAA","numChannels":1}`))
	if h.errorResponses() != 1 {
		t.Errorf("error responses = %d, want 1", h.errorResponses())
	}
	if len(h.audio) != 0 {
		t.Errorf("pushed %d audio frames, want none", len(h.audio))
	}
}

// TestRawAudioBatchWithABadChunkPushesNothing checks a batch is decoded whole
// before any of it is pushed: a chunk that is not base64 costs the whole batch,
// and the client is told.
func TestRawAudioBatchWithABadChunkPushesNothing(t *testing.T) {
	h := clientSends(t, clientMessage(rtvi.TypeRawAudioBatch,
		`{"base64AudioBatch":["AAAA","not base64!"],"sampleRate":16000,"numChannels":1}`))
	if h.errorResponses() != 1 {
		t.Errorf("error responses = %d, want 1", h.errorResponses())
	}
	if len(h.audio) != 0 {
		t.Errorf("pushed %d audio frames, want none", len(h.audio))
	}
}
