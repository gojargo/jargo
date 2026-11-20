package heygen

import (
	"bytes"
	"context"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/utils/events"
)

// envelope is HeyGen's error envelope for a voice it does not know.
const envelope = `{"error":{"code":"invalid_parameter","message":"Voice not found","request_id":"req-7"}}`

// step is one thing a test does to the running pipeline: queue a frame, wait,
// or run a function.
type step struct {
	frame frames.Frame
	sleep time.Duration
	run   func()
}

func queue(f frames.Frame) step  { return step{frame: f} }
func do(run func()) step         { return step{run: run} }
func sleep(d time.Duration) step { return step{sleep: d} }
func speak(text string) step     { return queue(frames.NewTTSSpeakFrame(text)) }
func llmText(text string) step   { return queue(frames.NewLLMTextFrame(text)) }
func responseStart() step        { return queue(frames.NewLLMFullResponseStartFrame()) }
func responseEnd() step          { return queue(frames.NewLLMFullResponseEndFrame()) }

// runTest runs the service in a pipeline through the steps, ends the pipeline,
// and returns the frames that left it downstream and upstream.
func runTest(t *testing.T, svc *tts.Base, steps ...step) (down, up []frames.Frame) {
	t.Helper()
	var mu sync.Mutex
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		down = append(down, f)
		mu.Unlock()
	})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		up = append(up, f)
		mu.Unlock()
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	for _, s := range steps {
		if s.frame != nil {
			task.QueueFrame(s.frame)
		}
		if s.run != nil {
			s.run()
		}
		if s.sleep > 0 {
			time.Sleep(s.sleep)
		}
	}
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(15 * time.Second):
		t.Fatal("the pipeline did not finish")
	}
	mu.Lock()
	defer mu.Unlock()
	return slices.Clone(down), slices.Clone(up)
}

func count[T frames.Frame](fs []frames.Frame) int {
	n := 0
	for _, f := range fs {
		if _, ok := f.(T); ok {
			n++
		}
	}
	return n
}

func audioFrames(fs []frames.Frame) []*frames.TTSAudioRawFrame {
	var out []*frames.TTSAudioRawFrame
	for _, f := range fs {
		if a, ok := f.(*frames.TTSAudioRawFrame); ok {
			out = append(out, a)
		}
	}
	return out
}

func wordFrames(fs []frames.Frame) []*frames.TTSTextFrame {
	var out []*frames.TTSTextFrame
	for _, f := range fs {
		if w, ok := f.(*frames.TTSTextFrame); ok {
			out = append(out, w)
		}
	}
	return out
}

func wordTexts(fs []frames.Frame) []string {
	out := []string{}
	for _, w := range wordFrames(fs) {
		out = append(out, w.Text)
	}
	return out
}

func errorFrames(fs []frames.Frame) []*frames.ErrorFrame {
	var out []*frames.ErrorFrame
	for _, f := range fs {
		if e, ok := f.(*frames.ErrorFrame); ok {
			out = append(out, e)
		}
	}
	return out
}

// captureWarnings records what is logged at warning level and above while the
// test runs.
func captureWarnings(t *testing.T) func() string {
	t.Helper()
	var mu sync.Mutex
	var buf bytes.Buffer
	prev := slog.Default()
	handler := slog.NewTextHandler(lockedWriter{mu: &mu, w: &buf}, &slog.HandlerOptions{Level: slog.LevelWarn})
	slog.SetDefault(slog.New(handler))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return func() string {
		mu.Lock()
		defer mu.Unlock()
		return buf.String()
	}
}

type lockedWriter struct {
	mu *sync.Mutex
	w  *bytes.Buffer
}

func (l lockedWriter) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.w.Write(p)
}

// Regional variants resolve to HeyGen's base ISO 639-1 codes.
func TestLanguageMapping(t *testing.T) {
	for l, want := range map[language.Language]string{
		language.English:      "en",
		language.EnglishGB:    "en",
		language.ChineseHK:    "zh",
		language.PortugueseBR: "pt",
	} {
		if got := heygenLanguage(l); got != want {
			t.Errorf("heygenLanguage(%s) = %q, want %q", l, got, want)
		}
	}
}

// A language HeyGen doesn't support logs that it isn't applied.
func TestUnsupportedLanguageIsLogged(t *testing.T) {
	logs := captureWarnings(t)
	heygenLanguage(language.English)
	if strings.Contains(logs(), "not applied") {
		t.Fatalf("a supported language was logged: %s", logs())
	}
	heygenLanguage(language.Norwegian)
	if n := strings.Count(logs(), "[no] not applied"); n != 1 {
		t.Fatalf("logged %d times that Norwegian is not applied, want once: %s", n, logs())
	}
}

// HeyGen's error envelope is unwrapped to its code, message and request ID; any
// other body is reported as it is.
func TestErrorDetail(t *testing.T) {
	for body, want := range map[string]string{
		envelope:                               "invalid_parameter: Voice not found (request_id: req-7)",
		`{"error":{"code":"x","message":"y"}}`: "x: y (request_id: none)",
		"<html>bad gateway</html>":             "<html>bad gateway</html>",
		`{"error":"just a string"}`:            `{"error":"just a string"}`,
	} {
		if got := errorDetail([]byte(body)); got != want {
			t.Errorf("errorDetail(%s) = %q, want %q", body, got, want)
		}
	}
	long := strings.Repeat("x", 600)
	if got := errorDetail([]byte(long)); len(got) != 500 {
		t.Errorf("a long body is reported as %d bytes, want the first 500", len(got))
	}
}
