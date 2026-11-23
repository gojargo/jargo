package zoomai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
	errs "github.com/gojargo/jargo/utils/errors"
	"github.com/gojargo/jargo/utils/events"
)

// Tests for the Zoom TTS service, against a local HTTP server.

const ttsPath = "/v2/aiservices/tts/synthesize"

// pcm is the audio the stand-in answers with.
func pcm() []byte { return bytes.Repeat([]byte{1, 2, 3, 4}, 2048) }

// TestHTTPTTSConfigValidate pins which fields the service requires.
func TestHTTPTTSConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: HTTPTTSConfig{}, Valid: false},
		{Name: "API key only", Cfg: HTTPTTSConfig{APIKey: "k"}, Valid: true},
	})
}

// TestNewHTTPTTS checks the constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNewHTTPTTS(t *testing.T) {
	providertest.Service(t, "ZoomAIHTTPTTS", NewHTTPTTS(HTTPTTSConfig{APIKey: "k"}))
}

// request is one request the stand-in was sent.
type request struct {
	header http.Header
	raw    []byte
}

// fakeZoomTTS is a local stand-in for the Zoom TTS endpoint that records each
// request.
type fakeZoomTTS struct {
	mu       sync.Mutex
	requests []request
	url      string
}

// streamPCM answers with the audio, split mid-sample, as network reads can.
func streamPCM(w http.ResponseWriter, _ *http.Request) {
	audio := pcm()
	w.Header().Set("Content-Type", "audio/pcm;rate=44100")
	_, _ = w.Write(audio[:4095])
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
	time.Sleep(10 * time.Millisecond)
	_, _ = w.Write(audio[4095:])
}

// reject answers with a Zoom error body.
func reject(status int, reason, message string, header map[string]string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		for k, v := range header {
			w.Header().Set(k, v)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(map[string]any{"code": status, "reason": reason, "message": message})
	}
}

func newFakeZoomTTS(t *testing.T, reply http.HandlerFunc) *fakeZoomTTS {
	t.Helper()
	if reply == nil {
		reply = streamPCM
	}
	f := &fakeZoomTTS{}
	mux := http.NewServeMux()
	mux.HandleFunc("POST "+ttsPath, func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		f.mu.Lock()
		f.requests = append(f.requests, request{header: r.Header.Clone(), raw: raw})
		f.mu.Unlock()
		reply(w, r)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	f.url = srv.URL + ttsPath
	return f
}

func (f *fakeZoomTTS) all() []request {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.requests)
}

// body is the JSON body of the nth request.
func (f *fakeZoomTTS) body(t *testing.T, n int) map[string]any {
	t.Helper()
	requests := f.all()
	if n >= len(requests) {
		t.Fatalf("%d requests were sent, want at least %d", len(requests), n+1)
	}
	var body map[string]any
	if err := json.Unmarshal(requests[n].raw, &body); err != nil {
		t.Fatalf("request body is not JSON: %v", err)
	}
	return body
}

// service builds the service against the stand-in, at Zoom's own rate.
func service(f *fakeZoomTTS, cfg HTTPTTSConfig) *tts.Base {
	cfg.APIKey = "test-key"
	cfg.URL = f.url
	cfg.SampleRate = 44100
	return NewHTTPTTS(cfg)
}

// step is one thing a test does to the running pipeline: queue a frame or
// wait.
type step struct {
	frame frames.Frame
	sleep time.Duration
}

func speak(text string) step     { return step{frame: frames.NewTTSSpeakFrame(text)} }
func sleep(d time.Duration) step { return step{sleep: d} }

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

func audioFrames(fs []frames.Frame) []*frames.TTSAudioRawFrame {
	var out []*frames.TTSAudioRawFrame
	for _, f := range fs {
		if a, ok := f.(*frames.TTSAudioRawFrame); ok {
			out = append(out, a)
		}
	}
	return out
}

// zoomErrors are the errors the service itself reported. The base service
// also reports a sentence that produced no audio.
func zoomErrors(fs []frames.Frame) []*frames.ErrorFrame {
	var out []*frames.ErrorFrame
	for _, f := range fs {
		if e, ok := f.(*frames.ErrorFrame); ok && strings.Contains(e.Error, "Zoom TTS") {
			out = append(out, e)
		}
	}
	return out
}

func TestStreamsSampleAlignedAudioAt44100(t *testing.T) {
	fake := newFakeZoomTTS(t, nil)
	down, up := runTest(t, service(fake, HTTPTTSConfig{}), speak("Hello from Zoom."))

	audio := audioFrames(down)
	if len(audio) == 0 {
		t.Fatal("no audio")
	}
	var joined []byte
	for _, f := range audio {
		if len(f.Audio)%2 != 0 || f.SampleRate != 44100 {
			t.Fatalf("a frame of %d bytes at %d Hz, want whole samples at 44100", len(f.Audio), f.SampleRate)
		}
		joined = append(joined, f.Audio...)
	}
	if !bytes.Equal(joined, pcm()) {
		t.Errorf("emitted %d bytes, not the %d Zoom sent", len(joined), len(pcm()))
	}
	for _, f := range up {
		if e, ok := f.(*frames.ErrorFrame); ok {
			t.Errorf("an error was reported: %s", e.Error)
		}
	}
}

func TestRequestCarriesTextVoiceAndAuth(t *testing.T) {
	fake := newFakeZoomTTS(t, nil)
	runTest(t, service(fake, HTTPTTSConfig{}), speak("Hello from Zoom."))

	want := map[string]any{"input": "Hello from Zoom.", "response_format": "pcm_44100", "voice": "jenna"}
	if got := fake.body(t, 0); !reflect.DeepEqual(got, want) {
		t.Errorf("body = %v, want %v", got, want)
	}
	header := fake.all()[0].header
	if got := header.Get("Authorization"); got != "Bearer test-key" {
		t.Errorf("Authorization = %q", got)
	}
	if got := header.Get("Content-Type"); got != "application/json" {
		t.Errorf("Content-Type = %q", got)
	}
	if got := header.Get("x-zm-trackingid"); !strings.HasPrefix(got, "zoom_ai_community_jargo_") {
		t.Errorf("x-zm-trackingid = %q", got)
	}
}

func TestSettingsMapToRequestFields(t *testing.T) {
	fake := newFakeZoomTTS(t, nil)
	speed, volume := 1.2, 0.8
	runTest(t, service(fake, HTTPTTSConfig{Voice: "ava", Speed: &speed, Volume: &volume}), speak("Hello from Zoom."))

	body := fake.body(t, 0)
	if body["voice"] != "ava" || body["speed"] != 1.2 || body["volume"] != 0.8 {
		t.Errorf("body = %v, want voice ava, speed 1.2 and volume 0.8", body)
	}
}

// Unescaped UTF-8 keeps the longest input within the API's body limit.
func TestNonASCIITextIsSentAsUTF8(t *testing.T) {
	fake := newFakeZoomTTS(t, nil)
	runTest(t, service(fake, HTTPTTSConfig{}), speak("Café crème."))

	raw := fake.all()[0].raw
	if !bytes.Contains(raw, []byte("Café crème.")) {
		t.Errorf("body = %s, want the text as UTF-8", raw)
	}
	if bytes.Contains(raw, []byte(`\u`)) {
		t.Errorf("body = %s, want no escapes", raw)
	}
}

func TestLongTextIsSplitAcrossRequests(t *testing.T) {
	fake := newFakeZoomTTS(t, nil)
	text := strings.TrimSpace(strings.Repeat("word ", maxInputChars/4))
	runTest(t, service(fake, HTTPTTSConfig{}), speak(text))

	var inputs []string
	for n := range fake.all() {
		input, _ := fake.body(t, n)["input"].(string)
		inputs = append(inputs, input)
	}
	if len(inputs) != 2 {
		t.Fatalf("sent %d requests, want 2", len(inputs))
	}
	for _, input := range inputs {
		if len([]rune(input)) > maxInputChars {
			t.Errorf("an input of %d code points, want at most %d", len([]rune(input)), maxInputChars)
		}
	}
	if strings.Join(inputs, " ") != text {
		t.Error("the inputs do not join back into the text")
	}
}

func TestSplitTextCutsAtWhitespace(t *testing.T) {
	for _, tc := range []struct {
		text  string
		limit int
		want  []string
	}{
		{"aaa bbb ccc", 7, []string{"aaa bbb", "ccc"}},
		{"abcdefghij", 4, []string{"abcd", "efgh", "ij"}},
		{"short", maxInputChars, []string{"short"}},
	} {
		if got := splitText(tc.text, tc.limit); !slices.Equal(got, tc.want) {
			t.Errorf("splitText(%q, %d) = %q, want %q", tc.text, tc.limit, got, tc.want)
		}
	}
}

func TestUnsupportedVoiceReportsZoomReasonAndMakesTheServiceUnusable(t *testing.T) {
	fake := newFakeZoomTTS(t, reject(http.StatusBadRequest, "UNSUPPORTED_VOICE", "Unsupported voice", nil))
	svc := service(fake, HTTPTTSConfig{Voice: "nobody"})
	down, up := runTest(t, svc, speak("Hello from Zoom."))

	errors := zoomErrors(up)
	if len(errors) != 1 {
		t.Fatalf("%d Zoom errors reported, want 1: %v", len(errors), errors)
	}
	for _, want := range []string{
		"HTTP 400 UNSUPPORTED_VOICE Unsupported voice", "x-zm-trackingid zoom_ai_community_jargo_",
	} {
		if !strings.Contains(errors[0].Error, want) {
			t.Errorf("error = %q, want it to contain %q", errors[0].Error, want)
		}
	}
	if errors[0].Category != errs.InvalidRequest {
		t.Errorf("category = %q, want %q", errors[0].Category, errs.InvalidRequest)
	}
	if len(audioFrames(down)) != 0 {
		t.Error("audio was emitted for a refused request")
	}
	if svc.Usable() {
		t.Error("the service is still usable after its voice was refused")
	}
}

func TestUnavailableServiceLeavesTheServiceUsable(t *testing.T) {
	fake := newFakeZoomTTS(t, reject(http.StatusServiceUnavailable, "SERVICE_UNAVAILABLE", "Try again later",
		map[string]string{"Retry-After": "10"}))
	svc := service(fake, HTTPTTSConfig{})
	_, up := runTest(t, svc, speak("Hello from Zoom."))

	errors := zoomErrors(up)
	if len(errors) != 1 {
		t.Fatalf("%d Zoom errors reported, want 1: %v", len(errors), errors)
	}
	if !strings.Contains(errors[0].Error, "HTTP 503 SERVICE_UNAVAILABLE") {
		t.Errorf("error = %q", errors[0].Error)
	}
	if errors[0].Category != errs.Server {
		t.Errorf("category = %q, want %q", errors[0].Category, errs.Server)
	}
	if !svc.Usable() {
		t.Error("the service is unusable after a failure it can recover from")
	}
}

// A synthesis failure after the 200 ends the chunked response without its last
// chunk.
func TestStreamThatEndsEarlyIsReported(t *testing.T) {
	cutOff := func(w http.ResponseWriter, _ *http.Request) {
		hijacker, ok := w.(http.Hijacker)
		if !ok {
			t.Error("the stand-in cannot take over its connection")
			return
		}
		conn, rw, err := hijacker.Hijack()
		if err != nil {
			t.Errorf("hijack: %v", err)
			return
		}
		defer func() { _ = conn.Close() }()
		out := bufio.NewWriter(rw)
		_, _ = out.WriteString("HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\nTransfer-Encoding: chunked\r\n\r\n")
		_, _ = fmt.Fprintf(out, "%x\r\n", 4096)
		_, _ = out.Write(pcm()[:4096])
		_, _ = out.WriteString("\r\n")
		_ = out.Flush()
		time.Sleep(10 * time.Millisecond)
	}
	fake := newFakeZoomTTS(t, cutOff)
	down, up := runTest(t, service(fake, HTTPTTSConfig{}), speak("Hello from Zoom."))

	if len(audioFrames(down)) == 0 {
		t.Error("the audio that arrived before the stream ended was not emitted")
	}
	errors := zoomErrors(up)
	if len(errors) != 1 {
		t.Fatalf("%d Zoom errors reported, want 1: %v", len(errors), errors)
	}
	if !strings.Contains(errors[0].Error, "stream ended early") {
		t.Errorf("error = %q, want it to say the stream ended early", errors[0].Error)
	}
	if errors[0].Category != errs.Server {
		t.Errorf("category = %q, want %q", errors[0].Category, errs.Server)
	}
}

// Closing the request is how Zoom TTS is told to stop synthesizing.
func TestInterruptionClosesTheRequest(t *testing.T) {
	closed := make(chan struct{})
	var once sync.Once
	slow := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "audio/pcm")
		flusher, _ := w.(http.Flusher)
		for range 100 {
			if _, err := w.Write(pcm()[:4096]); err != nil {
				once.Do(func() { close(closed) })
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
			select {
			case <-r.Context().Done():
				// The client went away.
				once.Do(func() { close(closed) })
				return
			case <-time.After(50 * time.Millisecond):
			}
		}
	}
	fake := newFakeZoomTTS(t, slow)
	runTest(t, service(fake, HTTPTTSConfig{}),
		speak("A long reply the user talks over."),
		sleep(300*time.Millisecond),
		step{frame: frames.NewInterruptionFrame()},
		sleep(300*time.Millisecond),
	)

	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("the request was still open after the interruption")
	}
}

// A plaintext endpoint on another host is warned about; one on this machine is
// not.
func TestCheckURLWarnsAboutAPlaintextRemoteEndpoint(t *testing.T) {
	for endpoint, wantWarning := range map[string]bool{
		"https://api.zoom.us/v2/aiservices/tts/synthesize": false,
		"http://localhost:8080/tts":                        false,
		"http://127.0.0.1:8080/tts":                        false,
		"http://tts.example.com/tts":                       true,
	} {
		logs := captureWarnings(t)
		if got := checkURL(endpoint, "Zoom TTS", "text"); got != endpoint {
			t.Errorf("checkURL(%q) = %q, want the URL back", endpoint, got)
		}
		if warned := strings.Contains(logs(), "not encrypted"); warned != wantWarning {
			t.Errorf("checkURL(%q) warned = %t, want %t", endpoint, warned, wantWarning)
		}
	}
}
