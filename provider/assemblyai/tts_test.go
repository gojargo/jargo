package assemblyai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"maps"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
	uctx "github.com/gojargo/jargo/utils/context"
	"github.com/gojargo/jargo/utils/events"
)

// Tests for the TTS service against a fake Streaming TTS server.

const (
	// chunkMs is the length of one Audio frame: 80 ms of 24 kHz 16-bit mono
	// audio. Each request is answered with two.
	chunkMs = 80
	// wordMs is how long the fake server times each word.
	wordMs = 50
)

// audioChunk is the PCM of one Audio frame.
func audioChunk() []byte { return []byte(strings.Repeat("\x01\x02", 1920)) }

// ttsServerOptions shape the fake server.
type ttsServerOptions struct {
	// noWordBoundaries answers each Flush without a WordBoundaries frame.
	noWordBoundaries bool
	// stallFirstFlush makes the first request send one Audio frame and then
	// wait, so it is still in flight when the client cancels it.
	stallFirstFlush bool
	// errorAfterBegin ends every session (or only the first, with errorOnce)
	// with that Error.
	errorAfterBegin map[string]any
	errorOnce       bool
	// errorBeforeBegin refuses every session with it in place of Begin.
	errorBeforeBegin map[string]any
	// configuration adds keys to Begin's configuration.
	configuration map[string]any
	// expiresIn sets the first session's expires_at; zero, and every later
	// session, lasts an hour.
	expiresIn time.Duration
}

// captured is what the fake server saw.
type captured struct {
	mu           sync.Mutex
	connections  int
	connectTimes []time.Time
	query        url.Values
	auth         string
	messages     []map[string]any
}

func (c *captured) connectionCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.connections
}

// sent is every message of the type the client sent.
func (c *captured) sent(msgType string) []map[string]any {
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []map[string]any
	for _, m := range c.messages {
		if m["type"] == msgType {
			out = append(out, m)
		}
	}
	return out
}

// types is the type of every message the client sent, in order.
func (c *captured) types() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]string, 0, len(c.messages))
	for _, m := range c.messages {
		s, _ := m["type"].(string)
		out = append(out, s)
	}
	return out
}

// generated is the text of every Generate the client sent.
func (c *captured) generated() []string {
	var out []string
	for _, m := range c.sent(ttsMsgGenerate) {
		s, _ := m["text"].(string)
		out = append(out, s)
	}
	return out
}

// fakeTTSServer starts a fake server following the documented Streaming TTS
// protocol. Each Flush is answered with two Audio frames, a FlushDone and
// (unless noWordBoundaries is set) a WordBoundaries frame placing the request's
// words on the session timeline.
func fakeTTSServer(t *testing.T, opts ttsServerOptions) (endpoint string, got *captured) {
	t.Helper()
	got = &captured{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got.mu.Lock()
		got.connections++
		connection := got.connections
		got.connectTimes = append(got.connectTimes, time.Now())
		got.query = r.URL.Query()
		got.auth = r.Header.Get("Authorization")
		got.mu.Unlock()

		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		ctx := r.Context()
		send := func(m map[string]any) bool {
			b, err := json.Marshal(m)
			if err != nil {
				t.Errorf("encoding a server message: %v", err)
				return false
			}
			return c.Write(ctx, websocket.MessageText, b) == nil
		}
		refuse := func(e map[string]any) {
			msg := map[string]any{"type": ttsMsgError}
			maps.Copy(msg, e)
			send(msg)
			code, _ := e["error_code"].(int)
			_ = c.Close(websocket.StatusCode(code), "See Error message for details")
		}

		if opts.errorBeforeBegin != nil {
			refuse(opts.errorBeforeBegin)
			return
		}
		expiresIn := time.Hour
		if connection == 1 && opts.expiresIn > 0 {
			expiresIn = opts.expiresIn
		}
		configuration := map[string]any{"voice": "jane", "word_boundaries": !opts.noWordBoundaries}
		maps.Copy(configuration, opts.configuration)
		send(map[string]any{
			"type":          ttsMsgBegin,
			"id":            "session-" + strconv.Itoa(connection),
			"expires_at":    float64(time.Now().Add(expiresIn).UnixNano()) / float64(time.Second),
			"configuration": configuration,
		})
		if opts.errorAfterBegin != nil && (!opts.errorOnce || connection == 1) {
			refuse(opts.errorAfterBegin)
			return
		}

		flushID, sessionMs := 0, 0
		var buffer strings.Builder
		stalled := opts.stallFirstFlush
		audio := base64.StdEncoding.EncodeToString(audioChunk())
		for {
			_, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			msg := map[string]any{}
			if err := json.Unmarshal(data, &msg); err != nil {
				t.Errorf("decoding a client message: %v", err)
				return
			}
			got.mu.Lock()
			got.messages = append(got.messages, msg)
			got.mu.Unlock()

			switch msg["type"] {
			case ttsMsgGenerate:
				text, _ := msg["text"].(string)
				buffer.WriteString(text)
			case ttsMsgFlush:
				frame := map[string]any{"type": ttsMsgAudio, "audio": audio, "flush_id": flushID}
				if stalled {
					stalled = false
					send(frame)
					// Canceled audio still holds its place on the timeline.
					sessionMs += chunkMs
					flushID++
					buffer.Reset()
					continue
				}
				send(frame)
				send(frame)
				duration := 2 * chunkMs
				send(map[string]any{"type": ttsMsgFlushDone, "flush_id": flushID, "audio_duration_ms": duration})
				if !opts.noWordBoundaries {
					words := []map[string]any{}
					for i, word := range strings.Fields(buffer.String()) {
						words = append(words, map[string]any{
							"word":       word,
							"start":      sessionMs + i*wordMs,
							"end":        sessionMs + (i+1)*wordMs,
							"confidence": 0.9,
						})
					}
					send(map[string]any{
						"type":           ttsMsgWordBoundaries,
						"flush_id":       flushID,
						"audio_start_ms": sessionMs,
						"words":          words,
					})
				}
				sessionMs += duration
				flushID++
				buffer.Reset()
			case ttsMsgCancel:
				//nolint:misspell // the protocol's own spelling
				send(map[string]any{"type": "Cancelled", "audio_duration_seconds": 0.08})
				flushID++
				buffer.Reset()
			}
		}
	}))
	t.Cleanup(srv.Close)
	return "ws" + strings.TrimPrefix(srv.URL, "http") + "/v1/ws", got
}

// testTTS builds the service against the fake server.
func testTTS(endpoint string, cfg TTSConfig) (*tts.Base, *synthesizer) {
	cfg.APIKey = "test-key"
	cfg.URL = endpoint
	cfg.SampleRate = 24000
	return newTTS(cfg)
}

// step is one thing a test does to the running pipeline: queue a frame, or
// wait.
type step struct {
	frame frames.Frame
	sleep time.Duration
}

func queue(f frames.Frame) step       { return step{frame: f} }
func wait(d time.Duration) step       { return step{sleep: d} }
func waitMs(ms int) step              { return wait(time.Duration(ms) * time.Millisecond) }
func llmText(text string) step        { return queue(frames.NewLLMTextFrame(text)) }
func speakStep(text string) step      { return queue(frames.NewTTSSpeakFrame(text)) }
func responseStart() step             { return queue(frames.NewLLMFullResponseStartFrame()) }
func responseEnd() step               { return queue(frames.NewLLMFullResponseEndFrame()) }
func botStopped() step                { return queue(frames.NewBotStoppedSpeakingFrame()) }
func interruption() step              { return queue(frames.NewInterruptionFrame()) }
func llmTexts(texts ...string) []step { return stepsOf(texts, llmText) }

func stepsOf(texts []string, build func(string) step) []step {
	out := make([]step, 0, len(texts))
	for _, text := range texts {
		out = append(out, build(text))
	}
	return out
}

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
	case <-time.After(10 * time.Second):
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
	var out []string
	for _, w := range wordFrames(fs) {
		out = append(out, w.Text)
	}
	return out
}

func errorTexts(fs []frames.Frame) []string {
	var out []string
	for _, f := range fs {
		if e, ok := f.(*frames.ErrorFrame); ok {
			out = append(out, e.Error)
		}
	}
	return out
}

func anyContains(texts []string, sub string) bool {
	return slices.ContainsFunc(texts, func(s string) bool { return strings.Contains(s, sub) })
}

func pts(t *testing.T, f frames.Frame) int64 {
	t.Helper()
	p, ok := f.Base().PTS()
	if !ok {
		t.Fatalf("%T has no presentation time", f)
	}
	return p
}

// A TTSSpeakFrame is sent as one flushed request and its audio and words come
// back.
func TestTTSSpeakStreamsAudioAndWords(t *testing.T) {
	endpoint, got := fakeTTSServer(t, ttsServerOptions{})
	svc, _ := testTTS(endpoint, TTSConfig{})
	down, up := runTest(t, svc, speakStep("Hello from AssemblyAI."), waitMs(300), botStopped())

	if errs := errorTexts(append(down, up...)); len(errs) > 0 {
		t.Errorf("errors = %q, want none", errs)
	}
	if n := count[*frames.TTSStartedFrame](down); n != 1 {
		t.Errorf("TTSStartedFrames = %d, want 1", n)
	}
	if n := count[*frames.TTSStoppedFrame](down); n != 1 {
		t.Errorf("TTSStoppedFrames = %d, want 1", n)
	}
	var audio []byte
	for _, f := range down {
		if a, ok := f.(*frames.TTSAudioRawFrame); ok {
			audio = append(audio, a.Audio...)
		}
	}
	if want := slices.Concat(audioChunk(), audioChunk()); string(audio) != string(want) {
		t.Errorf("audio is %d bytes, want the %d the server sent", len(audio), len(want))
	}
	if words, want := wordTexts(down), []string{"Hello", "from", "AssemblyAI."}; !slices.Equal(words, want) {
		t.Errorf("words = %q, want %q", words, want)
	}

	// The key goes in the header raw, and the session is configured in the URL.
	got.mu.Lock()
	auth, query := got.auth, got.query
	got.mu.Unlock()
	if auth != "test-key" {
		t.Errorf("Authorization = %q, want the raw key", auth)
	}
	if query.Get("sample_rate") != "24000" || query.Get("encoding") != "pcm_s16le" {
		t.Errorf("query = %v, want sample_rate 24000 and encoding pcm_s16le", query)
	}
	if types := got.types(); !slices.Equal(types, []string{ttsMsgGenerate, ttsMsgFlush}) {
		t.Errorf("messages = %q, want one Generate and one Flush", types)
	}
	if n := got.connectionCount(); n != 1 {
		t.Errorf("connections = %d, want 1", n)
	}
}

// Each sentence is its own request, and the turn's words stay in order.
func TestTTSLLMTurnFlushesEachSentenceIntoOneContext(t *testing.T) {
	endpoint, got := fakeTTSServer(t, ttsServerOptions{})
	svc, _ := testTTS(endpoint, TTSConfig{})
	tokens := strings.Split("Hi there! Thanks for calling. How can I help?", " ")
	steps := []step{responseStart()}
	for _, token := range tokens {
		steps = append(steps, llmText(token+" "))
	}
	steps = append(steps, responseEnd(), waitMs(300), botStopped())
	down, _ := runTest(t, svc, steps...)

	want := []string{"Hi there!", "Thanks for calling.", "How can I help?"}
	if gen := got.generated(); !slices.Equal(gen, want) {
		t.Errorf("generated = %q, want %q", gen, want)
	}
	if n := len(got.sent(ttsMsgFlush)); n != 3 {
		t.Errorf("flushes = %d, want 3", n)
	}
	if n := count[*frames.TTSStartedFrame](down); n != 1 {
		t.Errorf("TTSStartedFrames = %d, want 1", n)
	}
	if n := count[*frames.TTSStoppedFrame](down); n != 1 {
		t.Errorf("TTSStoppedFrames = %d, want 1", n)
	}
	words := wordFrames(down)
	if texts := wordTexts(down); !slices.Equal(texts, tokens) {
		t.Fatalf("words = %q, want %q", texts, tokens)
	}
	// Word times are on the session's timeline, measured from the turn's first
	// audio.
	var times []int64
	for _, w := range words {
		times = append(times, pts(t, w))
	}
	if !slices.IsSorted(times) {
		t.Errorf("word times %v are out of order", times)
	}
	if d := time.Duration(times[2] - times[0]); d != 2*chunkMs*time.Millisecond {
		t.Errorf("the second sentence starts %v after the first, want %v", d, 2*chunkMs*time.Millisecond)
	}
}

// Without WordBoundaries the context closes after the grace period and keeps
// its text.
func TestTTSMissingWordBoundariesStillCloseTheContext(t *testing.T) {
	endpoint, _ := fakeTTSServer(t, ttsServerOptions{noWordBoundaries: true})
	svc, s := testTTS(endpoint, TTSConfig{})
	s.wordBoundariesGrace = 50 * time.Millisecond
	down, _ := runTest(t, svc, speakStep("Hello from AssemblyAI."), waitMs(300), botStopped())

	if n := count[*frames.TTSStoppedFrame](down); n != 1 {
		t.Errorf("TTSStoppedFrames = %d, want 1", n)
	}
	if text := strings.Join(wordTexts(down), " "); text != "Hello from AssemblyAI." {
		t.Errorf("text = %q, want the whole sentence", text)
	}
}

// An interruption sends Cancel, drops the canceled audio and reuses the
// session.
func TestTTSInterruptionCancelsWithoutReconnecting(t *testing.T) {
	endpoint, got := fakeTTSServer(t, ttsServerOptions{stallFirstFlush: true})
	svc, s := testTTS(endpoint, TTSConfig{})
	down, _ := runTest(t, svc,
		speakStep("This reply gets cut off."), waitMs(200),
		interruption(), waitMs(100),
		speakStep("Go ahead."), waitMs(300),
		botStopped())

	if n := len(got.sent(ttsMsgCancel)); n != 1 {
		t.Errorf("cancels = %d, want 1", n)
	}
	if n := got.connectionCount(); n != 1 {
		t.Errorf("connections = %d, want 1", n)
	}
	words := wordTexts(down)
	if len(words) < 2 || !slices.Equal(words[len(words)-2:], []string{"Go", "ahead."}) {
		t.Errorf("words = %q, want them to end with the reply after the interruption", words)
	}
	for _, f := range down {
		if started, ok := f.(*frames.TTSStartedFrame); ok && svc.AudioContextAvailable(started.ContextID) {
			t.Errorf("audio context %s is still open", started.ContextID)
		}
	}
	s.mu.Lock()
	left := len(s.contexts)
	s.mu.Unlock()
	if left != 0 {
		t.Errorf("%d contexts still have bookkeeping, want none", left)
	}
}

// Once every request has been answered there is nothing to cancel on the
// server.
func TestTTSInterruptionAfterSynthesisFinishedSendsNoCancel(t *testing.T) {
	endpoint, got := fakeTTSServer(t, ttsServerOptions{})
	svc, _ := testTTS(endpoint, TTSConfig{})
	runTest(t, svc, speakStep("Hello from AssemblyAI."), waitMs(300), interruption(), waitMs(100))

	if n := len(got.sent(ttsMsgCancel)); n != 0 {
		t.Errorf("cancels = %d, want none", n)
	}
}

// ttsError is an Error frame's fields.
func ttsError(code int, errorType, text string, fields map[string]any) map[string]any {
	e := map[string]any{"error_code": code, "error": text}
	if errorType != "" {
		e["error_type"] = errorType
		e["retryable"] = false
		e["param"] = nil
	}
	maps.Copy(e, fields)
	return e
}

// A voice that can't be found fails the same way on every retry.
func TestTTSMissingVoiceMakesTheServiceUnusable(t *testing.T) {
	refusal := ttsError(3012, "voice_not_found", "The requested voice was not found.",
		map[string]any{"param": "voice_id"})
	endpoint, got := fakeTTSServer(t, ttsServerOptions{errorBeforeBegin: refusal})
	svc, _ := testTTS(endpoint, TTSConfig{VoiceID: "nobody"})
	_, up := runTest(t, svc, waitMs(300))

	if svc.Usable() {
		t.Error("the service is still usable")
	}
	if errs := errorTexts(up); !anyContains(errs, "3012 voice_not_found") {
		t.Errorf("errors = %q, want the voice_not_found refusal", errs)
	}
	if n := got.connectionCount(); n != 1 {
		t.Errorf("connections = %d, want 1", n)
	}
}

// A text the server can't split fails that text only; the service reconnects.
func TestTTSUnspeakableTextIsNotPermanent(t *testing.T) {
	refusal := ttsError(3006, "invalid_request",
		"Text contains a 900-character segment too long to synthesize. Add spaces or punctuation.", nil)
	endpoint, got := fakeTTSServer(t, ttsServerOptions{errorAfterBegin: refusal, errorOnce: true})
	svc, _ := testTTS(endpoint, TTSConfig{})
	_, up := runTest(t, svc, waitMs(300))

	if !svc.Usable() {
		t.Error("the service is no longer usable")
	}
	if len(errorTexts(up)) == 0 {
		t.Error("no error was reported")
	}
	if n := got.connectionCount(); n != 2 {
		t.Errorf("connections = %d, want 2", n)
	}
}

// A credential refused before Begin fails the same way on every retry.
func TestTTSRejectedKeyMakesTheServiceUnusable(t *testing.T) {
	refusal := ttsError(1008, "unauthorized", "Unauthorized: Invalid API key", nil)
	endpoint, got := fakeTTSServer(t, ttsServerOptions{errorBeforeBegin: refusal})
	svc, _ := testTTS(endpoint, TTSConfig{})
	_, up := runTest(t, svc, waitMs(300))

	if svc.Usable() {
		t.Error("the service is still usable")
	}
	if errs := errorTexts(up); !anyContains(errs, "1008 unauthorized") {
		t.Errorf("errors = %q, want the unauthorized refusal", errs)
	}
	if n := got.connectionCount(); n != 1 {
		t.Errorf("connections = %d, want 1", n)
	}
}

// An error after Begin is reported but left to the reconnect to settle.
func TestTTSErrorWithoutErrorTypeAfterBeginIsNotPermanent(t *testing.T) {
	refusal := ttsError(3006, "", "Unsupported preset voice 'nobody'.", nil)
	endpoint, got := fakeTTSServer(t, ttsServerOptions{errorAfterBegin: refusal, errorOnce: true})
	svc, _ := testTTS(endpoint, TTSConfig{})
	_, up := runTest(t, svc, waitMs(300))

	if !svc.Usable() {
		t.Error("the service is no longer usable")
	}
	if errs := errorTexts(up); !anyContains(errs, "error 3006: Unsupported") {
		t.Errorf("errors = %q, want the error reported", errs)
	}
	if n := got.connectionCount(); n != 2 {
		t.Errorf("connections = %d, want 2", n)
	}
}

// An expired session reconnects without reporting an error.
func TestTTSRoutineSessionEndIsNotAnError(t *testing.T) {
	refusal := ttsError(3008, "session_expired",
		"Session reached its maximum duration; open a new connection to continue",
		map[string]any{"retryable": true})
	endpoint, got := fakeTTSServer(t, ttsServerOptions{errorAfterBegin: refusal, errorOnce: true})
	svc, _ := testTTS(endpoint, TTSConfig{})
	_, up := runTest(t, svc, waitMs(300))

	if errs := errorTexts(up); len(errs) > 0 {
		t.Errorf("errors = %q, want none", errs)
	}
	if n := got.connectionCount(); n != 2 {
		t.Errorf("connections = %d, want 2", n)
	}
}

// A retryable refusal is retried no sooner than the server asks.
func TestTTSReconnectWaitsForRetryAfterSeconds(t *testing.T) {
	refusal := ttsError(3005, "at_capacity", "The TTS service is at capacity; please retry in 1 seconds.",
		map[string]any{"retryable": true, "retry_after_seconds": 0.4})
	endpoint, got := fakeTTSServer(t, ttsServerOptions{errorAfterBegin: refusal, errorOnce: true})
	svc, _ := testTTS(endpoint, TTSConfig{})
	runTest(t, svc, waitMs(1500))

	if !svc.Usable() {
		t.Error("the service is no longer usable")
	}
	got.mu.Lock()
	times := slices.Clone(got.connectTimes)
	got.mu.Unlock()
	if len(times) != 2 {
		t.Fatalf("connections = %d, want 2", len(times))
	}
	if d := times[1].Sub(times[0]); d < 400*time.Millisecond {
		t.Errorf("reconnected after %v, want no sooner than 400ms", d)
	}
}

// Generate texts are split at the limit Begin reports and joined as sent.
func TestTTSBeginLimitsSizeGenerateMessages(t *testing.T) {
	limits := map[string]any{"limits": map[string]any{"max_generate_text_length": 8}}
	endpoint, got := fakeTTSServer(t, ttsServerOptions{configuration: limits})
	svc, _ := testTTS(endpoint, TTSConfig{})
	runTest(t, svc, waitMs(100), speakStep("Hello from AssemblyAI."), waitMs(300), botStopped())

	gen := got.generated()
	for _, text := range gen {
		if utf8.RuneCountInString(text) > 8 {
			t.Errorf("Generate text %q is over the limit", text)
		}
	}
	if joined := strings.Join(gen, ""); joined != "Hello from AssemblyAI." {
		t.Errorf("joined = %q, want the text as sent", joined)
	}
}

// An idle session sends KeepAlive, since the server closes sessions that go
// quiet.
func TestTTSIdleSessionIsKeptAlive(t *testing.T) {
	endpoint, got := fakeTTSServer(t, ttsServerOptions{})
	svc, s := testTTS(endpoint, TTSConfig{})
	s.keepaliveInterval = 100 * time.Millisecond
	runTest(t, svc, waitMs(350))

	if n := len(got.sent(ttsMsgKeepAlive)); n < 2 {
		t.Errorf("keepalives = %d, want at least 2", n)
	}
}

// A session near expires_at is replaced before the next request, keeping the
// turn.
func TestTTSExpiringSessionIsRenewedBetweenRequests(t *testing.T) {
	endpoint, got := fakeTTSServer(t, ttsServerOptions{expiresIn: time.Second})
	svc, _ := testTTS(endpoint, TTSConfig{})
	down, _ := runTest(t, svc,
		speakStep("First reply."), waitMs(800), botStopped(),
		speakStep("Second reply."), waitMs(300), botStopped())

	if n := got.connectionCount(); n != 2 {
		t.Errorf("connections = %d, want 2", n)
	}
	if words, want := wordTexts(down), []string{"First", "reply.", "Second", "reply."}; !slices.Equal(words, want) {
		t.Errorf("words = %q, want %q", words, want)
	}
}

// A turn that spans a renewal keeps its word times measured from its first
// audio.
func TestTTSSessionRenewedMidTurnKeepsWordTimes(t *testing.T) {
	endpoint, got := fakeTTSServer(t, ttsServerOptions{expiresIn: time.Second})
	svc, _ := testTTS(endpoint, TTSConfig{})
	tokens := []string{"Hi ", "there! ", "Thanks ", "for ", "calling. ", "Bye."}
	steps := []step{responseStart()}
	steps = append(steps, llmTexts(tokens[:3]...)...)
	// The first sentence is spoken and the session nears expires_at.
	steps = append(steps, waitMs(800))
	steps = append(steps, llmTexts(tokens[3:]...)...)
	steps = append(steps, responseEnd(), waitMs(300), botStopped())
	down, _ := runTest(t, svc, steps...)

	if n := got.connectionCount(); n != 2 {
		t.Errorf("connections = %d, want 2", n)
	}
	words := wordFrames(down)
	want := make([]string, 0, len(tokens))
	for _, token := range tokens {
		want = append(want, strings.TrimSpace(token))
	}
	if texts := wordTexts(down); !slices.Equal(texts, want) {
		t.Fatalf("words = %q, want %q", texts, want)
	}
	// Each sentence's words follow the audio of the sentences before it.
	start := pts(t, words[0])
	var offsets []int64
	for _, w := range words {
		offsets = append(offsets, (pts(t, w)-start)/int64(time.Millisecond))
	}
	wantOffsets := []int64{0, wordMs, 2 * chunkMs, 2*chunkMs + wordMs, 2*chunkMs + 2*wordMs, 4 * chunkMs}
	if !slices.Equal(offsets, wantOffsets) {
		t.Errorf("word offsets = %v ms, want %v", offsets, wantOffsets)
	}
}

// recordingHost records the word times the service reports.
type recordingHost struct {
	tts.AudioContextHost
	mu    sync.Mutex
	words []uctx.WordTiming
}

func (h *recordingHost) AddWordTimestamps(contextID string, words []uctx.WordTiming, opts tts.WordTimingOptions) {
	h.mu.Lock()
	h.words = append(h.words, words...)
	h.mu.Unlock()
	h.AudioContextHost.AddWordTimestamps(contextID, words, opts)
}

// Words after a context times out mid-turn are timed from the recreated
// context's audio.
func TestTTSContextRecreatedAfterQuietTimeoutRestartsWordTimes(t *testing.T) {
	endpoint, _ := fakeTTSServer(t, ttsServerOptions{})
	svc, s := testTTS(endpoint, TTSConfig{})
	host := &recordingHost{AudioContextHost: s.host}
	s.host = host
	// The turn goes quiet for longer than the base's three-second stop-frame
	// timeout, which is not configurable.
	down, _ := runTest(t, svc,
		responseStart(), llmText("Hi there! "), llmText("Thanks "),
		wait(3500*time.Millisecond),
		llmText("for calling. "), llmText("Bye."),
		responseEnd(), waitMs(300), botStopped())

	if n := count[*frames.TTSStartedFrame](down); n != 2 {
		t.Errorf("TTSStartedFrames = %d, want 2", n)
	}
	host.mu.Lock()
	recorded := slices.Clone(host.words)
	host.mu.Unlock()
	var words []string
	var offsets []int
	for _, w := range recorded {
		words = append(words, w.Word)
		offsets = append(offsets, int(w.Offset*1000+0.5))
	}
	if want := []string{"Hi", "there!", "Thanks", "for", "calling.", "Bye."}; !slices.Equal(words, want) {
		t.Errorf("words = %q, want %q", words, want)
	}
	// The recreated context's words start again from zero.
	if want := []int{0, wordMs, 0, wordMs, 2 * wordMs, 2 * chunkMs}; !slices.Equal(offsets, want) {
		t.Errorf("word offsets = %v ms, want %v", offsets, want)
	}
}

// urlQuery is the query the service would connect with.
func urlQuery(t *testing.T, s *synthesizer) url.Values {
	t.Helper()
	u, err := url.Parse(s.buildURL())
	if err != nil {
		t.Fatal(err)
	}
	return u.Query()
}

func TestTTSLanguageIsSentWhenSet(t *testing.T) {
	_, s := newTTS(TTSConfig{APIKey: "test-key", VoiceID: "aai-jane", Language: language.Spanish, SampleRate: 16000})
	q := urlQuery(t, s)
	if q.Get("voice_id") != "aai-jane" || q.Get("language") != "spanish" || q.Get("sample_rate") != "16000" {
		t.Errorf("query = %v, want voice aai-jane, language spanish and sample rate 16000", q)
	}
}

func TestTTSLanguageMapsToLanguageName(t *testing.T) {
	_, s := newTTS(TTSConfig{APIKey: "test-key", Language: language.Language("de-DE")})
	if got := s.live.Language.Or(""); got != "german" {
		t.Errorf("language = %q, want german", got)
	}
}

func TestTTSDefaultVoiceIsSentAndLanguageIsLeftToTheServer(t *testing.T) {
	_, s := newTTS(TTSConfig{APIKey: "test-key"})
	q := urlQuery(t, s)
	if got := q.Get("voice_id"); got != defaultTTSVoiceID {
		t.Errorf("voice_id = %q, want the default voice", got)
	}
	if q.Has("language") {
		t.Errorf("language = %q, want it left to the server", q.Get("language"))
	}
}

// A voice given no value leaves the voice to the server.
func TestTTSClearedVoiceLeavesTheVoiceToTheServer(t *testing.T) {
	_, s := newTTS(TTSConfig{APIKey: "test-key"})
	delta := &TTSSettings{}
	delta.Voice = settings.Cleared[string]()
	if _, err := settings.Apply(s.live, delta); err != nil {
		t.Fatal(err)
	}
	if q := urlQuery(t, s); q.Has("voice_id") {
		t.Errorf("voice_id = %q, want it left to the server", q.Get("voice_id"))
	}
}

func TestTTSClonedVoiceAndModelAreSent(t *testing.T) {
	_, s := newTTS(TTSConfig{
		APIKey:  "test-key",
		VoiceID: "01a142a4-66f3-7315-b15f-668b6edfb595",
		Model:   "assemblyai-tts-1",
	})
	q := urlQuery(t, s)
	if q.Get("voice_id") != "01a142a4-66f3-7315-b15f-668b6edfb595" || q.Get("model_id") != "assemblyai-tts-1" {
		t.Errorf("query = %v, want the cloned voice and the model", q)
	}
	if q.Has("voice") || q.Has("language") {
		t.Errorf("query = %v, want neither voice nor language", q)
	}
}

// A settings update keyed voice_id, as other services take it, sets the voice.
func TestTTSVoiceIDKeySetsTheVoice(t *testing.T) {
	delta := &TTSSettings{}
	if err := settings.FromMap(delta, map[string]any{"voice_id": "aai-mary"}); err != nil {
		t.Fatal(err)
	}
	if got := delta.Voice.Or(""); got != "aai-mary" {
		t.Errorf("voice = %q, want aai-mary", got)
	}
}

func TestTTSConfigValidate(t *testing.T) {
	if err := (TTSConfig{}).Validate(); err == nil {
		t.Error("a config without an API key validated")
	}
	if err := (TTSConfig{APIKey: "k"}).Validate(); err != nil {
		t.Errorf("Validate() = %v, want nil", err)
	}
}
