package deepgram

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/utils/events"
	ttstext "github.com/gojargo/jargo/utils/text"
)

// The two chunks of audio the fake Flux endpoint answers a turn with, one each
// side of its Flushed acknowledgement.
//
//nolint:gochecknoglobals // test fixtures, read-only
var (
	fluxAudioChunk1 = bytes.Repeat([]byte{0x00, 0x01}, 512)
	fluxAudioChunk2 = bytes.Repeat([]byte{0x02, 0x03}, 512)
)

// fluxTTSServerOptions shape how the fake Flux endpoint answers.
type fluxTTSServerOptions struct {
	// warningFirst sends a Warning ahead of the first turn's SpeechStarted.
	warningFirst bool
	// holdTurn withholds SpeechMetadata, leaving the turn open the way it is
	// while Flux is still synthesizing.
	holdTurn bool
	// rejectConfigure answers a Configure with a ConfigureFailure.
	rejectConfigure bool
	// audioDelay is how long the endpoint takes to answer a Flush.
	audioDelay time.Duration
}

// fluxTTSServer is a fake Flux synthesis endpoint following the documented
// turn flow: it records what it was dialed with and every message it was sent.
type fluxTTSServer struct {
	opts fluxTTSServerOptions

	mu            sync.Mutex
	queries       []url.Values
	auth          string
	messages      []map[string]any
	dials         int
	speechStarted bool
}

// newFluxTTSServer starts a fake Flux endpoint and returns it with its URL.
func newFluxTTSServer(t *testing.T, opts fluxTTSServerOptions) (*fluxTTSServer, string) {
	t.Helper()
	f := &fluxTTSServer{opts: opts}
	srv := httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(srv.Close)
	return f, wsURL(srv.URL) + "/v2/speak"
}

func (f *fluxTTSServer) serve(w http.ResponseWriter, r *http.Request) {
	c, err := websocket.Accept(w, r, nil)
	if err != nil {
		return
	}
	defer func() { _ = c.CloseNow() }()
	f.mu.Lock()
	f.dials++
	f.queries = append(f.queries, r.URL.Query())
	f.auth = r.Header.Get("Authorization")
	f.mu.Unlock()
	ctx := r.Context()
	send := func(m map[string]any) bool { return c.Write(ctx, websocket.MessageText, mustJSON(m)) == nil }

	for {
		_, data, err := c.Read(ctx)
		if err != nil {
			return
		}
		msg := map[string]any{}
		if json.Unmarshal(data, &msg) != nil {
			return
		}
		f.mu.Lock()
		f.messages = append(f.messages, msg)
		first := msg["type"] == "Speak" && !f.speechStarted
		if first {
			f.speechStarted = true
		}
		f.mu.Unlock()

		switch {
		case first:
			send(map[string]any{"type": "Connected", "request_id": "test-request", "model_name": "flux-alexis-en"})
			if f.opts.warningFirst {
				send(map[string]any{
					"type": "Warning", "code": "TEXT_TRUNCATED", "description": "Input text was truncated.",
				})
			}
			send(map[string]any{"type": "SpeechStarted", "speech_id": "dg_sp_test"})
		case msg["type"] == "Configure" && f.opts.rejectConfigure:
			send(map[string]any{
				"type": "ConfigureFailure", "code": "SPEED_OUT_OF_RANGE", "field": "speed",
				"value": msg["speed"], "description": "Speed must be between 0.5 and 1.5.",
			})
		case msg["type"] == "Flush":
			time.Sleep(f.opts.audioDelay)
			// Flux acknowledges the flush ahead of the turn's remaining audio;
			// SpeechMetadata comes only after all of it.
			_ = c.Write(ctx, websocket.MessageBinary, fluxAudioChunk1)
			send(map[string]any{"type": "Flushed", "speech_id": "dg_sp_test"})
			_ = c.Write(ctx, websocket.MessageBinary, fluxAudioChunk2)
			if !f.opts.holdTurn {
				send(map[string]any{
					"type": "SpeechMetadata", "speech_id": "dg_sp_test", "audio_duration_ms": 100,
					"input_character_count": 17, "billable_character_count": 17,
				})
			}
		}
	}
}

// sent is every message the endpoint was sent, in order.
func (f *fluxTTSServer) sent() []map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.messages)
}

// sentTypes is the type of every message the endpoint was sent, in order.
func (f *fluxTTSServer) sentTypes() []string {
	var out []string
	for _, m := range f.sent() {
		typ, _ := m["type"].(string)
		out = append(out, typ)
	}
	return out
}

// dialCount is how many connections were opened.
func (f *fluxTTSServer) dialCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.dials
}

// lastQuery is the query the latest connection was opened with.
func (f *fluxTTSServer) lastQuery(t *testing.T) url.Values {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.queries) == 0 {
		t.Fatal("the endpoint was never dialed")
	}
	return f.queries[len(f.queries)-1]
}

// awaitSent waits for the endpoint to have been sent n messages.
func (f *fluxTTSServer) awaitSent(t *testing.T, n int) []map[string]any {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if got := f.sent(); len(got) >= n {
			return got
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("the endpoint was sent %v, want %d messages", f.sent(), n)
	return nil
}

// fluxTTSTestConfig is a config pointed at endpoint.
func fluxTTSTestConfig(endpoint string) FluxTTSConfig {
	return FluxTTSConfig{APIKey: "test-key", SpeakURL: endpoint, SampleRate: 24000}
}

// connectedFluxSynth builds the service against endpoint and connects it, the
// way the pipeline starting does.
func connectedFluxSynth(t *testing.T, cfg FluxTTSConfig) (*tts.Base, *fluxSynth) {
	t.Helper()
	b, s := newFluxTTS(cfg)
	s.Connect(t.Context())
	t.Cleanup(func() { s.Disconnect(context.Background()) })
	if !s.Connected() {
		t.Fatal("the synthesizer did not connect")
	}
	return b, s
}

// runFluxTTSPipeline runs the service in a pipeline, queueing each frame in
// order (a time.Duration waits that long instead), ends the pipeline, and
// returns the frames that left it downstream and upstream.
func runFluxTTSPipeline(t *testing.T, svc *tts.Base, steps ...any) (down, up []frames.Frame) {
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
		switch v := s.(type) {
		case time.Duration:
			time.Sleep(v)
		case frames.Frame:
			task.QueueFrame(v)
		default:
			t.Fatalf("cannot run step %T", s)
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

// framesOf is the frames of type T, in order.
func framesOf[T frames.Frame](fs []frames.Frame) []T {
	var out []T
	for _, f := range fs {
		if v, ok := f.(T); ok {
			out = append(out, v)
		}
	}
	return out
}

func TestFluxTTSValidate(t *testing.T) {
	if err := (FluxTTSConfig{APIKey: "k"}).Validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	if err := (FluxTTSConfig{}).Validate(); err == nil {
		t.Fatal("config without APIKey should be rejected")
	}
	if err := (FluxTTSConfig{APIKey: "k", SampleRate: 24000}).Validate(); err != nil {
		t.Fatalf("supported sample rate rejected: %v", err)
	}
	err := FluxTTSConfig{APIKey: "k", SampleRate: 22050}.Validate()
	if err == nil || !errors.Is(err, errFluxUnsupportedRate) {
		t.Fatalf("unsupported sample rate should be rejected, got %v", err)
	}
	if err := (FluxTTSConfig{APIKey: "k", Speed: new(1.05), Expressivity: new(-2)}).Validate(); err != nil {
		t.Fatalf("voice controls in range rejected: %v", err)
	}
	if err := (FluxTTSConfig{APIKey: "k", Speed: new(2.0)}).Validate(); err == nil {
		t.Fatal("a speed out of range should be rejected")
	}
	if err := (FluxTTSConfig{APIKey: "k", Expressivity: new(3)}).Validate(); err == nil {
		t.Fatal("an expressivity out of range should be rejected")
	}
}

func TestFluxTTSDefaults(t *testing.T) {
	cfg := FluxTTSConfig{APIKey: "k"}.withDefaults()
	if cfg.Voice != "flux-heather-en" {
		t.Fatalf("voice = %q, want flux-heather-en", cfg.Voice)
	}
	if cfg.SampleRate != defaultTTSSampleRate {
		t.Fatalf("sample_rate = %d, want %d", cfg.SampleRate, defaultTTSSampleRate)
	}
	if cfg.SpeakURL != fluxSpeakURL {
		t.Fatalf("speak URL = %q, want %q", cfg.SpeakURL, fluxSpeakURL)
	}

	// Explicit values are preserved.
	custom := FluxTTSConfig{APIKey: "k", Voice: "flux-other-en", SampleRate: 16000}.withDefaults()
	if custom.Voice != "flux-other-en" || custom.SampleRate != 16000 {
		t.Fatalf("explicit config overwritten: %+v", custom)
	}

	// The constructor names the service and reports the configured rate.
	svc, s := newFluxTTS(FluxTTSConfig{APIKey: "k"})
	if !strings.HasPrefix(svc.Name(), "DeepgramFluxTTS") {
		t.Fatalf("service name = %q", svc.Name())
	}
	if s.SampleRate() != defaultTTSSampleRate {
		t.Fatalf("SampleRate() mismatch")
	}
}

func TestFluxTTSQuery(t *testing.T) {
	mip := true
	cfg := FluxTTSConfig{
		Voice:      defaultFluxVoice,
		SampleRate: 24000,
		MipOptOut:  &mip,
		Tag:        []string{"prod", "agent"},
	}
	q := fluxTTSQuery(cfg, newFluxTTSSettings(cfg))

	if got := q.Get("model"); got != defaultFluxVoice {
		t.Fatalf("model = %q (voice is sent as model)", got)
	}
	if got := q.Get("encoding"); got != fluxEncoding {
		t.Fatalf("encoding = %q", got)
	}
	if got := q.Get("sample_rate"); got != "24000" {
		t.Fatalf("sample_rate = %q", got)
	}
	if got := q.Get("mip_opt_out"); got != "true" {
		t.Fatalf("mip_opt_out = %q", got)
	}
	if got := q["tag"]; len(got) != 2 || got[0] != "prod" || got[1] != "agent" {
		t.Fatalf("tag = %v", got)
	}
}

// Speak and Flush are sent, and the turn's audio is played, including what
// arrives after Flushed.
func TestFluxTTSProtocolRoundtrip(t *testing.T) {
	server, endpoint := newFluxTTSServer(t, fluxTTSServerOptions{})
	svc, _ := newFluxTTS(FluxTTSConfig{APIKey: "test-key", SpeakURL: endpoint, SampleRate: 24000})

	down, up := runFluxTTSPipeline(t, svc,
		frames.NewTTSSpeakFrame("Hello from Flux."),
		300*time.Millisecond,
		// The output transport sends this once playback ends.
		frames.NewBotStoppedSpeakingFrame(),
	)

	if len(framesOf[*frames.TTSStartedFrame](down)) == 0 || len(framesOf[*frames.TTSStoppedFrame](down)) == 0 {
		t.Errorf("the turn was not bracketed by started and stopped frames")
	}
	if errs := framesOf[*frames.ErrorFrame](slices.Concat(down, up)); len(errs) > 0 {
		t.Errorf("errors pushed: %v", errs)
	}
	audio := framesOf[*frames.TTSAudioRawFrame](down)
	var pcm []byte
	for _, f := range audio {
		if f.SampleRate != 24000 || f.NumChannels != 1 {
			t.Errorf("audio at %d Hz on %d channels, want 24000 Hz mono", f.SampleRate, f.NumChannels)
		}
		pcm = append(pcm, f.Audio...)
	}
	// Audio sent after Flushed but before SpeechMetadata must not be dropped.
	if !bytes.Equal(pcm, slices.Concat(fluxAudioChunk1, fluxAudioChunk2)) {
		t.Errorf("played %d bytes of audio, want both chunks", len(pcm))
	}

	server.mu.Lock()
	auth := server.auth
	server.mu.Unlock()
	if auth != "Token test-key" {
		t.Errorf("Authorization = %q", auth)
	}
	q := server.lastQuery(t)
	if q.Get("model") != "flux-heather-en" || q.Get("encoding") != "linear16" || q.Get("sample_rate") != "24000" {
		t.Errorf("query = %v", q)
	}
	if !slices.Contains(server.sentTypes(), "Flush") {
		t.Errorf("sent %v, want a Flush", server.sentTypes())
	}
	if speak := server.sent()[0]; speak["type"] != "Speak" || speak["text"] != "Hello from Flux." {
		t.Errorf("first message = %v, want the text spoken verbatim", speak)
	}
	if n := server.dialCount(); n != 1 {
		t.Errorf("dialed %d connections, want the one held for the call", n)
	}
}

// An uninterruptible utterance survives an interruption that comes before Flux
// returns its audio.
func TestFluxTTSUninterruptibleSpeechWaitsForAudioAfterInterruption(t *testing.T) {
	server, endpoint := newFluxTTSServer(t, fluxTTSServerOptions{audioDelay: 200 * time.Millisecond})
	svc, _ := newFluxTTS(FluxTTSConfig{APIKey: "test-key", SpeakURL: endpoint, SampleRate: 24000})
	speech := frames.NewTTSSpeakFrame("Mm-hmm.")
	speech.AppendToContext = false
	speech.SetInterruptible(false)

	down, up := runFluxTTSPipeline(t, svc,
		speech,
		50*time.Millisecond,
		frames.NewVADUserStartedSpeakingFrame(0, time.Now()),
		frames.NewInterruptionFrame(),
		300*time.Millisecond,
		frames.NewBotStoppedSpeakingFrame(),
	)

	if slices.Contains(server.sentTypes(), "Interrupt") {
		t.Errorf("sent %v, want no Interrupt for uninterruptible speech", server.sentTypes())
	}
	if errs := framesOf[*frames.ErrorFrame](slices.Concat(down, up)); len(errs) > 0 {
		t.Errorf("errors pushed: %v", errs)
	}
	audio := framesOf[*frames.TTSAudioRawFrame](down)
	var pcm []byte
	for _, f := range audio {
		pcm = append(pcm, f.Audio...)
		if frames.Interruptible(f) {
			t.Error("an audio frame of uninterruptible speech is interruptible")
		}
	}
	if !bytes.Equal(pcm, slices.Concat(fluxAudioChunk1, fluxAudioChunk2)) {
		t.Errorf("played %d bytes of audio, want both chunks", len(pcm))
	}
	var texts []string
	for _, f := range framesOf[*frames.TTSTextFrame](down) {
		texts = append(texts, f.Text)
		if frames.Interruptible(f) {
			t.Error("a text frame of uninterruptible speech is interruptible")
		}
	}
	if !slices.Equal(texts, []string{"Mm-hmm."}) {
		t.Errorf("spoken %q, want the utterance", texts)
	}
}

// Flux exists to start synthesizing on the first token rather than waiting for
// the sentence to finish, so the model's tokens reach it as written: one Speak
// each, with no space inserted between them and none taken away. Flux neither
// inserts nor strips whitespace between the units it is sent, so a space added
// here would split a word and a space removed would run two together.
func TestFluxTTSStreamsTokensAsWritten(t *testing.T) {
	server, endpoint := newFluxTTSServer(t, fluxTTSServerOptions{})
	svc, _ := newFluxTTS(fluxTTSTestConfig(endpoint))

	down, up := runFluxTTSPipeline(t, svc,
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMTextFrame("Unbelieva"),
		frames.NewLLMTextFrame("ble"),
		frames.NewLLMTextFrame(" isn't it?"),
		frames.NewLLMFullResponseEndFrame(),
		300*time.Millisecond,
		frames.NewBotStoppedSpeakingFrame(),
	)

	if errs := framesOf[*frames.ErrorFrame](slices.Concat(down, up)); len(errs) > 0 {
		t.Errorf("errors pushed: %v", errs)
	}
	if len(framesOf[*frames.TTSAudioRawFrame](down)) == 0 {
		t.Error("no audio was played")
	}
	var spoke []string
	flushes := 0
	for _, m := range server.sent() {
		switch m["type"] {
		case "Speak":
			text, _ := m["text"].(string)
			spoke = append(spoke, text)
		case "Flush":
			flushes++
		}
	}
	if want := []string{"Unbelieva", "ble", " isn't it?"}; !slices.Equal(spoke, want) {
		t.Errorf("spoke %q, want %q", spoke, want)
	}
	if flushes != 1 {
		t.Errorf("sent %d flushes, want one for the turn", flushes)
	}
}

// grouped into sentences, the text ends in a space, which keeps consecutive
// sentences apart.
func TestFluxTTSSentenceModeAppendsTrailingSpace(t *testing.T) {
	server, endpoint := newFluxTTSServer(t, fluxTTSServerOptions{})
	svc, _ := newFluxTTS(fluxTTSTestConfig(endpoint))
	svc.SetTextAggregator(ttstext.NewSimpleAggregator(frames.TextTypeSentence, "en"))

	down, up := runFluxTTSPipeline(t, svc,
		frames.NewTTSSpeakFrame("Hello from Flux."),
		300*time.Millisecond,
		frames.NewBotStoppedSpeakingFrame(),
	)

	if errs := framesOf[*frames.ErrorFrame](slices.Concat(down, up)); len(errs) > 0 {
		t.Errorf("errors pushed: %v", errs)
	}
	if speak := server.sent()[0]; speak["text"] != "Hello from Flux. " {
		t.Errorf("first message = %v, want the sentence with a trailing space", speak)
	}
}

// An interruption sends an Interrupt, with no other field, and keeps the
// connection.
func TestFluxTTSInterruptionSendsInterruptAndKeepsConnection(t *testing.T) {
	server, endpoint := newFluxTTSServer(t, fluxTTSServerOptions{})
	_, s := connectedFluxSynth(t, fluxTTSTestConfig(endpoint))

	s.OnAudioContextInterrupted(t.Context(), "test-context")

	sent := server.awaitSent(t, 1)
	if len(sent) != 1 || len(sent[0]) != 1 || sent[0]["type"] != "Interrupt" {
		t.Errorf("sent %v, want a bare Interrupt", sent)
	}
	if !s.Connected() || server.dialCount() != 1 {
		t.Errorf("connected %v after %d dials, want the connection kept", s.Connected(), server.dialCount())
	}
}

// an interruption sends an Interrupt and leaves no context for audio still in
// flight. Audio names no turn, so it is attributed to the context playing, and
// leaving none is what keeps audio generated before the Interrupt out of the
// next turn.
func TestFluxTTSInterruptionLeavesNoActiveAudioContext(t *testing.T) {
	server, endpoint := newFluxTTSServer(t, fluxTTSServerOptions{holdTurn: true})
	svc, _ := newFluxTTS(fluxTTSTestConfig(endpoint))

	runFluxTTSPipeline(t, svc,
		frames.NewTTSSpeakFrame("Hello from Flux."),
		200*time.Millisecond,
		frames.NewInterruptionFrame(),
		200*time.Millisecond,
	)

	if !slices.Contains(server.sentTypes(), "Interrupt") {
		t.Errorf("sent %v, want an Interrupt", server.sentTypes())
	}
	if id := svc.ActiveAudioContextID(); id != "" {
		t.Errorf("active audio context = %q, want none", id)
	}
	if ids := svc.AudioContextIDs(); len(ids) != 0 {
		t.Errorf("audio contexts %v left open", ids)
	}
}

// An interruption after the turn ended on the server sends no Interrupt. The
// context closes once its SpeechMetadata arrives, so there is no turn left to
// cancel.
func TestFluxTTSCompletedTurnIsNotInterrupted(t *testing.T) {
	server, endpoint := newFluxTTSServer(t, fluxTTSServerOptions{})
	svc, _ := newFluxTTS(fluxTTSTestConfig(endpoint))

	runFluxTTSPipeline(t, svc,
		frames.NewTTSSpeakFrame("Hello from Flux."),
		300*time.Millisecond,
		frames.NewInterruptionFrame(),
		200*time.Millisecond,
	)

	if slices.Contains(server.sentTypes(), "Interrupt") {
		t.Errorf("sent %v, want no Interrupt", server.sentTypes())
	}
}

// A Warning is logged, not pushed as an error.
func TestFluxTTSWarningIsNotFatal(t *testing.T) {
	_, endpoint := newFluxTTSServer(t, fluxTTSServerOptions{warningFirst: true})
	svc, _ := newFluxTTS(fluxTTSTestConfig(endpoint))

	down, up := runFluxTTSPipeline(t, svc,
		frames.NewTTSSpeakFrame("Hello from Flux."),
		300*time.Millisecond,
		frames.NewBotStoppedSpeakingFrame(),
	)

	if errs := framesOf[*frames.ErrorFrame](slices.Concat(down, up)); len(errs) > 0 {
		t.Errorf("errors pushed: %v", errs)
	}
	if len(framesOf[*frames.TTSAudioRawFrame](down)) == 0 {
		t.Error("no audio was played")
	}
}

// A rejected settings update reaches the application as a non-fatal error.
func TestFluxTTSConfigureFailurePushesError(t *testing.T) {
	server, endpoint := newFluxTTSServer(t, fluxTTSServerOptions{rejectConfigure: true})
	svc, _ := newFluxTTS(fluxTTSTestConfig(endpoint))

	down, up := runFluxTTSPipeline(t, svc,
		frames.NewTTSSpeakFrame("Hello from Flux."),
		300*time.Millisecond,
		frames.NewBotStoppedSpeakingFrame(),
		frames.NewTTSUpdateSettingsFrame(&FluxTTSSettings{Speed: settings.Set(1.1)}),
		300*time.Millisecond,
	)

	if !slices.Contains(server.sentTypes(), "Configure") {
		t.Errorf("sent %v, want a Configure", server.sentTypes())
	}
	errs := framesOf[*frames.ErrorFrame](slices.Concat(down, up))
	if len(errs) == 0 {
		t.Fatal("the ConfigureFailure was not pushed as an error")
	}
	if !strings.Contains(errs[0].Error, "SPEED_OUT_OF_RANGE") || errs[0].Fatal {
		t.Errorf("error = %q (fatal %v), want the rejection's code, not fatal", errs[0].Error, errs[0].Fatal)
	}
	if server.dialCount() != 1 {
		t.Errorf("dialed %d connections, want the speed applied to the open one", server.dialCount())
	}
}

// The connection's configuration and the settings reach the query.
func TestFluxTTSQueryParams(t *testing.T) {
	server, endpoint := newFluxTTSServer(t, fluxTTSServerOptions{})
	svc, s := newFluxTTS(FluxTTSConfig{
		APIKey:          "test-key",
		SpeakURL:        endpoint,
		SampleRate:      8000,
		MipOptOut:       new(true),
		Tag:             []string{"tag-a", "tag-b"},
		Voice:           "flux-thalia-en",
		Speed:           new(1.05),
		Expressivity:    new(-1),
		NormalizeVolume: new(false),
	})

	runFluxTTSPipeline(t, svc)

	q := server.lastQuery(t)
	want := url.Values{
		"model":            {"flux-thalia-en"},
		"encoding":         {"linear16"},
		"sample_rate":      {"8000"},
		"speed":            {"1.05"},
		"expressivity":     {"-1"},
		"normalize_volume": {"false"},
		"mip_opt_out":      {"true"},
		"tag":              {"tag-a", "tag-b"},
	}
	for k, v := range want {
		if !slices.Equal(q[k], v) {
			t.Errorf("%s = %v, want %v", k, q[k], v)
		}
	}
	if model, _ := s.live.Model.Value(); model != "flux-thalia-en" {
		t.Errorf("model setting = %q, want the voice", model)
	}
	if meta := s.Metadata(); meta.Model != "flux-thalia-en" {
		t.Errorf("Metadata().Model = %q, want the voice", meta.Model)
	}
}

// Speed, expressivity and volume normalization are left off the query when
// unset.
func TestFluxTTSUnsetVoiceControlsAreOmitted(t *testing.T) {
	server, endpoint := newFluxTTSServer(t, fluxTTSServerOptions{})
	svc, _ := newFluxTTS(fluxTTSTestConfig(endpoint))

	runFluxTTSPipeline(t, svc)

	q := server.lastQuery(t)
	for _, k := range []string{"speed", "expressivity", "normalize_volume"} {
		if q.Has(k) {
			t.Errorf("%s = %v sent, want it omitted", k, q[k])
		}
	}
}

// updateFluxSettings applies a settings delta the way the base does.
func updateFluxSettings(t *testing.T, s *fluxSynth, delta *FluxTTSSettings) {
	t.Helper()
	changed, err := settings.Apply(s.live, delta)
	if err != nil {
		t.Fatalf("applying the settings: %v", err)
	}
	if err := s.UpdateSettings(t.Context(), changed); err != nil {
		t.Fatalf("UpdateSettings: %v", err)
	}
}

// A speed change is applied on the open connection, without reconnecting.
func TestFluxTTSSpeedUpdateSendsConfigure(t *testing.T) {
	server, endpoint := newFluxTTSServer(t, fluxTTSServerOptions{})
	_, s := connectedFluxSynth(t, fluxTTSTestConfig(endpoint))

	updateFluxSettings(t, s, &FluxTTSSettings{Speed: settings.Set(1.1)})

	sent := server.awaitSent(t, 1)
	if len(sent) != 1 || sent[0]["type"] != "Configure" || sent[0]["speed"] != 1.1 {
		t.Errorf("sent %v, want one Configure with speed 1.1", sent)
	}
	if server.dialCount() != 1 {
		t.Errorf("dialed %d connections, want the open one kept", server.dialCount())
	}
	if speed, _ := s.live.Speed.Value(); speed != 1.1 {
		t.Errorf("speed setting = %v, want 1.1", speed)
	}
}

// clearing the speed restores Flux's default rate explicitly.
func TestFluxTTSClearedSpeedConfiguresDefaultRate(t *testing.T) {
	server, endpoint := newFluxTTSServer(t, fluxTTSServerOptions{})
	cfg := fluxTTSTestConfig(endpoint)
	cfg.Speed = new(1.15)
	_, s := connectedFluxSynth(t, cfg)

	updateFluxSettings(t, s, &FluxTTSSettings{Speed: settings.Cleared[float64]()})

	sent := server.awaitSent(t, 1)
	if len(sent) != 1 || sent[0]["type"] != "Configure" || sent[0]["speed"] != 1.0 {
		t.Errorf("sent %v, want one Configure with speed 1.0", sent)
	}
}

// expressivity is fixed for a connection, so a change reconnects with it.
func TestFluxTTSExpressivityUpdateReconnects(t *testing.T) {
	server, endpoint := newFluxTTSServer(t, fluxTTSServerOptions{})
	_, s := connectedFluxSynth(t, fluxTTSTestConfig(endpoint))

	updateFluxSettings(t, s, &FluxTTSSettings{Expressivity: settings.Set(2)})

	if server.dialCount() != 2 || !s.Connected() {
		t.Errorf("dialed %d connections (connected %v), want a second one", server.dialCount(), s.Connected())
	}
	if sent := server.sent(); len(sent) != 0 {
		t.Errorf("sent %v, want nothing on either connection", sent)
	}
	if v, _ := s.live.Expressivity.Value(); v != 2 {
		t.Errorf("expressivity setting = %v, want 2", v)
	}
	if got := server.lastQuery(t).Get("expressivity"); got != "2" {
		t.Errorf("the new connection's expressivity = %q, want 2", got)
	}
}

// A voice change reconnects with the new voice, and relabels the model the
// synthesis is reported against: Deepgram takes the voice as the model.
func TestFluxTTSVoiceUpdateFollowsTheModel(t *testing.T) {
	server, endpoint := newFluxTTSServer(t, fluxTTSServerOptions{})
	_, s := connectedFluxSynth(t, fluxTTSTestConfig(endpoint))

	changed, err := settings.Apply(s.live, &FluxTTSSettings{Voice: settings.Set("flux-thalia-en")})
	if err != nil {
		t.Fatalf("applying the settings: %v", err)
	}
	if changed["model"] != "flux-heather-en" || changed["voice"] != "flux-heather-en" {
		t.Errorf("changed = %v, want the voice and the model, from the default", changed)
	}
	if err := s.UpdateSettings(t.Context(), changed); err != nil {
		t.Fatalf("UpdateSettings: %v", err)
	}
	if meta := s.Metadata(); meta.Model != "flux-thalia-en" || meta.VoiceID != "flux-thalia-en" {
		t.Errorf("Metadata() = %+v, want the new voice as the model", meta)
	}
	if server.dialCount() != 2 || server.lastQuery(t).Get("model") != "flux-thalia-en" {
		t.Errorf("dialed %d connections, the last with %v, want one with the new voice",
			server.dialCount(), server.lastQuery(t))
	}
}

// A rate Flux does not synthesize is warned about when connecting, and still
// sent.
func TestFluxTTSUnsupportedSampleRateWarns(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	defer slog.SetDefault(prev)

	server, endpoint := newFluxTTSServer(t, fluxTTSServerOptions{})
	cfg := fluxTTSTestConfig(endpoint)
	cfg.SampleRate = 22050
	svc, _ := newFluxTTS(cfg)

	runFluxTTSPipeline(t, svc)

	if !strings.Contains(buf.String(), "22050") {
		t.Errorf("logged %q, want a warning naming the rate", buf.String())
	}
	if got := server.lastQuery(t).Get("sample_rate"); got != "22050" {
		t.Errorf("sample_rate = %q, want the configured rate", got)
	}
}

// A sentence that finds the connection closed by the server drops it and dials
// again in full, restarting the receive loop, and goes out on the new
// connection.
func TestFluxTTSRunTTSRedialsAClosedConnection(t *testing.T) {
	server, endpoint := newFluxTTSServer(t, fluxTTSServerOptions{})
	_, s := newFluxTTS(fluxTTSTestConfig(endpoint))
	t.Cleanup(func() { s.Disconnect(context.Background()) })
	// Dialed with no receive loop, then marked closed the way the loop marks a
	// connection whose read ended.
	if err := s.ConnectWebsocket(t.Context()); err != nil {
		t.Fatalf("connect: %v", err)
	}
	s.mu.Lock()
	s.closed = true
	s.mu.Unlock()

	if err := s.RunTTS(t.Context(), "hello", "c1", nil); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if server.dialCount() != 2 || !s.Connected() || !s.ws.ReceiveTaskRunning() {
		t.Errorf("dialed %d connections (connected %v, receiving %v), want a fresh one with its loop",
			server.dialCount(), s.Connected(), s.ws.ReceiveTaskRunning())
	}
	if sent := server.awaitSent(t, 1); sent[0]["type"] != "Speak" || sent[0]["text"] != "hello" {
		t.Errorf("sent %v, want the text on the new connection", sent)
	}
}

// A sentence that cannot get a connection fails, naming the service, rather
// than being taken as spoken.
func TestFluxTTSRunTTSWithoutAConnectionFails(t *testing.T) {
	_, s := newFluxTTS(fluxTTSTestConfig(refusingServer(t, http.StatusServiceUnavailable, "")))
	t.Cleanup(func() { s.Disconnect(context.Background()) })

	err := s.RunTTS(t.Context(), "hello", "c1", nil)
	if !errors.Is(err, errFluxTTSNotConnected) {
		t.Errorf("RunTTS = %v, want the service reported not connected", err)
	}
}

// The server's error message is pushed as a non-fatal error naming its code,
// and the turn's audio still plays.
func TestFluxTTSServerErrorIsPushed(t *testing.T) {
	_, s := connectedFluxSynth(t, fluxTTSTestConfig(fluxErrorServer(t)))
	b := s.base
	var mu sync.Mutex
	var pushed []*frames.ErrorFrame
	events.On(b.Events(), processor.EventError, func(_ context.Context, ef *frames.ErrorFrame) {
		mu.Lock()
		pushed = append(pushed, ef)
		mu.Unlock()
	})
	if err := s.RunTTS(t.Context(), "hello", "c1", nil); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		mu.Lock()
		n := len(pushed)
		mu.Unlock()
		if n > 0 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(pushed) != 1 || !strings.Contains(pushed[0].Error, "error INVALID_TEXT: text was empty") || pushed[0].Fatal {
		t.Errorf("pushed %v, want one non-fatal error naming the code and description", pushed)
	}
}

// fluxErrorServer answers every message with a Flux Error.
func fluxErrorServer(t *testing.T) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		for {
			if _, _, err := c.Read(r.Context()); err != nil {
				return
			}
			_ = c.Write(r.Context(), websocket.MessageText, mustJSON(map[string]any{
				"type": "Error", "code": "INVALID_TEXT", "description": "text was empty",
			}))
		}
	}))
	t.Cleanup(srv.Close)
	return wsURL(srv.URL)
}

// a setting Deepgram refuses would be refused again on a retry, so the refusal
// is reported with Deepgram's reason and the service gives up.
func TestFluxTTSRefusedSettingIsPermanent(t *testing.T) {
	reason := "'expressivity' must be between -2 and 2. Got: 99."
	b, s := newFluxTTS(fluxTTSTestConfig(refusingServer(t, http.StatusBadRequest, reason)))
	var mu sync.Mutex
	var connectionErrors []string
	events.On(b.Events(), service.EventConnectionError, func(_ context.Context, msg string) {
		mu.Lock()
		connectionErrors = append(connectionErrors, msg)
		mu.Unlock()
	})

	_ = s.ConnectWebsocket(t.Context())

	if b.Usable() {
		t.Error("the service is still usable after a refused setting")
	}
	// Event handlers run in the background; let this one finish.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		mu.Lock()
		n := len(connectionErrors)
		mu.Unlock()
		if n > 0 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	mu.Lock()
	defer mu.Unlock()
	want := "Deepgram refused the connection (HTTP 400): " + reason
	if !slices.Equal(connectionErrors, []string{want}) {
		t.Errorf("connection errors = %q, want %q", connectionErrors, want)
	}
}

// A server error is worth retrying, so the service stays usable.
func TestFluxTTSServerErrorIsNotPermanent(t *testing.T) {
	b, s := newFluxTTS(fluxTTSTestConfig(refusingServer(t, http.StatusServiceUnavailable, "")))

	_ = s.ConnectWebsocket(t.Context())

	if !b.Usable() {
		t.Error("a server error made the service unusable")
	}
}

func mustJSON(v any) []byte {
	b, _ := json.Marshal(v)
	return b
}
