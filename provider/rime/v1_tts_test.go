package rime

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/provider/rime/internal/rimepb"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
	errs "github.com/gojargo/jargo/utils/errors"
	"github.com/gojargo/jargo/utils/events"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

// v1Reply is how the fake Rime v1 server answers one request.
type v1Reply func(req *rimepb.WebSocketRequest) []*rimepb.WebSocketResponse

func v1Started(req *rimepb.WebSocketRequest) *rimepb.WebSocketResponse {
	return &rimepb.WebSocketResponse{
		ContextId: req.GetContextId(),
		Payload: &rimepb.WebSocketResponse_Started{
			Started: &rimepb.WebSocketStarted{RequestId: "request-1"},
		},
	}
}

func v1Audio(req *rimepb.WebSocketRequest, audio []byte) *rimepb.WebSocketResponse {
	return &rimepb.WebSocketResponse{
		ContextId: req.GetContextId(),
		Payload:   &rimepb.WebSocketResponse_Audio{Audio: audio},
	}
}

func v1Done(req *rimepb.WebSocketRequest) *rimepb.WebSocketResponse {
	return &rimepb.WebSocketResponse{
		ContextId: req.GetContextId(),
		Payload:   &rimepb.WebSocketResponse_Done{Done: &rimepb.WebSocketDone{}},
	}
}

func v1ContextError(req *rimepb.WebSocketRequest, kind string) *rimepb.WebSocketResponse {
	return &rimepb.WebSocketResponse{
		ContextId: req.GetContextId(),
		Payload: &rimepb.WebSocketResponse_Error{
			Error: &rimepb.WebSocketError{Kind: kind, Message: "rejected"},
		},
	}
}

func v1ReadyResponse() *rimepb.WebSocketResponse {
	return &rimepb.WebSocketResponse{
		Payload: &rimepb.WebSocketResponse_Ready{Ready: &rimepb.WebSocketReady{Protocol: 1}},
	}
}

// payloadName names the payload of a request the way the schema does.
func payloadName(req *rimepb.WebSocketRequest) string {
	switch req.GetPayload().(type) {
	case *rimepb.WebSocketRequest_Config:
		return "config"
	case *rimepb.WebSocketRequest_Start:
		return "start"
	case *rimepb.WebSocketRequest_Text:
		return "text"
	case *rimepb.WebSocketRequest_End:
		return "end"
	case *rimepb.WebSocketRequest_Cancel:
		return "cancel"
	default:
		return ""
	}
}

// speakingReply replies like Rime: started on start, audio per text, done on
// end.
func speakingReply(audio []byte) v1Reply {
	return func(req *rimepb.WebSocketRequest) []*rimepb.WebSocketResponse {
		switch payloadName(req) {
		case "start":
			return []*rimepb.WebSocketResponse{v1Started(req)}
		case "text":
			return []*rimepb.WebSocketResponse{v1Audio(req, audio)}
		case "end":
			return []*rimepb.WebSocketResponse{v1Done(req)}
		default:
			return nil
		}
	}
}

// fakeRimeV1 serves Rime v1 locally and records the requests it receives.
type fakeRimeV1 struct {
	t     *testing.T
	reply v1Reply
	ready *rimepb.WebSocketResponse
	// closeFirstAfterEnd closes the first connection once it has answered an
	// end.
	closeFirstAfterEnd bool
	url                string

	mu          sync.Mutex
	requests    []*rimepb.WebSocketRequest
	connections int
}

func newFakeRimeV1(t *testing.T, f *fakeRimeV1) *fakeRimeV1 {
	t.Helper()
	f.t = t
	if f.ready == nil {
		f.ready = v1ReadyResponse()
	}
	srv := httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(srv.Close)
	f.url = "ws" + strings.TrimPrefix(srv.URL, "http")
	return f
}

func (f *fakeRimeV1) serve(w http.ResponseWriter, r *http.Request) {
	c, err := websocket.Accept(w, r, &websocket.AcceptOptions{
		Subprotocols: []string{"rime.v1.binary", "rime.v1.json"},
	})
	if err != nil {
		return
	}
	defer func() { _ = c.CloseNow() }()
	f.mu.Lock()
	f.connections++
	connection := f.connections
	f.mu.Unlock()

	ctx := r.Context()
	binary := c.Subprotocol() == "rime.v1.binary"
	send := func(resp *rimepb.WebSocketResponse) error {
		if binary {
			data, _ := proto.Marshal(resp)
			return c.Write(ctx, websocket.MessageBinary, data)
		}
		data, _ := protojson.Marshal(resp)
		return c.Write(ctx, websocket.MessageText, data)
	}
	if send(f.ready) != nil {
		return
	}
	for {
		_, data, err := c.Read(ctx)
		if err != nil {
			return
		}
		req := &rimepb.WebSocketRequest{}
		if binary {
			err = proto.Unmarshal(data, req)
		} else {
			err = protojson.Unmarshal(data, req)
		}
		if err != nil {
			f.t.Errorf("decoding a client request: %v", err)
			return
		}
		f.mu.Lock()
		f.requests = append(f.requests, req)
		f.mu.Unlock()
		for _, resp := range f.reply(req) {
			if send(resp) != nil {
				return
			}
		}
		if f.closeFirstAfterEnd && connection == 1 && payloadName(req) == "end" {
			_ = c.Close(websocket.StatusNormalClosure, "")
			return
		}
	}
}

func (f *fakeRimeV1) received() []*rimepb.WebSocketRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.requests)
}

func (f *fakeRimeV1) connectionCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.connections
}

func payloadNames(reqs []*rimepb.WebSocketRequest) []string {
	out := make([]string, 0, len(reqs))
	for _, r := range reqs {
		out = append(out, payloadName(r))
	}
	return out
}

// step is one thing a test does to the running pipeline: queue a frame, or
// wait.
type step struct {
	frame frames.Frame
	sleep time.Duration
}

func queue(f frames.Frame) step  { return step{frame: f} }
func pause(d time.Duration) step { return step{sleep: d} }

// runTTSTest runs the service in a pipeline through the steps, ends the
// pipeline, and returns the frames that left it downstream and upstream.
func runTTSTest(t *testing.T, svc *tts.Base, steps ...step) (down, up []frames.Frame) {
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

// framesOf is every frame of type T.
func framesOf[T frames.Frame](fs []frames.Frame) []T {
	var out []T
	for _, f := range fs {
		if v, ok := f.(T); ok {
			out = append(out, v)
		}
	}
	return out
}

// spokenTexts is the text of every TTSTextFrame.
func spokenTexts(fs []frames.Frame) []string {
	var out []string
	for _, f := range framesOf[*frames.TTSTextFrame](fs) {
		out = append(out, f.Text)
	}
	return out
}

// audioBytes is the audio of every TTSAudioRawFrame, joined.
func audioBytes(fs []frames.Frame) []byte {
	var out []byte
	for _, f := range framesOf[*frames.TTSAudioRawFrame](fs) {
		out = append(out, f.Audio...)
	}
	return out
}

// indexOf is the position of the first frame of type T, or -1.
func indexOf[T frames.Frame](fs []frames.Frame) int {
	for i, f := range fs {
		if _, ok := f.(T); ok {
			return i
		}
	}
	return -1
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

// testV1TTS builds the V1 service, failing the test on a configuration error.
func testV1TTS(t *testing.T, cfg V1TTSConfig) (*tts.Base, *v1Synthesizer) {
	t.Helper()
	if cfg.APIKey == "" {
		cfg.APIKey = "key"
	}
	b, s, err := newV1TTS(cfg)
	if err != nil {
		t.Fatalf("newV1TTS: %v", err)
	}
	return b, s
}

// startOf builds the start request for a context and returns its start.
func startOf(s *v1Synthesizer) *rimepb.SynthesisRequest {
	return s.buildStartRequest("turn").GetStart()
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

func TestV1TTSCodaUsesBinaryAndPublicEndpoint(t *testing.T) {
	b, s := testV1TTS(t, V1TTSConfig{Model: "coda"})

	if s.cfg.URL != "wss://coda.api.rime.ai/ws" {
		t.Errorf("URL = %q, want the coda endpoint", s.cfg.URL)
	}
	if s.cfg.Protocol != V1ProtocolBinary {
		t.Errorf("protocol = %q, want binary", s.cfg.Protocol)
	}
	if got := s.live.Model.Or(""); got != "coda" {
		t.Errorf("model = %q, want coda", got)
	}
	if b.IsStreamingTokens() {
		t.Error("the service streams tokens, want sentences by default")
	}
}

func TestV1TTSModelSelectsThePublicEndpoint(t *testing.T) {
	for model, url := range map[string]string{
		"coda":   "wss://coda.api.rime.ai/ws",
		"mistv3": "wss://mist.api.rime.ai/ws",
	} {
		_, s := testV1TTS(t, V1TTSConfig{Model: model})
		if s.cfg.URL != url {
			t.Errorf("%s: URL = %q, want %q", model, s.cfg.URL, url)
		}
		if got := s.live.Model.Or(""); got != model {
			t.Errorf("%s: model = %q", model, got)
		}
	}
}

func TestV1TTSCustomEndpointUsesTheExplicitModel(t *testing.T) {
	for _, url := range []string{
		"wss://coda.api.rime.ai/ws",
		"wss://mist.api.rime.ai/ws",
		"wss://voice.customer.example/custom/stream?deployment=test",
		"wss://api.rime.ai/coda/ws",
		"wss://api.rime.ai/mist/ws",
	} {
		_, s := testV1TTS(t, V1TTSConfig{URL: url, Model: "custom-model"})
		if s.cfg.URL != url {
			t.Errorf("URL = %q, want %q", s.cfg.URL, url)
		}
		if got := s.live.Model.Or(""); got != "custom-model" {
			t.Errorf("%s: model = %q, want custom-model", url, got)
		}
		if s.endpointModel != "custom-model" {
			t.Errorf("%s: endpoint model = %q, want custom-model", url, s.endpointModel)
		}
	}
}

func TestV1TTSCustomModelRequiresAnEndpoint(t *testing.T) {
	_, err := NewV1TTS(V1TTSConfig{APIKey: "key", Model: "custom-model"})
	if !errors.Is(err, errV1URLRequired) {
		t.Errorf("err = %v, want %v", err, errV1URLRequired)
	}
}

func TestV1TTSEveryEndpointRequiresAnExplicitModel(t *testing.T) {
	for _, url := range []string{
		"",
		"wss://coda.api.rime.ai/ws",
		"wss://mist.api.rime.ai/ws",
		"wss://voice.customer.example/custom/stream",
		"wss://api.rime.ai/coda/ws",
		"wss://api.rime.ai/mist/ws",
	} {
		_, err := NewV1TTS(V1TTSConfig{APIKey: "key", URL: url})
		if !errors.Is(err, errV1ModelRequired) {
			t.Errorf("URL %q: err = %v, want %v", url, err, errV1ModelRequired)
		}
	}
}

func TestV1TTSRejectsInvalidConfiguration(t *testing.T) {
	_, err := NewV1TTS(V1TTSConfig{APIKey: "key", Model: "coda", Protocol: "xml"})
	if !errors.Is(err, errV1Protocol) {
		t.Errorf("err = %v, want %v", err, errV1Protocol)
	}
}

func TestV1TTSRejectsNonWebsocketURLs(t *testing.T) {
	for _, url := range []string{"https://coda.api.rime.ai/ws", "/ws", "wss:///ws"} {
		_, err := NewV1TTS(V1TTSConfig{APIKey: "key", Model: "coda", URL: url})
		if !errors.Is(err, errV1URLInvalid) {
			t.Errorf("URL %q: err = %v, want %v", url, err, errV1URLInvalid)
		}
	}
}

func TestV1TTSSettingsTheModelDoesNotAcceptAreLeftOutWithAWarning(t *testing.T) {
	cases := []struct {
		url     string
		cfg     V1TTSConfig
		ignored string
	}{
		{"wss://mist.api.rime.ai/ws", V1TTSConfig{Model: "mistv3", TextLookaheadTokens: new(4)}, "text_lookahead_tokens"},
		{
			"wss://voice.customer.example/ws",
			V1TTSConfig{Model: "arcana", TextLookaheadTokens: new(4)},
			"text_lookahead_tokens",
		},
		{"wss://coda.api.rime.ai/ws", V1TTSConfig{Model: "coda", PauseBetweenBrackets: new(true)}, "pause_between_brackets"},
		{"wss://coda.api.rime.ai/ws", V1TTSConfig{Model: "coda", SaveOovs: new(false)}, "save_oovs"},
		{"wss://customer.rime.ai/ws", V1TTSConfig{Model: "mistv2", TimeScaleFactor: new(1.2)}, "time_scale_factor"},
	}
	for _, tc := range cases {
		t.Run(tc.ignored+"/"+tc.cfg.Model, func(t *testing.T) {
			warnings := captureWarnings(t)
			tc.cfg.URL = tc.url
			_, s := testV1TTS(t, tc.cfg)

			start := startOf(s)
			if c := start.GetCodaParameters(); c != nil && c.TextLookaheadTokens != nil {
				t.Error("text_lookahead_tokens was sent")
			}
			if start.MistParameters != nil {
				t.Error("mist parameters were sent")
			}
			if start.GetAudioParameters().TimeScaleFactor != nil {
				t.Error("time_scale_factor was sent")
			}
			if _, ok := settings.Get(s.live, tc.ignored); !ok {
				t.Errorf("%s was not kept in the settings", tc.ignored)
			}
			if !strings.Contains(warnings(), tc.ignored) {
				t.Errorf("no warning names %s: %q", tc.ignored, warnings())
			}
		})
	}
}

func TestV1TTSUnknownSettingsAreIgnored(t *testing.T) {
	_, s := testV1TTS(t, V1TTSConfig{Model: "coda"})
	delta := &V1TTSSettings{}
	if err := settings.FromMap(delta, map[string]any{"model": "coda", "repetition_penalty": 1.1}); err != nil {
		t.Fatalf("FromMap: %v", err)
	}
	if _, err := settings.Apply(s.live, delta); err != nil {
		t.Fatalf("Apply: %v", err)
	}

	if got := startOf(s).GetSpeaker(); got != "" {
		t.Errorf("speaker = %q, want none", got)
	}
}

func TestV1TTSCodaSettingsMapToTheStartRequest(t *testing.T) {
	_, s := testV1TTS(t, V1TTSConfig{
		Model:               "coda",
		Speaker:             "astra",
		Language:            language.English,
		SampleRate:          22050,
		TextLookaheadTokens: new(0),
		TimeScaleFactor:     new(1.2),
	})

	start := startOf(s)
	if start.GetSpeaker() != "astra" {
		t.Errorf("speaker = %q, want astra", start.GetSpeaker())
	}
	if start.GetLanguage() != "en" {
		t.Errorf("language = %q, want en", start.GetLanguage())
	}
	audio := start.GetAudioParameters()
	if audio.GetAudioFormat() != "audio/pcm" {
		t.Errorf("audio format = %q, want audio/pcm", audio.GetAudioFormat())
	}
	if audio.GetSamplingRate() != 22050 {
		t.Errorf("sampling rate = %d, want 22050", audio.GetSamplingRate())
	}
	if got := audio.GetTimeScaleFactor(); got < 1.1999 || got > 1.2001 {
		t.Errorf("time scale factor = %v, want 1.2", got)
	}
	if c := start.GetCodaParameters(); c == nil || c.TextLookaheadTokens == nil || c.GetTextLookaheadTokens() != 0 {
		t.Errorf("text lookahead tokens = %v, want 0 sent", start.GetCodaParameters())
	}
	if start.MistParameters != nil {
		t.Error("mist parameters were sent to coda")
	}
}

func TestV1TTSTokenModeAsksRimeToAccumulateSentences(t *testing.T) {
	_, sentences := testV1TTS(t, V1TTSConfig{Model: "coda"})
	if startOf(sentences).SplitStrategy != nil {
		t.Errorf("split strategy = %v, want none", startOf(sentences).GetSplitStrategy())
	}

	_, tokens := testV1TTS(t, V1TTSConfig{Model: "coda", TextAggregation: frames.TextTypeToken})
	if got := startOf(tokens).GetSplitStrategy(); got != rimepb.SplitStrategy_SPLIT_STRATEGY_ACCUMULATE_SENTENCE {
		t.Errorf("split strategy = %v, want accumulate sentence", got)
	}
}

func TestV1TTSLanguagesMapToBCP47Tags(t *testing.T) {
	f := newFakeRimeV1(t, &fakeRimeV1{reply: speakingReply([]byte{1, 2})})
	b, s := testV1TTS(t, V1TTSConfig{Model: "coda", URL: f.url, Language: language.JapaneseJP})
	if got := startOf(s).GetLanguage(); got != "ja" {
		t.Errorf("language = %q, want ja", got)
	}

	update := &V1TTSSettings{}
	update.Language = settings.Set(string(language.GermanDE))
	runTTSTest(t, b, queue(frames.NewTTSUpdateSettingsFrame(update)), pause(50*time.Millisecond))

	if got := startOf(s).GetLanguage(); got != "de" {
		t.Errorf("language after the update = %q, want de", got)
	}
}

func TestV1TTSMistSettingsMapToTheStartRequest(t *testing.T) {
	_, s := testV1TTS(t, V1TTSConfig{
		Model:                "mistv3",
		URL:                  "wss://mist.api.rime.ai/ws",
		PauseBetweenBrackets: new(true),
		SaveOovs:             new(false),
	})

	start := startOf(s)
	mist := start.GetMistParameters()
	if mist.PauseBetweenBrackets == nil || !mist.GetPauseBetweenBrackets() {
		t.Errorf("pause_between_brackets = %v, want true", mist.PauseBetweenBrackets)
	}
	if mist.SaveOovs == nil || mist.GetSaveOovs() {
		t.Errorf("save_oovs = %v, want false", mist.SaveOovs)
	}
	if start.CodaParameters != nil {
		t.Error("coda parameters were sent to mist")
	}
}

func TestV1TTSRuntimeSettingsApplyToTheNextContext(t *testing.T) {
	warnings := captureWarnings(t)
	_, s := testV1TTS(t, V1TTSConfig{Model: "coda", Speaker: "old"})

	update := &V1TTSSettings{}
	update.Voice = settings.Set("new")
	changed, err := settings.Apply(s.live, update)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if !reflect.DeepEqual(changed, settings.Changed{"voice": "old"}) {
		t.Errorf("changed = %v, want voice from old", changed)
	}
	if got := startOf(s).GetSpeaker(); got != "new" {
		t.Errorf("speaker = %q, want new", got)
	}
	if w := warnings(); w != "" {
		t.Errorf("warned %q, want nothing", w)
	}

	// The endpoint serves one model, so a model change is reported, not
	// applied.
	update = &V1TTSSettings{}
	update.Model = settings.Set("mistv3")
	if _, err := settings.Apply(s.live, update); err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if startOf(s).MistParameters != nil {
		t.Error("mist parameters were sent to the coda endpoint")
	}
	if !strings.Contains(warnings(), "mistv3 is not applied") {
		t.Errorf("no warning that mistv3 is not applied: %q", warnings())
	}

	_, mist := testV1TTS(t, V1TTSConfig{Model: "mistv3", URL: "wss://mist.api.rime.ai/ws"})
	update = &V1TTSSettings{TextLookaheadTokens: settings.Set(4)}
	if _, err := settings.Apply(mist.live, update); err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if startOf(mist).CodaParameters != nil {
		t.Error("coda parameters were sent to mist")
	}
	if !strings.Contains(warnings(), "text_lookahead_tokens") {
		t.Errorf("no warning names text_lookahead_tokens: %q", warnings())
	}
}

// ---------------------------------------------------------------------------
// Wire format
// ---------------------------------------------------------------------------

// Fixed envelopes for each message the service sends or reads. The local server
// shares the schema with the service, so only these catch a schema change that
// renumbers or renames a field.
//
//nolint:gochecknoglobals // test fixture
var v1RequestEnvelopes = []struct{ envelope, binary string }{
	{
		`{"contextId": "ctx", "start": {"language": "en", "speaker": "luna", "text": "Hello.",` +
			` "audioParameters": {"audioFormat": "audio/pcm", "samplingRate": 24000, "timeScaleFactor": 1.25},` +
			` "splitStrategy": "SPLIT_STRATEGY_NONE", "codaParameters": {"textLookaheadTokens": 0}}}`,
		"0a036374781a2e1202656e1a046c756e61220648656c6c6f2e2a140a09617564696f2f70636d" +
			"10c0bb011d0000a03f300242024000",
	},
	{`{"contextId": "ctx", "text": "Hello."}`, "0a03637478220648656c6c6f2e"},
	{`{"contextId": "ctx", "end": {}}`, "0a036374782a00"},
	{`{"contextId": "ctx", "cancel": {}}`, "0a036374783200"},
}

//nolint:gochecknoglobals // test fixture
var v1ResponseEnvelopes = []struct{ envelope, binary string }{
	{
		`{"ready": {"protocol": 1, "languages": ["en", "es"], "defaultLanguage": "en"}}`,
		"120e08011202656e120265731a02656e",
	},
	{`{"contextId": "ctx", "started": {"requestId": "req"}}`, "0a036374781a050a03726571"},
	{`{"contextId": "ctx", "audio": "AAH+/w=="}`, "0a0363747822040001feff"},
	{`{"contextId": "ctx", "done": {}}`, "0a036374782a00"},
	{`{"contextId": "ctx", "cancelled": {}}`, "0a036374783200"}, //nolint:misspell // Rime's field name
	{
		`{"contextId": "ctx", "error": {"kind": "invalid_input", "message": "test error", "requestId": "req"}}`,
		"0a036374783a200a0d696e76616c69645f696e707574120a74657374206572726f721a03726571",
	},
}

// jsonValue decodes JSON into a plain value, for comparing two encodings.
func jsonValue(t *testing.T, data []byte) any {
	t.Helper()
	var v any
	if err := json.Unmarshal(data, &v); err != nil {
		t.Fatalf("decoding %s: %v", data, err)
	}
	return v
}

func TestV1TTSRequestsMatchFixedWireEnvelopes(t *testing.T) {
	_, binary := testV1TTS(t, V1TTSConfig{Model: "coda"})
	_, jsonSynth := testV1TTS(t, V1TTSConfig{Model: "coda", Protocol: V1ProtocolJSON})
	for _, tc := range v1RequestEnvelopes {
		req := &rimepb.WebSocketRequest{}
		if err := protojson.Unmarshal([]byte(tc.envelope), req); err != nil {
			t.Fatalf("parsing %s: %v", tc.envelope, err)
		}

		typ, got, err := binary.encode(req)
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		if typ != websocket.MessageBinary || hex.EncodeToString(got) != tc.binary {
			t.Errorf("binary %s = %x (%v), want %s", tc.envelope, got, typ, tc.binary)
		}

		typ, got, err = jsonSynth.encode(req)
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		if typ != websocket.MessageText || !reflect.DeepEqual(jsonValue(t, got), jsonValue(t, []byte(tc.envelope))) {
			t.Errorf("JSON = %s (%v), want %s", got, typ, tc.envelope)
		}
	}
}

func TestV1TTSResponsesMatchFixedWireEnvelopes(t *testing.T) {
	_, binary := testV1TTS(t, V1TTSConfig{Model: "coda"})
	_, jsonSynth := testV1TTS(t, V1TTSConfig{Model: "coda", Protocol: V1ProtocolJSON})
	for _, tc := range v1ResponseEnvelopes {
		raw, err := hex.DecodeString(tc.binary)
		if err != nil {
			t.Fatal(err)
		}
		fromBinary, err := binary.decode(websocket.MessageBinary, raw)
		if err != nil {
			t.Fatalf("decode binary %s: %v", tc.binary, err)
		}
		fromJSON, err := jsonSynth.decode(websocket.MessageText, []byte(tc.envelope))
		if err != nil {
			t.Fatalf("decode JSON %s: %v", tc.envelope, err)
		}
		want := jsonValue(t, []byte(tc.envelope))
		for name, resp := range map[string]*rimepb.WebSocketResponse{"binary": fromBinary, "JSON": fromJSON} {
			data, err := protojson.Marshal(resp)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(jsonValue(t, data), want) {
				t.Errorf("from %s = %s, want %s", name, data, tc.envelope)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Synthesis against a local Rime v1 server
// ---------------------------------------------------------------------------

func TestV1TTSRoundTripStreamsOneContextPerTurn(t *testing.T) {
	for _, protocol := range []string{V1ProtocolBinary, V1ProtocolJSON} {
		t.Run(protocol, func(t *testing.T) {
			f := newFakeRimeV1(t, &fakeRimeV1{reply: speakingReply([]byte{1, 2, 3, 4})})
			b, _ := testV1TTS(t, V1TTSConfig{
				Model:      "coda",
				URL:        f.url + "/custom/stream",
				Protocol:   protocol,
				SampleRate: 24000,
			})

			down, up := runTTSTest(t, b,
				queue(frames.NewTTSSpeakFrame("Hello from the bot. This is a second sentence.")),
				pause(200*time.Millisecond),
				queue(frames.NewBotStoppedSpeakingFrame()),
			)

			reqs := f.received()
			if got, want := payloadNames(reqs), []string{"start", "text", "text", "end"}; !slices.Equal(got, want) {
				t.Fatalf("requests = %v, want %v", got, want)
			}
			ids := map[string]bool{}
			for _, r := range reqs {
				ids[r.GetContextId()] = true
			}
			if len(ids) != 1 {
				t.Errorf("contexts = %v, want one", ids)
			}
			if got := reqs[0].GetStart().GetAudioParameters().GetSamplingRate(); got != 24000 {
				t.Errorf("sampling rate = %d, want 24000", got)
			}
			texts := []string{reqs[1].GetText(), reqs[2].GetText()}
			if want := []string{"Hello from the bot. ", "This is a second sentence. "}; !slices.Equal(texts, want) {
				t.Errorf("texts = %q, want %q", texts, want)
			}
			if errs := framesOf[*frames.ErrorFrame](slices.Concat(down, up)); len(errs) > 0 {
				t.Errorf("errors = %v", errs)
			}
			if got := audioBytes(down); !bytes.Equal(got, bytes.Repeat([]byte{1, 2, 3, 4}, 2)) {
				t.Errorf("audio = % x, want both sentences' audio", got)
			}
			started := indexOf[*frames.TTSStartedFrame](down)
			audio := indexOf[*frames.TTSAudioRawFrame](down)
			stopped := indexOf[*frames.TTSStoppedFrame](down)
			text := indexOf[*frames.TTSTextFrame](down)
			if started < 0 || audio < started || stopped < audio || text < 0 || stopped < text {
				t.Errorf("order: started %d, audio %d, text %d, stopped %d", started, audio, text, stopped)
			}
		})
	}
}

func TestV1TTSTokenModeStreamsTokensAndKeepsSpellTagsWhole(t *testing.T) {
	f := newFakeRimeV1(t, &fakeRimeV1{reply: speakingReply([]byte{1, 2})})
	b, _ := testV1TTS(t, V1TTSConfig{
		Model:           "coda",
		URL:             f.url + "/custom/stream",
		TextAggregation: frames.TextTypeToken,
	})

	steps := []step{queue(frames.NewLLMFullResponseStartFrame())}
	for _, token := range []string{"Say ", "spe", "ll(A", "B) ", "now", "."} {
		steps = append(steps, queue(frames.NewLLMTextFrame(token)))
	}
	steps = append(steps, queue(frames.NewLLMFullResponseEndFrame()), pause(200*time.Millisecond))
	down, up := runTTSTest(t, b, steps...)

	reqs := f.received()
	want := []string{"start", "text", "text", "text", "text", "end"}
	if got := payloadNames(reqs); !slices.Equal(got, want) {
		t.Fatalf("requests = %v, want %v", got, want)
	}
	if got := reqs[0].GetStart().GetSplitStrategy(); got != rimepb.SplitStrategy_SPLIT_STRATEGY_ACCUMULATE_SENTENCE {
		t.Errorf("split strategy = %v, want accumulate sentence", got)
	}
	var texts []string
	for _, r := range reqs[1:5] {
		texts = append(texts, r.GetText())
	}
	if want := []string{"Say ", "spell(AB) ", "now", "."}; !slices.Equal(texts, want) {
		t.Errorf("texts = %q, want %q", texts, want)
	}
	if errs := framesOf[*frames.ErrorFrame](slices.Concat(down, up)); len(errs) > 0 {
		t.Errorf("errors = %v", errs)
	}
	if n := len(framesOf[*frames.TTSStoppedFrame](down)); n != 1 {
		t.Errorf("stopped frames = %d, want 1", n)
	}
}

func TestV1TTSInterruptionCancelsTheContextAndDropsLateAudio(t *testing.T) {
	reply := func(req *rimepb.WebSocketRequest) []*rimepb.WebSocketResponse {
		switch payloadName(req) {
		case "start":
			return []*rimepb.WebSocketResponse{v1Started(req)}
		case "text":
			return []*rimepb.WebSocketResponse{v1Audio(req, []byte{1, 2})}
		case "cancel":
			return []*rimepb.WebSocketResponse{
				v1Audio(req, []byte{3, 4}),
				{
					ContextId: req.GetContextId(),
					Payload:   &rimepb.WebSocketResponse_Cancelled{Cancelled: &rimepb.WebSocketCancelled{}}, //nolint:misspell,lll // Rime's field name
				},
			}
		default:
			return nil
		}
	}
	f := newFakeRimeV1(t, &fakeRimeV1{reply: reply})
	b, _ := testV1TTS(t, V1TTSConfig{Model: "coda", URL: f.url + "/custom/stream"})

	down, up := runTTSTest(t, b,
		queue(frames.NewTTSSpeakFrame("Stop me.")),
		pause(50*time.Millisecond),
		queue(frames.NewInterruptionFrame()),
		pause(100*time.Millisecond),
	)

	if got, want := payloadNames(f.received()), []string{"start", "text", "end", "cancel"}; !slices.Equal(got, want) {
		t.Errorf("requests = %v, want %v", got, want)
	}
	if errs := framesOf[*frames.ErrorFrame](slices.Concat(down, up)); len(errs) > 0 {
		t.Errorf("errors = %v", errs)
	}
	if got := audioBytes(down); !bytes.Equal(got, []byte{1, 2}) {
		t.Errorf("audio = % x, want only what came before the interruption", got)
	}
}

func TestV1TTSContextErrorReportsAndSkipsTheRestOfTheTurn(t *testing.T) {
	reply := func(req *rimepb.WebSocketRequest) []*rimepb.WebSocketResponse {
		switch payloadName(req) {
		case "start":
			return []*rimepb.WebSocketResponse{v1Started(req)}
		case "text":
			return []*rimepb.WebSocketResponse{v1ContextError(req, "internal")}
		default:
			return nil
		}
	}
	failedTurn := func(first, second string) []step {
		return []step{
			queue(frames.NewLLMFullResponseStartFrame()),
			queue(frames.NewAggregatedTextFrame(first, frames.TextTypeSentence)),
			pause(100 * time.Millisecond),
			queue(frames.NewAggregatedTextFrame(second, frames.TextTypeSentence)),
			queue(frames.NewLLMFullResponseEndFrame()),
			pause(100 * time.Millisecond),
		}
	}
	f := newFakeRimeV1(t, &fakeRimeV1{reply: reply})
	b, _ := testV1TTS(t, V1TTSConfig{Model: "coda", URL: f.url + "/custom/stream"})

	startedAt := time.Now()
	down, up := runTTSTest(t, b,
		slices.Concat(failedTurn("First one.", "First two."), failedTurn("Second one.", "Second two."))...)
	elapsed := time.Since(startedAt)

	// Each turn's later sentence is skipped after Rime fails its context.
	if got, want := payloadNames(f.received()), []string{"start", "text", "start", "text"}; !slices.Equal(got, want) {
		t.Errorf("requests = %v, want %v", got, want)
	}
	var rimeErrors []*frames.ErrorFrame
	for _, e := range framesOf[*frames.ErrorFrame](up) {
		if strings.Contains(e.Error, "internal") {
			rimeErrors = append(rimeErrors, e)
		}
	}
	if len(rimeErrors) != 2 {
		t.Fatalf("Rime errors = %v, want 2", rimeErrors)
	}
	if rimeErrors[0].Category != errs.Server {
		t.Errorf("category = %q, want server", rimeErrors[0].Category)
	}
	if n := len(framesOf[*frames.TTSAudioRawFrame](down)); n != 0 {
		t.Errorf("audio frames = %d, want none", n)
	}
	// The failed context stays open for the rest of its turn instead of being
	// reopened empty, so each turn ends promptly with one started/stopped pair.
	if n := len(framesOf[*frames.TTSStartedFrame](down)); n != 2 {
		t.Errorf("started frames = %d, want 2", n)
	}
	if n := len(framesOf[*frames.TTSStoppedFrame](down)); n != 2 {
		t.Errorf("stopped frames = %d, want 2", n)
	}
	if elapsed >= 2*time.Second {
		t.Errorf("the turns took %v, want them to end promptly", elapsed)
	}
	// Nothing was spoken, so none of the text reaches the context.
	if texts := spokenTexts(down); len(texts) != 0 {
		t.Errorf("spoken texts = %q, want none", texts)
	}
	if !b.Usable() {
		t.Error("the service is unusable after a failed context")
	}
}

func TestV1TTSPermanentContextErrorLeavesTheServiceUsable(t *testing.T) {
	reply := func(req *rimepb.WebSocketRequest) []*rimepb.WebSocketResponse {
		if payloadName(req) == "start" {
			return []*rimepb.WebSocketResponse{v1ContextError(req, "invalid_input")}
		}
		return nil
	}
	f := newFakeRimeV1(t, &fakeRimeV1{reply: reply})
	b, _ := testV1TTS(t, V1TTSConfig{Model: "coda", URL: f.url + "/custom/stream"})

	down, up := runTTSTest(t, b,
		queue(frames.NewTTSSpeakFrame("Rejected text.")),
		pause(100*time.Millisecond),
	)

	errors := framesOf[*frames.ErrorFrame](up)
	if len(errors) == 0 || errors[0].Category != errs.Unknown {
		t.Errorf("errors = %v, want one categorized unknown", errors)
	}
	if texts := spokenTexts(down); len(texts) != 0 {
		t.Errorf("spoken texts = %q, want none", texts)
	}
	if !b.Usable() {
		t.Error("the service is unusable after a context Rime rejected")
	}
}

func TestV1TTSDoneWithoutAudioEndsTheContext(t *testing.T) {
	reply := func(req *rimepb.WebSocketRequest) []*rimepb.WebSocketResponse {
		switch payloadName(req) {
		case "start":
			return []*rimepb.WebSocketResponse{v1Started(req)}
		case "end":
			return []*rimepb.WebSocketResponse{v1Done(req)}
		default:
			return nil
		}
	}
	f := newFakeRimeV1(t, &fakeRimeV1{reply: reply})
	b, _ := testV1TTS(t, V1TTSConfig{Model: "coda", URL: f.url + "/custom/stream"})

	down, _ := runTTSTest(t, b,
		queue(frames.NewTTSSpeakFrame("This audio is empty.")),
		pause(100*time.Millisecond),
	)

	if n := len(framesOf[*frames.TTSAudioRawFrame](down)); n != 0 {
		t.Errorf("audio frames = %d, want none", n)
	}
	if n := len(framesOf[*frames.TTSStoppedFrame](down)); n == 0 {
		t.Error("no stopped frame, want the context ended")
	}
	if texts := spokenTexts(down); len(texts) != 0 {
		t.Errorf("spoken texts = %q, want none", texts)
	}
}

func TestV1TTSOddLengthAudioIsPaddedWithoutAnError(t *testing.T) {
	reply := func(req *rimepb.WebSocketRequest) []*rimepb.WebSocketResponse {
		switch payloadName(req) {
		case "start":
			return []*rimepb.WebSocketResponse{v1Started(req)}
		case "text":
			return []*rimepb.WebSocketResponse{v1Audio(req, []byte{1, 2, 3}), v1Audio(req, []byte{4, 5})}
		case "end":
			return []*rimepb.WebSocketResponse{v1Done(req)}
		default:
			return nil
		}
	}
	f := newFakeRimeV1(t, &fakeRimeV1{reply: reply})
	b, _ := testV1TTS(t, V1TTSConfig{Model: "coda", URL: f.url + "/custom/stream"})

	down, up := runTTSTest(t, b,
		queue(frames.NewTTSSpeakFrame("Odd audio.")),
		pause(200*time.Millisecond),
		queue(frames.NewBotStoppedSpeakingFrame()),
	)

	if errs := framesOf[*frames.ErrorFrame](slices.Concat(down, up)); len(errs) > 0 {
		t.Errorf("errors = %v", errs)
	}
	if got := audioBytes(down); !bytes.Equal(got, []byte{1, 2, 3, 4, 5, 0}) {
		t.Errorf("audio = % x, want the odd byte padded", got)
	}
}

func TestV1TTSReadyErrorIsReportedWithItsCategory(t *testing.T) {
	ready := &rimepb.WebSocketResponse{
		Payload: &rimepb.WebSocketResponse_Error{
			Error: &rimepb.WebSocketError{Kind: "unauthenticated", Message: "bad key"},
		},
	}
	f := newFakeRimeV1(t, &fakeRimeV1{reply: speakingReply([]byte{1, 2}), ready: ready})
	b, _ := testV1TTS(t, V1TTSConfig{APIKey: "bad", Model: "coda", URL: f.url + "/custom/stream"})

	_, up := runTTSTest(t, b, pause(50*time.Millisecond))

	errors := framesOf[*frames.ErrorFrame](up)
	if len(errors) == 0 || !strings.Contains(errors[0].Error, "unauthenticated") {
		t.Fatalf("errors = %v, want Rime's unauthenticated error", errors)
	}
	if errors[0].Category != errs.Authentication {
		t.Errorf("category = %q, want authentication", errors[0].Category)
	}
	if reqs := f.received(); len(reqs) != 0 {
		t.Errorf("requests = %v, want none", payloadNames(reqs))
	}
}

func TestV1TTSNextTurnReconnectsAfterRimeClosesTheSocket(t *testing.T) {
	f := newFakeRimeV1(t, &fakeRimeV1{reply: speakingReply([]byte{1, 2}), closeFirstAfterEnd: true})
	b, _ := testV1TTS(t, V1TTSConfig{Model: "coda", URL: f.url + "/custom/stream"})

	down, _ := runTTSTest(t, b,
		queue(frames.NewTTSSpeakFrame("First turn.")),
		pause(200*time.Millisecond),
		queue(frames.NewBotStoppedSpeakingFrame()),
		queue(frames.NewTTSSpeakFrame("Second turn.")),
		pause(200*time.Millisecond),
		queue(frames.NewBotStoppedSpeakingFrame()),
	)

	if n := f.connectionCount(); n != 2 {
		t.Errorf("connections = %d, want 2", n)
	}
	if got := audioBytes(down); !bytes.Equal(got, []byte{1, 2, 1, 2}) {
		t.Errorf("audio = % x, want both turns' audio", got)
	}
	if got, want := spokenTexts(down), []string{"First turn.", "Second turn."}; !slices.Equal(got, want) {
		t.Errorf("spoken texts = %q, want %q", got, want)
	}
}
