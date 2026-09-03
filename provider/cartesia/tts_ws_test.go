package cartesia

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"maps"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
	uctx "github.com/gojargo/jargo/utils/context"
)

// session is what the fake synthesis endpoint saw: the headers it was dialed
// with and every request the service sent on the connection.
type session struct {
	mu       sync.Mutex
	header   http.Header
	requests []map[string]any
}

func (s *session) record(m map[string]any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests = append(s.requests, m)
}

// first is the first request the endpoint saw, or nil if it saw none.
func (s *session) first() map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.requests) == 0 {
		return nil
	}
	return s.requests[0]
}

// all is every request the endpoint saw.
func (s *session) all() []map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]map[string]any(nil), s.requests...)
}

// await waits for the endpoint to have seen n requests.
func (s *session) await(t *testing.T, n int) []map[string]any {
	t.Helper()
	deadline := time.After(3 * time.Second)
	for {
		if got := s.all(); len(got) >= n {
			return got
		}
		select {
		case <-deadline:
			t.Fatalf("the endpoint saw %d requests, want %d", len(s.all()), n)
		case <-time.After(5 * time.Millisecond):
		}
	}
}

// ttsServer starts a fake Cartesia TTS endpoint. It answers each request the
// service sends with the scripted messages, stamping every one with the context
// the request named, the way Cartesia does.
func ttsServer(t *testing.T, reply []map[string]any) (endpoint string, got *session) {
	t.Helper()
	got = &session{}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got.mu.Lock()
		got.header = r.Header.Clone()
		got.mu.Unlock()
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		for {
			_, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			req := map[string]any{}
			if err := json.Unmarshal(data, &req); err != nil {
				t.Errorf("decoding the synthesis request: %v", err)
				return
			}
			got.record(req)
			// A cancel or a flush asks for nothing back.
			if req["cancel"] == true {
				continue
			}
			if req[fieldTranscript] == "" {
				continue
			}
			contextID, _ := req["context_id"].(string)
			for _, m := range reply {
				msg := map[string]any{"context_id": contextID}
				maps.Copy(msg, m)
				b, err := json.Marshal(msg)
				if err != nil {
					t.Errorf("encoding a reply message: %v", err)
					return
				}
				if c.Write(ctx, websocket.MessageText, b) != nil {
					return
				}
			}
		}
	}))
	t.Cleanup(srv.Close)

	return "ws" + strings.TrimPrefix(srv.URL, "http"), got
}

// chunk is a base64 audio message as Cartesia sends it.
func chunk(pcm []byte) map[string]any {
	return map[string]any{"type": "chunk", "data": base64.StdEncoding.EncodeToString(pcm)}
}

// ttsHost stands in for the tts.Base a provider appends its audio to.
type ttsHost struct {
	mu     sync.Mutex
	audio  []byte
	words  []uctx.WordTiming
	opts   tts.WordTimingOptions
	frames []string
	closed bool
}

func (h *ttsHost) AppendToAudioContext(_ string, f frames.Frame) {
	h.mu.Lock()
	defer h.mu.Unlock()
	switch fr := f.(type) {
	case *frames.TTSAudioRawFrame:
		h.audio = append(h.audio, fr.Audio...)
		h.frames = append(h.frames, "audio")
	case *frames.TTSStoppedFrame:
		h.frames = append(h.frames, "stopped")
	}
}

func (h *ttsHost) AppendWordToAudioContext(_, word string, offset float64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.words = append(h.words, uctx.WordTiming{Word: word, Offset: offset})
}

func (h *ttsHost) AddWordTimestamps(
	contextID string, words []uctx.WordTiming, opts tts.WordTimingOptions,
) {
	h.mu.Lock()
	h.opts = opts
	h.mu.Unlock()
	for _, w := range words {
		h.AppendWordToAudioContext(contextID, w.Word, w.Offset)
	}
}

func (h *ttsHost) RemoveAudioContext(string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.closed = true
}

func (h *ttsHost) AudioContextAvailable(string) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return !h.closed
}

// spokenWords is the timings reported so far.
func (h *ttsHost) spokenWords() []uctx.WordTiming {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]uctx.WordTiming(nil), h.words...)
}

// snapshot is what the host has been given so far.
func (h *ttsHost) snapshot() (pcm []byte, words []uctx.WordTiming, seen []string, closed bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]byte(nil), h.audio...), append([]uctx.WordTiming(nil), h.words...),
		append([]string(nil), h.frames...), h.closed
}

// waitForClose blocks until the provider has closed the context, or fails.
func (h *ttsHost) waitForClose(t *testing.T) {
	t.Helper()
	deadline := time.After(3 * time.Second)
	for {
		if _, _, _, closed := h.snapshot(); closed {
			return
		}
		select {
		case <-deadline:
			t.Fatal("the provider never closed the audio context")
		case <-time.After(5 * time.Millisecond):
		}
	}
}

// speak runs one synthesis on its own context and waits for it to finish.
func speak(t *testing.T, s *synthesizer, text string) *ttsHost {
	t.Helper()
	host := &ttsHost{}
	s.SetAudioContextHost(host)
	t.Cleanup(func() { _ = s.Close() })
	if err := s.RunTTS(t.Context(), text, "c1", nil); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	host.waitForClose(t)
	return host
}

// testVoiceID is the voice these tests synthesize with. Cartesia has no default
// one, so every configuration names it.
const testVoiceID = "694f9389-aac1-45b6-b726-9d9369183238"

// ttsConfig is a config pointed at endpoint with every default filled in the way
// NewTTS fills them, so the request under test is the real one.
func ttsConfig(endpoint string) Config {
	return Config{
		APIKey:     "test-key",
		URL:        endpoint,
		Version:    defaultVersion,
		Model:      defaultModel,
		VoiceID:    testVoiceID,
		SampleRate: defaultSampleRate,
		Encoding:   defaultEncoding,
		Container:  defaultContainer,
	}
}

// TestConfigValidate pins which fields the TTS and STT configs require.
func TestConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing TTS API key", Cfg: Config{VoiceID: testVoiceID}, Valid: false},
		{Name: "missing TTS voice", Cfg: Config{APIKey: "k"}, Valid: false},
		{Name: "TTS key and voice", Cfg: Config{APIKey: "k", VoiceID: testVoiceID}, Valid: true},
		{Name: "missing STT API key", Cfg: STTConfig{}, Valid: false},
		{Name: "STT API key only", Cfg: STTConfig{APIKey: "k"}, Valid: true},
	})
}

// TestNewServices checks each constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNewServices(t *testing.T) {
	providertest.Service(t, "CartesiaTTS", NewTTS(Config{APIKey: "k", VoiceID: testVoiceID}))
	providertest.Service(t, "CartesiaTTS",
		NewTTS(Config{APIKey: "k", VoiceID: testVoiceID, WordTimestamps: true}))
	providertest.Service(t, "CartesiaSTT", NewSTT(STTConfig{APIKey: "k"}))
}

// TestSynthesizerMetadata checks the service reports the model and voice the
// synthesis is billed against.
func TestSynthesizerMetadata(t *testing.T) {
	s := &synthesizer{cfg: Config{Model: "sonic-3.5", VoiceID: "voice-1", SampleRate: 24000}}
	meta := s.Metadata()
	if meta.Model != "sonic-3.5" || meta.VoiceID != "voice-1" {
		t.Errorf("Metadata() = %+v, want the configured model and voice", meta)
	}
	if got := s.SampleRate(); got != 24000 {
		t.Errorf("SampleRate() = %d, want the configured rate", got)
	}
}

// TestRunTTSRequestShape checks the session is dialed with the Cartesia headers
// and that the synthesis request names the model, voice, context and output
// format.
func TestRunTTSRequestShape(t *testing.T) {
	want := []byte{0x11, 0x22, 0x33, 0x44}
	endpoint, seen := ttsServer(t, []map[string]any{chunk(want), {"type": "done"}})

	s := &synthesizer{cfg: ttsConfig(endpoint)}
	host := speak(t, s, "hello there")

	pcm, _, order, _ := host.snapshot()
	if string(pcm) != string(want) {
		t.Errorf("PCM = % x, want % x", pcm, want)
	}
	// The stop frame rides the queue behind the audio, so it is pushed only once
	// the last of that audio has been.
	if len(order) == 0 || order[len(order)-1] != "stopped" {
		t.Errorf("frames = %v, want the stop frame behind the audio", order)
	}

	got := seen.first()
	if got == nil {
		t.Fatal("the endpoint saw no synthesis request")
	}
	if h := seen.header.Get("X-API-Key"); h != "test-key" {
		t.Errorf("X-API-Key = %q, want the configured key", h)
	}
	if h := seen.header.Get("Cartesia-Version"); h != defaultVersion {
		t.Errorf("Cartesia-Version = %q, want the pinned %q", h, defaultVersion)
	}
	if got[fieldTranscript] != "hello there" {
		t.Errorf("transcript = %v, want the text to speak", got[fieldTranscript])
	}
	if got["model_id"] != defaultModel {
		t.Errorf("model_id = %v, want %q", got["model_id"], defaultModel)
	}
	if got["context_id"] != "c1" {
		t.Errorf("context_id = %v, want the turn's context", got["context_id"])
	}
	// More of the turn may follow, which is what lets its sentences stream as
	// one utterance. The flush at the end is what closes it.
	if got["continue"] != true {
		t.Errorf("continue = %v, want true", got["continue"])
	}

	voice, _ := got["voice"].(map[string]any)
	if voice["mode"] != "id" || voice["id"] != testVoiceID {
		t.Errorf("voice = %v, want the configured voice in id mode", voice)
	}
	format, _ := got["output_format"].(map[string]any)
	if format["container"] != defaultContainer || format["encoding"] != defaultEncoding {
		t.Errorf("output_format = %v, want the raw PCM container", format)
	}
	if format["sample_rate"] != float64(defaultSampleRate) {
		t.Errorf("output_format.sample_rate = %v, want %d", format["sample_rate"], defaultSampleRate)
	}

	// The plain path asks for no timestamps, so the base takes the unaligned
	// route and nothing downstream waits on word timing.
	for _, f := range []string{"add_timestamps", "use_normalized_timestamps"} {
		if _, present := got[f]; present {
			t.Errorf("%s was requested on the plain path: %v", f, got[f])
		}
	}
}

// TestRunTTSOptionalFields checks the language, generation config and
// pronunciation dictionary are sent only when set.
func TestRunTTSOptionalFields(t *testing.T) {
	endpoint, seen := ttsServer(t, []map[string]any{{"type": "done"}})

	s := &synthesizer{cfg: ttsConfig(endpoint)}
	speak(t, s, "hi")
	got := seen.first()
	for _, f := range []string{"language", "generation_config", "pronunciation_dict_id"} {
		if _, present := got[f]; present {
			t.Errorf("%s was sent for an unset config: %v", f, got[f])
		}
	}

	endpoint, seen = ttsServer(t, []map[string]any{{"type": "done"}})
	speed := 1.2
	cfg := ttsConfig(endpoint)
	cfg.Language = language.French
	cfg.GenerationConfig = &GenerationConfig{Speed: &speed, Emotion: "excited"}
	cfg.PronunciationDictID = "dict-1"
	speak(t, &synthesizer{cfg: cfg}, "hi")

	got = seen.first()
	if got["language"] != "fr" {
		t.Errorf("language = %v, want the base code fr", got["language"])
	}
	if got["pronunciation_dict_id"] != "dict-1" {
		t.Errorf("pronunciation_dict_id = %v", got["pronunciation_dict_id"])
	}
	gen, _ := got["generation_config"].(map[string]any)
	if gen["speed"] != 1.2 || gen["emotion"] != "excited" {
		t.Errorf("generation_config = %v, want the configured speed and emotion", gen)
	}
	if _, present := gen["volume"]; present {
		t.Error("an unset generation control was sent")
	}
}

// TestRunTTSJoinsChunks checks every audio chunk before "done" reaches the host,
// and that a message the service does not model is skipped rather than ending
// the synthesis.
func TestRunTTSJoinsChunks(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{
		chunk([]byte{1, 2}),
		{"type": "flush_done"},
		chunk([]byte{3, 4}),
		{"type": "done"},
	})

	s := &synthesizer{cfg: ttsConfig(endpoint)}
	host := speak(t, s, "hi")
	if pcm, _, _, _ := host.snapshot(); string(pcm) != string([]byte{1, 2, 3, 4}) {
		t.Errorf("PCM = % x, want the chunks up to done", pcm)
	}
}

// TestRunTTSSharesOneConnection checks the sentences of a turn go out on one
// connection and one context, which is what lets them stream as a single
// utterance rather than paying a handshake each.
func TestRunTTSSharesOneConnection(t *testing.T) {
	endpoint, seen := ttsServer(t, []map[string]any{chunk([]byte{1, 2})})

	s := &synthesizer{cfg: ttsConfig(endpoint)}
	host := &ttsHost{}
	s.SetAudioContextHost(host)
	t.Cleanup(func() { _ = s.Close() })

	for _, sentence := range []string{"one.", "two.", "three."} {
		if err := s.RunTTS(t.Context(), sentence, "c1", nil); err != nil {
			t.Fatalf("RunTTS: %v", err)
		}
	}
	reqs := seen.await(t, 3)
	for i, r := range reqs {
		if r["context_id"] != "c1" {
			t.Errorf("request %d went out on context %v, want the turn's own", i, r["context_id"])
		}
		if r["continue"] != true {
			t.Errorf("request %d closed the context, want it left open", i)
		}
	}
}

// TestFlushAudioClosesTheTurn checks the turn's context is closed once its text
// has all been sent, which is what makes Cartesia generate what it was holding.
func TestFlushAudioClosesTheTurn(t *testing.T) {
	endpoint, seen := ttsServer(t, []map[string]any{chunk([]byte{1, 2})})

	s := &synthesizer{cfg: ttsConfig(endpoint)}
	host := &ttsHost{}
	s.SetAudioContextHost(host)
	t.Cleanup(func() { _ = s.Close() })

	if err := s.RunTTS(t.Context(), "hi", "c1", nil); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	seen.await(t, 1)
	s.FlushAudio(t.Context(), "c1")

	reqs := seen.await(t, 2)
	flush := reqs[1]
	if flush["continue"] != false {
		t.Errorf("continue = %v on the flush, want false so the context closes", flush["continue"])
	}
	if flush[fieldTranscript] != "" {
		t.Errorf("transcript = %v on the flush, want none", flush[fieldTranscript])
	}
	if flush["context_id"] != "c1" {
		t.Errorf("context_id = %v on the flush, want the turn's", flush["context_id"])
	}
}

// TestInterruptionCancelsTheContext checks Cartesia is told to stop generating
// into a context nobody is listening to any more.
func TestInterruptionCancelsTheContext(t *testing.T) {
	endpoint, seen := ttsServer(t, []map[string]any{chunk([]byte{1, 2})})

	s := &synthesizer{cfg: ttsConfig(endpoint)}
	host := &ttsHost{}
	s.SetAudioContextHost(host)
	t.Cleanup(func() { _ = s.Close() })

	if err := s.RunTTS(t.Context(), "hi", "c1", nil); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	seen.await(t, 1)
	s.OnAudioContextInterrupted(t.Context(), "c1")

	reqs := seen.await(t, 2)
	cancel := reqs[1]
	if cancel["cancel"] != true || cancel["context_id"] != "c1" {
		t.Errorf("cancel message = %v, want it naming the interrupted context", cancel)
	}
}

// Audio for a context the host has closed is dropped rather than spoken over the
// next turn. An interruption abandons a turn mid-flight, and what the server had
// already generated for it arrives afterwards.
func TestAudioForAClosedContextIsDropped(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{chunk([]byte{1, 2}), {"type": "done"}})

	s := &synthesizer{cfg: ttsConfig(endpoint)}
	host := &ttsHost{}
	host.RemoveAudioContext("c1") // the context is gone before any audio lands
	s.SetAudioContextHost(host)
	t.Cleanup(func() { _ = s.Close() })

	if err := s.RunTTS(t.Context(), "hi", "c1", nil); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	time.Sleep(200 * time.Millisecond)
	if pcm, _, seen, _ := host.snapshot(); len(pcm) != 0 || len(seen) != 0 {
		t.Errorf("the host was given %d bytes and %v for a closed context", len(pcm), seen)
	}
}

// TestRunTTSServerError checks an error message from Cartesia closes the context
// out rather than leaving the turn waiting on audio that is not coming.
func TestRunTTSServerError(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{{"type": "error", "message": "voice not found"}})

	s := &synthesizer{cfg: ttsConfig(endpoint)}
	host := speak(t, s, "hi")
	if _, _, order, _ := host.snapshot(); len(order) == 0 || order[len(order)-1] != "stopped" {
		t.Errorf("frames = %v, want the context closed out", order)
	}
}

// TestRunTTSDialError checks a session that cannot be opened is reported rather
// than treated as an empty synthesis.
func TestRunTTSDialError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()

	s := &synthesizer{cfg: ttsConfig("ws" + strings.TrimPrefix(srv.URL, "http"))}
	s.SetAudioContextHost(&ttsHost{})
	if err := s.RunTTS(t.Context(), "hi", "c1", nil); err == nil {
		t.Fatal("RunTTS on a refused session = nil, want an error")
	}
}

// TestRunTTSTimedRequestsTimestamps checks the word-aligned path asks for
// timestamps against the text as written, since the base matches the tokens back
// to that text rather than to a normalized form, and that the timings reach the
// host with the audio.
func TestRunTTSTimedRequestsTimestamps(t *testing.T) {
	endpoint, seen := ttsServer(t, []map[string]any{
		chunk([]byte{1, 2}),
		{"type": "timestamps", "word_timestamps": map[string]any{
			"words": []string{"Hello", "world"},
			"start": []float64{0.0, 0.5},
		}},
		{"type": "done"},
	})

	cfg := ttsConfig(endpoint)
	cfg.WordTimestamps = true
	s := &timedSynthesizer{synthesizer: &synthesizer{cfg: cfg}}
	host := &ttsHost{}
	s.SetAudioContextHost(host)
	t.Cleanup(func() { _ = s.Close() })

	if err := s.RunTTSTimed(t.Context(), "Hello world", "c1", nil, nil); err != nil {
		t.Fatalf("RunTTSTimed: %v", err)
	}
	host.waitForClose(t)

	req := seen.first()
	if req["add_timestamps"] != true {
		t.Errorf("add_timestamps = %v, want it requested on the word-aligned path", req["add_timestamps"])
	}
	if req["use_normalized_timestamps"] != false {
		t.Errorf("use_normalized_timestamps = %v, want false so the tokens match the text",
			req["use_normalized_timestamps"])
	}

	words := host.spokenWords()
	want := []uctx.WordTiming{{Word: "Hello", Offset: 0}, {Word: "world", Offset: 0.5}}
	if len(words) != len(want) {
		t.Fatalf("words = %v, want %v", words, want)
	}
	for i := range want {
		if words[i] != want[i] {
			t.Fatalf("word %d = %v, want %v", i, words[i], want[i])
		}
	}
}

// TestRunTTSTimedIgnoresTimestampsOnThePlainPath checks a service that did not
// ask for timings does not report any, so the base stays off the word-aligned
// path.
func TestRunTTSTimedIgnoresTimestampsOnThePlainPath(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{
		chunk([]byte{1, 2}),
		{"type": "timestamps", "word_timestamps": map[string]any{
			"words": []string{"Hello"},
			"start": []float64{0.0},
		}},
		{"type": "done"},
	})

	s := &synthesizer{cfg: ttsConfig(endpoint)}
	host := speak(t, s, "Hello")
	if _, words, _, _ := host.snapshot(); len(words) != 0 {
		t.Errorf("the plain path reported %v, want no word timings", words)
	}
}

// TestSpacelessLanguage checks which languages are treated as written without
// spaces between words, which decides whether a reported message is joined into
// one token.
func TestSpacelessLanguage(t *testing.T) {
	spaceless := []language.Language{language.Chinese, language.ChineseTW, language.Japanese}
	for _, l := range spaceless {
		s := &synthesizer{cfg: Config{Language: l}}
		if !s.spacelessLanguage() {
			t.Errorf("spacelessLanguage(%q) = false, want true", l)
		}
	}
	spaced := []language.Language{language.English, language.Korean, language.French, language.Language("")}
	for _, l := range spaced {
		s := &synthesizer{cfg: Config{Language: l}}
		if s.spacelessLanguage() {
			t.Errorf("spacelessLanguage(%q) = true, want false", l)
		}
	}
}

// TestCartesiaLanguage pins the synthesis codes. Cartesia takes the base code,
// and a language this provider was not verified against is still sent under its
// own base code rather than dropped. Only the zero value sends nothing, which
// leaves Cartesia on its own default.
func TestCartesiaLanguage(t *testing.T) {
	cases := map[language.Language]string{
		language.English:      "en",
		language.EnglishGB:    "en",
		language.French:       "fr",
		language.FrenchCA:     "fr",
		language.Spanish:      "es",
		language.German:       "de",
		language.PortugueseBR: "pt",
		language.Japanese:     "ja",
		language.Korean:       "ko",
		language.Chinese:      "zh",
		// The five the verified map gained alongside the rest.
		language.Arabic:    "ar",
		language.Bulgarian: "bg",
		language.Bengali:   "bn",
		language.Odia:      "or",
		language.Urdu:      "ur",
		// Unverified, so its own base code goes out with a warning.
		language.Language("cy"): "cy",
		// Unset, so nothing is sent.
		language.Language(""): "",
	}
	for in, want := range cases {
		if got := cartesiaLanguage(in); got != want {
			t.Errorf("cartesiaLanguage(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestStripCartesiaTags checks the markup this provider accepts comes back off
// the tokens it reports, so they can be matched against the written text.
func TestStripCartesiaTags(t *testing.T) {
	cases := map[string]string{
		"<spell>Hello</spell>":     "Hello",
		"<emotion name=\"sad\">hi": "hi",
		"<break time=\"1s\"/>":     "",
		"plain":                    "plain",
		"a <volume>  b":            "a b",
		// A tag between two words is the only thing separating them, so it
		// becomes a space.
		"to<spell>1234</spell>": "to 1234",
		// A tag between a word and its own punctuation must not separate them,
		// or the token no longer matches the text sent for synthesis.
		"<spell>1234</spell>.":   "1234.",
		"<spell>A.B.C.</spell>,": "A.B.C.,",
	}
	for in, want := range cases {
		if got := stripCartesiaTags(in); got != want {
			t.Errorf("stripCartesiaTags(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestMaxBufferDelayDefault checks the server-side buffering window follows the
// aggregation the client does. Cartesia warns against stacking client-side
// sentence aggregation on top of its own 3000ms buffer, so aggregating
// sentences here disables the server's buffering, streaming tokens leaves it
// alone, and a value the caller chose is kept either way.
func TestMaxBufferDelayDefault(t *testing.T) {
	sentences := ttsDefaults(Config{APIKey: "k", VoiceID: testVoiceID})
	if sentences.MaxBufferDelayMs == nil || *sentences.MaxBufferDelayMs != 0 {
		t.Errorf("aggregating sentences: max_buffer_delay_ms = %v, want 0", sentences.MaxBufferDelayMs)
	}

	tokens := ttsDefaults(Config{APIKey: "k", VoiceID: testVoiceID, TextAggregation: frames.AggregationToken})
	if tokens.MaxBufferDelayMs != nil {
		t.Errorf("streaming tokens: max_buffer_delay_ms = %v, want it unset", *tokens.MaxBufferDelayMs)
	}

	chosen := 250
	kept := ttsDefaults(Config{APIKey: "k", VoiceID: testVoiceID, MaxBufferDelayMs: &chosen})
	if kept.MaxBufferDelayMs == nil || *kept.MaxBufferDelayMs != chosen {
		t.Errorf("max_buffer_delay_ms = %v, want the configured %d", kept.MaxBufferDelayMs, chosen)
	}
}

// TestRunTTSSendsMaxBufferDelay checks the buffering window reaches Cartesia
// when it is set, and is left off the request when it is not.
func TestRunTTSSendsMaxBufferDelay(t *testing.T) {
	endpoint, seen := ttsServer(t, []map[string]any{{"type": "done"}})
	cfg := ttsConfig(endpoint)
	zero := 0
	cfg.MaxBufferDelayMs = &zero
	speak(t, &synthesizer{cfg: cfg}, "hi")
	if got := seen.first(); got["max_buffer_delay_ms"] != float64(0) {
		t.Errorf("max_buffer_delay_ms = %v, want 0", got["max_buffer_delay_ms"])
	}

	endpoint, seen = ttsServer(t, []map[string]any{{"type": "done"}})
	speak(t, &synthesizer{cfg: ttsConfig(endpoint)}, "hi")
	if _, present := seen.first()["max_buffer_delay_ms"]; present {
		t.Error("max_buffer_delay_ms was sent with no window configured")
	}
}

// TestSpellTagReachesTheServiceWhole checks the service holds off on a sentence
// boundary inside a spell tag. The periods in one end no sentence, and splitting
// it would hand Cartesia half a tag.
func TestSpellTagReachesTheServiceWhole(t *testing.T) {
	endpoint, seen := ttsServer(t, []map[string]any{chunk([]byte{1, 2}), {"type": "done"}})
	base := NewTTS(ttsConfig(endpoint))

	task := pipeline.NewWorker(pipeline.New(base), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Dial <spell>A.B.C.</spell> now. Then wait."))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(5 * time.Second):
		t.Fatal("the task did not finish")
	}

	got := seen.first()
	if got == nil {
		t.Fatal("the endpoint saw no synthesis request")
	}
	transcript, _ := got[fieldTranscript].(string)
	if !strings.Contains(transcript, "<spell>A.B.C.</spell>") {
		t.Errorf("transcript = %q, want the spell tag whole inside it", transcript)
	}
}
