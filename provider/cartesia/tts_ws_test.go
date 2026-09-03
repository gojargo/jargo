package cartesia

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
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
// with and the request the service sent once the socket was open.
type session struct {
	header  http.Header
	request map[string]any
}

// ttsServer starts a fake Cartesia TTS endpoint that reads one request and
// replies with the scripted messages. It reports the session back once the
// request has arrived.
func ttsServer(t *testing.T, reply []map[string]any) (endpoint string, seen func() *session) {
	t.Helper()
	sessions := make(chan *session, 1)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got := &session{header: r.Header.Clone()}
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		_, data, err := c.Read(ctx)
		if err != nil {
			return
		}
		got.request = map[string]any{}
		if err := json.Unmarshal(data, &got.request); err != nil {
			t.Errorf("decoding the synthesis request: %v", err)
			return
		}
		select {
		case sessions <- got:
		default:
		}

		for _, m := range reply {
			b, err := json.Marshal(m)
			if err != nil {
				t.Errorf("encoding a reply message: %v", err)
				return
			}
			if c.Write(ctx, websocket.MessageText, b) != nil {
				return
			}
		}
		for {
			if _, _, err := c.Read(ctx); err != nil {
				return
			}
		}
	}))
	t.Cleanup(srv.Close)

	return "ws" + strings.TrimPrefix(srv.URL, "http"), func() *session {
		select {
		case s := <-sessions:
			return s
		default:
			return nil
		}
	}
}

// chunk is a base64 audio message as Cartesia sends it.
func chunk(pcm []byte) map[string]any {
	return map[string]any{"type": "chunk", "data": base64.StdEncoding.EncodeToString(pcm)}
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

// collect runs one synthesis and returns the PCM the frames carried.
func collect(t *testing.T, s *synthesizer, text string) []byte {
	t.Helper()
	var pcm []byte
	err := s.RunTTS(t.Context(), text, "", func(f frames.Frame) error {
		audio, ok := f.(*frames.TTSAudioRawFrame)
		if !ok {
			t.Errorf("yielded %T, want a TTSAudioRawFrame", f)
			return nil
		}
		if audio.SampleRate != s.SampleRate() {
			t.Errorf("frame rate = %d, want the configured %d", audio.SampleRate, s.SampleRate())
		}
		pcm = append(pcm, audio.Audio...)
		return nil
	})
	if err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	return pcm
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
// and that the synthesis request names the model, voice and output format.
func TestRunTTSRequestShape(t *testing.T) {
	want := []byte{0x11, 0x22, 0x33, 0x44}
	endpoint, seen := ttsServer(t, []map[string]any{chunk(want), {"type": "done"}})

	s := &synthesizer{cfg: ttsConfig(endpoint)}
	if got := collect(t, s, "hello there"); string(got) != string(want) {
		t.Errorf("PCM = % x, want % x", got, want)
	}

	got := seen()
	if got == nil {
		t.Fatal("the endpoint saw no synthesis request")
	}
	if h := got.header.Get("X-API-Key"); h != "test-key" {
		t.Errorf("X-API-Key = %q, want the configured key", h)
	}
	if h := got.header.Get("Cartesia-Version"); h != defaultVersion {
		t.Errorf("Cartesia-Version = %q, want the pinned %q", h, defaultVersion)
	}
	if got.request["transcript"] != "hello there" {
		t.Errorf("transcript = %v, want the text to speak", got.request["transcript"])
	}
	if got.request["model_id"] != defaultModel {
		t.Errorf("model_id = %v, want %q", got.request["model_id"], defaultModel)
	}
	if got.request["continue"] != false {
		t.Errorf("continue = %v, want false", got.request["continue"])
	}

	voice, _ := got.request["voice"].(map[string]any)
	if voice["mode"] != "id" || voice["id"] != testVoiceID {
		t.Errorf("voice = %v, want the configured voice in id mode", voice)
	}
	format, _ := got.request["output_format"].(map[string]any)
	if format["container"] != defaultContainer || format["encoding"] != defaultEncoding {
		t.Errorf("output_format = %v, want the raw PCM container", format)
	}
	if format["sample_rate"] != float64(defaultSampleRate) {
		t.Errorf("output_format.sample_rate = %v, want %d", format["sample_rate"], defaultSampleRate)
	}

	// The plain path asks for no timestamps, so the base takes the unaligned
	// route and nothing downstream waits on word timing.
	for _, f := range []string{"add_timestamps", "use_normalized_timestamps"} {
		if _, present := got.request[f]; present {
			t.Errorf("%s was requested on the plain path: %v", f, got.request[f])
		}
	}
}

// TestRunTTSOptionalFields checks the language, generation config and
// pronunciation dictionary are sent only when set.
func TestRunTTSOptionalFields(t *testing.T) {
	endpoint, seen := ttsServer(t, []map[string]any{{"type": "done"}})

	s := &synthesizer{cfg: ttsConfig(endpoint)}
	collect(t, s, "hi")
	got := seen()
	for _, f := range []string{"language", "generation_config", "pronunciation_dict_id"} {
		if _, present := got.request[f]; present {
			t.Errorf("%s was sent for an unset config: %v", f, got.request[f])
		}
	}

	endpoint, seen = ttsServer(t, []map[string]any{{"type": "done"}})
	speed := 1.2
	cfg := ttsConfig(endpoint)
	cfg.Language = language.French
	cfg.GenerationConfig = &GenerationConfig{Speed: &speed, Emotion: "excited"}
	cfg.PronunciationDictID = "dict-1"
	collect(t, &synthesizer{cfg: cfg}, "hi")

	got = seen()
	if got.request["language"] != "fr" {
		t.Errorf("language = %v, want the base code fr", got.request["language"])
	}
	if got.request["pronunciation_dict_id"] != "dict-1" {
		t.Errorf("pronunciation_dict_id = %v", got.request["pronunciation_dict_id"])
	}
	gen, _ := got.request["generation_config"].(map[string]any)
	if gen["speed"] != 1.2 || gen["emotion"] != "excited" {
		t.Errorf("generation_config = %v, want the configured speed and emotion", gen)
	}
	if _, present := gen["volume"]; present {
		t.Error("an unset generation control was sent")
	}
}

// TestRunTTSJoinsChunks checks every audio chunk before "done" reaches the
// caller, and that a message the service does not model is skipped rather than
// ending the synthesis.
func TestRunTTSJoinsChunks(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{
		chunk([]byte{1, 2}),
		{"type": "flush_done"},
		chunk([]byte{3, 4}),
		{"type": "done"},
		chunk([]byte{5, 6}),
	})

	s := &synthesizer{cfg: ttsConfig(endpoint)}
	if got := collect(t, s, "hi"); string(got) != string([]byte{1, 2, 3, 4}) {
		t.Errorf("PCM = % x, want the chunks up to done", got)
	}
}

// TestRunTTSServerError checks an error message from Cartesia is reported rather
// than leaving the synthesis waiting for audio that will not come.
func TestRunTTSServerError(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{{"type": "error", "message": "voice not found"}})

	s := &synthesizer{cfg: ttsConfig(endpoint)}
	err := s.RunTTS(t.Context(), "hi", "", func(frames.Frame) error { return nil })
	if !errors.Is(err, errProtocol) {
		t.Fatalf("RunTTS error = %v, want errProtocol", err)
	}
	if !strings.Contains(err.Error(), "voice not found") {
		t.Errorf("error = %v, want it to carry the server's message", err)
	}
}

// TestRunTTSDialError checks a session that cannot be opened is reported rather
// than treated as an empty synthesis.
func TestRunTTSDialError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()

	cfg := ttsConfig("ws" + strings.TrimPrefix(srv.URL, "http"))
	s := &synthesizer{cfg: cfg}
	err := s.RunTTS(t.Context(), "hi", "", func(frames.Frame) error {
		t.Error("a frame was yielded for a session that never opened")
		return nil
	})
	if err == nil {
		t.Fatal("RunTTS on a refused session = nil, want an error")
	}
}

// TestRunTTSTimedRequestsTimestamps checks the word-aligned path asks for
// timestamps against the text as written, since the base matches the tokens back
// to that text rather than to a normalized form.
func TestRunTTSTimedRequestsTimestamps(t *testing.T) {
	endpoint, seen := ttsServer(t, []map[string]any{
		chunk([]byte{1, 2}),
		{"type": "timestamps", "word_timestamps": map[string]any{
			"words": []string{"Hello", "world"},
			"start": []float64{0.0, 0.5},
		}},
		{"type": "done"},
	})

	s := &timedSynthesizer{synthesizer: &synthesizer{cfg: ttsConfig(endpoint)}}
	var got []uctx.WordTiming
	var opts tts.WordTimingOptions
	err := s.RunTTSTimed(t.Context(), "Hello world", "",
		func(frames.Frame) error { return nil },
		func(words []uctx.WordTiming, o tts.WordTimingOptions) error {
			got = append(got, words...)
			opts = o
			return nil
		})
	if err != nil {
		t.Fatalf("RunTTSTimed: %v", err)
	}

	req := seen()
	if req.request["add_timestamps"] != true {
		t.Errorf("add_timestamps = %v, want true on the word-aligned path", req.request["add_timestamps"])
	}
	if req.request["use_normalized_timestamps"] != false {
		t.Errorf("use_normalized_timestamps = %v, want false so tokens match the written text",
			req.request["use_normalized_timestamps"])
	}
	if len(got) != 2 || got[0].Word != "Hello" || got[1].Word != "world" || got[1].Offset != 0.5 {
		t.Errorf("timings = %+v, want the reported words at their offsets", got)
	}
	if opts.IncludesInterFrameSpaces {
		t.Error("a language written with spaces was reported as carrying its own")
	}
}

// TestRunTTSTimedIgnoresTimestampsOnThePlainPath checks a timestamps message is
// dropped when the caller did not ask for word timing, rather than failing the
// synthesis.
func TestRunTTSTimedIgnoresTimestampsOnThePlainPath(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{
		{"type": "timestamps", "word_timestamps": map[string]any{
			"words": []string{"Hello"}, "start": []float64{0.0},
		}},
		chunk([]byte{7}),
		{"type": "done"},
	})

	s := &synthesizer{cfg: ttsConfig(endpoint)}
	if got := collect(t, s, "hi"); string(got) != string([]byte{7}) {
		t.Errorf("PCM = % x, want the one chunk", got)
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
	collect(t, &synthesizer{cfg: cfg}, "hi")
	if got := seen(); got.request["max_buffer_delay_ms"] != float64(0) {
		t.Errorf("max_buffer_delay_ms = %v, want 0", got.request["max_buffer_delay_ms"])
	}

	endpoint, seen = ttsServer(t, []map[string]any{{"type": "done"}})
	collect(t, &synthesizer{cfg: ttsConfig(endpoint)}, "hi")
	if _, present := seen().request["max_buffer_delay_ms"]; present {
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

	got := seen()
	if got == nil {
		t.Fatal("the endpoint saw no synthesis request")
	}
	transcript, _ := got.request["transcript"].(string)
	if !strings.Contains(transcript, "<spell>A.B.C.</spell>") {
		t.Errorf("transcript = %q, want the spell tag whole inside it", transcript)
	}
}
