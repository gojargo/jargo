package soniox

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
	uctx "github.com/gojargo/jargo/utils/context"
)

// TestTTSConfigValidate pins which TTSConfig fields the provider requires and
// the ranges Soniox accepts.
func TestTTSConfigValidate(t *testing.T) {
	slow, tooSlow := 0.7, 0.6
	fast, tooFast := 1.3, 1.4
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: TTSConfig{}, Valid: false},
		{Name: "API key only", Cfg: TTSConfig{APIKey: "k"}, Valid: true},
		{Name: "supported sample rate", Cfg: TTSConfig{APIKey: "k", SampleRate: 44100}, Valid: true},
		{Name: "unsupported sample rate", Cfg: TTSConfig{APIKey: "k", SampleRate: 22050}, Valid: false},
		{Name: "slowest speed", Cfg: TTSConfig{APIKey: "k", Speed: &slow}, Valid: true},
		{Name: "below the slowest speed", Cfg: TTSConfig{APIKey: "k", Speed: &tooSlow}, Valid: false},
		{Name: "fastest speed", Cfg: TTSConfig{APIKey: "k", Speed: &fast}, Valid: true},
		{Name: "above the fastest speed", Cfg: TTSConfig{APIKey: "k", Speed: &tooFast}, Valid: false},
	})
}

// TestNewTTS checks the constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNewTTS(t *testing.T) {
	providertest.Service(t, "SonioxTTS", NewTTS(TTSConfig{APIKey: "k"}))
}

// TestTTSConfigMessage checks the stream configuration Soniox expects.
func TestTTSConfigMessage(t *testing.T) {
	t.Run("defaults", func(t *testing.T) {
		s := &ttsSynthesizer{cfg: TTSConfig{APIKey: "k"}.withTTSDefaults()}
		cfg := s.config(true)

		want := map[string]any{
			"api_key":           "k",
			keyStreamID:         ttsStreamID,
			"model":             defaultTTSModel,
			"voice":             defaultVoice,
			"audio_format":      ttsAudioFormat,
			"sample_rate":       defaultTTSSampleRate,
			"return_timestamps": true,
		}
		for key, val := range want {
			if cfg[key] != val {
				t.Errorf("%s = %v, want %v", key, cfg[key], val)
			}
		}
		for _, key := range []string{"language", "speed"} {
			if _, ok := cfg[key]; ok {
				t.Errorf("%s = %v, want it omitted when unset", key, cfg[key])
			}
		}
	})

	t.Run("optional settings", func(t *testing.T) {
		speed := 1.1
		s := &ttsSynthesizer{cfg: TTSConfig{
			APIKey:   "k",
			Voice:    "9f2b0c1e",
			Language: language.FrenchCA,
			Speed:    &speed,
		}.withTTSDefaults()}
		cfg := s.config(false)

		if cfg["language"] != "fr" {
			t.Errorf("language = %v, want the base code fr", cfg["language"])
		}
		if cfg["speed"] != speed {
			t.Errorf("speed = %v, want %v", cfg["speed"], speed)
		}
		if cfg["voice"] != "9f2b0c1e" {
			t.Errorf("voice = %v, want the cloned-voice id", cfg["voice"])
		}
		if cfg["return_timestamps"] != false {
			t.Errorf("return_timestamps = %v, want false", cfg["return_timestamps"])
		}
	})
}

// TestSpacelessLanguage checks which languages get per-character word timings.
func TestSpacelessLanguage(t *testing.T) {
	cases := map[language.Language]bool{
		language.Chinese:   true,
		language.ChineseTW: true,
		language.Japanese:  true,
		language.English:   false,
		language.Korean:    false, // Korean does use spaces between words
		"":                 false,
	}
	for lang, want := range cases {
		s := &ttsSynthesizer{cfg: TTSConfig{Language: lang}}
		if got := s.spacelessLanguage(); got != want {
			t.Errorf("spacelessLanguage(%q) = %v, want %v", lang, got, want)
		}
	}
}

// ttsServer starts a fake Soniox endpoint that records the client's messages
// and replies with a scripted sequence.
func ttsServer(t *testing.T, events []map[string]any) (endpoint string, sent <-chan map[string]any) {
	t.Helper()
	messages := make(chan map[string]any, 8)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		// The client sends the config, the text, and the end marker before any
		// audio comes back.
		for range 3 {
			_, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			var msg map[string]any
			if json.Unmarshal(data, &msg) == nil {
				messages <- msg
			}
		}
		for _, ev := range events {
			b, _ := json.Marshal(ev)
			if c.Write(ctx, websocket.MessageText, b) != nil {
				return
			}
		}
		// Keep reading so the library can answer the close handshake.
		for {
			if _, _, err := c.Read(ctx); err != nil {
				return
			}
		}
	}))
	t.Cleanup(srv.Close)

	return "ws" + strings.TrimPrefix(srv.URL, "http"), messages
}

// audioEvent builds a synthesis message carrying pcm and optional timings.
func audioEvent(pcm []byte, chars []string, starts []float64) map[string]any {
	ev := map[string]any{}
	if len(pcm) > 0 {
		ev["audio"] = base64.StdEncoding.EncodeToString(pcm)
	}
	if chars != nil {
		ev["timestamps"] = map[string]any{
			"characters":                    chars,
			"character_start_times_seconds": starts,
		}
	}
	return ev
}

// TestTTSSynthesize checks the messages the service sends and the audio it
// emits.
func TestTTSSynthesize(t *testing.T) {
	endpoint, sent := ttsServer(t, []map[string]any{
		audioEvent([]byte{1, 2, 3, 4}, nil, nil),
		audioEvent([]byte{5, 6}, nil, nil),
		{"terminated": true},
	})

	s := &ttsSynthesizer{cfg: TTSConfig{APIKey: "test-key", URL: endpoint}.withTTSDefaults()}
	var got []byte
	if err := runPCM(s, context.Background(), "hello there", func(pcm []byte) error {
		got = append(got, pcm...)
		return nil
	}); err != nil {
		t.Fatalf("Synthesize: %v", err)
	}

	if string(got) != "\x01\x02\x03\x04\x05\x06" {
		t.Errorf("emitted %q, want the concatenated PCM chunks", got)
	}

	cfg := <-sent
	if cfg["api_key"] != "test-key" {
		t.Errorf("config api_key = %v, want the key", cfg["api_key"])
	}
	if cfg["stream_id"] != ttsStreamID {
		t.Errorf("config stream_id = %v, want %q", cfg["stream_id"], ttsStreamID)
	}

	text := <-sent
	if text["text"] != "hello there" || text["text_end"] != false {
		t.Errorf("text message = %v, want the sentence with text_end false", text)
	}
	end := <-sent
	if end["text"] != "" || end["text_end"] != true {
		t.Errorf("end message = %v, want an empty text with text_end true", end)
	}
}

// TestTTSSynthesizeTimed checks per-character timings are assembled into words,
// including a word split across two payloads and a trailing word closed only by
// the stream terminating.
func TestTTSSynthesizeTimed(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{
		// "hi the" arrives first, leaving "the" carried over mid-word.
		audioEvent([]byte{1, 2},
			[]string{"h", "i", " ", "t", "h", "e"},
			[]float64{0.0, 0.1, 0.2, 0.3, 0.4, 0.5}),
		// "re, pal" completes "there," then leaves "pal" unterminated.
		audioEvent(nil,
			[]string{"r", "e", ",", " ", "p", "a", "l"},
			[]float64{0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2}),
		{"terminated": true},
	})

	s := &timedTTSSynthesizer{ttsSynthesizer: &ttsSynthesizer{
		cfg: TTSConfig{APIKey: "k", URL: endpoint}.withTTSDefaults(),
	}}

	type timed struct {
		text   string
		offset float64
	}
	var words []timed
	if err := runPCMTimed(s, context.Background(), "hi there, pal",
		func([]byte) error { return nil },
		func(text string, offset float64) error {
			words = append(words, timed{text, offset})
			return nil
		}); err != nil {
		t.Fatalf("SynthesizeTimed: %v", err)
	}

	want := []timed{{"hi", 0.0}, {"there,", 0.3}, {"pal", 1.0}}
	if len(words) != len(want) {
		t.Fatalf("words = %+v, want %+v", words, want)
	}
	for i, w := range want {
		if words[i] != w {
			t.Errorf("word %d = %+v, want %+v", i, words[i], w)
		}
	}
}

// TestTTSSynthesizeTimedSpaceless checks a language written without spaces
// reports each character as its own word, since there is no boundary to split
// on. Punctuation carries no timing of its own and is dropped.
func TestTTSSynthesizeTimedSpaceless(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{
		audioEvent(nil, []string{"你", "好", "。"}, []float64{0.0, 0.2, 0.4}),
		{"terminated": true},
	})

	s := &timedTTSSynthesizer{ttsSynthesizer: &ttsSynthesizer{
		cfg: TTSConfig{APIKey: "k", URL: endpoint, Language: language.ChineseCN}.withTTSDefaults(),
	}}

	var words []string
	if err := runPCMTimed(s, context.Background(), "你好。",
		func([]byte) error { return nil },
		func(text string, _ float64) error {
			words = append(words, text)
			return nil
		}); err != nil {
		t.Fatalf("SynthesizeTimed: %v", err)
	}

	if len(words) != 2 || words[0] != "你" || words[1] != "好" {
		t.Errorf("words = %q, want one per character with punctuation dropped", words)
	}
}

// TestTTSServerError surfaces a stream failure to the caller.
func TestTTSServerError(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{
		{"error_code": 400, "error_type": "invalid_request", "error_message": "unknown voice"},
	})

	s := &ttsSynthesizer{cfg: TTSConfig{APIKey: "k", URL: endpoint}.withTTSDefaults()}
	err := runPCM(s, context.Background(), "hello", func([]byte) error { return nil })
	if err == nil {
		t.Fatal("Synthesize() = nil error, want the server error surfaced")
	}
	if !strings.Contains(err.Error(), "unknown voice") || !strings.Contains(err.Error(), "invalid_request") {
		t.Errorf("error = %v, want it to carry the server type and message", err)
	}
}

// TestTTSTimingLengthMismatch checks a malformed timing payload fails loudly
// rather than silently misaligning every word after it.
func TestTTSTimingLengthMismatch(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{
		audioEvent(nil, []string{"h", "i"}, []float64{0.0}),
	})

	s := &timedTTSSynthesizer{ttsSynthesizer: &ttsSynthesizer{
		cfg: TTSConfig{APIKey: "k", URL: endpoint}.withTTSDefaults(),
	}}
	err := runPCMTimed(s, context.Background(), "hi",
		func([]byte) error { return nil },
		func(string, float64) error { return nil })
	if err == nil {
		t.Fatal("SynthesizeTimed() = nil error, want the length mismatch reported")
	}
	if !strings.Contains(err.Error(), "timing length mismatch") {
		t.Errorf("error = %v, want it to name the mismatch", err)
	}
}

// TestTTSSynthesizerDescription checks the rate it emits and the model and
// voice the synthesis is billed against.
func TestTTSSynthesizerDescription(t *testing.T) {
	s := &ttsSynthesizer{cfg: TTSConfig{APIKey: "k", SampleRate: 16000}.withTTSDefaults()}
	if got := s.SampleRate(); got != 16000 {
		t.Errorf("SampleRate() = %d, want 16000", got)
	}
	meta := s.Metadata()
	if meta.Model != defaultTTSModel || meta.VoiceID != defaultVoice {
		t.Errorf("Metadata() = %+v, want the default model and voice", meta)
	}
}

// TestNewTTSWordTimestampPath checks the word-aligned path is on by default and
// can be turned off.
func TestNewTTSWordTimestampPath(t *testing.T) {
	off := false
	if !(TTSConfig{APIKey: "k"}).wordTimestamps() {
		t.Error("wordTimestamps() = false, want it on by default")
	}
	if (TTSConfig{APIKey: "k", WordTimestamps: &off}).wordTimestamps() {
		t.Error("wordTimestamps() = true, want it off when explicitly disabled")
	}
}

// runPCM drives a synthesizer the way the base does, handing back the raw audio
// it yields.
func runPCM(s tts.Synthesizer, ctx context.Context, text string, emit func(pcm []byte) error) error {
	return s.RunTTS(ctx, text, "", func(f frames.Frame) error {
		if af, ok := f.(*frames.TTSAudioRawFrame); ok {
			return emit(af.Audio)
		}
		return nil
	})
}

// runPCMTimed drives a word-timestamp synthesizer the way the base does, handing
// back the raw audio it yields.
func runPCMTimed(s tts.WordTimestamps, ctx context.Context, text string,
	emit func(pcm []byte) error, word func(text string, offset float64) error,
) error {
	return s.RunTTSTimed(ctx, text, "", func(f frames.Frame) error {
		if af, ok := f.(*frames.TTSAudioRawFrame); ok {
			return emit(af.Audio)
		}
		return nil
	}, func(words []uctx.WordTiming, _ tts.WordTimingOptions) error {
		for _, w := range words {
			if err := word(w.Word, w.Offset); err != nil {
				return err
			}
		}
		return nil
	})
}

// A language written without spaces reports its timings per character, so each
// token already reads as continuous text. The tokens have to say so, or every
// consumer joining them puts a space between characters that belong to one word.
func TestTTSTimedSpacelessTokensCarryTheirOwnSpacing(t *testing.T) {
	for _, tc := range []struct {
		name string
		lang language.Language
		want bool
	}{
		{"a language written without spaces", language.ChineseCN, true},
		{"a language written with them", language.English, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			endpoint, _ := ttsServer(t, []map[string]any{
				audioEvent(nil, []string{"你", "好"}, []float64{0.0, 0.2}),
				{"terminated": true},
			})
			s := &timedTTSSynthesizer{ttsSynthesizer: &ttsSynthesizer{
				cfg: TTSConfig{APIKey: "k", URL: endpoint, Language: tc.lang}.withTTSDefaults(),
			}}

			var got []bool
			err := s.RunTTSTimed(context.Background(), "你好", "",
				func(frames.Frame) error { return nil },
				func(words []uctx.WordTiming, opts tts.WordTimingOptions) error {
					for range words {
						got = append(got, opts.IncludesInterFrameSpaces)
					}
					return nil
				})
			if err != nil {
				t.Fatalf("RunTTSTimed: %v", err)
			}
			if len(got) == 0 {
				t.Fatal("no word timings were reported")
			}
			for i, carries := range got {
				if carries != tc.want {
					t.Fatalf("token %d says it carries its own spacing = %v, want %v", i, carries, tc.want)
				}
			}
		})
	}
}
