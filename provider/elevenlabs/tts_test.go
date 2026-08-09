package elevenlabs

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
	uctx "github.com/gojargo/jargo/utils/context"
)

// spoken is one synthesis request as the endpoint received it.
type spoken struct {
	path   string
	query  url.Values
	header http.Header
	body   map[string]any
}

// newTTSServer stands in for the with-timestamps synthesis endpoint. It replies
// with lines verbatim and records every request, since one turn makes several.
func newTTSServer(t *testing.T, lines []string) (*httptest.Server, *[]spoken) {
	t.Helper()
	var got []spoken
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		one := spoken{path: r.URL.Path, query: r.URL.Query(), header: r.Header.Clone(), body: map[string]any{}}
		if err := json.NewDecoder(r.Body).Decode(&one.body); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		got = append(got, one)
		for _, l := range lines {
			_, _ = w.Write([]byte(l + "\n"))
		}
	}))
	t.Cleanup(srv.Close)
	return srv, &got
}

// charAlignmentOf builds an alignment for text, one character per 100ms.
func charAlignmentOf(text string) *httpAlignment {
	a := &httpAlignment{}
	for i, r := range []rune(text) {
		a.Characters = append(a.Characters, string(r))
		a.StartTimesSecs = append(a.StartTimesSecs, float64(i)/10)
		a.EndTimesSeconds = append(a.EndTimesSeconds, float64(i+1)/10)
	}
	return a
}

// alignmentMap is the wire form of an alignment for text.
func alignmentMap(text string) map[string]any {
	a := charAlignmentOf(text)
	return map[string]any{
		"characters":                    a.Characters,
		"character_start_times_seconds": a.StartTimesSecs,
		"character_end_times_seconds":   a.EndTimesSeconds,
	}
}

// alignmentJSON renders text and its audio as one with-timestamps line.
func alignmentJSON(t *testing.T, text string, pcm []byte) string {
	t.Helper()
	b, err := json.Marshal(map[string]any{
		"audio_base64": base64.StdEncoding.EncodeToString(pcm),
		"alignment":    alignmentMap(text),
	})
	if err != nil {
		t.Fatalf("encoding a stream line: %v", err)
	}
	return string(b)
}

// ttsCfg is a config pointed at srv with the defaults NewTTS applies.
func ttsCfg(srv *httptest.Server) Config {
	return Config{
		APIKey:     "test-key",
		BaseURL:    srv.URL,
		VoiceID:    defaultVoiceID,
		Model:      defaultModel,
		SampleRate: defaultSampleRate,
	}
}

// timed is what one synthesis produced: the audio and the word timing.
type timed struct {
	pcm   []byte
	words []uctx.WordTiming
	opts  tts.WordTimingOptions
}

// speak runs one synthesis through s and collects both halves.
func speak(t *testing.T, s *synthesizer, text string) timed {
	t.Helper()
	var out timed
	err := s.RunTTSTimed(t.Context(), text, "",
		func(f frames.Frame) error {
			audio, ok := f.(*frames.TTSAudioRawFrame)
			if !ok {
				t.Errorf("yielded %T, want a TTSAudioRawFrame", f)
				return nil
			}
			out.pcm = append(out.pcm, audio.Audio...)
			return nil
		},
		func(words []uctx.WordTiming, opts tts.WordTimingOptions) error {
			out.words = append(out.words, words...)
			out.opts = opts
			return nil
		})
	if err != nil {
		t.Fatalf("RunTTSTimed: %v", err)
	}
	return out
}

// spokenWords is just the words, for comparing against upstream's expectations.
func spokenWords(got timed) []string {
	out := make([]string, 0, len(got.words))
	for _, w := range got.words {
		out = append(out, w.Word)
	}
	return out
}

// equalWords compares two word lists.
func equalWords(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

// TestSynthesizerImplementsWordTimestamps checks the HTTP service always reports
// word timing, which is what makes the base emit playback-aligned text and
// record only what was actually spoken before an interruption.
func TestSynthesizerImplementsWordTimestamps(t *testing.T) {
	var _ tts.WordTimestamps = (*synthesizer)(nil)
}

// TestSynthesizerMetadata checks the service reports the model and voice the
// synthesis is billed against.
func TestSynthesizerMetadata(t *testing.T) {
	s := &synthesizer{cfg: Config{Model: "eleven_flash_v2_5", VoiceID: "voice-1", SampleRate: 24000}}
	meta := s.Metadata()
	if meta.Model != "eleven_flash_v2_5" || meta.VoiceID != "voice-1" {
		t.Errorf("Metadata() = %+v, want the configured model and voice", meta)
	}
	if got := s.SampleRate(); got != 24000 {
		t.Errorf("SampleRate() = %d, want the configured rate", got)
	}
}

// TestRunTTSRequestShape checks the synthesis goes to the with-timestamps
// endpoint, authorized with the ElevenLabs key header, and that both the audio
// and the word timing come back.
func TestRunTTSRequestShape(t *testing.T) {
	pcm := bytes.Repeat([]byte{0x11, 0x22}, 8)
	srv, reqs := newTTSServer(t, []string{alignmentJSON(t, "Hello world", pcm)})

	cfg := ttsCfg(srv)
	cfg.VoiceID = "voice-1"
	got := speak(t, &synthesizer{cfg: cfg, http: &http.Client{}}, "Hello world")

	one := (*reqs)[0]
	if one.path != "/v1/text-to-speech/voice-1/stream/with-timestamps" {
		t.Errorf("path = %q, want the with-timestamps endpoint", one.path)
	}
	if h := one.header.Get("xi-api-key"); h != "test-key" {
		t.Errorf("xi-api-key = %q, want the configured key", h)
	}
	if h := one.header.Get("Content-Type"); h != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", h)
	}
	if q := one.query.Get("output_format"); q != "pcm_48000" {
		t.Errorf("output_format = %q, want pcm_48000", q)
	}
	if one.body["text"] != "Hello world" {
		t.Errorf("text = %v, want the text to speak", one.body["text"])
	}
	if one.body["model_id"] != defaultModel {
		t.Errorf("model_id = %v, want %q", one.body["model_id"], defaultModel)
	}
	if !bytes.Equal(got.pcm, pcm) {
		t.Errorf("PCM = % x, want the decoded audio", got.pcm)
	}
	if want := []string{"Hello", "world"}; !equalWords(spokenWords(got), want) {
		t.Fatalf("words = %v, want %v", spokenWords(got), want)
	}
	// Each word is timed from its own first character: "Hello" at 0.0s, and
	// "world" at the seventh character, 0.6s in.
	if got.words[0].Offset != 0.0 || got.words[1].Offset != 0.6 {
		t.Errorf("offsets = %v / %v, want the first character of each word",
			got.words[0].Offset, got.words[1].Offset)
	}
}

// TestPreviousTextCarriesAcrossTheTurn is upstream's
// test_http_payload_includes_previous_text_when_supported: what has already been
// spoken this turn is sent as context so the prosody carries across the sentence
// boundary.
func TestPreviousTextCarriesAcrossTheTurn(t *testing.T) {
	srv, reqs := newTTSServer(t, []string{alignmentJSON(t, "Hello!", nil)})
	s := &synthesizer{cfg: ttsCfg(srv), http: &http.Client{}}

	speak(t, s, "Hello!")
	if _, present := (*reqs)[0].body["previous_text"]; present {
		t.Errorf("previous_text = %v, want nothing on the turn's first sentence",
			(*reqs)[0].body["previous_text"])
	}

	speak(t, s, "How can I assist you today?")
	if got := (*reqs)[1].body["previous_text"]; got != "Hello!" {
		t.Errorf("previous_text = %v, want the first sentence", got)
	}

	speak(t, s, "Anything at all.")
	if got := (*reqs)[2].body["previous_text"]; got != "Hello! How can I assist you today?" {
		t.Errorf("previous_text = %v, want everything spoken so far, space-joined", got)
	}
}

// TestPreviousTextOmittedForUnsupportedModel is upstream's
// test_http_payload_omits_previous_text_for_eleven_v3: the model rejects the
// context parameters, so it is left off however much has been spoken.
func TestPreviousTextOmittedForUnsupportedModel(t *testing.T) {
	srv, reqs := newTTSServer(t, []string{alignmentJSON(t, "Hello!", nil)})
	cfg := ttsCfg(srv)
	cfg.Model = "eleven_v3"
	s := &synthesizer{cfg: cfg, http: &http.Client{}}

	speak(t, s, "Hello!")
	speak(t, s, "How can I assist you today?")

	if _, present := (*reqs)[1].body["previous_text"]; present {
		t.Errorf("previous_text = %v, want it omitted for this model",
			(*reqs)[1].body["previous_text"])
	}
	// The speech is still tracked; only the sending is gated on the model.
	s.mu.Lock()
	remembered := s.previousText
	s.mu.Unlock()
	if remembered != "Hello! How can I assist you today?" {
		t.Errorf("previousText = %q, want the turn's speech still tracked", remembered)
	}
}

// TestTurnResetDropsTheContext checks each thing that ends a turn drops what was
// carried across it, so the next turn does not open with the last one's text as
// context or its timings as an offset.
func TestTurnResetDropsTheContext(t *testing.T) {
	for _, tc := range []struct {
		name  string
		reset func(s *synthesizer)
	}{
		{"pipeline start", func(s *synthesizer) { s.Start(t.Context()) }},
		{"interruption", func(s *synthesizer) { s.OnAudioContextInterrupted(t.Context(), "ctx") }},
		{"turn completed", func(s *synthesizer) { s.OnTurnContextCompleted(t.Context(), "ctx") }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv, reqs := newTTSServer(t, []string{alignmentJSON(t, "Hello!", nil)})
			s := &synthesizer{cfg: ttsCfg(srv), http: &http.Client{}}

			speak(t, s, "Hello!")
			tc.reset(s)
			speak(t, s, "A new turn.")

			if _, present := (*reqs)[1].body["previous_text"]; present {
				t.Errorf("previous_text = %v, want the turn's context dropped",
					(*reqs)[1].body["previous_text"])
			}
			s.mu.Lock()
			cumulative := s.cumulative
			s.mu.Unlock()
			// The reply is "Hello!", six characters ending at 0.6s, and the new
			// turn's timeline starts from zero rather than from the last one.
			if cumulative != 0.6 {
				t.Errorf("cumulative = %v, want the new turn's timeline to start from zero", cumulative)
			}
		})
	}
}

// TestCumulativeTimeAdvancesAcrossSentences checks the second sentence of a turn
// is timed after the first rather than from zero, so the words line up with the
// audio actually playing.
func TestCumulativeTimeAdvancesAcrossSentences(t *testing.T) {
	srv, _ := newTTSServer(t, []string{alignmentJSON(t, "Hi there", nil)})
	s := &synthesizer{cfg: ttsCfg(srv), http: &http.Client{}}

	first := speak(t, s, "Hi there")
	second := speak(t, s, "Hi there")

	if first.words[0].Offset != 0 {
		t.Errorf("first sentence starts at %v, want 0", first.words[0].Offset)
	}
	// "Hi there" is eight characters at 0.1s each, so the utterance ends at 0.8s
	// and the next sentence's first word starts there.
	if second.words[0].Offset != 0.8 {
		t.Errorf("second sentence starts at %v, want it after the first", second.words[0].Offset)
	}
}

// TestLanguageCodeOnlyForMultilingualModels checks the language is sent only to
// the models that accept one.
func TestLanguageCodeOnlyForMultilingualModels(t *testing.T) {
	for model, want := range map[string]string{
		"eleven_flash_v2_5":      "fr",
		"eleven_turbo_v2_5":      "fr",
		"eleven_multilingual_v2": "",
		"eleven_v3":              "",
	} {
		srv, reqs := newTTSServer(t, []string{alignmentJSON(t, "bonjour", nil)})
		cfg := ttsCfg(srv)
		cfg.Model = model
		cfg.Language = language.French
		speak(t, &synthesizer{cfg: cfg, http: &http.Client{}}, "bonjour")

		got, _ := (*reqs)[0].body["language_code"].(string)
		if got != want {
			t.Errorf("model %q sent language_code %q, want %q", model, got, want)
		}
	}
}

// TestRunTTSOmitsUnsetOptions checks the optional body and query parameters stay
// off the request when unset, leaving the voice's own configuration in force.
func TestRunTTSOmitsUnsetOptions(t *testing.T) {
	srv, reqs := newTTSServer(t, []string{alignmentJSON(t, "hi", nil)})
	speak(t, &synthesizer{cfg: ttsCfg(srv), http: &http.Client{}}, "hi")

	one := (*reqs)[0]
	for _, f := range []string{
		"language_code", "voice_settings", "apply_text_normalization",
		"pronunciation_dictionary_locators", "previous_text",
	} {
		if _, present := one.body[f]; present {
			t.Errorf("%s was sent for an unset config: %v", f, one.body[f])
		}
	}
	for _, q := range []string{"optimize_streaming_latency", "enable_logging"} {
		if one.query.Has(q) {
			t.Errorf("%s was sent for an unset config: %q", q, one.query.Get(q))
		}
	}
}

// TestRunTTSOptionalOptions checks each option reaches the request where the API
// expects it: the generation controls in the body, the transport controls in the
// query string.
func TestRunTTSOptionalOptions(t *testing.T) {
	srv, reqs := newTTSServer(t, []string{alignmentJSON(t, "hi", nil)})
	stability := 0.4
	latency := 3
	logging := false
	cfg := ttsCfg(srv)
	cfg.VoiceSettings = &VoiceSettings{Stability: &stability}
	cfg.ApplyTextNormalization = "off"
	cfg.PronunciationDictionaryLocators = []PronunciationDictionaryLocator{{DictionaryID: "dict-1"}}
	cfg.OptimizeStreamingLatency = &latency
	cfg.EnableLogging = &logging
	speak(t, &synthesizer{cfg: cfg, http: &http.Client{}}, "hi")

	one := (*reqs)[0]
	settings, _ := one.body["voice_settings"].(map[string]any)
	if settings["stability"] != 0.4 {
		t.Errorf("voice_settings.stability = %v, want 0.4", settings["stability"])
	}
	if _, present := settings["similarity_boost"]; present {
		t.Error("an unset voice setting was sent, overriding the voice's own default")
	}
	if one.body["apply_text_normalization"] != "off" {
		t.Errorf("apply_text_normalization = %v, want off", one.body["apply_text_normalization"])
	}
	locators, _ := one.body["pronunciation_dictionary_locators"].([]any)
	if len(locators) != 1 {
		t.Fatalf("pronunciation_dictionary_locators = %v, want the one locator",
			one.body["pronunciation_dictionary_locators"])
	}
	locator, _ := locators[0].(map[string]any)
	if locator["pronunciation_dictionary_id"] != "dict-1" {
		t.Errorf("locator = %v, want the dictionary id", locator)
	}
	if _, present := locator["version_id"]; present {
		t.Error("an unset version_id was sent")
	}
	if one.query.Get("optimize_streaming_latency") != "3" {
		t.Errorf("optimize_streaming_latency = %q, want 3", one.query.Get("optimize_streaming_latency"))
	}
	if one.query.Get("enable_logging") != "false" {
		t.Errorf("enable_logging = %q, want false", one.query.Get("enable_logging"))
	}
}

// TestRunTTSStatusError checks a non-200 is reported rather than read as audio.
func TestRunTTSStatusError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte("invalid api key"))
	}))
	defer srv.Close()

	s := &synthesizer{cfg: ttsCfg(srv), http: &http.Client{}}
	err := s.RunTTS(t.Context(), "hi", "", func(frames.Frame) error {
		t.Error("a frame was yielded for a failed request")
		return nil
	})
	if !errors.Is(err, errTTSStatus) {
		t.Fatalf("RunTTS error = %v, want errTTSStatus", err)
	}
	if !strings.Contains(err.Error(), "invalid api key") {
		t.Errorf("error = %v, want it to carry the response body", err)
	}
}

// TestRunTTSSkipsUnparseableLines checks a line that is not JSON is skipped
// rather than failing the turn, since the audio around it is still good.
func TestRunTTSSkipsUnparseableLines(t *testing.T) {
	srv, _ := newTTSServer(t, []string{
		alignmentJSON(t, "Hi", []byte{1, 2}),
		"not json",
		"",
		alignmentJSON(t, " there", []byte{3, 4}),
	})

	got := speak(t, &synthesizer{cfg: ttsCfg(srv), http: &http.Client{}}, "Hi there")
	if !bytes.Equal(got.pcm, []byte{1, 2, 3, 4}) {
		t.Errorf("PCM = % x, want the audio from both good lines", got.pcm)
	}
}

// TestFlashAlignmentPreservesInterWordChunkSpace is upstream's
// test_elevenlabs_flash_alignment_preserves_inter_word_chunk_space: the flash
// models split a sentence mid-word and mark the join with a leading space on the
// next chunk, so only the utterance's own leading space may be stripped.
func TestFlashAlignmentPreservesInterWordChunkSpace(t *testing.T) {
	srv, _ := newTTSServer(t, []string{
		alignmentJSON(t, " Why did the math book", nil),
		alignmentJSON(t, " look so sad? ", nil),
		alignmentJSON(t, " Because it had too m", nil),
		alignmentJSON(t, "any problems. ", nil),
	})

	got := speak(t, &synthesizer{cfg: ttsCfg(srv), http: &http.Client{}}, "irrelevant")
	want := []string{
		"Why", "did", "the", "math", "book", "look", "so", "sad?",
		"Because", "it", "had", "too", "many", "problems.",
	}
	if !equalWords(spokenWords(got), want) {
		t.Errorf("words = %v, want %v", spokenWords(got), want)
	}
}

// TestStripLeadingSpaces is upstream's
// test_elevenlabs_alignment_strips_only_utterance_leading_spaces.
func TestStripLeadingSpaces(t *testing.T) {
	first := stripLeadingSpaces(charAlignmentOf("  Hello"), true)
	if strings.Join(first.Characters, "") != "Hello" {
		t.Errorf("first chunk = %q, want the leading spaces gone", strings.Join(first.Characters, ""))
	}
	if len(first.StartTimesSecs) != len(first.Characters) {
		t.Errorf("times = %d, want them cut alongside the %d characters",
			len(first.StartTimesSecs), len(first.Characters))
	}

	later := stripLeadingSpaces(charAlignmentOf(" world"), false)
	if strings.Join(later.Characters, "") != " world" {
		t.Errorf("later chunk = %q, want its separating space kept", strings.Join(later.Characters, ""))
	}
}

// TestSelectAlignment ports upstream's five _select_alignment cases. The plain
// alignment is the input as written and the normalized one is what was actually
// spoken, so the written form wins unless a pronunciation dictionary has
// rewritten the speech.
func TestSelectAlignment(t *testing.T) {
	plain := charAlignmentOf("Hello")
	normalized := charAlignmentOf(" Hello")

	both := &httpMessage{Alignment: plain, NormalizedAlignment: normalized}
	if got := selectAlignment(both, false); got != plain {
		t.Error("without a dictionary the written alignment must win")
	}
	if got := selectAlignment(both, true); got != normalized {
		t.Error("with a dictionary the normalized alignment must win")
	}

	// Either may be absent, so each falls back to the other.
	if got := selectAlignment(&httpMessage{NormalizedAlignment: normalized}, false); got != normalized {
		t.Error("a missing written alignment must fall back to the normalized one")
	}
	if got := selectAlignment(&httpMessage{Alignment: plain}, true); got != plain {
		t.Error("a missing normalized alignment must fall back to the written one")
	}
	if got := selectAlignment(&httpMessage{}, false); got != nil {
		t.Errorf("selectAlignment with neither = %v, want nil", got)
	}
	if got := selectAlignment(&httpMessage{}, true); got != nil {
		t.Errorf("selectAlignment with neither = %v, want nil", got)
	}
}

// TestSelectAlignmentPrefersNormalizedWithDictionary checks the choice end to
// end: a dictionary rewrites what is spoken, so its alignment is the one the
// timings belong to. Without one the written form is what the conversation
// should record.
func TestSelectAlignmentPrefersNormalizedWithDictionary(t *testing.T) {
	line, err := json.Marshal(map[string]any{
		"alignment":            alignmentMap("Doctor Smith"),
		"normalized_alignment": alignmentMap("Dr Smith"),
	})
	if err != nil {
		t.Fatalf("encoding the line: %v", err)
	}

	srv, _ := newTTSServer(t, []string{string(line)})
	written := speak(t, &synthesizer{cfg: ttsCfg(srv), http: &http.Client{}}, "Doctor Smith")
	if want := []string{"Doctor", "Smith"}; !equalWords(spokenWords(written), want) {
		t.Errorf("words = %v, want the written form %v", spokenWords(written), want)
	}

	srv2, _ := newTTSServer(t, []string{string(line)})
	cfg2 := ttsCfg(srv2)
	cfg2.PronunciationDictionaryLocators = []PronunciationDictionaryLocator{{DictionaryID: "d"}}
	normalized := speak(t, &synthesizer{cfg: cfg2, http: &http.Client{}}, "Doctor Smith")
	if want := []string{"Dr", "Smith"}; !equalWords(spokenWords(normalized), want) {
		t.Errorf("words = %v, want the spoken form %v", spokenWords(normalized), want)
	}
}

// TestSpacelessLanguageTimings ports upstream's
// test_elevenlabs_timestamp_spacing_languages and its Japanese reassembly case:
// a language written without spaces reports tokens that already read as
// continuous text, so nothing downstream adds spacing.
func TestSpacelessLanguageTimings(t *testing.T) {
	for _, l := range []language.Language{language.Japanese, language.ChineseCN, language.Chinese} {
		s := &synthesizer{cfg: Config{Language: l}}
		if !s.spacelessLanguage() {
			t.Errorf("spacelessLanguage(%q) = false, want true", l)
		}
	}
	for _, l := range []language.Language{
		language.English, language.French, language.Korean, language.Language(""),
	} {
		s := &synthesizer{cfg: Config{Language: l}}
		if s.spacelessLanguage() {
			t.Errorf("spacelessLanguage(%q) = true, want false", l)
		}
	}

	srv, _ := newTTSServer(t, []string{
		alignmentJSON(t, "どんなことでも気 ", nil),
		alignmentJSON(t, "軽に相談してくださいね。 ", nil),
	})
	cfg := ttsCfg(srv)
	cfg.Language = language.Japanese
	got := speak(t, &synthesizer{cfg: cfg, http: &http.Client{}}, "irrelevant")

	want := []string{"どんなことでも気", "軽に相談してくださいね。"}
	if !equalWords(spokenWords(got), want) {
		t.Errorf("words = %v, want %v", spokenWords(got), want)
	}
	if !got.opts.IncludesInterFrameSpaces {
		t.Error("a language written without spaces must report that it carries its own")
	}
}

// TestOutputFormat pins the PCM formats ElevenLabs offers. A rate it does not
// offer falls back to 24 kHz rather than asking for a format the API would
// reject.
func TestOutputFormat(t *testing.T) {
	for _, rate := range []int{8000, 16000, 22050, 24000, 32000, 44100, 48000} {
		want := "pcm_" + strconv.Itoa(rate)
		if got := outputFormat(rate); got != want {
			t.Errorf("outputFormat(%d) = %q, want %q", rate, got, want)
		}
	}
	for _, rate := range []int{0, 11025, 96000} {
		if got := outputFormat(rate); got != "pcm_24000" {
			t.Errorf("outputFormat(%d) = %q, want the pcm_24000 fallback", rate, got)
		}
	}
}

// TestElevenlabsLanguage pins the synthesis codes. Synthesis takes the plain
// base code, where transcription takes ISO-639-3, and a language ElevenLabs does
// not speak maps to nothing so the model detects it.
func TestElevenlabsLanguage(t *testing.T) {
	cases := map[language.Language]string{
		language.English:           "en",
		language.EnglishGB:         "en",
		language.French:            "fr",
		language.FrenchCA:          "fr",
		language.Spanish:           "es",
		language.German:            "de",
		language.PortugueseBR:      "pt",
		language.Chinese:           "zh",
		language.Japanese:          "ja",
		language.Language(langFil): langFil,
		// Not spoken, so nothing is sent and the model auto-detects.
		language.Language("cy"): "",
		language.Language(""):   "",
	}
	for in, want := range cases {
		if got := elevenlabsLanguage(in); got != want {
			t.Errorf("elevenlabsLanguage(%q) = %q, want %q", in, got, want)
		}
	}
}
