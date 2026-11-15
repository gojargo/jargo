package speechify

import (
	"encoding/base64"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
	uctx "github.com/gojargo/jargo/utils/context"
)

const text = "Hello world."

// marks decodes speech marks as they arrive on the wire.
func marks(t *testing.T, raw string) any {
	t.Helper()
	dec := json.NewDecoder(strings.NewReader(raw))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		t.Fatal(err)
	}
	return v
}

// mark builds a word speech mark.
func mark(value string, start, end, startTime, endTime int) string {
	b, _ := json.Marshal(map[string]any{
		"type": "word", "value": value, "start": start, "end": end,
		"start_time": startTime, "end_time": endTime,
	})
	return string(b)
}

// list joins marks into a JSON array.
func list(ms ...string) string { return "[" + strings.Join(ms, ",") + "]" }

// wordTimes feeds chunks of speech marks through an accumulator and collects
// its words.
func wordTimes(t *testing.T, txt string, offset float64, chunks ...string) []uctx.WordTiming {
	t.Helper()
	a := newSpeechMarkAccumulator(txt, offset)
	var out []uctx.WordTiming
	for _, c := range chunks {
		out = append(out, a.add(marks(t, c))...)
	}
	return append(out, a.flush()...)
}

// streamingMarks are the speech marks Speechify streams for text.
//
//nolint:gochecknoglobals // test fixture, read-only
var streamingMarks = list(mark("Hello", 0, 5, 0, 469), mark("world.", 6, 12, 469, 1152))

// wt is a word timed at offset seconds.
func wt(word string, offset float64) uctx.WordTiming {
	return uctx.WordTiming{Word: word, Offset: offset}
}

func assertWords(t *testing.T, got, want []uctx.WordTiming) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("words = %v, want %v", got, want)
	}
	for i, w := range want {
		if got[i].Word != w.Word || math.Abs(got[i].Offset-w.Offset) > 1e-9 {
			t.Fatalf("words = %v, want %v", got, want)
		}
	}
}

// Speech marks become (word, seconds) pairs and an utterance end time.

func TestMarksConvertMillisecondsToSeconds(t *testing.T) {
	assertWords(t, wordTimes(t, text, 0, streamingMarks), []uctx.WordTiming{wt("Hello", 0), wt("world.", 0.469)})
}

// The offset sequences words across the utterances of one turn.
func TestTimeOffsetShiftsWords(t *testing.T) {
	assertWords(t, wordTimes(t, text, 2.5, streamingMarks), []uctx.WordTiming{wt("Hello", 2.5), wt("world.", 2.969)})
}

func TestEndTimeTracksTheLastMarkAndIgnoresTheOffset(t *testing.T) {
	a := newSpeechMarkAccumulator(text, 2.5)
	a.add(marks(t, streamingMarks))
	if math.Abs(a.endTime-1.152) > 1e-9 {
		t.Errorf("end time = %v, want 1.152", a.endTime)
	}
}

func TestNonWordMarksAreSkipped(t *testing.T) {
	ms := list(`{"type": "sentence", "start": 0, "end": 12, "start_time": 0, "end_time": 1152}`,
		mark("Hello", 0, 5, 0, 469))
	assertWords(t, wordTimes(t, text, 0, ms), []uctx.WordTiming{wt("Hello", 0)})
}

func TestEmptyMarksProduceNothing(t *testing.T) {
	for _, ms := range []string{"null", "[]", "[{}]", `[{"type": "word", "value": ""}]`} {
		t.Run(ms, func(t *testing.T) {
			assertWords(t, wordTimes(t, text, 0, ms), nil)
		})
	}
}

// Words come from the synthesized text, not from the mark's normalized value.

// Speechify flattens typographic apostrophes; the offsets recover them.
func TestNormalizedValueIsReplacedByTheOriginalSpelling(t *testing.T) {
	assertWords(t, wordTimes(t, "you’d like", 0, list(mark("you'd", 0, 5, 0, 0))), []uctx.WordTiming{wt("you’d", 0)})
}

func TestUnusableOffsetsFallBackToTheValue(t *testing.T) {
	for name, unusable := range map[string]string{
		"no offsets":       `{"type": "word", "value": "world."}`,
		"end past text":    `{"type": "word", "start": 6, "end": 99, "value": "world."}`,
		"empty span":       `{"type": "word", "start": 6, "end": 6, "value": "world."}`,
		"not integers":     `{"type": "word", "start": "6", "end": "12", "value": "world."}`,
		"fractional start": `{"type": "word", "start": 6.0, "end": 12, "value": "world."}`,
	} {
		t.Run(name, func(t *testing.T) {
			assertWords(t, wordTimes(t, text, 0, list(unusable)), []uctx.WordTiming{wt("world.", 0)})
		})
	}
}

// Marks that abut in the text are one word, even across chunk boundaries.

// hyphenated is marked by Speechify as five separate marks for
// "text-to-speech".
const hyphenated = "I like text-to-speech systems."

func TestAbuttingMarksJoinIntoOneWord(t *testing.T) {
	ms := list(mark("text", 7, 11, 0, 0), mark("-", 11, 12, 0, 0), mark("to", 12, 14, 0, 0),
		mark("-", 14, 15, 0, 0), mark("speech", 15, 21, 0, 0), mark("systems.", 22, 30, 0, 0))
	assertWords(t, wordTimes(t, hyphenated, 0, ms), []uctx.WordTiming{wt("text-to-speech", 0), wt("systems.", 0)})
}

// A run of abutting marks can straddle two speech.chunk events.
func TestASplitWordIsJoinedAcrossChunks(t *testing.T) {
	got := wordTimes(t, hyphenated, 0,
		list(mark("text", 7, 11, 0, 0), mark("-", 11, 12, 0, 0), mark("to", 12, 14, 0, 0), mark("-", 14, 15, 0, 0)),
		list(mark("speech", 15, 21, 0, 0), mark("systems.", 22, 30, 0, 0)))
	assertWords(t, got, []uctx.WordTiming{wt("text-to-speech", 0), wt("systems.", 0)})
}

func TestAJoinedWordKeepsTheFirstMarksTimestamp(t *testing.T) {
	ms := list(mark("text", 7, 11, 500, 0), mark("-", 11, 12, 900, 0))
	assertWords(t, wordTimes(t, hyphenated, 0, ms), []uctx.WordTiming{wt("text-", 0.5)})
}

// Only words the text can still extend are held back for the next chunk.
func TestAWordIsReleasedAsSoonAsTheTextShowsItCannotContinue(t *testing.T) {
	a := newSpeechMarkAccumulator(hyphenated, 0)
	got := a.add(marks(t, list(mark("I", 0, 1, 0, 0), mark("like", 2, 6, 0, 0))))
	assertWords(t, got, []uctx.WordTiming{wt("I", 0), wt("like", 0)})
}

// The text ends mid-run when the trailing mark is not the final character.
func TestAWordLeftPendingAtTheEndOfTheStreamIsFlushed(t *testing.T) {
	a := newSpeechMarkAccumulator(hyphenated, 0)
	a.add(marks(t, list(mark("text", 7, 11, 0, 0), mark("-", 11, 12, 0, 0))))
	assertWords(t, a.flush(), []uctx.WordTiming{wt("text-", 0)})
	assertWords(t, a.flush(), nil)
}

// Server-sent events blocks decode to an event name and payload.

func assertEvent(t *testing.T, block, wantName, wantData string) {
	t.Helper()
	got, ok := parseSSEEvent(block)
	if !ok {
		t.Fatalf("%q decoded to nothing", block)
	}
	want, _ := marks(t, wantData).(map[string]any)
	if got.name != wantName || !reflect.DeepEqual(got.data, want) {
		t.Errorf("event = %q %v, want %q %v", got.name, got.data, wantName, want)
	}
}

func TestSSEChunkWithAudioAndMarks(t *testing.T) {
	assertEvent(t, "event: speech.chunk\ndata: {\"audio\":\"SUQzBAA=\",\"speech_marks\":[]}",
		"speech.chunk", `{"audio": "SUQzBAA=", "speech_marks": []}`)
}

func TestSSEDoneEvent(t *testing.T) {
	assertEvent(t, "event: speech.done\ndata: {\"billable_characters_count\":40,\"audio_duration_ms\":4350}",
		"speech.done", `{"billable_characters_count": 40, "audio_duration_ms": 4350}`)
}

func TestSSEUnknownEventNameIsPreservedForTheCallerToIgnore(t *testing.T) {
	assertEvent(t, "event: speech.something-new\ndata: {}", "speech.something-new", "{}")
}

func TestSSEMultiLineDataIsJoined(t *testing.T) {
	assertEvent(t, "event: speech.done\ndata: {\"audio_duration_ms\":\ndata: 4350}",
		"speech.done", `{"audio_duration_ms": 4350}`)
}

func TestSSECommentLinesAreIgnored(t *testing.T) {
	assertEvent(t, ": keep-alive\nevent: speech.done\ndata: {}", "speech.done", "{}")
}

func TestSSEUndecodableBlocksYieldNothing(t *testing.T) {
	for _, block := range []string{"event: speech.done", `data: {"audio":`, ""} {
		if _, ok := parseSSEEvent(block); ok {
			t.Errorf("%q decoded to an event", block)
		}
	}
}

// Sample rates map to Speechify's PCM output formats.

func TestSupportedSampleRatesPassThrough(t *testing.T) {
	for _, rate := range []int{8000, 16000, 22050, 24000, 44100, 48000} {
		if format, got := outputFormat(rate); format != "pcm_"+strconv.Itoa(rate) || got != rate {
			t.Errorf("outputFormat(%d) = %s, %d", rate, format, got)
		}
	}
}

// Speechify has no pcm_32000, so the caller is told the real synthesis rate.
func TestUnsupportedSampleRateFallsBack(t *testing.T) {
	if format, got := outputFormat(32000); format != "pcm_24000" || got != 24000 {
		t.Errorf("outputFormat(32000) = %s, %d, want pcm_24000, 24000", format, got)
	}
}

// Languages resolve to the regional tags Speechify documents.

func TestLanguageMapping(t *testing.T) {
	for l, want := range map[language.Language]string{
		// Base languages resolve to supported regional tags.
		language.English: "en-US", language.Portuguese: "pt-BR", language.German: "de-DE",
		// Regional variants are preserved.
		language.SpanishMX: "es-MX", language.EnglishGB: "en-GB",
		// Unmapped languages pass through as BCP-47 tags.
		language.JapaneseJP: "ja-JP",
	} {
		if got := speechifyLanguage(l); got != want {
			t.Errorf("speechifyLanguage(%s) = %q, want %q", l, got, want)
		}
	}
}

// The request carries the pinned version and the settings, the audio of each
// chunk is yielded at the synthesized rate, the words are timed on the turn's
// timeline, and the next utterance of the turn starts where this one ended.
func TestHTTPTTSStreamsAudioAndWords(t *testing.T) {
	pcm := []byte{1, 2, 3, 4}
	audio := base64.StdEncoding.EncodeToString(pcm)
	body := ": keep-alive\n\n" +
		"event: speech.chunk\ndata: {\"audio\":\"" + audio + "\",\"speech_marks\":" + streamingMarks + "}\n\n" +
		"event: speech.done\ndata: {\"audio_duration_ms\":2000}\n\n"
	var payload map[string]any
	var header http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &payload)
		header = r.Header.Clone()
		if r.URL.Path != "/v1/audio/stream/with-timestamps" {
			t.Errorf("path = %q", r.URL.Path)
		}
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	norm := true
	_, s := newHTTPTTS(HTTPTTSConfig{
		APIKey: "k", URL: srv.URL, SampleRate: 32000, TextNormalization: &norm, SafetyIdentifier: "user-1",
	})

	var out []frames.Frame
	var words []uctx.WordTiming
	run := func() {
		err := s.RunTTSTimed(t.Context(), text, "c1", func(f frames.Frame) error {
			out = append(out, f)
			return nil
		}, func(w []uctx.WordTiming, _ tts.WordTimingOptions) error {
			words = append(words, w...)
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	run()

	if header.Get("Authorization") != "Bearer k" || header.Get("Speechify-Version") != apiVersion ||
		header.Get("Speechify-Caller") != "jargo" {
		t.Errorf("headers = %v", header)
	}
	want := map[string]any{
		"input": text, "voice_id": "geffen_32", "model": "simba-3.5", "output_format": "pcm_24000",
		"safety_identifier": "user-1", "options": map[string]any{"text_normalization": true},
	}
	if !reflect.DeepEqual(payload, want) {
		t.Errorf("payload = %v, want %v", payload, want)
	}
	if len(out) != 1 {
		t.Fatalf("yielded %v, want one audio frame", out)
	}
	if f, ok := out[0].(*frames.TTSAudioRawFrame); !ok || string(f.Audio) != string(pcm) || f.SampleRate != 24000 {
		t.Errorf("yielded %v, want the audio at the synthesized 24 kHz", out[0])
	}
	assertWords(t, words, []uctx.WordTiming{wt("Hello", 0), wt("world.", 0.469)})

	words = nil
	run()
	assertWords(t, words, []uctx.WordTiming{wt("Hello", 2), wt("world.", 2.469)})

	s.OnTurnContextCompleted(t.Context(), "c1")
	words = nil
	run()
	assertWords(t, words, []uctx.WordTiming{wt("Hello", 0), wt("world.", 0.469)})
}

// A refused request and a failure reported midway are yielded as errors.
func TestHTTPTTSErrors(t *testing.T) {
	for name, tc := range map[string]struct {
		status int
		body   string
		want   string
	}{
		"refused": {http.StatusUnauthorized, "bad key", "Speechify API error: bad key"},
		"midway": {
			http.StatusOK,
			"event: speech.error\ndata: {\"error\":{\"code\":\"model_retired\",\"message\":\"gone\"}}\n\n",
			"Speechify API error (model_retired): gone",
		},
	} {
		t.Run(name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tc.status)
				_, _ = io.WriteString(w, tc.body)
			}))
			t.Cleanup(srv.Close)
			_, s := newHTTPTTS(HTTPTTSConfig{APIKey: "k", URL: srv.URL})
			var out []frames.Frame
			if err := s.RunTTS(t.Context(), text, "c1", func(f frames.Frame) error {
				out = append(out, f)
				return nil
			}); err != nil {
				t.Fatal(err)
			}
			if len(out) != 1 {
				t.Fatalf("yielded %v, want one error", out)
			}
			if ef, ok := out[0].(*frames.ErrorFrame); !ok || ef.Error != tc.want {
				t.Errorf("yielded %v, want %q", out[0], tc.want)
			}
		})
	}
}
