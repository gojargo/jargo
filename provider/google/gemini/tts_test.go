package gemini

import (
	"bytes"
	"encoding/base64"
	"errors"
	"net/http"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
)

// errDownstream stands in for a downstream consumer that has gone away.
//
//nolint:gochecknoglobals // sentinel error for the tests below
var errDownstream = errors.New("downstream closed")

// wavReply renders a synthesize response carrying pcm behind the canonical
// 44-byte header, which is what the service strips back off.
func wavReply(pcm []byte) string {
	wav := append(bytes.Repeat([]byte{0}, wavHeaderBytes), pcm...)
	return `{"audioContent":"` + base64.StdEncoding.EncodeToString(wav) + `"}`
}

// synthesize runs one synthesis through syn and returns the PCM the frames
// carried, along with the rate they declared.
func synthesize(t *testing.T, syn *ttsSynthesizer, text string) ([]byte, int) {
	t.Helper()
	var pcm bytes.Buffer
	rate := 0
	err := syn.RunTTS(t.Context(), text, "", func(f frames.Frame) error {
		audio, ok := f.(*frames.TTSAudioRawFrame)
		if !ok {
			t.Errorf("yielded %T, want a TTSAudioRawFrame", f)
			return nil
		}
		rate = audio.SampleRate
		pcm.Write(audio.Audio)
		return nil
	})
	if err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	return pcm.Bytes(), rate
}

// TestSynthesizerReportsConfiguredRate checks the rate the service announces is
// the one it asked the API for, since the frames it emits are labeled with it
// and everything downstream resamples against that label.
func TestSynthesizerReportsConfiguredRate(t *testing.T) {
	rt := &roundTripper{body: wavReply([]byte{1, 2})}
	syn := &ttsSynthesizer{
		cfg:  TTSConfig{APIKey: "k", VoiceName: defaultVoiceName, SampleRate: 16000},
		http: &http.Client{Transport: rt},
	}

	if got := syn.SampleRate(); got != 16000 {
		t.Errorf("SampleRate() = %d, want the configured 16000", got)
	}
	if _, rate := synthesize(t, syn, "hi"); rate != 16000 {
		t.Errorf("frame rate = %d, want the configured 16000", rate)
	}
	audio, _ := rt.sent["audioConfig"].(map[string]any)
	if audio["sampleRateHertz"] != float64(16000) {
		t.Errorf("sampleRateHertz = %v, want the rate the frames are labeled with", audio["sampleRateHertz"])
	}
}

// TestRunTTSRequestShape checks the synthesis request names the voice, language
// and audio format, and that the PCM behind the WAV header reaches the caller.
func TestRunTTSRequestShape(t *testing.T) {
	want := bytes.Repeat([]byte{0x11, 0x22}, 32)
	rt := &roundTripper{body: wavReply(want)}
	syn := &ttsSynthesizer{
		cfg: TTSConfig{
			APIKey:     "test-key",
			VoiceName:  "en-US-Chirp3-HD-Charon",
			Language:   language.French,
			SampleRate: 24000,
		},
		http: &http.Client{Transport: rt},
	}

	pcm, rate := synthesize(t, syn, "hello there")
	if !bytes.Equal(pcm, want) {
		t.Errorf("PCM = % x, want the samples behind the WAV header", pcm)
	}
	if rate != 24000 {
		t.Errorf("frame rate = %d, want the configured 24000", rate)
	}

	if rt.url != ttsEndpoint+"?key=test-key" {
		t.Errorf("URL = %q, want the synthesize endpoint carrying the key", rt.url)
	}
	if got := rt.header.Get("Content-Type"); got != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", got)
	}

	input, _ := rt.sent["input"].(map[string]any)
	if input["text"] != "hello there" {
		t.Errorf("input text = %v, want the text to speak", input["text"])
	}
	voice, _ := rt.sent["voice"].(map[string]any)
	if voice["name"] != "en-US-Chirp3-HD-Charon" {
		t.Errorf("voice name = %v", voice["name"])
	}
	if voice["languageCode"] != "fr-FR" {
		t.Errorf("voice languageCode = %v, want fr-FR", voice["languageCode"])
	}
	audio, _ := rt.sent["audioConfig"].(map[string]any)
	if audio["audioEncoding"] != "LINEAR16" {
		t.Errorf("audioEncoding = %v, want LINEAR16", audio["audioEncoding"])
	}
	if audio["sampleRateHertz"] != float64(24000) {
		t.Errorf("sampleRateHertz = %v, want the configured rate", audio["sampleRateHertz"])
	}
}

// TestRunTTSHeaderOnlyResponse checks a response carrying nothing but the WAV
// header yields no audio rather than a slice read past its end.
func TestRunTTSHeaderOnlyResponse(t *testing.T) {
	rt := &roundTripper{body: wavReply(nil)}
	syn := &ttsSynthesizer{cfg: TTSConfig{APIKey: "k", SampleRate: 24000}, http: &http.Client{Transport: rt}}

	pcm, _ := synthesize(t, syn, "hi")
	if len(pcm) != 0 {
		t.Errorf("PCM = % x, want nothing", pcm)
	}
}

// TestRunTTSStatusError checks a non-200 is reported rather than decoded.
func TestRunTTSStatusError(t *testing.T) {
	rt := &roundTripper{status: http.StatusUnauthorized, body: "bad key"}
	syn := &ttsSynthesizer{cfg: TTSConfig{APIKey: "k", SampleRate: 24000}, http: &http.Client{Transport: rt}}

	err := syn.RunTTS(t.Context(), "hi", "", func(frames.Frame) error {
		t.Error("a frame was yielded for a failed request")
		return nil
	})
	if !errors.Is(err, errStatus) {
		t.Fatalf("RunTTS error = %v, want errStatus", err)
	}
}

// TestRunTTSBadBase64 checks an audioContent that is not base64 is reported
// rather than passed downstream as audio.
func TestRunTTSBadBase64(t *testing.T) {
	rt := &roundTripper{body: `{"audioContent":"!!!not base64!!!"}`}
	syn := &ttsSynthesizer{cfg: TTSConfig{APIKey: "k", SampleRate: 24000}, http: &http.Client{Transport: rt}}

	err := syn.RunTTS(t.Context(), "hi", "", func(frames.Frame) error { return nil })
	if err == nil {
		t.Fatal("RunTTS on an undecodable body = nil, want an error")
	}
}

// TestRunTTSYieldError checks an error from downstream is reported back.
func TestRunTTSYieldError(t *testing.T) {
	rt := &roundTripper{body: wavReply([]byte{1, 2, 3, 4})}
	syn := &ttsSynthesizer{cfg: TTSConfig{APIKey: "k", SampleRate: 24000}, http: &http.Client{Transport: rt}}

	err := syn.RunTTS(t.Context(), "hi", "", func(frames.Frame) error { return errDownstream })
	if !errors.Is(err, errDownstream) {
		t.Fatalf("RunTTS error = %v, want the downstream error", err)
	}
}

// TestLanguageToGoogleTTS pins the synthesis language codes. They are not the
// recognition codes: Google synthesizes Portuguese as Brazilian and Chinese as
// plain Mandarin, where it recognizes European Portuguese and Simplified
// Chinese.
func TestLanguageToGoogleTTS(t *testing.T) {
	cases := map[language.Language]string{
		language.English:      "en-US",
		language.EnglishGB:    "en-US",
		language.French:       "fr-FR",
		language.Spanish:      "es-ES",
		language.German:       "de-DE",
		language.Italian:      "it-IT",
		language.Dutch:        "nl-NL",
		language.Portuguese:   "pt-BR",
		language.Polish:       "pl-PL",
		language.Russian:      "ru-RU",
		language.Japanese:     "ja-JP",
		language.Korean:       "ko-KR",
		language.Chinese:      "cmn-CN",
		language.Language(""): defaultLangCode,
		// Not modeled above, so the configured code is passed through as-is.
		language.Language("sv-SE"): "sv-SE",
	}
	for in, want := range cases {
		if got := languageToGoogleTTS(in); got != want {
			t.Errorf("languageToGoogleTTS(%q) = %q, want %q", in, got, want)
		}
	}

	// The two maps genuinely disagree for these, which is the reason there are
	// two of them.
	for _, l := range []language.Language{language.Portuguese, language.Chinese} {
		if languageToGoogleTTS(l) == languageToGoogleSTT(l) {
			t.Errorf("%q maps the same way for synthesis and recognition", l)
		}
	}
}
