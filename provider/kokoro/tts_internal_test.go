package kokoro

import (
	"context"
	"errors"
	"iter"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
)

// Tests for the language codes the service phonemizes in.

// Mandarin and French use espeak-ng voice names, which differ from the ISO
// code. espeak-ng has no "zh" or bare "fr"; a name it does not know fails at
// synthesis time rather than falling back.
func TestKokoroUsesEspeakNamesNotISOCodes(t *testing.T) {
	assertLanguage(t, language.Chinese, "cmn")
	assertLanguage(t, language.French, "fr-fr")
}

// Kokoro's zf_ and zm_ voices are Mandarin, whatever the script.
func TestKokoroChineseVariantsAllMapToMandarin(t *testing.T) {
	assertLanguage(t, language.ChineseCN, "cmn")
	assertLanguage(t, language.ChineseHK, "cmn")
	assertLanguage(t, language.ChineseTW, "cmn")
	assertLanguage(t, language.MandarinChinese, "cmn")
}

// Regions espeak-ng ships separately keep their own name.
func TestKokoroRegionalVariantsUseTheirOwnEspeakVoice(t *testing.T) {
	assertLanguage(t, language.EnglishGB, "en-gb")
	assertLanguage(t, language.FrenchBE, "fr-be")
	assertLanguage(t, language.FrenchCH, "fr-ch")
	assertLanguage(t, language.PortugueseBR, "pt-br")
	// espeak-ng has no Canadian French; fr-fr is the nearest it ships.
	assertLanguage(t, language.FrenchCA, "fr-fr")
}

func TestKokoroBaseLanguagesPassThrough(t *testing.T) {
	assertLanguage(t, language.English, "en-us")
	assertLanguage(t, language.Spanish, "es")
	assertLanguage(t, language.Hindi, "hi")
	assertLanguage(t, language.Italian, "it")
	assertLanguage(t, language.Japanese, "ja")
	assertLanguage(t, language.Portuguese, "pt")
}

func assertLanguage(t *testing.T, l language.Language, want string) {
	t.Helper()
	if got := languageToKokoroLanguage(l); got != want {
		t.Errorf("languageToKokoroLanguage(%s) = %q, want %q", l, got, want)
	}
}

// Tests for the voice validation at construction. The model and voices file
// are replaced with stand-ins so nothing is downloaded.

// fakeModel is a model whose voices file holds the voices named.
type fakeModel struct {
	voices map[string]bool
	closed bool
}

func (m *fakeModel) hasVoice(name string) bool { return m.voices[name] }

func (*fakeModel) createStream(context.Context, string, string, string, float64) iter.Seq2[[]float32, error] {
	return func(func([]float32, error) bool) {}
}

func (m *fakeModel) close() error {
	m.closed = true
	return nil
}

// build builds the service with the stand-ins, the voices file holding
// af_heart.
func build(t *testing.T, cfg Config) (*synthesizer, *fakeModel, error) {
	t.Helper()
	model := &fakeModel{voices: map[string]bool{"af_heart": true}}
	cfg.CacheDir = t.TempDir()
	noDownload := func(context.Context, Config, string, string) error { return nil }
	load := func(string, string, EspeakConfig) (kokoroModel, error) { return model, nil }
	_, s, err := newTTS(cfg, noDownload, load)
	return s, model, err
}

func TestAKnownVoiceConstructs(t *testing.T) {
	s, _, err := build(t, Config{Voice: "af_heart"})
	if err != nil {
		t.Fatalf("newTTS: %v", err)
	}
	if v, _ := s.live.Voice.Value(); v != "af_heart" {
		t.Errorf("voice = %q, want af_heart", v)
	}
}

func TestMissingOrBlankVoiceIsRejected(t *testing.T) {
	// An empty Voice is how a configuration gives no voice, and a blank one.
	_, model, err := build(t, Config{Voice: ""})
	if err == nil || !strings.Contains(err.Error(), "Kokoro TTS voice must be specified") {
		t.Fatalf("err = %v, want the missing voice reported", err)
	}
	if !model.closed {
		t.Error("the model loaded for a rejected configuration was not released")
	}
}

func TestUnknownVoiceIsRejectedAtConstruction(t *testing.T) {
	_, _, err := build(t, Config{Voice: "af_nope"})
	if err == nil || !strings.Contains(err.Error(), "'af_nope' is not in the voices file") {
		t.Fatalf("err = %v, want the unknown voice reported", err)
	}
}

// The settings start with no model, the configured voice, the language named
// the way espeak-ng names it once the base has named it, and a speed of 1.
func TestSettingsStartFromTheConfig(t *testing.T) {
	s, _, err := build(t, Config{Voice: "af_heart"})
	if err != nil {
		t.Fatalf("newTTS: %v", err)
	}
	if _, ok := s.live.Model.Value(); ok || !s.live.Model.IsGiven() {
		t.Errorf("model = %+v, want given with no value", s.live.Model)
	}
	if lang := s.live.Language.Or(""); lang != "en-us" {
		t.Errorf("language = %q, want en-us", lang)
	}
	if speed := s.live.Speed.Or(0); speed != 1.0 {
		t.Errorf("speed = %v, want 1", speed)
	}
	if s.SampleRate() != SampleRate {
		t.Errorf("sample rate = %d, want %d", s.SampleRate(), SampleRate)
	}

	speed := 1.5
	s, _, err = build(t, Config{Voice: "af_heart", Language: language.ChineseCN, Speed: &speed, SampleRate: 16000})
	if err != nil {
		t.Fatalf("newTTS: %v", err)
	}
	if lang := s.live.Language.Or(""); lang != "cmn" {
		t.Errorf("language = %q, want cmn", lang)
	}
	if got := s.live.Speed.Or(0); got != 1.5 {
		t.Errorf("speed = %v, want 1.5", got)
	}
	if s.SampleRate() != 16000 {
		t.Errorf("sample rate = %d, want 16000", s.SampleRate())
	}
}

// A model or voices file not configured is looked for in the cache directory,
// and downloaded there when missing.
func TestFilesDefaultToTheCacheDirectory(t *testing.T) {
	dir := t.TempDir()
	var gotModel, gotVoices string
	ensure := func(_ context.Context, _ Config, modelPath, voicesPath string) error {
		gotModel, gotVoices = modelPath, voicesPath
		return nil
	}
	load := func(string, string, EspeakConfig) (kokoroModel, error) {
		return &fakeModel{voices: map[string]bool{"af_heart": true}}, nil
	}
	if _, _, err := newTTS(Config{Voice: "af_heart", CacheDir: dir}, ensure, load); err != nil {
		t.Fatalf("newTTS: %v", err)
	}
	if gotModel != dir+"/kokoro-v1.0.onnx" || gotVoices != dir+"/voices-v1.0.bin" {
		t.Errorf("files = %q, %q", gotModel, gotVoices)
	}
}

// failingModel is a model whose synthesis fails, after a first batch.
type failingModel struct {
	fakeModel
	err error
}

func (m *failingModel) createStream(context.Context, string, string, string, float64) iter.Seq2[[]float32, error] {
	return func(yield func([]float32, error) bool) {
		if yield([]float32{0.5, -1}, nil) {
			yield(nil, m.err)
		}
	}
}

// errBoom is the failure the failing model reports.
var errBoom = errors.New("boom")

// The audio is yielded as 16-bit PCM at the model's rate, and a failure while
// synthesizing as an error frame, as the service reports it.
func TestASynthesisFailureIsYieldedAsAnErrorFrame(t *testing.T) {
	s, _, err := build(t, Config{Voice: "af_heart"})
	if err != nil {
		t.Fatalf("newTTS: %v", err)
	}
	s.model = &failingModel{err: errBoom}
	var got []frames.Frame
	err = s.RunTTS(t.Context(), "hello", "", func(f frames.Frame) error {
		got = append(got, f)
		return nil
	})
	if err != nil {
		t.Fatalf("RunTTS = %v, want the failure yielded", err)
	}
	if len(got) != 2 {
		t.Fatalf("yielded %d frames, want 2", len(got))
	}
	audio, ok := got[0].(*frames.TTSAudioRawFrame)
	if !ok || audio.SampleRate != SampleRate || audio.NumChannels != 1 {
		t.Fatalf("first frame = %#v, want audio at %d Hz", got[0], SampleRate)
	}
	// 0.5 and -1 scaled by 32767 and truncated: 16383 and -32767.
	if want := []byte{0xff, 0x3f, 0x01, 0x80}; string(audio.Audio) != string(want) {
		t.Errorf("pcm = %x, want %x", audio.Audio, want)
	}
	ef, ok := got[1].(*frames.ErrorFrame)
	if !ok || ef.Error != "Unknown error occurred: boom" {
		t.Fatalf("second frame = %#v, want the error", got[1])
	}
}
