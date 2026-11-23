package piper

import (
	"context"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/settings"
)

// Tests for the in-process service, with the library and the voice download
// replaced by stand-ins so nothing native is loaded and nothing is fetched.

// Errors the stand-ins fail with.
var (
	errNoDownload = errors.New("no download in this test")
	errBoom       = errors.New("boom")
)

// fakeVoice is a loaded voice that records what it is asked to synthesize and
// answers with chunks, or one short chunk when it has none.
type fakeVoice struct {
	chunks [][]float32
	err    error

	mu     sync.Mutex
	texts  []string
	opts   []synthesisOptions
	closed bool
}

func (v *fakeVoice) synthesize(
	_ context.Context, text string, opts synthesisOptions, yield func([]float32, int) error,
) error {
	v.mu.Lock()
	v.texts = append(v.texts, text)
	v.opts = append(v.opts, opts)
	v.mu.Unlock()
	if v.err != nil {
		return v.err
	}
	chunks := v.chunks
	if chunks == nil {
		// Some audio, so the utterance does not wait out the no-audio timeout.
		chunks = [][]float32{{0.5, -0.5}}
	}
	for _, c := range chunks {
		if err := yield(c, 22050); err != nil {
			return err
		}
	}
	return nil
}

func (v *fakeVoice) close() { v.closed = true }

// fakeLibrary loads the fake voice, recording the paths it is asked for.
type fakeLibrary struct {
	voice     *fakeVoice
	models    []string
	espeakDir []string
}

func (l *fakeLibrary) loadVoice(modelPath, espeakDataPath string) (voiceModel, error) {
	l.models = append(l.models, modelPath)
	l.espeakDir = append(l.espeakDir, espeakDataPath)
	return l.voice, nil
}

// writeVoice puts a voice's files in dir: an empty model and a configuration
// giving rate.
func writeVoice(t *testing.T, dir, voice string, rate int) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, voice+".onnx"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	config := []byte(`{"audio": {"sample_rate": ` + strconv.Itoa(rate) + `, "quality": "high"}}`)
	if err := os.WriteFile(filepath.Join(dir, voice+".onnx.json"), config, 0o600); err != nil {
		t.Fatal(err)
	}
}

// fakeDeps are the stand-ins: lib for the library, and a download that fails
// the test unless allowed, recording the voices it is asked for.
type fakeDeps struct {
	lib       *fakeLibrary
	downloads []string
	download  func(dir, voice string) (string, error)
	opened    bool
}

func newFakeDeps(voice *fakeVoice) *fakeDeps {
	return &fakeDeps{lib: &fakeLibrary{voice: voice}}
}

func (d *fakeDeps) deps(t *testing.T) dependencies {
	t.Helper()
	return dependencies{
		openLibrary: func(string) (voiceLoader, error) {
			d.opened = true
			return d.lib, nil
		},
		downloadVoice: func(_ context.Context, _ Config, voice, dir string) (string, error) {
			d.downloads = append(d.downloads, voice)
			if d.download == nil {
				t.Errorf("the voice %q was downloaded", voice)
				return "", errNoDownload
			}
			return d.download(dir, voice)
		},
	}
}

// A missing voice is rejected before anything is downloaded. An empty Voice
// is how a configuration gives no voice, and a blank one.
func TestLocalPiperRejectsAMissingOrBlankVoiceBeforeDownloading(t *testing.T) {
	d := newFakeDeps(&fakeVoice{})
	_, _, err := newTTS(Config{Voice: "", DownloadDir: t.TempDir()}, d.deps(t))
	if err == nil || !strings.Contains(err.Error(), "Piper TTS voice must be specified") {
		t.Fatalf("err = %v, want the missing voice reported", err)
	}
	if len(d.downloads) > 0 || len(d.lib.models) > 0 {
		t.Errorf("downloaded %v and loaded %v for a configuration with no voice", d.downloads, d.lib.models)
	}
}

// The synthesis settings reach the voice with each utterance, as set, and
// unset ones are left to the voice.
func TestLocalPiperPassesSynthesisSettingsToTheVoice(t *testing.T) {
	speaker, length, noise, noiseW := 2, 1.3, 0.5, 0.6
	for name, tc := range map[string]struct {
		cfg  Config
		want synthesisOptions
	}{
		"set": {
			Config{SpeakerID: &speaker, LengthScale: &length, NoiseScale: &noise, NoiseWScale: &noiseW},
			synthesisOptions{speakerID: &speaker, lengthScale: &length, noiseScale: &noise, noiseWScale: &noiseW},
		},
		"unset": {Config{}, synthesisOptions{}},
	} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			writeVoice(t, dir, "en_US-ryan-high", 22050)
			voice := &fakeVoice{}
			cfg := tc.cfg
			cfg.Voice, cfg.DownloadDir = "en_US-ryan-high", dir
			b, _, err := newTTS(cfg, newFakeDeps(voice).deps(t))
			if err != nil {
				t.Fatalf("newTTS: %v", err)
			}

			runTest(t, b, speakFrame("Hello.", false))

			if len(voice.opts) != 1 {
				t.Fatalf("synthesized %d times, want once", len(voice.opts))
			}
			if !reflect.DeepEqual(voice.opts[0], tc.want) {
				t.Errorf("options = %+v, want %+v", describe(voice.opts[0]), describe(tc.want))
			}
		})
	}
}

// describe is the options with their values rather than their addresses.
func describe(o synthesisOptions) []any {
	var out []any
	for _, p := range []any{o.speakerID, o.lengthScale, o.noiseScale, o.noiseWScale} {
		switch v := p.(type) {
		case *int:
			if v != nil {
				out = append(out, *v)
				continue
			}
		case *float64:
			if v != nil {
				out = append(out, *v)
				continue
			}
		}
		out = append(out, nil)
	}
	return out
}

// The voice's audio is streamed as Piper hands it over, labeled with the
// voice's rate, between the started and stopped frames; the service outputs
// at the voice's rate unless configured otherwise.
func TestLocalPiperStreamsTheVoicesAudio(t *testing.T) {
	dir := t.TempDir()
	writeVoice(t, dir, "en_US-ryan-high", 22050)
	voice := &fakeVoice{chunks: [][]float32{{0.25, -0.5}, {0.1, 0.2, -0.4}}}
	b, s, err := newTTS(Config{Voice: "en_US-ryan-high", DownloadDir: dir}, newFakeDeps(voice).deps(t))
	if err != nil {
		t.Fatalf("newTTS: %v", err)
	}
	if s.SampleRate() != 22050 {
		t.Errorf("sample rate = %d, want the voice's 22050", s.SampleRate())
	}

	down, _ := runTest(t, b, speakFrame("Hello world.", false))

	started, stopped := indexOf[*frames.TTSStartedFrame](down), indexOf[*frames.TTSStoppedFrame](down)
	if started < 0 || stopped < started {
		t.Fatalf("frames = %v, want started then stopped", frameNames(down))
	}
	audio := audioFrames(down)
	if len(audio) != 2 {
		t.Fatalf("got %d audio frames, want one per chunk", len(audio))
	}
	// Each chunk is scaled so its loudest sample reaches full scale.
	want := [][]int16{{16383, -32767}, {8191, 16383, -32767}}
	for i, a := range audio {
		if a.SampleRate != 22050 {
			t.Errorf("audio at %d Hz, want 22050", a.SampleRate)
		}
		got := make([]int16, len(a.Audio)/2)
		for j := range got {
			got[j] = int16(binary.LittleEndian.Uint16(a.Audio[2*j:])) //nolint:gosec // reading PCM back
		}
		if !slices.Equal(got, want[i]) {
			t.Errorf("chunk %d = %v, want %v", i, got, want[i])
		}
	}

	_, s, err = newTTS(Config{Voice: "en_US-ryan-high", DownloadDir: dir, SampleRate: 16000},
		newFakeDeps(&fakeVoice{}).deps(t))
	if err != nil {
		t.Fatalf("newTTS: %v", err)
	}
	if s.SampleRate() != 16000 {
		t.Errorf("sample rate = %d, want the configured 16000", s.SampleRate())
	}
}

// A synthesis that fails is reported as an error frame.
func TestLocalPiperReportsAFailedSynthesis(t *testing.T) {
	dir := t.TempDir()
	writeVoice(t, dir, "en_US-ryan-high", 22050)
	voice := &fakeVoice{err: errBoom}
	b, _, err := newTTS(Config{Voice: "en_US-ryan-high", DownloadDir: dir}, newFakeDeps(voice).deps(t))
	if err != nil {
		t.Fatalf("newTTS: %v", err)
	}

	_, up := runTest(t, b, speakFrame("Hello.", false))

	if errs := errorsOf(up); len(errs) == 0 || errs[0] != "Unknown error occurred: boom" {
		t.Errorf("errors = %q, want the failure reported", errs)
	}
}

// A voice whose model is not in the download directory is downloaded into it,
// and one that is, is not.
func TestLocalPiperDownloadsAMissingVoiceOnce(t *testing.T) {
	dir := t.TempDir()
	d := newFakeDeps(&fakeVoice{})
	d.download = func(dir, voice string) (string, error) {
		writeVoice(t, dir, voice, 22050)
		return voice, nil
	}
	cfg := Config{Voice: "en_US-ryan-high", DownloadDir: dir}
	if _, _, err := newTTS(cfg, d.deps(t)); err != nil {
		t.Fatalf("newTTS: %v", err)
	}
	if _, _, err := newTTS(cfg, d.deps(t)); err != nil {
		t.Fatalf("newTTS: %v", err)
	}
	if !slices.Equal(d.downloads, []string{"en_US-ryan-high"}) {
		t.Errorf("downloads = %v, want the voice downloaded once", d.downloads)
	}
	want := filepath.Join(dir, "en_US-ryan-high.onnx")
	if !slices.Equal(d.lib.models, []string{want, want}) {
		t.Errorf("loaded %v, want %s each time", d.lib.models, want)
	}
}

// A voice that was renamed is loaded under the name it was downloaded as.
func TestLocalPiperLoadsARenamedVoiceUnderItsCurrentName(t *testing.T) {
	dir := t.TempDir()
	d := newFakeDeps(&fakeVoice{})
	d.download = func(dir, _ string) (string, error) {
		writeVoice(t, dir, "de_DE-karlsson-low", 16000)
		return "de_DE-karlsson-low", nil
	}
	_, s, err := newTTS(Config{Voice: "de-karlsson-low", DownloadDir: dir}, d.deps(t))
	if err != nil {
		t.Fatalf("newTTS: %v", err)
	}
	if want := filepath.Join(dir, "de_DE-karlsson-low.onnx"); !slices.Equal(d.lib.models, []string{want}) {
		t.Errorf("loaded %v, want %s", d.lib.models, want)
	}
	if s.SampleRate() != 16000 {
		t.Errorf("sample rate = %d, want the voice's 16000", s.SampleRate())
	}
}

// The espeak-ng data directory configured is the one the voice is loaded
// with.
func TestLocalPiperLoadsWithTheConfiguredEspeakData(t *testing.T) {
	dir := t.TempDir()
	writeVoice(t, dir, "en_US-ryan-high", 22050)
	d := newFakeDeps(&fakeVoice{})
	cfg := Config{Voice: "en_US-ryan-high", DownloadDir: dir, EspeakDataPath: "/opt/espeak-ng-data"}
	if _, _, err := newTTS(cfg, d.deps(t)); err != nil {
		t.Fatalf("newTTS: %v", err)
	}
	if !slices.Equal(d.lib.espeakDir, []string{"/opt/espeak-ng-data"}) {
		t.Errorf("espeak data = %v, want the configured directory", d.lib.espeakDir)
	}
}

// A voice configuration that gives no sample rate fails the construction, and
// the voice loaded for it is released.
func TestLocalPiperRejectsAVoiceWithNoSampleRate(t *testing.T) {
	dir := t.TempDir()
	writeVoice(t, dir, "en_US-ryan-high", 22050)
	if err := os.WriteFile(filepath.Join(dir, "en_US-ryan-high.onnx.json"), []byte(`{}`), 0o600); err != nil {
		t.Fatal(err)
	}
	voice := &fakeVoice{}
	_, _, err := newTTS(Config{Voice: "en_US-ryan-high", DownloadDir: dir}, newFakeDeps(voice).deps(t))
	if !errors.Is(err, errNoSampleRate) {
		t.Fatalf("err = %v, want errNoSampleRate", err)
	}
	if !voice.closed {
		t.Error("the voice loaded for a rejected configuration was not released")
	}
}

// warnings records the settings reported as not applied.
type warnings struct{ fields [][]string }

func (w *warnings) WarnUnhandledUpdatedSettings(_ context.Context, unhandled []string) {
	if len(unhandled) > 0 {
		w.fields = append(w.fields, unhandled)
	}
}

// A synthesis setting changed while running applies to the next utterance; a
// voice change is stored but reported as not applied.
func TestLocalPiperAppliesSynthesisUpdatesAndWarnsOnTheRest(t *testing.T) {
	dir := t.TempDir()
	writeVoice(t, dir, "en_US-ryan-high", 22050)
	voice := &fakeVoice{}
	b, s, err := newTTS(Config{Voice: "en_US-ryan-high", DownloadDir: dir}, newFakeDeps(voice).deps(t))
	if err != nil {
		t.Fatalf("newTTS: %v", err)
	}
	w := &warnings{}
	s.svc = w

	runTest(t, b,
		frames.NewTTSUpdateSettingsFrame(&TTSSettings{LengthScale: settings.Set(1.5)}),
		speakFrame("One.", false),
		frames.NewTTSUpdateSettingsFrame(&TTSSettings{Voice: settings.Set("en_US-lessac-medium")}),
		speakFrame("Two.", false),
	)

	if len(voice.opts) != 2 || voice.opts[0].lengthScale == nil || *voice.opts[0].lengthScale != 1.5 {
		t.Fatalf("options = %v, want the length scale applied", voice.opts)
	}
	if !reflect.DeepEqual(w.fields, [][]string{{"voice"}}) {
		t.Errorf("warned about %v, want only the voice", w.fields)
	}
	if v, _ := s.live.Voice.Value(); v != "en_US-lessac-medium" {
		t.Errorf("voice = %q, want the update stored", v)
	}
}

// The settings start with no model and no language, the configured voice, and
// the synthesis controls given, the others holding no value.
func TestLocalPiperSettingsStartFromTheConfig(t *testing.T) {
	dir := t.TempDir()
	writeVoice(t, dir, "en_US-ryan-high", 22050)
	length := 0.8
	_, s, err := newTTS(Config{Voice: "en_US-ryan-high", DownloadDir: dir, LengthScale: &length},
		newFakeDeps(&fakeVoice{}).deps(t))
	if err != nil {
		t.Fatalf("newTTS: %v", err)
	}
	for name, o := range map[string]interface {
		IsGiven() bool
		AsAny() any
	}{
		"model": s.live.Model, "language": s.live.Language, "speaker_id": s.live.SpeakerID,
		"noise_scale": s.live.NoiseScale, "noise_w_scale": s.live.NoiseWScale,
	} {
		if !o.IsGiven() || o.AsAny() != nil {
			t.Errorf("%s = %v, want given with no value", name, o)
		}
	}
	if v, _ := s.live.Voice.Value(); v != "en_US-ryan-high" {
		t.Errorf("voice = %q", v)
	}
	if v, _ := s.live.LengthScale.Value(); v != 0.8 {
		t.Errorf("length scale = %v, want 0.8", v)
	}
}

// The service frees the voice when the pipeline closes it.
func TestLocalPiperCloseReleasesTheVoice(t *testing.T) {
	dir := t.TempDir()
	writeVoice(t, dir, "en_US-ryan-high", 22050)
	voice := &fakeVoice{}
	_, s, err := newTTS(Config{Voice: "en_US-ryan-high", DownloadDir: dir}, newFakeDeps(voice).deps(t))
	if err != nil {
		t.Fatalf("newTTS: %v", err)
	}
	if err := s.Close(); err != nil || !voice.closed {
		t.Errorf("Close = %v, closed = %v", err, voice.closed)
	}
}

// Audio is scaled to full scale, a silent chunk stays silent, and samples are
// truncated toward zero.
func TestAudioInt16(t *testing.T) {
	for name, tc := range map[string]struct {
		in   []float32
		want []int16
	}{
		"scaled":    {[]float32{0.5, -0.25}, []int16{32767, -16383}},
		"silent":    {[]float32{0, 0}, []int16{0, 0}},
		"truncated": {[]float32{1, 0.00001}, []int16{32767, 0}},
	} {
		t.Run(name, func(t *testing.T) {
			pcm := audioInt16(tc.in)
			got := make([]int16, len(pcm)/2)
			for i := range got {
				got[i] = int16(binary.LittleEndian.Uint16(pcm[2*i:])) //nolint:gosec // reading PCM back
			}
			if !slices.Equal(got, tc.want) {
				t.Errorf("audioInt16(%v) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}
