package piper

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"os"
	"path/filepath"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
)

// errNoSampleRate reports a voice configuration that gives no sample rate.
var errNoSampleRate = errors.New("piper: the voice configuration gives no audio sample rate")

// Config configures the in-process Piper TTS service.
type Config struct {
	// Voice is the Piper voice to speak in, such as "en_US-ryan-high". Required.
	// The voices are listed in the rhasspy/piper-voices repository on Hugging
	// Face.
	Voice string

	// SpeakerID is which speaker to use, for voices trained on several
	// speakers; nil uses the voice's first speaker.
	SpeakerID *int
	// LengthScale is the phoneme length scale: below 1 is faster, above 1 is
	// slower. Nil uses the voice's default.
	LengthScale *float64
	// NoiseScale is the amount of generator noise; nil uses the voice's
	// default.
	NoiseScale *float64
	// NoiseWScale is the amount of phoneme width noise; nil uses the voice's
	// default.
	NoiseWScale *float64

	// LibraryPath is the libpiper shared library to load; empty looks for it
	// by its conventional name on the loader's default search path.
	LibraryPath string
	// EspeakDataPath is the espeak-ng data directory libpiper phonemizes
	// with; empty uses share/espeak-ng-data beside the library's lib
	// directory, where libpiper installs it, when LibraryPath names the
	// library and the directory exists, and the directory espeak-ng was built
	// to read otherwise.
	EspeakDataPath string
	// DownloadDir is where the voice files are stored; empty uses
	// DefaultCacheDir. A voice whose model is not there is downloaded into it.
	DownloadDir string
	// ForceRedownload downloads the voice files again even when they exist,
	// when the voice's model is to be downloaded.
	ForceRedownload bool
	// HTTPClient downloads the voice; nil uses http.DefaultClient.
	HTTPClient *http.Client

	// SampleRate is the rate the service outputs; 0 uses the voice's own
	// rate. The voice's audio is resampled to another rate.
	SampleRate int `validate:"gte=0"`
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// synthesisOptions are the synthesis controls one utterance is synthesized
// with; a nil one leaves the voice's own default.
type synthesisOptions struct {
	speakerID   *int
	lengthScale *float64
	noiseScale  *float64
	noiseWScale *float64
}

// voiceModel is a Piper voice loaded into the library.
type voiceModel interface {
	// synthesize synthesizes text with opts, handing yield each chunk of
	// samples, in [-1, 1] as the model produces them, and its rate.
	synthesize(ctx context.Context, text string, opts synthesisOptions,
		yield func(samples []float32, sampleRate int) error) error
	// close frees the voice.
	close()
}

// voiceLoader loads voices.
type voiceLoader interface {
	// loadVoice loads the voice model at modelPath, with its configuration at
	// modelPath + ".json" and espeak-ng's data at espeakDataPath ("" for the
	// library's default).
	loadVoice(modelPath, espeakDataPath string) (voiceModel, error)
}

// dependencies are what the service is built over: the library and the voice
// download, replaced in tests.
type dependencies struct {
	openLibrary   func(path string) (voiceLoader, error)
	downloadVoice func(ctx context.Context, cfg Config, voice, dir string) (string, error)
}

// nativeDependencies are libpiper and the voices repository.
func nativeDependencies() dependencies {
	return dependencies{
		openLibrary: func(path string) (voiceLoader, error) {
			lib, err := openLibrary(path)
			if err != nil {
				return nil, err
			}
			return lib, nil
		},
		downloadVoice: func(ctx context.Context, cfg Config, voice, dir string) (string, error) {
			d := &voiceDownloader{endpoint: huggingFace, client: cfg.HTTPClient}
			return d.downloadVoice(ctx, voice, dir, cfg.ForceRedownload)
		},
	}
}

// NewTTS builds a Piper TTS service, which runs the voice in process and
// streams the audio of each utterance as libpiper synthesizes it.
//
// The library is loaded, and the voice downloaded (the first time) and loaded
// here, so a missing library, a missing voice or a voice that fails to load is
// reported now rather than on the first utterance. A missing library is
// ErrLibraryNotAvailable.
//
// The synthesis controls (the speaker, the length scale and the two noise
// scales) are applied to each utterance, so a TTSUpdateSettingsFrame changing
// them takes effect from the next one. The voice is loaded once, so a change to
// it is stored but not applied, and is reported as not supported; NewHTTPTTS
// switches voices while running.
func NewTTS(cfg Config) (*tts.Base, error) {
	b, _, err := newTTS(cfg, nativeDependencies())
	return b, err
}

// newTTS builds the service over deps, returning its synthesizer alongside it.
func newTTS(cfg Config, deps dependencies) (*tts.Base, *synthesizer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	s := &synthesizer{live: newTTSSettings(cfg.Voice, synthesisConfig{
		speakerID:   cfg.SpeakerID,
		lengthScale: cfg.LengthScale,
		noiseScale:  cfg.NoiseScale,
		noiseWScale: cfg.NoiseWScale,
	})}

	dir := cfg.DownloadDir
	if dir == "" {
		var err error
		if dir, err = DefaultCacheDir(); err != nil {
			return nil, nil, err
		}
	}
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return nil, nil, fmt.Errorf("piper: %w", err)
	}

	voice, err := requireVoice(s.live.Voice)
	if err != nil {
		return nil, nil, err
	}

	// The library is loaded before the voice is downloaded, so a missing
	// install is reported without fetching a model it could not run.
	lib, err := deps.openLibrary(cfg.LibraryPath)
	if err != nil {
		return nil, nil, err
	}

	modelPath := filepath.Join(dir, voice+".onnx")
	if _, statErr := os.Stat(modelPath); statErr != nil {
		slog.Debug("downloading Piper model", "voice", voice)
		name, dlErr := deps.downloadVoice(context.Background(), cfg, voice, dir)
		if dlErr != nil {
			return nil, nil, dlErr
		}
		// A voice that was renamed is downloaded, and loaded, under its
		// current name.
		modelPath = filepath.Join(dir, name+".onnx")
	}

	slog.Debug("loading Piper model", "voice", voice, "path", modelPath)
	espeak := cfg.EspeakDataPath
	if espeak == "" {
		espeak = espeakDataDir(cfg.LibraryPath)
	}
	model, err := lib.loadVoice(modelPath, espeak)
	if err != nil {
		return nil, nil, err
	}
	rate, err := voiceSampleRate(modelPath + ".json")
	if err != nil {
		model.close()
		return nil, nil, err
	}
	slog.Debug("loaded Piper model", "voice", voice)

	s.voice = model
	s.voiceRate = rate
	s.sampleRate = cfg.SampleRate
	if s.sampleRate == 0 {
		s.sampleRate = rate
	}
	b := tts.New("PiperTTS", s)
	s.svc = b
	return b, s, nil
}

// voiceSampleRate is the rate the voice whose configuration is at path
// synthesizes at.
func voiceSampleRate(path string) (int, error) {
	raw, err := os.ReadFile(path) //nolint:gosec // the configuration of the voice loaded
	if err != nil {
		return 0, fmt.Errorf("piper: reading the voice configuration: %w", err)
	}
	var config struct {
		Audio struct {
			SampleRate int `json:"sample_rate"`
		} `json:"audio"`
	}
	if err := json.Unmarshal(raw, &config); err != nil {
		return 0, fmt.Errorf("piper: reading the voice configuration %s: %w", path, err)
	}
	if config.Audio.SampleRate <= 0 {
		return 0, fmt.Errorf("%w: %s", errNoSampleRate, path)
	}
	return config.Audio.SampleRate, nil
}

// unhandledWarner reports settings changes the service does not apply.
type unhandledWarner interface {
	WarnUnhandledUpdatedSettings(ctx context.Context, unhandled []string)
}

// synthesizer runs a Piper voice for the service.
type synthesizer struct {
	svc   unhandledWarner
	voice voiceModel
	// voiceRate is the rate the voice synthesizes at, which its audio is
	// labeled with; sampleRate is the rate the service outputs.
	voiceRate  int
	sampleRate int
	// live is the settings store, which the base updates on the frame that
	// changes it and which each synthesis reads.
	live TTSSettings
}

// SampleRate is the rate the service outputs. The voice's audio is labeled
// with the rate it is synthesized at, and resampled to this one.
func (s *synthesizer) SampleRate() int { return s.sampleRate }

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *synthesizer) Settings() any { return &s.live }

// UpdateSettings acts on a change, implementing tts.SettingsUpdater. The
// synthesis controls take effect on the next utterance; the rest are stored
// but not applied to the loaded voice, and are reported as such.
func (s *synthesizer) UpdateSettings(ctx context.Context, changed settings.Changed) error {
	if len(changed) == 0 {
		return nil
	}
	s.svc.WarnUnhandledUpdatedSettings(ctx, changed.Except(synthesisFields...))
	return nil
}

// Close releases the voice, implementing tts.Closer.
func (s *synthesizer) Close() error {
	s.voice.close()
	return nil
}

// RunTTS synthesizes text, yielding its audio a chunk at a time as libpiper
// produces it. A failure is yielded as an error frame.
func (s *synthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	defer slog.Debug("finished TTS", "service", "PiperTTS", "text", text)
	err := s.runTTS(ctx, text, yield)
	if err == nil || ctx.Err() != nil {
		return err
	}
	slog.Error("Piper TTS failed", "err", err)
	ef := frames.NewErrorFrame(fmt.Sprintf("Unknown error occurred: %v", err))
	ef.Err = err
	return yield(ef)
}

// runTTS is RunTTS, returning what fails.
func (s *synthesizer) runTTS(ctx context.Context, text string, yield func(f frames.Frame) error) error {
	opts := synthesisOptions{
		speakerID:   s.live.SpeakerID.Ptr(),
		lengthScale: s.live.LengthScale.Ptr(),
		noiseScale:  s.live.NoiseScale.Ptr(),
		noiseWScale: s.live.NoiseWScale.Ptr(),
	}
	tts.StartUsageMetrics(ctx, text)
	return s.voice.synthesize(ctx, text, opts, func(samples []float32, _ int) error {
		pcm := audioInt16(samples)
		if len(pcm) == 0 {
			return nil
		}
		return yield(frames.NewTTSAudioRawFrame(pcm, s.voiceRate, 1))
	})
}

// maxWavValue is the full scale of 16-bit PCM.
const maxWavValue = 32767

// audioInt16 is a chunk of the voice's samples as 16-bit little-endian PCM,
// the way Piper hands its audio over: scaled so the loudest sample reaches
// full scale (a silent chunk stays silent), clipped to [-1, 1], then scaled
// by 32767 and truncated toward zero.
func audioInt16(samples []float32) []byte {
	var peak float32
	for _, v := range samples {
		peak = max(peak, float32(math.Abs(float64(v))))
	}
	pcm := make([]byte, 2*len(samples))
	if peak < 1e-8 {
		return pcm
	}
	for i, v := range samples {
		x := min(max(v/peak, -1), 1)
		n := int16(min(max(x*maxWavValue, -maxWavValue), maxWavValue))
		pcm[2*i] = byte(n)
		pcm[2*i+1] = byte(uint16(n) >> 8)
	}
	return pcm
}

// Compile-time checks that the synthesizer takes settings the way the base
// applies them.
var (
	_ tts.SettingsUpdater = (*synthesizer)(nil)
	_ tts.Closer          = (*synthesizer)(nil)
	_ voiceLoader         = (*library)(nil)
)
