package eval

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/gojargo/jargo/audio"
	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/tts"
	"github.com/google/uuid"
)

// The user's voice: a caching TTS in the harness pipeline.
//
// In audio mode the harness synthesizes each user turn and streams it to the
// bot. CachingTTSService wraps a real TTS service and caches its audio on
// disk, keyed by service, voice, model, language, speed and text, so a scripted
// utterance is synthesized once and reused across runs and bots.
//
// Only local and HTTP services fit, since the wrapper drives the inner
// service's RunTTS directly.

const (
	// DefaultSampleRate is the rate the user's audio is streamed at when the
	// block names none.
	DefaultSampleRate = 16000
	// CacheSampleRate is the rate the cache stores audio at, and the built-in
	// services synthesize at, so one cached utterance serves every pipeline
	// rate.
	CacheSampleRate = 24000
)

// errWebsocketTTS reports a TTS whose RunTTS yields no audio, so it cannot be
// cached around.
var errWebsocketTTS = errors.New("eval: CachingTTSService supports only local or HTTP TTS services")

// errWAV reports a cache file that is not mono 16-bit PCM.
var errWAV = errors.New("eval: not a mono 16-bit PCM WAV file")

// errSynthesis is a synthesis the wrapped service reported failed.
var errSynthesis = errors.New("eval: user TTS failed")

// DefaultCacheDir is where cached user audio is stored when no directory is
// given: jargo/eval/tts under the user's cache directory.
func DefaultCacheDir() (string, error) {
	base, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("eval: no user cache directory for the TTS cache: %w", err)
	}
	return filepath.Join(base, "jargo", "eval", "tts"), nil
}

// TTSSampleRate is the sample rate a `user.speech:` block asks for (default
// 16 kHz).
func TTSSampleRate(cfg *SpeechConfig) int {
	if cfg == nil || cfg.SampleRate == 0 {
		return DefaultSampleRate
	}
	return cfg.SampleRate
}

// TTSCacheKey is a stable identity for a `user.speech:` block: its service,
// voice, model, language and speed, not its sample rate.
func TTSCacheKey(cfg *SpeechConfig) string {
	if cfg == nil {
		cfg = &SpeechConfig{}
	}
	lang := ""
	if s, ok := cfg.Language.(string); ok {
		lang = s
	} else if cfg.Language != nil && cfg.Language != false {
		lang = fmt.Sprint(cfg.Language)
	}
	// An absent speed and the service's default rate render the same audio.
	speed := ""
	if cfg.Speed != nil && *cfg.Speed != 1.0 {
		speed = strconv.FormatFloat(*cfg.Speed, 'f', -1, 64)
		if !strings.ContainsAny(speed, ".eE") {
			speed += ".0"
		}
	}
	return strings.Join([]string{strings.ToLower(cfg.Service), cfg.Voice, cfg.Model, lang, speed}, "\x00")
}

// CachingTTSConfig configures a CachingTTSService.
type CachingTTSConfig struct {
	// CacheKey is a stable identity for the inner service's config (see
	// TTSCacheKey); combined with the text, it keys the cache.
	CacheKey string
	// CacheDir is where cached audio is stored; empty uses DefaultCacheDir.
	CacheDir string
	// UseCache is whether cached audio is read and new audio written; nil
	// uses the cache.
	UseCache *bool
	// SampleRate is the rate the service's audio is played at; zero uses
	// DefaultSampleRate.
	SampleRate int
}

// CachingTTSService is a pipeline TTS that wraps a real service and caches its
// audio on disk.
//
// For each utterance it emits the cached audio when present, otherwise it runs
// the wrapped service, caches the audio, and forwards it. The cache stores
// audio at CacheSampleRate, and the audio is resampled to the service's rate as
// it plays. The inner service's start and close are forwarded so it works
// inside the pipeline. Only local and HTTP inner services are supported.
type CachingTTSService struct {
	*tts.Base
	syn *cachingSynthesizer
}

// NewCachingTTSService wraps inner, the local or HTTP synthesizer that does the
// synthesis, in a cache. A WebSocket-streaming synthesizer, or one that appends
// its audio to the service's contexts rather than yielding it, is refused: its
// RunTTS yields no audio to cache.
func NewCachingTTSService(inner tts.Synthesizer, cfg CachingTTSConfig) (*CachingTTSService, error) {
	switch inner.(type) {
	case tts.WebsocketService, tts.ContextSynthesizer:
		return nil, fmt.Errorf("%w, not the websocket-streaming %T; use an HTTP variant or a local service",
			errWebsocketTTS, inner)
	}
	syn := &cachingSynthesizer{
		inner:      inner,
		cacheKey:   cfg.CacheKey,
		cacheDir:   cfg.CacheDir,
		useCache:   cfg.UseCache == nil || *cfg.UseCache,
		sampleRate: cfg.SampleRate,
	}
	if syn.sampleRate == 0 {
		syn.sampleRate = DefaultSampleRate
	}
	return &CachingTTSService{Base: tts.New("CachingTTSService", syn), syn: syn}, nil
}

// Inner is the synthesizer the service wraps.
func (s *CachingTTSService) Inner() tts.Synthesizer { return s.syn.inner }

// SampleRate is the rate the service's audio is played at.
func (s *CachingTTSService) SampleRate() int { return s.syn.sampleRate }

// RunTTS emits the audio for text from the cache, or by driving the inner
// service: TTSStartedFrame, TTSAudioRawFrame and TTSStoppedFrame on a cache
// hit, the inner service's own frames on a miss.
func (s *CachingTTSService) RunTTS(
	ctx context.Context, text, contextID string, yield func(f frames.Frame) error,
) error {
	return s.syn.RunTTS(ctx, text, contextID, yield)
}

// Synthesize is the audio for text, from the cache or by driving the inner
// service, as 16-bit mono PCM at the service's sample rate.
func (s *CachingTTSService) Synthesize(ctx context.Context, text string) ([]byte, error) {
	ctx = withEvalPipeline(ctx, PipelineSpeech)
	var (
		pcm      []byte
		reported error
	)
	rate := s.syn.sampleRate
	err := s.syn.RunTTS(ctx, text, uuid.NewString(), func(f frames.Frame) error {
		switch fr := f.(type) {
		case *frames.TTSAudioRawFrame:
			pcm = append(pcm, fr.Audio...)
			rate = fr.SampleRate
		case *frames.ErrorFrame:
			// A service reports a failed synthesis as an error frame.
			if reported == nil {
				reported = fmt.Errorf("%w: %s", errSynthesis, fr.Error)
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if reported != nil {
		return nil, reported
	}
	return resampleUtterance(pcm, rate, s.syn.sampleRate)
}

// cacheFile is the WAV cache path for text, keyed by the config's identity and
// the text.
func (s *CachingTTSService) cacheFile(text string) (string, error) { return s.syn.cacheFile(text) }

// cachingSynthesizer is the synthesis of a CachingTTSService.
type cachingSynthesizer struct {
	inner      tts.Synthesizer
	cacheKey   string
	cacheDir   string
	useCache   bool
	sampleRate int
}

// SampleRate is the rate the service plays at; the base resamples the cache's
// audio to it.
func (c *cachingSynthesizer) SampleRate() int { return c.sampleRate }

// Start starts the inner service, when it has a start of its own.
func (c *cachingSynthesizer) Start(ctx context.Context) {
	if s, ok := c.inner.(tts.Starter); ok {
		s.Start(ctx)
	}
}

// Close closes the inner service, when it holds something open.
func (c *cachingSynthesizer) Close() error {
	if closer, ok := c.inner.(tts.Closer); ok {
		return closer.Close()
	}
	return nil
}

// RunTTS implements tts.Synthesizer: the cached audio, or the inner service's
// frames, its audio cached at CacheSampleRate.
func (c *cachingSynthesizer) RunTTS(
	ctx context.Context, text, contextID string, yield func(f frames.Frame) error,
) error {
	var cacheFile string
	if c.useCache {
		path, err := c.cacheFile(text)
		if err != nil {
			return err
		}
		cacheFile = path
		if hit, err := c.playCached(ctx, text, cacheFile, yield); hit || err != nil {
			return err
		}
	}

	// Cache miss: drive the inner service, capturing its audio as it is
	// forwarded.
	var captured []byte
	capturedRate := CacheSampleRate
	err := c.inner.RunTTS(ctx, text, contextID, func(f frames.Frame) error {
		switch fr := f.(type) {
		case *frames.TTSAudioRawFrame:
			captured = append(captured, fr.Audio...)
			capturedRate = fr.SampleRate
		case *frames.ErrorFrame:
			slog.WarnContext(ctx, "eval: TTS error during synthesis", "text", text, "error", fr.Error)
		}
		return yield(f)
	})
	if err != nil {
		return err
	}
	if cacheFile == "" || len(captured) == 0 {
		return nil
	}
	pcm, err := resampleUtterance(captured, capturedRate, CacheSampleRate)
	if err != nil {
		return err
	}
	return writeWAV(cacheFile, pcm, CacheSampleRate)
}

// playCached yields the cached audio for text, reporting whether there was
// any at the cache's rate. A file at another rate, or one that cannot be read,
// is regenerated.
func (c *cachingSynthesizer) playCached(
	ctx context.Context, text, cacheFile string, yield func(f frames.Frame) error,
) (bool, error) {
	if _, err := os.Stat(cacheFile); err != nil {
		return false, nil //nolint:nilerr // no cached audio is a miss, not an error
	}
	pcm, rate, err := readWAV(cacheFile)
	switch {
	case err != nil:
		slog.WarnContext(ctx, "eval: TTS cache read failed, regenerating", "file", cacheFile, "error", err)
		return false, nil
	case rate != CacheSampleRate:
		slog.InfoContext(ctx, "eval: cached TTS audio is at another rate, regenerating",
			"text", text, "rate", rate, "want", CacheSampleRate)
		return false, nil
	}
	for _, f := range []frames.Frame{
		frames.NewTTSStartedFrame(),
		frames.NewTTSAudioRawFrame(pcm, CacheSampleRate, 1),
		frames.NewTTSStoppedFrame(),
	} {
		if err := yield(f); err != nil {
			return true, err
		}
	}
	return true, nil
}

// cacheFile is the WAV cache path for text, keyed by the config's identity and
// the text, creating the cache directory when it is not there.
func (c *cachingSynthesizer) cacheFile(text string) (string, error) {
	dir := c.cacheDir
	if dir == "" {
		d, err := DefaultCacheDir()
		if err != nil {
			return "", err
		}
		dir = d
	}
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return "", fmt.Errorf("eval: cannot create the TTS cache directory: %w", err)
	}
	h := sha256.New()
	for _, part := range []string{c.cacheKey, text} {
		// Lower-cased so trivial casing differences hit the same slot.
		h.Write([]byte(strings.ToLower(part)))
		h.Write([]byte{0})
	}
	return filepath.Join(dir, hex.EncodeToString(h.Sum(nil))+".wav"), nil
}

// resampleUtterance resamples a whole utterance of 16-bit mono PCM, dropping a
// trailing odd byte.
func resampleUtterance(pcm []byte, inRate, outRate int) ([]byte, error) {
	pcm = pcm[:len(pcm)-len(pcm)%2]
	if len(pcm) == 0 || inRate == outRate {
		return pcm, nil
	}
	return resample.Resample(pcm, inRate, outRate, 1)
}

// writeWAV writes mono 16-bit PCM to a WAV file.
func writeWAV(path string, pcm []byte, sampleRate int) error {
	return os.WriteFile(path, audio.PCMToWAV(pcm, sampleRate, 1), 0o600)
}

// readWAV reads a mono 16-bit PCM WAV file, returning its samples and sample
// rate.
func readWAV(path string) (pcm []byte, sampleRate int, err error) {
	data, err := os.ReadFile(path) //nolint:gosec // the path is the cache's own file
	if err != nil {
		return nil, 0, err
	}
	if len(data) < 12 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("%w: %s: not a RIFF WAVE file", errWAV, path)
	}
	var formatSeen bool
	for off := 12; off+8 <= len(data); {
		id := string(data[off : off+4])
		size := int(binary.LittleEndian.Uint32(data[off+4 : off+8]))
		body := off + 8
		if size < 0 || body+size > len(data) {
			size = len(data) - body
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return nil, 0, fmt.Errorf("%w: %s: short fmt chunk", errWAV, path)
			}
			channels := binary.LittleEndian.Uint16(data[body+2 : body+4])
			bits := binary.LittleEndian.Uint16(data[body+14 : body+16])
			if channels != 1 {
				return nil, 0, fmt.Errorf("%w: %s: expected mono, got %d channels", errWAV, path, channels)
			}
			if bits != 16 {
				return nil, 0, fmt.Errorf("%w: %s: expected 16-bit, got %d-bit", errWAV, path, bits)
			}
			sampleRate = int(binary.LittleEndian.Uint32(data[body+4 : body+8]))
			formatSeen = true
		case wavDataChunk:
			if !formatSeen {
				return nil, 0, fmt.Errorf("%w: %s: data before fmt", errWAV, path)
			}
			return data[body : body+size], sampleRate, nil
		}
		off = body + size + size%2
	}
	return nil, 0, fmt.Errorf("%w: %s: no data chunk", errWAV, path)
}
