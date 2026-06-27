// Package piper is a text-to-speech provider for a local Piper HTTP server
// (python -m piper.http_server). The server accepts the text to speak as the
// request body and returns a WAV file; this provider strips the WAV header and
// streams the raw PCM samples downstream.
//
// Piper's sample rate depends on the voice the server was launched with (16 kHz
// for "low" voices, 22.05 kHz for "medium"). Set SampleRate to match the voice;
// the pipeline resamples to the output rate from there.
package piper

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/tts"
)

// defaultSampleRate is the rate of Piper's "medium" voices.
const defaultSampleRate = 22050

// emitChunk is the size of each PCM chunk pushed downstream.
const emitChunk = 4096

// errStatus is returned when the server responds with a non-200 status.
//
//nolint:gochecknoglobals // sentinel error
var errStatus = errors.New("piper: unexpected status")

// errBadWAV is returned when the response is not a parseable WAV file.
//
//nolint:gochecknoglobals // sentinel error
var errBadWAV = errors.New("piper: response is not a WAV file")

// Config configures the Piper TTS provider.
type Config struct {
	// BaseURL is the Piper HTTP server's synthesis endpoint (e.g.
	// http://localhost:5000). The text to speak is POSTed as the body. Required.
	BaseURL string `validate:"required,url"`
	// SampleRate is the PCM rate of the configured Piper voice; 0 uses 22050.
	SampleRate int
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// NewTTS builds a Piper TTS service.
func NewTTS(cfg Config) *tts.Base {
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	return tts.New("PiperTTS", &synthesizer{cfg: cfg, http: &http.Client{}})
}

type synthesizer struct {
	cfg  Config
	http *http.Client
}

// SampleRate reports the configured Piper voice rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// Synthesize POSTs text to the Piper server, strips the returned WAV header, and
// streams the PCM downstream in chunks.
func (s *synthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.BaseURL, bytes.NewReader([]byte(text)))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "text/plain")

	resp, err := s.http.Do(req) //nolint:gosec // request target is the configured local endpoint
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg)
	}

	wav, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	pcm, err := pcmFromWAV(wav)
	if err != nil {
		return err
	}
	for len(pcm) > 0 {
		n := min(emitChunk, len(pcm))
		if err := emit(pcm[:n]); err != nil {
			return err
		}
		pcm = pcm[n:]
	}
	return nil
}

// pcmFromWAV returns the bytes of the "data" chunk of a RIFF/WAVE file, walking
// the chunk list so it tolerates headers carrying extra chunks (e.g. LIST).
func pcmFromWAV(b []byte) ([]byte, error) {
	if len(b) < 12 || string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return nil, errBadWAV
	}
	off := 12
	for off+8 <= len(b) {
		id := string(b[off : off+4])
		size := int(binary.LittleEndian.Uint32(b[off+4 : off+8]))
		off += 8
		if size > len(b)-off {
			// Tolerate a streamed/placeholder size by taking the remainder.
			size = len(b) - off
		}
		if id == "data" {
			return b[off : off+size], nil
		}
		off += size
		if size%2 == 1 && off < len(b) {
			off++ // chunks are word-aligned
		}
	}
	return nil, errBadWAV
}
