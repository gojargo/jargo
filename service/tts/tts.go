// Package tts is the shared base for text-to-speech services. The base
// aggregates incoming text into sentences, hands each sentence to a provider's
// Synthesizer, and brackets the resulting audio with TTSStarted/TTSStopped
// frames. Providers implement only Synthesize; sentence aggregation, the frame
// contract, and the HTTP response streaming helper live here.
package tts

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// errStatus is returned when a provider responds with a non-200 status.
//
//nolint:gochecknoglobals // sentinel error
var errStatus = errors.New("tts: unexpected status")

// readChunk is the size of each audio read from an HTTP stream.
const readChunk = 4096

// Synthesizer turns text into speech audio. SampleRate reports the PCM rate of
// the audio it produces; Synthesize streams 16-bit mono PCM to emit.
type Synthesizer interface {
	SampleRate() int
	Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error
}

// Base is the shared TTS processor. It aggregates text into sentences and
// synthesizes each one.
type Base struct {
	*processor.Base
	syn         Synthesizer
	aggregation string
}

// New builds a TTS Base named name driven by syn. The concrete service passes
// itself as syn and embeds the returned Base.
func New(name string, syn Synthesizer) *Base {
	b := &Base{syn: syn}
	b.Base = processor.New(name, b)
	return b
}

// ProcessFrame aggregates text into sentences and synthesizes them.
func (b *Base) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := b.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.LLMTextFrame:
		if err := b.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		return b.aggregate(ctx, fr.Text)
	case *frames.TextFrame:
		if err := b.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		return b.aggregate(ctx, fr.Text)
	case *frames.TTSSpeakFrame:
		// Speak fixed text immediately, bypassing sentence aggregation.
		if err := b.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		return b.synthesize(ctx, fr.Text)
	case *frames.LLMFullResponseEndFrame:
		if err := b.flush(ctx); err != nil {
			return err
		}
		return b.PushFrame(ctx, f, dir)
	default:
		return b.PushFrame(ctx, f, dir)
	}
}

// aggregate buffers text and synthesizes once a sentence is complete.
func (b *Base) aggregate(ctx context.Context, text string) error {
	b.aggregation += text
	if endOfSentence(b.aggregation) {
		sentence := b.aggregation
		b.aggregation = ""
		return b.synthesize(ctx, sentence)
	}
	return nil
}

// flush synthesizes any buffered text that didn't end on a sentence boundary.
func (b *Base) flush(ctx context.Context) error {
	if strings.TrimSpace(b.aggregation) == "" {
		b.aggregation = ""
		return nil
	}
	sentence := b.aggregation
	b.aggregation = ""
	return b.synthesize(ctx, sentence)
}

// synthesize requests speech for text and streams it downstream as audio.
func (b *Base) synthesize(ctx context.Context, text string) error {
	if strings.TrimSpace(text) == "" {
		return nil
	}
	if err := b.PushFrame(ctx, frames.NewTTSStartedFrame(), processor.Downstream); err != nil {
		return err
	}
	rate := b.syn.SampleRate()
	emit := func(pcm []byte) error {
		if len(pcm) == 0 {
			return nil
		}
		return b.PushFrame(ctx, frames.NewTTSAudioRawFrame(pcm, rate, 1), processor.Downstream)
	}
	if err := b.syn.Synthesize(ctx, text, emit); err != nil && ctx.Err() == nil {
		b.PushError(ctx, "tts synthesis failed", err, false)
	}
	return b.PushFrame(ctx, frames.NewTTSStoppedFrame(), processor.Downstream)
}

// StreamResponse issues req and streams the raw-PCM response body to emit in
// chunks. It is the shared body-reading loop for HTTP TTS providers.
func StreamResponse(client *http.Client, req *http.Request, emit func(pcm []byte) error) error {
	resp, err := client.Do(req) //nolint:gosec // request target is the service's configured endpoint
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg)
	}
	buf := make([]byte, readChunk)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			chunk := make([]byte, n)
			copy(chunk, buf[:n])
			if perr := emit(chunk); perr != nil {
				return perr
			}
		}
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
	}
}

// endOfSentence reports whether text ends on a sentence-terminating mark.
func endOfSentence(text string) bool {
	trimmed := strings.TrimRight(text, " \t\n\"')]}")
	if trimmed == "" {
		return false
	}
	switch trimmed[len(trimmed)-1] {
	case '.', '!', '?', ':', ';':
		return true
	}
	// Catch full-width CJK terminators.
	for _, suffix := range []string{"。", "！", "？", "…"} {
		if strings.HasSuffix(trimmed, suffix) {
			return true
		}
	}
	return false
}
