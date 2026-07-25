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
	"time"
	"unicode/utf8"

	"github.com/gojargo/jargo/audio/onset"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/telemetry/metrics"
	"github.com/gojargo/jargo/telemetry/tracing"
	uctx "github.com/gojargo/jargo/utils/context"
	ttstext "github.com/gojargo/jargo/utils/text"
	"go.opentelemetry.io/otel/attribute"
)

// errStatus is returned when a provider responds with a non-200 status.
//
//nolint:gochecknoglobals // sentinel error
var errStatus = errors.New("tts: unexpected status")

// readChunk is the size of each audio read from an HTTP stream.
const readChunk = 4096

// ttfaMaxBufferSeconds bounds how much leading audio is buffered while scanning
// for a speech onset before giving up on the time-to-first-audio measurement.
const ttfaMaxBufferSeconds = 3

// Synthesizer turns text into speech audio. SampleRate reports the PCM rate of
// the audio it produces; Synthesize streams 16-bit mono PCM to emit.
type Synthesizer interface {
	SampleRate() int
	Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error
}

// WordTimestamps is an optional interface a Synthesizer may implement to report
// per-word timing aligned to the audio it produces. When the provider driving a
// Base implements it, the Base tracks word completion and emits playback-aligned
// TTSTextFrames that map each spoken word back to its original written form — so
// the assistant context records only what was actually spoken before an
// interruption, in its written form. A Synthesizer that does not implement this
// interface is unaffected: the Base pushes text and synthesizes audio exactly as
// before, and no TTSTextFrames are produced.
type WordTimestamps interface {
	Synthesizer
	// SynthesizeTimed streams 16-bit mono PCM to emit like Synthesize, and
	// additionally reports word timing via word(text, offset): text is a single
	// spoken token, offset its start time in seconds from the beginning of this
	// synthesis. Tokens are reported in spoken order; punctuation the provider
	// splits into its own token should be merged into the preceding word first
	// (see utils/context.MergePunctTokens).
	SynthesizeTimed(
		ctx context.Context,
		text string,
		emit func(pcm []byte) error,
		word func(text string, offset float64) error,
	) error
}

// Metadata describes a TTS service to the observability layer.
type Metadata struct {
	// Model is the provider's model identifier, e.g. "eleven_flash_v2_5". It
	// labels the metrics and is what a cost-tracking backend prices the
	// synthesis against, so it should be the identifier the provider bills
	// under.
	Model string
	// VoiceID is the provider's voice identifier, or "" when the service has no
	// notion of a voice separate from the model.
	VoiceID string
}

// Describer is an optional interface a Synthesizer implements to describe the
// model it synthesizes with. A Synthesizer that does not implement it still
// reports character usage on its spans; only the model label is missing, and
// with it the ability to price the synthesis.
type Describer interface {
	Metadata() Metadata
}

// pendingWord is a TTSTextFrame awaiting the point in the emitted audio where
// its word begins, so it is pushed downstream in step with playback.
type pendingWord struct {
	offset float64
	frame  *frames.TTSTextFrame
}

// Base is the shared TTS processor. It aggregates text into sentences and
// synthesizes each one.
type Base struct {
	*processor.Base
	syn         Synthesizer
	meta        Metadata
	aggregation string
	filters     []ttstext.Filter
}

// New builds a TTS Base named name driven by syn. The concrete service passes
// itself as syn and embeds the returned Base.
func New(name string, syn Synthesizer) *Base {
	b := &Base{syn: syn}
	if d, ok := syn.(Describer); ok {
		b.meta = d.Metadata()
	}
	b.Base = processor.New(name, b)
	return b
}

// SetTextFilters sets the text-normalization filters applied to each sentence
// just before synthesis — for example a text.VoiceFormatter that strips
// Markdown and spells out numbers, currency and dates. Filters run in order.
// Call this before the pipeline starts; the filter set is not safe to change
// while it is running.
func (b *Base) SetTextFilters(filters ...ttstext.Filter) {
	b.filters = filters
}

// ProcessFrame aggregates text into sentences and synthesizes them.
func (b *Base) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := b.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.LLMTextFrame:
		return b.handleText(ctx, &fr.TextFrame, f, dir)
	case *frames.TextFrame:
		return b.handleText(ctx, fr, f, dir)
	case *frames.TTSSpeakFrame:
		// Speak fixed text immediately, bypassing sentence aggregation.
		if b.wordPath() {
			// The spoken words drive the context via TTSTextFrames; don't also let
			// the aggregator record the whole fixed text (it would double-count).
			fr.AppendToContext = false
		}
		if err := b.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		return b.synthesize(ctx, fr.Text)
	case *frames.LLMFullResponseEndFrame:
		if err := b.flush(ctx); err != nil {
			return err
		}
		return b.PushFrame(ctx, f, dir)
	case *frames.StartFrame:
		if err := b.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		b.broadcastMetadata(ctx)
		return nil
	default:
		return b.PushFrame(ctx, f, dir)
	}
}

// handleText forwards a text frame downstream and buffers its text for
// synthesis. On the word-timestamp path the frame is excluded from the assistant
// context (the playback-aligned TTSTextFrames drive it instead) while still
// flowing to other consumers such as transcripts. tf is the (embedded) TextFrame
// carrying the flags; orig is the concrete frame to forward.
func (b *Base) handleText(ctx context.Context, tf *frames.TextFrame, orig frames.Frame, dir processor.Direction) error {
	if b.wordPath() {
		tf.AppendToContext = false
	}
	if err := b.PushFrame(ctx, orig, dir); err != nil {
		return err
	}
	return b.aggregate(ctx, tf.Text)
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

// wordPath reports whether the driving provider reports word timings, in which
// case the base emits playback-aligned TTSTextFrames for interruption-accurate
// context.
func (b *Base) wordPath() bool {
	_, ok := b.syn.(WordTimestamps)
	return ok
}

// synthesize requests speech for original and streams it downstream as audio.
// original is the pre-filter text; the configured filters produce the text sent
// to the provider. When the provider reports word timings, the base also emits
// TTSTextFrames mapping each spoken word back to original.
func (b *Base) synthesize(ctx context.Context, original string) error {
	filtered := original
	for _, f := range b.filters {
		filtered = f.Filter(filtered)
	}
	if strings.TrimSpace(filtered) == "" {
		return nil
	}
	ctx, span := tracing.Tracer().Start(ctx, "tts")
	defer span.End()
	rate := b.syn.SampleRate()
	// Providers bill per character, so count runes: len would charge an accented
	// character twice.
	chars := utf8.RuneCountInString(filtered)
	span.SetAttributes(
		attribute.String("tts.service", b.Name()),
		attribute.Int("tts.chars", chars),
		attribute.Int("tts.sample_rate", rate),
		attribute.String("gen_ai.output.type", "speech"),
	)
	if b.meta.VoiceID != "" {
		span.SetAttributes(attribute.String("gen_ai.request.voice", b.meta.VoiceID))
	}
	tracing.SetTTSUsage(ctx, b.meta.Model, chars)
	if err := b.PushFrame(ctx, frames.NewTTSStartedFrame(), processor.Downstream); err != nil {
		return err
	}
	start := time.Now()
	meter := ttfaMeter{rate: rate}
	emit := func(pcm []byte) error {
		if len(pcm) == 0 {
			return nil
		}
		meter.observe(pcm, start)
		return b.PushFrame(ctx, frames.NewTTSAudioRawFrame(pcm, rate, 1), processor.Downstream)
	}

	var synthErr error
	if wt, ok := b.syn.(WordTimestamps); ok {
		synthErr = b.synthesizeTimed(ctx, wt, original, filtered, rate, emit)
	} else {
		synthErr = b.syn.Synthesize(ctx, filtered, emit)
	}
	if synthErr != nil && ctx.Err() == nil {
		span.RecordError(synthErr)
		b.PushError(ctx, "tts synthesis failed", synthErr, false)
	}
	if meter.hadTTFB {
		span.SetAttributes(attribute.Int64("tts.ttfb_ms", meter.ttfb.Milliseconds()))
	}
	if meter.hadTTFA {
		span.SetAttributes(attribute.Int64("tts.ttfa_ms", meter.ttfa.Milliseconds()))
	}
	b.emitTiming(ctx, chars, &meter, time.Since(start))
	return b.PushFrame(ctx, frames.NewTTSStoppedFrame(), processor.Downstream)
}

// synthesizeTimed drives a word-timestamp provider. It tracks word completion
// against a segment map built from filtered (sent to the provider) and original
// (written form), and interleaves a TTSTextFrame for each spoken word into the
// audio stream at the point its audio begins to play. Words whose audio has not
// yet been emitted are held back, so an interruption (a canceled ctx) leaves
// only the spoken words downstream. When the synthesis finishes normally, any
// trailing words are released and any original text the provider under-reported
// is emitted to close out the sentence.
func (b *Base) synthesizeTimed(
	ctx context.Context,
	wt WordTimestamps,
	original, filtered string,
	rate int,
	emit func(pcm []byte) error,
) error {
	tracker := uctx.NewWordCompletionTracker(filtered, original, original)
	var (
		pending []pendingWord
		elapsed float64 // seconds of audio emitted so far
	)
	release := func(upTo float64) error {
		for len(pending) > 0 && pending[0].offset <= upTo {
			if err := b.PushFrame(ctx, pending[0].frame, processor.Downstream); err != nil {
				return err
			}
			pending = pending[1:]
		}
		return nil
	}
	timedEmit := func(pcm []byte) error {
		if len(pcm) == 0 {
			return nil
		}
		if err := emit(pcm); err != nil {
			return err
		}
		elapsed += float64(len(pcm)) / float64(rate*2) // 16-bit mono
		// Release every word whose audio has now been emitted, so each spoken
		// word's frame follows its audio downstream. A word is "spoken" once its
		// audio has played, so an interruption after this chunk keeps it.
		return release(elapsed)
	}
	word := func(text string, offset float64) error {
		if fr := trackWord(tracker, text); fr != nil {
			pending = append(pending, pendingWord{offset: offset, frame: fr})
		}
		return nil
	}

	err := wt.SynthesizeTimed(ctx, filtered, timedEmit, word)
	if ctx.Err() != nil {
		return err // interrupted: leave held-back words unspoken
	}
	for _, p := range pending {
		if perr := b.PushFrame(ctx, p.frame, processor.Downstream); perr != nil {
			return perr
		}
	}
	if rem := tracker.RemainingRawText(); rem != "" {
		f := frames.NewTTSTextFrame(rem)
		f.RawText = rem
		if perr := b.PushFrame(ctx, f, processor.Downstream); perr != nil {
			return perr
		}
	}
	return err
}

// trackWord advances the tracker by one spoken token and builds the
// TTSTextFrame to emit for it, or nil when the token produced no frame text. An
// intermediate token of a transformed span is marked to stay out of the context;
// the completing token carries the original written text as RawText.
func trackWord(tracker *uctx.WordCompletionTracker, text string) *frames.TTSTextFrame {
	tracker.AddWord(text)
	frameWord, ok := tracker.FrameWord()
	if !ok || frameWord == "" {
		return nil
	}
	f := frames.NewTTSTextFrame(frameWord)
	if raw, has := tracker.RawText(); has {
		f.RawText = raw
	}
	if tracker.Suppress() {
		f.AppendToContext = false
	}
	return f
}

// emitTiming records the synthesis's time-to-first-byte, time-to-first-audible
// sample, processing time and character count to OpenTelemetry (always) and,
// when in-band metrics are enabled, downstream as a MetricsFrame for the RTVI
// client.
func (b *Base) emitTiming(ctx context.Context, chars int, m *ttfaMeter, processing time.Duration) {
	metrics.RecordProcessing(ctx, "tts", b.Name(), b.meta.Model, processing.Seconds())
	metrics.RecordTTSCharacters(ctx, b.Name(), b.meta.Model, int64(chars))
	if m.hadTTFB {
		metrics.RecordTTFB(ctx, "tts", b.Name(), b.meta.Model, m.ttfb.Seconds())
	}
	if m.hadTTFA {
		metrics.RecordTTFA(ctx, "tts", b.Name(), b.meta.Model, m.ttfa.Seconds())
	}
	if !b.MetricsEnabled() {
		return
	}
	mf := frames.NewMetricsFrame(b.Name())
	mf.Processing = &processing
	mf.Characters = &chars
	if m.hadTTFB {
		mf.TTFB = &m.ttfb
	}
	if m.hadTTFA {
		mf.TTFA = &m.ttfa
		mf.LeadingSilence = &m.leadingSilence
	}
	_ = b.PushFrame(ctx, mf, processor.Downstream)
}

// broadcastMetadata pushes the TTS service's metadata frame downstream at
// pipeline start so downstream processors can discover the service.
func (b *Base) broadcastMetadata(ctx context.Context) {
	_ = b.PushFrame(ctx, frames.NewServiceMetadataFrame(b.Name()), processor.Downstream)
}

// ttfaMeter measures a synthesis's time-to-first-byte and time-to-first-audible
// sample from the emitted PCM. TTFA extends TTFB: once the first byte arrives it
// scans the leading audio for a sustained speech onset and reports TTFB plus the
// leading-silence duration.
type ttfaMeter struct {
	rate int

	hadTTFB bool
	ttfb    time.Duration

	active         bool
	buf            []byte
	hadTTFA        bool
	ttfa           time.Duration
	leadingSilence time.Duration
}

// observe folds one non-empty PCM chunk into the measurement, relative to the
// synthesis start time.
func (m *ttfaMeter) observe(pcm []byte, start time.Time) {
	if !m.hadTTFB {
		m.hadTTFB = true
		m.ttfb = time.Since(start)
		m.active = true
	}
	if !m.active {
		return
	}
	m.buf = append(m.buf, pcm...)
	if idx := onset.Detect(m.buf, m.rate, 1); idx >= 0 {
		m.leadingSilence = time.Duration(float64(idx) / float64(m.rate) * float64(time.Second))
		m.ttfa = m.ttfb + m.leadingSilence
		m.hadTTFA = true
		m.active = false
		m.buf = nil
		return
	}
	if len(m.buf) >= ttfaMaxBufferSeconds*m.rate*2 {
		// No onset within a bounded window of audio; stop scanning.
		m.active = false
		m.buf = nil
	}
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
