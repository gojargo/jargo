// Package tts is the shared base for text-to-speech services. The base
// aggregates incoming text into sentences and sends each one on the turn's audio
// context, a queue that holds a turn's audio in the order it was generated and
// is drained downstream by a goroutine of its own. Bracketing a turn with
// TTSStarted/TTSStopped, sentence aggregation, and the HTTP response streaming
// helper live here; the context machinery is in audiocontext.go.
//
// A provider implements Synthesize and has its audio appended for it, or
// implements ContextSynthesizer and appends the audio itself as its receive loop
// reads it. Either way the audio reaches the pipeline through a context, so one
// sentence never waits on the one before it.
package tts

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"reflect"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/gojargo/jargo/audio/onset"
	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/expressive/formatter"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/telemetry/metrics"
	uctx "github.com/gojargo/jargo/utils/context"
	errs "github.com/gojargo/jargo/utils/errors"
	ttstext "github.com/gojargo/jargo/utils/text"
	"github.com/google/uuid"
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

// Synthesizer turns text into speech. SampleRate reports the PCM rate of the
// audio it produces.
//
// RunTTS synthesizes one unit of text and hands each frame it produces to yield,
// which routes them to the synthesis context they belong to. Most providers
// yield audio and nothing else, and can build those frames with PCMYielder;
// yielding is the contract so a provider that has more to report, such as an
// error mid-stream, can say so in the same ordered stream as its audio.
//
// contextID names the synthesis the text belongs to. A provider that keeps a
// server-side stream open across a turn needs it; one that answers each call on
// its own can ignore it.
type Synthesizer interface {
	SampleRate() int
	RunTTS(ctx context.Context, text, contextID string, yield func(f frames.Frame) error) error
}

// PCMYielder adapts a frame yield to a callback taking raw 16-bit mono PCM, for
// a provider whose output is audio and nothing else.
func PCMYielder(yield func(f frames.Frame) error, rate int) func(pcm []byte) error {
	return func(pcm []byte) error {
		if len(pcm) == 0 {
			return nil
		}
		return yield(frames.NewTTSAudioRawFrame(pcm, rate, 1))
	}
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
	// RunTTSTimed synthesizes like RunTTS and additionally reports word timing
	// via word: each call carries a batch of tokens in spoken order, with the
	// offset of each in seconds from the beginning of this synthesis. A batch
	// rather than one token at a time, because normalizing a token stream can
	// need the token before it, which the base cannot see one call at a time.
	RunTTSTimed(
		ctx context.Context,
		text, contextID string,
		yield func(f frames.Frame) error,
		word func(words []uctx.WordTiming, opts WordTimingOptions) error,
	) error
}

// WordTimingOptions says how a batch of word timings should be treated, for the
// shapes a provider's token stream can take.
type WordTimingOptions struct {
	// IncludesInterFrameSpaces marks each token as carrying whatever spacing
	// separates it from the ones around it, so a consumer joining them adds none
	// of its own. Set it for a language written without spaces between words,
	// Chinese and Japanese, where a provider reports tokens that already read as
	// continuous text and inserting spaces between them is wrong. Off by default:
	// a stream of words in a spaced language needs the separator supplied.
	//
	// Leave it off when PreMergeTokens is set. Merging produces clean word
	// strings, which is the spaced shape.
	IncludesInterFrameSpaces bool
	// PreMergeTokens merges punctuation- and space-only tokens into the word
	// before them, for a provider that reports those as tokens of their own
	// rather than attached to the word they belong to. Off by default: a
	// provider whose stream is already word-level needs no merging, and merging
	// it anyway would join tokens that were meant to stand apart.
	PreMergeTokens bool
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

// SettingsHolder is an optional interface a Synthesizer implements when part of
// what it was built with can change while the pipeline runs: the voice it speaks
// in, the language, the model. The value returned is the provider's own store, a
// pointer to a settings value, which an update is merged into.
type SettingsHolder interface {
	Settings() any
}

// SettingsUpdater is an optional interface a Synthesizer implements to act on a
// settings change, with what changed and what each field held before. A provider
// whose new settings only take effect on a fresh connection reconnects here,
// which it can do because it owns its connection. A Synthesizer that holds
// settings without implementing this still has them updated; it picks them up
// the next time it synthesizes.
type SettingsUpdater interface {
	SettingsHolder
	UpdateSettings(ctx context.Context, changed settings.Changed) error
}

// LanguageNamer is an optional interface a Synthesizer implements to name a
// language the way its provider does. Providers disagree on the codes, so a
// caller naming a language neutrally has it converted before it is stored,
// leaving the store holding the code the provider itself uses. Without that the
// stored value and the next update would be in different vocabularies, and a
// change would be reported for two spellings of the same language.
type LanguageNamer interface {
	ServiceLanguage(l language.Language) string
}

// TrailingSpaceRequirer is an optional interface a Synthesizer implements when
// its provider reads trailing punctuation aloud, saying "dot" for the full stop
// ending a sentence, unless the text it is given ends in a space. The Base
// appends one before handing the text over.
//
// It applies to sentence-sized text only. When the aggregator streams tokens the
// text's own whitespace is what separates the words, so a space appended to
// every token would break them apart.
type TrailingSpaceRequirer interface {
	// RequiresTrailingSpace reports whether text sent for synthesis has to end
	// in a space.
	RequiresTrailingSpace() bool
}

// Closer is an optional interface a Synthesizer implements when it holds a
// resource open across syntheses, such as a connection it reuses rather than
// redialing per sentence. The Base closes it when the pipeline tears down. A
// Synthesizer that does not implement it is unaffected.
type Closer interface {
	Close() error
}

// Starter is an optional interface a Synthesizer implements when it has setup to
// do before the first synthesis, such as dialing the connection it will reuse.
// The Base calls it when the pipeline starts — the counterpart to Closer at
// teardown — so the cost lands while the transport is still negotiating rather
// than in front of the first thing the bot says. It runs off the frame goroutine.
// A Synthesizer that does not implement it is unaffected.
type Starter interface {
	Start(ctx context.Context)
}

// Base is the shared TTS processor. It aggregates text into sentences,
// synthesizes each one, and routes the audio through an audio context so it
// reaches the pipeline in the order it was generated. See audiocontext.go.
type Base struct {
	*service.Base
	syn     Synthesizer
	meta    Metadata
	filters []ttstext.Filter
	// transforms reshape a unit for the provider after the filters have
	// normalized it. See TextTransform.
	transforms []TextTransformer
	// skippingPronunciations is whether pronunciation transforms are being
	// skipped, so the warning is logged once each time they start being
	// skipped, not for every sentence.
	skippingPronunciations bool
	// skipTypes name the ways of grouping text whose units are not spoken. See
	// SetSkipTextTypes.
	skipTypes []frames.TextType

	// appendTrailingSpace says the provider needs the text it synthesizes to end
	// in a space. See TrailingSpaceRequirer.
	appendTrailingSpace bool

	// aggregator groups streamed text into the units the provider is given.
	aggregator ttstext.Aggregator
	// skippedAggregator cuts model text marked to skip synthesis into
	// sentences, so a reply that is not spoken still reaches a client as the
	// bot's output. It is separate from aggregator because that one yields
	// single tokens when streaming them, and only spoken tokens are regrouped
	// into sentences afterwards, by the sequencer.
	skippedAggregator ttstext.Aggregator
	// sequencer keeps the frames of a synthesis in the order the text was
	// spoken, so the conversation records it that way.
	sequencer *uctx.AggregatedFrameSequencer

	// Audio contexts. serial orders playback across contexts; audioContexts
	// holds each open context's queue. See audiocontext.go.
	audioCtxMu    sync.Mutex
	audioContexts map[string]*audioContext
	// audioCtxSeq numbers contexts as they are created, so the open ones can be
	// listed in creation order.
	audioCtxSeq uint64
	// ttsContexts records what each context carries beyond its audio. It is read
	// on the drain goroutine, long after the call that set it, which is why the
	// answers are kept here rather than passed along with each frame.
	ttsContexts map[string]ttsContext
	// audioRemainders holds, per context, the trailing bytes of a sample split
	// across audio chunks. See alignAudioFrame.
	audioRemainders map[string]audioRemainder
	serial          *serialQueue
	ctxCancel       context.CancelFunc
	ctxWG           sync.WaitGroup
	// audioTornDown records that the queue was torn down rather than drained, so
	// a graceful end has nothing left to wait for.
	audioTornDown bool
	// playingContext is the context the drain goroutine is playing, or "" between
	// contexts. An interruption cuts it off only when it is interruptible.
	playingContext string
	// uninterruptibleLastPTS is when the last word of uninterruptible speech is
	// spoken. Those words outlive an interruption, so the next words are timed
	// after them.
	uninterruptibleLastPTS time.Duration
	// stopFrameTimeout is how long a context waits for more audio before it is
	// taken as finished.
	stopFrameTimeout time.Duration

	// yieldsSync records whether the provider answered the last call with its
	// audio, which is what says who closes the context out.
	yieldsSync bool

	// wordLastPTS is when the last word pushed downstream was spoken, so a
	// context starting while another is still playing times its words after it.
	wordLastPTS time.Duration
	// replyStartPTS is what wordLastPTS was when the current model response
	// started playing, which tells whether the response timed any words.
	replyStartPTS time.Duration

	// turnContext is the context every sentence of the current turn is sent on,
	// empty between turns. Reusing it is what keeps the provider from opening a
	// cold context per sentence.
	turnContext string

	// llmResponseStarted reports whether a model response is driving the speech.
	// It is what tells an utterance the service says on its own from one the
	// model composed: the first has no response around it to close the assistant
	// turn, so the service closes it itself.
	llmResponseStarted bool

	// Pausing frame handling while a turn's audio is generated. See pause.go.
	pauseStateMu sync.Mutex
	// pauseOpts is how the service was configured; pausing is off by default.
	pauseOpts PauseOptions

	// outMu guards how what leaves the service is shaped: the silence padding
	// each utterance and the transport stream the audio is addressed to.
	outMu       sync.Mutex
	silence     SilenceOptions
	destination string

	// Contexts that complete with no audio at all. See silentcontext.go.
	silentMu sync.Mutex
	// zeroAudioLimit is how many such contexts in a row write the service off;
	// 0 reports them without ever doing so.
	zeroAudioLimit int
	// zeroAudioRun is how many have completed in a row.
	zeroAudioRun int

	// textAgg times how long grouping text into sentences takes. See
	// textaggregation.go.
	textAgg textAggregationMetrics

	// tokens carries the whitespace bookkeeping a streamed token needs. See
	// textaggregation.go.
	tokens tokenSpacing

	// metaMu guards meta, which the model labeling the metrics is read from
	// and which a settings update can change while a turn is being measured.
	metaMu sync.Mutex

	// exprFormatter reads the expression tags an LLM writes, when the pipeline
	// has expressive conversation on, and is nil otherwise. See expression.go.
	exprFormatter *formatter.ExpressionFormatter
	// exprMu guards exprSettings, what the expression adapter asked to send
	// beside the latest request's text.
	exprMu       sync.Mutex
	exprSettings map[string]any
}

// model is the identifier the synthesis is measured and priced against.
func (b *Base) model() string {
	b.metaMu.Lock()
	defer b.metaMu.Unlock()
	return b.meta.Model
}

// New builds a TTS Base named name driven by syn. The concrete service passes
// itself as syn and embeds the returned Base.
func New(name string, syn Synthesizer) *Base {
	b := &Base{syn: syn, zeroAudioLimit: defaultZeroAudioContextLimit, stopFrameTimeout: defaultStopFrameTimeout}
	// Sentences are found in the language the service speaks, English when its
	// settings name none.
	b.aggregator = ttstext.NewSimpleAggregator(frames.TextTypeSentence, settingsLanguage(syn))
	b.skippedAggregator = ttstext.NewSimpleAggregator(frames.TextTypeSentence, settingsLanguage(syn))
	b.sequencer = uctx.NewAggregatedFrameSequencer(name, false)
	if d, ok := syn.(Describer); ok {
		b.meta = d.Metadata()
	}
	if t, ok := syn.(TrailingSpaceRequirer); ok {
		b.appendTrailingSpace = t.RequiresTrailingSpace()
	}
	b.Base = service.New(name, b)
	b.Events().Register(EventTTSRequest, false)
	if cs, ok := syn.(ContextSynthesizer); ok {
		cs.SetAudioContextHost(b)
	}
	return b
}

// Cleanup releases the Synthesizer's resources, when it holds any, and tears
// down the processor.
func (b *Base) Cleanup(ctx context.Context) error {
	b.stopAudioContexts()
	if c, ok := b.syn.(Closer); ok {
		_ = c.Close()
	}
	return b.Base.Cleanup(ctx)
}

// settingsLanguage is the language the provider's settings name, or "" when it
// keeps none.
func settingsLanguage(syn Synthesizer) string {
	holder, ok := syn.(SettingsHolder)
	if !ok {
		return ""
	}
	value, _ := settings.Get(holder.Settings(), "language")
	code, _ := value.(string)
	return code
}

// TextAggregationLanguage is the language sentence boundaries are found in,
// derived from the service's settings.
func (b *Base) TextAggregationLanguage() string { return b.aggregator.Language() }

// SetTextAggregator sets how streamed text is grouped into the units handed to
// the provider, replacing the default (sentences in the service's language), or
// one that aggregates by token to stream text through as it arrives. Call this
// before the pipeline starts.
func (b *Base) SetTextAggregator(a ttstext.Aggregator) {
	b.aggregator = a
	// The sequencer has to be told how the text reaching it was grouped. Given
	// tokens it assembles them back into sentences and opens a slot only once a
	// boundary is confirmed; given whole units it opens one per unit. Built here
	// rather than read at push time because the mode is fixed for the run.
	b.sequencer = uctx.NewAggregatedFrameSequencer(b.Name(), a.Type() == frames.TextTypeToken)
}

// EventTTSRequest fires just before each unit of text is handed to the
// synthesizer, carrying a TTSRequest.
//
//	events.On(svc.Events(), tts.EventTTSRequest,
//	    func(ctx context.Context, r tts.TTSRequest) { … })
const EventTTSRequest = "on_tts_request"

// TTSRequest is one unit of text on its way to the synthesizer.
type TTSRequest struct {
	// ContextID identifies the synthesis context the text belongs to, which is
	// how the audio it produces is grouped.
	ContextID string
	// Text is what the synthesizer is given, after the filters, the transforms
	// and whatever final shaping the provider needs.
	Text string
}

// TextTransform reshapes one unit of text on its way to the provider, and is
// told how that unit was aggregated so it can treat a sentence differently from
// a token or from the content of a matched pattern.
//
// It is for changes the provider needs and the conversation should not record:
// inserting a tag that asks for a word to be spelled out or said in a particular
// tone, or writing "@" as "at". The text that goes into the conversation is
// taken before any of this runs, so what is recorded is what the model wrote
// rather than what the provider had to be told.
//
// Returning an error abandons the unit: it produces no audio and the failure is
// reported as an application error. Speaking the untransformed text instead is
// not safe, because a transform may exist precisely to take something out.
type TextTransform func(ctx context.Context, text string, aggregatedBy frames.TextType) (string, error)

// TextTransformer pairs a transform with the text type whose units it applies
// to.
type TextTransformer struct {
	// TextType names the units this transform applies to. Use
	// frames.AnyTextType for every unit, whatever its type.
	TextType frames.TextType
	// AggregatedBy is the former name of TextType, read when TextType is empty.
	//
	// Deprecated: use TextType.
	AggregatedBy frames.TextType
	// Transform is the reshaping itself.
	Transform TextTransform

	// pronunciation marks a transform built by PronunciationTransformIPA, which
	// is skipped while the provider cannot read pronunciation markup.
	pronunciation bool
}

// PronunciationFormatter is an optional interface a Synthesizer implements to
// take pronunciation hints. FormatPronunciation renders a word's pronunciation,
// given in IPA, in the provider's markup, and reports false when the provider
// cannot use it. A Synthesizer that does not implement it takes none.
type PronunciationFormatter interface {
	FormatPronunciation(word, ipa string) (string, bool)
}

// PronunciationSupporter is an optional interface a Synthesizer implements when
// whether it reads pronunciation markup depends on its model or settings.
// Pronunciation transforms are skipped while SupportsPronunciations reports
// false, so their words are spoken as written. It is asked for every text sent,
// so it follows settings updates. A Synthesizer that does not implement it
// reads its markup always.
type PronunciationSupporter interface {
	SupportsPronunciations() bool
}

// PronunciationTransformIPA returns a text transform that makes this service say
// words as IPA describes, applied to every unit.
//
// Each word is replaced with the provider's FormatPronunciation output, so the
// same IPA works with any service that supports it. Words the provider cannot
// use are reported once and spoken as written. While the provider cannot read
// pronunciation markup at all (see PronunciationSupporter), the transform is
// skipped.
//
// Register it last among the text transformers. They run in order, and one that
// runs after it (stripping markdown or symbols, replacing text) can rewrite the
// markup it inserts, such as Cartesia's <<…>> blocks or an SSML <phoneme> tag,
// and break the hint:
//
//	pronounce := t.PronunciationTransformIPA(map[string]string{"Metformin": "mɛtˈfɔɹmɪn"})
//	t.SetTextTransformers(stripMarkdown, pronounce) // last, so nothing rewrites its markup
func (b *Base) PronunciationTransformIPA(pronunciations map[string]string) TextTransformer {
	format := func(string, string) (string, bool) { return "", false }
	if f, ok := b.syn.(PronunciationFormatter); ok {
		format = f.FormatPronunciation
	}
	name, _, _ := strings.Cut(b.Name(), "#")
	t := ttstext.NewPronunciationTransform(pronunciations, format, name)
	return TextTransformer{
		TextType: frames.AnyTextType,
		Transform: func(_ context.Context, text string, _ frames.TextType) (string, error) {
			return t.Apply(text), nil
		},
		pronunciation: true,
	}
}

// canApplyPronunciations reports whether the provider reads pronunciation markup
// with its current settings, warning once each time it stops.
func (b *Base) canApplyPronunciations() bool {
	supported := true
	if s, ok := b.syn.(PronunciationSupporter); ok {
		supported = s.SupportsPronunciations()
	}
	if !supported && !b.skippingPronunciations {
		slog.Warn("the service does not read pronunciation markup with its current settings, so "+
			"pronunciation hints are spoken as written", "service", b.Name(), "model", b.model())
	}
	b.skippingPronunciations = !supported
	return supported
}

// SetTextTransformers sets the transforms applied to each unit just before the
// provider is given it, replacing any set before. They run in order, after the
// text filters, each one seeing what the one before it produced.
//
// Replacing the list is also how a transform is withdrawn: Go function values
// cannot be compared, so there is no removing one by naming it.
func (b *Base) SetTextTransformers(ts ...TextTransformer) {
	b.transforms = make([]TextTransformer, len(ts))
	for i, t := range ts {
		b.transforms[i] = t.withTextType()
	}
}

// AddTextTransformer appends one transform to those already set, for a caller
// registering them as it builds the service rather than all at once.
func (b *Base) AddTextTransformer(t TextTransformer) {
	b.transforms = append(b.transforms, t.withTextType())
}

// withTextType fills TextType from the deprecated AggregatedBy when it is empty.
func (t TextTransformer) withTextType() TextTransformer {
	if t.TextType == "" {
		t.TextType = t.AggregatedBy //nolint:staticcheck // the deprecated field is honored
	}
	return t
}

// transformText applies the transforms whose type matches how this unit was
// aggregated. It reports false when one of them failed, which abandons the unit.
func (b *Base) transformText(
	ctx context.Context, text string, aggregatedBy frames.TextType,
) (string, bool) {
	for _, t := range b.transforms {
		if t.Transform == nil {
			continue
		}
		if t.TextType != aggregatedBy && t.TextType != frames.AnyTextType {
			continue
		}
		if t.pronunciation && !b.canApplyPronunciations() {
			continue
		}
		out, err := t.Transform(ctx, text, aggregatedBy)
		if err != nil {
			// The failure came from application code rather than the provider, so
			// it is reported as such and the service stays usable. The unit is
			// dropped rather than spoken as it stands: a transform may be there to
			// remove something, and saying it anyway would defeat it.
			b.PushError(ctx, fmt.Sprintf("transforming text for synthesis [%s]", text), err, false,
				processor.WithErrorCategory(errs.Application))
			return "", false
		}
		text = out
	}
	return text, true
}

// SetSkipTextTypes names the ways of grouping text whose units are not to
// be spoken. A unit grouped that way is passed downstream unsynthesized, in its
// place among the units around it, so a consumer recording the conversation
// still sees it and only the audio is missing.
//
// It is how a pattern-pair aggregator's own types are kept out of the speech: a
// block of code cut out under a "code" type reads terribly aloud but still
// belongs in the transcript.
func (b *Base) SetSkipTextTypes(types ...frames.TextType) {
	b.skipTypes = types
}

// SetSkipAggregatorTypes is the former name of SetSkipTextTypes.
//
// Deprecated: use SetSkipTextTypes.
func (b *Base) SetSkipAggregatorTypes(types ...frames.TextType) {
	b.SetSkipTextTypes(types...)
}

// skipsSynthesis reports whether units grouped this way are passed on unspoken.
func (b *Base) skipsSynthesis(t frames.TextType) bool {
	return slices.Contains(b.skipTypes, t)
}

// SetTextFilters sets the text-normalization filters applied to each sentence
// just before synthesis, a text.MarkdownFilter that strips Markdown structure
// being the case. Filters run in order.
//
// A filter changes the text itself. To reshape a unit for the provider alone,
// leaving the conversation's record of it untouched, register a transform with
// SetTextTransformers instead: a text.VoiceFormatter goes there.
// Call this before the pipeline starts; the filter set is not safe to change
// while it is running.
func (b *Base) SetTextFilters(filters ...ttstext.Filter) {
	b.filters = filters
}

// Setup opens the synthesizer's connection, for a synthesizer that dials one.
//
// Dialing here rather than on the StartFrame is what keeps it off the frame
// path: a pipeline sets its processors up at once, so several services connect
// together instead of one after another as the frame reaches them, and a service
// that cannot connect is reported and left unusable before the pipeline starts,
// in time for a switcher to move off it.
func (b *Base) Setup(ctx context.Context, s processor.Setup) error {
	if err := b.Base.Setup(ctx, s); err != nil {
		return err
	}
	b.setupExpression(s)
	if st, ok := b.syn.(Starter); ok {
		// The connection outlives this call, so it is not tied to a context that
		// ends with it.
		st.Start(context.WithoutCancel(ctx))
	}
	return nil
}

// ProcessFrame aggregates text into sentences and synthesizes them.
func (b *Base) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := b.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	// Text the LLM service stamped to skip synthesis is not spoken, but it still
	// belongs in the conversation, and a client still shows it. The response
	// brackets are stamped with it too, so the turn they describe is not opened
	// and closed here for speech that never happens. A reply that is not spoken
	// is queued behind the audio still playing, so it comes out after that
	// audio, its start, sentences and end in order.
	if skipsTTS(f) {
		return b.handleSkipped(ctx, f, dir)
	}
	switch fr := f.(type) {
	case *frames.LLMTextFrame:
		return b.handleText(ctx, &fr.TextFrame, f, dir)
	case *frames.TextFrame:
		return b.handleText(ctx, fr, f, dir)
	case *frames.TTSSpeakFrame:
		return b.handleSpeak(ctx, fr, dir)
	case *frames.BotBackchannelFrame:
		// Spoken like any TTSSpeakFrame. The frame's own interruptibility is
		// carried over first: a backchannel is uninterruptible unless made
		// otherwise, which the speak frame inside it does not know by its type.
		fr.TTSSpeakFrame.SetInterruptible(frames.Interruptible(fr))
		return b.handleSpeak(ctx, &fr.TTSSpeakFrame, dir)
	case *frames.LLMFullResponseStartFrame:
		b.setLLMResponding(true)
		// A new turn gets one context id, shared by all of its sentences.
		b.setTurnContextID(b.createContextID())
		if c, ok := b.syn.(TurnContextCreator); ok {
			c.OnTurnContextCreated(ctx, b.turnContextID())
		}
		// Through the serialization queue, so this frame is emitted only after any
		// context already queued has drained rather than racing ahead of it.
		return b.queueSerial(ctx, f, dir)
	case *frames.LLMFullResponseEndFrame, *frames.EndFrame:
		return b.handleTurnEnd(ctx, f, dir)
	case *frames.InterruptionFrame:
		// An interruption stops every clock this service is running, this one
		// included: what it was measuring will never finish.
		b.stopTextAggregationMetrics(ctx)
		b.handleInterruption(ctx)
		return b.PushFrame(ctx, f, dir)
	case *frames.CancelFrame:
		b.handleInterruption(ctx)
		b.stopAudioContexts()
		return b.PushFrame(ctx, f, dir)
	case *frames.StartFrame:
		return b.handleStart(ctx, f, dir)
	case *frames.TTSUpdateSettingsFrame:
		return b.handleSettingsUpdate(ctx, fr, dir)
	default:
		return b.queueSerial(ctx, f, dir)
	}
}

// handleSettingsUpdate applies an update addressed to this service, and leaves
// one meant for another service untouched for that one.
func (b *Base) handleSettingsUpdate(
	ctx context.Context, f *frames.TTSUpdateSettingsFrame, dir processor.Direction,
) error {
	if !f.TargetsService(b) {
		return b.PushFrame(ctx, f, dir)
	}
	b.updateSettings(ctx, f)
	return nil
}

// handleSkipped handles a frame stamped to skip synthesis: the brackets and
// sentences of a reply that is not spoken are queued behind the audio, and any
// other text, such as an IVR tag, is pushed on as it is.
func (b *Base) handleSkipped(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	switch fr := f.(type) {
	case *frames.LLMFullResponseStartFrame:
		return b.queueSerial(ctx, f, dir)
	case *frames.LLMTextFrame:
		return b.handleSkippedLLMText(ctx, fr, dir)
	case *frames.LLMFullResponseEndFrame:
		return b.handleSkippedResponseEnd(ctx, f, dir)
	default:
		return b.PushFrame(ctx, f, dir)
	}
}

// handleSkippedLLMText queues each sentence of model text that is not spoken.
func (b *Base) handleSkippedLLMText(ctx context.Context, f *frames.LLMTextFrame, dir processor.Direction) error {
	for _, agg := range b.skippedAggregator.Aggregate(f.Text) {
		if err := b.queueSkippedLLMText(ctx, agg, dir); err != nil {
			return err
		}
	}
	return nil
}

// handleSkippedResponseEnd queues the rest of a reply that is not spoken, then
// the end of the reply.
func (b *Base) handleSkippedResponseEnd(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if rest, ok := b.skippedAggregator.Flush(); ok {
		if err := b.queueSkippedLLMText(ctx, rest, dir); err != nil {
			return err
		}
	}
	return b.queueSerial(ctx, f, dir)
}

// queueSkippedLLMText queues a sentence of model text that is not spoken, as
// the bot's output: an AggregatedTextFrame marked as skipping synthesis, which
// RTVI sends as bot-output and the assistant aggregator writes to the
// conversation.
func (b *Base) queueSkippedLLMText(ctx context.Context, agg ttstext.Aggregation, dir processor.Direction) error {
	if strings.TrimSpace(agg.Text) == "" {
		return nil
	}
	f := b.aggregatedFrame(agg, "")
	skip := true
	f.SkipTTS = &skip
	f.SetTransportDestination(b.destinationName())
	return b.queueSerial(ctx, f, dir)
}

// skipsTTS reports whether the frame was stamped to bypass synthesis. Only the
// frames describing the model's output carry the stamp, and an unstamped frame
// is not one to skip.
func skipsTTS(f frames.Frame) bool {
	var skip *bool
	switch fr := f.(type) {
	case *frames.LLMTextFrame:
		skip = fr.SkipTTS
	case *frames.TextFrame:
		skip = fr.SkipTTS
	case *frames.LLMFullResponseStartFrame:
		skip = fr.SkipTTS
	case *frames.LLMFullResponseEndFrame:
		skip = fr.SkipTTS
	}
	return skip != nil && *skip
}

// CurrentSettings is the service's current settings, for code outside the
// service to read, or nil when its provider keeps none. They are still changed
// through a TTSUpdateSettingsFrame.
func (b *Base) CurrentSettings() any {
	if holder, ok := b.syn.(SettingsHolder); ok {
		return holder.Settings()
	}
	return nil
}

// updateSettings merges an update into the provider's own settings and lets it
// act on what changed. A provider whose new settings only take effect on a
// fresh connection reconnects for itself: unlike a transcription session, which
// the base owns, a synthesizer owns its own connection. With expressive
// conversation on, the service's metadata is broadcast again when the change
// alters what it can perform.
func (b *Base) updateSettings(ctx context.Context, f *frames.TTSUpdateSettingsFrame) {
	holder, ok := b.syn.(SettingsHolder)
	if !ok {
		slog.Warn("settings update for a service whose provider has none", "service", b.Name())
		return
	}
	store := holder.Settings()

	delta, ok, err := settings.Resolve(&f.ServiceUpdateSettingsFrame, store)
	if err != nil {
		b.PushError(ctx, "tts: settings update", err, false)
		return
	}
	if !ok {
		return
	}

	// The language as the update gave it, before it is named the provider's
	// way, is what sentence boundaries are found in. A language given no value
	// goes back to English.
	givenFields, _ := settings.Given(delta)
	given, hasLanguage := givenFields["language"]
	lang, _ := given.(string)

	// Naming the language the provider's way before applying is what keeps the
	// comparison honest: the store holds the provider's code, so a neutral name
	// meaning the same language must be converted first or it reads as a change
	// when nothing changed.
	b.nameLanguage(delta)

	// A setting such as the model can change what the service performs, which
	// is broadcast again with expressive conversation on.
	var capabilities *expressive.ExpressionCapabilities
	if b.exprFormatter != nil {
		capabilities = b.exprFormatter.Capabilities()
	}
	defer func() {
		if b.exprFormatter != nil && !reflect.DeepEqual(b.exprFormatter.Capabilities(), capabilities) {
			b.BroadcastServiceMetadata(ctx)
		}
	}()

	changed, err := settings.Apply(store, delta)
	if err != nil {
		b.PushError(ctx, "tts: settings update", err, false)
		return
	}
	if hasLanguage {
		// Applied with the settings, whether or not the stored value changed,
		// and without clearing what the aggregator has buffered.
		b.aggregator.SetLanguage(lang)
		b.skippedAggregator.SetLanguage(lang)
	}
	if len(changed) == 0 {
		return
	}
	slog.Info("updated settings", "service", b.Name(), "fields", changed.String())
	b.SettingsUpdated(ctx)

	if changed.Has("model") {
		// The model labels the synthesis this service reports and is what the
		// characters are priced against, so a model that changed mid-call has to
		// relabel what follows.
		name, _ := settings.Get(store, "model")
		model, _ := name.(string)
		b.metaMu.Lock()
		b.meta.Model = model
		b.metaMu.Unlock()
	}

	if updater, ok := b.syn.(SettingsUpdater); ok {
		if err := updater.UpdateSettings(ctx, changed); err != nil {
			b.PushError(ctx, "tts: settings update", err, false)
		}
	}
}

// nameLanguage rewrites a language the delta gives into the code the provider
// uses, when the provider says how. A code the provider does not recognize is
// left as it came, since it may be one the service accepts directly.
func (b *Base) nameLanguage(delta any) {
	namer, ok := b.syn.(LanguageNamer)
	if !ok {
		return
	}
	value, ok := settings.Get(delta, "language")
	if !ok {
		return
	}
	code, ok := value.(string)
	if !ok || code == "" {
		return
	}
	named := namer.ServiceLanguage(language.Language(code))
	if named == "" || named == code {
		return
	}
	if err := settings.SetNamed(delta, "language", named); err != nil {
		slog.Warn("naming the language the provider's way failed", "service", b.Name(), "err", err)
	}
}

// handleTurnEnd closes a turn out: an LLMFullResponseEndFrame ends the model's
// response, an EndFrame ends the pipeline. Both flush whatever text did not land
// on a sentence boundary and close the context it was sent on.
func (b *Base) handleTurnEnd(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	_, isEnd := f.(*frames.EndFrame)
	if isEnd {
		// The pipeline is ending. The serialization queue is shut down and
		// waited on first, so the audio a provider is still delivering, and
		// anything queued behind it, reaches the output before the frame that
		// stops the pipeline rather than being cut off by it.
		b.drainAudioContexts()
	}
	if err := b.flush(ctx); err != nil {
		return err
	}
	if b.exprFormatter != nil {
		b.exprFormatter.Reset()
	}
	// Streaming tokens, a sentence the response ended without punctuating is
	// still pending in the sequencer, waiting on a boundary that is not coming.
	// It is promoted now, or its words would never be spoken into the context.
	b.pushSequencerFrames(ctx, b.sequencer.Finalize(b.turnContextID()), b.turnContextID())
	b.endTagsOnlyReply(ctx, b.turnContextID())
	b.holdFramesWhileAudioContextsOpen()
	// The turn's context is done, so the whitespace opening the next one is its
	// own to drop again.
	b.setSentNonWhitespace(false)
	// A filter's state belongs to one reply, so it is cleared as an
	// interruption clears it: a code block one reply left open must not mute
	// the next.
	for _, f := range b.filters {
		if i, ok := f.(ttstext.InterruptibleFilter); ok {
			i.HandleInterruption()
		}
	}
	b.onTurnContextCompleted(ctx)
	if isEnd {
		return b.PushFrame(ctx, f, dir)
	}
	b.setLLMResponding(false)
	// Queued behind the audio contexts, so the end follows the response's audio
	// and text. It goes out whether or not the response produced any speech.
	return b.queueSerial(ctx, f, dir)
}

// queueSerial routes a downstream frame through the serialization queue so it is
// emitted in the order it arrived relative to the contexts already queued. A
// system frame, or anything going upstream, overtakes the queue by design.
func (b *Base) queueSerial(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if _, isSystem := f.(frames.SystemFrame); isSystem || dir != processor.Downstream {
		return b.PushFrame(ctx, f, dir)
	}
	b.audioCtxMu.Lock()
	serial := b.serial
	b.audioCtxMu.Unlock()
	if serial == nil {
		return b.PushFrame(ctx, f, dir)
	}
	serial.push(serialItem{frame: f})
	return nil
}

// createContextID returns the turn's context id, so every sentence of a turn is
// synthesized on one context, or a fresh one outside a turn.
func (b *Base) createContextID() string {
	if id := b.turnContextID(); id != "" {
		// Keep the open context from timing out while the next sentence is still
		// being generated.
		b.refreshAudioContext(id)
		return id
	}
	return uuid.NewString()
}

// turnContextID is the context the current turn is spoken on, or "" between
// turns. It is read by a provider's receive loop as well as the frame
// goroutine, which is why it goes through the lock.
func (b *Base) turnContextID() string {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	return b.turnContext
}

// setTurnContextID records the context the current turn is spoken on.
func (b *Base) setTurnContextID(id string) {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	b.turnContext = id
}

// onTurnContextCompleted closes the turn out once its text has all been sent.
// The audio for it may still be arriving: for a provider that streams on its own
// receive loop the context is removed when the provider marks it final, and for
// one that returns its audio inline the base closes it here.
func (b *Base) onTurnContextCompleted(ctx context.Context) {
	id := b.turnContextID()
	b.setTurnContextID("")
	if id == "" {
		return
	}
	if ctx.Err() != nil {
		// Interrupted while the turn's last sentence was still being synthesized.
		// The interruption tears the context down, and the words it was holding
		// back are meant to stay unspoken.
		//
		// The check is here because a goroutine cannot be stopped from outside:
		// canceling only cancels the context, so the call that was in flight
		// runs to its end and has to notice for itself. Closing the turn out here
		// would speak what the interruption was meant to cut off.
		return
	}
	if b.yieldingSync() && b.AudioContextAvailable(id) {
		stopped := frames.NewTTSStoppedFrame()
		stopped.ContextID = id
		b.AppendToAudioContext(id, stopped)
		b.RemoveAudioContext(id)
	}
	// Anything the provider is still holding is flushed before the context is
	// closed, so a provider that waits to be told generates what it has.
	if f, ok := b.syn.(AudioFlusher); ok && b.AudioContextAvailable(id) {
		f.FlushAudio(ctx, id)
	}
	if c, ok := b.syn.(TurnContextCompleter); ok && b.AudioContextAvailable(id) {
		c.OnTurnContextCompleted(ctx, id)
	}
}

// handleInterruption drops everything queued for playback. The audio is no
// longer wanted, and the provider is told so it stops generating into contexts
// nobody is listening to.
func (b *Base) handleInterruption(ctx context.Context) {
	b.setSentNonWhitespace(false)
	b.setLLMResponding(false)
	if b.aggregator != nil {
		b.aggregator.HandleInterruption()
	}
	b.skippedAggregator.HandleInterruption()
	if b.exprFormatter != nil {
		b.exprFormatter.Reset()
	}
	for _, f := range b.filters {
		if i, ok := f.(ttstext.InterruptibleFilter); ok {
			i.HandleInterruption()
		}
	}
	// Interruptible contexts are dropped, along with their place in the
	// sequencer and the turn they were spoken on. Uninterruptible speech carries
	// on, and the drain goroutine is restarted only if it was stopped.
	b.interruptAudioContexts(ctx)
	b.createAudioContextTask(ctx)
	// Uninterruptible speech still open holds frames, even when the reply it was
	// part of never ended to hold them. The contexts dropped never end on their
	// own, so frames held for them are released here.
	b.holdFramesWhileAudioContextsOpen()
	b.releaseFramesIfAudioContextsEnded()
}

// handleSpeak speaks fixed text immediately, bypassing sentence aggregation.
//
// The frame is consumed rather than forwarded: it is a request to speak, and
// what the pipeline behind this service is told is the speech it produced. The
// conversation is built from what was actually spoken, never from the text on
// its way to the synthesizer, so this utterance reaches it as TTSTextFrames, per
// word where the provider times them and as one whole unit where it does not.
// The caller's answer about whether it belongs there is carried to those frames,
// so an utterance the service says rather than the assistant (a phrase covering
// a tool call, a stall while something is fetched) stays out of the conversation
// when the caller asked for that.
func (b *Base) handleSpeak(ctx context.Context, fr *frames.TTSSpeakFrame, _ processor.Direction) error {
	appendToContext := fr.AppendToContext
	// With no model response driving this, nothing else will close the assistant
	// turn the speech opens, so the service closes it once the words are out.
	pushAssistantAggregation := appendToContext && !b.llmResponding()
	// A fixed utterance is independent of any LLM turn, so it gets a context of
	// its own that is opened and closed around it. Whether a turn was mid-flight
	// is saved and put back, so speaking this does not look like the end of one.
	saved := b.turnContextID()
	// The utterance opens a context of its own, so its leading whitespace is its
	// own to drop; whatever the interrupted turn had sent is put back after.
	savedSentNonWhitespace := b.sentNonWhitespace()
	b.setSentNonWhitespace(false)
	// The turn's context is cleared first, so the utterance gets a fresh id
	// rather than the one the turn around it is spoken on.
	b.setTurnContextID("")
	speakContextID := b.createContextID()
	// Recorded before the id becomes the turn's, so an interruption arriving
	// while uninterruptible speech is being synthesized already finds it
	// uninterruptible and leaves it in place.
	interruptible := frames.Interruptible(fr)
	b.setTTSContext(speakContextID, ttsContext{
		appendToContext:          appendToContext,
		pushAssistantAggregation: pushAssistantAggregation,
		interruptible:            interruptible,
	})
	b.setTurnContextID(speakContextID)
	if c, ok := b.syn.(TurnContextCreator); ok {
		c.OnTurnContextCreated(ctx, speakContextID)
	}
	// A fixed utterance was never aggregated, so it stands as one sentence and
	// the text is its own written form.
	textType := fr.TextType
	if textType == "" {
		textType = frames.TextTypeSentence
	}
	spoken := ttstext.Aggregation{Text: fr.Text, Type: textType}
	if err := b.pushSpeech(ctx, spoken, interruptible); err != nil {
		return err
	}
	// Streaming tokens, the utterance is held as a pending sentence with no
	// following token to confirm its boundary, so it is promoted here or it is
	// never spoken into the context.
	b.pushSequencerFrames(ctx, b.sequencer.Finalize(b.turnContextID()), b.turnContextID())
	b.onTurnContextCompleted(ctx)
	if !b.AudioContextAvailable(speakContextID) {
		b.deleteTTSContext(speakContextID)
	}
	b.setTurnContextID(saved)
	b.setSentNonWhitespace(savedSentNonWhitespace)
	// Inside an LLM turn, the end of the turn holds the next frames.
	if saved == "" {
		b.holdFramesWhileAudioContextsOpen()
	}
	return nil
}

// handleStart forwards the StartFrame, announces the service, and gives the
// provider its chance to set up before the first sentence.
func (b *Base) handleStart(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := b.PushFrame(ctx, f, dir); err != nil {
		return err
	}
	b.startAudioContexts(ctx)
	return nil
}

// handleText folds a text frame's text into the aggregator. The frame itself is
// not forwarded: what reaches the pipeline is the text the synthesizer actually
// spoke, as TTSTextFrames timed to the audio, so the conversation records what
// was said rather than what was generated. A consumer that wants the text as the
// model produced it watches for it where the model pushes it.
func (b *Base) handleText(ctx context.Context, tf *frames.TextFrame, _ frames.Frame, _ processor.Direction) error {
	// The model has started answering, which is when the wait for the first
	// sentence starts. A transcription is a distinct type and does not land here.
	b.startTextAggregationMetrics()
	return b.aggregate(ctx, tf.Text)
}

// aggregate folds text into the aggregator and synthesizes every unit it
// completes.
func (b *Base) aggregate(ctx context.Context, text string) error {
	if b.aggregator == nil {
		return nil
	}
	for _, agg := range b.aggregator.Aggregate(text) {
		if agg.Type != frames.TextTypeToken {
			// The first sentence is complete, which is what the aggregation clock
			// was waiting for. Later ones find it already stopped.
			b.stopTextAggregationMetrics(ctx)
		}
		if err := b.pushTTSFrames(ctx, agg, true, false); err != nil {
			return err
		}
	}
	return nil
}

// flush synthesizes text left buffered when a response ends without a boundary.
func (b *Base) flush(ctx context.Context) error {
	if b.aggregator == nil {
		return nil
	}
	rest, ok := b.aggregator.Flush()
	// The response is over. Stop the clock whether or not anything was left, for
	// a response that never completed a sentence at all.
	b.stopTextAggregationMetrics(ctx)
	if !ok || strings.TrimSpace(rest.Text) == "" {
		return nil
	}
	return b.pushTTSFrames(ctx, rest, true, false)
}

// setLLMResponding records whether a model response is driving the speech.
func (b *Base) setLLMResponding(v bool) {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	b.llmResponseStarted = v
}

// llmResponding reports whether a model response is driving the speech.
func (b *Base) llmResponding() bool {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	return b.llmResponseStarted
}

// setYieldsSync records whether the provider yielded its audio.
func (b *Base) setYieldsSync(v bool) {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	b.yieldsSync = v
}

// yieldingSync reports whether the provider answers with its audio.
func (b *Base) yieldingSync() bool {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	return b.yieldsSync
}

// wordPath reports whether the driving provider reports word timings, in which
// case the base emits playback-aligned TTSTextFrames for interruption-accurate
// context.
func (b *Base) wordPath() bool {
	_, ok := b.syn.(WordTimestamps)
	return ok
}

// PushFrame pushes a frame on, closing the assistant turn behind an utterance
// the service said on its own.
//
// It is done here rather than where the stop frame is built because there is
// more than one place that builds one, the base's own and a provider's, and
// every one of them passes through here.
func (b *Base) PushFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	b.applyContextInterruptible(f)
	if stopped, ok := f.(*frames.TTSStoppedFrame); ok {
		b.reportUnreportedExpression(ctx, stopped, dir)
		interruptible := frames.Interruptible(stopped)
		if stopped.ContextID != "" {
			if err := b.maybeCloseAssistantTurn(ctx, stopped.ContextID, interruptible); err != nil {
				return err
			}
		}
		// Ahead of the stop frame, so the padding is part of the utterance
		// rather than something arriving after it has been called finished.
		if err := b.pushTrailingSilence(ctx, interruptible); err != nil {
			return err
		}
	}
	b.stampDestination(f)
	if b.exprFormatter != nil {
		sounds, keep := b.reportExpression(f)
		for _, sound := range sounds {
			if err := b.PushFrame(ctx, sound, dir); err != nil {
				return err
			}
		}
		if !keep {
			return nil
		}
	}
	return b.Base.PushFrame(ctx, f, dir)
}

// maybeCloseAssistantTurn tells the aggregator to commit, for a context that
// asked for it when it was opened. The answer is taken rather than read, so the
// turn is closed once however many stop frames the context produces. The commit
// is as interruptible as the speech it closes.
func (b *Base) maybeCloseAssistantTurn(ctx context.Context, contextID string, interruptible bool) error {
	c, known := b.takeTTSContext(contextID)
	if !known || !c.pushAssistantAggregation {
		return nil
	}
	push := frames.NewLLMAssistantPushAggregationFrame()
	push.Base().SetInterruptible(interruptible)
	// The words of this utterance travel the transport's clock queue, which
	// holds each until its moment. A frame with no timing of its own takes the
	// other queue and would overtake them, closing the turn before the last
	// words had been said, so it is timed just past the last of them.
	if pts := b.lastWordPTS(); pts > 0 {
		push.Base().SetPTS(int64(pts) + 1)
	}
	return b.Base.PushFrame(ctx, push, processor.Downstream)
}

// trimIncomingWhitespace takes the whitespace off the front of a unit and
// reports whether what is left is worth speaking at all.
//
// What comes off depends on how the text was grouped. Streaming tokens, the
// whitespace opening a context attaches to nothing and goes, but once a token
// carrying something has been sent, whitespace is kept as written: it is what
// holds the words apart, and it shapes the prosody between them. Aggregating
// sentences, only leading newlines come off, and a unit that is nothing but
// whitespace is not worth speaking.
func (b *Base) trimIncomingWhitespace(text string, streaming bool) (string, bool) {
	if streaming {
		if !b.sentNonWhitespace() {
			text = strings.TrimLeftFunc(text, unicode.IsSpace)
		}
		return text, text != ""
	}
	text = strings.TrimLeft(text, "\n")
	return text, strings.TrimSpace(text) != ""
}

// keepsFilteredText reports whether what the filters and transforms left is
// still worth speaking, and records that this context has now carried
// something. The same question is asked again after them because a filter or a
// transform can strip a unit to nothing of its own, and that has to be treated
// like text that arrived so: a lone space is not sent to the synthesizer.
func (b *Base) keepsFilteredText(text string, streaming bool) bool {
	if !streaming {
		return strings.TrimSpace(text) != ""
	}
	if text == "" || (strings.TrimSpace(text) == "" && !b.sentNonWhitespace()) {
		return false
	}
	b.setSentNonWhitespace(true)
	return true
}

// aggregatedFrame announces one unit of text, before the audio describing it
// opens. A consumer that wants the text ahead of hearing it reads this: an RTVI
// client starts its segment from it, and the progress frames that follow refer
// back to it. It carries the text as written, before the filters, and does not
// itself go into the conversation, which is built from what was actually spoken.
func (b *Base) aggregatedFrame(
	agg ttstext.Aggregation, contextID string,
) *frames.AggregatedTextFrame {
	f := frames.NewAggregatedTextFrame(agg.Text, agg.Type)
	// The written form this unit was cut from. It differs from the text only
	// when the unit came from a matched pattern, where the text is the content
	// and the delimiters around it are what the model actually wrote.
	f.RawText = agg.Original()
	f.ContextID = contextID
	return f
}

// prepareText applies the last shaping the provider needs before it is given
// the text: a trailing space, for a provider that would otherwise read the
// punctuation ending a sentence aloud. See TrailingSpaceRequirer.
func (b *Base) prepareText(text string) string {
	if b.appendTrailingSpace && !b.isStreamingTokens() && !strings.HasSuffix(text, " ") {
		return text + " "
	}
	return text
}

// pushTTSFrames synthesizes one piece of text on the turn's audio context,
// opening the context if this is the turn's first sentence. original is the
// pre-filter text; the configured filters produce the text sent to the provider.
// interruptible says whether an interruption may discard this speech. What the
// context carries (see ttsContext) is recorded for it the first time, unless a
// TTSSpeakFrame recorded it already, and is stamped onto every frame emitted
// from it.
//
// The audio does not go downstream from here. It is appended to the context and
// pushed by the context's drain loop, so a provider whose audio arrives later on
// its own receive loop and one that returns it inline reach the pipeline the
// same way, in the order the audio was generated.
//
// whole says the text stands on its own, apart from any response, as a
// TTSSpeakFrame's does; it is only read with expressive conversation on.
func (b *Base) pushTTSFrames(ctx context.Context, agg ttstext.Aggregation, interruptible, whole bool) error {
	original := agg.Text
	streaming := b.isStreamingTokens()

	// The id is settled first, before anything can drop this unit, because
	// reading it is also what keeps an open context from timing out while the
	// turn is still being generated.
	contextID := b.createContextID()

	// A unit grouped a way that is not spoken goes straight on downstream, held
	// by the sequencer until whatever was spoken before it has been said, so it
	// lands in the order it was written rather than ahead of the audio it
	// follows.
	if b.skipsSynthesis(agg.Type) {
		skipped := b.aggregatedFrame(agg, contextID)
		// Marked so whatever is downstream knows the service left it out.
		skip := true
		skipped.SkipTTS = &skip
		b.pushSequencerFrames(ctx, b.sequencer.RegisterSkipped(skipped, contextID, b.destinationName()), contextID)
		return nil
	}

	// With expressive conversation on, the text an LLM writes its replies in is
	// read for expression tags. Other text, such as a pattern's match, is spoken
	// as it is.
	if b.exprFormatter != nil && expressiveTextType(agg.Type) {
		return b.speakExpression(ctx, agg, contextID, interruptible, whole)
	}

	text, worthSpeaking := b.trimIncomingWhitespace(original, streaming)
	if !worthSpeaking {
		return nil
	}

	// Reshaped for the provider before anything is announced. A transform that
	// fails, or that leaves nothing but whitespace, abandons the unit here,
	// before it is announced or any audio is opened for it, so the turn carries
	// on with this much unspoken rather than waiting on audio that is never
	// coming.
	prepared, ok := b.transformUnit(ctx, text, agg.Type)
	if !ok {
		return nil
	}
	return b.speakText(ctx, agg, contextID, spokenText{text: original, raw: writtenText(agg)}, prepared, interruptible)
}

// transformUnit turns text to speak into the text sent to the provider: the
// filters, then the transforms, then the last shaping the provider needs. It
// reports false when nothing is left to say or a transform failed.
func (b *Base) transformUnit(ctx context.Context, text string, by frames.TextType) (string, bool) {
	filtered := text
	for _, f := range b.filters {
		// A stateful filter is told the interruption is over just before it is
		// asked for text again, which is the point it can resume from.
		if i, ok := f.(ttstext.InterruptibleFilter); ok {
			i.ResetInterruption()
		}
		filtered = f.Filter(filtered)
	}
	transformed, ok := b.transformText(ctx, filtered, by)
	if !ok || !b.keepsFilteredText(transformed, b.isStreamingTokens()) {
		return "", false
	}
	// The last shaping before the provider sees the text. Everything measured
	// against the synthesis counts it, because it is what was synthesized.
	return b.prepareText(transformed), true
}

// speakText announces a unit, opens its context if it is the turn's first, and
// asks the provider to speak prepared, the text as the provider is sent it.
// said is the unit's text as it is reported spoken.
func (b *Base) speakText(
	ctx context.Context, agg ttstext.Aggregation, contextID string, said spokenText, prepared string,
	interruptible bool,
) error {
	aggregated := b.aggregatedFrame(agg, contextID)
	aggregated.WillBeSpoken = true
	// It is going to be spoken, so it does not go into the conversation here: the
	// text that lands there is what was actually said, which an interruption can
	// cut short. A unit that is never spoken is the other case, and keeps the
	// default, since nothing later will put it there.
	aggregated.AppendToContext = false
	if !interruptible {
		// Queued for serialization before its context exists, so it carries the
		// mark itself to survive an interruption.
		aggregated.SetInterruptible(false)
	}

	opening := !b.AudioContextAvailable(contextID)
	if opening {
		// Serialized so it lands immediately before the start of the context it
		// describes, rather than racing ahead of audio still draining from the
		// turn before.
		_ = b.queueSerial(ctx, aggregated, processor.Downstream)
	} else {
		b.AppendToAudioContext(contextID, aggregated)
	}

	// Recorded before the context opens, so the drain loop has the answers in
	// hand by the time the first frame of the context reaches it.
	tc := b.ttsContextOrDefault(contextID, ttsContext{appendToContext: true, interruptible: interruptible})
	appendToContext := tc.appendToContext

	if opening {
		b.CreateAudioContext(contextID)
		started := frames.NewTTSStartedFrame()
		started.ContextID = contextID
		b.AppendToAudioContext(contextID, started)
	}
	c := b.audioContextFor(contextID)
	if c == nil {
		return nil
	}
	// Announced before the provider is asked, with the text exactly as it will
	// be sent. It is the last point at which what the bot is about to say can be
	// seen in the form the synthesizer receives it, filters and transforms
	// included, which is what a caller logging or auditing speech wants.
	b.Events().Call(ctx, EventTTSRequest, b, TTSRequest{ContextID: contextID, Text: prepared})
	c.addText(prepared)
	b.pushSequencerFrames(ctx, b.sequencer.RegisterSpoken(
		aggregated, contextID, prepared, appendToContext, b.wordPath(), false, b.TextAggregationLanguage()),
		contextID)
	return b.runTTS(ctx, c, contextID, prepared, aggregated, said, appendToContext)
}

// runTTS asks the provider to speak one unit of text and routes what it yields
// to the synthesis context. A provider whose audio arrives later on its own
// receive loop yields nothing here and appends it itself. aggregated is the
// frame announcing the unit, which the spoken text is reported as part of.
func (b *Base) runTTS(
	ctx context.Context, c *audioContext, contextID, prepared string,
	aggregated *frames.AggregatedTextFrame, said spokenText, appendToContext bool,
) error {
	// Every frame the provider yields is routed to its context, so audio and
	// anything else it reports stay in the order it produced them.
	yielded := false
	yield := func(f frames.Frame) error {
		if _, audio := f.(*frames.TTSAudioRawFrame); audio {
			yielded = true
		}
		b.AppendToAudioContext(contextID, f)
		return nil
	}
	var err error
	usable := b.Usable()
	if !usable {
		// A service that can no longer work cannot synthesize anything, and one
		// that connects on demand would attempt a handshake per request. The
		// bookkeeping around this still runs, so the turn completes with no audio
		// rather than stalling. The text that goes unspoken is named: silence
		// from the bot is otherwise hard to trace back to the service that caused
		// it.
		slog.WarnContext(ctx, "service is no longer usable, not speaking",
			"service", b.Name(), "text", prepared)
	} else if wt, ok := b.syn.(WordTimestamps); ok {
		word := func(words []uctx.WordTiming, opts WordTimingOptions) error {
			b.AddWordTimestamps(contextID, words, opts)
			return nil
		}
		err = wt.RunTTSTimed(ctx, prepared, contextID, yield, word)
	} else {
		err = b.syn.RunTTS(ctx, prepared, contextID, yield)
	}
	if err != nil && ctx.Err() == nil {
		c.span.RecordError(err)
		b.PushError(ctx, "tts synthesis failed", err, false)
	}
	// Whether the provider answered here decides who closes the context: one that
	// yielded its audio is finished, one that did not is still delivering. A
	// service that can no longer work was never asked, so nothing is coming and
	// the base closes the context itself rather than leaving the turn open on
	// audio that will never arrive.
	b.setYieldsSync(yielded || !usable)
	b.pushSpokenText(ctx, aggregated, said, contextID, appendToContext)
	return nil
}

// spokenText is a unit's text as it is reported spoken: its words, and the text
// the LLM wrote, when that differs, such as a pattern's match with its
// delimiters or a unit with expression tags in it.
type spokenText struct {
	text string
	raw  string
}

// writtenText is a unit's text as the LLM wrote it: a pattern's match with its
// delimiters, and any other unit's text as it is.
func writtenText(agg ttstext.Aggregation) string {
	if agg.Type == frames.TextTypeSentence || agg.Type == frames.TextTypeToken {
		return agg.Text
	}
	return agg.Original()
}

// pushSpokenText queues the spoken text behind its audio, for a provider
// without word timings.
//
// With no word timings there is nothing to place the text against, so the
// whole unit goes in as one, carrying the text as written rather than what the
// filters made of it: the frame carries the text as the LLM wrote it, which the
// conversation records, as it does for a provider with word timings.
//
// This is the only thing that puts the turn into the conversation: the text
// the model produced is consumed here and never forwarded, and without word
// timings no per-word frames are produced either. It cannot depend on the
// provider having answered inline. One that delivers its audio later on its own
// receive loop yields nothing here, and gating on that left every one of its
// turns out of the context entirely.
//
// Streaming tokens, the sequencer does this instead as it promotes each
// sentence, since one token is not the unit the conversation should record.
func (b *Base) pushSpokenText(
	ctx context.Context, src *frames.AggregatedTextFrame, said spokenText, contextID string, appendToContext bool,
) {
	if b.wordPath() || b.isStreamingTokens() {
		return
	}
	f := frames.NewTTSTextFrame(said.text, src.TextType)
	f.RawText = said.raw
	f.SegmentID = src.ID()
	f.ContextID = contextID
	f.WillBeSpoken = true
	f.AppendToContext = appendToContext
	// Appended to the audio context, so it keeps its place. A unit that is only
	// tags may come before anything is spoken, with no audio context.
	if b.AudioContextAvailable(contextID) {
		b.AppendToAudioContext(contextID, f)
	} else {
		_ = b.PushFrame(ctx, f, processor.Downstream)
	}
	// The text frame is queued, so the slot is complete and whatever was held
	// behind it (a code block, say) can follow it, in order.
	b.pushSequencerFrames(ctx, b.sequencer.CompleteSpokenSlot(), contextID)
}

// emitTiming records the synthesis's time-to-first-byte, time-to-first-audible
// sample, processing time and character count to OpenTelemetry (always) and,
// when in-band metrics are enabled, downstream as a MetricsFrame for the RTVI
// client.
func (b *Base) emitTiming(ctx context.Context, chars int, m *ttfaMeter, processing time.Duration) {
	model := b.model()
	metrics.RecordProcessing(ctx, "tts", b.TypeName(), model, processing.Seconds())
	metrics.RecordTTSCharacters(ctx, b.TypeName(), model, int64(chars))
	if m.hadTTFB {
		metrics.RecordTTFB(ctx, "tts", b.TypeName(), model, m.ttfb.Seconds())
	}
	if m.hadTTFA {
		metrics.RecordTTFA(ctx, "tts", b.TypeName(), model, m.ttfa.Seconds())
	}
	if !b.MetricsEnabled() {
		return
	}
	base := frames.BaseMetricsData{Processor: b.Name(), Model: b.model()}
	data := []frames.MetricsData{
		frames.ProcessingMetricsData{BaseMetricsData: base, Value: processing},
		frames.TTSUsageMetricsData{BaseMetricsData: base, Value: chars},
	}
	if m.hadTTFB {
		data = append(data, frames.TTFBMetricsData{BaseMetricsData: base, Value: m.ttfb})
	}
	if m.hadTTFA {
		data = append(data, frames.TTFAMetricsData{
			BaseMetricsData: base,
			TTFA:            m.ttfa,
			TTFB:            m.ttfb,
			LeadingSilence:  m.leadingSilence,
		})
	}
	_ = b.PushFrame(ctx, frames.NewMetricsFrame(data...), processor.Downstream)
}

// ServiceMetadataFrame implements service.MetadataDescriber, describing this
// synthesizer to the rest of the pipeline. With expressive conversation on, a
// TTSMetadataFrame carries its expression adapter's capabilities; otherwise
// nothing is broadcast.
func (b *Base) ServiceMetadataFrame() frames.ServiceMetadata {
	if b.exprFormatter == nil {
		return nil
	}
	f := frames.NewTTSMetadataFrame(b.Name())
	f.Expression = b.ExpressionAdapter().Capabilities()
	return f
}

// ttfaMeter measures a synthesis's time-to-first-byte and time-to-first-audible
// sample from the emitted PCM. TTFA extends TTFB: once the first byte arrives it
// scans the leading audio for a sustained speech onset and reports TTFB plus the
// leading-silence duration.
type ttfaMeter struct {
	rate int
	// report says whether this synthesis is measured at all. It is false once
	// the pipeline has had the only measurement it asked for, and nothing is
	// timed or scanned from then on.
	report bool

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
	if !m.report {
		return
	}
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
// chunks. It is the shared body-reading loop for HTTP TTS providers. Pair it
// with PCMYielder to turn the chunks into frames.
func StreamResponse(client *http.Client, req *http.Request, emit func(pcm []byte) error) error {
	resp, err := client.Do(req) //nolint:gosec // request target is the service's configured endpoint
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return errs.NewHTTPStatusError(resp.StatusCode, fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg))
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

// CanGenerateMetrics reports that this service times synthesis and reports
// the result, so the pipeline counts it when it collects the processors that
// report metrics.
func (b *Base) CanGenerateMetrics() bool { return true }
