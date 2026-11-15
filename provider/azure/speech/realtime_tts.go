package speech

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"unicode"

	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
	uctx "github.com/gojargo/jargo/utils/context"
	ttstext "github.com/gojargo/jargo/utils/text"
)

// RealtimeTTSConfig configures the Azure AI Speech streaming TTS service.
type RealtimeTTSConfig struct {
	// APIKey is the Speech resource key, sent as Ocp-Apim-Subscription-Key.
	// Required.
	APIKey string `validate:"required"`
	// Region is the Speech resource region, e.g. "eastus" or "francecentral".
	// Required unless PrivateEndpoint is set, which takes precedence.
	Region string `validate:"required_without=PrivateEndpoint"`
	// PrivateEndpoint is a custom endpoint for the Speech resource, for Private
	// Link or a custom domain, e.g. https://my-resource.cognitiveservices.azure.com/.
	PrivateEndpoint string
	// Voice is the SSML voice name; empty uses "en-US-SaraNeural".
	Voice string
	// Language sets the SSML xml:lang; the zero value uses en-US. A base language
	// takes the locale Azure defaults it to.
	Language language.Language
	// SampleRate is the PCM output rate; empty uses 24 kHz. Must be one Azure
	// offers as raw PCM (8000, 16000, 22050, 24000, 44100, 48000).
	SampleRate int
	// Effect names the audio effect processor that compensates for playback
	// distortion on the device the audio is bound for, as SSML's effect
	// attribute on <voice>: "eq_car" for an enclosed vehicle, "eq_telecomhp8k"
	// for narrowband telephony. Empty leaves it off.
	Effect string
	// ForceLocale wraps the text in SSML's <lang> element, so a multilingual
	// voice speaks in Language rather than the language it reads out of the
	// text. Pair it with a language the voice speaks: it synthesizes nothing at
	// all for one it does not.
	ForceLocale bool
	// VoiceParameters sets the model parameters of an HD voice, as SSML's
	// semicolon-separated parameters attribute on <voice>. Empty leaves it off.
	VoiceParameters string
	// Style is the speaking style ("cheerful", "sad"); empty speaks in the
	// voice's own.
	Style string
	// StyleDegree is the intensity of the style, from 0.01 to 2.0.
	StyleDegree string
	// Role is the role the voice plays ("YoungAdultFemale").
	Role string
	// Rate, Pitch and Volume adjust the prosody; empty leaves each as it is.
	Rate   string
	Pitch  string
	Volume string
	// Emphasis is the level of emphasis ("strong", "moderate", "reduced").
	Emphasis string
	// TextAggregation is how streamed text is grouped before synthesis:
	// frames.TextTypeSentence (the zero value) synthesizes a sentence at a time,
	// frames.TextTypeToken each token as it arrives.
	TextAggregation frames.TextType
}

// Validate reports whether the configuration is usable.
func (c RealtimeTTSConfig) Validate() error { return validate.Struct(c) }

// withDefaults fills in what the caller left unset.
func (c RealtimeTTSConfig) withDefaults() RealtimeTTSConfig {
	if c.Voice == "" {
		c.Voice = defaultTTSVoice
	}
	if c.SampleRate == 0 {
		c.SampleRate = defaultTTSRate
	}
	return c
}

// NewRealtimeTTS builds an Azure AI Speech streaming TTS service. It speaks
// Azure's synthesis WebSocket protocol over a connection reused across
// sentences: each sentence is sent as an SSML document, and its audio and word
// boundaries stream back as they are generated, so playback starts before the
// sentence is synthesized and the words are reported as they are spoken.
// Unlike NewTTS, which posts each sentence over HTTP, it reports the words, so
// the conversation keeps only what was heard before an interruption.
func NewRealtimeTTS(cfg RealtimeTTSConfig) *tts.Base {
	b, _ := newRealtimeTTS(cfg)
	return b
}

// newRealtimeTTS builds the service and returns its synthesizer alongside it.
func newRealtimeTTS(cfg RealtimeTTSConfig) (*tts.Base, *realtimeSynthesizer) {
	cfg = cfg.withDefaults()
	if cfg.Region != "" && cfg.PrivateEndpoint != "" {
		slog.Warn("azurespeech: both a region and a private endpoint were given; the region is ignored")
	}
	s := &realtimeSynthesizer{
		cfg:               cfg,
		live:              newTTSSettings(cfg),
		audioQueue:        newQueue[audioItem](),
		wordBoundaryQueue: newQueue[*wordItem](),
	}
	endpoint, err := synthesisEndpoint(cfg.Region, cfg.PrivateEndpoint)
	if err != nil {
		slog.Error("azurespeech: invalid endpoint", "err", err)
	}
	s.speech = &speechSynthesizer{
		endpoint:       endpoint,
		apiKey:         cfg.APIKey,
		outputFormat:   outputFormat(cfg.SampleRate),
		bytesPerSecond: cfg.SampleRate * 2,
		handlers: synthesisHandlers{
			synthesizing: s.handleSynthesizing,
			wordBoundary: s.handleWordBoundary,
			completed:    s.handleCompleted,
			canceled:     s.handleCanceled,
		},
	}
	b := tts.New("AzureRealtimeTTS", s)
	s.base = b
	if cfg.TextAggregation == frames.TextTypeToken {
		b.SetTextAggregator(ttstext.NewTokenAggregator())
	}
	// Frames are held behind a reply until its audio has all played.
	b.SetPauseFrameProcessing(tts.PauseOptions{Enabled: true})
	return b, s
}

// audioItem is one entry on the audio queue: a chunk of audio, the error that
// canceled the synthesis, or the end of the synthesis.
type audioItem struct {
	audio []byte
	err   error
	end   bool
}

// wordItem is one word to report, with where it starts in the turn's audio, in
// seconds. A nil one on the queue is the end of the synthesis.
type wordItem struct {
	word      string
	timestamp float64
}

// realtimeSynthesizer synthesizes one sentence at a time, streaming its audio
// as it arrives. The synthesis reports its audio, its word boundaries and its
// end through handlers called from the goroutine reading the connection; the
// audio goes on a queue that RunTTS yields from, and the words on a queue a
// goroutine of their own reports to the base. The end of the synthesis is
// routed through the word queue, so every word is reported before the audio
// stream ends.
type realtimeSynthesizer struct {
	cfg    RealtimeTTSConfig
	live   *TTSSettings
	base   *tts.Base
	speech *speechSynthesizer

	audioQueue        *queue[audioItem]
	wordBoundaryQueue *queue[*wordItem]

	// taskMu guards the word processor.
	taskMu            sync.Mutex
	wordProcessorStop context.CancelFunc
	wordProcessorDone chan struct{}

	// mu guards everything below, which the handlers write from the goroutine
	// reading the connection.
	mu sync.Mutex
	// cumulativeAudioOffset is how much audio the turn's earlier sentences
	// produced, in seconds, where the next sentence's words are timed from.
	cumulativeAudioOffset float64
	// currentSentenceBaseOffset is cumulativeAudioOffset as it stood when the
	// current sentence started, which its word boundaries are timed from.
	currentSentenceBaseOffset float64
	// currentSentenceDuration is the length of the current sentence's audio, as
	// the completed synthesis reports it.
	currentSentenceDuration float64
	// currentSentenceMaxWordOffset is the latest word boundary of the current
	// sentence, which stands in for its length at 8 kHz, where the reported
	// length does not match the word boundaries.
	currentSentenceMaxWordOffset float64
	// lastWord is a word held back so the punctuation reported after it can be
	// merged into it, and lastTimestamp where it starts.
	lastWord      *string
	lastTimestamp float64
	// currentContextID is the context the words are reported on.
	currentContextID string
	// styleMarker is an HD voice's style marker still being reported, across
	// word boundaries.
	styleMarker *string
	// hdVoice and cjk are what the current sentence is spoken with: a voice
	// that performs style markers, and a language written without spaces.
	hdVoice bool
	cjk     bool
}

// SampleRate reports the PCM output rate.
func (s *realtimeSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// Metadata reports the voice synthesis is billed against.
func (s *realtimeSynthesizer) Metadata() tts.Metadata {
	return tts.Metadata{VoiceID: s.live.Voice.Or(s.cfg.Voice)}
}

// ExpressionAdapter implements tts.ExpressionAdapterProvider: style markers on
// HD voices, and none on the others.
func (s *realtimeSynthesizer) ExpressionAdapter() ttsadapter.ExpressionAdapter {
	return voiceExpressionAdapter(s.live.Voice.Or(""))
}

// Start starts the word processor, implementing tts.Starter.
func (s *realtimeSynthesizer) Start(ctx context.Context) {
	s.taskMu.Lock()
	defer s.taskMu.Unlock()
	if s.wordProcessorStop != nil {
		return
	}
	runCtx, stop := context.WithCancel(context.WithoutCancel(ctx))
	done := make(chan struct{})
	s.wordProcessorStop, s.wordProcessorDone = stop, done
	go func() {
		defer close(done)
		s.wordProcessorTaskHandler(runCtx)
	}()
}

// Close stops the word processor and closes the connection, implementing
// tts.Closer.
func (s *realtimeSynthesizer) Close() error {
	s.stopTasks()
	return s.speech.close()
}

// stopTasks stops the word processor. It is idempotent.
func (s *realtimeSynthesizer) stopTasks() {
	s.taskMu.Lock()
	stop, done := s.wordProcessorStop, s.wordProcessorDone
	s.wordProcessorStop, s.wordProcessorDone = nil, nil
	s.taskMu.Unlock()
	if stop != nil {
		stop()
		<-done
	}
}

// isCJKLanguage reports whether a language is Chinese, Japanese or Korean.
func isCJKLanguage(lang string) bool {
	lang = strings.ToLower(lang)
	for _, prefix := range []string{"zh", "ja", "ko", "cmn", "yue", "wuu"} {
		if strings.HasPrefix(lang, prefix) {
			return true
		}
	}
	return false
}

// isPunctuationOnly reports whether text is all punctuation and whitespace.
func isPunctuationOnly(text string) bool {
	if text == "" {
		return false
	}
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			return false
		}
	}
	return true
}

// handleWordBoundary reports a word boundary. Azure reports punctuation as
// boundaries of their own, and breaks CJK text into single characters or
// particles, so the boundaries are merged into words before they are reported.
func (s *realtimeSynthesizer) handleWordBoundary(evt wordBoundaryEvent) {
	word := evt.text
	sentenceRelative := float64(evt.audioOffset) / ticksPerSecond

	s.mu.Lock()
	defer s.mu.Unlock()
	// Timed from the offset the sentence started at, captured when it was sent.
	absolute := s.currentSentenceBaseOffset + sentenceRelative
	// The latest boundary times the sentence where the reported length cannot.
	if sentenceRelative > s.currentSentenceMaxWordOffset {
		s.currentSentenceMaxWordOffset = sentenceRelative
	}
	if word == "" {
		return
	}
	words := []string{word}
	if s.hdVoice {
		words = s.splitStyleMarkers(word)
	}
	for _, w := range words {
		if s.cjk {
			s.handleCJKWordBoundary(w, absolute)
		} else {
			s.handleNonCJKWordBoundary(w, absolute)
		}
	}
}

// splitStyleMarkers splits an HD voice's style and sound markers out of a word
// boundary's text. Azure reports a marker such as "[slow]" across boundaries
// and joined to the words around it, as "[" then "slow] A". Each marker is
// reported as a word of its own, "[slow]" then "A", so a marker unfinished at
// the end of one boundary carries over to the next. The caller holds mu.
func (s *realtimeSynthesizer) splitStyleMarkers(word string) []string {
	var pieces []string
	var text strings.Builder
	for _, r := range word {
		switch {
		case r == '[' && s.styleMarker == nil:
			if t := strings.TrimSpace(text.String()); t != "" {
				pieces = append(pieces, t)
			}
			text.Reset()
			marker := "["
			s.styleMarker = &marker
		case s.styleMarker != nil:
			marker := *s.styleMarker + string(r)
			if r == ']' {
				pieces = append(pieces, marker)
				s.styleMarker = nil
			} else {
				s.styleMarker = &marker
			}
		default:
			text.WriteRune(r)
		}
	}
	if t := strings.TrimSpace(text.String()); t != "" {
		pieces = append(pieces, t)
	}
	return pieces
}

// emitPendingWord queues the word held back, if there is one. The caller holds
// mu.
func (s *realtimeSynthesizer) emitPendingWord() {
	if s.lastWord != nil {
		s.wordBoundaryQueue.put(&wordItem{word: *s.lastWord, timestamp: s.lastTimestamp})
		s.lastWord = nil
		s.lastTimestamp = 0
	}
}

// handleCJKWordBoundary merges the characters of a language written without
// spaces into words, reported at natural breaks: punctuation or whitespace.
// The caller holds mu.
func (s *realtimeSynthesizer) handleCJKWordBoundary(word string, timestamp float64) {
	// The first word is held.
	if s.lastWord == nil {
		s.lastWord = &word
		s.lastTimestamp = timestamp
		return
	}
	// Punctuation is merged, and ends the word.
	if isPunctuationOnly(word) {
		merged := *s.lastWord + word
		s.lastWord = &merged
		s.emitPendingWord()
		return
	}
	// Whitespace ends the word before it, and starts a new one.
	if strings.TrimSpace(word) != word {
		s.emitPendingWord()
		s.lastWord = &word
		s.lastTimestamp = timestamp
		return
	}
	// Otherwise the characters go on being merged.
	merged := *s.lastWord + word
	s.lastWord = &merged
}

// handleNonCJKWordBoundary reports each word once the next has started, with
// the punctuation reported after it merged into it. The caller holds mu.
func (s *realtimeSynthesizer) handleNonCJKWordBoundary(word string, timestamp float64) {
	// Punctuation is merged into the word before it, which is still held.
	if isPunctuationOnly(word) && s.lastWord != nil {
		merged := *s.lastWord + word
		s.lastWord = &merged
		return
	}
	// A word reports the one before it, and is held in turn.
	if s.lastWord != nil {
		s.wordBoundaryQueue.put(&wordItem{word: *s.lastWord, timestamp: s.lastTimestamp})
	}
	s.lastWord = &word
	s.lastTimestamp = timestamp
}

// wordProcessorTaskHandler reports the queued words on the current context.
// The end of a synthesis is queued behind its words, and is passed on to the
// audio queue once they have all been reported, so the synthesis ends only
// after its last word.
func (s *realtimeSynthesizer) wordProcessorTaskHandler(ctx context.Context) {
	for {
		item, err := s.wordBoundaryQueue.get(ctx)
		if err != nil {
			return
		}
		if item == nil {
			// Every word is reported: the audio stream can end.
			s.audioQueue.put(audioItem{end: true})
			continue
		}
		s.mu.Lock()
		contextID := s.currentContextID
		s.mu.Unlock()
		if contextID != "" {
			s.base.AddWordTimestamps(contextID,
				[]uctx.WordTiming{{Word: item.word, Offset: item.timestamp}}, tts.WordTimingOptions{})
		}
	}
}

// handleSynthesizing queues a chunk of audio as it arrives.
func (s *realtimeSynthesizer) handleSynthesizing(audio []byte) {
	if len(audio) > 0 {
		s.audioQueue.put(audioItem{audio: audio})
	}
}

// handleCompleted records how long the sentence's audio is, reports the word
// still held, and queues the end of the synthesis behind the words.
func (s *realtimeSynthesizer) handleCompleted(result synthesisResult) {
	s.mu.Lock()
	if result.audioDuration > 0 {
		s.currentSentenceDuration = result.audioDuration.Seconds()
	}
	s.emitPendingWord()
	s.mu.Unlock()
	// Through the word queue, so the words are all reported before the audio
	// stream ends. The last word would otherwise come after the stop frame,
	// where observers miss it.
	s.wordBoundaryQueue.put(nil)
}

// handleCanceled ends the audio stream of a canceled synthesis: quietly when
// it was stopped by an interruption, with the error otherwise.
func (s *realtimeSynthesizer) handleCanceled(details cancellationDetails) {
	if details.reason == cancellationCancelledByUser {
		slog.Debug("azurespeech: speech synthesis canceled by user (interruption)")
		s.audioQueue.put(audioItem{end: true})
		return
	}
	s.audioQueue.put(audioItem{err: &synthesisCanceledError{details: details}})
}

// synthesisCanceledError is a synthesis that failed, with the reason Azure gave.
type synthesisCanceledError struct {
	details cancellationDetails
}

func (e *synthesisCanceledError) Error() string {
	msg := fmt.Sprintf("Azure TTS synthesis canceled: %s", e.details.reason)
	if e.details.errorDetails != "" {
		msg += " - " + e.details.errorDetails
	}
	return msg
}

// resetState drops what is carried between the sentences of a turn.
func (s *realtimeSynthesizer) resetState() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cumulativeAudioOffset = 0
	s.currentSentenceBaseOffset = 0
	s.currentSentenceDuration = 0
	s.currentSentenceMaxWordOffset = 0
	s.lastWord = nil
	s.lastTimestamp = 0
	s.currentContextID = ""
}

// OnTurnContextCompleted drops the turn's state once its text has all been
// spoken, implementing tts.TurnContextCompleter: the next turn's words are
// timed from zero.
func (s *realtimeSynthesizer) OnTurnContextCompleted(context.Context, string) {
	s.resetState()
}

// OnAudioContextInterrupted stops the synthesis in progress, so no more of its
// words are reported, and drops what it had already queued, implementing
// tts.AudioContextInterrupter.
func (s *realtimeSynthesizer) OnAudioContextInterrupted(ctx context.Context, _ string) {
	if err := s.speech.stopSpeaking(context.WithoutCancel(ctx)); err != nil {
		s.base.PushError(ctx, fmt.Sprintf("Unknown error occurred: %v", err), err, false)
	}
	s.resetState()
	s.audioQueue.drain()
	s.wordBoundaryQueue.drain()
}

// RunTTS synthesizes one sentence, yielding its audio as it arrives. It returns
// once the synthesis has ended, after every word of it has been reported.
func (s *realtimeSynthesizer) RunTTS(
	ctx context.Context, text, contextID string, yield func(f frames.Frame) error,
) error {
	// Audio left from a synthesis cut short would otherwise be taken for this
	// one, and its end would cut this one off.
	s.audioQueue.drain()

	s.mu.Lock()
	s.currentContextID = contextID
	// Captured before the synthesis starts, since the word boundaries are timed
	// from it as they arrive.
	s.currentSentenceBaseOffset = s.cumulativeAudioOffset
	s.currentSentenceDuration = 0
	s.currentSentenceMaxWordOffset = 0
	s.styleMarker = nil
	s.hdVoice = hdVoice(s.live.Voice.Or(""))
	s.cjk = isCJKLanguage(s.live.Language.Or(""))
	s.mu.Unlock()

	ssml := constructSSML(text, s.live.ssmlParams())
	s.speech.speakSSML(ctx, ssml)
	tts.StartUsageMetrics(ctx, text)

	var synthErr error
	for {
		item, err := s.audioQueue.get(ctx)
		if err != nil {
			// Interrupted: the interruption stops the synthesis.
			return nil //nolint:nilerr // the interruption is not a failure
		}
		if item.end {
			break
		}
		if item.err != nil {
			synthErr = item.err
			break
		}
		f := frames.NewTTSAudioRawFrame(item.audio, s.SampleRate(), 1)
		f.ContextID = contextID
		if err := yield(f); err != nil {
			return err
		}
	}

	// The next sentence's words are timed after this one's audio. At 8 kHz the
	// reported length does not match the word boundaries, so the latest
	// boundary stands in for it.
	s.mu.Lock()
	if s.SampleRate() == 8000 {
		s.cumulativeAudioOffset += s.currentSentenceMaxWordOffset
	} else {
		s.cumulativeAudioOffset += s.currentSentenceDuration
	}
	s.mu.Unlock()
	if synthErr != nil {
		// Reported with the reason the synthesis gave, not as a failed call.
		s.base.PushError(ctx, synthErr.Error(), synthErr, false)
	}
	return nil
}

// RunTTSTimed synthesizes like RunTTS, implementing tts.WordTimestamps. The
// words are reported by the word processor as the synthesis reaches them, so
// the word callback is not used here; implementing the interface is what puts
// the base on the word-aligned path.
func (s *realtimeSynthesizer) RunTTSTimed(
	ctx context.Context,
	text, contextID string,
	yield func(f frames.Frame) error,
	_ func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	return s.RunTTS(ctx, text, contextID, yield)
}

// queue is an unbounded FIFO queue. A put never blocks, so the goroutine
// reading the connection is never held up by a consumer.
type queue[T any] struct {
	mu    sync.Mutex
	items []T
	ready chan struct{}
}

func newQueue[T any]() *queue[T] { return &queue[T]{ready: make(chan struct{}, 1)} }

// put adds v to the back of the queue.
func (q *queue[T]) put(v T) {
	q.mu.Lock()
	q.items = append(q.items, v)
	q.mu.Unlock()
	q.signal()
}

// get takes the item at the front of the queue, waiting for one until ctx is
// done.
func (q *queue[T]) get(ctx context.Context) (T, error) {
	for {
		q.mu.Lock()
		if len(q.items) > 0 {
			v := q.items[0]
			q.items = q.items[1:]
			more := len(q.items) > 0
			q.mu.Unlock()
			if more {
				q.signal()
			}
			return v, nil
		}
		q.mu.Unlock()
		select {
		case <-ctx.Done():
			var zero T
			return zero, ctx.Err()
		case <-q.ready:
		}
	}
}

// drain empties the queue.
func (q *queue[T]) drain() {
	q.mu.Lock()
	q.items = nil
	q.mu.Unlock()
}

// signal wakes a waiting get.
func (q *queue[T]) signal() {
	select {
	case q.ready <- struct{}{}:
	default:
	}
}

// Compile-time checks that the synthesizer answers to what the base asks of it.
var (
	_ tts.WordTimestamps            = (*realtimeSynthesizer)(nil)
	_ tts.Starter                   = (*realtimeSynthesizer)(nil)
	_ tts.Closer                    = (*realtimeSynthesizer)(nil)
	_ tts.TurnContextCompleter      = (*realtimeSynthesizer)(nil)
	_ tts.AudioContextInterrupter   = (*realtimeSynthesizer)(nil)
	_ tts.Describer                 = (*realtimeSynthesizer)(nil)
	_ tts.ExpressionAdapterProvider = (*realtimeSynthesizer)(nil)
)
