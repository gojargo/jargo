package rime

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"regexp"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/audio"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
	errs "github.com/gojargo/jargo/utils/errors"
)

const (
	// sttFallbackSampleRate is the rate audio is sent at when Rime does not
	// take the input rate.
	sttFallbackSampleRate = 16000
	// maxContextTermsBytes bounds the context terms joined with ", ".
	maxContextTermsBytes = 1024
	// sttDrainWindow is the longest a segment still streaming when the pipeline
	// ends can take to finish: connecting, being accepted, and completing.
	sttDrainWindow = sttConnectionTimeout + sttAcceptanceTimeout + sttCompletionTimeout
)

// sttSupportedSampleRates are the rates Rime takes audio at.
//
//nolint:gochecknoglobals // a fixed table
var sttSupportedSampleRates = []int{8000, 16000, 24000, 48000}

// controlMarker matches a model control marker, which a context term may not
// carry.
//
//nolint:gochecknoglobals // a compiled pattern
var controlMarker = regexp.MustCompile(`(?s)<\|.*?\|>`)

// Errors a settings value is refused with.
//
//nolint:gochecknoglobals // sentinel errors
var (
	errSTTModel                = errors.New("rime stt: model selection is not supported")
	errSTTLanguage             = errors.New("rime stt: language must be a nonempty BCP-47 string")
	errSTTMode                 = errors.New("rime stt: mode must be 'written' or 'verbatim'")
	errSTTContextTermsSequence = errors.New("rime stt: context_terms must be a sequence of strings")
	errSTTContextTermsEmpty    = errors.New("rime stt: context_terms must be nonempty strings")
	errSTTContextTermsMarkers  = errors.New("rime stt: context_terms cannot contain model control markers")
	errSTTContextTermsSize     = errors.New("rime stt: context_terms must fit in 1,024 UTF-8 bytes when joined")
	errSTTOutputContract       = errors.New(
		"rime stt: output_contract must be revised_hypotheses or append_fragments")
	errSTTExtra = errors.New("rime stt: additional recognition settings are not supported")
	// errSTTSettingsRejected wraps the reason an update was refused.
	errSTTSettingsRejected = errors.New("rime stt: settings update rejected")
	// errSTTSessionClosed ends the read of a session that was closed.
	errSTTSessionClosed = errors.New("rime stt: session closed")
)

// STTConfig configures the Rime STT service.
type STTConfig struct {
	// APIKey is the Rime API key. Required.
	APIKey string `validate:"required"`
	// SampleRate is the input audio rate; 0 uses the pipeline's input rate.
	// Audio is sent to Rime at that rate when Rime takes it (8, 16, 24 or 48
	// kHz) and resampled to 16 kHz otherwise.
	SampleRate int
	// Language is the spoken language as a BCP-47 tag, passed to Rime
	// unchanged. Rime documents "en" and "es". Empty uses "en".
	Language language.Language
	// Mode is "written" for written formatting or "verbatim" for verbatim
	// recognition; empty uses "written".
	Mode string `validate:"omitempty,oneof=written verbatim"`
	// ContextTerms are recognition hints, kept in order. Each term must be
	// nonempty, and the terms joined with ", " must fit in 1,024 UTF-8 bytes.
	ContextTerms []string
	// OutputContract is "revised_hypotheses" for replacement hypotheses or
	// "append_fragments" for append-only fragments; empty uses
	// "revised_hypotheses". Both produce cumulative interim transcripts and one
	// final.
	OutputContract string `validate:"omitempty,oneof=revised_hypotheses append_fragments"`
	// TTFSP99 overrides the transcript latency the turn strategies size their
	// wait by; nil uses stt.RimeTTFSP99, and zero means no wait.
	TTFSP99 *time.Duration
}

// Validate reports whether the configuration is usable.
func (c STTConfig) Validate() error {
	if err := validate.Struct(c); err != nil {
		return err
	}
	return validateSTTSettings(newSTTSettings(c))
}

// STTSettings is the Rime STT configuration that can change while the pipeline
// runs, through an STTUpdateSettingsFrame. A change applies to the stretches of
// speech that start after it.
//
// Rime selects the recognizer itself, so the model is always unset, and Rime
// takes no other recognition settings: an update naming a model or an extra
// setting has it ignored with a warning. An update with an invalid value is
// refused whole and reported as an error.
type STTSettings struct {
	settings.STT
	// Mode is "written" or "verbatim".
	Mode settings.Opt[string] `settings:"mode"`
	// ContextTerms are the recognition hints, kept in order.
	ContextTerms settings.Opt[[]string] `settings:"context_terms"`
	// OutputContract is "revised_hypotheses" or "append_fragments".
	OutputContract settings.Opt[string] `settings:"output_contract"`
}

// newSTTSettings is the settings store a configuration starts the service
// with.
func newSTTSettings(cfg STTConfig) *STTSettings {
	s := &STTSettings{}
	s.Model = settings.Cleared[string]()
	s.Language = settings.Set(string(language.English))
	if cfg.Language != "" {
		s.Language = settings.Set(string(cfg.Language))
	}
	s.Mode = settings.Set(modeWritten)
	if cfg.Mode != "" {
		s.Mode = settings.Set(cfg.Mode)
	}
	s.ContextTerms = settings.Set(slices.Clone(cfg.ContextTerms))
	if cfg.ContextTerms == nil {
		s.ContextTerms = settings.Set([]string{})
	}
	s.OutputContract = settings.Set(contractRevised)
	if cfg.OutputContract != "" {
		s.OutputContract = settings.Set(cfg.OutputContract)
	}
	return s
}

// validateSTTSettings reports whether a settings store holds what Rime takes.
func validateSTTSettings(s *STTSettings) error {
	if _, ok := s.Model.Value(); ok {
		return errSTTModel
	}
	if l, ok := s.Language.Value(); !ok || strings.TrimSpace(l) == "" {
		return errSTTLanguage
	}
	if m, ok := s.Mode.Value(); !ok || (m != modeWritten && m != modeVerbatim) {
		return errSTTMode
	}
	if err := validateContextTerms(s.ContextTerms); err != nil {
		return err
	}
	if c, ok := s.OutputContract.Value(); !ok || (c != contractRevised && c != contractAppendFragments) {
		return errSTTOutputContract
	}
	if len(s.Extra) > 0 {
		return errSTTExtra
	}
	return nil
}

// validateContextTerms reports whether the context terms are what Rime takes:
// nonempty strings with no model control markers, which fit in 1,024 UTF-8
// bytes once joined with ", ".
func validateContextTerms(o settings.Opt[[]string]) error {
	terms, ok := o.Value()
	if !ok {
		return errSTTContextTermsSequence
	}
	if slices.ContainsFunc(terms, func(term string) bool { return strings.TrimSpace(term) == "" }) {
		return errSTTContextTermsEmpty
	}
	if slices.ContainsFunc(terms, controlMarker.MatchString) {
		return errSTTContextTermsMarkers
	}
	if len(strings.Join(terms, ", ")) > maxContextTermsBytes {
		return errSTTContextTermsSize
	}
	return nil
}

// streamOptions is what a stream started with these settings is configured
// with, for audio at sampleRate.
func (s *STTSettings) streamOptions(sampleRate int) streamOptions {
	terms, _ := s.ContextTerms.Value()
	return streamOptions{
		language:       s.Language.Or(""),
		mode:           s.Mode.Or(modeWritten),
		contextTerms:   slices.Clone(terms),
		outputContract: s.OutputContract.Or(contractRevised),
		sampleRate:     sampleRate,
	}
}

// NewSTT builds a Rime STT service.
//
// Rime transcribes one utterance per stream, so the service needs VAD frames
// (from the user aggregator or a VAD processor): each VAD speech segment is
// streamed as its own request, starting with the audio from just before the
// VAD reported speech. Ending the segment's audio asks Rime for the final
// transcript. Finals are pushed in speech order, and one is finalized only
// while its segment is the user's latest speech.
//
// Rime's model runs out of context after about 90 to 110 seconds of speech, so
// a segment with no pause that long fails with no transcript.
func NewSTT(cfg STTConfig) *stt.StreamService {
	return newSTT(cfg, nil).svc
}

// newSTT builds the service's connector, creating each session's client with
// newClient, or on Rime's endpoint when it is nil.
func newSTT(cfg STTConfig, newClient func(apiKey string) (transcriber, error)) *sttConnector {
	c := &sttConnector{cfg: cfg, live: newSTTSettings(cfg), newClient: newClient}
	c.current = c.snapshot()
	c.svc = stt.NewStream("RimeSTT", c, cfg.SampleRate)
	return c
}

// sttConnector opens the sessions of a Rime STT service. A session is the
// connection to Rime for the length of the call, and streams each stretch of
// speech as its own request on it.
type sttConnector struct {
	cfg STTConfig
	// live is the settings store updates are merged into.
	live *STTSettings
	// svc is the service this connector drives, for its mute and usability and
	// for reporting the errors a segment meets. It is nil for a connector built
	// outside NewSTT.
	svc *stt.StreamService
	// newClient creates the client a session streams on; nil creates one on
	// Rime's endpoint.
	newClient func(apiKey string) (transcriber, error)

	mu sync.Mutex
	// current is a copy of the settings as they stand, which a segment takes
	// its own copy of when it starts. The store itself is written on the
	// goroutine applying updates, and a segment starts on another.
	current STTSettings
}

// snapshot copies the settings store.
func (c *sttConnector) snapshot() STTSettings {
	s := *c.live
	if terms, ok := s.ContextTerms.Value(); ok {
		s.ContextTerms = settings.Set(slices.Clone(terms))
	}
	return s
}

// segmentSettings is the settings a segment starting now streams with.
func (c *sttConnector) segmentSettings() STTSettings {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.current
}

// Metadata reports the transcript latency the turn strategies size their wait
// by.
func (c *sttConnector) Metadata() stt.Metadata {
	return stt.Metadata{TTFSP99: stt.TTFSP99Or(c.cfg.TTFSP99, stt.RimeTTFSP99)}
}

// Settings is the configuration a caller may change while the pipeline runs.
func (c *sttConnector) Settings() any { return c.live }

// ServiceLanguage passes the language tag to Rime unchanged.
func (c *sttConnector) ServiceLanguage(l language.Language) string { return string(l) }

// SendSampleRate is the input rate, or 16 kHz when Rime does not take it.
func (c *sttConnector) SendSampleRate(sampleRate int) int {
	if slices.Contains(sttSupportedSampleRates, sampleRate) {
		return sampleRate
	}
	return sttFallbackSampleRate
}

// PrepareSettings drops what Rime cannot take from an update, with a warning,
// and refuses an update whose values Rime would reject, whole, before any of it
// is applied.
func (c *sttConnector) PrepareSettings(delta any) error {
	var base *settings.Base
	switch d := delta.(type) {
	case *STTSettings:
		base = &d.Base
	case *settings.STT:
		base = &d.Base
	}
	if base != nil {
		var unsupported []string
		for name := range base.Extra {
			unsupported = append(unsupported, name)
		}
		if _, ok := base.Model.Value(); ok {
			unsupported = append(unsupported, "model")
		}
		if c.svc != nil {
			c.svc.WarnUnhandledUpdatedSettings(context.Background(), unsupported)
		}
		base.Model = settings.Opt[string]{}
		base.Extra = nil
	}
	candidate := c.snapshot()
	if _, err := settings.Apply(&candidate, delta); err != nil {
		return err
	}
	if err := validateSTTSettings(&candidate); err != nil {
		return fmt.Errorf("%w: %w", errSTTSettingsRejected, err)
	}
	return nil
}

// UpdateSettings takes the settings now in force for the segments that start
// afterwards. Rime is told the settings as each segment's stream opens, so
// there is no session to reopen.
func (c *sttConnector) UpdateSettings(context.Context, settings.Changed) (bool, error) {
	current := c.snapshot()
	c.mu.Lock()
	c.current = current
	c.mu.Unlock()
	return false, nil
}

// ClassifyError says what a Rime failure means.
func (c *sttConnector) ClassifyError(err error) errs.Category {
	if errors.Is(err, errSTTSettingsRejected) {
		return errs.Application
	}
	if re := asRimeError(err); re != nil {
		return errorCategory(re)
	}
	return errs.Unset
}

// errorCategory is the category of a Rime failure.
func errorCategory(err *rimeError) errs.Category {
	switch err.kind {
	case kindAuthentication:
		return errs.Authentication
	case kindPermission:
		return errs.Authorization
	case kindBilling:
		// Credits can be added, so the service stays usable and recovers once
		// they are.
		return errs.Quota
	case kindTimeout, kindUnavailable:
		return errs.Connectivity
	case kindInput:
		return errs.InvalidRequest
	case kindAudioFormat:
		return errs.Application
	case kindStream, kindResourceLimit, kindCancelled:
		return errs.Unknown
	default:
		return errs.Unknown
	}
}

// Connect creates the client the session streams its segments on, for audio at
// sampleRate. The connection is made by the first segment.
func (c *sttConnector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	newClient := c.newClient
	if newClient == nil {
		newClient = newRimeSTTClient
	}
	client, err := newClient(c.cfg.APIKey)
	if err != nil {
		return nil, err
	}
	// Never started, so it keeps only the audio from just before speech.
	preSpeech := audio.NewSpeechSegmentBuffer(audio.DefaultPreSpeech, 0)
	preSpeech.SetSampleRate(sampleRate)
	segmentCtx, cancel := context.WithCancel(ctx)
	return &sttSession{
		c:         c,
		ctx:       segmentCtx,
		cancel:    cancel,
		client:    client,
		sendRate:  sampleRate,
		preSpeech: preSpeech,
		wake:      make(chan struct{}, 1),
	}, nil
}

// muted reports whether the service is muted.
func (c *sttConnector) muted() bool { return c.svc != nil && c.svc.Muted() }

// usable reports whether the service can still do its job.
func (c *sttConnector) usable() bool { return c.svc == nil || c.svc.Usable() }

// reportError puts a failed request on the pipeline. Rime's messages can echo
// request data such as context terms, so the report names only the kind of
// failure and the request.
func (c *sttConnector) reportError(ctx context.Context, err error) {
	name := kindStream.String()
	request := ""
	if re := asRimeError(err); re != nil {
		name = re.kind.String()
		if re.requestID != "" {
			request = " (request " + re.requestID + ")"
		}
	}
	msg := "Rime STT request failed: " + name + request
	if c.svc == nil {
		slog.Error(msg, "err", err)
		return
	}
	category := errs.Unknown
	if re := asRimeError(err); re != nil {
		category = errorCategory(re)
	}
	c.svc.PushError(ctx, msg, err, false, processor.WithErrorCategory(category))
}

// frameLanguage is the language a transcript is labeled with: tag when it
// names a language jargo knows, and none otherwise.
func frameLanguage(tag string) string {
	if slices.Contains(language.All(), language.Language(tag)) {
		return tag
	}
	return ""
}

// segment is one VAD speech segment and the stream that transcribes it.
type segment struct {
	// number is the segment's position in the user's speech, from 1.
	number int
	// settings is what the segment's stream is configured with.
	settings STTSettings
	// previous is the segment before this one, until its final is published.
	previous *segment
	// audio is the audio for the stream, ended by close.
	audio *audioQueue
	// published is closed once this segment's final is published or
	// abandoned.
	published chan struct{}
}

// isPublished reports whether the segment's final is out.
func (s *segment) isPublished() bool {
	select {
	case <-s.published:
		return true
	default:
		return false
	}
}

// sessionItem is what a session hands the service: a result, or a failure to
// report.
type sessionItem struct {
	result stt.Result
	err    error
}

// sttSession is one Rime client for the length of the call. Rime transcribes one
// utterance per stream, so each VAD speech segment is streamed as its own
// request: the audio between segments is kept only as the pre-speech of the
// next one.
type sttSession struct {
	c *sttConnector
	// ctx is the segments' context, canceled when the session closes.
	ctx    context.Context //nolint:containedctx // the session's lifetime
	cancel context.CancelFunc
	client transcriber
	// sendRate is the rate of the audio the session is given.
	sendRate int
	// wake tells the read that something changed.
	wake chan struct{}
	wg   sync.WaitGroup

	mu    sync.Mutex
	queue []sessionItem
	// preSpeech holds the audio from just before speech, which starts the next
	// segment.
	preSpeech *audio.SpeechSegmentBuffer
	// segment is the segment audio is streamed to, nil between segments.
	segment *segment
	// lastSegment is the segment started last, which the next one follows.
	lastSegment  *segment
	segmentCount int
	// finalizeRequested records that the speech ended with a segment
	// streaming, so the latest segment's final is the one that closes it.
	finalizeRequested bool
	// usage is the audio submitted to Rime since it was last reported.
	usage int64
	// running counts the segments whose streams have not finished.
	running int
	// stopping is set when the pipeline ends gracefully: the segments in
	// progress finish, and no new one starts.
	stopping bool
	closed   bool
}

// signal wakes the read.
func (s *sttSession) signal() {
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

// enqueue hands an item to the read.
func (s *sttSession) enqueue(item sessionItem) {
	s.mu.Lock()
	s.queue = append(s.queue, item)
	s.mu.Unlock()
	s.signal()
}

// Send streams audio within a speech segment, and keeps the audio just before
// one. The audio kept has been through the same resampling as the audio
// streamed, so a segment's audio stays continuous where its pre-speech ends.
func (s *sttSession) Send(pcm []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	if s.segment != nil {
		s.usage += int64(len(pcm))
		s.segment.audio.put(pcm)
		return nil
	}
	s.preSpeech.Append(pcm)
	return nil
}

// TakeAudioUsage returns the audio submitted to Rime since it was last called:
// the segments' audio, their pre-speech included.
func (s *sttSession) TakeAudioUsage() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := s.usage
	s.usage = 0
	return n
}

// SpeechStarted starts a segment, streamed as its own request.
func (s *sttSession) SpeechStarted() {
	s.mu.Lock()
	// A finalize from the speech before is not this one's.
	s.finalizeRequested = false
	s.endSegmentLocked()
	stopped := s.closed || s.stopping
	s.mu.Unlock()
	if stopped || s.c.muted() || !s.c.usable() {
		return
	}
	segmentSettings := s.c.segmentSettings()

	s.mu.Lock()
	defer s.mu.Unlock()
	s.segmentCount++
	seg := &segment{
		number:    s.segmentCount,
		settings:  segmentSettings,
		previous:  s.lastSegment,
		audio:     newAudioQueue(),
		published: make(chan struct{}),
	}
	s.segment, s.lastSegment = seg, seg
	if preSpeech := s.preSpeech.Stop(); len(preSpeech) > 0 {
		s.usage += int64(len(preSpeech))
		seg.audio.put(preSpeech)
	}
	s.running++
	s.wg.Go(func() { s.transcribe(seg) })
}

// Finalize ends the segment, which asks Rime for its final transcript. With a
// segment streaming, the finalize is recorded with the service, so the
// segment's final confirms it.
func (s *sttSession) Finalize() error {
	s.mu.Lock()
	streaming := s.segment != nil
	if streaming && s.c != nil && s.c.svc != nil {
		// Recorded before the segment ends, so its final has a request to
		// confirm.
		s.c.svc.RequestFinalize()
	}
	if streaming {
		s.finalizeRequested = true
	}
	s.endSegmentLocked()
	s.mu.Unlock()
	return nil
}

// Stop ends the segment in progress so it can finish, and starts no other: the
// pipeline is ending gracefully.
func (s *sttSession) Stop() error {
	s.mu.Lock()
	s.endSegmentLocked()
	s.stopping = true
	s.mu.Unlock()
	s.signal()
	return nil
}

// Drain gives the segments still streaming when the pipeline ends gracefully
// the time they can take to finish. The read ends once they all have.
func (s *sttSession) Drain() (time.Duration, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.stopping {
		return 0, nil
	}
	return sttDrainWindow, nil
}

// Close stops at once, dropping the segments in progress, and closes the
// client.
func (s *sttSession) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	s.endSegmentLocked()
	s.mu.Unlock()
	s.cancel()
	s.wg.Wait()
	s.signal()
	return s.client.close()
}

// endSegmentLocked ends the audio of the segment streaming, if any.
func (s *sttSession) endSegmentLocked() {
	if s.segment != nil {
		s.segment.audio.close()
		s.segment = nil
	}
}

// Recv returns the next result, reporting the failures that come before it. It
// ends when the session closes, or once the segments have finished after the
// pipeline asked it to stop.
func (s *sttSession) Recv() ([]stt.Result, error) {
	for {
		s.mu.Lock()
		if len(s.queue) > 0 {
			item := s.queue[0]
			s.queue = s.queue[1:]
			s.mu.Unlock()
			if item.err != nil {
				s.c.reportError(s.ctx, item.err)
				continue
			}
			return []stt.Result{item.result}, nil
		}
		closed := s.closed
		drained := s.stopping && s.running == 0
		s.mu.Unlock()
		if closed {
			return nil, errSTTSessionClosed
		}
		if drained {
			return nil, io.EOF
		}
		<-s.wake
	}
}

// transcribe streams a segment to Rime and hands its transcripts on in speech
// order.
func (s *sttSession) transcribe(seg *segment) {
	defer func() {
		close(seg.published)
		seg.previous = nil
		s.mu.Lock()
		if s.segment == seg {
			s.segment = nil
		}
		s.running--
		s.mu.Unlock()
		s.signal()
	}()

	segmentSettings := &seg.settings
	interimLanguage := frameLanguage(segmentSettings.Language.Or(""))
	text := ""
	final, err := s.client.stream(s.ctx, segmentSettings.streamOptions(s.sendRate), seg.audio,
		func(update transcriptUpdate) {
			if update.kind == updateDelta {
				text += update.text
			} else {
				text = update.text
			}
			// Drop a later segment's interims until the earlier final is out;
			// the next interim replaces them.
			previous := seg.previous
			if text != "" && (previous == nil || previous.isPublished()) {
				s.enqueue(sessionItem{result: stt.Result{Text: text, Language: interimLanguage}})
			}
		})
	if err != nil {
		if re := asRimeError(err); re != nil && re.kind == kindCancelled {
			return
		}
		s.enqueue(sessionItem{err: err})
		return
	}
	if previous := seg.previous; previous != nil {
		select {
		case <-previous.published:
		case <-s.ctx.Done():
			return
		}
	}
	s.pushFinal(seg, final)
}

// pushFinal hands a segment's final transcript on.
func (s *sttSession) pushFinal(seg *segment, final transcriptFinal) {
	s.mu.Lock()
	// Once the user speaks again, this final covers only the words before they
	// resumed, so only the latest segment's final is finalized.
	latest := seg.number == s.segmentCount
	// An empty final has no words to add, but the latest one still tells the
	// user turn that no more transcripts are coming.
	if final.text == "" && !latest {
		s.mu.Unlock()
		return
	}
	fromFinalize := latest && s.finalizeRequested
	if fromFinalize {
		s.finalizeRequested = false
	}
	s.mu.Unlock()
	s.enqueue(sessionItem{result: stt.Result{
		Text:         final.text,
		Final:        true,
		Language:     frameLanguage(final.language),
		FromFinalize: fromFinalize,
		KeepEmpty:    true,
	}})
}

// audioQueue is the audio of one segment, in order, until it is closed.
type audioQueue struct {
	mu     sync.Mutex
	items  [][]byte
	closed bool
	ready  chan struct{}
}

func newAudioQueue() *audioQueue { return &audioQueue{ready: make(chan struct{}, 1)} }

func (q *audioQueue) notify() {
	select {
	case q.ready <- struct{}{}:
	default:
	}
}

// put adds a chunk.
func (q *audioQueue) put(pcm []byte) {
	q.mu.Lock()
	if !q.closed {
		q.items = append(q.items, pcm)
	}
	q.mu.Unlock()
	q.notify()
}

// close ends the audio.
func (q *audioQueue) close() {
	q.mu.Lock()
	q.closed = true
	q.mu.Unlock()
	q.notify()
}

// next returns the next chunk, or false once the audio has ended.
func (q *audioQueue) next(ctx context.Context) ([]byte, bool, error) {
	for {
		q.mu.Lock()
		if len(q.items) > 0 {
			pcm := q.items[0]
			q.items = q.items[1:]
			q.mu.Unlock()
			return pcm, true, nil
		}
		closed := q.closed
		q.mu.Unlock()
		if closed {
			return nil, false, nil
		}
		select {
		case <-q.ready:
		case <-ctx.Done():
			return nil, false, ctx.Err()
		}
	}
}
