// Package perception reads how each user turn was said by asking classifiers
// (alpha).
//
// A [Reader] goes between the STT service and the user aggregator, and sets
// each final transcription's Perception from the answers of its sources: a
// classifier each, with the questions it is asked. Part of expressive
// conversation (alpha). These types are likely to change between releases.
package perception

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"slices"
	"sync"
	"time"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/pytext"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/utils/events"
)

// speechEndWait is the longest a transcription waits for the user to stop
// speaking: as long as the user aggregator keeps a turn open without user
// activity by default. A variable so tests can shorten it.
//
//nolint:gochecknoglobals // a seam for tests
var speechEndWait = 5 * time.Second

// errNoAnswer reports a classifier that left one of its questions unanswered.
var errNoAnswer = errors.New("perception: the classifier gave no answer for")

// Config configures a Reader.
type Config struct {
	// Name is the processor's name. Empty is "PerceptionReader".
	Name string
	// Sources is the classifiers to ask, each with its questions and its
	// timeout. At least one; a classifier goes in one source, and a signal's
	// name in one.
	Sources []Source
	// ReplacePerception makes the reader's reading replace the one a
	// transcription carries, such as the STT service's, instead of being added
	// to it.
	ReplacePerception bool
}

// Reader reads how each user turn was said by asking classifiers.
//
// It goes between the STT service and the user aggregator. Each final
// transcription gets the classifiers' answers as its Perception, one signal per
// question, named after it. Every classifier's reading is also broadcast in a
// frames.PerceptionFrame as soon as it is made, and the signals are declared at
// start in a frames.PerceptionMetadataFrame.
//
// A text classifier reads each final transcription, with the bot's reply before
// it. A classifier that can hear, as its SupportsAudio says, hears the user's
// speech each time they stop: the latest, topped up with the end of what they
// said before it since the bot last spoke when that is shorter than the
// classifier answers best. It hears one clip at a time, and of the clips that
// come while it is busy, only the newest. The transcriptions of that speech
// wait for its answers, up to its source's timeout after the user stopped.
//
// A reader asks each of its sources about the same turn, all at once, and their
// signals go in the turn's one reading, in the sources' order. A transcription
// goes on once every source has read it or run out of time. A source that is
// late or whose classifier fails leaves its signals out, and a transcription
// none of them read passes unread. A classifier that cannot be set up leaves
// the reader unusable.
//
// A transcription that already carries a reading, such as the STT service's,
// gets the reader's added to it, and the reader declares the signals declared
// before it together with its own. With Config.ReplacePerception, the reader's
// reading replaces the one the transcription carries instead.
//
// Turn expressive conversation on with
// pipeline.Params.EnableExpressiveConversation for the LLM to read the turns:
//
//	reader, err := perception.New(perception.Config{Sources: []perception.Source{frustration}})
//	p := pipeline.New(transport.Input(), stt, reader, userAggregator, llm, ...)
type Reader struct {
	*processor.Base
	sources []Source
	// textSources and audioSources are who to ask, by index into sources: the
	// sources that read transcriptions, and the ones that hear.
	textSources  []int
	audioSources []int
	replace      bool

	mu sync.Mutex
	// declaredBefore is the signals declared before the reader, which its
	// transcriptions can carry too.
	declaredBefore *expressive.PerceptionCapabilities
	// previous is the bot's last reply, which the user's turn is read against.
	previous string
	// audio is the user's recent speech, for the sources that hear; clipReads
	// is their reads of the clips that end with the user's latest speech, and
	// clipTasks each one's latest read, which its next one waits for.
	audio     *recentSpeech
	clipReads *reads
	clipTasks map[int]*read
	// held is the transcriptions held until their speech is read, and release
	// is closed once the goroutine pushing them on is done, or nil when none
	// runs.
	held       []heldTranscription
	release    chan struct{}
	sttLatency time.Duration

	// ctx is what every read and the release run under; cleaning up cancels it
	// and waits for them in wg.
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// New builds a Reader. It fails with ErrInvalidConfig when there are no
// sources, or when two report the same signal or ask the same classifier.
func New(cfg Config) (*Reader, error) {
	if cfg.Name == "" {
		cfg.Name = "PerceptionReader"
	}
	r := &Reader{
		sources:   slices.Clone(cfg.Sources),
		replace:   cfg.ReplacePerception,
		clipTasks: map[int]*read{},
	}
	r.Base = processor.New(cfg.Name, r)
	if len(r.sources) == 0 {
		return nil, fmt.Errorf("%w: %s needs at least one source", ErrInvalidConfig, r.Name())
	}
	var names []string
	for i, source := range r.sources {
		if source == nil || source.Classifier() == nil {
			return nil, fmt.Errorf("%w: %s: source %d has no classifier", ErrInvalidConfig, r.Name(), i)
		}
		for _, other := range r.sources[:i] {
			if other.Classifier() == source.Classifier() {
				return nil, fmt.Errorf("%w: %s: a classifier goes in one source, with all its questions",
					ErrInvalidConfig, r.Name())
			}
		}
		names = append(names, source.Names()...)
	}
	for i, name := range names {
		if slices.Contains(names[:i], name) {
			return nil, fmt.Errorf("%w: %s: two sources report %q, and a reading has one signal of a name",
				ErrInvalidConfig, r.Name(), name)
		}
	}
	hearing := make([]classifier.Classifier, 0, len(r.sources))
	for i, source := range r.sources {
		if source.Classifier().SupportsAudio() {
			r.audioSources = append(r.audioSources, i)
			hearing = append(hearing, source.Classifier())
		} else {
			r.textSources = append(r.textSources, i)
		}
		events.On(source.Classifier().Events(), classifier.EventMetrics, r.onClassifierMetrics)
	}
	r.audio = newRecentSpeech(hearing)
	return r, nil
}

// Sources is the classifiers asked about each turn, each with its questions.
func (r *Reader) Sources() []Source { return slices.Clone(r.sources) }

// Setup sets the processor and its classifiers up. A classifier that cannot be
// set up is reported as an error that leaves the reader unusable.
func (r *Reader) Setup(ctx context.Context, s processor.Setup) error {
	if err := r.Base.Setup(ctx, s); err != nil {
		return err
	}
	r.mu.Lock()
	r.audio.setSampleRate(s.AudioInSampleRate)
	r.ctx, r.cancel = context.WithCancel(ctx)
	r.mu.Unlock()
	for _, source := range r.sources {
		if err := source.Classifier().Setup(ctx); err != nil {
			if !errors.Is(err, classifier.ErrClassifier) {
				return err
			}
			r.PushError(ctx, fmt.Sprintf("%s: %v", r.Name(), err), err, false, processor.ForceTreatAsPermanent())
			break
		}
	}
	return nil
}

// Cleanup stops the reads and the release of the transcriptions held, and
// cleans the classifiers up.
func (r *Reader) Cleanup(ctx context.Context) error {
	err := r.Base.Cleanup(ctx)
	r.mu.Lock()
	if r.cancel != nil {
		r.cancel()
	}
	r.mu.Unlock()
	r.wg.Wait()
	r.mu.Lock()
	r.clipReads = nil
	r.mu.Unlock()
	for _, source := range r.sources {
		err = errors.Join(err, source.Classifier().Cleanup(ctx))
	}
	return err
}

// ProcessFrame reads the user's turns, and keeps the bot's replies to read them
// against.
//
//nolint:cyclop // one arm per frame the reader handles
func (r *Reader) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := r.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case frames.InputAudioFrame:
		r.handleAudio(fr)
	case *frames.StartFrame:
		// The declaration follows the StartFrame, which must reach every
		// processor before anything else.
		if err := r.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		return r.declareSignals(ctx)
	case *frames.EndFrame:
		// The transcriptions held go on before the pipeline ends.
		if err := r.waitForRelease(ctx); err != nil {
			return err
		}
	case *frames.TranscriptionFrame:
		if r.holds(fr) {
			// Pushed on with the reading once the user's speech is over.
			r.holdTranscription(fr)
			return nil
		}
		if err := r.handleTranscription(ctx, fr); err != nil {
			return err
		}
	case *frames.PerceptionMetadataFrame:
		if dir == processor.Downstream {
			// Declared again with the reader's signals, which its transcriptions
			// carry too.
			r.mu.Lock()
			declared := fr.Perception
			r.declaredBefore = &declared
			r.mu.Unlock()
			return r.declareSignals(ctx)
		}
	case *frames.STTMetadataFrame:
		r.handleSTTMetadata(fr)
	case *frames.VADUserStartedSpeakingFrame:
		r.mu.Lock()
		r.audio.start()
		r.mu.Unlock()
	case *frames.VADUserStoppedSpeakingFrame:
		r.handleVADUserStoppedSpeaking(fr)
	case *frames.UserMuteStartedFrame:
		r.handleMute(true)
	case *frames.UserMuteStoppedFrame:
		r.handleMute(false)
	case *frames.BotStoppedSpeakingFrame:
		r.mu.Lock()
		r.audio.botStoppedSpeaking()
		r.mu.Unlock()
	case *frames.LLMContextAssistantTurnFrame:
		r.mu.Lock()
		r.previous = fr.Text
		r.mu.Unlock()
	}
	return r.PushFrame(ctx, f, dir)
}

func (r *Reader) handleAudio(f frames.InputAudioFrame) {
	if len(r.audioSources) == 0 {
		return
	}
	r.mu.Lock()
	r.audio.append(f.AudioData().Audio)
	r.mu.Unlock()
}

// holds reports whether the transcription waits for an audio reading.
func (r *Reader) holds(f *frames.TranscriptionFrame) bool {
	return len(r.audioSources) > 0 && pytext.Strip(f.Text) != "" && r.Usable()
}

func (r *Reader) holdTranscription(f *frames.TranscriptionFrame) {
	// The text sources read it while it waits for its speech to be heard.
	reads := r.readText(f.Text)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.held = append(r.held, heldTranscription{frame: f, reads: reads})
	if r.release == nil {
		done := make(chan struct{})
		r.release = done
		r.wg.Go(func() { r.releaseTranscriptions(r.ctx, done) })
	}
}

func (r *Reader) handleTranscription(ctx context.Context, f *frames.TranscriptionFrame) error {
	if pytext.Strip(f.Text) == "" || !r.Usable() {
		return nil
	}
	readings := r.wait(ctx, r.readText(f.Text))
	if err := ctx.Err(); err != nil {
		return err
	}
	r.setReading(f, r.reading(readings))
	return nil
}

func (r *Reader) handleSTTMetadata(f *frames.STTMetadataFrame) {
	r.mu.Lock()
	r.sttLatency = f.TTFSP99Latency
	r.mu.Unlock()
	// Transcriptions held for an audio reading reach the user aggregator up to
	// the longest timeout later than the STT sends them, so the stop strategies
	// wait that much longer.
	if len(r.audioSources) > 0 && r.Usable() {
		var longest time.Duration
		for _, source := range r.sources {
			longest = max(longest, source.Timeout())
		}
		f.TTFSP99Latency += longest
	}
}

func (r *Reader) handleVADUserStoppedSpeaking(f *frames.VADUserStoppedSpeakingFrame) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.audio.stop(f.StopSecs)
	if len(r.audioSources) > 0 && r.Usable() {
		r.readSpeech()
	}
}

func (r *Reader) handleMute(muted bool) {
	r.mu.Lock()
	r.audio.mute(muted)
	r.mu.Unlock()
	if muted {
		r.cancelClipReads()
	}
}

// waitForRelease waits for the transcriptions held to go on.
func (r *Reader) waitForRelease(ctx context.Context) error {
	r.mu.Lock()
	done := r.release
	r.mu.Unlock()
	if done == nil {
		return nil
	}
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// declareSignals pushes the signals the reader's transcriptions can carry, if
// any.
func (r *Reader) declareSignals(ctx context.Context) error {
	capabilities := r.capabilities()
	if len(capabilities.Signals) == 0 {
		return nil
	}
	return r.PushFrame(ctx, frames.NewPerceptionMetadataFrame(r.Name(), capabilities), processor.Downstream)
}

// capabilities is the signals declared before the reader, and its sources'
// when it can report them.
func (r *Reader) capabilities() expressive.PerceptionCapabilities {
	var own expressive.PerceptionCapabilities
	if r.Usable() {
		for _, source := range r.sources {
			questions, err := source.Questions()
			if err != nil {
				continue
			}
			for _, q := range questions {
				own.Signals = append(own.Signals, expressive.PerceptionTerm{
					Name: q.Name, Description: q.Question.Description, Labels: labels(q.Question.Question),
				})
			}
		}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.declaredBefore != nil {
		return r.declaredBefore.Merge(own)
	}
	return own
}

// labels is the labels a signal can report: a choice's options, or none.
func labels(q classifier.Question) []string {
	if choice, ok := q.(classifier.ChoiceQuestion); ok {
		return choice.OptionNames()
	}
	return nil
}

// readText starts each text source's read of a transcription, with the bot's
// reply before it.
func (r *Reader) readText(text string) *reads {
	r.mu.Lock()
	previous := r.previous
	r.mu.Unlock()
	state := map[string]any{"user": text}
	if previous != "" {
		state["assistant"] = previous
	}
	tasks := make([]sourceRead, 0, len(r.textSources))
	for _, i := range r.textSources {
		tasks = append(tasks, sourceRead{source: i, read: r.startRead(func(ctx context.Context, _ *read) result {
			return r.read(ctx, i, state)
		})})
	}
	return &reads{startedAt: time.Now(), tasks: tasks}
}

// readSpeech starts each audio source's read of the clip that ends with the
// speech just finished. The caller holds mu.
func (r *Reader) readSpeech() {
	r.clipReads = nil
	endedAt := r.audio.endedAt
	if endedAt.IsZero() {
		return
	}
	state := map[string]any{}
	if r.previous != "" {
		state["assistant"] = r.previous
	}
	var tasks []sourceRead
	for _, i := range r.audioSources {
		source := r.sources[i]
		// Each classifier takes clips of its own lengths.
		clip, ok := r.audio.clip(source.Classifier())
		if !ok {
			continue
		}
		slog.Debug(fmt.Sprintf("%s: %s is reading %.1fs of the user's audio",
			r.Name(), source.Classifier().Name(), clip.Duration().Seconds()))
		tasks = append(tasks, sourceRead{source: i, read: r.readClip(i, state, clip)})
	}
	if len(tasks) > 0 {
		r.clipReads = &reads{startedAt: endedAt, tasks: tasks}
	}
}

// readClip starts a source's read of a clip, which asks once the source's
// earlier read is done. The caller holds mu.
//
// A classifier that hears is asked about one clip at a time, and of the clips
// that came while it was busy, only about the newest.
func (r *Reader) readClip(i int, state map[string]any, clip classifier.Audio) *read {
	earlier := r.clipTasks[i]
	current := r.startRead(func(ctx context.Context, self *read) result {
		if earlier != nil {
			// A wait that doesn't cancel the earlier read, nor fails when a mute
			// cancels it.
			select {
			case <-earlier.done:
			case <-ctx.Done():
				return result{}
			}
			r.mu.Lock()
			superseded := r.clipTasks[i] != self
			r.mu.Unlock()
			if superseded {
				return result{}
			}
		}
		return r.read(ctx, i, state, clip)
	})
	r.clipTasks[i] = current
	return current
}

// startRead runs a read on a goroutine of its own, which cleaning up stops.
func (r *Reader) startRead(run func(ctx context.Context, self *read) result) *read {
	ctx, cancel := context.WithCancel(r.ctx)
	rd := &read{done: make(chan struct{}), cancel: cancel}
	r.wg.Go(func() {
		defer close(rd.done)
		defer cancel()
		res := run(ctx, rd)
		// A read stopped while it ran found nothing.
		if ctx.Err() == nil {
			rd.result = res
		}
	})
	return rd
}

// read is a source's reading, broadcast once made, late or not, or nothing
// when its classifier fails.
func (r *Reader) read(ctx context.Context, i int, state map[string]any, inputs ...classifier.Input) result {
	source := r.sources[i]
	signals, err := r.ask(ctx, source, state, inputs)
	switch {
	case ctx.Err() != nil:
		// Stopped: a mute or cleaning up.
		return result{}
	case err == nil:
		if berr := r.Broadcast(ctx, func() frames.Frame {
			return frames.NewPerceptionFrame(slices.Clone(signals))
		}); berr != nil {
			slog.DebugContext(ctx, "perception: broadcasting a reading failed", "reader", r.Name(), "error", berr)
		}
		return result{signals: signals, ok: true}
	case errors.Is(err, classifier.ErrClassifier):
		r.PushError(ctx, fmt.Sprintf("%s: %v", r.Name(), err), err, false)
	default:
		slog.ErrorContext(ctx, "perception: a classifier failed", "reader", r.Name(),
			"classifier", source.Classifier().Name(), "error", err)
	}
	return result{}
}

// ask asks a source's classifier its questions, and returns the answers as the
// signals they report.
func (r *Reader) ask(
	ctx context.Context, source Source, state map[string]any, inputs []classifier.Input,
) ([]expressive.PerceptionSignal, error) {
	questions, err := source.Questions()
	if err != nil {
		return nil, err
	}
	results, err := source.Classifier().Ask(ctx, state, classifierQuestions(questions), inputs...)
	if err != nil {
		return nil, err
	}
	return signals(results, questions)
}

// releaseTranscriptions pushes the held transcriptions on with the reading, in
// order.
func (r *Reader) releaseTranscriptions(ctx context.Context, done chan struct{}) {
	defer close(done)
	for {
		r.mu.Lock()
		if len(r.held) == 0 || ctx.Err() != nil {
			r.release = nil
			r.mu.Unlock()
			return
		}
		r.mu.Unlock()
		r.releaseSpeech(ctx)
	}
}

// releaseSpeech pushes the transcriptions of the user's speech on together,
// once it is over and read.
func (r *Reader) releaseSpeech(ctx context.Context) {
	speechEnd := r.waitForSpeechEnd(ctx)
	r.mu.Lock()
	held := r.held
	r.held = nil
	clipReads := r.takeClipReads(speechEnd)
	r.mu.Unlock()
	heard := r.wait(ctx, clipReads)
	readings := make([][]expressive.PerceptionSignal, len(held))
	for i, t := range held {
		merged := maps.Clone(heard)
		maps.Copy(merged, r.wait(ctx, t.reads))
		readings[i] = r.reading(merged)
	}
	if ctx.Err() != nil {
		return
	}
	// The stop strategies end the turn at once on a finalized transcription, so
	// only the speech's last one is, and none when the user has spoken again
	// since.
	r.mu.Lock()
	over := !speechEnd.IsZero() && r.audio.endedAt.Equal(speechEnd)
	r.mu.Unlock()
	for i, t := range held {
		r.setReading(t.frame, readings[i])
		t.frame.Finalized = over && i == len(held)-1
		if err := r.PushFrame(ctx, t.frame, processor.Downstream); err != nil {
			slog.DebugContext(ctx, "perception: pushing a held transcription failed", "reader", r.Name(), "error", err)
		}
	}
}

// waitForSpeechEnd is when the user stopped speaking, once the STT's latency
// has passed, or zero if they have not yet.
func (r *Reader) waitForSpeechEnd(ctx context.Context) time.Time {
	r.mu.Lock()
	speaking, ended := r.audio.speaking, r.audio.ended
	r.mu.Unlock()
	if speaking {
		slog.Debug(r.Name() + ": transcriptions are waiting for the user to stop speaking")
		timer := time.NewTimer(speechEndWait)
		select {
		case <-ended:
		case <-timer.C:
		case <-ctx.Done():
		}
		timer.Stop()
	}
	r.mu.Lock()
	speechEnd, latency := r.audio.endedAt, r.sttLatency
	r.mu.Unlock()
	if !speechEnd.IsZero() {
		sleep(ctx, time.Until(speechEnd.Add(latency)))
	}
	return speechEnd
}

// takeClipReads is the reads of the clips that end with the speech, if any.
// The caller holds mu.
func (r *Reader) takeClipReads(speechEnd time.Time) *reads {
	// Reads are used once, and only for the speech their clips end with, so a
	// transcription of no new speech gets no reading.
	clipReads := r.clipReads
	r.clipReads = nil
	if clipReads != nil && !speechEnd.IsZero() && clipReads.startedAt.Equal(speechEnd) {
		return clipReads
	}
	return nil
}

// wait is what each source read, waited for up to its timeout from when the
// reads started. A source that read nothing is left out.
func (r *Reader) wait(ctx context.Context, rs *reads) map[int][]expressive.PerceptionSignal {
	readings := map[int][]expressive.PerceptionSignal{}
	if rs == nil {
		return readings
	}
	for _, t := range rs.tasks {
		source := r.sources[t.source]
		// A wait that doesn't stop a late read, which still broadcasts its
		// reading, nor fails when a mute stops it.
		if !sleepUntilDone(ctx, t.read.done, time.Until(rs.startedAt.Add(source.Timeout()))) && ctx.Err() == nil {
			slog.Warn(fmt.Sprintf("%s: no reading from %s after %s, the turn goes on without it",
				r.Name(), source.Classifier().Name(), source.Timeout()))
		}
	}
	for _, t := range rs.tasks {
		if res, ok := t.read.found(); ok {
			readings[t.source] = res
		}
	}
	return readings
}

// reading is the sources' readings as one, in the sources' order, or nil when
// none read the turn.
func (r *Reader) reading(readings map[int][]expressive.PerceptionSignal) []expressive.PerceptionSignal {
	var (
		out  []expressive.PerceptionSignal
		read bool
	)
	for i := range r.sources {
		if signals, ok := readings[i]; ok {
			read = true
			out = append(out, signals...)
		}
	}
	if !read {
		return nil
	}
	if out == nil {
		out = []expressive.PerceptionSignal{}
	}
	return out
}

// setReading sets the reading on a transcription, replacing or added to the
// one it carries.
func (r *Reader) setReading(f *frames.TranscriptionFrame, reading []expressive.PerceptionSignal) {
	if len(reading) == 0 {
		return
	}
	if r.replace {
		f.Perception = reading
		return
	}
	f.Perception = expressive.MergePerception(f.Perception, reading)
}

// cancelClipReads stops the reads of the latest clips still running, if any.
func (r *Reader) cancelClipReads() {
	r.mu.Lock()
	clipReads := r.clipReads
	r.clipReads = nil
	r.mu.Unlock()
	if clipReads == nil {
		return
	}
	for _, t := range clipReads.tasks {
		if !t.read.isDone() {
			slog.Debug(r.Name() + ": dropping a clip still being read")
			t.read.cancel()
			<-t.read.done
		}
	}
}

// classifierQuestions is the questions as the classifier is asked them.
func classifierQuestions(questions []NamedQuestion) []classifier.Named[classifier.Question] {
	out := make([]classifier.Named[classifier.Question], 0, len(questions))
	for _, q := range questions {
		out = append(out, classifier.Named[classifier.Question]{Name: q.Name, Question: q.Question.Question})
	}
	return out
}

// signals is the answers as the signals they report, in the questions' order.
func signals(results map[string]classifier.Result, questions []NamedQuestion) ([]expressive.PerceptionSignal, error) {
	out := make([]expressive.PerceptionSignal, 0, len(questions))
	for _, q := range questions {
		res, ok := results[q.Name]
		if !ok {
			return nil, fmt.Errorf("%w: %q", errNoAnswer, q.Name)
		}
		out = append(out, signal(q, res))
	}
	return out, nil
}

// signal is one answer as the signal it reports.
func signal(q NamedQuestion, res classifier.Result) expressive.PerceptionSignal {
	switch res := res.(type) {
	case classifier.ChoiceResult:
		likeliest := choiceProbabilities(q.Question.Question, res.Probabilities)
		slices.SortStableFunc(likeliest, func(a, b expressive.PerceptionLabel) int {
			return cmp.Compare(b.Confidence, a.Confidence)
		})
		return expressive.PerceptionSignal{Name: q.Name, Labels: likeliest[:min(q.Question.maxLabels(), len(likeliest))]}
	case classifier.ScoreResult:
		value := pytext.Round(res.Score, 2)
		return expressive.PerceptionSignal{Name: q.Name, Value: &value}
	case classifier.YesNoResult:
		value := pytext.Round(res.Probability, 2)
		return expressive.PerceptionSignal{Name: q.Name, Value: &value}
	default:
		return expressive.PerceptionSignal{Name: q.Name}
	}
}

// choiceProbabilities is a choice's probabilities as labels, in the order of
// the question's options, then any other label the classifier gave in name
// order, so labels as likely as each other keep a stable order.
func choiceProbabilities(q classifier.Question, probabilities map[string]float64) []expressive.PerceptionLabel {
	out := make([]expressive.PerceptionLabel, 0, len(probabilities))
	var options []string
	if choice, ok := q.(classifier.ChoiceQuestion); ok {
		options = choice.OptionNames()
	}
	for _, option := range options {
		if p, ok := probabilities[option]; ok {
			out = append(out, expressive.PerceptionLabel{Label: option, Confidence: p})
		}
	}
	for _, label := range slices.Sorted(maps.Keys(probabilities)) {
		if !slices.Contains(options, label) {
			out = append(out, expressive.PerceptionLabel{Label: label, Confidence: probabilities[label]})
		}
	}
	return out
}

// onClassifierMetrics pushes a classifier's metrics into the pipeline, when
// metrics are on.
func (r *Reader) onClassifierMetrics(ctx context.Context, data []frames.MetricsData) {
	if !r.MetricsEnabled() {
		return
	}
	if err := r.PushFrame(ctx, frames.NewMetricsFrame(data...), processor.Downstream); err != nil {
		slog.DebugContext(ctx, "perception: pushing classifier metrics failed", "reader", r.Name(), "error", err)
	}
}

// result is what a read found: a source's signals, or nothing when it is late,
// was stopped or failed.
type result struct {
	signals []expressive.PerceptionSignal
	ok      bool
}

// read is one source's read of a turn, running on a goroutine of its own.
type read struct {
	done   chan struct{}
	cancel context.CancelFunc
	// result is set before done is closed.
	result result
}

// isDone reports whether the read has finished.
func (rd *read) isDone() bool {
	select {
	case <-rd.done:
		return true
	default:
		return false
	}
}

// found is what the read found, and false when it is late, was stopped or
// failed.
func (rd *read) found() ([]expressive.PerceptionSignal, bool) {
	if !rd.isDone() {
		return nil, false
	}
	return rd.result.signals, rd.result.ok
}

// reads is the sources' reads of a turn, and when they started: each timeout
// counts from then. A transcription's reads start when it arrives, and the
// reads of the clips of the user's speech when the user stopped it.
type reads struct {
	startedAt time.Time
	tasks     []sourceRead
}

// sourceRead is one source's read, by the source's index.
type sourceRead struct {
	source int
	read   *read
}

// heldTranscription is a transcription held until its speech is read, and the
// text sources' reads of it.
type heldTranscription struct {
	frame *frames.TranscriptionFrame
	reads *reads
}

// sleep waits for d, or until ctx ends.
func sleep(ctx context.Context, d time.Duration) {
	if d <= 0 {
		return
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
	case <-ctx.Done():
	}
}

// sleepUntilDone waits up to d for done to close, or until ctx ends, and
// reports whether it closed.
func sleepUntilDone(ctx context.Context, done <-chan struct{}, d time.Duration) bool {
	timer := time.NewTimer(max(d, 0))
	defer timer.Stop()
	select {
	case <-done:
		return true
	case <-timer.C:
	case <-ctx.Done():
	}
	select {
	case <-done:
		return true
	default:
		return false
	}
}
