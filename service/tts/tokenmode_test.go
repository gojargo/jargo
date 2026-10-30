package tts_test

// Ported from upstream's TTS frame-ordering suite: text streamed token by token
// is regrouped into sentences by the sequencer, which only promotes a sentence
// once a following token confirms its boundary. The end of a response and a
// TTSSpeakFrame have no following token, so the sentence they leave pending has
// to be promoted explicitly or its words never reach the conversation.

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/service/tts"
	uctx "github.com/gojargo/jargo/utils/context"
	"github.com/gojargo/jargo/utils/events"
	ttstext "github.com/gojargo/jargo/utils/text"
)

// tokenStreamingSynth is a provider streaming tokens over its own connection:
// one call per token, all on the turn's context, each delivering its own word
// timings and audio on a receive loop. It never closes a context itself; the
// context is closed when the base flushes it at the end of the turn.
type tokenStreamingSynth struct {
	mu        sync.Mutex
	host      tts.AudioContextHost
	wordTimes [][]uctx.WordTiming
	wg        sync.WaitGroup
	// last is closed once the delivery queued last is done, so deliveries land
	// in the order they were queued, as on a single receive loop.
	last chan struct{}
}

// deliver runs fn on the receive loop after delay, behind every delivery queued
// before it. The caller holds s.mu.
func (s *tokenStreamingSynth) deliver(delay time.Duration, fn func()) {
	prev, done := s.last, make(chan struct{})
	s.last = done
	s.wg.Go(func() {
		defer close(done)
		if prev != nil {
			<-prev
		}
		time.Sleep(delay)
		fn()
	})
}

func (s *tokenStreamingSynth) SampleRate() int { return orderingSampleRate }

func (s *tokenStreamingSynth) SetAudioContextHost(h tts.AudioContextHost) {
	s.mu.Lock()
	s.host = h
	s.mu.Unlock()
}

func (s *tokenStreamingSynth) RunTTS(context.Context, string, string, func(frames.Frame) error) error {
	return nil
}

func (s *tokenStreamingSynth) RunTTSTimed(
	_ context.Context, _, contextID string,
	_ func(frames.Frame) error,
	_ func([]uctx.WordTiming, tts.WordTimingOptions) error,
) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	host := s.host
	var words []uctx.WordTiming
	if len(s.wordTimes) > 0 {
		words, s.wordTimes = s.wordTimes[0], s.wordTimes[1:]
	}
	s.deliver(10*time.Millisecond, func() {
		if len(words) > 0 {
			host.AddWordTimestamps(contextID, words, tts.WordTimingOptions{})
		}
		host.AppendToAudioContext(contextID,
			frames.NewTTSAudioRawFrame(make([]byte, orderingChunkBytes), orderingSampleRate, 1))
	})
	return nil
}

// FlushAudio closes the context once the audio still on its way has landed, the
// way a provider ends a context it was told no more text is coming for.
func (s *tokenStreamingSynth) FlushAudio(_ context.Context, contextID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	host := s.host
	s.deliver(20*time.Millisecond, func() {
		host.AppendToAudioContext(contextID, frames.NewTTSStoppedFrame())
		host.RemoveAudioContext(contextID)
	})
}

// runTokenMode streams send through a service aggregating by token and returns
// everything that reached the end of the pipeline.
func runTokenMode(t *testing.T, synth *tokenStreamingSynth, send ...frames.Frame) []frames.Frame {
	t.Helper()
	var mu sync.Mutex
	var got []frames.Frame
	base := tts.New("TokenTTS", synth)
	base.SetTextAggregator(ttstext.NewTokenAggregator())
	task := pipeline.NewWorker(pipeline.New(base), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		got = append(got, f)
		mu.Unlock()
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	for _, f := range send {
		task.QueueFrame(f)
	}
	time.Sleep(300 * time.Millisecond)
	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(5 * time.Second):
		t.Fatal("task did not finish")
	}
	synth.wg.Wait()
	mu.Lock()
	defer mu.Unlock()
	return got
}

func spokenWords(down []frames.Frame) []string {
	var out []string
	for _, f := range down {
		if tf, ok := f.(*frames.TTSTextFrame); ok {
			out = append(out, tf.Text)
		}
	}
	return out
}

// TestIsolatedSpeakFrameFinalizesInTokenMode covers a lone TTSSpeakFrame while
// streaming tokens. It is a sentence of its own with no following token to
// confirm its boundary, so without being finalized its words are never emitted.
func TestIsolatedSpeakFrameFinalizesInTokenMode(t *testing.T) {
	synth := &tokenStreamingSynth{wordTimes: [][]uctx.WordTiming{
		{{Word: "hello", Offset: 0}, {Word: "world", Offset: 0.2}},
	}}
	speak := frames.NewTTSSpeakFrame("hello world")
	speak.AppendToContext = false

	got := spokenWords(runTokenMode(t, synth, speak))

	if strings.Join(got, "|") != "hello|world" {
		t.Fatalf("words = %q, want [hello world]", got)
	}
}

// TestTokenModeTracksTheWholeSentence covers a sentence streamed as two tokens.
// They are regrouped into one sentence: one anchor announces it, the progress
// frames grow across all of it, and its words reach the conversation even
// though the response ends with no token after the sentence's last one.
func TestTokenModeTracksTheWholeSentence(t *testing.T) {
	synth := &tokenStreamingSynth{wordTimes: [][]uctx.WordTiming{
		{{Word: "Hi", Offset: 0}},
		{{Word: "there.", Offset: 0.2}},
	}}

	down := runTokenMode(t, synth,
		frames.NewLLMFullResponseStartFrame(),
		frames.NewTextFrame("Hi"),
		frames.NewTextFrame(" there."),
		frames.NewLLMFullResponseEndFrame(),
	)

	var progress []*frames.AggregatedTextProgressFrame
	var anchors []*frames.AggregatedTextFrame
	for _, f := range down {
		switch fr := f.(type) {
		case *frames.AggregatedTextProgressFrame:
			progress = append(progress, fr)
		case *frames.AggregatedTextFrame:
			if fr.TextType == frames.TextTypeSentence {
				anchors = append(anchors, fr)
			}
		}
	}
	if len(progress) != 2 {
		t.Fatalf("%d progress frames, want 2", len(progress))
	}
	if progress[0].AccumulatedText != "Hi" || progress[0].RemainingText != " there." {
		t.Fatalf("first progress = %q / %q", progress[0].AccumulatedText, progress[0].RemainingText)
	}
	if progress[1].AccumulatedText != "Hi there." || progress[1].RemainingText != "" {
		t.Fatalf("second progress = %q / %q", progress[1].AccumulatedText, progress[1].RemainingText)
	}
	if len(anchors) != 1 || anchors[0].Text != "Hi there." || !anchors[0].WillBeSpoken {
		t.Fatalf("sentence anchors = %v, want one for %q", anchors, "Hi there.")
	}
	if progress[0].SegmentID != anchors[0].ID() || progress[1].SegmentID != anchors[0].ID() {
		t.Fatal("progress frames do not refer to the sentence anchor")
	}
	if got := strings.Join(spokenWords(down), " "); got != "Hi there." {
		t.Fatalf("words = %q, want %q", got, "Hi there.")
	}
}

// TestTokenModeSpeaksAnUnpunctuatedTrailingSentence covers a response whose
// last sentence has no terminal punctuation. Nothing confirms its boundary, so
// it is the end of the response that has to promote it.
func TestTokenModeSpeaksAnUnpunctuatedTrailingSentence(t *testing.T) {
	synth := &tokenStreamingSynth{wordTimes: [][]uctx.WordTiming{
		{{Word: "Hello", Offset: 0}, {Word: "there.", Offset: 0.1}},
		{{Word: "Sure", Offset: 0}},
		{{Word: "thing", Offset: 0}},
	}}

	down := runTokenMode(t, synth,
		frames.NewLLMFullResponseStartFrame(),
		frames.NewTextFrame("Hello there. "),
		frames.NewTextFrame("Sure "),
		frames.NewTextFrame("thing"),
		frames.NewLLMFullResponseEndFrame(),
	)

	if got := strings.Join(spokenWords(down), " "); got != "Hello there. Sure thing" {
		t.Fatalf("words = %q, want %q", got, "Hello there. Sure thing")
	}
}

// TestTokenModeWithoutWordTimestampsRecordsEachSentenceOnce covers a provider
// with no word timings streaming tokens. The sequencer records each sentence it
// promotes, so the tokens it was built from are not recorded as well.
func TestTokenModeWithoutWordTimestampsRecordsEachSentenceOnce(t *testing.T) {
	convo := frames.NewLLMContext("")
	pair := aggregators.New(convo)
	base := tts.New("PlainTTS", &plainSynth{rate: 24000, chunk: []byte{1, 2, 3, 4}})
	base.SetTextAggregator(ttstext.NewTokenAggregator())

	runFrames(t, []frames.Frame{
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMTextFrame("Hello"),
		frames.NewLLMTextFrame(" there."),
		frames.NewLLMFullResponseEndFrame(),
	}, base, pair.Assistant())

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Text != "Hello there." {
		t.Fatalf("messages = %+v, want one assistant %q", msgs, "Hello there.")
	}
}
