package tts

// A context's words are timed from where the words before it end, for a
// provider that reports word ends. Its audio is pushed as soon as it arrives,
// often while the context before it is still playing, so the moment its audio
// is pushed says nothing about when it is heard.

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	uctx "github.com/gojargo/jargo/utils/context"
	"github.com/gojargo/jargo/utils/timeutil"
)

// perCallWordSynth answers inline, the way an HTTP provider does, reporting the
// next of its word lists for each call before the call's audio. An empty list
// reports no words for that call.
type perCallWordSynth struct {
	mu    sync.Mutex
	calls [][]uctx.WordTiming
}

func (s *perCallWordSynth) SampleRate() int { return alignSampleRate }

func (s *perCallWordSynth) RunTTS(_ context.Context, _, _ string, yield func(frames.Frame) error) error {
	return yield(frames.NewTTSAudioRawFrame(make([]byte, 640), alignSampleRate, 1))
}

func (s *perCallWordSynth) RunTTSTimed(
	ctx context.Context, text, contextID string,
	yield func(frames.Frame) error,
	word func([]uctx.WordTiming, WordTimingOptions) error,
) error {
	s.mu.Lock()
	var words []uctx.WordTiming
	if len(s.calls) > 0 {
		words, s.calls = s.calls[0], s.calls[1:]
	}
	s.mu.Unlock()
	if len(words) > 0 {
		if err := word(words, WordTimingOptions{}); err != nil {
			return err
		}
	}
	return s.RunTTS(ctx, text, contextID, yield)
}

// unrecordedSpeak is a speak frame whose text stays out of the conversation.
func unrecordedSpeak(text string) *frames.TTSSpeakFrame {
	f := frames.NewTTSSpeakFrame(text)
	f.AppendToContext = false
	return f
}

// wordPTSByText maps each word frame's text to its presentation timestamp.
func wordPTSByText(down []frames.Frame) map[string]int64 {
	pts := map[string]int64{}
	for _, f := range down {
		if tf, ok := f.(*frames.TTSTextFrame); ok {
			pts[tf.Text], _ = tf.PTS()
		}
	}
	return pts
}

// TestNextContextWordsStartWherePreviousWordsEnd covers a context whose audio
// arrives while the one before is still playing. It is timed from where that
// context's words end, not where its last word starts.
func TestNextContextWordsStartWherePreviousWordsEnd(t *testing.T) {
	synth := &perCallWordSynth{calls: [][]uctx.WordTiming{
		{{Word: "Hello", Offset: 0.0, End: 0.3}, {Word: "there.", Offset: 0.3, End: 0.8}},
		{{Word: "Bye", Offset: 0.0, End: 0.2}, {Word: "now.", Offset: 0.2, End: 0.5}},
	}}
	down := runBase(t, New("WordEndsTTS", synth),
		unrecordedSpeak("Hello there."),
		unrecordedSpeak("Bye now."),
		sleepFor(100*time.Millisecond),
	)

	pts := wordPTSByText(down)
	if got, want := pts["Bye"]-pts["Hello"], int64(timeutil.SecondsToDuration(0.8)); got != want {
		t.Errorf("Bye is timed %v after Hello, want %v", time.Duration(got), time.Duration(want))
	}
}

// TestInterruptionForgetsWhereDroppedWordsEnd covers words dropped by an
// interruption: where they would have ended does not delay the next context's
// words.
func TestInterruptionForgetsWhereDroppedWordsEnd(t *testing.T) {
	synth := &perCallWordSynth{calls: [][]uctx.WordTiming{
		{{Word: "Hello.", Offset: 0.0, End: 30.0}},
		{{Word: "Bye.", Offset: 0.0, End: 0.2}},
	}}
	down := runBase(t, New("WordEndsTTS", synth),
		unrecordedSpeak("Hello."),
		sleepFor(100*time.Millisecond),
		frames.NewInterruptionFrame(),
		unrecordedSpeak("Bye."),
		sleepFor(100*time.Millisecond),
	)

	pts := wordPTSByText(down)
	if _, ok := pts["Bye."]; !ok {
		t.Fatalf("Bye. was not spoken: %s", frameNames(down))
	}
	if got := pts["Bye."] - pts["Hello."]; got >= int64(time.Second) {
		t.Errorf("Bye. is timed %v after Hello., want under a second", time.Duration(got))
	}
}

// TestWordTimingStartsWhereTheWordsBeforeEnd checks the next context's words
// start after the last word's end, not its start.
func TestWordTimingStartsWhereTheWordsBeforeEnd(t *testing.T) {
	b := New("WordEndsTTS", &perCallWordSynth{})
	first := newAudioContext(false, context.Background(), nil)
	first.startWordTimestamps(b.wordTimingBaseline(10 * time.Second))
	last, _ := first.wordPTS(1.0)
	b.setLastWordPTS(last)
	end, _ := first.wordPTS(1.5)
	b.endWord(end)

	next := newAudioContext(false, context.Background(), nil)
	next.startWordTimestamps(b.wordTimingBaseline(10*time.Second + time.Second/5))
	if got, _ := next.wordPTS(0); got != 11*time.Second+time.Second/2 {
		t.Errorf("the next context's words start at %v, want 11.5s", got)
	}
}

// TestWordTimingStartsNowOnceTheWordsBeforeHaveEnded checks a context that
// starts after the words before it ended is timed from now.
func TestWordTimingStartsNowOnceTheWordsBeforeHaveEnded(t *testing.T) {
	b := New("WordEndsTTS", &perCallWordSynth{})
	first := newAudioContext(false, context.Background(), nil)
	first.startWordTimestamps(b.wordTimingBaseline(10 * time.Second))
	end, _ := first.wordPTS(1.5)
	b.endWord(end)

	if got := b.wordTimingBaseline(20 * time.Second); got != 20*time.Second {
		t.Errorf("baseline = %v, want 20s", got)
	}
}
