package tts_test

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
)

// timedWord is one spoken token the fake provider reports, with its start offset
// (seconds) and the audio chunk that carries it.
type timedWord struct {
	text   string
	offset float64
	pcm    []byte
}

// fakeTimedSynth is a Synthesizer that reports word timings (implements
// tts.WordTimestamps). It emits each preset word's audio chunk and timing in
// order; after blockAfter words it blocks until the context is canceled, letting
// a test interrupt partway through "playback".
type fakeTimedSynth struct {
	rate       int
	words      []timedWord
	blockAfter int
}

// Asserted, because a synthesizer that stops satisfying this interface does not
// fail to build: the base just stops reporting word timings for it.
var _ tts.WordTimestamps = (*fakeTimedSynth)(nil)

func (s *fakeTimedSynth) SampleRate() int { return s.rate }

// Synthesize is the plain fallback; unused because the base takes the timed path.
func (s *fakeTimedSynth) RunTTS(_ context.Context, _, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	for _, w := range s.words {
		if err := emit(w.pcm); err != nil {
			return err
		}
	}
	return nil
}

func (s *fakeTimedSynth) RunTTSTimed(
	ctx context.Context,
	_, _ string,
	yield func(f frames.Frame) error,
	word func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	for i, w := range s.words {
		if err := word([]uctx.WordTiming{{Word: w.text, Offset: w.offset}}, tts.WordTimingOptions{}); err != nil {
			return err
		}
		if err := emit(w.pcm); err != nil {
			return err
		}
		if s.blockAfter > 0 && i+1 == s.blockAfter {
			<-ctx.Done() // hold "mid-utterance" until interrupted
			return ctx.Err()
		}
	}
	return nil
}

// currencyFilter mimics a voice formatter expanding a written amount into spoken
// words, so the base must map the spoken words back to the written form.
type currencyFilter struct{}

func (currencyFilter) Filter(text string) string {
	return strings.ReplaceAll(text, "$42.50", "forty two dollars and fifty cents")
}

// wordsFor builds the spoken-word timeline for the expanded balance sentence.
// Each word gets a 0.1s chunk and a start offset one chunk after the last.
func wordsFor(rate int) []timedWord {
	spoken := []string{"Your", "balance", "is", "forty", "two", "dollars", "and", "fifty", "cents"}
	chunk := make([]byte, rate/10*2) // 0.1s of 16-bit mono silence
	out := make([]timedWord, len(spoken))
	for i, w := range spoken {
		out[i] = timedWord{text: w, offset: float64(i) * 0.1, pcm: chunk}
	}
	return out
}

// runConversation wires the TTS base to an assistant aggregator sharing convo,
// speaks the sentence, optionally interrupts once the aggregator has recorded
// the spoken words, and returns after the task stops.
func runConversation(t *testing.T, convo *frames.LLMContext, syn tts.Synthesizer, interrupt bool) {
	t.Helper()
	base := tts.New("FakeTTS", syn)
	base.SetTextFilters(currencyFilter{})
	pair := aggregators.New(convo)

	task := pipeline.NewTask(pipeline.New(base, pair.Assistant()), pipeline.TaskParams{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Your balance is $42.50"))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	// Let the spoken words flow into the context before deciding to interrupt.
	time.Sleep(400 * time.Millisecond)
	if interrupt {
		task.QueueFrame(frames.NewInterruptionFrame())
		time.Sleep(200 * time.Millisecond)
	}
	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}
}

// TestWordTimestampsFullUtteranceRecordsOriginalForm proves the assistant
// context records the whole sentence in its original written form once it is
// spoken in full — the transformed span maps back to "$42.50".
func TestWordTimestampsFullUtteranceRecordsOriginalForm(t *testing.T) {
	convo := frames.NewLLMContext("system")
	syn := &fakeTimedSynth{rate: 24000, words: wordsFor(24000)}
	runConversation(t, convo, syn, false)

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Role != frames.RoleAssistant {
		t.Fatalf("messages = %+v, want one assistant message", msgs)
	}
	if msgs[0].Text != "Your balance is $42.50" {
		t.Fatalf("assistant text = %q, want %q", msgs[0].Text, "Your balance is $42.50")
	}
}

// TestWordTimestampsInterruptionTruncatesToSpoken proves that interrupting
// before the transformed span completes records only the words actually spoken,
// in their original form, and excludes the partially-spoken amount.
func TestWordTimestampsInterruptionTruncatesToSpoken(t *testing.T) {
	convo := frames.NewLLMContext("system")
	// Speak "Your", "balance", "is", "forty", "two", then hold: the "$42.50" span
	// is only half spoken, so it must not appear.
	words := wordsFor(24000)
	syn := &fakeTimedSynth{rate: 24000, words: words, blockAfter: 5}
	runConversation(t, convo, syn, true)

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Role != frames.RoleAssistant {
		t.Fatalf("messages = %+v, want one assistant message", msgs)
	}
	if msgs[0].Text != "Your balance is" {
		t.Fatalf("assistant text = %q, want %q (spoken words only, amount excluded)", msgs[0].Text, "Your balance is")
	}
}

// TestWordTimestampsInterruptionAfterSpanRecordsAmount proves that interrupting
// after the transformed span completes records the amount in its written form.
func TestWordTimestampsInterruptionAfterSpanRecordsAmount(t *testing.T) {
	convo := frames.NewLLMContext("system")
	// Speak through "cents" (completing the span) then hold.
	words := wordsFor(24000)
	syn := &fakeTimedSynth{rate: 24000, words: words, blockAfter: 9}
	runConversation(t, convo, syn, true)

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Text != "Your balance is $42.50" {
		t.Fatalf("assistant text = %q, want %q", messageText(msgs), "Your balance is $42.50")
	}
}

func messageText(msgs []frames.Message) string {
	if len(msgs) == 0 {
		return ""
	}
	return msgs[0].Text
}

// plainSynth is a Synthesizer with no word timings — the pre-existing contract.
type plainSynth struct {
	rate  int
	chunk []byte
}

func (s *plainSynth) SampleRate() int { return s.rate }

func (s *plainSynth) RunTTS(_ context.Context, _, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	return emit(s.chunk)
}

// TestNoTimestampsRecordsFullLLMText proves a provider without word timings
// behaves exactly as before: the assistant context is driven by the LLM text and
// no TTSTextFrames are involved.
func TestNoTimestampsRecordsFullLLMText(t *testing.T) {
	convo := frames.NewLLMContext("system")
	base := tts.New("PlainTTS", &plainSynth{rate: 24000, chunk: []byte{1, 2, 3, 4}})
	pair := aggregators.New(convo)

	task := pipeline.NewTask(pipeline.New(base, pair.Assistant()), pipeline.TaskParams{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Hello there."))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	time.Sleep(300 * time.Millisecond)
	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Role != frames.RoleAssistant || msgs[0].Text != "Hello there." {
		t.Fatalf("messages = %+v, want one assistant 'Hello there.'", msgs)
	}
}

// TestWordTimestampsCarryPresentationTimestamps proves each spoken word is
// stamped with the moment it is heard, timed from the first chunk of the
// context's audio. That timestamp is what lets the output transport hold the
// word until its audio plays, instead of it landing wherever buffering left it.
func TestWordTimestampsCarryPresentationTimestamps(t *testing.T) {
	syn := &fakeTimedSynth{rate: 24000, words: wordsFor(24000)}
	base := tts.New("FakeTTS", syn)
	base.SetTextFilters(currencyFilter{})

	var mu sync.Mutex
	type stamped struct {
		text string
		pts  int64
		ok   bool
	}
	var got []stamped
	var firstAudio int64 = -1
	task := pipeline.NewTask(pipeline.New(base), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			defer mu.Unlock()
			switch fr := f.(type) {
			case *frames.TTSAudioRawFrame:
				if firstAudio < 0 {
					firstAudio = 0
				}
			case *frames.TTSTextFrame:
				pts, ok := fr.Base().PTS()
				got = append(got, stamped{text: fr.Text, pts: pts, ok: ok})
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Your balance is $42.50"))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	time.Sleep(400 * time.Millisecond)
	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}

	mu.Lock()
	defer mu.Unlock()
	if len(got) == 0 {
		t.Fatal("no word frames reached downstream")
	}
	for _, w := range got {
		if !w.ok {
			t.Fatalf("word %q carries no presentation timestamp", w.text)
		}
	}
	// Words are 0.1s apart, so their timestamps must climb by that much.
	for i := 1; i < len(got); i++ {
		if got[i].pts <= got[i-1].pts {
			t.Fatalf("word %q at %d does not follow %q at %d",
				got[i].text, got[i].pts, got[i-1].text, got[i-1].pts)
		}
	}
	if spread := got[len(got)-1].pts - got[0].pts; spread < int64(100*time.Millisecond) {
		t.Fatalf("word timestamps span %v, want at least one word gap", time.Duration(spread))
	}
}

// asyncSynth answers on its own receive loop rather than inline: RunTTS returns
// having yielded nothing, and the audio reaches the context later. Realtime
// WebSocket providers work this way.
type asyncSynth struct{ rate int }

func (s *asyncSynth) SampleRate() int { return s.rate }

func (s *asyncSynth) RunTTS(_ context.Context, _, _ string, _ func(f frames.Frame) error) error {
	return nil
}

// TestNoTimestampsRecordsTurnWhenAudioArrivesLater covers a provider that
// answers on its own receive loop instead of inline.
//
// With no word timings, the whole-unit text frame is the only thing carrying the
// turn into the conversation: the model's own text is folded into the aggregator
// and never forwarded, and no per-word frames are produced. Emitting it only
// when the provider had answered inline left every turn of such a provider out
// of the context, so each new turn saw nothing but a run of user messages and
// the bot repeated itself.
func TestNoTimestampsRecordsTurnWhenAudioArrivesLater(t *testing.T) {
	convo := frames.NewLLMContext("system")
	base := tts.New("AsyncTTS", &asyncSynth{rate: 24000})
	pair := aggregators.New(convo)

	task := pipeline.NewTask(pipeline.New(base, pair.Assistant()), pipeline.TaskParams{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Hello there."))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	time.Sleep(300 * time.Millisecond)
	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Role != frames.RoleAssistant || msgs[0].Text != "Hello there." {
		t.Fatalf("messages = %+v, want one assistant 'Hello there.': the turn never reached the context", msgs)
	}
}

// A fixed utterance the caller kept out of the conversation must stay out of it,
// whichever way the provider reports what it spoke. The flag is the caller's
// answer for text the service says rather than the assistant: a phrase covering
// a tool call, a stall while something is fetched. Recording it tells the model
// it said something it never composed, and an utterance that announces an action
// with no tool call beside it is a pattern the model reproduces.
//
// Both emission paths carry it. A provider that times its words emits them
// through the sequencer; one that does not emits the unit whole. Stamping only
// one of them leaves every provider on the other side of that line recording
// what it was told not to.
func TestSpeakFrameHonoursAppendToContext(t *testing.T) {
	for _, tc := range []struct {
		name string
		syn  tts.Synthesizer
	}{
		{"audio inline", &plainSynth{rate: 24000, chunk: []byte{1, 2, 3, 4}}},
		{"audio later", &asyncSynth{rate: 24000}},
		{"word timings", &fakeTimedSynth{rate: 24000, words: []timedWord{
			{text: "One", offset: 0, pcm: make([]byte, 4800)},
			{text: "moment", offset: 0.1, pcm: make([]byte, 4800)},
			{text: "please.", offset: 0.2, pcm: make([]byte, 4800)},
		}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			convo := frames.NewLLMContext("system")
			base := tts.New("FixedTTS", tc.syn)
			pair := aggregators.New(convo)

			task := pipeline.NewTask(pipeline.New(base, pair.Assistant()), pipeline.TaskParams{})
			runDone := make(chan error, 1)
			go func() { runDone <- task.Run(context.Background()) }()

			speak := frames.NewTTSSpeakFrame("One moment please.")
			speak.AppendToContext = false
			task.QueueFrame(speak)
			time.Sleep(500 * time.Millisecond)
			task.StopWhenDone()
			select {
			case <-runDone:
			case <-time.After(3 * time.Second):
				t.Fatal("task did not finish")
			}

			if msgs := convo.Messages(); len(msgs) != 0 {
				t.Fatalf("messages = %+v, want none: the caller asked for this utterance "+
					"to stay out of the conversation", msgs)
			}
		})
	}
}

// TestSpeakFrameEntersTheContextOnce covers a fixed utterance on a provider with
// no word timings.
//
// The conversation is built from what was actually spoken, so the utterance
// reaches the context as the whole-unit text frame. The aggregator also records
// a TTSSpeakFrame that asks to be appended, which was guarded against only on
// the word path: once the whole-unit frame was emitted whichever way the audio
// arrived, both wrote and the utterance was said once but stored twice.
func TestSpeakFrameEntersTheContextOnce(t *testing.T) {
	for _, tc := range []struct {
		name string
		syn  tts.Synthesizer
	}{
		{"audio inline", &plainSynth{rate: 24000, chunk: []byte{1, 2, 3, 4}}},
		{"audio later", &asyncSynth{rate: 24000}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			convo := frames.NewLLMContext("system")
			base := tts.New("FixedTTS", tc.syn)
			pair := aggregators.New(convo)

			task := pipeline.NewTask(pipeline.New(base, pair.Assistant()), pipeline.TaskParams{})
			runDone := make(chan error, 1)
			go func() { runDone <- task.Run(context.Background()) }()

			speak := frames.NewTTSSpeakFrame("One moment please.")
			speak.AppendToContext = true
			task.QueueFrame(speak)
			time.Sleep(300 * time.Millisecond)
			task.StopWhenDone()
			select {
			case <-runDone:
			case <-time.After(3 * time.Second):
				t.Fatal("task did not finish")
			}

			msgs := convo.Messages()
			if len(msgs) != 1 {
				t.Fatalf("messages = %+v, want exactly one assistant message", msgs)
			}
			if msgs[0].Role != frames.RoleAssistant || msgs[0].Text != "One moment please." {
				t.Fatalf("message = %+v, want assistant 'One moment please.'", msgs[0])
			}
		})
	}
}

// An utterance the service says on its own has no model response around it to
// mark where the assistant turn ends, so the service closes it: once the last
// word is out it tells the aggregator to commit. Without that the utterance
// would sit in the aggregation until something else happened to close the turn,
// and would be recorded merged with whatever the model said next.
//
// It is sent only for an utterance that belongs in the conversation. One the
// caller kept out of it has nothing to commit.
func TestSpeakFrameClosesTheAssistantTurnItOpened(t *testing.T) {
	for _, tc := range []struct {
		name            string
		appendToContext bool
		want            bool
	}{
		{"recorded", true, true},
		{"kept out of the conversation", false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var mu sync.Mutex
			var pushes int
			synth := &fakeTimedSynth{rate: 24000, words: []timedWord{
				{text: "One", offset: 0, pcm: make([]byte, 4800)},
				{text: "moment.", offset: 0.1, pcm: make([]byte, 4800)},
			}}
			base := tts.New("FixedTTS", synth)
			task := pipeline.NewTask(pipeline.New(base), pipeline.TaskParams{
				OnReachedDownstream: func(f frames.Frame) {
					if _, ok := f.(*frames.LLMAssistantPushAggregationFrame); ok {
						mu.Lock()
						pushes++
						mu.Unlock()
					}
				},
			})
			runDone := make(chan error, 1)
			go func() { runDone <- task.Run(context.Background()) }()

			speak := frames.NewTTSSpeakFrame("One moment.")
			speak.AppendToContext = tc.appendToContext
			task.QueueFrame(speak)
			time.Sleep(500 * time.Millisecond)
			task.StopWhenDone()
			select {
			case <-runDone:
			case <-time.After(3 * time.Second):
				t.Fatal("task did not finish")
			}

			mu.Lock()
			defer mu.Unlock()
			switch {
			case tc.want && pushes != 1:
				t.Fatalf("the assistant turn was closed %d times, want once", pushes)
			case !tc.want && pushes != 0:
				t.Fatalf("the assistant turn was closed %d times for an utterance that "+
					"was never meant to be recorded", pushes)
			}
		})
	}
}
