package tts_test

import (
	"context"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/service/tts"
	uctx "github.com/gojargo/jargo/utils/context"
	"github.com/gojargo/jargo/utils/text"
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
		ReachedDownstreamFilter: pipeline.AnyFrame,
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
				ReachedDownstreamFilter: pipeline.AnyFrame,
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

// The end of a response lands behind the words it ends, and reports the same
// response once.
//
// Word frames carry the moment they are spoken and travel the transport's queue
// for timed frames, which holds each until then. An end frame with no timing of
// its own takes the other queue, so it would overtake the tail of the very
// response it closes: a consumer keying off it, a turn observer or a client's
// bot-stopped event, would see the response end while its last words were still
// being said. It is held until the audio for the turn has been heard and stamped
// with the last word's moment, and the frame held is the frame pushed, so a
// consumer that recognizes a frame it has already seen is not told twice.
func TestResponseEndFollowsTheWordsItEnds(t *testing.T) {
	var mu sync.Mutex
	var order []string
	var endPTS []int64
	var endIDs []uint64

	synth := &fakeTimedSynth{rate: 24000, words: []timedWord{
		{text: "Hello", offset: 0, pcm: make([]byte, 4800)},
		{text: "there.", offset: 0.1, pcm: make([]byte, 4800)},
	}}
	base := tts.New("TimedTTS", synth)
	task := pipeline.NewTask(pipeline.New(base), pipeline.TaskParams{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			defer mu.Unlock()
			switch fr := f.(type) {
			case *frames.TTSTextFrame:
				order = append(order, "word:"+fr.Text)
			case *frames.LLMFullResponseEndFrame:
				order = append(order, "end")
				pts, _ := fr.PTS()
				endPTS = append(endPTS, pts)
				endIDs = append(endIDs, fr.ID())
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Hello there."))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	time.Sleep(600 * time.Millisecond)
	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}

	mu.Lock()
	defer mu.Unlock()
	if len(endIDs) != 1 {
		t.Fatalf("the response was reported as ending %d times, want once: %v", len(endIDs), order)
	}
	if len(order) == 0 || order[len(order)-1] != "end" {
		t.Fatalf("frames reached the pipeline as %v, want the end of the response last", order)
	}
	if endPTS[0] <= 0 {
		t.Fatalf("the end of the response was timed at %d, want the moment of the last word "+
			"so it cannot overtake it", endPTS[0])
	}
}

// Ported from upstream's frame-ordering suite. The frame closing the assistant
// turn has to be timed past the last word of the utterance. Word frames carry
// their moment and travel the transport's queue for timed frames; an untimed
// frame takes the other queue and would overtake them, leaving the last words
// out of the message it commits.
func TestPushAggregationTimedAfterTheLastWord(t *testing.T) {
	var mu sync.Mutex
	var wordPTS []int64
	var pushPTS []int64
	var pushTimed []bool

	synth := &fakeTimedSynth{rate: 24000, words: []timedWord{
		{text: "hello", offset: 0, pcm: make([]byte, 4800)},
		{text: "world", offset: 0.2, pcm: make([]byte, 4800)},
	}}
	base := tts.New("TimedTTS", synth)
	task := pipeline.NewTask(pipeline.New(base), pipeline.TaskParams{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			defer mu.Unlock()
			switch fr := f.(type) {
			case *frames.TTSTextFrame:
				pts, _ := fr.PTS()
				wordPTS = append(wordPTS, pts)
			case *frames.LLMAssistantPushAggregationFrame:
				pts, has := fr.PTS()
				pushPTS = append(pushPTS, pts)
				pushTimed = append(pushTimed, has)
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	speak := frames.NewTTSSpeakFrame("hello world")
	speak.AppendToContext = true
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
	if len(pushPTS) != 1 {
		t.Fatalf("the assistant turn was closed %d times, want once", len(pushPTS))
	}
	if len(wordPTS) == 0 {
		t.Fatal("no spoken words reached the pipeline")
	}
	last := slices.Max(wordPTS)
	if !pushTimed[0] || pushPTS[0] <= last {
		t.Fatalf("the turn was closed at %d, want a moment past the last word at %d",
			pushPTS[0], last)
	}
}

// The other half of the same rule: with no word timings there is nothing to
// come after, and every frame travels the one queue in order, so timing the
// frame that closes the turn would route it through the other queue for no
// reason.
func TestPushAggregationUntimedWithoutWordTimestamps(t *testing.T) {
	var mu sync.Mutex
	var timed []bool

	base := tts.New("PlainTTS", &plainSynth{rate: 24000, chunk: []byte{1, 2, 3, 4}})
	task := pipeline.NewTask(pipeline.New(base), pipeline.TaskParams{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			if fr, ok := f.(*frames.LLMAssistantPushAggregationFrame); ok {
				_, has := fr.PTS()
				mu.Lock()
				timed = append(timed, has)
				mu.Unlock()
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	speak := frames.NewTTSSpeakFrame("hello world")
	speak.AppendToContext = true
	task.QueueFrame(speak)
	time.Sleep(400 * time.Millisecond)
	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}

	mu.Lock()
	defer mu.Unlock()
	if len(timed) != 1 {
		t.Fatalf("the assistant turn was closed %d times, want once", len(timed))
	}
	if timed[0] {
		t.Fatal("the frame closing the turn was timed, with no word timings to come after it")
	}
}

// Ported from upstream. The started frame is what tells the aggregator whether
// this speech opens an assistant turn, so it has to carry the caller's answer
// either way round.
func TestStartedFrameCarriesAppendToContext(t *testing.T) {
	for _, appendToContext := range []bool{true, false} {
		t.Run(map[bool]string{true: "recorded", false: "not recorded"}[appendToContext], func(t *testing.T) {
			var mu sync.Mutex
			var got []bool
			base := tts.New("PlainTTS", &plainSynth{rate: 24000, chunk: []byte{1, 2, 3, 4}})
			task := pipeline.NewTask(pipeline.New(base), pipeline.TaskParams{
				ReachedDownstreamFilter: pipeline.AnyFrame,
				OnReachedDownstream: func(f frames.Frame) {
					if fr, ok := f.(*frames.TTSStartedFrame); ok {
						mu.Lock()
						got = append(got, fr.AppendToContext)
						mu.Unlock()
					}
				},
			})
			runDone := make(chan error, 1)
			go func() { runDone <- task.Run(context.Background()) }()

			speak := frames.NewTTSSpeakFrame("hello world")
			speak.AppendToContext = appendToContext
			task.QueueFrame(speak)
			time.Sleep(400 * time.Millisecond)
			task.StopWhenDone()
			select {
			case <-runDone:
			case <-time.After(3 * time.Second):
				t.Fatal("task did not finish")
			}

			mu.Lock()
			defer mu.Unlock()
			if len(got) != 1 {
				t.Fatalf("%d started frames reached the pipeline, want one", len(got))
			}
			if got[0] != appendToContext {
				t.Fatalf("the started frame says append=%v, want %v", got[0], appendToContext)
			}
		})
	}
}

// Ported from upstream's frame-ordering suite. A language written without
// spaces between words, Chinese or Japanese, has its tokens reported as
// continuous text, so each says it carries whatever spacing separates it from
// the ones around it and a consumer joining them adds none of its own. Without
// that the turn is assembled with a space between every token, which is how the
// conversation ends up holding "こんにちは、私は... AIアシスタントです。" with
// spaces that were never spoken.
func TestWordsCarryingTheirOwnSpacingAssembleWithNone(t *testing.T) {
	var mu sync.Mutex
	var parts []text.Part

	rate := 24000
	chunk := make([]byte, rate/10*2)
	synth := &spacingSynth{rate: rate, words: []timedWord{
		{text: "こんにちは、私はあなたのお手伝いをする", offset: 0, pcm: chunk},
		{text: "AIアシスタントです。", offset: 0.1, pcm: chunk},
	}}
	base := tts.New("CJKTTS", synth)
	task := pipeline.NewTask(pipeline.New(base), pipeline.TaskParams{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			if fr, ok := f.(*frames.TTSTextFrame); ok {
				mu.Lock()
				parts = append(parts, text.Part{
					Text:                    fr.Original(),
					IncludesInterPartSpaces: fr.IncludesInterFrameSpaces,
				})
				mu.Unlock()
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	speak := frames.NewTTSSpeakFrame("こんにちは、私はあなたのお手伝いをするAIアシスタントです。")
	speak.AppendToContext = false
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
	const want = "こんにちは、私はあなたのお手伝いをするAIアシスタントです。"
	if got := text.Concatenate(parts); got != want {
		t.Fatalf("the turn assembles to %q, want %q with no spacing added between tokens", got, want)
	}
}

// spacingSynth reports word timings whose tokens carry their own spacing, the
// way a provider does for a language written without spaces between words.
type spacingSynth struct {
	rate  int
	words []timedWord
}

func (s *spacingSynth) SampleRate() int { return s.rate }

func (s *spacingSynth) RunTTS(_ context.Context, _, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	for _, w := range s.words {
		if err := emit(w.pcm); err != nil {
			return err
		}
	}
	return nil
}

func (s *spacingSynth) RunTTSTimed(
	_ context.Context,
	_, _ string,
	yield func(f frames.Frame) error,
	word func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	for _, w := range s.words {
		err := word([]uctx.WordTiming{{Word: w.text, Offset: w.offset}},
			tts.WordTimingOptions{IncludesInterFrameSpaces: true})
		if err != nil {
			return err
		}
		if err := emit(w.pcm); err != nil {
			return err
		}
	}
	return nil
}
