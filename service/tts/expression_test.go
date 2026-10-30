package tts_test

// How a TTS service reads expression tags in an
// expressive conversation.

import (
	"bytes"
	"context"
	"log/slog"
	"reflect"
	"regexp"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
	uctx "github.com/gojargo/jargo/utils/context"
	"github.com/gojargo/jargo/utils/events"
	ttstext "github.com/gojargo/jargo/utils/text"
)

const written = "<<cheerful>> It's [[laugh]] ^^back up^^ now. [[pause 400]] Bye!"

func reply() []string {
	return []string{"<<cheerful>> It's [[la", "ugh]] ^^back up^^ now. [[pause 400]] Bye!"}
}

// bracketAdapter formats delivery as <delivery> where it changes, sounds as
// <name>, pauses as <Nms>, set apart.
type bracketAdapter struct {
	ttsadapter.BaseExpressionAdapter
}

func (bracketAdapter) Capabilities() *expressive.ExpressionCapabilities {
	return &expressive.ExpressionCapabilities{
		Sounds:       []expressive.ExpressionTerm{{Name: "laugh"}},
		Delivery:     []expressive.ExpressionTerm{{Name: "cheerful"}, {Name: "calm"}},
		PauseRangeMS: &[2]int{100, 1000},
		Emphasis:     true,
	}
}

func (bracketAdapter) FormatText(text string, delivery, previous []string) string {
	if previous == nil {
		previous = []string{}
	}
	if !slices.Equal(delivery, previous) && len(delivery) > 0 {
		return "<" + strings.Join(delivery, ",") + "> " + text
	}
	return text
}

func (bracketAdapter) FormatSound(name string) string { return " <" + name + "> " }

func (bracketAdapter) FormatPause(ms int) string { return " <" + itoa(ms) + "ms> " }

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var digits []byte
	for ; n > 0; n /= 10 {
		digits = append([]byte{byte('0' + n%10)}, digits...)
	}
	return string(digits)
}

// pushTextSynth speaks without word timestamps, answering with its audio.
type pushTextSynth struct {
	adapter ttsadapter.ExpressionAdapter
}

func (s *pushTextSynth) SampleRate() int { return orderingSampleRate }

func (s *pushTextSynth) RunTTS(_ context.Context, _, _ string, yield func(frames.Frame) error) error {
	return yield(frames.NewTTSAudioRawFrame(make([]byte, orderingChunkBytes), orderingSampleRate, 1))
}

func (s *pushTextSynth) ExpressionAdapter() ttsadapter.ExpressionAdapter { return s.adapter }

//nolint:gochecknoglobals // test fixtures
var (
	markupRe      = regexp.MustCompile(`<[^>]*>`)
	spacedPunctRe = regexp.MustCompile(` ([!?.,;:])`)
)

// wordsOf is the words a provider reports for text: its markup left out, set
// apart from the words, which keep their punctuation.
func wordsOf(text string) []string {
	return strings.Fields(spacedPunctRe.ReplaceAllString(markupRe.ReplaceAllString(text, " "), "$1"))
}

// timedSynth reports timestamps for the words it speaks. By default it leaves
// out its markup; reporting markup reports each token it is sent, markup
// included, as ElevenLabs' alignment does; dropLast leaves the last words
// unreported, as a provider can drop events.
type timedSynth struct {
	adapter   ttsadapter.ExpressionAdapter
	reporting bool
	dropLast  int
}

func (s *timedSynth) SampleRate() int { return orderingSampleRate }

func (s *timedSynth) RunTTS(context.Context, string, string, func(frames.Frame) error) error {
	return nil
}

func (s *timedSynth) RunTTSTimed(
	_ context.Context, text, _ string,
	yield func(frames.Frame) error,
	word func([]uctx.WordTiming, tts.WordTimingOptions) error,
) error {
	words := wordsOf(text)
	if s.reporting {
		words = strings.Fields(text)
	}
	words = words[:max(0, len(words)-s.dropLast)]
	timings := make([]uctx.WordTiming, 0, len(words))
	for i, w := range words {
		timings = append(timings, uctx.WordTiming{Word: w, Offset: float64(i) * 0.1})
	}
	if len(timings) > 0 {
		if err := word(timings, tts.WordTimingOptions{}); err != nil {
			return err
		}
	}
	return yield(frames.NewTTSAudioRawFrame(make([]byte, orderingChunkBytes), orderingSampleRate, 1))
}

func (s *timedSynth) ExpressionAdapter() ttsadapter.ExpressionAdapter { return s.adapter }

type serviceCase struct {
	name string
	new  func() tts.Synthesizer
}

func services() []serviceCase {
	return []serviceCase{
		{"push text", func() tts.Synthesizer { return &pushTextSynth{adapter: bracketAdapter{}} }},
		{"word timestamps", func() tts.Synthesizer { return &timedSynth{adapter: bracketAdapter{}} }},
	}
}

func response(chunks ...string) []frames.Frame {
	out := []frames.Frame{frames.NewLLMFullResponseStartFrame()}
	for _, c := range chunks {
		out = append(out, frames.NewLLMTextFrame(c))
	}
	return append(out, frames.NewLLMFullResponseEndFrame())
}

// recordRequests records the text the provider is sent.
func recordRequests(base *tts.Base) func() []string {
	var mu sync.Mutex
	var requests []string
	events.On(base.Events(), tts.EventTTSRequest, func(_ context.Context, r tts.TTSRequest) {
		mu.Lock()
		requests = append(requests, r.Text)
		mu.Unlock()
	})
	return func() []string {
		mu.Lock()
		defer mu.Unlock()
		return append([]string(nil), requests...)
	}
}

// pause is a frame the runner waits on instead of queueing.
type pause struct {
	frames.BaseDataFrame
	d time.Duration
}

func sleep(d time.Duration) frames.Frame {
	return &pause{BaseDataFrame: frames.NewBaseDataFrame("pause"), d: d}
}

// runExpressive runs feed through procs with expressive conversation on, or off
// when plain, and returns the frames that reached each end of the pipeline.
func runExpressive(
	t *testing.T, plain bool, feed []frames.Frame, procs ...processor.Processor,
) (down, up []frames.Frame) {
	t.Helper()
	var mu sync.Mutex
	task := pipeline.NewWorker(pipeline.New(procs...), pipeline.WorkerConfig{
		Params:                  pipeline.Params{EnableExpressiveConversation: !plain},
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		down = append(down, f)
		mu.Unlock()
	})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		up = append(up, f)
		mu.Unlock()
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	for _, f := range feed {
		if p, ok := f.(*pause); ok {
			time.Sleep(p.d)
			continue
		}
		task.QueueFrame(f)
	}
	time.Sleep(100 * time.Millisecond)
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not finish")
	}
	mu.Lock()
	defer mu.Unlock()
	return append([]frames.Frame(nil), down...), append([]frames.Frame(nil), up...)
}

func newBase(syn tts.Synthesizer, mode frames.TextType) *tts.Base {
	base := tts.New("ExpressiveTTS", syn)
	if mode == frames.TextTypeToken {
		base.SetTextAggregator(ttstext.NewTokenAggregator())
	}
	return base
}

// contextAfter is the last message the conversation holds once feed is spoken.
func contextAfter(t *testing.T, base *tts.Base, feed []frames.Frame) string {
	t.Helper()
	convo := frames.NewLLMContext("")
	pair := aggregators.New(convo)
	runExpressive(t, false, feed, base, pair.Assistant())
	msgs := convo.Messages()
	if len(msgs) == 0 {
		t.Fatal("the conversation holds nothing")
	}
	return msgs[len(msgs)-1].Text
}

type spokenFrame struct {
	textType frames.TextType
	text     string
	delivery []string
}

func spokenFrames(down []frames.Frame) []spokenFrame {
	var out []spokenFrame
	for _, f := range down {
		if tf, ok := f.(*frames.TTSTextFrame); ok {
			out = append(out, spokenFrame{tf.TextType, tf.Text, tf.Delivery})
		}
	}
	return out
}

func checkSpoken(t *testing.T, got, want []spokenFrame) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("spoken\n got: %+v\nwant: %+v", got, want)
	}
}

func checkRequests(t *testing.T, got, want []string) {
	t.Helper()
	if !slices.Equal(got, want) {
		t.Fatalf("requests\n got: %q\nwant: %q", got, want)
	}
}

func TestTheTTSIsSentTheWordsWithTheirFormattedExpression(t *testing.T) {
	for _, svc := range services() {
		t.Run(svc.name, func(t *testing.T) {
			base := newBase(svc.new(), frames.TextTypeSentence)
			requests := recordRequests(base)
			runExpressive(t, false, response(reply()...), base)

			checkRequests(t, requests(), []string{
				"<cheerful> It's <laugh> <cheerful,emphasis> back up <cheerful> now.",
				"<400ms> <cheerful> Bye!",
			})
		})
	}
}

func TestAppsGetTheWordsWhileTheContextKeepsTheTags(t *testing.T) {
	convo := frames.NewLLMContext("")
	pair := aggregators.New(convo)
	var mu sync.Mutex
	var stopped []string
	events.On(pair.Assistant().Events(), aggregators.EventAssistantTurnStopped,
		func(_ context.Context, m aggregators.AssistantTurnStopped) {
			mu.Lock()
			stopped = append(stopped, m.Content)
			mu.Unlock()
		})

	down, _ := runExpressive(t, false, response(reply()...),
		newBase(&timedSynth{adapter: bracketAdapter{}}, frames.TextTypeSentence), pair.Assistant())

	msgs := convo.Messages()
	if got := msgs[len(msgs)-1].Text; got != written {
		t.Fatalf("context = %q", got)
	}
	mu.Lock()
	defer mu.Unlock()
	if !slices.Equal(stopped, []string{"It's back up now. Bye!"}) {
		t.Fatalf("turn stopped with %q", stopped)
	}
	var turns []string
	for _, f := range down {
		if tf, ok := f.(*frames.LLMContextAssistantTurnFrame); ok {
			turns = append(turns, tf.Text)
		}
	}
	if !slices.Equal(turns, []string{"It's back up now. Bye!"}) {
		t.Fatalf("assistant turns = %q", turns)
	}
}

func TestTheContextKeepsOnlyTheWordsWhenATaggerWroteTheTags(t *testing.T) {
	for _, switched := range []bool{false, true} {
		name := "tts"
		if switched {
			name = "tts switcher"
		}
		t.Run(name, func(t *testing.T) {
			convo := frames.NewLLMContext("")
			pair := aggregators.New(convo)
			var svc processor.Processor = newBase(&timedSynth{adapter: bracketAdapter{}}, frames.TextTypeSentence)
			if switched {
				sw, err := pipeline.NewServiceSwitcher([]processor.Processor{
					svc, newBase(&timedSynth{adapter: bracketAdapter{}}, frames.TextTypeSentence),
				}, nil)
				if err != nil {
					t.Fatal(err)
				}
				svc = sw
			}

			feed := append([]frames.Frame{frames.NewExpressionMetadataFrame("tagger")}, response(reply()...)...)
			runExpressive(t, false, feed, svc, pair.Assistant())

			msgs := convo.Messages()
			if len(msgs) == 0 {
				t.Fatal("the conversation holds nothing")
			}
			if got := msgs[len(msgs)-1].Text; got != "It's back up now. Bye!" {
				t.Fatalf("context = %q, want the words alone", got)
			}
		})
	}
}

func TestTheContextKeepsTheReplyAsWritten(t *testing.T) {
	for _, svc := range services() {
		t.Run(svc.name, func(t *testing.T) {
			if got := contextAfter(t, newBase(svc.new(), frames.TextTypeSentence), response(reply()...)); got != written {
				t.Fatalf("context = %q", got)
			}
		})
	}
}

func TestWithoutWordTimestampsASentenceCarriesItsDeliveryAndSounds(t *testing.T) {
	down, _ := runExpressive(t, false, response(reply()...),
		newBase(&pushTextSynth{adapter: bracketAdapter{}}, frames.TextTypeSentence))

	// Each sentence is reported whole, in the delivery it starts in, after its
	// sounds.
	checkSpoken(t, spokenFrames(down), []spokenFrame{
		{frames.TextTypeExpressionSound, "laugh", nil},
		{frames.TextTypeSentence, "It's back up now.", []string{"cheerful"}},
		{frames.TextTypeExpressionPause, "400", nil},
		{frames.TextTypeSentence, "Bye!", []string{"cheerful"}},
	})
}

func TestWordTimestampsReportEachWordsDeliveryAndTheSoundsBetween(t *testing.T) {
	down, _ := runExpressive(t, false, response(reply()...),
		newBase(&timedSynth{adapter: bracketAdapter{}}, frames.TextTypeSentence))

	checkSpoken(t, spokenFrames(down), []spokenFrame{
		{frames.TextTypeWord, "It's", []string{"cheerful"}},
		{frames.TextTypeExpressionSound, "laugh", nil},
		{frames.TextTypeWord, "back", []string{"cheerful", "emphasis"}},
		{frames.TextTypeWord, "up", []string{"cheerful", "emphasis"}},
		{frames.TextTypeWord, "now.", []string{"cheerful"}},
		{frames.TextTypeExpressionPause, "400", nil},
		{frames.TextTypeWord, "Bye!", []string{"cheerful"}},
	})
}

// vocabularyAdapter performs the universal warm and laugh by names of its own.
type vocabularyAdapter struct{ bracketAdapter }

func (vocabularyAdapter) Capabilities() *expressive.ExpressionCapabilities {
	return &expressive.ExpressionCapabilities{
		Sounds:   []expressive.ExpressionTerm{{Name: "laughs"}},
		Delivery: []expressive.ExpressionTerm{{Name: "warmly"}},
		ServiceVocabulary: map[expressive.UniversalName][]string{
			expressive.DeliveryWarm: {"warmly"},
			expressive.SoundLaugh:   {"laughs"},
		},
	}
}

func TestUniversalNamesAreSentByTheServicesNamesAndReportedAsWritten(t *testing.T) {
	base := newBase(&timedSynth{adapter: vocabularyAdapter{}}, frames.TextTypeSentence)
	requests := recordRequests(base)
	down, _ := runExpressive(t, false, response("<<warm>> Hi [[laugh]] there."), base)

	checkRequests(t, requests(), []string{"<warmly> Hi <laughs> there."})
	checkSpoken(t, spokenFrames(down), []spokenFrame{
		{frames.TextTypeWord, "Hi", []string{"warm"}},
		{frames.TextTypeExpressionSound, "laugh", nil},
		{frames.TextTypeWord, "there.", []string{"warm"}},
	})
}

func TestASentenceIsOneSegmentShownWithoutItsTags(t *testing.T) {
	down, _ := runExpressive(t, false, response("<<calm>> It's ^^back up^^ now."),
		newBase(&timedSynth{adapter: bracketAdapter{}}, frames.TextTypeSentence))

	type segment struct{ text, raw string }
	var segments []segment
	type progress struct {
		accumulated string
		delivery    []string
	}
	var progresses []progress
	for _, f := range down {
		switch fr := f.(type) {
		case *frames.AggregatedTextFrame:
			if fr.WillBeSpoken {
				segments = append(segments, segment{fr.Text, fr.RawText})
			}
		case *frames.AggregatedTextProgressFrame:
			progresses = append(progresses, progress{fr.AccumulatedText, fr.Delivery})
		}
	}
	if want := []segment{{"It's back up now.", "<<calm>> It's ^^back up^^ now."}}; !reflect.DeepEqual(segments, want) {
		t.Fatalf("segments = %+v", segments)
	}
	// Its progress carries the delivery of each word as it is spoken.
	want := []progress{
		{"It's", []string{"calm"}},
		{"It's back", []string{"calm", "emphasis"}},
		{"It's back up", []string{"calm", "emphasis"}},
		{"It's back up now.", []string{"calm"}},
	}
	if !reflect.DeepEqual(progresses, want) {
		t.Fatalf("progress = %+v", progresses)
	}
}

func TestAUnitThatIsOnlyTagsReachesTheContext(t *testing.T) {
	// A tag with no words of its own, performed or not, is text the LLM wrote.
	for _, svc := range services() {
		for _, tt := range []struct {
			mode   frames.TextType
			chunks []string
		}{
			{frames.TextTypeToken, []string{"<<chee", "rful>>", " It's", " fine."}},
			{frames.TextTypeToken, []string{"It's", " fine.", " [[laugh]]"}},
			{frames.TextTypeToken, []string{"It's", " fine.", " <<calm>>"}},
			{frames.TextTypeSentence, []string{"It's fine. Bye. [[laugh]]"}},
			{frames.TextTypeSentence, []string{"It's fine. Bye. [[snort]]"}},
			{frames.TextTypeSentence, []string{"It's fine. Bye. <<calm>>"}},
		} {
			t.Run(svc.name+"/"+string(tt.mode)+"/"+strings.Join(tt.chunks, ""), func(t *testing.T) {
				if svc.name == "word timestamps" && tt.mode == frames.TextTypeToken {
					// Streaming tokens to a provider that times its words inline,
					// a word can be emitted ahead of the sentence promoted for it at
					// the end of the response, with or without expression tags.
					t.Skip("token mode with inline word timestamps can reorder the words of the last sentence")
				}
				got := contextAfter(t, newBase(svc.new(), tt.mode), response(tt.chunks...))
				if want := strings.Join(tt.chunks, ""); got != want {
					t.Fatalf("context = %q, want %q", got, want)
				}
			})
		}
	}
}

func TestAReplyThatIsOnlyTagsIsRecordedAndEnded(t *testing.T) {
	// It speaks nothing, so no audio context ends it: the reply's end does.
	for _, svc := range services() {
		for _, mode := range []frames.TextType{frames.TextTypeSentence, frames.TextTypeToken} {
			t.Run(svc.name+"/"+string(mode), func(t *testing.T) {
				convo := frames.NewLLMContext("")
				pair := aggregators.New(convo)
				feed := append(response("<<calm>>"), sleep(100*time.Millisecond))
				runExpressive(t, false, append(feed, response("Hi there.")...), newBase(svc.new(), mode), pair.Assistant())

				var texts []string
				for _, m := range convo.Messages() {
					texts = append(texts, m.Text)
				}
				if !slices.Equal(texts, []string{"<<calm>>", "Hi there."}) {
					t.Fatalf("messages = %q", texts)
				}
			})
		}
	}
}

func TestWhitespaceBeforeAnythingIsSaidIsntSentInTokenMode(t *testing.T) {
	// As without expressive conversation: nothing has been said for it to
	// separate.
	for _, svc := range services() {
		for _, tt := range []struct{ first, request string }{
			{"\n", "Hi"}, {" ", "Hi"}, {"<<cheerful>>", "<cheerful> Hi"},
		} {
			t.Run(svc.name+"/"+tt.first, func(t *testing.T) {
				base := newBase(svc.new(), frames.TextTypeToken)
				requests := recordRequests(base)
				runExpressive(t, false, response(tt.first, " ", "Hi", " there."), base)

				checkRequests(t, requests(), []string{tt.request, " there."})
			})
		}
	}
}

func TestAReplyOfOnlyWhitespaceSaysNothingInTokenMode(t *testing.T) {
	for _, svc := range services() {
		t.Run(svc.name, func(t *testing.T) {
			base := newBase(svc.new(), frames.TextTypeToken)
			requests := recordRequests(base)
			down, _ := runExpressive(t, false, response("\n", "\n\n", " "), base)

			checkRequests(t, requests(), nil)
			for _, f := range down {
				if _, ok := f.(*frames.TTSStartedFrame); ok {
					t.Fatal("a context was started")
				}
			}
		})
	}
}

func TestAServiceKeepsItsOwnTagsWholeToo(t *testing.T) {
	// It keeps its own tags whole, as Cartesia keeps <spell> tags.
	base := tts.New("SpellingTTS", &pushTextSynth{adapter: bracketAdapter{}})
	base.SetTextAggregator(ttstext.NewSkipTagsAggregator(frames.TextTypeSentence, "",
		[]ttstext.StartEndTags{{Start: "<spell>", End: "</spell>"}}))
	requests := recordRequests(base)

	runExpressive(t, false, response("Your code is <spell>A. B. C.</spell> okay? <<calm>> Bye."), base)

	checkRequests(t, requests(), []string{"Your code is <spell>A. B. C.</spell> okay?", "<calm> Bye."})
}

func TestTagsSplitAcrossTokensAreReadInTokenMode(t *testing.T) {
	base := newBase(&pushTextSynth{adapter: bracketAdapter{}}, frames.TextTypeToken)
	requests := recordRequests(base)
	got := contextAfter(t, base, response("<<chee", "rful>> It's", " [[la", "ugh]] fine."))

	if joined := strings.Join(requests(), ""); joined != "<cheerful> It's <laugh> fine." {
		t.Fatalf("requests = %q", joined)
	}
	if got != "<<cheerful>> It's [[laugh]] fine." {
		t.Fatalf("context = %q", got)
	}
}

func TestATTSSpeakFrameIsReadToo(t *testing.T) {
	for _, svc := range services() {
		t.Run(svc.name, func(t *testing.T) {
			base := newBase(svc.new(), frames.TextTypeSentence)
			requests := recordRequests(base)
			got := contextAfter(t, base, []frames.Frame{frames.NewTTSSpeakFrame("<<cheerful>> Welcome back!")})

			checkRequests(t, requests(), []string{"<cheerful> Welcome back!"})
			if got != "<<cheerful>> Welcome back!" {
				t.Fatalf("context = %q", got)
			}
		})
	}
}

// trailingSpaceSynth needs its inputs to end in a space.
type trailingSpaceSynth struct{ pushTextSynth }

func (trailingSpaceSynth) RequiresTrailingSpace() bool { return true }

func TestAServiceThatEndsInputsInASpaceGetsOne(t *testing.T) {
	// A service whose inputs are joined verbatim needs the space, or one
	// sentence's last word runs into the next one's first.
	base := newBase(&trailingSpaceSynth{pushTextSynth{adapter: bracketAdapter{}}}, frames.TextTypeSentence)
	requests := recordRequests(base)
	runExpressive(t, false, response(reply()...), base)

	got := requests()
	if len(got) == 0 {
		t.Fatal("nothing was sent")
	}
	for _, r := range got {
		if !strings.HasSuffix(r, " ") {
			t.Fatalf("request %q does not end in a space", r)
		}
	}
}

func TestWordsTheServiceNeverReportsReachTheContextAsWritten(t *testing.T) {
	for _, text := range []string{
		"<<calm>> I'm so sorry your order is late.",
		"<<calm>> Your code is <<slow: A B 7>>, okay?",
	} {
		base := newBase(&timedSynth{adapter: bracketAdapter{}, dropLast: 2}, frames.TextTypeSentence)
		if got := contextAfter(t, base, response(text)); got != text {
			t.Fatalf("context = %q, want %q", got, text)
		}
	}
}

func TestATagCutOffByAnInterruptionIsNeverSpoken(t *testing.T) {
	base := newBase(&pushTextSynth{adapter: bracketAdapter{}}, frames.TextTypeToken)
	requests := recordRequests(base)
	feed := []frames.Frame{
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMTextFrame("Sure "),
		frames.NewLLMTextFrame("[[la"),
		sleep(50 * time.Millisecond),
		frames.NewInterruptionFrame(),
		sleep(50 * time.Millisecond),
	}
	runExpressive(t, false, append(feed, response("ugh]] Okay.")...), base)

	checkRequests(t, requests(), []string{"Sure ", "ugh]] Okay."})
}

func TestTagsAreSpokenAsWrittenWithoutExpressiveConversation(t *testing.T) {
	base := newBase(&pushTextSynth{adapter: bracketAdapter{}}, frames.TextTypeSentence)
	requests := recordRequests(base)
	runExpressive(t, true, response("<<cheerful>> Hi!"), base)

	checkRequests(t, requests(), []string{"<<cheerful>> Hi!"})
}

func TestAServiceWithoutAnAdapterSpeaksOnlyTheWords(t *testing.T) {
	plain := &plainSynth{rate: orderingSampleRate, chunk: make([]byte, orderingChunkBytes)}
	base := newBase(plain, frames.TextTypeSentence)
	requests := recordRequests(base)
	down, _ := runExpressive(t, false, response(reply()...), base)

	checkRequests(t, requests(), []string{"It's back up now.", "Bye!"})
	// What clients see matches what was spoken: no delivery, and no sounds.
	checkSpoken(t, spokenFrames(down), []spokenFrame{
		{frames.TextTypeSentence, "It's back up now.", []string{}},
		{frames.TextTypeSentence, "Bye!", []string{}},
	})
	got := contextAfter(t, newBase(&plainSynth{rate: orderingSampleRate, chunk: make([]byte, 4)},
		frames.TextTypeSentence), response(reply()...))
	if got != written {
		t.Fatalf("context = %q", got)
	}
}

func TestADeliveryTheServiceCantPerformIsLeftOut(t *testing.T) {
	base := newBase(&pushTextSynth{adapter: bracketAdapter{}}, frames.TextTypeSentence)
	requests := recordRequests(base)
	down, _ := runExpressive(t, false, response("<<whisper>> Shh."), base)

	checkRequests(t, requests(), []string{"Shh."})
	checkSpoken(t, spokenFrames(down), []spokenFrame{{frames.TextTypeSentence, "Shh.", []string{}}})
}

func TestEachUnperformedTagIsWarnedAboutAsWritten(t *testing.T) {
	var logs bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(prev) })

	runExpressive(t, false, response(reply()...),
		newBase(&plainSynth{rate: orderingSampleRate, chunk: make([]byte, 4)}, frames.TextTypeSentence))

	var unperformed []string
	for line := range strings.SplitSeq(logs.String(), "\n") {
		if !strings.Contains(line, "can't perform") {
			continue
		}
		_, tag, _ := strings.Cut(line, "tag=")
		unperformed = append(unperformed, strings.Trim(tag, `"`))
	}
	if want := []string{"<<cheerful>>", "[[laugh]]", "^^emphasis^^", "[[pause 400]]"}; !slices.Equal(unperformed, want) {
		t.Fatalf("warned about %q, want %q", unperformed, want)
	}
}

func TestASoundTheServiceCantPerformIsLeftOut(t *testing.T) {
	base := newBase(&pushTextSynth{adapter: bracketAdapter{}}, frames.TextTypeSentence)
	requests := recordRequests(base)
	got := contextAfter(t, base, response("Ha [[snort]] okay."))

	checkRequests(t, requests(), []string{"Ha okay."})
	// The context keeps the reply as written.
	if got != "Ha [[snort]] okay." {
		t.Fatalf("context = %q", got)
	}
}

func TestAPauseIsKeptWithinTheServicesRange(t *testing.T) {
	base := newBase(&pushTextSynth{adapter: bracketAdapter{}}, frames.TextTypeSentence)
	requests := recordRequests(base)
	runExpressive(t, false, response("One. [[pause 5000]] Two."), base)

	checkRequests(t, requests(), []string{"One.", "<1000ms> Two."})
}

// styleAdapter is for a service that takes delivery with each request.
type styleAdapter struct {
	ttsadapter.BaseExpressionAdapter
}

func (styleAdapter) Capabilities() *expressive.ExpressionCapabilities {
	return &expressive.ExpressionCapabilities{
		Delivery:      []expressive.ExpressionTerm{{Name: "calm"}, {Name: "cheerful"}},
		DeliveryScope: expressive.DeliveryScopeSentence,
	}
}

func (styleAdapter) RequestSettings(delivery []string) map[string]any {
	if len(delivery) == 0 {
		return nil
	}
	return map[string]any{"style": strings.Join(delivery, " ")}
}

// styleSynth records the delivery each request is spoken in, as a style.
type styleSynth struct {
	pushTextSynth
	base   *tts.Base
	mu     sync.Mutex
	styles [][2]string
}

func (s *styleSynth) RunTTS(ctx context.Context, text, contextID string, yield func(frames.Frame) error) error {
	style, ok := s.base.ExpressionSettings()["style"].(string)
	if !ok {
		style = "plain"
	}
	s.mu.Lock()
	s.styles = append(s.styles, [2]string{text, style})
	s.mu.Unlock()
	return s.pushTextSynth.RunTTS(ctx, text, contextID, yield)
}

func newStyleService() (*tts.Base, *styleSynth) {
	syn := &styleSynth{}
	syn.adapter = styleAdapter{}
	syn.base = tts.New("StyleTTS", syn)
	return syn.base, syn
}

func (s *styleSynth) recorded() [][2]string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([][2]string(nil), s.styles...)
}

func TestEachRequestIsGivenTheDeliveryItIsSpokenIn(t *testing.T) {
	base, syn := newStyleService()
	feed := response("One. <<calm>> Two. Three. <<neutral>> Four. <<cheerful>> Five.")
	runExpressive(t, false, append(feed, response("Six.")...), base)

	want := [][2]string{
		{"One.", "plain"},
		{"Two.", "calm"},
		{"Three.", "calm"},
		{"Four.", "plain"},
		{"Five.", "cheerful"},
		{"Six.", "plain"},
	}
	if got := syn.recorded(); !reflect.DeepEqual(got, want) {
		t.Fatalf("styles = %q", got)
	}
}

func TestEachSentenceStatesItsDelivery(t *testing.T) {
	// Not every service keeps delivery from one request to the next.
	base := newBase(&pushTextSynth{adapter: bracketAdapter{}}, frames.TextTypeSentence)
	requests := recordRequests(base)
	runExpressive(t, false, response("<<calm>> One. Two. <<neutral>> Three."), base)

	checkRequests(t, requests(), []string{"<calm> One.", "<calm> Two.", "Three."})
}

func TestStreamedTokensStateDeliveryOnlyWhereItChanges(t *testing.T) {
	base := newBase(&pushTextSynth{adapter: bracketAdapter{}}, frames.TextTypeToken)
	requests := recordRequests(base)
	runExpressive(t, false, response("<<calm>> One.", " Two."), base)

	checkRequests(t, requests(), []string{"<calm> One.", " Two."})
}

func TestEachResponseStartsFromTheBaselineWhenStreamingTokens(t *testing.T) {
	base := newBase(&pushTextSynth{adapter: bracketAdapter{}}, frames.TextTypeToken)
	requests := recordRequests(base)
	// The first response ends in text the parser holds until its end.
	runExpressive(t, false, append(response("<<calm>> One ^^two"), response("<<calm>> Three.")...), base)

	got := requests()
	if len(got) == 0 || got[len(got)-1] != "<calm> Three." {
		t.Fatalf("requests = %q", got)
	}
}

func TestASpanSetsTheDeliveryOfItsWholeRequest(t *testing.T) {
	base, syn := newStyleService()
	runExpressive(t, false, response("Your code is <<calm: A B 7>>. Anything else?"), base)

	want := [][2]string{{"Your code is A B 7.", "calm"}, {"Anything else?", "plain"}}
	if got := syn.recorded(); !reflect.DeepEqual(got, want) {
		t.Fatalf("styles = %q", got)
	}
}

func TestARequestIsSpokenInTheDeliveryItsWordsStartIn(t *testing.T) {
	base, syn := newStyleService()
	down, _ := runExpressive(t, false, response("The code is <<calm>> four two. Anything else?"), base)

	// The change takes effect from the next request, as a delivery lasts until
	// the next tag.
	want := [][2]string{{"The code is four two.", "plain"}, {"Anything else?", "calm"}}
	if got := syn.recorded(); !reflect.DeepEqual(got, want) {
		t.Fatalf("styles = %q", got)
	}
	checkSpoken(t, spokenFrames(down), []spokenFrame{
		{frames.TextTypeSentence, "The code is four two.", []string{}},
		{frames.TextTypeSentence, "Anything else?", []string{"calm"}},
	})
}

// markupAdapter formats tags as bracketed markup, which its service reports
// back with its words.
type markupAdapter struct {
	ttsadapter.BaseExpressionAdapter
}

func (markupAdapter) Capabilities() *expressive.ExpressionCapabilities {
	return &expressive.ExpressionCapabilities{
		Sounds:       []expressive.ExpressionTerm{{Name: "laugh"}},
		Delivery:     []expressive.ExpressionTerm{{Name: "calm"}},
		PauseRangeMS: &[2]int{100, 1000},
	}
}

func (markupAdapter) FormatText(text string, delivery, previous []string) string {
	if previous == nil {
		previous = []string{}
	}
	if len(delivery) > 0 && !slices.Equal(delivery, previous) {
		return "[" + strings.Join(delivery, " ") + "] " + text
	}
	return text
}

func (markupAdapter) FormatSound(name string) string { return " [" + name + "] " }

func (markupAdapter) FormatPause(int) string { return " [pause] " }

func reportingSynth() tts.Synthesizer { return &timedSynth{adapter: markupAdapter{}, reporting: true} }

type timedWord2 struct {
	text string
	pts  int64
}

// spokenWithWarnings runs text through syn and returns the text frames, with
// their timestamps, and what was logged as a warning.
func spokenWithWarnings(t *testing.T, syn tts.Synthesizer, text string) ([]timedWord2, string) {
	t.Helper()
	var logs bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelWarn})))
	down, _ := runExpressive(t, false, response(text), newBase(syn, frames.TextTypeSentence))
	slog.SetDefault(prev)
	var words []timedWord2
	for _, f := range down {
		if tf, ok := f.(*frames.TTSTextFrame); ok {
			pts, _ := tf.PTS()
			words = append(words, timedWord2{tf.Text, pts})
		}
	}
	return words, logs.String()
}

func textsOf(words []timedWord2) []string {
	out := make([]string, 0, len(words))
	for _, w := range words {
		out = append(out, w.text)
	}
	return out
}

func TestASoundIsTimedByTheMarkupItsServiceReports(t *testing.T) {
	words, warnings := spokenWithWarnings(t, reportingSynth(), "It's [[laugh]] back.")

	// The markup the service reports times the sound.
	if got := textsOf(words); !slices.Equal(got, []string{"It's", "laugh", "back."}) {
		t.Fatalf("words = %q", got)
	}
	if words[0].pts >= words[1].pts || words[1].pts >= words[2].pts {
		t.Fatalf("timestamps out of order: %+v", words)
	}
	if strings.Contains(warnings, "dropping") {
		t.Fatalf("a word was dropped:\n%s", warnings)
	}
}

func TestASoundItsServiceReportsIsReportedAsPerformed(t *testing.T) {
	base := newBase(reportingSynth(), frames.TextTypeSentence)
	down, _ := runExpressive(t, false, response("It's [[laugh]] back."), base)

	for _, f := range down {
		if tf, ok := f.(*frames.TTSTextFrame); ok && tf.TextType == frames.TextTypeExpressionSound {
			// Clients are told a sound was performed when it is spoken, in a
			// segment of its own.
			if !tf.WillBeSpoken || tf.SegmentID != 0 {
				t.Fatalf("sound = %+v", tf)
			}
			return
		}
	}
	t.Fatal("no sound was reported")
}

func TestReportedDeliveryMarkupIsNotSpoken(t *testing.T) {
	words, warnings := spokenWithWarnings(t, reportingSynth(), "<<calm>> It's [[laugh]] back.")

	if got := textsOf(words); !slices.Equal(got, []string{"It's", "laugh", "back."}) {
		t.Fatalf("words = %q", got)
	}
	if strings.Contains(warnings, "dropping") {
		t.Fatalf("a word was dropped:\n%s", warnings)
	}
}

func timedServices() []serviceCase {
	return []serviceCase{
		{"reporting markup", reportingSynth},
		{"word timestamps", func() tts.Synthesizer { return &timedSynth{adapter: bracketAdapter{}} }},
	}
}

func TestASpansWordsAreSpokenInItsDelivery(t *testing.T) {
	for _, svc := range timedServices() {
		for _, tt := range []struct {
			text   string
			spoken []spokenFrame
		}{
			{"<<calm>> It's <<neutral: A B 7>>, okay.", []spokenFrame{
				{frames.TextTypeWord, "It's", []string{"calm"}},
				{frames.TextTypeWord, "A", []string{}},
				{frames.TextTypeWord, "B", []string{}},
				{frames.TextTypeWord, "7,", []string{}},
				{frames.TextTypeWord, "okay.", []string{"calm"}},
			}},
			{"Email <<calm: docs@acme.com>> today.", []spokenFrame{
				{frames.TextTypeWord, "Email", []string{}},
				{frames.TextTypeWord, "docs@acme.com", []string{"calm"}},
				{frames.TextTypeWord, "today.", []string{}},
			}},
		} {
			t.Run(svc.name+"/"+tt.text, func(t *testing.T) {
				down, _ := runExpressive(t, false, response(tt.text), newBase(svc.new(), frames.TextTypeSentence))

				checkSpoken(t, spokenFrames(down), tt.spoken)
				if got := contextAfter(t, newBase(svc.new(), frames.TextTypeSentence), response(tt.text)); got != tt.text {
					t.Fatalf("context = %q", got)
				}
			})
		}
	}
}

func TestASoundBeforePunctuationIsReportedAfterTheWordsItFollows(t *testing.T) {
	for _, svc := range timedServices() {
		for _, tt := range []struct {
			text   string
			spoken []string
		}{
			{"Hi [[laugh]], there.", []string{"Hi,", "laugh", "there."}},
			{"It's ^^back^^ [[laugh]]! Bye.", []string{"It's", "back!", "laugh", "Bye."}},
			{"Really [[laugh]]? Yes.", []string{"Really?", "laugh", "Yes."}},
			{"Wait [[pause 300]]. Okay.", []string{"Wait.", "300", "Okay."}},
			{"See you [[laugh]]", []string{"See", "you", "laugh"}},
		} {
			t.Run(svc.name+"/"+tt.text, func(t *testing.T) {
				words, warnings := spokenWithWarnings(t, svc.new(), tt.text)

				if got := textsOf(words); !slices.Equal(got, tt.spoken) {
					t.Fatalf("words = %q, want %q", got, tt.spoken)
				}
				if strings.Contains(warnings, "dropping") {
					t.Fatalf("a word was dropped:\n%s", warnings)
				}
				if got := contextAfter(t, newBase(svc.new(), frames.TextTypeSentence), response(tt.text)); got != tt.text {
					t.Fatalf("context = %q", got)
				}
			})
		}
	}
}

func TestPunctuationAfterEmphasisStaysWithItInTheContext(t *testing.T) {
	for _, svc := range services() {
		t.Run(svc.name, func(t *testing.T) {
			got := contextAfter(t, newBase(svc.new(), frames.TextTypeSentence), response("It's ^^back^^! ^^Wow^^, nice."))
			if got != "It's ^^back^^! ^^Wow^^, nice." {
				t.Fatalf("context = %q", got)
			}
		})
	}
}

func TestSpacingBetweenPiecesIsSentAsWritten(t *testing.T) {
	base := newBase(&plainSynth{rate: orderingSampleRate, chunk: make([]byte, 4)}, frames.TextTypeSentence)
	requests := recordRequests(base)
	runExpressive(t, false, response("That is ^^very^^ ^^good^^ news."), base)

	checkRequests(t, requests(), []string{"That is very good news."})
}

func TestSpacingBetweenPiecesIsSentAsWrittenWhenStreamingTokens(t *testing.T) {
	base := newBase(&plainSynth{rate: orderingSampleRate, chunk: make([]byte, 4)}, frames.TextTypeToken)
	requests := recordRequests(base)
	runExpressive(t, false, response("^^Wow^^ that's great."), base)

	if got := strings.Join(requests(), ""); got != "Wow that's great." {
		t.Fatalf("requests = %q", got)
	}
}

func TestAResponseAfterATTSSpeakFrameStatesItsDelivery(t *testing.T) {
	base := newBase(&pushTextSynth{adapter: bracketAdapter{}}, frames.TextTypeToken)
	requests := recordRequests(base)
	feed := []frames.Frame{frames.NewTTSSpeakFrame("<<calm>> Hi.")}
	runExpressive(t, false, append(feed, response("<<calm>> Three.")...), base)

	checkRequests(t, requests(), []string{"<calm> Hi.", "<calm> Three."})
}

// streamingSynth streams each word's audio and timestamp after the request, as
// a WebSocket service does.
type streamingSynth struct {
	mu   sync.Mutex
	host tts.AudioContextHost
	wg   sync.WaitGroup
}

func (s *streamingSynth) SampleRate() int { return orderingSampleRate }

func (s *streamingSynth) SetAudioContextHost(h tts.AudioContextHost) {
	s.mu.Lock()
	s.host = h
	s.mu.Unlock()
}

func (s *streamingSynth) ExpressionAdapter() ttsadapter.ExpressionAdapter { return bracketAdapter{} }

func (s *streamingSynth) RunTTS(context.Context, string, string, func(frames.Frame) error) error {
	return nil
}

func (s *streamingSynth) RunTTSTimed(
	_ context.Context, text, contextID string,
	_ func(frames.Frame) error,
	_ func([]uctx.WordTiming, tts.WordTimingOptions) error,
) error {
	s.mu.Lock()
	host := s.host
	s.mu.Unlock()
	words := wordsOf(text)
	s.wg.Go(func() {
		for i, word := range words {
			time.Sleep(50 * time.Millisecond)
			host.AppendToAudioContext(contextID,
				frames.NewTTSAudioRawFrame(make([]byte, 320), orderingSampleRate, 1))
			host.AddWordTimestamps(contextID, []uctx.WordTiming{{Word: word, Offset: float64(i) * 0.05}},
				tts.WordTimingOptions{})
		}
		host.AppendToAudioContext(contextID, frames.NewTTSStoppedFrame())
		host.RemoveAudioContext(contextID)
	})
	return nil
}

func TestUninterruptibleSpeechKeepsItsDeliveryThroughAnInterruption(t *testing.T) {
	speech := frames.NewTTSSpeakFrame("<<calm>> One two three four five six.")
	speech.SetInterruptible(false)
	syn := &streamingSynth{}
	base := tts.New("StreamingTTS", syn)

	down, _ := runExpressive(t, false, []frames.Frame{
		speech, sleep(150 * time.Millisecond), frames.NewInterruptionFrame(), sleep(800 * time.Millisecond),
	}, base)
	syn.wg.Wait()

	var want []spokenFrame
	for _, word := range []string{"One", "two", "three", "four", "five", "six."} {
		want = append(want, spokenFrame{frames.TextTypeWord, word, []string{"calm"}})
	}
	checkSpoken(t, spokenFrames(down), want)
}

// declaringSynth never speaks, and declares the given capabilities.
type declaringSynth struct {
	capabilities *expressive.ExpressionCapabilities
}

type declaringAdapter struct {
	ttsadapter.BaseExpressionAdapter
	capabilities *expressive.ExpressionCapabilities
}

func (a declaringAdapter) Capabilities() *expressive.ExpressionCapabilities { return a.capabilities }

func (s *declaringSynth) SampleRate() int { return orderingSampleRate }

func (s *declaringSynth) RunTTS(context.Context, string, string, func(frames.Frame) error) error {
	return nil
}

func (s *declaringSynth) ExpressionAdapter() ttsadapter.ExpressionAdapter {
	return declaringAdapter{capabilities: s.capabilities}
}

func metadataFrames(fs []frames.Frame) []*frames.TTSMetadataFrame {
	var out []*frames.TTSMetadataFrame
	for _, f := range fs {
		if m, ok := f.(*frames.TTSMetadataFrame); ok {
			out = append(out, m)
		}
	}
	return out
}

func TestMetadataIsBroadcastBothWaysAtStart(t *testing.T) {
	// The LLM sits upstream of the TTS, so the metadata goes both ways.
	base := tts.New("SilentTTS", &declaringSynth{})
	down, up := runExpressive(t, false, nil, base)

	d, u := metadataFrames(down), metadataFrames(up)
	if len(d) != 1 || len(u) != 1 || d[0].ServiceName != base.Name() || u[0].ServiceName != base.Name() {
		t.Fatalf("metadata down %v, up %v", d, u)
	}
}

func TestAServiceDeclaresNoExpressionByDefault(t *testing.T) {
	down, _ := runExpressive(t, false, nil, tts.New("SilentTTS", &declaringSynth{}))

	if d := metadataFrames(down); len(d) != 1 || d[0].Expression != nil {
		t.Fatalf("metadata = %v", d)
	}
}

func TestAServiceBroadcastsTheExpressionItDeclares(t *testing.T) {
	capabilities := &expressive.ExpressionCapabilities{
		Sounds:        []expressive.ExpressionTerm{{Name: "laugh", Description: "Something is genuinely funny."}},
		Delivery:      []expressive.ExpressionTerm{{Name: "cheerful"}, {Name: "whisper"}},
		PauseRangeMS:  &[2]int{100, 3000},
		Emphasis:      true,
		DeliveryScope: expressive.DeliveryScopeSentence,
	}
	down, _ := runExpressive(t, false, nil, tts.New("SilentTTS", &declaringSynth{capabilities: capabilities}))

	if d := metadataFrames(down); len(d) != 1 || !reflect.DeepEqual(d[0].Expression, capabilities) {
		t.Fatalf("metadata = %v", d)
	}
}

// modelSynth performs expression on one model only, as ElevenLabs performs
// audio tags on Eleven v3 alone.
type modelSynth struct{ settingsSynth }

func (s *modelSynth) ExpressionAdapter() ttsadapter.ExpressionAdapter {
	s.mu.Lock()
	defer s.mu.Unlock()
	if model, _ := s.store.Model.Value(); model == "expressive" {
		return bracketAdapter{}
	}
	return ttsadapter.BaseExpressionAdapter{}
}

func TestAModelChangeToOneWithoutExpressionIsBroadcast(t *testing.T) {
	syn := &modelSynth{settingsSynth{store: settings.TTS{Model: settings.Set("expressive")}}}

	_, up := runExpressive(t, false, []frames.Frame{
		frames.NewTTSUpdateSettingsFrame(&settings.TTS{Model: settings.Set("plain")}),
	}, tts.New("ModelTTS", syn))

	metadata := metadataFrames(up)
	if len(metadata) != 2 || metadata[0].Expression == nil || metadata[1].Expression != nil {
		t.Fatalf("metadata = %v, want the expression at start and none after the change", metadata)
	}
}

func TestNothingIsBroadcastWithoutExpressiveConversation(t *testing.T) {
	capabilities := &expressive.ExpressionCapabilities{Emphasis: true}
	down, up := runExpressive(t, true, nil, tts.New("SilentTTS", &declaringSynth{capabilities: capabilities}))

	if got := metadataFrames(append(down, up...)); len(got) != 0 {
		t.Fatalf("metadata = %v", got)
	}
}
