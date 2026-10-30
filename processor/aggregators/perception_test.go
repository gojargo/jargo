package aggregators_test

// How the user aggregator writes how the user spoke into the conversation.

import (
	"context"
	"reflect"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/utils/events"
)

//nolint:gochecknoglobals // test fixtures
var (
	expressiveOn = pipeline.Params{EnableExpressiveConversation: true}
	frustrated   = []expressive.PerceptionSignal{{Name: "tone", Labels: []expressive.PerceptionLabel{
		{Label: "frustrated", Confidence: 0.86}, {Label: "neutral", Confidence: 0.10},
	}}}
	neutralTone = []expressive.PerceptionSignal{{Name: "tone", Labels: []expressive.PerceptionLabel{
		{Label: "neutral", Confidence: 0.81}, {Label: "positive", Confidence: 0.12},
	}}}
)

// frustratedMessage is the user message for the words of these tests, read as
// frustrated.
const frustratedMessage = "([tone frustrated=0.86, neutral=0.10]) Can you tell me where my order is?"

func readTranscript(words string, perception []expressive.PerceptionSignal) frames.Frame {
	f := frames.NewTranscriptionFrame(words, "", "now")
	f.Perception = perception
	return f
}

func readInterim(words string, perception []expressive.PerceptionSignal) frames.Frame {
	f := frames.NewInterimTranscriptionFrame(words, "", "now")
	f.Perception = perception
	return f
}

// readTurn is one user turn of words read as perception, with before sent
// between the user starting to speak and the transcript.
func readTurn(words string, perception []expressive.PerceptionSignal, before ...frames.Frame) []any {
	turn := []any{vadStart(), flush{}}
	for _, f := range before {
		turn = append(turn, f)
	}
	return append(turn, readTranscript(words, perception), pause(50*time.Millisecond), vadStop(),
		pause(speechTimeout+200*time.Millisecond))
}

func speechTimeoutPair(convo *frames.LLMContext) *aggregators.Pair {
	return aggregators.New(convo, aggregators.WithTurns(turns.Config{
		Strategies: turns.UserTurnStrategies{Stop: []turns.StopStrategy{
			turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{UserSpeechTimeout: speechTimeout}),
		}},
	}))
}

// userMessages runs the turns through a user aggregator and returns the user
// messages the conversation holds, and the frames that went downstream.
func userMessages(t *testing.T, params pipeline.Params, turnsToSend ...[]any) ([]string, []frames.Frame) {
	t.Helper()
	convo := frames.NewLLMContext("")
	pair := speechTimeoutPair(convo)
	down := sendAndCollectWith(t, params, []processor.Processor{pair.User()}, slices.Concat(turnsToSend...)...)
	var said []string
	for _, m := range convo.Messages() {
		if m.Role == frames.RoleUser {
			said = append(said, m.Text)
		}
	}
	return said, down
}

func userTurnFrames(down []frames.Frame) []*frames.LLMContextUserTurnFrame {
	var out []*frames.LLMContextUserTurnFrame
	for _, f := range down {
		if tf, ok := f.(*frames.LLMContextUserTurnFrame); ok {
			out = append(out, tf)
		}
	}
	return out
}

func TestTheReadingIsWrittenAtTheHeadOfTheUserMessage(t *testing.T) {
	said, down := userMessages(t, expressiveOn, readTurn("Can you tell me where my order is?", frustrated))

	if want := []string{frustratedMessage}; !slices.Equal(said, want) {
		t.Errorf("user messages = %q, want %q", said, want)
	}
	turn := userTurnFrames(down)
	if len(turn) == 0 || turn[0].Text != "Can you tell me where my order is?" ||
		!reflect.DeepEqual(turn[0].Perception, frustrated) {
		t.Errorf("user turn frames = %+v, want the words with their reading", turn)
	}
}

func TestNoReadingIsWrittenWithoutExpressiveConversation(t *testing.T) {
	said, down := userMessages(t, pipeline.Params{}, readTurn("Can you tell me where my order is?", frustrated))

	if want := []string{"Can you tell me where my order is?"}; !slices.Equal(said, want) {
		t.Errorf("user messages = %q, want %q", said, want)
	}
	if turn := userTurnFrames(down); len(turn) == 0 || turn[0].Perception != nil {
		t.Errorf("user turn frames = %+v, want no reading", turn)
	}
}

func TestAProvisionalReadingIsNotWritten(t *testing.T) {
	said, _ := userMessages(t, expressiveOn,
		readTurn("Can you tell me where my order is?", nil, readInterim("Can you tell me", frustrated)))

	if want := []string{"Can you tell me where my order is?"}; !slices.Equal(said, want) {
		t.Errorf("user messages = %q, want %q", said, want)
	}
}

func TestTheTurnsLatestFinalReadingIsWritten(t *testing.T) {
	said, _ := userMessages(t, expressiveOn,
		readTurn("where my order is?", frustrated, readTranscript("Can you tell me", neutralTone)))

	if want := []string{frustratedMessage}; !slices.Equal(said, want) {
		t.Errorf("user messages = %q, want %q", said, want)
	}
}

func TestAReadingBelongsToItsOwnTurn(t *testing.T) {
	said, _ := userMessages(t, expressiveOn,
		readTurn("Can you tell me where my order is?", frustrated), readTurn("Thanks.", nil))

	want := []string{frustratedMessage, "Thanks."}
	if !slices.Equal(said, want) {
		t.Errorf("user messages = %q, want %q", said, want)
	}
}

// earlyTurn is the user message an early reply is generated from, and the turn
// broadcast ahead of it.
func earlyTurn(t *testing.T, params pipeline.Params, perception []expressive.PerceptionSignal) (
	string, *frames.LLMContextUserTurnFrame,
) {
	t.Helper()
	pair := aggregators.New(frames.NewLLMContext(""), aggregators.WithTurns(turns.Config{
		Strategies: turns.EagerStrategies(turns.EagerStrategiesConfig{}),
	}))
	eager := frames.NewEagerTranscriptionFrame("Can you tell me where my order is?", "", "now")
	eager.Perception = perception
	down := sendAndCollectWith(t, params, []processor.Processor{pair.User()},
		frames.NewProposedUserStartedSpeakingFrame(), pause(50*time.Millisecond), eager, pause(50*time.Millisecond))

	var message string
	for _, cf := range contextFramesIn(down) {
		if cf.Speculation {
			msgs := cf.Context.Messages()
			message = msgs[len(msgs)-1].Text
			break
		}
	}
	for _, tf := range userTurnFrames(down) {
		if tf.Speculation {
			return message, tf
		}
	}
	t.Fatal("no speculative user turn was broadcast")
	return "", nil
}

func TestAnEarlyReplyReadsTheEagerReading(t *testing.T) {
	message, turn := earlyTurn(t, expressiveOn, frustrated)

	if want := frustratedMessage; message != want {
		t.Errorf("early message = %q, want %q", message, want)
	}
	if !reflect.DeepEqual(turn.Perception, frustrated) {
		t.Errorf("speculative turn reading = %+v, want %+v", turn.Perception, frustrated)
	}
}

func TestAnEagerReadingIsNotWrittenWithoutExpressiveConversation(t *testing.T) {
	message, turn := earlyTurn(t, pipeline.Params{}, frustrated)

	if want := "Can you tell me where my order is?"; message != want {
		t.Errorf("early message = %q, want %q", message, want)
	}
	if turn.Perception != nil {
		t.Errorf("speculative turn reading = %+v, want none", turn.Perception)
	}
}

// stoppedTurns runs the turns and returns the reports of each one stopping.
func stoppedTurns(t *testing.T, params pipeline.Params, turnsToSend ...[]any) []aggregators.UserTurnStopped {
	t.Helper()
	pair := speechTimeoutPair(frames.NewLLMContext(""))
	var mu sync.Mutex
	var stopped []aggregators.UserTurnStopped
	events.On(pair.User().Events(), aggregators.EventUserTurnStopped,
		func(_ context.Context, s aggregators.UserTurnStopped) {
			mu.Lock()
			stopped = append(stopped, s)
			mu.Unlock()
		})
	sendAndCollectWith(t, params, []processor.Processor{pair.User()}, slices.Concat(turnsToSend...)...)
	mu.Lock()
	defer mu.Unlock()
	return stopped
}

func TestTheTurnStopsWithItsFinalReading(t *testing.T) {
	stopped := stoppedTurns(t, expressiveOn,
		readTurn("Can you tell me where my order is?", frustrated), readTurn("Thanks.", nil))

	if len(stopped) != 2 || !reflect.DeepEqual(stopped[0].Perception, frustrated) || stopped[1].Perception != nil {
		t.Errorf("turns stopped = %+v, want the first with its reading and the second with none", stopped)
	}
}

func TestAProvisionalReadingIsNotReportedWithTheTurn(t *testing.T) {
	stopped := stoppedTurns(t, expressiveOn,
		readTurn("Can you tell me where my order is?", nil, readInterim("Can you tell me", neutralTone)))

	if len(stopped) == 0 || stopped[0].Perception != nil {
		t.Errorf("turns stopped = %+v, want no reading", stopped)
	}
}

func TestNoReadingIsReportedWithoutExpressiveConversation(t *testing.T) {
	stopped := stoppedTurns(t, pipeline.Params{}, readTurn("Can you tell me where my order is?", frustrated))

	if len(stopped) == 0 || stopped[0].Perception != nil {
		t.Errorf("turns stopped = %+v, want no reading", stopped)
	}
}
