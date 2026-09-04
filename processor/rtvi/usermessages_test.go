package rtvi_test

import (
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
)

// The message types below have no tests upstream, so the ones here are ours.
// What they pin is the behavior of upstream's implementation, which is what was
// ported.

// texts returns the text of every message of the given type, in order.
func textsOf(msgs []rtvi.Message, msgType string) []string {
	var out []string
	for _, m := range msgs {
		if m.Type != msgType {
			continue
		}
		if d, ok := m.Data.(rtvi.TextData); ok {
			out = append(out, d.Text)
		}
	}
	return out
}

// A conversation frame carries what the user said as the model is about to read
// it, which is not always what the transcription service heard: the turn may
// have been assembled from several transcripts, or written by the client.
func TestUserLLMTextReportsTheMessagePutToTheModel(t *testing.T) {
	convo := frames.NewLLMContext("you are a bot")
	convo.AddUserMessage("what is the weather")

	msgs := observerHarness(t, rtvi.DefaultObserverParams(), frames.NewLLMContextFrame(convo))

	if got := textsOf(msgs, rtvi.TypeUserLLMText); len(got) != 1 || got[0] != "what is the weather" {
		t.Errorf("user-llm-text = %v, want the user's message", got)
	}
}

// A conversation whose last message is not the user's is not the user being put
// to the model, so nothing is reported for it.
func TestUserLLMTextIgnoresAConversationNotEndingOnTheUser(t *testing.T) {
	convo := frames.NewLLMContext("you are a bot")
	convo.AddUserMessage("what is the weather")
	convo.AddMessage(frames.Message{Role: frames.RoleAssistant, Text: "sunny"})

	msgs := observerHarness(t, rtvi.DefaultObserverParams(), frames.NewLLMContextFrame(convo))

	if got := textsOf(msgs, rtvi.TypeUserLLMText); len(got) != 0 {
		t.Errorf("user-llm-text = %v, want nothing: the model is not answering the user", got)
	}
}

func TestUserLLMCategoryCanBeTurnedOff(t *testing.T) {
	convo := frames.NewLLMContext("you are a bot")
	convo.AddUserMessage("what is the weather")
	params := rtvi.DefaultObserverParams()
	params.UserLLMEnabled = off()

	msgs := observerHarness(t, params, frames.NewLLMContextFrame(convo))

	if sent(msgs, rtvi.TypeUserLLMText) {
		t.Errorf("user-llm-text was sent with the category off, got %v", types(msgs))
	}
}

// The user's input being suppressed and released again is reported, so a client
// can show that it is not being listened to.
func TestUserMuteIsReported(t *testing.T) {
	msgs := observerHarness(t, rtvi.DefaultObserverParams(),
		frames.NewUserMuteStartedFrame(), frames.NewUserMuteStoppedFrame())

	for _, want := range []string{rtvi.TypeUserMuteStarted, rtvi.TypeUserMuteStopped} {
		if !sent(msgs, want) {
			t.Errorf("no %s message, got %v", want, types(msgs))
		}
	}
}

func TestUserMuteCategoryCanBeTurnedOff(t *testing.T) {
	params := rtvi.DefaultObserverParams()
	params.UserMuteEnabled = off()

	msgs := observerHarness(t, params,
		frames.NewUserMuteStartedFrame(), frames.NewUserMuteStoppedFrame())

	if sent(msgs, rtvi.TypeUserMuteStarted) || sent(msgs, rtvi.TypeUserMuteStopped) {
		t.Errorf("mute was reported with the category off, got %v", types(msgs))
	}
}

// The bot transcription says the same thing as the model's tokens, a sentence at
// a time, for a client rendering whole sentences rather than a growing stream.
func TestBotTranscriptionIsAssembledFromTheModelsTokens(t *testing.T) {
	msgs := observerHarness(t, rtvi.DefaultObserverParams(),
		frames.NewLLMTextFrame("Hello"),
		frames.NewLLMTextFrame(" there."),
		// A sentence boundary is only confirmed by what follows it.
		frames.NewLLMTextFrame(" Next"),
		frames.NewLLMTextFrame(" one."),
		frames.NewLLMTextFrame(" X"),
	)

	if got := textsOf(msgs, rtvi.TypeBotTranscription); len(got) != 2 ||
		got[0] != "Hello there." || got[1] != " Next one." {
		t.Errorf("bot-transcription = %q, want one message per sentence", got)
	}
	// Every token still goes out on its own channel.
	if got := len(textsOf(msgs, rtvi.TypeBotLLMText)); got != 5 {
		t.Errorf("got %d bot-llm-text messages, want one per token", got)
	}
}

// Turning off what the model produced silences the sentences assembled from it
// as well as the tokens themselves.
func TestBotTranscriptionIsSilencedWithTheModelsText(t *testing.T) {
	params := rtvi.DefaultObserverParams()
	params.BotLLMEnabled = off()

	msgs := observerHarness(t, params,
		frames.NewLLMTextFrame("Hello there."), frames.NewLLMTextFrame(" X"))

	if sent(msgs, rtvi.TypeBotTranscription) || sent(msgs, rtvi.TypeBotLLMText) {
		t.Errorf("the model's text was reported with the category off, got %v", types(msgs))
	}
}
