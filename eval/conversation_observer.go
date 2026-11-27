package eval

import (
	"context"
	"encoding/json"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/rtvi"
)

// The observer that records a bot's conversation as a conversation file holds
// it. It lets a call the harness didn't drive, such as a production one, be
// judged with `jargo eval judge` against a rubric (Rubric) or a simulation's
// success and metrics.

// EventConversationEnded fires once the pipeline a ConversationObserver watches
// is done, with a ConversationEndedMessage holding the conversation:
//
//	event.On(&observer.Registry, eval.EventConversationEnded,
//		func(ctx context.Context, m eval.ConversationEndedMessage) { … })
const EventConversationEnded = "on_conversation_ended"

// ConversationEndedMessage is the message accompanying EventConversationEnded.
type ConversationEndedMessage struct {
	// Conversation is the conversation's messages, as a conversation file
	// holds them (see Conversation).
	Conversation []map[string]any
}

// ConversationObserver records a bot's conversation, and reports it once the
// pipeline is done.
//
// Each side's turn is what the context aggregators write to the LLM context, so
// the user's turns are what the bot heard and the bot's turns are what it said,
// cut where the user interrupted it. Text a client sends with RTVI's send-text is
// a user turn too, since it goes to the context as is. Each turn is recorded as
// it is written, so summarizing the context later doesn't lose it, and the
// context's other messages, such as a flow's tool changes, stay out.
//
// It records what passes through the worker it is attached to: in a bot with
// several workers, that is the one holding the conversation's context, and
// calls made in other workers are not in it.
//
//	observer := eval.NewConversationObserver()
//	event.On(&observer.Registry, eval.EventConversationEnded,
//		func(_ context.Context, m eval.ConversationEndedMessage) {
//			data, _ := json.MarshalIndent(m.Conversation, "", "  ")
//			_ = os.WriteFile("call.conversation.json", data, 0o600)
//		})
//	worker := pipeline.NewWorker(p, pipeline.WorkerConfig{Observers: []processor.Observer{observer}})
type ConversationObserver struct {
	event.Registry

	conversation *Conversation
}

// NewConversationObserver builds the observer.
func NewConversationObserver() *ConversationObserver {
	o := &ConversationObserver{conversation: NewConversation()}
	o.Register(EventConversationEnded, false)
	return o
}

// ObserveEveryPush reports that the observer sees only a frame's first push.
func (*ConversationObserver) ObserveEveryPush() bool { return false }

// OnPushFrame adds what a frame says about the conversation to it.
func (o *ConversationObserver) OnPushFrame(data processor.FramePushed) {
	f := data.Frame
	// These frames are broadcast, arriving as two frames, each pushed for the
	// first time once. Read the downstream one.
	if _, paired := f.Base().BroadcastSiblingID(); paired && data.Direction != processor.Downstream {
		return
	}
	switch fr := f.(type) {
	case *frames.LLMContextUserTurnFrame:
		// A speculation is added once its turn ends, as written to the
		// context.
		if !fr.Speculation {
			o.conversation.AddUserMessage(fr.Text)
		}
	case *rtvi.SendTextFrame:
		o.conversation.AddUserMessage(fr.Text)
	case *frames.LLMContextAssistantTurnFrame:
		o.conversation.AddAssistantMessage(fr.Text)
	case *frames.FunctionCallInProgressFrame:
		o.conversation.AddToolCall(fr.ToolName, decodedArguments(fr.Args))
	case *frames.ExternalFunctionCallInProgressFrame:
		o.conversation.AddToolCall(fr.ToolName, decodedArguments(fr.Args))
	case *frames.FunctionCallCancelFrame:
		o.conversation.CancelToolCall(fr.ToolName)
	case *frames.ExternalFunctionCallCancelFrame:
		o.conversation.CancelToolCall(fr.ToolName)
	}
}

// Cleanup reports the conversation: the pipeline is done, and so is the bot's
// last turn.
func (o *ConversationObserver) Cleanup(ctx context.Context) error {
	o.conversation.EndBotTurn()
	o.Call(ctx, EventConversationEnded, o, ConversationEndedMessage{Conversation: o.conversation.Messages()})
	o.Registry.Cleanup(ctx)
	return nil
}

// decodedArguments are a call's JSON arguments as a mapping, empty when it has
// none.
func decodedArguments(args json.RawMessage) map[string]any {
	decoded := map[string]any{}
	if len(args) > 0 {
		_ = json.Unmarshal(args, &decoded)
	}
	if decoded == nil {
		decoded = map[string]any{}
	}
	return decoded
}
