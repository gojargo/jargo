package rtvi

import (
	"context"
	"log/slog"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Processor bridges a pipeline to an RTVI client. It completes the handshake —
// replying to client-ready with bot-ready — and reports pipeline events
// (transcriptions, speaking state, errors, LLM text) to the client as RTVI
// messages. Place it in the pipeline upstream of the output transport, which
// carries the messages to the client.
//
// Incoming client messages arrive as InputTransportMessageFrames; outgoing
// messages are pushed downstream as OutputTransportMessageFrames.
type Processor struct {
	*processor.Base
}

// NewProcessor builds an RTVI processor.
func NewProcessor() *Processor {
	p := &Processor{}
	p.Base = processor.New("RTVI", p)
	return p
}

// ProcessFrame handles RTVI client messages and converts pipeline frames into
// RTVI messages, forwarding every frame on.
func (p *Processor) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}

	switch fr := f.(type) {
	case *frames.InputTransportMessageFrame:
		// Client messages are consumed here, not forwarded downstream.
		return p.handleIncoming(ctx, fr)
	case *frames.TranscriptionFrame:
		return p.emitAndForward(ctx, f, dir,
			UserTranscription(fr.Text, fr.UserID, fr.Timestamp, true))
	case *frames.InterimTranscriptionFrame:
		return p.emitAndForward(ctx, f, dir,
			UserTranscription(fr.Text, fr.UserID, fr.Timestamp, false))
	case *frames.UserStartedSpeakingFrame:
		return p.emitAndForward(ctx, f, dir, event(TypeUserStartedSpeaking))
	case *frames.UserStoppedSpeakingFrame:
		return p.emitAndForward(ctx, f, dir, event(TypeUserStoppedSpeaking))
	case *frames.BotStartedSpeakingFrame:
		return p.emitAndForward(ctx, f, dir, event(TypeBotStartedSpeaking))
	case *frames.BotStoppedSpeakingFrame:
		return p.emitAndForward(ctx, f, dir, event(TypeBotStoppedSpeaking))
	case *frames.LLMTextFrame:
		return p.emitAndForward(ctx, f, dir, BotLLMText(fr.Text))
	case *frames.TTSSpeakFrame:
		return p.emitAndForward(ctx, f, dir, BotTTSText(fr.Text))
	case *frames.ErrorFrame:
		return p.emitAndForward(ctx, f, dir, Error(fr.Error, fr.Fatal))
	default:
		return p.PushFrame(ctx, f, dir)
	}
}

// handleIncoming processes a message received from the client.
func (p *Processor) handleIncoming(ctx context.Context, f *frames.InputTransportMessageFrame) error {
	in, err := ParseIncoming(f.Message)
	if err != nil {
		slog.Warn("invalid RTVI message", "err", err)
		return nil
	}
	if in.Label != MessageLabel {
		// Not an RTVI message (e.g. transport signaling); ignore.
		return nil
	}
	switch in.Type {
	case TypeClientReady:
		slog.Debug("RTVI client-ready", "id", in.ID)
		return p.send(ctx, BotReady(in.ID))
	default:
		slog.Debug("unhandled RTVI message", "type", in.Type)
		return nil
	}
}

// emitAndForward sends an RTVI message and forwards the originating frame.
func (p *Processor) emitAndForward(ctx context.Context, f frames.Frame, dir processor.Direction, msg Message) error {
	if err := p.send(ctx, msg); err != nil {
		return err
	}
	return p.PushFrame(ctx, f, dir)
}

// send pushes an RTVI message toward the output transport.
func (p *Processor) send(ctx context.Context, msg Message) error {
	return p.PushFrame(ctx, frames.NewOutputTransportMessageFrame(msg), processor.Downstream)
}
