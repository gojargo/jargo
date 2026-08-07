package rtvi

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Processor bridges a pipeline to an RTVI client. It completes the handshake,
// replying to client-ready with bot-ready, and carries out what the client asks
// of the pipeline. Place it upstream of the output transport, which carries its
// messages to the client.
//
// It does not report pipeline events: pair it with an Observer, which watches
// the whole pipeline and sends through this processor. Incoming client messages
// arrive as InputTransportMessageFrames; outgoing messages are pushed downstream
// as OutputTransportMessageUrgentFrames.
type Processor struct {
	*processor.Base

	// mu guards baseCtx, which an Observer uses to send from its own goroutine.
	mu      sync.Mutex
	baseCtx context.Context //nolint:containedctx // outlives the frame that set it

	// messages carries client messages from the frame path to the goroutine that
	// carries them out. Acting on one can mean waiting for the pipeline to
	// settle, and the probe that reports settling has to travel through this
	// processor, so the wait cannot happen on the frame path.
	messages chan Incoming
	// messagesWG tracks that goroutine, so cleanup does not race it.
	messagesWG sync.WaitGroup
}

// messageBuffer is how many client messages may be queued for the handler
// goroutine. A client sends control messages, not a stream, so a handful in
// flight is generous.
const messageBuffer = 32

// NewProcessor builds an RTVI processor.
func NewProcessor() *Processor {
	p := &Processor{}
	p.Base = processor.New("RTVI", p)
	return p
}

// Setup records the context an out-of-band send runs under, and starts the
// goroutine that carries out client messages.
func (p *Processor) Setup(ctx context.Context, s processor.Setup) error {
	p.mu.Lock()
	p.baseCtx = ctx
	p.mu.Unlock()
	if err := p.Base.Setup(ctx, s); err != nil {
		return err
	}
	p.messages = make(chan Incoming, messageBuffer)
	p.messagesWG.Add(1)
	go p.messageLoop(ctx)
	return nil
}

// Cleanup stops the message goroutine and waits for it.
func (p *Processor) Cleanup(ctx context.Context) error {
	err := p.Base.Cleanup(ctx)
	if p.messages != nil {
		close(p.messages)
		p.messagesWG.Wait()
		p.messages = nil
	}
	return err
}

// messageLoop carries out client messages, one at a time and in order, off the
// frame path.
func (p *Processor) messageLoop(ctx context.Context) {
	defer p.messagesWG.Done()
	for {
		select {
		case in, ok := <-p.messages:
			if !ok {
				return
			}
			if err := p.handleMessage(ctx, in); err != nil {
				slog.Warn("RTVI message failed", "type", in.Type, "err", err)
			}
		case <-ctx.Done():
			return
		}
	}
}

// ProcessFrame handles messages arriving from the client and forwards every
// frame on. Events going the other way are reported by an Observer.
func (p *Processor) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}

	// Client messages are consumed here, not forwarded downstream.
	if fr, ok := f.(*frames.InputTransportMessageFrame); ok {
		return p.handleIncoming(ctx, fr)
	}
	return p.PushFrame(ctx, f, dir)
}

// messageFor maps a pipeline frame to the RTVI server message it should emit,
// reporting false for frames that produce no message. The mapping is split into
// user- and bot-originated frames to keep each dispatch small; the tool-call
// frames are separate again because how much of them is reported depends on the
// observer's per-function report level.
func (o *Observer) messageFor(f frames.Frame) (Message, bool) {
	if msg, ok := userMessageFor(f); ok {
		return msg, true
	}
	if msg, ok := o.functionCallMessageFor(f); ok {
		return msg, true
	}
	return botMessageFor(f)
}

// functionCallMessageFor maps a tool-call frame to its message, reporting only
// what the function's report level allows. A disabled function produces no
// message at all.
func (o *Observer) functionCallMessageFor(f frames.Frame) (Message, bool) {
	switch fr := f.(type) {
	case *frames.FunctionCallInProgressFrame:
		level := o.reportLevelFor(fr.ToolName)
		if level == ReportDisabled {
			return Message{}, false
		}
		return LLMFunctionCall(fr.ToolName, fr.ToolCallID, fr.Args, level), true
	case *frames.FunctionCallResultFrame:
		level := o.reportLevelFor(fr.ToolName)
		if level == ReportDisabled {
			return Message{}, false
		}
		return LLMFunctionCallResult(fr.ToolName, fr.ToolCallID, fr.Result, level), true
	default:
		return Message{}, false
	}
}

// userMessageFor maps user- and system-originated frames.
func userMessageFor(f frames.Frame) (Message, bool) {
	switch fr := f.(type) {
	case *frames.TranscriptionFrame:
		return UserTranscription(fr.Text, fr.UserID, fr.Timestamp, true), true
	case *frames.InterimTranscriptionFrame:
		return UserTranscription(fr.Text, fr.UserID, fr.Timestamp, false), true
	case *frames.UserStartedSpeakingFrame:
		return event(TypeUserStartedSpeaking), true
	case *frames.UserStoppedSpeakingFrame:
		return event(TypeUserStoppedSpeaking), true
	case *frames.ErrorFrame:
		return Error(fr.Error, fr.Fatal), true
	case *frames.MetricsFrame:
		return metricsMessage(fr), true
	default:
		return Message{}, false
	}
}

// botMessageFor maps bot-originated frames (speaking, LLM, TTS, tool calls).
func botMessageFor(f frames.Frame) (Message, bool) {
	switch fr := f.(type) {
	case *frames.BotStartedSpeakingFrame:
		return event(TypeBotStartedSpeaking), true
	case *frames.BotStoppedSpeakingFrame:
		return event(TypeBotStoppedSpeaking), true
	case *frames.InterruptionFrame:
		// The bot's in-flight output was cut off, by a VAD barge-in or by a
		// programmatic interrupt such as send-text with run_immediately. A client
		// drops whatever the bot was mid-saying.
		return event(TypeBotInterrupted), true
	case *frames.LLMFullResponseStartFrame:
		return event(TypeBotLLMStarted), true
	case *frames.LLMFullResponseEndFrame:
		return event(TypeBotLLMStopped), true
	case *frames.LLMTextFrame:
		return BotLLMText(fr.Text), true
	case *frames.TTSStartedFrame:
		return event(TypeBotTTSStarted), true
	case *frames.TTSStoppedFrame:
		return event(TypeBotTTSStopped), true
	case *frames.TTSSpeakFrame:
		return BotTTSText(fr.Text), true
	default:
		return Message{}, false
	}
}

// metricsMessage converts a MetricsFrame into an RTVI metrics message, grouping
// its measurements by kind. A frame can carry several kinds, and measurements
// from more than one processor, so each kind is a list.
func metricsMessage(f *frames.MetricsFrame) Message {
	var d MetricsData
	for _, m := range f.Data {
		p, model := m.MetricsProcessor(), m.MetricsModel()
		switch v := m.(type) {
		case frames.TTFBMetricsData:
			d.TTFB = append(d.TTFB, MetricData{Processor: p, Value: v.Value.Seconds(), Model: model})
		case frames.TTFAMetricsData:
			d.TTFA = append(d.TTFA, TTFAMetricData{
				Processor:      p,
				Model:          model,
				TTFA:           v.TTFA.Seconds(),
				TTFB:           v.TTFB.Seconds(),
				LeadingSilence: v.LeadingSilence.Seconds(),
			})
		case frames.ProcessingMetricsData:
			d.Processing = append(d.Processing, MetricData{Processor: p, Value: v.Value.Seconds(), Model: model})
		case frames.TTSUsageMetricsData:
			d.Characters = append(d.Characters, MetricData{Processor: p, Value: float64(v.Value), Model: model})
		case frames.STTUsageMetricsData:
			d.STTUsage = append(d.STTUsage, MetricData{Processor: p, Value: v.Value.AudioSeconds, Model: model})
		case frames.TextAggregationMetricsData:
			d.TextAggregation = append(d.TextAggregation,
				MetricData{Processor: p, Value: v.Value.Seconds(), Model: model})
		case frames.TurnMetricsData:
			d.Turn = append(d.Turn, TurnMetricData{
				Processor:    p,
				Complete:     v.Complete,
				Probability:  v.Probability,
				ProcessingMs: float64(v.E2EProcessing.Microseconds()) / 1000,
			})
		case frames.LLMUsageMetricsData:
			d.Tokens = append(d.Tokens, TokenMetricData{
				Processor:        p,
				Model:            model,
				PromptTokens:     v.Value.PromptTokens,
				CompletionTokens: v.Value.CompletionTokens,
				TotalTokens:      v.Value.TotalTokens,
			})
		}
	}
	return Metrics(d)
}

// handleIncoming parses a message received from the client and hands it to the
// goroutine that carries it out. Parsing is cheap and stays on the frame path;
// acting on the message does not, because it may have to wait for the pipeline
// to settle.
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
	select {
	case p.messages <- in:
	case <-ctx.Done():
	}
	return nil
}

// handleMessage carries out one client message. It runs on the message
// goroutine, never the frame path.
func (p *Processor) handleMessage(ctx context.Context, in Incoming) error {
	switch in.Type {
	case TypeClientReady:
		slog.Debug("RTVI client-ready", "id", in.ID)
		return p.send(ctx, BotReady(in.ID))
	case TypeSendText:
		return p.handleSendText(ctx, in)
	default:
		slog.Debug("unhandled RTVI message", "type", in.Type)
		return nil
	}
}

// handleSendText injects a text user turn. The processor sits downstream of the
// context aggregator, so the injected frames are pushed upstream to reach it:
// the append adds the user message to the shared context, and, unless the client
// opted out, the run makes the LLM respond immediately, bypassing the
// VAD/turn-taking gating that governs spoken turns.
//
// Running immediately also cuts the bot off mid-answer, which is what makes it a
// barge-in the client typed rather than spoke.
func (p *Processor) handleSendText(ctx context.Context, in Incoming) error {
	d, err := ParseSendTextData(in.Data)
	if err != nil {
		slog.Warn("invalid RTVI send-text", "err", err)
		return nil
	}
	if d.Content == "" {
		return nil
	}
	if d.RunImmediately() {
		if err := p.interruptAndSettle(ctx); err != nil {
			return err
		}
	}
	appendMsg := frames.NewLLMMessagesAppendFrame([]frames.Message{{Role: frames.RoleUser, Text: d.Content}})
	if err := p.PushFrame(ctx, appendMsg, processor.Upstream); err != nil {
		return err
	}
	if d.RunImmediately() {
		return p.PushFrame(ctx, frames.NewLLMRunFrame(), processor.Upstream)
	}
	return nil
}

// interruptAndSettle cuts off whatever the bot is saying, then waits for the
// pipeline to drain.
//
// The wait is the point. An interruption commits the in-progress assistant
// response into the context, and draining guarantees that lands before the new
// user message is appended and run. Without it the append can overtake the
// commit, and the model, seeing the new message ahead of what it was already
// saying, carries on with the turn it was interrupted out of.
//
// The wait is bounded: a processor that swallows the probe would otherwise stop
// the client being able to say anything at all. On timeout the turn goes ahead
// without the guarantee, which is what the client asked for, rather than
// nothing happening.
func (p *Processor) interruptAndSettle(ctx context.Context) error {
	err := p.Broadcast(ctx, func() frames.Frame { return frames.NewInterruptionFrame() })
	if err != nil {
		return err
	}
	flushCtx, cancel := context.WithTimeout(ctx, flushTimeout)
	defer cancel()
	if err := p.FlushPipeline(flushCtx); err != nil {
		slog.Warn("RTVI pipeline flush did not settle", "within", flushTimeout, "err", err)
	}
	return nil
}

// flushTimeout bounds the wait for the pipeline to settle after an interruption.
const flushTimeout = 5 * time.Second

// send pushes an RTVI message toward the output transport.
func (p *Processor) send(ctx context.Context, msg Message) error {
	return p.PushFrame(ctx, frames.NewOutputTransportMessageUrgentFrame(msg), processor.Downstream)
}
