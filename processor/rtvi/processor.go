package rtvi

import (
	"context"
	"encoding/json"
	"log/slog"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Processor bridges a pipeline to an RTVI client. It completes the handshake,
// replying to client-ready with bot-ready, and carries out what the client asks
// of the pipeline.
//
// Place it at the top of the pipeline, ahead of the input transport. What the
// client injects is pushed downstream from there, so it travels the pipeline by
// the same path a real caller's input takes: a typed message reaches the
// context aggregator, a keypress reaches the DTMF handling, and each arrives in
// order with everything the turn is made of. Its own messages to the client are
// pushed downstream too, and reach the output transport at the far end.
//
// It does not report pipeline events: pair it with an Observer, which watches
// the whole pipeline and sends through this processor. Incoming client messages
// arrive as InputTransportMessageFrames, which the input transport broadcasts so
// they reach this processor from either side; outgoing messages are pushed
// downstream as OutputTransportMessageUrgentFrames.
type Processor struct {
	*processor.Base

	// mu guards baseCtx, which an Observer uses to send from its own goroutine,
	// and llmSkipTTS, which the message goroutine reads while the frame path
	// updates it.
	mu      sync.Mutex
	baseCtx context.Context //nolint:containedctx // outlives the frame that set it
	// llmSkipTTS is the output configuration the LLM service is running under,
	// so a turn that changes it for itself can put back what it found. It starts
	// false, which is what an LLM service nothing has configured does.
	llmSkipTTS bool

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
	// Whatever configures the LLM's output is followed, so a turn asking for a
	// different setting knows what to put back after it.
	if fr, ok := f.(*frames.LLMConfigureOutputFrame); ok {
		p.mu.Lock()
		p.llmSkipTTS = fr.SkipTTS
		p.mu.Unlock()
	}
	return p.PushFrame(ctx, f, dir)
}

// messageFor maps a pipeline frame to the RTVI server message it should emit,
// reporting false for frames that produce no message. The mapping is split into
// user- and bot-originated frames to keep each dispatch small; the tool-call
// frames are separate again because how much of them is reported depends on the
// observer's per-function report level.
// A frame usually maps to one message, but a tool-call batch reports each call
// in it separately, so the result is a list.
func (o *Observer) messagesFor(f frames.Frame) []Message {
	if msg, ok := userMessageFor(f); ok {
		return []Message{msg}
	}
	if msgs, ok := o.functionCallMessagesFor(f); ok {
		return msgs
	}
	if msg, ok := o.vadMessageFor(f); ok {
		return []Message{msg}
	}
	if msg, send, isAudio := o.audioLevelMessageFor(f); isAudio {
		if send {
			return []Message{msg}
		}
		return nil
	}
	if msg, ok := botMessageFor(f); ok {
		return []Message{msg}
	}
	return nil
}

// vadMessageFor maps the raw VAD speaking frames, which are reported only when
// the observer is configured to expose them. They reflect the VAD signal
// directly, where the user-started/stopped-speaking events reflect the turn a
// strategy may gate or defer.
func (o *Observer) vadMessageFor(f frames.Frame) (Message, bool) {
	switch f.(type) {
	case *frames.VADUserStartedSpeakingFrame:
		return event(TypeVADUserStarted), o.vadUserSpeakingEnabled()
	case *frames.VADUserStoppedSpeakingFrame:
		return event(TypeVADUserStopped), o.vadUserSpeakingEnabled()
	default:
		return Message{}, false
	}
}

// functionCallMessagesFor maps a tool-call frame to its messages, reporting only
// what each function's report level allows. A disabled function produces no
// message at all, and the second result reports whether the frame was a
// tool-call frame in the first place.
func (o *Observer) functionCallMessagesFor(f frames.Frame) ([]Message, bool) {
	switch fr := f.(type) {
	case *frames.FunctionCallsStartedFrame:
		// The model asked for these calls; none has begun executing yet. Each is
		// reported on its own, at its own level.
		var msgs []Message
		for _, call := range fr.Calls {
			if level := o.reportLevelFor(call.Name); level != ReportDisabled {
				msgs = append(msgs, LLMFunctionCallStart(call.Name, level))
			}
		}
		return msgs, true
	case *frames.FunctionCallInProgressFrame:
		return o.callMessage(fr.ToolName, func(level FunctionCallReportLevel) Message {
			return LLMFunctionCall(fr.ToolName, fr.ToolCallID, fr.Args, level)
		})
	case *frames.FunctionCallResultFrame:
		return o.callMessage(fr.ToolName, func(level FunctionCallReportLevel) Message {
			return LLMFunctionCallStopped(fr.ToolName, fr.ToolCallID, fr.Result, false, level)
		})
	case *frames.FunctionCallCancelFrame:
		return o.callMessage(fr.ToolName, func(level FunctionCallReportLevel) Message {
			return LLMFunctionCallStopped(fr.ToolName, fr.ToolCallID, "", true, level)
		})
	default:
		return nil, false
	}
}

// callMessage builds one tool-call message at the function's report level, or
// none when the function is disabled.
func (o *Observer) callMessage(name string, build func(FunctionCallReportLevel) Message) ([]Message, bool) {
	level := o.reportLevelFor(name)
	if level == ReportDisabled {
		return nil, true
	}
	return []Message{build(level)}, true
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
	case *frames.TTSTextFrame:
		// The text the TTS reports speaking, aligned to playback, which is what a
		// client renders as the spoken caption. Not TTSSpeakFrame: that is text on
		// its way into the service, and it never reaches a client as something
		// spoken, because nothing guarantees the synthesizer accepted it.
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
		// The client is ready, so ask the input transport to start streaming
		// audio. A transport configured to hold it back until now (see
		// transport.Params.AudioInStreamOnStart) opens its source here; one that
		// was already streaming does nothing.
		if err := p.PushFrame(ctx, frames.NewInputTransportStartAudioStreamingFrame(),
			processor.Downstream); err != nil {
			return err
		}
		return p.send(ctx, BotReady(in.ID))
	case TypeSendText:
		return p.handleSendText(ctx, in)
	case TypeDTMF:
		return p.handleDTMF(ctx, in)
	default:
		slog.Debug("unhandled RTVI message", "type", in.Type)
		return nil
	}
}

// handleSendText injects a text user turn. The processor sits at the top of the
// pipeline, so the injected frames are pushed downstream to reach the context
// aggregator: the append adds the user message to the shared context, and,
// unless the client opted out, the run makes the LLM respond immediately,
// bypassing the VAD/turn-taking gating that governs spoken turns.
//
// Running immediately also cuts the bot off mid-answer, which is what makes it a
// barge-in the client typed rather than spoke.
//
// A turn the client asked not to be spoken is bracketed by the output
// configuration it wants and the one that was in effect before it, so the
// setting applies to this turn alone. The three travel one behind the other in
// the same direction, which is what keeps the restore from overtaking the turn
// it closes.
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

	p.mu.Lock()
	current := p.llmSkipTTS
	p.mu.Unlock()
	skip := !d.AudioResponse()
	toggle := current != skip
	if toggle {
		if err := p.PushFrame(ctx, frames.NewLLMConfigureOutputFrame(skip), processor.Downstream); err != nil {
			return err
		}
	}

	appendMsg := frames.NewLLMMessagesAppendFrame([]frames.Message{{Role: frames.RoleUser, Text: d.Content}})
	if err := p.PushFrame(ctx, appendMsg, processor.Downstream); err != nil {
		return err
	}
	if d.RunImmediately() {
		if err := p.PushFrame(ctx, frames.NewLLMRunFrame(), processor.Downstream); err != nil {
			return err
		}
	}
	if toggle {
		return p.PushFrame(ctx, frames.NewLLMConfigureOutputFrame(current), processor.Downstream)
	}
	return nil
}

// handleDTMF injects the keys the client pressed, one frame each. A
// DTMFAggregator in the pipeline accumulates them and flushes, on its
// terminator key or its idle timeout, into a transcription the bot reacts to:
// the same path a telephony caller's keypress takes.
//
// The keys go downstream, like the user message a send-text appends, because
// this processor sits at the top of the pipeline. A keypress is user input
// arriving from the client, so it travels the way the rest of it does, reaching
// the input transport and the DTMF handling behind it.
func (p *Processor) handleDTMF(ctx context.Context, in Incoming) error {
	var d DTMFData
	if err := json.Unmarshal(in.Data, &d); err != nil {
		slog.Warn("invalid RTVI dtmf", "err", err)
		return nil
	}
	for _, button := range d.Buttons {
		key := frames.KeypadEntry(button)
		if !key.Valid() {
			slog.Warn("ignoring invalid DTMF key", "key", button)
			continue
		}
		if err := p.PushFrame(ctx, frames.NewInputDTMFFrame(key), processor.Downstream); err != nil {
			return err
		}
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
	if err := p.BroadcastInterruption(ctx); err != nil {
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
