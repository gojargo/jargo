package rtvi

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"log/slog"
	"sync"

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
	// clientVersion is the protocol version the client declared in its
	// client-ready, read by anything whose wire shape depends on it.
	clientVersion [3]int
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
	p.Events().Register(EventClientMessage, false)
	return p
}

// EventClientMessage fires when the client sends a message of its own, one the
// protocol has no message for. Its argument is the *ClientMessageFrame, which
// also travels downstream; answer it with a ServerResponseFrame or through
// SendServerResponse.
const EventClientMessage = "on_client_message"

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
	if msg, ok := o.userMessageFor(f); ok {
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
	if msgs, ok := o.botSpeakingMessages(f); ok {
		return msgs
	}
	if msgs, ok := o.llmTextMessages(f); ok {
		return msgs
	}
	if msg, ok := o.botMessageFor(f); ok {
		return []Message{msg}
	}
	if msgs, ok := o.outputMessages(f); ok {
		return msgs
	}
	if msg, ok := serverMessageFor(f); ok {
		return []Message{msg}
	}
	return nil
}

// serverMessageFor maps the frames a bot pushes to say something of its own: an
// unprompted message, and the answer to a message the client sent. Unlike
// everything else here they are not derived from what the pipeline did but
// written by the bot outright, which is why they carry their data as it stands.
func serverMessageFor(f frames.Frame) (Message, bool) {
	switch fr := f.(type) {
	case *ServerMessageFrame:
		return ServerMessage(fr.Data), true
	case *ServerResponseFrame:
		if fr.ClientMsg == nil {
			slog.Warn("RTVI server response names no client message, dropping it")
			return Message{}, false
		}
		if fr.Error != "" {
			return ErrorResponse(fr.ClientMsg.MsgID, fr.Error), true
		}
		return ServerResponse(fr.ClientMsg.MsgID, fr.ClientMsg.Type, fr.Data), true
	default:
		return Message{}, false
	}
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

// userMessageFor maps user- and system-originated frames. The second result
// reports whether the frame was one of them at all, so a category the observer
// was told not to report stops the dispatch rather than falling through to
// another mapping.
func (o *Observer) userMessageFor(f frames.Frame) (Message, bool) {
	switch fr := f.(type) {
	case *frames.TranscriptionFrame:
		return o.gated(UserTranscription(fr.Text, fr.UserID, fr.Timestamp, true), transcriptionOf)
	case *frames.InterimTranscriptionFrame:
		return o.gated(UserTranscription(fr.Text, fr.UserID, fr.Timestamp, false), transcriptionOf)
	case *frames.UserStartedSpeakingFrame:
		return o.gated(event(TypeUserStartedSpeaking), userSpeakingOf)
	case *frames.UserStoppedSpeakingFrame:
		return o.gated(event(TypeUserStoppedSpeaking), userSpeakingOf)
	case *frames.UserMuteStartedFrame:
		return o.gated(event(TypeUserMuteStarted), userMuteOf)
	case *frames.UserMuteStoppedFrame:
		return o.gated(event(TypeUserMuteStopped), userMuteOf)
	case *frames.LLMContextFrame:
		// What the user said as the model is about to read it, which is not
		// always what the transcription service heard: the turn may have been
		// assembled from several transcripts, or written by the client outright.
		text, ok := lastUserText(fr.Context)
		if !ok {
			return Message{}, false
		}
		return o.gated(UserLLMText(text), userLLMOf)
	case *frames.ErrorFrame:
		// An error is not a category a client turns off: it is what explains a
		// conversation that stopped working.
		return Error(fr.Error, fr.Fatal), true
	case *frames.MetricsFrame:
		return o.gated(metricsMessage(fr), metricsOf)
	default:
		return Message{}, false
	}
}

// The categories a message can belong to, named so a gate reads as the thing it
// controls rather than as a field lookup.
func transcriptionOf(p ObserverParams) *bool { return p.UserTranscriptionEnabled }
func userSpeakingOf(p ObserverParams) *bool  { return p.UserSpeakingEnabled }
func metricsOf(p ObserverParams) *bool       { return p.MetricsEnabled }
func botSpeakingOf(p ObserverParams) *bool   { return p.BotSpeakingEnabled }
func botLLMOf(p ObserverParams) *bool        { return p.BotLLMEnabled }
func botTTSOf(p ObserverParams) *bool        { return p.BotTTSEnabled }
func botOutputOf(p ObserverParams) *bool     { return p.BotOutputEnabled }
func userLLMOf(p ObserverParams) *bool       { return p.UserLLMEnabled }
func userMuteOf(p ObserverParams) *bool      { return p.UserMuteEnabled }

// gated reports msg unless its category was turned off, and reports either way
// that the frame was recognized: a frame the observer is silent about is still
// not one for another mapping to pick up.
func (o *Observer) gated(msg Message, pick func(ObserverParams) *bool) (Message, bool) {
	if !o.enabled(pick) {
		return Message{}, false
	}
	return msg, true
}

// botMessageFor maps bot-originated frames (the model's output, the
// synthesizer's brackets, tool calls).
func (o *Observer) botMessageFor(f frames.Frame) (Message, bool) {
	switch f.(type) {
	case *frames.InterruptionFrame:
		// The bot's in-flight output was cut off, by a VAD barge-in or by a
		// programmatic interrupt such as send-text with run_immediately. A client
		// drops whatever the bot was mid-saying.
		return o.gated(event(TypeBotInterrupted), botSpeakingOf)
	case *frames.LLMFullResponseStartFrame:
		return o.gated(event(TypeBotLLMStarted), botLLMOf)
	case *frames.LLMFullResponseEndFrame:
		return o.gated(event(TypeBotLLMStopped), botLLMOf)
	case *frames.TTSStartedFrame:
		return o.gated(event(TypeBotTTSStarted), botTTSOf)
	case *frames.TTSStoppedFrame:
		return o.gated(event(TypeBotTTSStopped), botTTSOf)
	default:
		return Message{}, false
	}
}

// lastUserText is the text of the conversation's last message when the user
// wrote it, which is the message this context frame is putting to the model.
// Anything else means the model is not being asked to answer the user just now.
func lastUserText(convo *frames.LLMContext) (string, bool) {
	if convo == nil {
		return "", false
	}
	msgs := convo.Messages()
	if len(msgs) == 0 {
		return "", false
	}
	last := msgs[len(msgs)-1]
	if last.Role != frames.RoleUser || last.Text == "" {
		return "", false
	}
	return last.Text, true
}

// llmTextMessages maps the text the model streams: each token as it arrives,
// and the sentences assembled from them.
//
// The sentences are the superseded bot-transcription messages, which say the
// same thing as the tokens a sentence at a time. They are gathered here rather
// than anywhere else in the pipeline because nothing else sees the model's raw
// output on its way past.
func (o *Observer) llmTextMessages(f frames.Frame) ([]Message, bool) {
	fr, ok := f.(*frames.LLMTextFrame)
	if !ok {
		return nil, false
	}
	if !o.enabled(botLLMOf) {
		return nil, true
	}
	msgs := []Message{BotLLMText(fr.Text)}
	if sentence, done := o.gatherTranscription(fr.Text); done {
		msgs = append(msgs, BotTranscription(sentence))
	}
	return msgs, true
}

// gatherTranscription folds one token into the bot transcription and reports the
// text when it has completed a sentence, starting the next one afresh.
func (o *Observer) gatherTranscription(token string) (string, bool) {
	o.transcriptMu.Lock()
	defer o.transcriptMu.Unlock()
	if o.tokenizer == nil {
		return "", false
	}
	o.botTranscription += token
	if o.botTranscription == "" || o.tokenizer.MatchEndOfSentence(o.botTranscription) == 0 {
		return "", false
	}
	sentence := o.botTranscription
	o.botTranscription = ""
	return sentence, true
}

// segment is one unit of the bot's output on its way to the client, and whether
// it is text the synthesizer has spoken or text it is about to.
type segment struct {
	frame *frames.AggregatedTextFrame
	// spoken reports that this is text the synthesizer has said, rather than a
	// unit announced before synthesis. The two describe the same segment at
	// opposite ends of its playback.
	spoken bool
}

// botSpeakingMessages maps the bot's audio starting and stopping.
//
// Starting also releases the segments held while the bot was silent, so a
// client is told what the bot is saying as it becomes audible rather than
// before. The speaking state is tracked whether or not the speaking events
// themselves are reported: a client that has turned them off still wants the
// output, and holding the segments for an event nobody asked for would keep
// them for good.
func (o *Observer) botSpeakingMessages(f frames.Frame) ([]Message, bool) {
	switch f.(type) {
	case *frames.BotStartedSpeakingFrame:
		held := o.startedSpeaking()
		var msgs []Message
		if msg, ok := o.gated(event(TypeBotStartedSpeaking), botSpeakingOf); ok {
			msgs = append(msgs, msg)
		}
		for _, seg := range held {
			msgs = append(msgs, o.segmentMessages(seg)...)
		}
		return msgs, true
	case *frames.BotStoppedSpeakingFrame:
		o.stoppedSpeaking()
		if msg, ok := o.gated(event(TypeBotStoppedSpeaking), botSpeakingOf); ok {
			return []Message{msg}, true
		}
		return nil, true
	default:
		return nil, false
	}
}

// outputMessages maps the frames carrying what the bot is saying: a segment of
// its output, and how far through one it has spoken.
func (o *Observer) outputMessages(f frames.Frame) ([]Message, bool) {
	switch fr := f.(type) {
	case *frames.TTSTextFrame:
		return o.reportSegment(segment{frame: &fr.AggregatedTextFrame, spoken: true}), true
	case *frames.AggregatedTextFrame:
		return o.reportSegment(segment{frame: fr}), true
	case *frames.AggregatedTextProgressFrame:
		return o.progressMessages(fr), true
	default:
		return nil, false
	}
}

// reportSegment reports a segment now, or holds it until the bot is audible.
func (o *Observer) reportSegment(seg segment) []Message {
	if o.holdUnlessSpeaking(seg) {
		return nil
	}
	return o.segmentMessages(seg)
}

// segmentMessages builds the messages for one segment of the bot's output: what
// it says, and separately the caption for text the synthesizer has spoken.
func (o *Observer) segmentMessages(seg segment) []Message {
	agg := seg.frame
	if o.skipsAggregation(agg.AggregatedBy) {
		return nil
	}

	// A word or a token is not a segment of its own to a current client: the
	// segment is the sentence it belongs to, and the words within it are
	// reported as progress through that. An older client has no progress
	// reports, so it is told about each one as output in its own right.
	suppressed := !o.isLegacyClient() && (agg.AggregatedBy == frames.AggregationWord ||
		agg.AggregatedBy == frames.AggregationToken)

	// Rewritten once, so the segment and the caption for it say the same thing.
	text := o.transform(BotOutputText{Text: agg.Text, AggregatedBy: agg.AggregatedBy}).Text

	var msgs []Message
	if o.enabled(botOutputOf) && !suppressed {
		id, spoken, willBeSpoken := agg.ID(), seg.spoken, agg.WillBeSpoken
		d := BotOutputData{
			Text:         text,
			AggregatedBy: agg.AggregatedBy,
			SegmentID:    &id,
			Spoken:       &spoken,
			WillBeSpoken: &willBeSpoken,
		}
		switch {
		case !willBeSpoken:
			// Nothing is going to speak it, so it has no playback to be
			// anywhere in.
		case seg.spoken:
			// Spoken text arrives once synthesis is done, so the segment is
			// finished by the time the client hears about it.
			d.SpokenStatus = SpokenCompleted
			d.SpokenProgress = &SpokenProgressData{AccumulatedText: text}
		default:
			// The segment is announced before synthesis starts, so none of it
			// has been spoken yet.
			d.SpokenStatus = SpokenNew
			d.SpokenProgress = &SpokenProgressData{RemainingText: text}
		}
		msgs = append(msgs, BotOutput(d))
	}
	// The caption is a channel of its own and is not suppressed with the
	// segment: a client rendering spoken text word by word still wants each one.
	if seg.spoken && o.enabled(botTTSOf) {
		msgs = append(msgs, BotTTSText(text))
	}
	return msgs
}

// progressMessages reports how far through a segment the bot has spoken.
//
// An older client is told nothing: it has no notion of progress within a
// segment, and hears about each word as output of its own instead.
func (o *Observer) progressMessages(fr *frames.AggregatedTextProgressFrame) []Message {
	if o.isLegacyClient() || !o.enabled(botOutputOf) {
		return nil
	}
	// Rewritten as one, so the split stays consistent with the text it splits.
	seg := o.transform(BotOutputText{
		Text:         fr.Text,
		AggregatedBy: fr.AggregatedBy,
		Accumulated:  fr.AccumulatedText,
		Remaining:    fr.RemainingText,
		Progress:     true,
	})

	id, willBeSpoken := fr.SegmentID, true
	status := SpokenInProgress
	if seg.Remaining == "" {
		status = SpokenCompleted
	}
	return []Message{BotOutput(BotOutputData{
		Text:         seg.Text,
		AggregatedBy: fr.AggregatedBy,
		SegmentID:    &id,
		WillBeSpoken: &willBeSpoken,
		SpokenStatus: status,
		SpokenProgress: &SpokenProgressData{
			AccumulatedText: seg.Accumulated,
			RemainingText:   seg.Remaining,
		},
	})}
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
	if !isRTVIMessage(f.Message) {
		// Not an RTVI message (e.g. transport signaling); ignore.
		return nil
	}
	in, err := ParseIncoming(f.Message)
	if err != nil {
		// The label says this was meant for us, so the client is told its
		// message was unreadable rather than left waiting on a reply. The id is
		// part of what could not be read, so this is the general error rather
		// than a response to a particular request.
		slog.Warn("invalid RTVI message", "err", err)
		return p.send(ctx, Error("invalid RTVI transport message: "+err.Error(), false))
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
		return p.handleClientReady(ctx, in)
	case TypeSendText:
		return p.handleSendText(ctx, in)
	case TypeDTMF:
		return p.handleDTMF(ctx, in)
	case TypeDisconnectBot:
		// The client is done. Ending the worker is what stops the pipeline
		// gracefully, letting what is in flight finish rather than cutting it.
		return p.PushFrame(ctx, frames.NewEndWorkerFrame(), processor.Downstream)
	case TypeClientMessage:
		return p.handleClientMessage(ctx, in)
	case TypeFunctionCallResult:
		return p.handleFunctionCallResult(ctx, in)
	case TypeRawAudio, TypeRawAudioBatch:
		return p.handleRawAudio(ctx, in)
	default:
		slog.Debug("unhandled RTVI message", "type", in.Type)
		return p.send(ctx, ErrorResponse(in.ID, "unsupported type "+in.Type))
	}
}

// isRTVIMessage reports whether raw carries the RTVI label, and so was meant
// for this processor. Anything else belongs to the transport or another
// protocol sharing the channel, and is not ours to answer or complain about.
func isRTVIMessage(raw []byte) bool {
	var envelope struct {
		Label string `json:"label"`
	}
	return json.Unmarshal(raw, &envelope) == nil && envelope.Label == MessageLabel
}

// handleClientMessage passes a client's own message into the pipeline and
// announces it.
//
// The protocol has no message for whatever the client is asking, so the bot
// answers it: the frame travels downstream to whatever processor knows the
// answer, which pushes a ServerResponseFrame back naming this request. A
// listener attached to EventClientMessage sees the same request, for an answer
// that comes from outside the pipeline.
func (p *Processor) handleClientMessage(ctx context.Context, in Incoming) error {
	d, err := ParseRawClientMessageData(in.Data)
	if err != nil {
		slog.Warn("invalid RTVI client-message", "err", err)
		return p.send(ctx, ErrorResponse(in.ID, "invalid message: "+err.Error()))
	}
	f := NewClientMessageFrame(in.ID, d.T, d.D)
	if err := p.PushFrame(ctx, f, processor.Downstream); err != nil {
		return err
	}
	p.Events().Call(ctx, EventClientMessage, p, f)
	return nil
}

// handleFunctionCallResult delivers the result of a tool the client ran on the
// bot's behalf.
//
// A tool call the bot cannot run itself is reported to the client, which runs it
// and sends the result back here. The result frame goes downstream to the
// assistant aggregator, which fills in the placeholder the call left in the
// conversation, exactly as a result produced in-process does.
func (p *Processor) handleFunctionCallResult(ctx context.Context, in Incoming) error {
	d, err := ParseFunctionCallResultData(in.Data)
	if err != nil {
		slog.Warn("invalid RTVI llm-function-call-result", "err", err)
		return p.send(ctx, ErrorResponse(in.ID, "invalid message: "+err.Error()))
	}
	f := frames.NewFunctionCallResultFrame(d.ToolCallID, d.FunctionName, d.Arguments, d.ResultText())
	return p.PushFrame(ctx, f, processor.Downstream)
}

// handleRawAudio feeds the pipeline audio the client captured itself.
//
// This processor sits at the top of the pipeline, so the frames go downstream to
// the input transport, which runs them through its own processing: a client
// doing its own capture is heard exactly as one streaming a media track is.
func (p *Processor) handleRawAudio(ctx context.Context, in Incoming) error {
	d, err := ParseRawAudioData(in.Data)
	if err != nil {
		slog.Warn("invalid RTVI raw-audio", "err", err)
		return p.send(ctx, ErrorResponse(in.ID, "invalid message: "+err.Error()))
	}
	for _, chunk := range d.Chunks() {
		pcm, err := base64.StdEncoding.DecodeString(chunk)
		if err != nil {
			slog.Warn("invalid RTVI raw-audio chunk", "err", err)
			return p.send(ctx, ErrorResponse(in.ID, "invalid audio: "+err.Error()))
		}
		f := frames.NewInputAudioRawFrame(pcm, d.SampleRate, d.NumChannels)
		if err := p.PushFrame(ctx, f, processor.Downstream); err != nil {
			return err
		}
	}
	return nil
}

// SendServerMessage sends an unprompted message to the client, for a caller
// holding the processor rather than pushing a ServerMessageFrame.
func (p *Processor) SendServerMessage(ctx context.Context, data any) error {
	return p.send(ctx, ServerMessage(data))
}

// SendServerResponse answers a client message.
func (p *Processor) SendServerResponse(ctx context.Context, msg *ClientMessageFrame, data any) error {
	return p.send(ctx, ServerResponse(msg.MsgID, msg.Type, data))
}

// SendErrorResponse refuses a client message, giving reason.
func (p *Processor) SendErrorResponse(ctx context.Context, msg *ClientMessageFrame, reason string) error {
	return p.send(ctx, ErrorResponse(msg.MsgID, reason))
}

// handleClientReady completes the handshake.
//
// The version the client declares settles what the session speaks. A client of
// this protocol generation is answered with this implementation's version; one
// of the older generation is answered with its own, so it stays on the paths it
// understands rather than being pushed onto ones it has no code for. Any other
// version is told it is incompatible, and the session goes ahead anyway: the
// client is better placed to decide whether to carry on than the bot is to hang
// up on it.
//
// Whatever the version says, the client is ready, so the input transport is
// asked to start streaming audio. A transport configured to hold it back until
// now (see transport.Params.AudioInStreamOnStart) opens its source here; one
// that was already streaming does nothing.
func (p *Processor) handleClientReady(ctx context.Context, in Incoming) error {
	d, err := ParseClientReadyData(in.Data)
	if err != nil {
		slog.Warn("invalid RTVI client-ready data, treating the version as unknown", "err", err)
		d = ClientReadyData{}
	}
	slog.Debug("RTVI client-ready", "id", in.ID, "version", d.Version, "client", d.About.Library)

	version, complaint := p.negotiate(d.Version)
	p.mu.Lock()
	p.clientVersion = version
	p.mu.Unlock()

	if err := p.PushFrame(ctx, frames.NewInputTransportStartAudioStreamingFrame(),
		processor.Downstream); err != nil {
		return err
	}
	if complaint != "" {
		slog.Warn("RTVI client version", "err", complaint)
		if err := p.send(ctx, ErrorResponse(in.ID, complaint)); err != nil {
			return err
		}
	}
	return p.send(ctx, BotReady(in.ID, p.botReadyVersion(version), nil))
}

// negotiate reads the version a client declared, returning it parsed and the
// complaint to send back, which is empty when there is nothing to complain
// about. An unreadable or absent version leaves the parsed version at zero,
// which is no generation at all and so takes the bot-ready's own version.
func (p *Processor) negotiate(declared string) ([3]int, string) {
	const mayNotWork = " Compatibility issues may occur."
	if declared == "" {
		return [3]int{}, "client version unknown." + mayNotWork
	}
	version, ok := parseProtocolVersion(declared)
	if !ok {
		return [3]int{}, "invalid client version format (" + declared + ")." + mayNotWork
	}
	switch version[0] {
	case protocolMajor(), LegacySupportedMajor:
		return version, ""
	default:
		return version, "RTVI version " + declared + " is not compatible with server protocol " +
			ProtocolVersion + "." + mayNotWork
	}
}

// botReadyVersion is the version the bot-ready declares. A client of the older
// generation is told its own version back: this implementation would otherwise
// advertise a generation whose paths the client does not have, and the two would
// disagree about what the wire carries.
func (p *Processor) botReadyVersion(client [3]int) string {
	if client[0] == LegacySupportedMajor {
		return formatProtocolVersion(client)
	}
	return ProtocolVersion
}

// ClientVersion is the protocol version the client declared, as major, minor and
// patch. It is all zeros until a client-ready arrives.
func (p *Processor) ClientVersion() [3]int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.clientVersion
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
// The wait is bounded by the flush itself, which gives up once the pipeline has
// gone quiet without the probe coming back: a processor that swallows it would
// otherwise stop the client being able to say anything at all. On giving up the
// turn goes ahead without the guarantee, which is what the client asked for,
// rather than nothing happening. A pipeline that is still working keeps the wait
// alive however long it takes.
func (p *Processor) interruptAndSettle(ctx context.Context) error {
	if err := p.BroadcastInterruption(ctx); err != nil {
		return err
	}
	if err := p.FlushPipeline(ctx); err != nil {
		slog.Warn("RTVI pipeline flush did not settle", "err", err)
	}
	return nil
}

// send pushes an RTVI message toward the output transport.
func (p *Processor) send(ctx context.Context, msg Message) error {
	return p.PushFrame(ctx, frames.NewOutputTransportMessageUrgentFrame(msg), processor.Downstream)
}
