package eval

import (
	"cmp"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"mime"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/pytext"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/pipeline/capabilities"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/transport/wsclient"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
)

// The harness's connection to the bot under test.
//
// Client runs the eval pipeline, a pipeline acting as an RTVI client: it waits
// for the bot to listen, connects, completes the handshake, sends the user's
// turns (text, synthesized speech, a recording, DTMF keys, an image), records
// the conversation, and tears down. The bot's output reaches the rest of the
// harness as events on the stream the client is given.

// connectRetry is how often the harness tries the bot again while it is not yet
// accepting connections.
const connectRetry = 250 * time.Millisecond

// stopTimeout is how long the pipeline has to wind down once ended before it is
// canceled.
const stopTimeout = 5 * time.Second

// botVADStopSecs is how long the bot has to stay silent for the harness's VAD
// to end a segment of its speech.
const botVADStopSecs = 0.4

// errNotStarted is a user turn sent before the pipeline was built.
var errNotStarted = errors.New("eval: the client's pipeline is not started")

// errNoUserTTS is a turn with pauses and no voice to speak it.
var errNoUserTTS = errors.New("eval: speaking a turn with pauses needs a user voice")

// closeDescription is how the bot closed the connection, with its close code
// and reason when it sent them.
func closeDescription(err error) string {
	var ce websocket.CloseError
	if !errors.As(err, &ce) {
		return "the bot closed the connection before bot-ready"
	}
	reason := ""
	if ce.Reason != "" {
		reason = " " + ce.Reason
	}
	return fmt.Sprintf("the bot closed the connection before bot-ready (%d%s)", int(ce.Code), reason)
}

// segmentStartedAtKey is the metadata a transcription of one segment of the
// bot's speech carries: when that segment's audio began, as a time.Time.
const segmentStartedAtKey = "eval.segment_started_at"

// segmentStartedAt is when the segment a transcription transcribes began, and
// false for a transcription that does not say.
func segmentStartedAt(f *frames.TranscriptionFrame) (time.Time, bool) {
	at, ok := f.Metadata()[segmentStartedAtKey].(time.Time)
	return at, ok
}

// botSegmentTranscribedFrame closes a segment's transcription run, whatever it
// yielded.
//
// Sent after the run's frames, it reaches the sink once the aggregator has taken
// them in, so the sink reports the segment as accounted for only then: a
// transcript that opens a turn of its own has opened it by the time this frame
// passes. Uninterruptible because a transcript that opens a turn also raises an
// interruption, which flushes the queued frames behind it.
type botSegmentTranscribedFrame struct {
	frames.BaseDataFrame
	frames.UninterruptibleMixin
}

func newBotSegmentTranscribedFrame() *botSegmentTranscribedFrame {
	return &botSegmentTranscribedFrame{BaseDataFrame: frames.NewBaseDataFrame("BotSegmentTranscribedFrame")}
}

// segmentStarts is the gate's record of when each ended segment of the bot's
// speech began, oldest first, for the STT's tagging.
type segmentStarts struct {
	mu     sync.Mutex
	starts []time.Time
}

func (q *segmentStarts) push(at time.Time) {
	q.mu.Lock()
	q.starts = append(q.starts, at)
	q.mu.Unlock()
}

// pop takes the oldest start, and false when none is recorded.
func (q *segmentStarts) pop() (time.Time, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.starts) == 0 {
		return time.Time{}, false
	}
	at := q.starts[0]
	q.starts = q.starts[1:]
	return at, true
}

func (q *segmentStarts) len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.starts)
}

// tagBotSegments has the STT's transcriptions carry when their segment's audio
// began.
//
// A segmented STT transcribes each speech segment on its own, in order, once the
// segment ends, and its transcript can land seconds after the audio it came
// from. The gate ahead of it records when each segment began, one entry per
// segment as it ends, and this pairs each transcription run with the matching
// entry, so a transcript is attributed by its audio's time and not by its
// arrival. A run that yields no transcription still takes its entry, which keeps
// the pairing exact, and every run ends with a botSegmentTranscribedFrame, so
// the sink can tell when nothing is left to transcribe.
//
// A tagged transcription stays a *frames.TranscriptionFrame, every field it was
// built with kept, so the turn aggregator takes it in as one; the start rides in
// its metadata (see segmentStartedAt).
func tagBotSegments(service *stt.SegmentService, starts *segmentStarts) {
	run := service.RunSTT()
	service.SetRunSTT(func(ctx context.Context, audio []byte, sampleRate int) iter.Seq[frames.Frame] {
		return func(yield func(frames.Frame) bool) {
			startedAt, ok := starts.pop()
			if !ok {
				startedAt = time.Now()
			}
			slog.Debug("bot speech: transcribing a segment", "starts_left", starts.len())
			for f := range run(ctx, audio, sampleRate) {
				if tf, ok := f.(*frames.TranscriptionFrame); ok {
					tf.Metadata()[segmentStartedAtKey] = startedAt
				}
				if !yield(f) {
					return
				}
			}
			yield(newBotSegmentTranscribedFrame())
		}
	})
}

// botSpeechGate sits between the bot-speech STT and the turn aggregator: after
// an interrupting send, only the bot's reply gets through.
//
// The aggregator's VAD reports the bot's speech segments upstream, past this
// gate, to the STT; the gate notes when each began and, as each ends, hands the
// start to the STT's tagging (see tagBotSegments) and tells the stream a segment
// awaits transcription. An ordinary send waits for the bot's speech to be
// transcribed first, so nothing of it is in flight. A send that talks over the
// bot cannot wait, and the bot goes on speaking until it notices the user: a
// tagged transcript whose segment began before the bot started speaking again is
// the bot finishing what it was saying, however late it lands, and is dropped
// here so it cannot pass for the reply or be folded into it. A turn that sends
// nothing drops the same way what the bot says before it starts speaking again.
//
// Both need a segmented STT, which transcribes one segment at a time, so the
// client accepts no other.
type botSpeechGate struct {
	*processor.Base
	stream *EventStream
	trace  *Trace
	starts *segmentStarts

	// segmentStartedAt is when the segment being heard began; zero between
	// segments. Only the frame goroutine touches it.
	segmentStartedAt time.Time
}

func newBotSpeechGate(stream *EventStream, trace *Trace, starts *segmentStarts) *botSpeechGate {
	g := &botSpeechGate{stream: stream, trace: trace, starts: starts}
	g.Base = processor.New("EvalBotSpeechGate", g)
	return g
}

// ProcessFrame records the bot's segments and drops what it said before its
// reply.
func (g *botSpeechGate) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := g.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.VADUserStartedSpeakingFrame:
		slog.Debug("bot speech: segment began", "direction", dir)
		g.segmentStartedAt = time.Now()
		g.stream.BotSegmentStarted()
	case *frames.VADUserStoppedSpeakingFrame:
		slog.Debug("bot speech: segment ended", "direction", dir)
		started := g.segmentStartedAt
		if started.IsZero() {
			started = time.Now()
		}
		g.starts.push(started)
		g.segmentStartedAt = time.Time{}
		g.stream.BotSegmentEnded()
	case *frames.TranscriptionFrame:
		startedAt, tagged := segmentStartedAt(fr)
		if until, ok := g.stream.TalkedOverUntil(); tagged && ok && startedAt.Before(until) {
			g.trace.Log(fmt.Sprintf("discard: bot speech from before the turn's reply %q", fr.Text))
			return nil
		}
	}
	return g.PushFrame(ctx, f, dir)
}

// frameInjector pushes a user turn into the pipeline, past the bot's side: the
// sink.
type frameInjector interface {
	inject(ctx context.Context, f frames.Frame) error
}

// botTurnResetter is the aggregator of the bot's turns, as a send that talks
// over the bot drops what it holds.
type botTurnResetter interface {
	Reset()
}

// botFrameSink is where the bot's output stops and the user's turns start.
//
// The bot's RTVI messages become events on the stream and go no further. The
// user's turns are injected here, past the bot's side of the pipeline, so an
// interruption there cannot drop them.
//
// Two rules apply. The aggregator's interruption, raised when the bot speaks,
// stops here for a scripted turn (it would flush the user audio being sent) and
// passes for a persona (a caller who hears the bot keep talking stops). And the
// persona answers from here: in text mode it hears each finished bot response
// through the sink, in audio mode the aggregator hands it the bot's turn; either
// way nothing reaches it once it hung up.
//
// Once the persona has hung up, its LLM no longer marks the bot's turns
// complete. The sink marks the closing turn complete in its place once the bot
// reports it stopped speaking, so a pause between two of the bot's sentences
// does not end it.
type botFrameSink struct {
	*processor.Base
	stream  *EventStream
	persona *Persona
	// personaHears is whether the persona hears the bot's responses through
	// the sink (text mode, with no aggregator to hand it the turn).
	personaHears bool

	mu sync.Mutex
	// closingInference is the closing turn's latest inference, marked complete
	// once the bot stopped speaking.
	closingInference *frames.LLMContextFrame
}

// newBotFrameSink builds the sink. persona is the simulated caller, or nil;
// personaHears is whether it hears the bot's responses through the sink.
func newBotFrameSink(stream *EventStream, persona *Persona, personaHears bool) *botFrameSink {
	s := &botFrameSink{stream: stream, persona: persona, personaHears: personaHears && persona != nil}
	s.Base = processor.New("EvalBotFrameSink", s)
	return s
}

// inject pushes a user turn into the pipeline, past the bot's side.
func (s *botFrameSink) inject(ctx context.Context, f frames.Frame) error {
	return s.PushFrame(ctx, f, processor.Downstream)
}

// ProcessFrame turns the bot's RTVI messages into events.
func (s *botFrameSink) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	// The bot's side arrives downstream, from the input transport. What
	// travels upstream is the user side's own (a persona's function call, a
	// TTS's lifecycle), not bot output, and passes through as is.
	if dir != processor.Downstream {
		return s.PushFrame(ctx, f, dir)
	}
	if message, ok := f.(*frames.InputTransportMessageFrame); ok {
		// The bot's RTVI messages become events and stop here. Everything
		// downstream of the sink speaks for the user (a persona LLM, the user
		// TTS, the output) and must never take them as its own input.
		return s.botMessage(ctx, message)
	}
	// Not one of the bot's RTVI messages, but activity on its side all the same.
	s.stream.Touch()
	if _, ok := f.(*botSegmentTranscribedFrame); ok {
		s.stream.BotSegmentTranscribed()
		return nil
	}
	switch fr := f.(type) {
	case *frames.InterruptionFrame:
		if s.persona == nil {
			return nil
		}
	case *frames.LLMContextFrame:
		// The aggregator handing the persona the bot's turn (audio mode).
		if s.persona != nil && s.persona.HungUp() {
			s.mu.Lock()
			s.closingInference = fr
			s.mu.Unlock()
			return s.maybeCompleteClosingTurn(ctx)
		}
		return s.answer(ctx, fr)
	}
	return s.PushFrame(ctx, f, dir)
}

// botMessage appends the event one of the bot's RTVI messages maps to, if any,
// and has a persona that hears through the sink hear it.
func (s *botFrameSink) botMessage(ctx context.Context, message *frames.InputTransportMessageFrame) error {
	event := s.stream.RTVIMessageToEvent(message.Message)
	if event == nil {
		return nil
	}
	s.stream.Append(event)
	if s.personaHears {
		if err := s.answer(ctx, s.persona.Hear(event)); err != nil {
			return err
		}
	}
	return s.maybeCompleteClosingTurn(ctx)
}

// maybeCompleteClosingTurn marks the closing turn complete for the persona once
// the bot stopped speaking.
func (s *botFrameSink) maybeCompleteClosingTurn(ctx context.Context) error {
	s.mu.Lock()
	inference := s.closingInference
	if inference == nil || s.stream.BotSpeaking() {
		s.mu.Unlock()
		return nil
	}
	s.closingInference = nil
	s.mu.Unlock()
	return s.PushFrame(ctx, frames.NewUserTurnCompletionFrame(MarkerComplete, "", inference.ID()), processor.Upstream)
}

// answer pushes the frame that has the persona answer, unless there is none or
// it hung up.
func (s *botFrameSink) answer(ctx context.Context, f *frames.LLMContextFrame) error {
	if f == nil || s.persona == nil || s.persona.HungUp() {
		return nil
	}
	return s.PushFrame(ctx, f, processor.Downstream)
}

// EventPersonaTurn is the event the relay appends for each response the persona
// completes.
const EventPersonaTurn = "persona_turn"

// personaTurnRelay sends the persona's replies to the bot and reports each one
// as a turn.
//
// In text mode a whole reply goes to the bot as one send-text; in audio mode the
// user TTS has already spoken it and this only logs the text. A reply with words
// in it is one persona_turn event, up to where the bot's speech cut it off if it
// did; a reply that only calls end_call, or that the bot's speech cut off before
// a word went out, is not a turn.
//
// The persona's end_call event follows the turn of the reply that made the
// call: in audio mode the reply is still being spoken when the persona calls
// it.
type personaTurnRelay struct {
	*processor.Base
	textTurnMessage func(text string) rtvi.Message
	trace           *Trace
	stream          *EventStream

	mu   sync.Mutex
	text []string
	// spoken is what the persona said in the current response: the one text
	// turn, or the sentences the user TTS spoke.
	spoken []string
	// replying is whether a response is on its way to the bot, and endCall the
	// end_call event held until it has gone out.
	replying bool
	endCall  map[string]any
}

// newPersonaTurnRelay builds the relay: textTurnMessage builds the RTVI
// send-text message for a turn, and stream is where the persona's turns are
// counted.
func newPersonaTurnRelay(
	textTurnMessage func(string) rtvi.Message, trace *Trace, stream *EventStream,
) *personaTurnRelay {
	r := &personaTurnRelay{textTurnMessage: textTurnMessage, trace: trace, stream: stream}
	r.Base = processor.New("EvalPersonaTurnRelay", r)
	return r
}

// reportEndCall appends the persona's end_call event once the response in
// progress has gone out.
func (r *personaTurnRelay) reportEndCall(event map[string]any) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.replying {
		r.endCall = event
		return
	}
	r.stream.Append(event)
}

// ProcessFrame relays the persona's replies.
func (r *personaTurnRelay) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := r.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	r.stream.Touch()
	var send string
	r.mu.Lock()
	switch fr := f.(type) {
	case *frames.LLMFullResponseStartFrame:
		r.text, r.spoken = nil, nil
		r.replying = true
	case *frames.InterruptionFrame:
		// The bot heard what the user TTS spoke before the cut; a text-mode
		// reply is sent only once it is complete.
		r.text = nil
		r.endReplyLocked()
	case *frames.LLMTextFrame:
		if fr.SkipTTS != nil && *fr.SkipTTS {
			r.text = append(r.text, fr.Text)
		}
	case *frames.LLMFullResponseEndFrame:
		text := pytext.Strip(strings.Join(r.text, ""))
		r.text = nil
		if fr.SkipTTS != nil && *fr.SkipTTS && text != "" {
			r.trace.Log(fmt.Sprintf("send: %q (persona, text)", text))
			send = text
			r.spoken = append(r.spoken, text)
		}
	case *frames.TTSTextFrame:
		r.trace.Log(fmt.Sprintf("send: %q (persona, audio)", fr.Text))
		r.spoken = append(r.spoken, fr.Text)
	}
	r.mu.Unlock()
	if send != "" {
		message := frames.NewOutputTransportMessageUrgentFrame(r.textTurnMessage(send))
		if err := r.PushFrame(ctx, message, processor.Downstream); err != nil {
			return err
		}
	}
	if _, ok := f.(*frames.LLMFullResponseEndFrame); ok {
		r.mu.Lock()
		r.endReplyLocked()
		r.mu.Unlock()
	}
	return r.PushFrame(ctx, f, dir)
}

// endReplyLocked appends what went out of the reply as a turn, then the
// end_call event held for it.
func (r *personaTurnRelay) endReplyLocked() {
	if len(r.spoken) > 0 {
		r.stream.Append(map[string]any{keyType: EventPersonaTurn, keyText: strings.Join(r.spoken, " ")})
	}
	r.spoken = nil
	r.replying = false
	if r.endCall != nil {
		r.stream.Append(r.endCall)
		r.endCall = nil
	}
}

// ClientParams is what the scenario asks of the bot; the session derives it
// from its scenario.
type ClientParams struct {
	// BotAudio is whether the bot speaks; in text mode it is asked to skip TTS
	// at connect.
	BotAudio bool
	// UserAudio is whether the user's turns reach the bot as audio.
	UserAudio bool
	// UserSpeech is the TTS config the user's turns are synthesized with, or
	// nil; it sets the user audio rate.
	UserSpeech *SpeechConfig
	// CaptureBotAudio is whether the bot forwards its synthesized audio, for
	// the response transcription.
	CaptureBotAudio bool
	// CaptureBotImages is whether the bot reports the images it outputs.
	CaptureBotImages bool
	// ReportLevel is the function-call report level to ask of the bot, or ""
	// for its default.
	ReportLevel rtvi.FunctionCallReportLevel
	// VADEvents is whether to ask the bot for its raw VAD events.
	VADEvents bool
	// MarkerEvents is whether to ask the bot for the sideband markers its LLM
	// emits.
	MarkerEvents bool
	// Context is the messages the bot's context starts from, sent right after
	// the handshake; empty sends nothing.
	Context []map[string]any
	// TriggerDisconnect is whether the scenario itself asks a bot serving
	// eval.Handler to fire its client-disconnected handlers when this
	// connection ends; the run's SessionParams can ask too.
	TriggerDisconnect bool
}

// BotUnavailableError is a bot that refused the connection, closed it before
// bot-ready, or cannot carry the run's audio.
type BotUnavailableError struct {
	// Kind is the failure kind the run records: FailureConnectFailed when the
	// bot refused the connection, FailureClosedBeforeReady when it closed the
	// connection before bot-ready, FailureAudioUnsupported when its bot-ready
	// reported its transport off for audio the run uses, and FailureError when
	// the harness's own pipeline failed first.
	Kind string
	// Reason is what happened, with the close code and reason when there was
	// one.
	Reason string
}

func (e *BotUnavailableError) Error() string { return e.Reason }

// handshakeTimeoutError is a bot that never sent bot-ready.
type handshakeTimeoutError struct{ timeout time.Duration }

func (e *handshakeTimeoutError) Error() string {
	return fmt.Sprintf("bot-ready not received within %dms", e.timeout.Milliseconds())
}

// Client is the eval pipeline that talks to the bot, and the user's sends into
// it.
//
// Built once per run: WaitForBot, Start, Handshake, the turns (SendText, Say,
// Play, SendDTMF, SendImage), then Stop. The pipeline is
//
//	input -> [STT -> speech gate -> bot-turn aggregator] -> sink -> [persona LLM]
//	      -> [user TTS] -> [persona relay] -> output -> [assistant aggregator]
//
// The bracketed stages exist in audio mode (the STT that transcribes the bot's
// speech, the gate that drops transcripts of audio from before a send that
// talked over the bot, the aggregator that cuts the transcripts into the bot's
// turns, the user TTS) or in a simulation (the persona LLM, its relay, and the
// assistant aggregator that keeps the persona's own replies in its context). The
// bot's output comes in through the input and stops at the sink, as events; the
// user's turns start at the sink and leave through the output. In a simulation
// the persona answers the bot on its own from the sink. A session builds it,
// with the params its scenario asks for.
type Client struct {
	botURL        string
	stream        *EventStream
	trace         *Trace
	params        ClientParams
	sessionParams SessionParams
	userTTS       *CachingTTSService
	botSTT        *stt.SegmentService
	// persona is a simulation's simulated caller, or nil.
	persona *Persona
	// relay sends the persona's replies to the bot (built with the
	// processors), or nil.
	relay *personaTurnRelay
	// botTurns is the aggregator that turns the bot's transcribed speech into
	// turns (audio mode), kept so a send that talks over the bot can drop what
	// it holds of the bot's speech from before the send.
	botTurns botTurnResetter
	// newBotVAD builds the VAD the bot-turn aggregator hears the bot's speech
	// with; a test stands in for it.
	newBotVAD func() (vad.Analyzer, error)
	// botTurnStop are the stop strategies that hear the bot stop speaking, or
	// nil for the defaults; a test stands in for the default turn model this
	// way.
	botTurnStop []turns.StopStrategy
	// recorder records the conversation audio (bot and user) when the run
	// records and is in audio mode; fed raw audio by the transport, written on
	// Stop.
	recorder *ClientRecorder

	transport *ClientTransport
	// transportBuilt, when set, is handed the transport as it is built, before
	// the pipeline runs; a test hooks its edges this way.
	transportBuilt func(*ClientTransport)
	sink           *botFrameSink
	// injector is where the user's turns enter the pipeline: the sink, or what
	// a test stands in for it.
	injector frameInjector
	worker   *pipeline.Worker
	// runDone receives what running the worker returned.
	runDone chan error
	// stopping is set once Stop begins: a disconnect from then on is our own
	// teardown.
	stopping atomic.Bool
	// connected is set once the connection to the bot is open.
	connected atomic.Bool

	// handshakeOver is closed once the handshake is over: by the bot's
	// bot-ready, or by the bot refusing or closing the connection first, which
	// leaves refusal set. Handshake waits on it.
	mu            sync.Mutex
	handshakeOver chan struct{}
	refusal       *BotUnavailableError

	nextID atomic.Uint64
	// send is how a message reaches the pipeline; a test captures it instead.
	send func(rtvi.Message)
}

// NewClient builds the client for the bot at botURL. stream is where the bot's
// output is appended as events; userTTS speaks the user's text turns, or is nil
// for text turns; botSTT transcribes the bot's audio into the response event,
// or is nil when unused; persona is a simulation's simulated caller, whose LLM
// rides in the pipeline and answers the bot, or nil.
//
// The bot's speech is transcribed by a segmented STT or not at all: only a
// segmented STT's transcripts can be traced to when their audio began, which is
// how the client tells the speech a send talked over from the bot's reply.
func NewClient(
	botURL string, params ClientParams, sessionParams SessionParams,
	stream *EventStream, trace *Trace, userTTS *CachingTTSService, botSTT *stt.SegmentService, persona *Persona,
) *Client {
	c := &Client{
		botURL:        botURL,
		stream:        stream,
		trace:         trace,
		params:        params,
		sessionParams: sessionParams.withDefaults(),
		userTTS:       userTTS,
		botSTT:        botSTT,
		persona:       persona,
		newBotVAD:     newSileroBotVAD,
		handshakeOver: make(chan struct{}),
	}
	c.send = c.queueMessage
	return c
}

// newSileroBotVAD is the Silero VAD the bot-turn aggregator hears the bot with.
func newSileroBotVAD() (vad.Analyzer, error) {
	params := vad.DefaultParams()
	params.StopSecs = botVADStopSecs
	return vad.NewSilero(vad.WithParams(params))
}

// HasUserTTS reports whether text user turns are spoken to the bot rather than
// sent as text.
func (c *Client) HasUserTTS() bool { return c.userTTS != nil }

// SendsUserAudio reports whether the user's side streams audio to the bot.
func (c *Client) SendsUserAudio() bool { return c.userTTS != nil || c.params.UserAudio }

// WaitForBot waits until the bot accepts connections.
//
// It retries a plain TCP connect until the bot is listening, and stops short of
// the WebSocket handshake on purpose: a full connection would make the bot
// greet, and that connection would then be thrown away. It returns the last
// connect error when the bot never accepted within the connect timeout.
func (c *Client) WaitForBot(ctx context.Context) error {
	u, err := url.Parse(c.botURL)
	if err != nil {
		return err
	}
	host, port := u.Hostname(), u.Port()
	if host == "" {
		host = "localhost"
	}
	if port == "" {
		port = "80"
		if u.Scheme == "wss" {
			port = "443"
		}
	}
	addr := net.JoinHostPort(host, port)
	deadline := time.Now().Add(c.sessionParams.ConnectTimeout)
	var (
		dialer       net.Dialer
		connectError error
	)
	for time.Now().Before(deadline) {
		attemptCtx, cancel := context.WithDeadline(ctx, deadline)
		conn, err := dialer.DialContext(attemptCtx, "tcp", addr)
		cancel()
		if err == nil {
			_ = conn.Close()
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		// Not accepting connections yet.
		connectError = err
		if err := sleep(ctx, connectRetry); err != nil {
			return err
		}
	}
	if connectError == nil {
		return ErrTimeout
	}
	return connectError
}

// Start builds the eval pipeline and starts running it, which connects to the
// bot. It fails when a stage of the pipeline cannot be built: the VAD the bot's
// speech is heard with.
func (c *Client) Start(ctx context.Context) error {
	c.transport = c.newTransport()
	if c.transportBuilt != nil {
		c.transportBuilt(c.transport)
	}
	processors, err := c.processors(c.transport)
	if err != nil {
		return err
	}
	noRTVI, noIdleCancel := false, false
	c.worker = pipeline.NewWorker(pipeline.New(processors...),
		pipeline.WorkerConfig{
			Name:                "EvalClient",
			Params:              c.pipelineParams(),
			EnableRTVI:          &noRTVI,
			CancelOnIdleTimeout: &noIdleCancel,
		})

	botConnection := []any{c.transport.Input(), c.transport.Output()}
	event.On(&c.worker.Registry, pipeline.EventPipelineError, func(_ context.Context, ef *frames.ErrorFrame) {
		// An error before bot-ready ends the handshake. The transport connects
		// to the bot as it is set up, so a refused connection (an HTTP 403 for
		// a spent token, nothing listening) surfaces here, as does the
		// connection failing later. Any other error is the harness's: a
		// service failing, or a bug in the harness's own transport code.
		reason := describeError(ef)
		if fromBotConnection(ef, botConnection) && (!c.connected.Load() || isConnectionError(ef.Err)) {
			c.refuse(FailureConnectFailed, reason)
		} else {
			c.refuse(FailureError, reason)
		}
		c.harnessFailed(reason)
	})

	// Before the pipeline runs, so the persona is configured before it can
	// hear the bot, which may greet as soon as the input transport starts.
	if c.persona != nil {
		if err := c.configurePersona(ctx, c.persona.Instruction()); err != nil {
			return err
		}
	}

	c.runDone = make(chan error, 1)
	go func() {
		err := c.worker.Run(context.WithoutCancel(ctx))
		if err != nil && !c.stopping.Load() {
			// The worker fails to run when its processors cannot be set up,
			// and the transport connects to the bot as it is set up: a refused
			// connection (an HTTP 403 for a spent token, nothing listening)
			// surfaces here.
			if !c.connected.Load() {
				c.refuse(FailureConnectFailed, err.Error())
			} else {
				c.refuse(FailureError, err.Error())
			}
			c.harnessFailed(err.Error())
		}
		c.runDone <- err
	}()
	return nil
}

// harnessFailed records the harness's own pipeline failing, and reports it as an
// event.
func (c *Client) harnessFailed(reason string) {
	c.stream.HarnessFailed(reason)
	c.stream.Append(map[string]any{keyType: EventHarnessError, keyText: reason})
}

// fromBotConnection reports whether the error frame came from the bot
// connection's transports.
func fromBotConnection(ef *frames.ErrorFrame, botConnection []any) bool {
	for _, p := range botConnection {
		if ef.Source != nil && any(ef.Source) == p {
			return true
		}
	}
	return false
}

// isConnectionError reports whether err is a connection failing: a network
// error, or the WebSocket closing.
func isConnectionError(err error) bool {
	if err == nil {
		return false
	}
	var netErr net.Error
	var closeErr websocket.CloseError
	return errors.As(err, &netErr) || errors.As(err, &closeErr) ||
		errors.Is(err, io.EOF) || errors.Is(err, net.ErrClosed)
}

// describeError is what an error frame says: its message, and the error under
// it when there is one.
func describeError(ef *frames.ErrorFrame) string {
	if ef.Err == nil || strings.Contains(ef.Error, ef.Err.Error()) {
		return ef.Error
	}
	return ef.Error + ": " + ef.Err.Error()
}

// Handshake waits for bot-ready, then sends what the scenario asks of the bot
// and its context.
//
// A bot that never says ready is not a usable eval target, so this fails
// instead of sending turns to a half-started bot: with a *BotUnavailableError
// when the bot refuses the connection, closes it before bot-ready, or reports
// its transport off for audio the run uses, and with an error naming the
// timeout when bot-ready never comes.
func (c *Client) Handshake(ctx context.Context) error {
	// The transport sends client-ready on connect and fires its bot-ready
	// event when the bot answers; the handler closes handshakeOver.
	timeout := c.sessionParams.BotReadyTimeout
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case <-c.handshakeOver:
	case <-timer.C:
		return &handshakeTimeoutError{timeout: timeout}
	case <-ctx.Done():
		return ctx.Err()
	}
	c.mu.Lock()
	refusal := c.refusal
	c.mu.Unlock()
	if refusal != nil {
		return refusal
	}

	// Ask the bot's RTVI observer to expose what this scenario needs, for the
	// duration of this eval only (bots keep their defaults, and only the eval
	// transport understands this): raise the function-call report level if it
	// asserts on call names or arguments, and enable raw VAD speaking events
	// and LLM markers if it uses them.
	config := map[string]any{}
	if c.params.ReportLevel != "" {
		config["function_call_report_level"] = map[string]any{"*": c.params.ReportLevel}
	}
	if c.params.VADEvents {
		config["vad_user_speaking"] = true
	}
	if c.params.MarkerEvents {
		config["llm_markers"] = true
	}
	if len(config) > 0 {
		c.sendEval(evalws.ConfigureMessageType, config)
	}

	// Only send the eval-context when the scenario provides starting context.
	// An implicit empty one would race with the bot's own startup (a greeting
	// added as the client connected), wiping the bot's context right after it
	// set it up.
	if len(c.params.Context) > 0 {
		c.sendEval(evalws.ContextMessageType, map[string]any{"messages": c.params.Context})
	}
	return nil
}

// Stop saves the recording, ends the bot's session, and ends the pipeline.
func (c *Client) Stop(ctx context.Context) {
	c.stopping.Store(true)
	// Write the recording first: the recorder is the harness's own and fed raw
	// audio by the transport, so nothing below clears it, but writing here lands
	// it even if the teardown fails.
	c.writeRecording()
	// End the bot's session whatever its disconnect handler does, so it does
	// not wait out its pipeline's idle timeout.
	c.sendCancel()
	if c.worker == nil || c.runDone == nil {
		return
	}
	// End the worker (which disconnects the transport), falling back to cancel
	// if it does not wind down cleanly.
	c.worker.StopWhenDone()
	timer := time.NewTimer(stopTimeout)
	defer timer.Stop()
	select {
	case <-c.runDone:
	case <-timer.C:
		c.worker.Cancel(context.WithoutCancel(ctx), "eval client stop timed out")
		<-c.runDone
	}
}

// Send sends an RTVI client message to the bot through the transport pipeline.
func (c *Client) Send(message rtvi.Message) { c.send(message) }

// queueMessage queues message into the pipeline, which sends it to the bot.
func (c *Client) queueMessage(message rtvi.Message) {
	if c.worker == nil {
		return
	}
	c.worker.QueueFrame(frames.NewOutputTransportMessageUrgentFrame(message))
}

// SendText sends a text user turn as an RTVI send-text message.
func (c *Client) SendText(text string) {
	c.dropBotSpeechIfTalkingOver()
	c.Send(c.textTurnMessage(text))
}

// SendDTMF sends a DTMF keypress turn as one RTVI dtmf message. The bot handles
// the keys the way it handles a phone keypad.
func (c *Client) SendDTMF(keys string) {
	c.dropBotSpeechIfTalkingOver()
	buttons := make([]string, 0, len(keys))
	for _, key := range keys {
		buttons = append(buttons, string(key))
	}
	c.Send(c.message(rtvi.TypeDTMF, rtvi.DTMFData{Buttons: buttons}))
}

// SendImage registers an image for the current turn. The bot's eval transport
// serves it back when the bot asks for a user image. The file is sent as is.
func (c *Client) SendImage(imagePath string) error {
	data, err := os.ReadFile(imagePath) //nolint:gosec // the turn's image, as the scenario names it
	if err != nil {
		return err
	}
	format := mime.TypeByExtension(filepath.Ext(imagePath))
	if format == "" {
		format = "image/jpeg"
	}
	if i := strings.IndexByte(format, ';'); i >= 0 {
		format = format[:i]
	}
	c.trace.Log(fmt.Sprintf("send: image %s (%s)", filepath.Base(imagePath), format))
	c.sendEval(evalws.ImageMessageType, map[string]any{
		"image": base64.StdEncoding.EncodeToString(data), "format": format,
	})
	return nil
}

// Say speaks text as the user. The user TTS synthesizes it (cached) and the
// output paces it to the bot as live audio.
func (c *Client) Say(ctx context.Context, text string) error {
	c.dropBotSpeechIfTalkingOver()
	return c.inject(ctx, frames.NewTTSSpeakFrame(text))
}

// SayWithPauses speaks the user's text with real silence between its parts.
//
// Each text is synthesized by the user TTS (cached), and the turn goes out like
// a recording: one stretch of audio, paced to the bot and recorded.
func (c *Client) SayWithPauses(ctx context.Context, parts []TurnPart) error {
	if c.userTTS == nil {
		return errNoUserTTS
	}
	c.dropBotSpeechIfTalkingOver()
	rate := c.userTTS.SampleRate()
	var pcm []byte
	for _, part := range parts {
		if part.Text == "" {
			pcm = append(pcm, make([]byte, 2*(rate*part.PauseMS/1000))...)
			continue
		}
		audio, err := c.userTTS.Synthesize(ctx, part.Text)
		if err != nil {
			return err
		}
		pcm = append(pcm, audio...)
	}
	return c.injectUtterance(ctx, pcm, rate)
}

// Play plays a recording to the bot as the user's turn, instead of
// synthesizing one.
//
// It goes out like a spoken turn: resampled to the user audio rate, paced to the
// bot, and recorded.
func (c *Client) Play(ctx context.Context, path string) error {
	c.dropBotSpeechIfTalkingOver()
	pcm, rate, err := LoadUserAudio(path)
	if err != nil {
		return err
	}
	return c.injectUtterance(ctx, pcm, rate)
}

// injectUtterance pushes one stretch of the user's audio, bracketed like the
// user TTS's output, so the output flushes its final partial chunk on the stop
// frame.
func (c *Client) injectUtterance(ctx context.Context, pcm []byte, sampleRate int) error {
	for _, f := range []frames.Frame{
		frames.NewTTSStartedFrame(),
		frames.NewTTSAudioRawFrame(pcm, sampleRate, 1),
		frames.NewTTSStoppedFrame(),
	} {
		if err := c.inject(ctx, f); err != nil {
			return err
		}
	}
	return nil
}

// inject pushes a user turn into the pipeline at the sink.
func (c *Client) inject(ctx context.Context, f frames.Frame) error {
	if c.injector == nil {
		return errNotStarted
	}
	return c.injector.inject(ctx, f)
}

// dropBotSpeechIfTalkingOver drops, before a send that talks over the bot, its
// speech held in an open turn.
//
// An ordinary send waits for the bot's speech to be transcribed and closed into
// a turn. A send over a speaking bot cannot, and the bot's turn open at that
// moment holds transcripts of what it said before the send; left there, they
// would close into a turn after the bot's LLM has restarted and pass for the
// reply. The transcripts still with the STT are dropped by the gate as they land
// (see botSpeechGate).
func (c *Client) dropBotSpeechIfTalkingOver() {
	if c.botTurns == nil || !c.stream.BotSpeaking() {
		return
	}
	c.trace.Log("discard: bot speech held in its open turn before the interrupting send")
	c.botTurns.Reset()
}

// HangUp ends the persona's part of the conversation: it answers nothing more.
func (c *Client) HangUp() {
	if c.persona != nil {
		c.persona.HangUp()
	}
}

// ReportEndCall appends the persona's end_call event once the reply that made
// the call has gone out.
func (c *Client) ReportEndCall(event map[string]any) {
	if c.relay == nil {
		// No pipeline to send a reply through: nothing is going out.
		c.stream.Append(event)
		return
	}
	c.relay.reportEndCall(event)
}

// configurePersona gives the persona LLM its instruction, and says whether its
// replies are spoken or sent as text.
func (c *Client) configurePersona(ctx context.Context, instruction string) error {
	if err := c.sink.inject(ctx, frames.NewLLMUpdateSettingsFrame(&settings.LLM{
		SystemInstruction: settings.Set(instruction),
	})); err != nil {
		return err
	}
	return c.sink.inject(ctx, frames.NewLLMConfigureOutputFrame(!c.params.UserAudio))
}

// checkAudioSupport refuses the run when the bot reports its transport is off
// for audio the run uses.
//
// A transport that drops the caller's audio, or never sends the bot's, would
// otherwise pass for a bot ignoring the caller and use up the run's turn
// budget. A bot that reports nothing (nil) runs unchecked.
func (c *Client) checkAudioSupport(caps *capabilities.BotCapabilities) {
	if c.sessionParams.AudioSupportCheck != AudioSupportCheckWhenAdvertised || caps == nil {
		return
	}
	var off []string
	if c.SendsUserAudio() && caps.AudioIn != nil && !*caps.AudioIn {
		off = append(off, "receive the caller's audio (audio_in_enabled)")
	}
	if c.params.BotAudio && caps.AudioOut != nil && !*caps.AudioOut {
		off = append(off, "send its audio (audio_out_enabled)")
	}
	if len(off) > 0 {
		c.refuse(FailureAudioUnsupported,
			"The bot's eval transport is set not to "+strings.Join(off, " or "))
	}
}

// refuse ends the handshake with a refusal, unless it is already over.
func (c *Client) refuse(kind, reason string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	select {
	case <-c.handshakeOver:
		return
	default:
	}
	c.refusal = &BotUnavailableError{Kind: kind, Reason: reason}
	// A bot can greet as the run is torn down; the persona must not answer it.
	if c.persona != nil {
		c.persona.HangUp()
	}
	close(c.handshakeOver)
}

// endHandshake ends the handshake, unless it is already over.
func (c *Client) endHandshake() {
	c.mu.Lock()
	defer c.mu.Unlock()
	select {
	case <-c.handshakeOver:
	default:
		close(c.handshakeOver)
	}
}

// connectionFlags is what this connection asks of the bot: skip_tts in text
// mode, capture_bot_audio when the harness needs the bot's audio,
// capture_bot_images when it needs the bot's images.
func (c *Client) connectionFlags() evalws.ConnectionFlags {
	return evalws.ConnectionFlags{
		SkipTTS: !c.params.BotAudio,
		// Forward the bot's audio when something listens to it, or when
		// recording an audio run.
		CaptureBotAudio:  c.params.CaptureBotAudio || (c.sessionParams.RecordPath != "" && c.params.BotAudio),
		CaptureBotImages: c.params.CaptureBotImages,
	}
}

// connectURL is the bot's URL with this connection's eval flags as query
// parameters: the flags, and trigger_disconnect when the run asks for it, for a
// bot serving eval.Handler.
func (c *Client) connectURL() string {
	connection := c.connectionFlags()
	var flags []string
	if connection.SkipTTS {
		flags = append(flags, evalws.SkipTTSQueryParam+"=true")
	}
	if connection.CaptureBotAudio {
		flags = append(flags, evalws.CaptureAudioQueryParam+"=true")
	}
	if connection.CaptureBotImages {
		flags = append(flags, evalws.CaptureImagesQueryParam+"=true")
	}
	if c.sessionParams.TriggerDisconnect || c.params.TriggerDisconnect {
		flags = append(flags, evalws.TriggerDisconnectQueryParam+"=true")
	}
	if len(flags) == 0 {
		return c.botURL
	}
	sep := "?"
	if strings.Contains(c.botURL, "?") {
		sep = "&"
	}
	return c.botURL + sep + strings.Join(flags, "&")
}

// userAudioRate is the rate the user's audio is synthesized at; 0 in text mode.
func (c *Client) userAudioRate() int {
	if c.params.UserSpeech != nil {
		return TTSSampleRate(c.params.UserSpeech)
	}
	if c.userTTS != nil {
		return c.userTTS.SampleRate()
	}
	return 0
}

// newTransport is the transport to the bot, with the recorder when the run
// records, and its handlers.
func (c *Client) newTransport() *ClientTransport {
	params := wsclient.Params{
		AudioInEnabled: c.params.BotAudio,
		// The bot's audio travels on to the STT and the turn aggregator.
		AudioInPassthrough: true,
		AudioOutEnabled:    c.SendsUserAudio(),
		AudioOutSampleRate: c.userAudioRate(),
		Serializer:         NewClientSerializer(),
	}
	if c.params.BotAudio {
		params.AudioInSampleRate = evalws.STTSampleRate
	}
	// The recorder is fed the raw audio by both edges; the paced streams would
	// make the recording stutter.
	if c.sessionParams.RecordPath != "" && c.params.BotAudio {
		c.recorder = NewClientRecorder(cmp.Or(c.userAudioRate(), evalws.STTSampleRate))
	}
	tr := NewClientTransport(c.connectURL(), params, c.connectionFlags(), c.recorder)

	event.On(tr.Events(), wsclient.EventConnected, func(context.Context, *websocket.Conn) {
		c.connected.Store(true)
	})
	event.OnSignal(tr.Events(), wsclient.EventBotReady, func(context.Context) {
		c.connected.Store(true)
		c.checkAudioSupport(tr.BotCapabilities())
		c.endHandshake()
	})
	event.On(tr.Events(), wsclient.EventDisconnected, func(context.Context, *websocket.Conn) {
		// The bot ended the call by closing the connection; our own teardown
		// closes it too, and that is not the bot's doing.
		if c.stopping.Load() {
			return
		}
		// Before bot-ready a close is a refusal, such as a hosting service
		// turning away a session over its concurrency limit.
		c.refuse(FailureClosedBeforeReady, closeDescription(tr.CloseError()))
		c.stream.BotEnded()
		// The persona has nobody left to answer, so the sink marks the bot's
		// last turn complete in its place.
		if c.persona != nil {
			c.persona.HangUp()
		}
		c.stream.Append(map[string]any{keyType: EventBotEnded})
	})
	return tr
}

// processors are the pipeline's processors: inbound from the bot, the sink,
// outbound to the bot.
//
// The STT, the gate and the turn aggregator exist in audio mode, when the bot's
// speech is transcribed; the persona LLM, its relay, and the assistant
// aggregator in a simulation.
func (c *Client) processors(tr *ClientTransport) ([]processor.Processor, error) {
	// The context the aggregators keep: the persona's in a simulation, else a
	// throwaway that only gives the bot-turn aggregator somewhere to put turns.
	persona := c.persona
	convo := frames.NewLLMContext("")
	if persona != nil {
		convo = persona.Context()
	}
	// With no STT there is no aggregator to hand the persona the bot's turns;
	// it hears them through the sink instead.
	c.sink = newBotFrameSink(c.stream, persona, c.botSTT == nil)
	c.injector = c.sink

	inbound := []processor.Processor{tr.Input()}
	var aggregatorPair *aggregators.Pair
	if c.botSTT != nil {
		starts := &segmentStarts{}
		tagBotSegments(c.botSTT, starts)
		pair, err := c.botTurnAggregator(convo, persona != nil)
		if err != nil {
			return nil, err
		}
		aggregatorPair = pair
		turns := pair.User()
		c.botTurns = turns
		inbound = append(inbound, c.botSTT, newBotSpeechGate(c.stream, c.trace, starts), turns)
	}

	var speech []processor.Processor
	if c.userTTS != nil {
		speech = []processor.Processor{c.userTTS}
	}
	var outbound []processor.Processor
	if persona == nil {
		outbound = slices.Concat(speech, []processor.Processor{tr.Output()})
	} else {
		if aggregatorPair == nil {
			aggregatorPair = aggregators.New(convo)
		}
		c.relay = newPersonaTurnRelay(c.textTurnMessage, c.trace, c.stream)
		outbound = append(append([]processor.Processor{newPersonaStage(persona.LLM())}, speech...),
			c.relay, tr.Output(), aggregatorPair.Assistant())
	}
	return append(append(inbound, c.sink), outbound...), nil
}

// personaStage holds the persona LLM, and sets it up under a context that tags
// what it logs as the persona's (see PipelinePersona), so a run's debug log
// files the persona's logs under their own heading.
type personaStage struct {
	*pipeline.Pipeline
}

func newPersonaStage(service PersonaLLM) *personaStage {
	return &personaStage{Pipeline: pipeline.New(service)}
}

// Setup sets the persona LLM up under the persona's log tag.
func (p *personaStage) Setup(ctx context.Context, s processor.Setup) error {
	return p.Pipeline.Setup(withEvalPipeline(ctx, PipelinePersona), s)
}

// botTurnAggregator is the aggregator pair whose user half cuts the bot's
// transcribed speech into turns (audio mode), over convo.
//
// It takes the STT's transcriptions in to build the bot's turn, so the response
// comes from the aggregated turn text here rather than from a frame at the
// sink. Transcripts of audio from before an interrupting send never reach it
// (see botSpeechGate); the stream still holds a turn back until the bot's LLM
// has answered the send (see EventStream.BotTurnStopped). Its turn hooks give
// the stream the timing of the bot's replies, and its finished turn is the
// persona's next user message.
//
// persona is whether a persona answers the bot's turns (a simulation). Its LLM
// then also judges whether the bot finished what it was saying: the turn model
// hears a finished turn at the end of every sentence the bot's TTS speaks, and
// the persona waits for the rest of a reply that reads unfinished. Once it
// hangs up, the bot's report that it stopped speaking decides instead (see
// botFrameSink).
func (c *Client) botTurnAggregator(convo *frames.LLMContext, persona bool) (*aggregators.Pair, error) {
	analyzer, err := c.newBotVAD()
	if err != nil {
		return nil, fmt.Errorf("eval: the VAD that hears the bot's speech: %w", err)
	}
	options := []aggregators.Option{aggregators.WithVAD(aggregators.VADConfig{Analyzer: analyzer})}
	if strategies, set := c.botTurnStrategies(persona); set {
		options = append(options, aggregators.WithTurns(turns.Config{Strategies: strategies}))
	}
	pair := aggregators.New(convo, options...)
	user := pair.User()
	user.Events().Add(aggregators.EventUserTurnStarted, func(context.Context, any, ...any) {
		c.stream.BotTurnStarted()
	})
	user.Events().Add(aggregators.EventUserTurnStopped, func(_ context.Context, _ any, args ...any) {
		var text string
		if len(args) > 0 {
			if stopped, ok := args[0].(aggregators.UserTurnStopped); ok {
				text = stopped.Content
			}
		}
		c.stream.BotTurnStopped(text)
	})
	return pair, nil
}

// botTurnStrategies are the turn strategies of the bot-turn aggregator, and
// false when it keeps the defaults: a scripted run's turns are the turn
// model's, and a simulation's end once the persona's LLM marks them complete.
func (c *Client) botTurnStrategies(persona bool) (turns.UserTurnStrategies, bool) {
	if persona {
		return turns.FilterIncompleteUserTurnStrategies(c.botTurnStop, llm.UserTurnCompletionConfig{}), true
	}
	if c.botTurnStop != nil {
		return turns.UserTurnStrategies{Stop: c.botTurnStop}, true
	}
	return turns.UserTurnStrategies{}, false
}

// writeRecording writes the recorded conversation audio (bot and user) to the
// run's record path.
func (c *Client) writeRecording() {
	path := c.sessionParams.RecordPath
	if c.recorder == nil || path == "" || !c.recorder.HasAudio() {
		return
	}
	written, err := c.recorder.Write(path)
	if err != nil {
		slog.Warn("eval: could not save the recording", "path", path, "err", err)
		return
	}
	if written {
		c.trace.Log("recording saved: " + path)
	}
}

// pipelineParams are the StartFrame's rates: the bot STT reads the input rate,
// the user TTS produces the output rate. In text mode the STT rate is a
// placeholder.
func (c *Client) pipelineParams() pipeline.Params {
	out := c.userAudioRate()
	if out == 0 {
		out = evalws.STTSampleRate
	}
	return pipeline.Params{AudioInSampleRate: evalws.STTSampleRate, AudioOutSampleRate: out}
}

// message is one RTVI client message, numbered in order of creation.
func (c *Client) message(messageType string, data any) rtvi.Message {
	id := c.nextID.Add(1)
	return rtvi.Message{Label: rtvi.MessageLabel, Type: messageType, ID: strconv.FormatUint(id, 10), Data: data}
}

// sendEval sends one of the eval transport's own messages, in the
// client-message envelope.
func (c *Client) sendEval(evalType string, data any) {
	c.Send(c.message(rtvi.TypeClientMessage, map[string]any{"t": evalType, "d": data}))
}

// textTurnMessage is the RTVI send-text message for a user turn. The bot
// answers right away, and speaks the answer only in audio mode.
func (c *Client) textTurnMessage(text string) rtvi.Message {
	runImmediately, audioResponse := true, c.params.BotAudio
	return c.message(rtvi.TypeSendText, rtvi.SendTextData{
		Content: text,
		Options: &rtvi.SendTextOptions{RunImmediately: &runImmediately, AudioResponse: &audioResponse},
	})
}

// sendCancel asks the bot to cancel its pipeline, ending its session.
// Best-effort: the connection may already be gone.
func (c *Client) sendCancel() {
	c.sendEval(evalws.CancelMessageType, map[string]any{})
}
