package eval

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"mime"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/pipeline/capabilities"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport/wsclient"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
)

// The harness's connection to the bot under test.
//
// Client runs the eval pipeline, a pipeline acting as an RTVI client: it waits
// for the bot to listen, connects, completes the handshake, sends the user's
// turns (text, speech, DTMF keys, an image), and tears down. The bot's output
// reaches the rest of the harness as events on the stream the client is given.

// connectRetry is how often the harness tries the bot again while it is not yet
// accepting connections.
const connectRetry = 250 * time.Millisecond

// stopTimeout is how long the pipeline has to wind down once ended before it is
// canceled.
const stopTimeout = 5 * time.Second

// errNotYet is a turn this harness cannot send yet.
var errNotYet = errors.New("eval: not supported yet")

// botFrame reports whether f is a frame the bot produced. The sink turns these
// into events and stops them there: everything downstream of the sink speaks
// for the user, and must never see the bot's text or reports as its own input.
// The lifecycle frames pass.
func botFrame(f frames.Frame) bool {
	switch f.(type) {
	case *frames.LLMFullResponseStartFrame, *frames.LLMTextFrame, *frames.LLMMarkerResponseFrame,
		*frames.LLMFullResponseEndFrame, *frames.TTSTextFrame, *frames.BotStartedSpeakingFrame,
		*frames.BotStoppedSpeakingFrame, *frames.FunctionCallInProgressFrame, *frames.FunctionCallResultFrame,
		*FunctionCallCancelFrame, *frames.InputTransportMessageFrame:
		return true
	}
	return false
}

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

// botFrameSink is where the bot's output stops and the user's turns start.
//
// The bot's frames become events on the stream and go no further. The user's
// turns are injected here, past the bot's side of the pipeline, so an
// interruption there cannot drop them. An interruption reaching the sink stops
// here: for a scripted turn it would flush the user audio being sent.
type botFrameSink struct {
	*processor.Base
	stream *EventStream
}

func newBotFrameSink(stream *EventStream) *botFrameSink {
	s := &botFrameSink{stream: stream}
	s.Base = processor.New("EvalBotFrameSink", s)
	return s
}

// inject pushes a user turn into the pipeline, past the bot's side.
func (s *botFrameSink) inject(ctx context.Context, f frames.Frame) error {
	return s.PushFrame(ctx, f, processor.Downstream)
}

// ProcessFrame turns the bot's frames into events.
func (s *botFrameSink) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	// The bot's frames arrive downstream, from the input transport. What
	// travels upstream is the user side's own, not bot output, and passes
	// through as is.
	if dir != processor.Downstream {
		return s.PushFrame(ctx, f, dir)
	}
	if event := s.stream.FrameToEvent(f); event != nil {
		s.stream.Append(event)
	}
	switch f.(type) {
	case *frames.InterruptionFrame, *frames.LLMContextFrame:
		return nil
	}
	if botFrame(f) {
		return nil
	}
	return s.PushFrame(ctx, f, dir)
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
// SendDTMF, SendImage), then Stop. The pipeline is
//
//	input -> sink -> output
//
// The bot's output comes in through the input and stops at the sink, as events;
// the user's turns start at the sink and leave through the output. A session
// builds it, with the params its scenario asks for.
type Client struct {
	botURL        string
	stream        *EventStream
	trace         *Trace
	params        ClientParams
	sessionParams SessionParams
	userTTS       *CachingTTSService

	transport *ClientTransport
	sink      *botFrameSink
	worker    *pipeline.Worker
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
// for text turns.
func NewClient(
	botURL string, params ClientParams, sessionParams SessionParams,
	stream *EventStream, trace *Trace, userTTS *CachingTTSService,
) *Client {
	c := &Client{
		botURL:        botURL,
		stream:        stream,
		trace:         trace,
		params:        params,
		sessionParams: sessionParams.withDefaults(),
		userTTS:       userTTS,
		handshakeOver: make(chan struct{}),
	}
	c.send = c.queueMessage
	return c
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
// bot.
func (c *Client) Start(ctx context.Context) {
	c.transport = c.newTransport()
	c.sink = newBotFrameSink(c.stream)
	noRTVI, noIdleCancel := false, false
	c.worker = pipeline.NewWorker(pipeline.New(c.transport.Input(), c.sink, c.transport.Output()),
		pipeline.WorkerConfig{
			Name:                "EvalClient",
			Params:              c.pipelineParams(),
			EnableRTVI:          &noRTVI,
			CancelOnIdleTimeout: &noIdleCancel,
		})

	botConnection := []any{c.transport.Input(), c.transport.Output()}
	events.On(&c.worker.Registry, pipeline.EventPipelineError, func(_ context.Context, ef *frames.ErrorFrame) {
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

// Stop ends the bot's session and the pipeline.
func (c *Client) Stop(ctx context.Context) {
	c.stopping.Store(true)
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
	c.Send(c.textTurnMessage(text))
}

// SendDTMF sends a DTMF keypress turn as one RTVI dtmf message. The bot handles
// the keys the way it handles a phone keypad.
func (c *Client) SendDTMF(keys string) {
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

// Say speaks text as the user: the user TTS synthesizes it (cached) and it is
// streamed to the bot as live audio.
func (c *Client) Say(ctx context.Context, text string) error {
	pcm, err := c.userTTS.Synthesize(ctx, text)
	if err != nil {
		return err
	}
	return c.streamUserAudio(ctx, pcm, c.userTTS.SampleRate())
}

// SayWithPauses speaks the user's text with real silence between its parts:
// each text is synthesized by the user TTS (cached), and the turn goes out as
// one stretch of audio.
func (c *Client) SayWithPauses(ctx context.Context, parts []TurnPart) error {
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
	return c.streamUserAudio(ctx, pcm, rate)
}

// Play plays a recording to the bot as the user's turn, instead of synthesizing
// one.
func (c *Client) Play(_ context.Context, path string) error {
	return fmt.Errorf("%w: playing an audio file as a turn (%s)", errNotYet, path)
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

// newTransport is the transport to the bot, with its handlers.
func (c *Client) newTransport() *ClientTransport {
	params := wsclient.Params{
		AudioInEnabled: c.params.BotAudio,
		// The user's audio is streamed to the bot as raw-audio messages the
		// client builds, so the output carries no audio of its own.
		AudioOutEnabled:    false,
		AudioOutSampleRate: c.userAudioRate(),
		Serializer:         NewClientSerializer(),
	}
	if c.params.BotAudio {
		params.AudioInSampleRate = evalws.STTSampleRate
	}
	tr := NewClientTransport(c.connectURL(), params, c.connectionFlags())

	events.On(tr.Events(), wsclient.EventConnected, func(context.Context, *websocket.Conn) {
		c.connected.Store(true)
	})
	events.OnSignal(tr.Events(), wsclient.EventBotReady, func(context.Context) {
		c.connected.Store(true)
		c.checkAudioSupport(tr.BotCapabilities())
		c.endHandshake()
	})
	events.On(tr.Events(), wsclient.EventDisconnected, func(context.Context, *websocket.Conn) {
		// The bot ended the call by closing the connection; our own teardown
		// closes it too, and that is not the bot's doing.
		if c.stopping.Load() {
			return
		}
		// Before bot-ready a close is a refusal, such as a hosting service
		// turning away a session over its concurrency limit.
		c.refuse(FailureClosedBeforeReady, closeDescription(tr.CloseError()))
		c.stream.BotEnded()
		c.stream.Append(map[string]any{keyType: EventBotEnded})
	})
	return tr
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
