package resembleai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsutil"
	uctx "github.com/gojargo/jargo/utils/context"
)

const (
	// bufferThresholdBytes is the size of the chunks the audio is passed on in.
	// Resemble AI may send an odd number of bytes even for 16-bit PCM, so the
	// audio is regrouped into even-sized chunks.
	bufferThresholdBytes = 2208
	// jitterBufferBytes is how much of a request's audio is gathered before any
	// of it is played, about a second at 22.05 kHz. Resemble AI sends its audio
	// in bursts with gaps of 300 to 450 ms between them, and playback that
	// started on the first burst would stall in each gap.
	jitterBufferBytes = 44100
)

// errNotConnected is returned by a send that found no connection to send on,
// the dial having failed and been reported.
//
//nolint:gochecknoglobals // sentinel error
var errNotConnected = errors.New("resembleai tts: websocket not connected")

// NewTTS builds a Resemble AI TTS service.
//
// It holds one connection open for the call, opened when the pipeline starts.
// When it drops it is restored with backoff, and when that keeps failing the
// service reports a permanent error and becomes unusable. Every sentence is a
// request of its own, which Resemble AI ends on its own once its audio is
// done, so each is spoken on an audio context of its own.
func NewTTS(cfg Config) *tts.Base {
	b, _ := newTTS(cfg)
	return b
}

// newTTS builds the service and returns its synthesizer alongside it.
func newTTS(cfg Config) (*tts.Base, *synthesizer) {
	s := newSynthesizer(cfg)
	b := tts.New("ResembleTTS", s)
	// A request cannot be added to once sent, and Resemble AI ends it with its
	// own audio_end, so the sentences of a turn cannot share a context.
	b.SetReuseContextIDWithinTurn(false)
	s.svc = b
	s.ws = tts.NewWebsocket(b, s, s.cfg.Websocket)
	return b, s
}

// newSynthesizer builds the synthesizer, with the defaults applied and its
// settings taken from cfg.
func newSynthesizer(cfg Config) *synthesizer {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	if cfg.Precision == "" {
		cfg.Precision = defaultPrecision
	}
	if cfg.OutputFormat == "" {
		cfg.OutputFormat = defaultOutputFormat
	}
	if cfg.Websocket.CloseTimeout == nil {
		// Resemble AI never acknowledges the closing handshake, so the close is
		// sent without waiting for an answer that does not come.
		var noWait time.Duration
		cfg.Websocket.CloseTimeout = &noWait
	}
	return &synthesizer{
		cfg:              cfg,
		live:             newTTSSettings(cfg),
		requestToContext: map[int64]string{},
		audioBuffers:     map[string][]byte{},
		playbackStarted:  map[string]bool{},
	}
}

// ttsService is the part of the TTS service the synthesizer reports through.
// tts.Base implements it.
type ttsService interface {
	Name() string
	StopAllMetrics(ctx context.Context)
}

// synthesizer speaks over one connection held open for the call. A sentence is
// sent as a numbered request and the call returns; its audio arrives on the
// receive loop, labeled with the request's number.
type synthesizer struct {
	cfg  Config
	host tts.AudioContextHost
	svc  ttsService
	ws   *tts.Websocket

	// live is the settings store, which the base updates on the frame
	// goroutine, where every request is built from it.
	live *TTSSettings

	// dialMu serializes opening and closing the connection: the receive loop
	// redials while a sentence may be reconnecting on the frame goroutine.
	dialMu sync.Mutex
	// writeMu serializes writes; the connection permits one writer at a time.
	writeMu sync.Mutex

	mu   sync.Mutex
	conn *wsConn

	// stateMu guards the request bookkeeping, which a sentence writes on the
	// frame goroutine and the receive loop reads. It is never held across a
	// call into the host: appending blocks while the audio plays out.
	stateMu sync.Mutex
	// requestIDCounter numbers the requests; Resemble AI only takes a number.
	requestIDCounter int64
	// requestToContext is the audio context each request was sent for.
	requestToContext map[int64]string
	// audioBuffers holds each context's audio not yet passed on, so it goes
	// out in even-sized chunks.
	audioBuffers map[string][]byte
	// playbackStarted says whether a context's jitter buffer has filled.
	playbackStarted map[string]bool
}

// wsConn is one connection, with the goroutine reading it.
//
// The reading runs apart from the receive loop so stopping the loop does not
// cut the connection before the close goes out.
type wsConn struct {
	conn *wsutil.Conn
	// reads carries each message read, and finally the error that ended the
	// reading.
	reads chan wsRead
	// stop is closed when the connection is being closed, which ends the reading
	// goroutine once nothing takes its messages any more.
	stop     chan struct{}
	stopOnce sync.Once
	// readEnded is set once the reading has failed, which is the connection
	// closing as far as the service is concerned.
	readEnded atomic.Bool
}

// wsRead is one message read from the connection, or the error that ended the
// reading.
type wsRead struct {
	data []byte
	err  error
}

func newWSConn(conn *wsutil.Conn) *wsConn {
	c := &wsConn{conn: conn, reads: make(chan wsRead), stop: make(chan struct{})}
	go c.read()
	return c
}

// read reads the connection until it ends, handing each message to whoever
// receives from reads.
func (c *wsConn) read() {
	for {
		_, data, err := c.conn.Read(context.Background())
		if err != nil {
			c.readEnded.Store(true)
		}
		select {
		case c.reads <- wsRead{data: data, err: err}:
		case <-c.stop:
			return
		}
		if err != nil {
			return
		}
	}
}

// halt stops handing messages on, for a connection about to be closed.
func (c *wsConn) halt() { c.stopOnce.Do(func() { close(c.stop) }) }

// SetAudioContextHost records the host this provider appends its audio to,
// implementing tts.ContextSynthesizer.
func (s *synthesizer) SetAudioContextHost(h tts.AudioContextHost) { s.host = h }

// SampleRate reports the requested output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// Start connects when the pipeline starts, implementing tts.Starter, so the
// handshake is not paid in front of the bot's first words.
func (s *synthesizer) Start(ctx context.Context) {
	s.Connect(ctx)
}

// Connect opens the connection and starts the receive loop on it, implementing
// tts.WebsocketService. A connection that cannot be opened is reported, and
// leaves no loop running: the next sentence tries again.
func (s *synthesizer) Connect(ctx context.Context) {
	s.ws.Connect()
	_ = s.ConnectWebsocket(ctx)
	if s.Connected() {
		s.ws.StartReceiveTask(ctx)
	}
}

// Disconnect ends the audio contexts of the connection, stops the receive loop
// and closes the connection, implementing tts.WebsocketService.
func (s *synthesizer) Disconnect(ctx context.Context) {
	s.ws.Disconnect()
	s.ws.CancelReceiveTask()
	_ = s.DisconnectWebsocket(ctx)
}

// ConnectWebsocket opens a connection unless one is open, implementing
// tts.WebsocketHandler. A failure is reported here and leaves no connection
// behind, which a redial reads as a failed attempt.
func (s *synthesizer) ConnectWebsocket(ctx context.Context) error {
	s.dialMu.Lock()
	defer s.dialMu.Unlock()
	if s.Connected() {
		return nil
	}
	slog.Debug("connecting to resemble ai tts")
	header := http.Header{}
	header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	conn, err := s.ws.Dial(ctx, s.cfg.URL, header, wsutil.DefaultReadLimit)
	if err != nil {
		s.mu.Lock()
		old := s.conn
		s.conn = nil
		s.mu.Unlock()
		if old != nil {
			old.halt()
			_ = old.conn.CloseNow()
		}
		ef := frames.NewErrorFrame(fmt.Sprintf("Unknown error occurred: %v", err))
		ef.Err = err
		s.ws.ReportError(ctx, ef, false)
		return nil
	}
	s.mu.Lock()
	old := s.conn
	s.conn = newWSConn(conn)
	s.mu.Unlock()
	if old != nil {
		// A connection whose reading had already ended, replaced rather than
		// closed: nothing is left to say to the server on it.
		old.halt()
		_ = old.conn.CloseNow()
	}
	return nil
}

// DisconnectWebsocket closes the connection, implementing
// tts.WebsocketHandler. What the service was timing is reported as far as it
// got first, and the bookkeeping of the requests sent on the connection is
// dropped: their audio will not come on another.
func (s *synthesizer) DisconnectWebsocket(ctx context.Context) error {
	s.dialMu.Lock()
	defer s.dialMu.Unlock()
	s.svc.StopAllMetrics(ctx)
	s.mu.Lock()
	c := s.conn
	s.conn = nil
	s.mu.Unlock()
	defer s.clearRequests()
	if c == nil {
		return nil
	}
	slog.Debug("disconnecting from resemble ai tts")
	c.halt()
	if err := c.conn.Close(websocket.StatusNormalClosure, ""); err != nil && s.host != nil {
		s.host.PushError(ctx, fmt.Sprintf("Unknown error occurred: %v", err), err, false)
	}
	return nil
}

// clearRequests drops the audio buffered and the requests tracked.
func (s *synthesizer) clearRequests() {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	clear(s.audioBuffers)
	clear(s.playbackStarted)
	clear(s.requestToContext)
}

// Connected reports whether a connection is open, implementing
// tts.WebsocketHandler.
func (s *synthesizer) Connected() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.conn != nil && !s.conn.readEnded.Load()
}

// ReceiveMessages handles the connection's messages until it ends, implementing
// tts.WebsocketHandler. The server closing the connection ends it without an
// error, which the base reads as a connection to restore.
func (s *synthesizer) ReceiveMessages(ctx context.Context) error {
	s.mu.Lock()
	c := s.conn
	s.mu.Unlock()
	if c == nil {
		return errNotConnected
	}
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-c.stop:
			// The connection was closed on this side, by an error message that
			// redialed: its reading is over.
			return nil
		case r := <-c.reads:
			if r.err != nil {
				switch websocket.CloseStatus(r.err) {
				case websocket.StatusNormalClosure, websocket.StatusGoingAway:
					return nil
				default:
					return r.err
				}
			}
			redialed, err := s.handleMessage(ctx, r.data)
			if err != nil {
				return err
			}
			if redialed {
				return nil
			}
		}
	}
}

// wsMessage is the subset of a Resemble AI WebSocket message read.
type wsMessage struct {
	Type            string        `json:"type"`
	RequestID       *int64        `json:"request_id"`
	AudioContent    string        `json:"audio_content"`
	AudioTimestamps *wsTimestamps `json:"audio_timestamps"`
	ErrorName       *string       `json:"error_name"`
	Message         *string       `json:"message"`
	StatusCode      int           `json:"status_code"`
}

// wsTimestamps is the timing Resemble AI reports with a chunk of audio: the
// characters spoken, each with its start and end in seconds.
type wsTimestamps struct {
	GraphChars []string    `json:"graph_chars"`
	GraphTimes [][]float64 `json:"graph_times"`
}

// contextFor is the audio context a request was sent for. A request this side
// never sent maps to its own number, which names no context.
func (s *synthesizer) contextFor(requestID *int64) string {
	if requestID == nil {
		return ""
	}
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	if id, ok := s.requestToContext[*requestID]; ok {
		return id
	}
	return strconv.FormatInt(*requestID, 10)
}

// handleMessage handles one server message, dispatching it to the context of
// the request it names. A message for a context that is no longer open is
// skipped: an interruption abandons a context mid-flight, and audio the server
// had already generated for it arrives afterwards. It reports whether an error
// message redialed, which leaves the connection being read closed.
func (s *synthesizer) handleMessage(ctx context.Context, data []byte) (bool, error) {
	m, empty, ok := decodeMessage(data)
	if !ok {
		s.host.PushError(ctx, "Received invalid JSON: "+string(data), nil, false)
		return false, nil
	}
	if empty {
		return false, nil
	}

	contextID := s.contextFor(m.RequestID)
	if !s.host.AudioContextAvailable(contextID) {
		return false, nil
	}

	switch m.Type {
	case "audio":
		return false, s.handleAudio(contextID, m)
	case "audio_end":
		s.handleAudioEnd(contextID, m.RequestID)
	case "error":
		return s.handleError(ctx, contextID, m), nil
	default:
		slog.Warn("resemble ai tts sent an unknown message type", "type", m.Type)
	}
	return false, nil
}

// decodeMessage decodes a message, reporting whether it is one at all and
// whether it is an empty object.
func decodeMessage(data []byte) (m *wsMessage, empty, ok bool) {
	var fields map[string]json.RawMessage
	if json.Unmarshal(data, &fields) != nil {
		return nil, false, false
	}
	m = &wsMessage{}
	if json.Unmarshal(data, m) != nil {
		return nil, false, false
	}
	return m, len(fields) == 0, true
}

// handleAudio buffers a chunk of audio and passes on what can be played: once
// the context's jitter buffer has filled, every whole chunk gathered. A chunk
// that is not base64 fails the connection rather than being played as rubbish.
func (s *synthesizer) handleAudio(contextID string, m *wsMessage) error {
	if m.AudioContent == "" {
		return nil
	}
	audio, err := base64.StdEncoding.DecodeString(m.AudioContent)
	if err != nil {
		return fmt.Errorf("resembleai tts: decode audio: %w", err)
	}
	if len(audio) == 0 {
		return nil
	}

	s.stateMu.Lock()
	if _, ok := s.audioBuffers[contextID]; !ok {
		s.audioBuffers[contextID] = nil
		s.playbackStarted[contextID] = false
	}
	s.audioBuffers[contextID] = append(s.audioBuffers[contextID], audio...)
	buffer := s.audioBuffers[contextID]
	if !s.playbackStarted[contextID] {
		// Nothing plays until the jitter buffer has filled, and the timings of
		// a chunk held back with it are not reported.
		if len(buffer) < jitterBufferBytes {
			s.stateMu.Unlock()
			return nil
		}
		s.playbackStarted[contextID] = true
	}
	var chunks [][]byte
	for len(buffer) >= bufferThresholdBytes {
		chunkSize := bufferThresholdBytes - bufferThresholdBytes%2
		chunks = append(chunks, bytes.Clone(buffer[:chunkSize]))
		buffer = buffer[chunkSize:]
	}
	s.audioBuffers[contextID] = buffer
	s.stateMu.Unlock()

	for _, chunk := range chunks {
		s.appendAudio(contextID, chunk)
	}

	if ts := m.AudioTimestamps; ts != nil {
		var words []uctx.WordTiming
		for i, char := range ts.GraphChars {
			if i >= len(ts.GraphTimes) {
				break
			}
			if times := ts.GraphTimes[i]; len(times) >= 2 {
				words = append(words, uctx.WordTiming{Word: char, Offset: times[0]})
			}
		}
		if len(words) > 0 {
			s.host.AddWordTimestamps(contextID, words, tts.WordTimingOptions{})
		}
	}
	return nil
}

// handleAudioEnd closes a request's context out: the audio still buffered goes
// out, trimmed to a whole number of 16-bit samples, then the stop frame rides
// the queue behind it, and the context is marked done. Resemble AI has ended
// the request on its side, so nothing more is sent for it.
func (s *synthesizer) handleAudioEnd(contextID string, requestID *int64) {
	s.stateMu.Lock()
	remaining := s.audioBuffers[contextID]
	if len(remaining)%2 != 0 {
		remaining = remaining[:len(remaining)-1]
	}
	delete(s.audioBuffers, contextID)
	if _, ok := s.playbackStarted[contextID]; ok {
		delete(s.playbackStarted, contextID)
		if requestID != nil {
			delete(s.requestToContext, *requestID)
		}
	}
	s.stateMu.Unlock()

	if len(remaining) > 0 {
		s.appendAudio(contextID, remaining)
	}
	stopped := frames.NewTTSStoppedFrame()
	stopped.ContextID = contextID
	s.host.AppendToAudioContext(contextID, stopped)
	s.host.RemoveAudioContext(contextID)
}

// handleError reports an error message: the request's speech stops now rather
// than behind audio that is not coming, and what was being timed is reported
// as far as it got. Rejected credentials close the connection and open a new
// one, and it reports whether it did.
func (s *synthesizer) handleError(ctx context.Context, contextID string, m *wsMessage) bool {
	errorName, message := "Unknown", "Unknown error"
	if m.ErrorName != nil {
		errorName = *m.ErrorName
	}
	if m.Message != nil {
		message = *m.Message
	}
	s.host.PushError(ctx, fmt.Sprintf("Error: %s (status %d): %s", errorName, m.StatusCode, message), nil, false)

	s.stateMu.Lock()
	delete(s.audioBuffers, contextID)
	delete(s.playbackStarted, contextID)
	s.stateMu.Unlock()

	stopped := frames.NewTTSStoppedFrame()
	stopped.ContextID = contextID
	_ = s.host.PushFrame(ctx, stopped, processor.Downstream)
	s.svc.StopAllMetrics(ctx)
	s.host.PushError(ctx, fmt.Sprintf("%s error: %s - %s", s.svc.Name(), errorName, message), nil, false)

	if m.StatusCode == http.StatusUnauthorized || m.StatusCode == http.StatusForbidden {
		_ = s.DisconnectWebsocket(ctx)
		_ = s.ConnectWebsocket(ctx)
		return true
	}
	return false
}

// appendAudio appends a chunk of audio to its context.
func (s *synthesizer) appendAudio(contextID string, audio []byte) {
	f := frames.NewTTSAudioRawFrame(audio, s.cfg.SampleRate, 1)
	f.ContextID = contextID
	s.host.AppendToAudioContext(contextID, f)
}

// OnAudioContextInterrupted stops the metrics of the context the user cut off,
// implementing tts.AudioContextInterrupter. Resemble AI has no way to cancel a
// request, so its audio still arrives and is dropped as it does.
func (s *synthesizer) OnAudioContextInterrupted(ctx context.Context, _ string) {
	s.svc.StopAllMetrics(ctx)
}

// RunTTS sends one sentence as a request of its own and returns: the audio
// arrives on the receive loop. A connection that has closed is opened again
// first. A send that fails ends the context's synthesis with a stop frame, is
// reported, and resets the connection, whose state on the server is unknown
// from here.
func (s *synthesizer) RunTTS(ctx context.Context, text, contextID string, yield func(f frames.Frame) error) error {
	if !s.Connected() {
		s.Connect(ctx)
	}

	s.stateMu.Lock()
	s.requestToContext[s.requestIDCounter] = contextID
	payload, err := s.buildMsg(text)
	s.stateMu.Unlock()
	if err == nil {
		err = s.send(ctx, payload)
	}
	if err == nil {
		tts.StartUsageMetrics(ctx, text)
		return nil
	}

	if yield != nil {
		stopped := frames.NewTTSStoppedFrame()
		stopped.ContextID = contextID
		_ = yield(stopped)
	}
	s.Disconnect(ctx)
	s.Connect(ctx)
	return fmt.Errorf("resembleai tts: %w", err)
}

// buildMsg is the request for text, numbered with the next request id. The
// caller holds stateMu.
func (s *synthesizer) buildMsg(text string) ([]byte, error) {
	var voice any
	if v, ok := s.live.Voice.Value(); ok {
		voice = v
	}
	var applyCustomPronunciations any
	if v, ok := s.live.ApplyCustomPronunciations.Value(); ok {
		applyCustomPronunciations = v
	}
	msg := map[string]any{
		"voice_uuid": voice,
		"data":       text,
		// JSON frames rather than binary ones, which carry the timestamps.
		"binary_response":             false,
		"request_id":                  s.requestIDCounter,
		"output_format":               s.cfg.OutputFormat,
		"sample_rate":                 s.cfg.SampleRate,
		"precision":                   s.cfg.Precision,
		"no_audio_header":             true,
		"apply_custom_pronunciations": applyCustomPronunciations,
	}
	s.requestIDCounter++
	return json.Marshal(msg)
}

// send writes one message on the open connection, serialized against the other
// writers, failing with errNotConnected when there is none.
func (s *synthesizer) send(ctx context.Context, payload []byte) error {
	s.mu.Lock()
	c := s.conn
	s.mu.Unlock()
	if c == nil {
		return errNotConnected
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return c.conn.Write(ctx, websocket.MessageText, payload)
}

var (
	_ tts.Synthesizer             = (*synthesizer)(nil)
	_ tts.ContextSynthesizer      = (*synthesizer)(nil)
	_ tts.WebsocketService        = (*synthesizer)(nil)
	_ tts.Starter                 = (*synthesizer)(nil)
	_ tts.AudioContextInterrupter = (*synthesizer)(nil)
)
