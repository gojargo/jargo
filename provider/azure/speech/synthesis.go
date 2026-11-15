package speech

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/service/wsutil"
)

// This file speaks Azure's speech synthesis WebSocket protocol: the part of the
// Speech SDK's synthesizer the streaming TTS service relies on. A sentence is a
// turn on a connection held open across turns. The client describes the
// synthesis (synthesis.context) and sends the SSML (ssml); the service answers
// with turn.start, the audio stream's id (response), the audio itself (binary
// audio messages), word boundaries and the end of the session
// (audio.metadata), and turn.end. Every message carries the id of the turn it
// belongs to, so a turn stopped early is told apart from the next one.

const (
	// synthesisPath is the synthesis endpoint.
	synthesisPath = "/tts/cognitiveservices/websocket/v1"
	// synthesisConnectionImplParam is the query parameter that asks for the
	// WebSocket synthesis connection.
	synthesisConnectionImplParam = "synthesizer.synthesis.connection.synthesisConnectionImpl"
	// synthesisReadLimit bounds a single inbound message.
	synthesisReadLimit = 1 << 20
	// ticksPerSecond converts the protocol's offsets, in 100-nanosecond ticks.
	ticksPerSecond = 10_000_000
)

// Message paths of the synthesis protocol, beyond those it shares with
// recognition.
const (
	// pathSynthesisContext describes the synthesis about to start.
	pathSynthesisContext = "synthesis.context"
	// pathSSML carries the document to synthesize.
	pathSSML = "ssml"
	// pathSynthesisControl steers a synthesis in progress.
	pathSynthesisControl = "synthesis.control"
	// pathTurnStart opens a turn.
	pathTurnStart = "turn.start"
	// pathResponse names the audio stream of the turn.
	pathResponse = "response"
	// pathAudioMetadata carries word boundaries and the end of the session.
	pathAudioMetadata = "audio.metadata"
)

// Metadata types reported in audio.metadata.
const (
	metadataWordBoundary     = "WordBoundary"
	metadataSentenceBoundary = "SentenceBoundary"
	metadataSessionEnd       = "SessionEnd"
)

// errSynthesisNotConnected is returned when there is no connection to send on.
//
//nolint:gochecknoglobals // sentinel error
var errSynthesisNotConnected = errors.New("azurespeech: synthesis websocket not connected")

// cancellationReason says why a synthesis was canceled.
type cancellationReason int

const (
	// cancellationError is a synthesis that failed.
	cancellationError cancellationReason = iota + 1
	// cancellationCancelledByUser is a synthesis the client stopped.
	cancellationCancelledByUser
)

// String names the reason the way Azure does.
func (r cancellationReason) String() string {
	switch r {
	case cancellationError:
		return "Error"
	case cancellationCancelledByUser:
		return "CancelledByUser"
	default:
		return fmt.Sprintf("CancellationReason(%d)", int(r))
	}
}

// wordBoundaryEvent is a word, or a punctuation mark, the synthesis has
// reached, with where it starts in the turn's audio, in ticks.
type wordBoundaryEvent struct {
	text        string
	audioOffset int64
}

// synthesisResult is a completed synthesis.
type synthesisResult struct {
	// audioDuration is how long the synthesized audio is.
	audioDuration time.Duration
}

// cancellationDetails is why a synthesis was canceled.
type cancellationDetails struct {
	reason       cancellationReason
	errorDetails string
}

// synthesisHandlers are the events a synthesis reports. They are called one at
// a time, from the goroutine reading the connection or, for a cancellation,
// from the one that stopped or failed it.
type synthesisHandlers struct {
	synthesizing func(audio []byte)
	wordBoundary func(evt wordBoundaryEvent)
	completed    func(result synthesisResult)
	canceled     func(details cancellationDetails)
}

// synthesisTurn is the synthesis in progress: one turn on the connection.
type synthesisTurn struct {
	requestID    string
	streamID     string
	synthesizing bool
	audioBytes   int
	// sessionEnd is the offset the session ended at, in ticks, which is the
	// length of the audio.
	sessionEnd    int64
	hasSessionEnd bool
}

// speechSynthesizer synthesizes SSML over Azure's synthesis WebSocket protocol.
// The connection is opened with the first synthesis and reused by the next,
// and opened again when it has dropped.
type speechSynthesizer struct {
	endpoint     string
	apiKey       string
	outputFormat string
	// bytesPerSecond is the rate of the PCM requested, which times the audio
	// when the service does not report its length.
	bytesPerSecond int
	handlers       synthesisHandlers

	// writeMu serializes writes; the connection permits one writer at a time.
	writeMu sync.Mutex

	// mu guards everything below. It is never held across a handler or a write.
	mu        sync.Mutex
	conn      *wsutil.Conn
	connected bool
	turn      synthesisTurn
}

// synthesisEndpoint builds the URL the synthesizer connects to: the region's
// host, or a private endpoint given as its https URL, which takes the
// synthesis path unless it names one of its own.
func synthesisEndpoint(region, privateEndpoint string) (string, error) {
	var u *url.URL
	if privateEndpoint != "" {
		parsed, err := url.Parse(privateEndpoint)
		if err != nil {
			return "", fmt.Errorf("azurespeech: private endpoint: %w", err)
		}
		u = parsed
		switch strings.ToLower(u.Scheme) {
		case "https":
			u.Scheme = "wss"
		case "http":
			u.Scheme = "ws"
		}
		if u.Path == "" || u.Path == "/" || strings.EqualFold(u.Path, synthesisPath) {
			u.Path = synthesisPath
		}
	} else {
		u = &url.URL{Scheme: "wss", Host: region + ".tts.speech" + hostSuffix(region), Path: synthesisPath}
	}
	q := u.Query()
	q.Set(synthesisConnectionImplParam, "websocket")
	u.RawQuery = q.Encode()
	return u.String(), nil
}

// hostSuffix is the domain of a region's cloud: the China and US Government
// regions have their own.
func hostSuffix(region string) string {
	lower := strings.ToLower(region)
	switch {
	case strings.HasPrefix(lower, "china"):
		return ".azure.cn"
	case strings.HasPrefix(lower, "usgov"):
		return ".azure.us"
	default:
		return ".microsoft.com"
	}
}

// synthesisContextBody describes the synthesis to the service: the audio
// format, and the metadata wanted alongside it. Word and punctuation
// boundaries are reported, and the end of the session, whose offset is the
// length of the audio.
func synthesisContextBody(outputFormat string) string {
	body := map[string]any{
		"synthesis": map[string]any{
			"audio": map[string]any{
				"metadataOptions": map[string]any{
					"bookmarkEnabled":            false,
					"punctuationBoundaryEnabled": true,
					"sentenceBoundaryEnabled":    false,
					"sessionEndEnabled":          true,
					"visemeEnabled":              false,
					"wordBoundaryEnabled":        true,
				},
				"outputFormat": outputFormat,
			},
			"language": map[string]any{"autoDetection": false},
		},
	}
	b, _ := json.Marshal(body) //nolint:errchkjson // map of known-serializable values
	return string(b)
}

// speakSSML starts synthesizing ssml as a new turn, connecting first if there
// is no connection. It returns once the request is sent; the audio and the
// word boundaries are reported through the handlers as they arrive, and the
// synthesis ends by completing or being canceled. A synthesis that cannot be
// sent is canceled with the error.
func (s *speechSynthesizer) speakSSML(ctx context.Context, ssml string) {
	requestID := newRequestID()
	s.mu.Lock()
	s.turn = synthesisTurn{requestID: requestID, synthesizing: true}
	s.mu.Unlock()

	conn, err := s.fetchConnection(ctx, requestID)
	if err == nil {
		err = s.send(ctx, conn, pathSynthesisContext, requestID, "application/json",
			synthesisContextBody(s.outputFormat))
	}
	if err == nil {
		err = s.send(ctx, conn, pathSSML, requestID, "application/ssml+xml", ssml)
	}
	if err != nil {
		s.cancelSynthesisLocal(requestID, cancellationError, err.Error())
	}
}

// stopSpeaking stops the synthesis in progress: the service is told to stop,
// and the synthesis is canceled as stopped by the user, so what it was still
// to deliver is dropped.
func (s *speechSynthesizer) stopSpeaking(ctx context.Context) error {
	s.mu.Lock()
	turn, conn := s.turn, s.conn
	s.mu.Unlock()
	if !turn.synthesizing {
		return nil
	}
	var err error
	if conn != nil {
		err = s.send(ctx, conn, pathSynthesisControl, turn.requestID, "application/json", `{"action":"stop"}`)
	}
	s.cancelSynthesisLocal(turn.requestID, cancellationCancelledByUser, "")
	return err
}

// cancelSynthesisLocal cancels the synthesis requestID names, if it is still
// in progress, and reports it canceled.
func (s *speechSynthesizer) cancelSynthesisLocal(requestID string, reason cancellationReason, details string) {
	s.mu.Lock()
	if !s.turn.synthesizing || s.turn.requestID != requestID {
		s.mu.Unlock()
		return
	}
	s.turn.synthesizing = false
	s.mu.Unlock()
	s.handlers.canceled(cancellationDetails{reason: reason, errorDetails: details})
}

// fetchConnection returns the open connection, or opens one. A new connection
// is described to the service before anything else is sent on it, and starts
// being read.
func (s *speechSynthesizer) fetchConnection(ctx context.Context, requestID string) (*wsutil.Conn, error) {
	s.mu.Lock()
	if s.conn != nil && s.connected {
		conn := s.conn
		s.mu.Unlock()
		return conn, nil
	}
	s.mu.Unlock()

	header := http.Header{}
	header.Set("Ocp-Apim-Subscription-Key", s.apiKey)
	header.Set("X-ConnectionId", newRequestID())
	conn, err := wsutil.Dial(ctx, s.endpoint, header, synthesisReadLimit)
	if err != nil {
		return nil, err
	}
	if err := s.send(ctx, conn, pathSpeechConfig, requestID, "application/json", speechConfigBody()); err != nil {
		_ = conn.CloseNow()
		return nil, err
	}
	s.mu.Lock()
	old := s.conn
	s.conn, s.connected = conn, true
	s.mu.Unlock()
	if old != nil {
		_ = old.CloseNow()
	}
	go s.receive(context.WithoutCancel(ctx), conn)
	return conn, nil
}

// send writes a text message: the headers, a blank line, then the body.
func (s *speechSynthesizer) send(
	ctx context.Context, conn *wsutil.Conn, path, requestID, contentType, body string,
) error {
	if conn == nil {
		return errSynthesisNotConnected
	}
	payload := messageHeaders(path, requestID, contentType) + "\r\n" + body
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return conn.Write(ctx, websocket.MessageText, []byte(payload))
}

// receive reads the connection until it ends, dispatching each message. A
// connection lost while a synthesis is in progress cancels it with the error.
func (s *speechSynthesizer) receive(ctx context.Context, conn *wsutil.Conn) {
	for {
		typ, data, err := conn.Read(ctx)
		if err != nil {
			s.connectionClosed(conn, err)
			return
		}
		msg, err := parseMessage(typ, data)
		if err != nil {
			continue
		}
		s.dispatch(msg)
	}
}

// connectionClosed records that conn has ended, and cancels the synthesis it
// was carrying. A connection the synthesizer closed or replaced itself is no
// longer its connection, and ends nothing.
func (s *speechSynthesizer) connectionClosed(conn *wsutil.Conn, err error) {
	s.mu.Lock()
	if s.conn != conn {
		s.mu.Unlock()
		return
	}
	s.connected = false
	turn := s.turn
	s.mu.Unlock()
	if !turn.synthesizing {
		return
	}
	details := err.Error()
	if closeErr, ok := errors.AsType[websocket.CloseError](err); ok {
		details = fmt.Sprintf("%s websocket error code: %d", closeErr.Reason, int(closeErr.Code))
	} else if errors.Is(err, net.ErrClosed) {
		details = "connection closed"
	}
	s.cancelSynthesisLocal(turn.requestID, cancellationError, details)
}

// synthesisResponse is the body of a response message.
type synthesisResponse struct {
	Audio struct {
		StreamID string `json:"streamId"` //nolint:tagliatelle // Azure wire field
	} `json:"audio"`
}

// synthesisMetadata is the body of an audio.metadata message.
type synthesisMetadata struct {
	Metadata []struct {
		Type string `json:"Type"` //nolint:tagliatelle // Azure wire field
		Data struct {
			Offset int64 `json:"Offset"` //nolint:tagliatelle // Azure wire field
			Text   struct {
				Text string `json:"Text"` //nolint:tagliatelle // Azure wire field
			} `json:"text"`
		} `json:"Data"` //nolint:tagliatelle // Azure wire field
	} `json:"Metadata"` //nolint:tagliatelle // Azure wire field
}

// dispatch handles one message of the synthesis in progress. A message for any
// other turn is dropped: a turn that was stopped goes on arriving for a while.
func (s *speechSynthesizer) dispatch(msg sttMessage) {
	requestID := msg.header("X-RequestId")
	s.mu.Lock()
	if !s.turn.synthesizing || !strings.EqualFold(requestID, s.turn.requestID) {
		s.mu.Unlock()
		return
	}
	s.mu.Unlock()

	switch strings.ToLower(msg.path) {
	case pathTurnStart:
		// The turn has started; nothing to do until its audio arrives.
	case pathResponse:
		var r synthesisResponse
		if json.Unmarshal(msg.body, &r) == nil {
			s.mu.Lock()
			s.turn.streamID = r.Audio.StreamID
			s.mu.Unlock()
		}
	case pathAudio:
		s.mu.Lock()
		ours := strings.EqualFold(s.turn.streamID, msg.header("X-StreamId"))
		if ours {
			s.turn.audioBytes += len(msg.body)
		}
		s.mu.Unlock()
		if ours && len(msg.body) > 0 {
			s.handlers.synthesizing(msg.body)
		}
	case pathAudioMetadata:
		s.handleMetadata(msg.body)
	case pathTurnEnd:
		s.mu.Lock()
		if !s.turn.synthesizing {
			s.mu.Unlock()
			return
		}
		s.turn.synthesizing = false
		turn := s.turn
		s.mu.Unlock()
		s.handlers.completed(synthesisResult{audioDuration: s.audioDuration(turn)})
	}
}

// handleMetadata reports the word boundaries a metadata message carries, and
// records where the session ended.
func (s *speechSynthesizer) handleMetadata(body []byte) {
	var m synthesisMetadata
	if json.Unmarshal(body, &m) != nil {
		return
	}
	for _, md := range m.Metadata {
		switch md.Type {
		case metadataWordBoundary, metadataSentenceBoundary:
			s.handlers.wordBoundary(wordBoundaryEvent{text: md.Data.Text.Text, audioOffset: md.Data.Offset})
		case metadataSessionEnd:
			s.mu.Lock()
			s.turn.sessionEnd, s.turn.hasSessionEnd = md.Data.Offset, true
			s.mu.Unlock()
		}
	}
}

// audioDuration is how long a turn's audio is: where its session ended, or,
// when that was not reported, the length of the audio received.
func (s *speechSynthesizer) audioDuration(turn synthesisTurn) time.Duration {
	if turn.hasSessionEnd {
		return time.Duration(turn.sessionEnd) * time.Second / ticksPerSecond
	}
	if s.bytesPerSecond <= 0 {
		return 0
	}
	return time.Duration(turn.audioBytes) * time.Second / time.Duration(s.bytesPerSecond)
}

// close closes the connection, if one is open.
func (s *speechSynthesizer) close() error {
	s.mu.Lock()
	conn := s.conn
	s.conn, s.connected = nil, false
	s.mu.Unlock()
	if conn == nil {
		return nil
	}
	return conn.Close(websocket.StatusNormalClosure, "")
}
