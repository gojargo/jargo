package speechmatics

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
)

// EventSpeakersResult fires when Speechmatics answers a GetSpeakers request
// (see SendMessage) with the speakers its diarization identified. The handler
// is given the raw SpeakersResult message. It fires only with diarization on.
//
//	events.On(svc.Events(), speechmatics.EventSpeakersResult,
//	    func(ctx context.Context, msg map[string]any) { … })
const EventSpeakersResult = "on_speakers_result"

// fieldMessage is the field every message names its type in.
const fieldMessage = "message"

// Message types the Agent STT service sends.
const (
	msgRecognitionStarted = "RecognitionStarted"
	msgAddPartialSegment  = "AddPartialSegment"
	msgAddSegment         = "AddSegment"
	msgStartOfTurn        = "StartOfTurn"
	msgEndOfTurn          = "EndOfTurn"
	msgSpeakersResult     = "SpeakersResult"
	msgEndOfTranscript    = "EndOfTranscript"
	msgWarning            = "Warning"
	msgError              = "Error"
)

// errUnsupportedSampleRate is returned for a session at a rate the service
// does not accept.
//
//nolint:gochecknoglobals // sentinel error
var errUnsupportedSampleRate = errors.New("speechmatics stt: unsupported sample rate")

// errNotConnected is returned by SendMessage when no session is open.
//
//nolint:gochecknoglobals // sentinel error
var errNotConnected = errors.New("speechmatics stt: session is not connected")

// errNotSpeechmatics is returned by a call given a service this package did not
// build.
//
//nolint:gochecknoglobals // sentinel error
var errNotSpeechmatics = errors.New("speechmatics stt: not a Speechmatics STT service")

// NewSTT builds a Speechmatics streaming STT service on the Agent STT API.
func NewSTT(cfg Config) *stt.StreamService {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	c := &connector{cfg: cfg, live: newSTTSettings(cfg)}
	svc := stt.NewStream("SpeechmaticsSTT", c, cfg.SampleRate)
	c.svc = svc
	// Registered whatever the configuration: diarization can be turned on at
	// runtime, and a handler added while it was off would otherwise be dropped.
	svc.Events().Register(EventSpeakersResult, false)
	return svc
}

// SendMessage sends a message to the Speechmatics session svc holds open, with
// fields alongside the message type: GetSpeakers, say, which Speechmatics
// answers with the SpeakersResult that EventSpeakersResult reports. It fails
// when svc is not a Speechmatics STT service, when no session is open, and when
// the message could not be sent.
func SendMessage(svc *stt.StreamService, message string, fields map[string]any) error {
	c, ok := svc.Connector().(*connector)
	if !ok {
		return errNotSpeechmatics
	}
	s := c.current()
	if s == nil {
		return fmt.Errorf("%w: cannot send %s", errNotConnected, message)
	}
	payload := maps.Clone(fields)
	if payload == nil {
		payload = map[string]any{}
	}
	payload[fieldMessage] = message
	slog.Debug("speechmatics stt: sending message", "payload", payload)
	if err := s.writeJSON(payload); err != nil {
		return fmt.Errorf("speechmatics stt: error sending %s: %w", message, err)
	}
	return nil
}

// SessionID is the id Speechmatics gave the session svc holds open, or "" when
// none is open.
func SessionID(svc *stt.StreamService) string {
	c, ok := svc.Connector().(*connector)
	if !ok {
		return ""
	}
	if s := c.current(); s != nil {
		return s.sessionID
	}
	return ""
}

type connector struct {
	cfg Config
	// live is what may change while the pipeline runs.
	live *Settings
	// svc is the service this connector drives, for reporting the errors
	// Speechmatics sends during a session and the speakers it identifies. It
	// is nil for a connector built outside NewSTT.
	svc *stt.StreamService

	mu sync.Mutex
	// stream is the session open now, nil when there is none.
	stream *stream
}

// errorTypeNotAuthorised is the Error type for a rejected credential, spelled
// as Speechmatics spells it.
const errorTypeNotAuthorised = "not_authorised" //nolint:misspell // the provider's own spelling

// permanentServerErrorTypes are the server Error types that reject the request
// or the account, so a fresh session with the same configuration would fail the
// same way. Every other type (idle and session timeouts, buffer, data and
// internal errors) ends the session, but a new one can succeed.
//
//nolint:gochecknoglobals // read-only lookup table
var permanentServerErrorTypes = map[string]bool{
	"invalid_message":       true,
	"invalid_model":         true,
	"invalid_config":        true,
	"invalid_audio_type":    true,
	"invalid_output_format": true,
	errorTypeNotAuthorised:  true,
	"insufficient_funds":    true,
	"not_allowed":           true,
	"protocol_error":        true,
	"quota_exceeded":        true,
}

// serverError is an Error message Speechmatics sent during a session. It
// always ends the session.
type serverError struct {
	kind   string
	status string
}

func (e *serverError) Error() string { return "speechmatics stt error: " + e.status }

func (e *serverError) Unwrap() error { return errServer }

// permanent reports whether the error rejects the request or the account, so
// that reconnecting would only be refused the same way.
func (e *serverError) permanent() bool { return permanentServerErrorTypes[e.kind] }

// startRejectedError is Speechmatics ending a session before it started: an
// Error, or an EndOfTranscript, in place of RecognitionStarted. A session
// opened with the same configuration is rejected the same way, so it never
// clears on retry.
type startRejectedError struct {
	status string
}

func (e *startRejectedError) Error() string {
	return "Speechmatics STT rejected the session: " + e.status
}

func (e *startRejectedError) Unwrap() error { return errServer }

// reportServerError puts an error Speechmatics sent during a session on the
// pipeline. One that rejects the request or the account is reported as
// permanent, which costs the service its usability and keeps the read loop from
// reconnecting into it; any other is reported as recoverable, and the session
// it ended is reopened.
func (c *connector) reportServerError(ctx context.Context, err *serverError) {
	if c.svc == nil {
		return
	}
	var opts []processor.ErrorOption
	if err.permanent() {
		opts = append(opts, processor.ForceTreatAsPermanent())
	}
	c.svc.PushError(ctx, err.Error(), err, false, opts...)
}

// speakersResult reports the speakers Speechmatics identified, through
// EventSpeakersResult.
func (c *connector) speakersResult(ctx context.Context, msg map[string]any) {
	slog.Debug("speechmatics stt: speakers result received")
	if c.svc == nil {
		return
	}
	c.svc.Events().Call(ctx, EventSpeakersResult, c.svc, msg)
}

// Metadata reports the transcript latency the turn strategies size their wait
// by, and the model. When Speechmatics closes turns itself it proposes where
// they start and end, so it also recommends the external turn strategies that
// resolve those proposals, carrying the should-interrupt setting. They apply
// unless the application configured its own.
func (c *connector) Metadata() stt.Metadata {
	m := stt.Metadata{
		TTFSP99: stt.TTFSP99Or(c.cfg.TTFSP99, stt.SpeechmaticsTTFSP99),
		Model:   c.live.Model.Or(defaultModel),
	}
	if c.live.serviceClosesTurns() {
		m.UserTurnStrategies = turns.ExternalStrategies(turns.ExternalStrategiesConfig{
			EnableInterruptions: c.cfg.ShouldInterrupt,
		})
	}
	return m
}

// Settings is the configuration a caller may change while the pipeline runs.
func (c *connector) Settings() any { return c.live }

// UpdateSettings asks for the session to be reopened when a field Speechmatics
// is told when a session opens changed. The speaker format is applied locally,
// so a change to it alone takes effect at once.
func (c *connector) UpdateSettings(_ context.Context, changed settings.Changed) (bool, error) {
	for name := range changed {
		if !localFields[name] {
			slog.Debug("speechmatics stt: settings update requires reconnect", "fields", changed.String())
			return true, nil
		}
	}
	slog.Debug("speechmatics stt: local settings update, no reconnect", "fields", changed.String())
	return false, nil
}

// WebsocketOptions hands the stream base the connection settings the
// configuration gives, implementing wsservice.Optioner.
func (c *connector) WebsocketOptions() wsservice.Options { return c.cfg.Websocket }

// current is the session open now, or nil.
func (c *connector) current() *stream {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.stream
}

// setCurrent records s as the session open now.
func (c *connector) setCurrent(s *stream) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.stream = s
}

// forget clears s as the session open now, once it has closed.
func (c *connector) forget(s *stream) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.stream == s {
		c.stream = nil
	}
}

// Connect dials the Agent STT WebSocket, sends StartRecognition, and waits for
// the RecognitionStarted acknowledgement before audio flows. A sample rate the
// service does not accept is refused for good before anything is dialed: every
// attempt would be refused the same way.
func (c *connector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	if !slices.Contains(supportedSampleRates, sampleRate) {
		return nil, stt.Permanent(fmt.Errorf(
			"%w: %d Hz (supported: 8000, 16000)", errUnsupportedSampleRate, sampleRate))
	}

	header := http.Header{}
	header.Set("Authorization", "Bearer "+c.cfg.APIKey)

	conn, err := c.cfg.Websocket.Dial(ctx, resolveURL(c.cfg.URL), header, readLimit)
	if err != nil {
		return nil, err
	}

	s := &stream{
		conn:        conn,
		ctx:         ctx,
		sampleRate:  sampleRate,
		closesTurns: c.live.serviceClosesTurns(),
		diarization: c.live.EnableDiarization.Or(false),
		report:      c.reportServerError,
		speakers:    c.speakersResult,
		closed:      c.forget,
	}
	if err := conn.Write(ctx, websocket.MessageText, c.startRecognition(sampleRate)); err != nil {
		_ = conn.Close(websocket.StatusInternalError, "start failed")
		return nil, err
	}
	if err := s.awaitStarted(ctx); err != nil {
		_ = conn.Close(websocket.StatusInternalError, "start failed")
		if _, rejected := errors.AsType[*startRejectedError](err); rejected {
			// A session opened with the same configuration is rejected the same
			// way, so the service reports it as permanent and does not retry.
			return nil, stt.Permanent(err)
		}
		return nil, err
	}
	c.setCurrent(s)
	if s.closesTurns {
		return s, nil
	}
	// The pipeline closes turns, by having the session finalize when its VAD
	// reports the speech over.
	return &externalStream{stream: s}, nil
}

// startRecognition is the StartRecognition message a session opens with. The
// turn config is a sibling of the transcription config, not a member of it.
func (c *connector) startRecognition(sampleRate int) []byte {
	msg := map[string]any{
		fieldMessage: "StartRecognition",
		"audio_format": map[string]any{
			"type":        "raw",
			"encoding":    audioEncoding,
			"sample_rate": sampleRate,
		},
		"transcription_config": c.live.transcriptionConfig(),
		"turn_config": map[string]any{
			"turn_detection_mode": c.live.TurnDetectionMode.Or(defaultTurnDetectionMode),
		},
	}
	b, _ := json.Marshal(msg) //nolint:errchkjson // map of known-serializable values
	return b
}

// resolveURL is the Agent STT URL for a real-time URL: the agent path segment
// is appended unless it is already there, and the application is reported as
// sm-app, keeping any other parameters on the URL. A URL that does not parse is
// left as it is, for the dial to report.
func resolveURL(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return raw
	}
	var segments []string
	for segment := range strings.SplitSeq(u.Path, "/") {
		if segment != "" {
			segments = append(segments, segment)
		}
	}
	if !slices.Contains(segments, agentPathSegment) {
		segments = append(segments, agentPathSegment)
	}
	u.Path = "/" + strings.Join(segments, "/")
	q := u.Query()
	q.Set("sm-app", appName)
	u.RawQuery = q.Encode()
	return u.String()
}

// speechmaticsLanguages are the input languages Speechmatics supports, each
// under the code it takes for it.
//
//nolint:gochecknoglobals // lookup table, read-only
var speechmaticsLanguages = map[language.Language]string{
	language.Arabic:              "ar",
	language.Bashkir:             "ba",
	language.Basque:              "eu",
	language.Belarusian:          "be",
	language.Bulgarian:           "bg",
	language.Bengali:             "bn",
	language.YueChineseCantonese: "yue",
	language.Catalan:             "ca",
	language.Croatian:            "hr",
	language.Czech:               "cs",
	language.Danish:              "da",
	language.Dutch:               "nl",
	language.English:             "en",
	language.Esperanto:           "eo",
	language.Estonian:            "et",
	language.Persian:             "fa",
	language.Finnish:             "fi",
	language.French:              "fr",
	language.Galician:            "gl",
	language.German:              "de",
	language.Greek:               "el",
	language.Hebrew:              "he",
	language.Hindi:               "hi",
	language.Hungarian:           "hu",
	language.Italian:             "it",
	language.Indonesian:          "id",
	language.Irish:               "ga",
	language.Japanese:            "ja",
	language.Korean:              "ko",
	language.Latvian:             "lv",
	language.Lithuanian:          "lt",
	language.Malay:               "ms",
	language.Maltese:             "mt",
	language.MandarinChinese:     "cmn",
	language.Marathi:             "mr",
	language.Mongolian:           "mn",
	language.Norwegian:           "no",
	language.Polish:              "pl",
	language.Portuguese:          "pt",
	language.Romanian:            "ro",
	language.Russian:             "ru",
	language.Slovak:              "sk",
	language.Slovenian:           "sl",
	language.Spanish:             "es",
	language.Swedish:             "sv",
	language.Swahili:             "sw",
	// Speechmatics' Tagalog pack also covers Filipino, its standardized
	// register, so both map onto the one code the provider offers.
	language.Tagalog:    "tl",
	language.Filipino:   "tl",
	language.Tamil:      "ta",
	language.Thai:       "th",
	language.Turkish:    "tr",
	language.Uyghur:     "ug",
	language.Ukrainian:  "uk",
	language.Urdu:       "ur",
	language.Vietnamese: "vi",
	language.Welsh:      "cy",
}

// speechmaticsLanguage maps a Language to the language code Speechmatics takes.
// It takes the base code, so a regional variant resolves through its base
// language, and a language not listed is still sent under its own base code,
// with a warning.
func speechmaticsLanguage(l language.Language) string {
	return language.Resolve(l, speechmaticsLanguages, true)
}

// speechmaticsLocales are the output locales Speechmatics offers, by the
// language code they belong to.
//
//nolint:gochecknoglobals // read-only lookup table
var speechmaticsLocales = map[string]map[language.Language]string{
	"en": {
		language.EnglishGB: "en-GB",
		language.EnglishUS: "en-US",
		language.EnglishAU: "en-AU",
	},
}

// speechmaticsLocale is the output locale Speechmatics takes for a regional
// language whose code is base, or "" when the language names no region, or a
// region with no output locale, which leaves the transcript in the base
// language.
func speechmaticsLocale(base string, l language.Language) string {
	locales, ok := speechmaticsLocales[base]
	if !strings.Contains(string(l), "-") || !ok {
		return ""
	}
	if locale, ok := locales[l]; ok {
		return locale
	}
	slog.Warn("unsupported Speechmatics output locale, defaulting to the base language",
		"locale", string(l), "language", base)
	return ""
}

type stream struct {
	conn       *wsutil.Conn
	ctx        context.Context
	sampleRate int
	writeMu    sync.Mutex
	seqNo      atomic.Uint64
	// audioBytes is the audio sent so far, which stamps a finalize with the
	// point in the audio the speech ended at.
	audioBytes atomic.Int64
	// closesTurns is whether Speechmatics closes turns in this session, which
	// is when its turn events are acted on.
	closesTurns bool
	// diarization is whether this session tells speakers apart, which is when
	// its speakers results are reported.
	diarization bool
	// sessionID is the id Speechmatics gave the session when it started.
	sessionID string
	// report puts an error Speechmatics sent during the session on the
	// pipeline, before the session ends on it.
	report func(ctx context.Context, err *serverError)
	// speakers reports the speakers Speechmatics identified.
	speakers func(ctx context.Context, msg map[string]any)
	// closed is told once the session is closed.
	closed func(s *stream)
	// ending is set once EndOfStream has been sent. What the session says
	// after it is read only to see it end, and is not reported.
	ending atomic.Bool
}

// externalStream is a session whose turns the pipeline closes: it finalizes
// when the VAD reports the speech over.
type externalStream struct {
	*stream
}

// Finalize closes the current turn, implementing stt.Finalizer. It sends
// ForceEndOfUtterance stamped with the audio sent so far, so Speechmatics
// closes the turn where the VAD heard the speech end rather than wherever the
// message lands. The segment it flushes arrives as a normal AddSegment.
func (s *externalStream) Finalize() error {
	seconds := float64(s.audioBytes.Load()) / float64(s.sampleRate*bytesPerSample)
	return s.writeJSON(map[string]any{fieldMessage: "ForceEndOfUtterance", "timestamp": seconds})
}

// awaitStarted reads until RecognitionStarted. An Error, or an EndOfTranscript,
// arriving first means the session was rejected; one that has not started
// within startTimeout has timed out, which a retry may clear.
func (s *stream) awaitStarted(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, startTimeout)
	defer cancel()
	for {
		var m message
		if _, err := s.readContext(ctx, &m); err != nil {
			if ctx.Err() != nil {
				return fmt.Errorf("speechmatics stt: timed out waiting for RecognitionStarted: %w", err)
			}
			return err
		}
		switch m.Message {
		case msgRecognitionStarted:
			s.sessionID = m.ID
			return nil
		case msgError:
			return &startRejectedError{status: m.describeStatus()}
		case msgEndOfTranscript:
			return &startRejectedError{status: "Session ended before RecognitionStarted"}
		}
	}
}

// Send writes a chunk of PCM as a binary frame (Speechmatics' AddAudio).
func (s *stream) Send(audio []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if err := s.conn.Write(s.ctx, websocket.MessageBinary, audio); err != nil {
		return err
	}
	s.seqNo.Add(1)
	s.audioBytes.Add(int64(len(audio)))
	return nil
}

// writeJSON sends a message on the session.
func (s *stream) writeJSON(msg map[string]any) error {
	b, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageText, b)
}

// Recv reads the next results. Each partial segment is an interim transcript
// and each final segment a final one, carrying the speaker diarization
// attributed it to. When Speechmatics closes turns, its StartOfTurn and
// EndOfTurn are reported as speech boundaries. Once EndOfStream has been sent
// the session is only read to its end: the pipeline is stopping, and what
// arrives then is not reported.
func (s *stream) Recv() ([]stt.Result, error) {
	for {
		results, err := s.next()
		if s.ending.Load() {
			if err != nil {
				return nil, err
			}
			continue
		}
		if err != nil || len(results) > 0 {
			return results, err
		}
	}
}

// next reads messages until one yields results or ends the session.
func (s *stream) next() ([]stt.Result, error) {
	for {
		var m message
		data, err := s.read(&m)
		if err != nil {
			return nil, err
		}
		if results, err := s.handle(&m, data); err != nil || len(results) > 0 {
			return results, err
		}
	}
}

// handle reads one message into the results it yields, or the error that ends
// the session on it.
func (s *stream) handle(m *message, data []byte) ([]stt.Result, error) {
	switch m.Message {
	case msgAddPartialSegment:
		if t := m.Segment.Transcript; t != "" {
			return []stt.Result{{Text: t, Speaker: m.Segment.Speaker}}, nil
		}
	case msgAddSegment:
		if t := m.Segment.Transcript; t != "" {
			// A final segment answers a finalize when one was asked for,
			// which marks it as the transcript that closes the utterance.
			return []stt.Result{{Text: t, Final: true, Speaker: m.Segment.Speaker, FromFinalize: true}}, nil
		}
	case msgStartOfTurn, msgEndOfTurn:
		return s.turnEvent(m.Message), nil
	case msgSpeakersResult:
		s.reportSpeakers(data)
	case msgWarning:
		// The session continues, possibly with adjusted config.
		slog.Warn("speechmatics stt warning", "status", m.describeStatus())
	case msgEndOfTranscript:
		return nil, io.EOF
	case msgError:
		// An Error always ends the session. It is reported here, where its
		// type says whether a new session could succeed, and the session
		// ending on it is what has the read loop reopen one.
		err := &serverError{kind: m.Type, status: m.describeStatus()}
		if s.report != nil && !s.ending.Load() {
			s.report(s.ctx, err)
		}
		return nil, err
	default:
		slog.Debug("speechmatics stt: message", "message", m.Message)
	}
	return nil, nil
}

// turnEvent is the speech boundary a turn event reports, when Speechmatics
// closes turns in this session. Otherwise the pipeline does, and the event
// yields nothing.
func (s *stream) turnEvent(kind string) []stt.Result {
	if !s.closesTurns {
		return nil
	}
	slog.Debug("speechmatics stt: turn event received", "message", kind)
	if kind == msgStartOfTurn {
		return []stt.Result{{Speech: stt.SpeechStarted}}
	}
	return []stt.Result{{Speech: stt.SpeechStopped}}
}

// reportSpeakers reports a speakers result, raw, when this session tells
// speakers apart.
func (s *stream) reportSpeakers(data []byte) {
	if !s.diarization || s.speakers == nil || s.ending.Load() {
		return
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return
	}
	s.speakers(s.ctx, raw)
}

func (s *stream) read(m *message) ([]byte, error) { return s.readContext(s.ctx, m) }

func (s *stream) readContext(ctx context.Context, m *message) ([]byte, error) {
	_, data, err := s.conn.Read(ctx)
	if err != nil {
		return nil, err
	}
	return data, json.Unmarshal(data, m)
}

// Drain ends the session, implementing stt.Drainer: it sends EndOfStream with
// the number of audio chunks sent, while the socket is still open, and asks the
// service to wait up to endOfStreamTimeout for the EndOfTranscript that answers
// it. That ends the read, which ends the wait early. The service drains a
// session it ends deliberately, as the pipeline stops or is canceled.
func (s *stream) Drain() (time.Duration, error) {
	if err := s.endOfStream(); err != nil {
		return 0, err
	}
	return endOfStreamTimeout, nil
}

// Close closes the socket. A session the service did not drain first, one
// closed to replace it after it dropped or failed, is still sent EndOfStream
// first, as every session is ended, but there is no read left to wait on its
// answer with. The send is best effort: the socket is closed either way.
func (s *stream) Close() error {
	if s.closed != nil {
		s.closed(s)
	}
	if !s.ending.Load() {
		if err := s.endOfStream(); err != nil {
			slog.Debug("speechmatics stt: EndOfStream send failed", "err", err)
		}
	}
	return s.conn.Close(websocket.StatusNormalClosure, "")
}

// endOfStream sends EndOfStream with the number of audio chunks sent, once.
func (s *stream) endOfStream() error {
	end, _ := json.Marshal(map[string]any{ //nolint:errchkjson // known-serializable values
		fieldMessage:  "EndOfStream",
		"last_seq_no": s.seqNo.Load(),
	})
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if s.ending.Swap(true) {
		return nil
	}
	return s.conn.Write(s.ctx, websocket.MessageText, end)
}

// message is the subset of the Agent STT messages we use.
type message struct {
	Message string `json:"message"`
	// ID is the session id RecognitionStarted carries.
	ID     string `json:"id"`
	Type   string `json:"type"`
	Code   any    `json:"code"`
	Reason string `json:"reason"`
	// Segment is the segment AddSegment and AddPartialSegment carry.
	Segment struct {
		Transcript string `json:"transcript"`
		Speaker    string `json:"speaker"`
	} `json:"segment"`
}

// describeStatus reads a status message as its type, code and reason, any of
// which may be absent.
func (m *message) describeStatus() string {
	var parts []string
	if m.Type != "" {
		parts = append(parts, m.Type)
	}
	if m.Code != nil {
		parts = append(parts, fmt.Sprint(m.Code))
	}
	if m.Reason != "" {
		parts = append(parts, m.Reason)
	}
	if len(parts) == 0 {
		return m.Message
	}
	return strings.Join(parts, " ")
}
