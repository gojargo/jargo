package speechmatics

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
)

// NewSTT builds a Speechmatics streaming STT service.
func NewSTT(cfg Config) *stt.StreamService {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.Language == "" {
		cfg.Language = language.EnglishUS
	}
	if cfg.OperatingPoint == "" {
		cfg.OperatingPoint = defaultOperatingPoint
	}
	if cfg.MaxDelay == 0 {
		cfg.MaxDelay = defaultMaxDelay
	}
	c := &connector{cfg: cfg}
	svc := stt.NewStream("SpeechmaticsSTT", c, cfg.SampleRate)
	c.svc = svc
	return svc
}

type connector struct {
	cfg Config
	// svc is the service this connector drives, for reporting the errors
	// Speechmatics sends during a session.
	svc *stt.StreamService
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

// Metadata reports the transcript latency the turn strategies size their
// wait by.
func (c *connector) Metadata() stt.Metadata {
	return stt.Metadata{TTFSP99: stt.TTFSP99Or(c.cfg.TTFSP99, stt.SpeechmaticsTTFSP99)}
}

// Connect dials the real-time WebSocket, sends StartRecognition, and waits for
// the RecognitionStarted acknowledgement before audio flows.
// WebsocketOptions hands the stream base the connection settings the
// configuration gives, implementing wsservice.Optioner.
func (c *connector) WebsocketOptions() wsservice.Options { return c.cfg.Websocket }

func (c *connector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	header := http.Header{}
	header.Set("Authorization", "Bearer "+c.cfg.APIKey)

	conn, err := c.cfg.Websocket.Dial(ctx, c.cfg.URL, header, readLimit)
	if err != nil {
		return nil, err
	}

	s := &stream{conn: conn, ctx: ctx, report: c.reportServerError}
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
	return s, nil
}

func (c *connector) startRecognition(sampleRate int) []byte {
	tc := map[string]any{
		"language":        speechmaticsLanguage(c.cfg.Language),
		"operating_point": c.cfg.OperatingPoint,
		"enable_partials": c.cfg.EnablePartials == nil || *c.cfg.EnablePartials,
		"max_delay":       c.cfg.MaxDelay,
	}
	eou := defaultEndOfUtteranceSilence
	if c.cfg.EndOfUtteranceSilence != nil {
		eou = *c.cfg.EndOfUtteranceSilence
	}
	if eou > 0 {
		tc["conversation_config"] = map[string]any{"end_of_utterance_silence_trigger": eou}
	}
	msg := map[string]any{
		"message": "StartRecognition",
		"audio_format": map[string]any{
			"type":        "raw",
			"encoding":    "pcm_s16le",
			"sample_rate": sampleRate,
		},
		"transcription_config": tc,
	}
	b, _ := json.Marshal(msg) //nolint:errchkjson // map of known-serializable values
	return b
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

type stream struct {
	conn     *wsutil.Conn
	ctx      context.Context
	writeMu  sync.Mutex
	seqNo    atomic.Uint64
	finalBuf string
	// report puts an error Speechmatics sent during the session on the
	// pipeline, before the session ends on it.
	report func(ctx context.Context, err *serverError)
	// ending is set once EndOfStream has been sent. What the session says
	// after it is read only to see it end, and is not reported.
	ending atomic.Bool
}

// awaitStarted reads until RecognitionStarted. An Error, or an EndOfTranscript,
// arriving first means the session was rejected; one that has not started
// within startTimeout has timed out, which a retry may clear.
func (s *stream) awaitStarted(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, startTimeout)
	defer cancel()
	for {
		var m message
		if err := s.readContext(ctx, &m); err != nil {
			if ctx.Err() != nil {
				return fmt.Errorf("speechmatics stt: timed out waiting for RecognitionStarted: %w", err)
			}
			return err
		}
		switch m.Message {
		case "RecognitionStarted":
			return nil
		case "Error":
			return &startRejectedError{status: m.describeStatus()}
		case "EndOfTranscript":
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
	return nil
}

// Recv reads the next transcript. Partials are interim; finalized segments
// accumulate and are emitted together at end-of-utterance with EndOfTurn set.
// Once EndOfStream has been sent the session is only read to its end: the
// pipeline is stopping, and what arrives then is not reported.
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
		if err := s.read(&m); err != nil {
			return nil, err
		}
		switch m.Message {
		case "AddPartialTranscript":
			if t := m.Metadata.Transcript; t != "" {
				return []stt.Result{{Text: t, Final: false}}, nil
			}
		case "AddTranscript":
			if t := m.Metadata.Transcript; t != "" {
				if s.finalBuf != "" {
					s.finalBuf += " "
				}
				s.finalBuf += t
			}
		case "EndOfUtterance":
			if s.finalBuf != "" {
				text := s.finalBuf
				s.finalBuf = ""
				return []stt.Result{{Text: text, Final: true, EndOfTurn: true}}, nil
			}
		case "EndOfTranscript":
			return nil, io.EOF
		case "Error":
			// An Error always ends the session. It is reported here, where its
			// type says whether a new session could succeed, and the session
			// ending on it is what has the read loop reopen one.
			err := &serverError{kind: m.Type, status: m.describeStatus()}
			if s.report != nil && !s.ending.Load() {
				s.report(s.ctx, err)
			}
			return nil, err
		}
	}
}

func (s *stream) read(m *message) error { return s.readContext(s.ctx, m) }

func (s *stream) readContext(ctx context.Context, m *message) error {
	_, data, err := s.conn.Read(ctx)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, m)
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
		"message":     "EndOfStream",
		"last_seq_no": s.seqNo.Load(),
	})
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if s.ending.Swap(true) {
		return nil
	}
	return s.conn.Write(s.ctx, websocket.MessageText, end)
}

// message is the subset of Speechmatics' real-time messages we use.
type message struct {
	Message  string `json:"message"`
	Type     string `json:"type"`
	Code     any    `json:"code"`
	Reason   string `json:"reason"`
	Metadata struct {
		Transcript string `json:"transcript"`
	} `json:"metadata"`
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
