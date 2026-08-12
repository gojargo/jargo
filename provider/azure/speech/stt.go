package speech

import (
	"cmp"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsutil"
	"github.com/google/uuid"
)

const (
	// sttPath is the continuous-recognition endpoint. Azure calls this
	// "conversation" mode: it recognizes until the client stops sending audio,
	// rather than stopping after the first utterance.
	sttPath = "/speech/recognition/conversation/cognitiveservices/v1"
	// defaultSTTLocale is the recognition locale used when none is configured.
	defaultSTTLocale = "en-US"
	// sttReadLimit bounds a single inbound message.
	sttReadLimit = 1 << 20
	// wavHeaderLen is the size of the RIFF header that opens each turn.
	wavHeaderLen = 44
	// pcmFormatTag marks uncompressed PCM in a WAV header.
	pcmFormatTag = 1
	// bitsPerSample is the pipeline's sample width.
	bitsPerSample = 16
)

// Message paths in Azure's speech protocol.
const (
	// pathSpeechConfig opens the session and describes the client.
	pathSpeechConfig = "speech.config"
	// pathSpeechContext configures the recognition about to start.
	pathSpeechContext = "speech.context"
	// pathAudio carries the WAV header and the PCM that follows it.
	pathAudio = "audio"
	// pathHypothesis is an interim transcript, revised as more audio arrives.
	pathHypothesis = "speech.hypothesis"
	// pathPhrase is a finalized transcript for one utterance.
	pathPhrase = "speech.phrase"
	// pathTurnEnd closes a turn. In continuous recognition the next turn has to
	// be opened by the client.
	pathTurnEnd = "turn.end"
)

// recognitionSuccess is the status on a phrase that produced a transcript.
// Anything else (NoMatch, InitialSilenceTimeout, BabbleTimeout, EndOfDictation)
// carries no usable text.
const recognitionSuccess = "Success"

// errSTTProtocol is returned when Azure sends a message the service cannot make
// sense of.
//
//nolint:gochecknoglobals // sentinel error
var errSTTProtocol = errors.New("azurespeech: stt protocol error")

// STTConfig configures the Azure AI Speech streaming STT service.
type STTConfig struct {
	// APIKey is the Speech resource key, sent as Ocp-Apim-Subscription-Key.
	// Required.
	APIKey string `validate:"required"`
	// Region is the Speech resource region, e.g. "eastus" or "francecentral".
	// Required unless Host is set.
	Region string `validate:"required_without=Host"`
	// Host overrides the full recognition host, for a private endpoint,
	// sovereign cloud or custom domain, e.g.
	// wss://my-resource.stt.speech.azure.us. Empty derives it from Region.
	Host string
	// Language is the recognition locale; the zero value uses en-US. Azure wants
	// a full locale ("fr-FR"), so a base language is expanded to its default
	// region.
	Language language.Language
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
	// Profanity is how Azure treats profanities in the transcript: "raw" leaves
	// them, "masked" replaces them with asterisks, "removed" drops them. Empty
	// uses Azure's default of masked. Non-English deployments often want "raw":
	// the profanity list is over-eager and masks ordinary words, which breaks
	// downstream matching.
	Profanity string `validate:"omitempty,oneof=raw masked removed"`
	// EndpointID selects a custom speech model by its deployment id; empty uses
	// the base model. Azure ignores the language when a custom model is chosen.
	EndpointID string

	// TTFSP99 overrides the measured transcript latency the turn strategies
	// size their wait by; 0 uses stt.AzureTTFSP99.
	TTFSP99 time.Duration
}

// Validate reports whether the configuration is usable.
func (c STTConfig) Validate() error { return validate.Struct(c) }

// NewSTT builds an Azure AI Speech streaming speech-to-text service. Azure
// detects utterance boundaries itself, so a finalized phrase marks the end of
// the user's turn.
func NewSTT(cfg STTConfig) *stt.StreamService {
	return stt.NewStream("AzureSTT", &sttConnector{cfg: cfg}, cfg.SampleRate)
}

type sttConnector struct {
	cfg STTConfig
}

// Metadata reports Azure's time-to-final-segment latency to downstream
// processors.
func (c *sttConnector) Metadata() stt.Metadata {
	return stt.Metadata{TTFSP99: cmp.Or(c.cfg.TTFSP99, stt.AzureTTFSP99)}
}

// azureBaseLocales expands the common base language codes to the locale Azure
// expects. An already-qualified or unmapped code passes through unchanged.
//
//nolint:gochecknoglobals // static lookup table
var azureBaseLocales = map[language.Language]string{
	language.English:    "en-US",
	language.French:     "fr-FR",
	language.German:     "de-DE",
	language.Spanish:    "es-ES",
	language.Italian:    "it-IT",
	language.Portuguese: "pt-PT",
	language.Dutch:      "nl-NL",
	language.Russian:    "ru-RU",
	language.Chinese:    "zh-CN",
	language.Japanese:   "ja-JP",
	language.Korean:     "ko-KR",
	language.Arabic:     "ar-EG",
	language.Hindi:      "hi-IN",
	language.Polish:     "pl-PL",
	language.Turkish:    "tr-TR",
	language.Swedish:    "sv-SE",
	language.Danish:     "da-DK",
	language.Norwegian:  "nb-NO",
	language.Finnish:    "fi-FI",
	language.Greek:      "el-GR",
	language.Czech:      "cs-CZ",
	language.Hebrew:     "he-IL",
	language.Thai:       "th-TH",
	language.Vietnamese: "vi-VN",
	language.Indonesian: "id-ID",
	language.Hungarian:  "hu-HU",
	language.Romanian:   "ro-RO",
	language.Ukrainian:  "uk-UA",
	language.Bulgarian:  "bg-BG",
	language.Croatian:   "hr-HR",
	language.Slovak:     "sk-SK",
	language.Malay:      "ms-MY",
	language.Filipino:   "fil-PH",
	language.Tamil:      "ta-IN",
}

// azureLocale maps a Language to an Azure recognition locale.
func azureLocale(l language.Language) string {
	if l == "" {
		return defaultSTTLocale
	}
	if locale, ok := azureBaseLocales[l]; ok {
		return locale
	}
	return l.Code()
}

// endpoint builds the recognition URL. The recognition parameters travel as
// query parameters rather than in a configuration message.
func (c *sttConnector) endpoint() string {
	host := c.cfg.Host
	if host == "" {
		host = "wss://" + c.cfg.Region + ".stt.speech.microsoft.com"
	}

	q := url.Values{}
	// A custom model is selected by deployment id, and Azure derives the
	// language from the model rather than the request.
	if c.cfg.EndpointID != "" {
		q.Set("cid", c.cfg.EndpointID)
	} else {
		q.Set("language", azureLocale(c.cfg.Language))
	}
	// The simple format returns the display text alone, which is all the
	// pipeline consumes.
	q.Set("format", "simple")
	if c.cfg.Profanity != "" {
		q.Set("profanity", strings.ToLower(c.cfg.Profanity))
	}
	return strings.TrimSuffix(host, "/") + sttPath + "?" + q.Encode()
}

// Connect opens a recognition session and starts its first turn.
func (c *sttConnector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	header := http.Header{}
	header.Set("Ocp-Apim-Subscription-Key", c.cfg.APIKey)

	conn, err := wsutil.Dial(ctx, c.endpoint(), header, sttReadLimit)
	if err != nil {
		return nil, err
	}

	s := &sttStream{
		conn:       conn,
		ctx:        ctx,
		sampleRate: sampleRate,
		lang:       azureLocale(c.cfg.Language),
		requestID:  newRequestID(),
	}
	if err := s.openSession(); err != nil {
		_ = conn.Close(websocket.StatusInternalError, "session setup failed")
		return nil, err
	}
	return s, nil
}

// newRequestID is the correlation id for one turn: a UUID with the dashes
// removed, which is the form Azure's X-RequestId header takes.
func newRequestID() string {
	return strings.ReplaceAll(uuid.NewString(), "-", "")
}

type sttStream struct {
	conn       *wsutil.Conn
	ctx        context.Context
	sampleRate int
	// lang is the configured locale echoed on results.
	lang string

	writeMu sync.Mutex

	// idMu guards requestID, which correlates every message of the current turn.
	// Azure ends a turn after each utterance and the next runs under a fresh id,
	// so the send side reads it while the receive side rewrites it. It is
	// deliberately not writeMu: rendering a message's headers reads the id
	// before taking the write lock.
	idMu      sync.RWMutex
	requestID string
}

// openSession sends the client description, then opens the first turn.
func (s *sttStream) openSession() error {
	if err := s.sendText(pathSpeechConfig, speechConfigBody()); err != nil {
		return err
	}
	return s.startTurn()
}

// startTurn opens a turn: it configures the recognition and describes the audio
// that follows with a WAV header. Azure expects both at the start of every turn,
// not just the first.
func (s *sttStream) startTurn() error {
	if err := s.sendText(pathSpeechContext, "{}"); err != nil {
		return err
	}
	return s.sendBinary(pathAudio, "audio/x-wav", wavHeader(s.sampleRate))
}

// speechConfigBody describes the client to the service. Azure records it for
// diagnostics; the recognition itself is configured by the URL.
func speechConfigBody() string {
	body := map[string]any{
		"context": map[string]any{
			"system": map[string]any{"name": "jargo", "version": "1", "build": "go", "lang": "Go"},
			"os":     map[string]any{"platform": "Go", "name": "Go", "version": "1"},
		},
	}
	b, _ := json.Marshal(body) //nolint:errchkjson // map of known-serializable values
	return string(b)
}

// wavHeader is the 44-byte RIFF header that opens a turn's audio. The two length
// fields are zero because the stream's length is not known up front.
func wavHeader(sampleRate int) []byte {
	const channels = 1
	blockAlign := channels * bitsPerSample / 8
	byteRate := sampleRate * blockAlign

	h := make([]byte, wavHeaderLen)
	copy(h[0:], "RIFF")
	binary.LittleEndian.PutUint32(h[4:], 0) // streaming: length unknown
	copy(h[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(h[16:], 16) // fmt chunk size
	binary.LittleEndian.PutUint16(h[20:], pcmFormatTag)
	binary.LittleEndian.PutUint16(h[22:], channels)
	binary.LittleEndian.PutUint32(h[24:], uint32(sampleRate)) //nolint:gosec // a sample rate is small and positive
	binary.LittleEndian.PutUint32(h[28:], uint32(byteRate))   //nolint:gosec // derived from the sample rate
	binary.LittleEndian.PutUint16(h[32:], uint16(blockAlign))
	binary.LittleEndian.PutUint16(h[34:], bitsPerSample)
	copy(h[36:], "data")
	binary.LittleEndian.PutUint32(h[40:], 0) // streaming: length unknown
	return h
}

// headers renders the message headers Azure's framing puts ahead of every body.
func (s *sttStream) headers(path, contentType string) string {
	var b strings.Builder
	b.WriteString("Path: ")
	b.WriteString(path)
	b.WriteString("\r\nX-RequestId: ")
	b.WriteString(s.currentRequestID())
	b.WriteString("\r\nX-Timestamp: ")
	b.WriteString(time.Now().UTC().Format("2006-01-02T15:04:05.000Z"))
	b.WriteString("\r\n")
	if contentType != "" {
		b.WriteString("Content-Type: ")
		b.WriteString(contentType)
		b.WriteString("\r\n")
	}
	return b.String()
}

// sendText writes a text message: the headers, a blank line, then the body.
func (s *sttStream) sendText(path, body string) error {
	payload := s.headers(path, "application/json") + "\r\n" + body
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageText, []byte(payload))
}

// sendBinary writes a binary message: a two-byte big-endian header length, the
// headers, then the payload.
func (s *sttStream) sendBinary(path, contentType string, payload []byte) error {
	headers := []byte(s.headers(path, contentType))
	frame := make([]byte, 2+len(headers)+len(payload))
	binary.BigEndian.PutUint16(frame, uint16(len(headers))) //nolint:gosec // headers are a few dozen bytes
	copy(frame[2:], headers)
	copy(frame[2+len(headers):], payload)

	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageBinary, frame)
}

// currentRequestID reports the turn the next message belongs to.
func (s *sttStream) currentRequestID() string {
	s.idMu.RLock()
	defer s.idMu.RUnlock()
	return s.requestID
}

// Send writes a chunk of PCM audio for the current turn.
func (s *sttStream) Send(audio []byte) error {
	return s.sendBinary(pathAudio, "", audio)
}

// sttMessage is a parsed protocol message: the path that names it and the body
// it carries.
type sttMessage struct {
	path string
	body []byte
}

// parseMessage splits Azure's framing into a path and a body. A text message is
// headers, a blank line, then the body; a binary one is a two-byte big-endian
// header length, the headers, then the body.
func parseMessage(typ websocket.MessageType, data []byte) (sttMessage, error) {
	var headers, body []byte
	switch typ {
	case websocket.MessageText:
		head, rest, found := strings.Cut(string(data), "\r\n\r\n")
		if !found {
			return sttMessage{}, fmt.Errorf("%w: text message has no header separator", errSTTProtocol)
		}
		headers, body = []byte(head), []byte(rest)
	case websocket.MessageBinary:
		if len(data) < 2 {
			return sttMessage{}, fmt.Errorf("%w: binary message is too short for a header length", errSTTProtocol)
		}
		n := int(binary.BigEndian.Uint16(data))
		if len(data) < 2+n {
			return sttMessage{}, fmt.Errorf("%w: binary message is shorter than its header length", errSTTProtocol)
		}
		headers, body = data[2:2+n], data[2+n:]
	}
	return sttMessage{path: headerValue(headers, "path"), body: body}, nil
}

// headerValue reads one header, matching the name case-insensitively.
func headerValue(headers []byte, name string) string {
	for line := range strings.SplitSeq(string(headers), "\r\n") {
		key, value, found := strings.Cut(line, ":")
		if found && strings.EqualFold(strings.TrimSpace(key), name) {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

// sttHypothesis is an interim transcript.
type sttHypothesis struct {
	Text string `json:"Text"` //nolint:tagliatelle // Azure wire field
}

// sttPhrase is a finalized transcript. A status other than Success means the
// segment produced no usable text.
type sttPhrase struct {
	RecognitionStatus string `json:"RecognitionStatus"` //nolint:tagliatelle // Azure wire field
	DisplayText       string `json:"DisplayText"`       //nolint:tagliatelle // Azure wire field
}

// Recv reads the next transcript. Azure ends a turn after each utterance, so a
// turn.end silently opens the next one and the loop continues.
func (s *sttStream) Recv() ([]stt.Result, error) {
	for {
		typ, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, err
		}
		msg, err := parseMessage(typ, data)
		if err != nil {
			return nil, err
		}
		switch msg.path {
		case pathHypothesis:
			var h sttHypothesis
			if json.Unmarshal(msg.body, &h) != nil || h.Text == "" {
				continue
			}
			return []stt.Result{{Text: h.Text, Final: false, Language: s.lang}}, nil
		case pathPhrase:
			var p sttPhrase
			if json.Unmarshal(msg.body, &p) != nil {
				continue
			}
			if p.RecognitionStatus != recognitionSuccess || p.DisplayText == "" {
				continue
			}
			return []stt.Result{{Text: p.DisplayText, Final: true, EndOfTurn: true, Language: s.lang}}, nil
		case pathTurnEnd:
			// Continuous recognition is a sequence of turns, each opened by the
			// client. Without this the session goes quiet after one utterance.
			if err := s.nextTurn(); err != nil {
				return nil, err
			}
		}
	}
}

// nextTurn rolls the request id forward and opens the following turn.
func (s *sttStream) nextTurn() error {
	s.idMu.Lock()
	s.requestID = newRequestID()
	s.idMu.Unlock()
	return s.startTurn()
}

// Close tells Azure the audio is complete, then tears the session down. An
// empty audio message is the end-of-stream signal.
func (s *sttStream) Close() error {
	_ = s.sendBinary(pathAudio, "", nil)
	return s.conn.Close(websocket.StatusNormalClosure, "")
}
