package speechmatics

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"sync/atomic"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/stt"
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
	return stt.NewStream("SpeechmaticsSTT", &connector{cfg: cfg}, cfg.SampleRate)
}

type connector struct {
	cfg Config
}

// Metadata reports the transcript latency the turn strategies size their
// wait by.
func (c *connector) Metadata() stt.Metadata {
	return stt.Metadata{TTFSP99: cmp.Or(c.cfg.TTFSP99, stt.SpeechmaticsTTFSP99)}
}

// Connect dials the real-time WebSocket, sends StartRecognition, and waits for
// the RecognitionStarted acknowledgement before audio flows.
func (c *connector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	header := http.Header{}
	header.Set("Authorization", "Bearer "+c.cfg.APIKey)

	conn, err := wsutil.Dial(ctx, c.cfg.URL, header, readLimit)
	if err != nil {
		return nil, err
	}

	s := &stream{conn: conn, ctx: ctx}
	if err := conn.Write(ctx, websocket.MessageText, c.startRecognition(sampleRate)); err != nil {
		_ = conn.Close(websocket.StatusInternalError, "start failed")
		return nil, err
	}
	if err := s.awaitStarted(); err != nil {
		_ = conn.Close(websocket.StatusInternalError, "start failed")
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
}

// awaitStarted reads until RecognitionStarted, failing on an error message.
func (s *stream) awaitStarted() error {
	for {
		var m message
		if err := s.read(&m); err != nil {
			return err
		}
		switch m.Message {
		case "RecognitionStarted":
			return nil
		case "Error":
			return fmt.Errorf("%w: %s", errServer, m.Reason)
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
func (s *stream) Recv() ([]stt.Result, error) {
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
			return nil, fmt.Errorf("%w: %s", errServer, m.Reason)
		}
	}
}

func (s *stream) read(m *message) error {
	_, data, err := s.conn.Read(s.ctx)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, m)
}

// Close sends EndOfStream and closes the socket.
func (s *stream) Close() error {
	end, _ := json.Marshal(map[string]any{ //nolint:errchkjson // known-serializable values
		"message":     "EndOfStream",
		"last_seq_no": s.seqNo.Load(),
	})
	s.writeMu.Lock()
	_ = s.conn.Write(context.Background(), websocket.MessageText, end)
	s.writeMu.Unlock()
	return s.conn.Close(websocket.StatusNormalClosure, "")
}

// message is the subset of Speechmatics' real-time messages we use.
type message struct {
	Message  string `json:"message"`
	Reason   string `json:"reason"`
	Metadata struct {
		Transcript string `json:"transcript"`
	} `json:"metadata"`
}
