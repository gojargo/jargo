package live

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"sync"
	"sync/atomic"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/wsutil"
)

// Service is the Gemini Live speech-to-speech processor.
type Service struct {
	*processor.Base
	cfg Config

	mu        sync.Mutex
	conn      *wsutil.Conn
	connCtx   context.Context
	cancel    context.CancelFunc
	writeMu   sync.Mutex
	wg        sync.WaitGroup
	ready     atomic.Bool
	speaking  bool
	connector Connector
}

// Connector customizes how the Live session is addressed and authorized, so a
// deployment with a different endpoint or auth scheme (Vertex AI, which
// addresses models per project and location and authorizes with an OAuth token)
// can reuse this implementation.
type Connector interface {
	// Endpoint returns the WebSocket URL to dial and the headers to dial it
	// with. It takes a context because a scheme may have to mint or refresh a
	// token first.
	Endpoint(ctx context.Context) (string, http.Header, error)
	// ModelPath returns the resource name the setup message identifies the model
	// by.
	ModelPath(model string) string
}

// apiKeyConnector is the standard Gemini Live addressing: the api key travels as
// a query parameter and the model is named relative to the API.
type apiKeyConnector struct {
	baseURL string
	apiKey  string
}

func (c apiKeyConnector) Endpoint(context.Context) (string, http.Header, error) {
	return c.baseURL + "?key=" + url.QueryEscape(c.apiKey), nil, nil
}

func (apiKeyConnector) ModelPath(model string) string { return "models/" + model }

// New builds a Gemini Live service.
func New(cfg Config) *Service {
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	return NewWithConnector("GeminiLive", apiKeyConnector{baseURL: cfg.BaseURL, apiKey: cfg.APIKey}, cfg)
}

// NewWithConnector builds a Gemini Live service that dials through conn. It is
// the base for deployments that do not use the Gemini API's own endpoint or
// api-key auth; name is the processor label.
func NewWithConnector(name string, conn Connector, cfg Config) *Service {
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.Voice == "" {
		cfg.Voice = defaultVoice
	}
	s := &Service{cfg: cfg, connector: conn}
	s.Base = processor.New(name, s)
	return s
}

// ProcessFrame opens the session on StartFrame, forwards input audio to the
// model, and tears the session down when the pipeline ends.
func (s *Service) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.StartFrame:
		if err := s.connect(ctx); err != nil {
			s.PushError(ctx, "gemini live connect failed", err, true)
		}
		return s.PushFrame(ctx, f, dir)
	case *frames.InputAudioRawFrame:
		if dir == processor.Downstream {
			s.sendAudio(fr.Audio, fr.SampleRate)
			return nil // the model consumes the audio; it does not flow on
		}
		return s.PushFrame(ctx, f, dir)
	case *frames.EndFrame, *frames.CancelFrame:
		s.disconnect()
		return s.PushFrame(ctx, f, dir)
	default:
		return s.PushFrame(ctx, f, dir)
	}
}

// Cleanup tears down the session and stops the read loop.
func (s *Service) Cleanup(ctx context.Context) error {
	s.disconnect()
	return s.Base.Cleanup(ctx)
}

// connect dials the Live WebSocket, sends the setup message, and starts the read
// loop.
func (s *Service) connect(ctx context.Context) error {
	endpoint, header, err := s.connector.Endpoint(ctx)
	if err != nil {
		return err
	}
	conn, err := wsutil.Dial(ctx, endpoint, header, readLimit)
	if err != nil {
		return err
	}

	connCtx, cancel := context.WithCancel(context.Background())
	s.mu.Lock()
	s.conn = conn
	s.connCtx = connCtx
	s.cancel = cancel
	s.mu.Unlock()

	if err := s.send(s.setup()); err != nil {
		cancel()
		_ = conn.Close(websocket.StatusInternalError, "setup failed")
		return err
	}

	s.wg.Add(1)
	go s.readLoop(conn, connCtx)
	return nil
}

// setup is the initial session-configuration message.
func (s *Service) setup() map[string]any {
	setup := map[string]any{
		"model": s.connector.ModelPath(s.cfg.Model),
		"generationConfig": map[string]any{
			"responseModalities": []string{modalityAudio},
			"speechConfig": map[string]any{
				"voiceConfig": map[string]any{
					"prebuiltVoiceConfig": map[string]any{"voiceName": s.cfg.Voice},
				},
			},
		},
		"inputAudioTranscription":  map[string]any{},
		"outputAudioTranscription": map[string]any{},
	}
	if s.cfg.Instructions != "" {
		setup["systemInstruction"] = map[string]any{
			"parts": []any{map[string]any{"text": s.cfg.Instructions}},
		}
	}
	return map[string]any{"setup": setup}
}

// sendAudio streams a chunk of input PCM to the model once the session is ready.
func (s *Service) sendAudio(pcm []byte, sampleRate int) {
	if len(pcm) == 0 || !s.ready.Load() {
		return
	}
	_ = s.send(map[string]any{
		"realtimeInput": map[string]any{
			"audio": map[string]any{
				"data":     base64.StdEncoding.EncodeToString(pcm),
				"mimeType": fmt.Sprintf("audio/pcm;rate=%d", sampleRate),
			},
		},
	})
}

// send marshals v and writes it as a text frame, serializing concurrent writes.
func (s *Service) send(v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	s.mu.Lock()
	conn, connCtx := s.conn, s.connCtx
	s.mu.Unlock()
	if conn == nil {
		return nil
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return conn.Write(connCtx, websocket.MessageText, data)
}

// disconnect cancels the session, closes the socket, and waits for the read loop.
func (s *Service) disconnect() {
	s.mu.Lock()
	conn, cancel := s.conn, s.cancel
	s.conn, s.cancel, s.connCtx = nil, nil, nil
	s.mu.Unlock()
	s.ready.Store(false)
	if cancel != nil {
		cancel()
	}
	if conn != nil {
		_ = conn.Close(websocket.StatusNormalClosure, "")
	}
	s.wg.Wait()
}

// serverMessage is the subset of Live API server messages the service handles.
// The JSON field names below are Gemini's wire protocol (camelCase), so the
// snake_case house style does not apply.

type serverMessage struct {
	SetupComplete *json.RawMessage `json:"setupComplete"` //nolint:tagliatelle // Gemini wire field
	ServerContent *serverContent   `json:"serverContent"` //nolint:tagliatelle // Gemini wire field
	UsageMetadata *usageMetadata   `json:"usageMetadata"` //nolint:tagliatelle // Gemini wire field
}

// usageMetadata is the per-turn token accounting the Live API sends alongside a
// completed turn. The *Details lists break the prompt and response token counts
// down by modality (TEXT vs AUDIO), which is how a native-audio model exposes
// how many of its billed tokens were speech.
type usageMetadata struct {
	PromptTokenCount        int64                `json:"promptTokenCount"`        //nolint:tagliatelle // Gemini wire field
	ResponseTokenCount      int64                `json:"responseTokenCount"`      //nolint:tagliatelle // Gemini wire field
	TotalTokenCount         int64                `json:"totalTokenCount"`         //nolint:tagliatelle // Gemini wire field
	CachedContentTokenCount int64                `json:"cachedContentTokenCount"` //nolint:tagliatelle // Gemini wire field
	PromptTokensDetails     []modalityTokenCount `json:"promptTokensDetails"`     //nolint:tagliatelle // Gemini wire field
	ResponseTokensDetails   []modalityTokenCount `json:"responseTokensDetails"`   //nolint:tagliatelle // Gemini wire field
}

// modalityTokenCount is a token count attributed to one modality (e.g. TEXT or
// AUDIO) within a prompt or response.
type modalityTokenCount struct {
	Modality   string `json:"modality"`
	TokenCount int64  `json:"tokenCount"` //nolint:tagliatelle // Gemini wire field
}

// tokenUsage converts the wire accounting into the framework's usage shape,
// folding the per-modality detail lists into the audio and text breakdowns.
func (u usageMetadata) tokenUsage() frames.LLMTokenUsage {
	usage := frames.LLMTokenUsage{
		PromptTokens:     u.PromptTokenCount,
		CompletionTokens: u.ResponseTokenCount,
		TotalTokens:      u.TotalTokenCount,
		CacheReadTokens:  u.CachedContentTokenCount,
	}
	if usage.TotalTokens == 0 {
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
	}
	for _, d := range u.PromptTokensDetails {
		switch d.Modality {
		case modalityAudio:
			usage.InputAudioTokens += d.TokenCount
		case modalityText:
			usage.InputTextTokens += d.TokenCount
		}
	}
	for _, d := range u.ResponseTokensDetails {
		switch d.Modality {
		case modalityAudio:
			usage.OutputAudioTokens += d.TokenCount
		case modalityText:
			usage.OutputTextTokens += d.TokenCount
		}
	}
	return usage
}

type serverContent struct {
	ModelTurn *struct {
		Parts []part `json:"parts"`
	} `json:"modelTurn"` //nolint:tagliatelle // Gemini wire field
	InputTranscription  *textPayload `json:"inputTranscription"`  //nolint:tagliatelle // Gemini wire field
	OutputTranscription *textPayload `json:"outputTranscription"` //nolint:tagliatelle // Gemini wire field
	Interrupted         bool         `json:"interrupted"`
	GenerationComplete  bool         `json:"generationComplete"` //nolint:tagliatelle // Gemini wire field
}

type part struct {
	Text       string `json:"text"`
	InlineData *struct {
		MimeType string `json:"mimeType"` //nolint:tagliatelle // Gemini wire field
		Data     string `json:"data"`
	} `json:"inlineData"` //nolint:tagliatelle // Gemini wire field
}

type textPayload struct {
	Text string `json:"text"`
}

// readLoop reads server messages until the connection is closed or canceled.
func (s *Service) readLoop(conn *wsutil.Conn, connCtx context.Context) {
	defer s.wg.Done()
	for {
		_, data, err := conn.Read(connCtx)
		if err != nil {
			if connCtx.Err() == nil {
				slog.Debug("gemini live read ended", "err", err)
			}
			return
		}
		var msg serverMessage
		if json.Unmarshal(data, &msg) != nil {
			continue
		}
		s.handle(connCtx, msg)
	}
}

// handle maps a server message onto downstream pipeline frames.
func (s *Service) handle(ctx context.Context, msg serverMessage) {
	if msg.SetupComplete != nil {
		s.ready.Store(true)
	}
	if msg.UsageMetadata != nil && s.UsageMetricsEnabled() {
		_ = s.PushTokenUsage(ctx, s.cfg.Model, msg.UsageMetadata.tokenUsage())
	}
	sc := msg.ServerContent
	if sc == nil {
		return
	}
	if sc.Interrupted {
		s.setSpeaking(ctx, false)
		_ = s.PushFrame(ctx, frames.NewUserStartedSpeakingFrame(), processor.Downstream)
		_ = s.PushFrame(ctx, frames.NewInterruptionFrame(), processor.Downstream)
	}
	if sc.InputTranscription != nil && sc.InputTranscription.Text != "" {
		_ = s.PushFrame(ctx, frames.NewTranscriptionFrame(sc.InputTranscription.Text, "", ""), processor.Downstream)
	}
	if sc.OutputTranscription != nil && sc.OutputTranscription.Text != "" {
		_ = s.PushFrame(ctx, frames.NewLLMTextFrame(sc.OutputTranscription.Text), processor.Downstream)
	}
	if sc.ModelTurn != nil {
		for _, p := range sc.ModelTurn.Parts {
			s.handlePart(ctx, p)
		}
	}
	if sc.GenerationComplete {
		s.setSpeaking(ctx, false)
	}
}

// handlePart emits the audio and any text carried by one model-turn part.
func (s *Service) handlePart(ctx context.Context, p part) {
	if p.Text != "" {
		_ = s.PushFrame(ctx, frames.NewLLMTextFrame(p.Text), processor.Downstream)
	}
	if p.InlineData == nil {
		return
	}
	pcm, err := base64.StdEncoding.DecodeString(p.InlineData.Data)
	if err != nil || len(pcm) == 0 {
		return
	}
	s.setSpeaking(ctx, true)
	_ = s.PushFrame(ctx, frames.NewTTSAudioRawFrame(pcm, outputSampleRate, 1), processor.Downstream)
}

// setSpeaking emits a bot-speaking transition frame on a change of state.
func (s *Service) setSpeaking(ctx context.Context, speaking bool) {
	s.mu.Lock()
	changed := s.speaking != speaking
	s.speaking = speaking
	s.mu.Unlock()
	if !changed {
		return
	}
	if speaking {
		_ = s.PushFrame(ctx, frames.NewBotStartedSpeakingFrame(), processor.Downstream)
	} else {
		_ = s.PushFrame(ctx, frames.NewBotStoppedSpeakingFrame(), processor.Downstream)
	}
}

// CanGenerateMetrics reports that this service times the conversation and reports
// the result, so the pipeline counts it when it collects the processors that
// report metrics.
func (s *Service) CanGenerateMetrics() bool { return true }
