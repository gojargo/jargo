// Package novasonic is a speech-to-speech service built on Amazon Nova Sonic,
// served over Bedrock's bidirectional streaming API. A single SigV4-signed
// HTTP/2 duplex stream carries the conversation: input audio streams up and the
// model streams its spoken reply, its transcript, and the user's transcript
// back down. The AWS SDK handles signing, HTTP/2 and event-stream framing; this
// service drives the Nova Sonic event protocol on top.
//
// Place the service where the STT/LLM/TTS stack would go, between the transport
// input and output. Nova Sonic takes 16 kHz mono PCM in and returns 24 kHz mono
// PCM out, so run the transport input at 16000 and output at 24000.
package novasonic

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"log/slog"
	"maps"
	"sync"
	"sync/atomic"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/processor"
	"github.com/google/uuid"
)

const (
	defaultModel     = "amazon.nova-sonic-v1:0"
	defaultVoice     = "matthew"
	inputSampleRate  = 16000
	outputSampleRate = 24000

	keyPromptName  = "promptName"
	keyContentName = "contentName"
	keyMediaType   = "mediaType"
)

// Config configures the Nova Sonic service.
type Config struct {
	// Region is the AWS region (Nova Sonic launched in us-east-1); empty uses the
	// default chain (AWS_REGION, shared config).
	Region string
	// AccessKeyID and SecretAccessKey set static credentials; leave both empty to
	// use the default AWS credential chain (environment, shared config, IAM role).
	AccessKeyID     string
	SecretAccessKey string
	// SessionToken is the optional session token for temporary credentials.
	SessionToken string
	// Model is the Nova Sonic model id; empty uses a current default.
	Model string
	// Voice is the output voice id (e.g. matthew, tiffany, ambre, florian); empty
	// uses a default.
	Voice string
	// Instructions is the system prompt for the session.
	Instructions string
	// MaxTokens caps the model's response length; 0 leaves the API default.
	MaxTokens int
	// Temperature and TopP are optional sampling controls; nil leaves the API
	// default in place.
	Temperature *float64
	TopP        *float64
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// Service is the Nova Sonic speech-to-speech processor.
type Service struct {
	*processor.Base
	cfg Config

	mu      sync.Mutex
	stream  *bedrockruntime.InvokeModelWithBidirectionalStreamEventStream
	connCtx context.Context
	cancel  context.CancelFunc
	writeMu sync.Mutex
	wg      sync.WaitGroup
	ready   atomic.Bool

	promptName           string
	audioContent         string
	speaking             bool
	assistantSpeculative bool
}

// New builds a Nova Sonic service.
func New(cfg Config) *Service {
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.Voice == "" {
		cfg.Voice = defaultVoice
	}
	s := &Service{cfg: cfg}
	s.Base = processor.New("NovaSonic", s)
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
			s.PushError(ctx, "nova sonic connect failed", err, true)
		}
		return s.PushFrame(ctx, f, dir)
	case *frames.InputAudioRawFrame:
		if dir == processor.Downstream {
			s.sendAudio(fr.Audio)
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

// connect loads AWS config, opens the bidirectional stream, sends the session
// handshake, and starts the read loop.
func (s *Service) connect(ctx context.Context) error {
	awsCfg, err := awsconfig.LoadDefaultConfig(ctx, s.cfg.loadOptions()...)
	if err != nil {
		return err
	}
	client := bedrockruntime.NewFromConfig(awsCfg)

	connCtx, cancel := context.WithCancel(context.Background())
	out, err := client.InvokeModelWithBidirectionalStream(connCtx,
		&bedrockruntime.InvokeModelWithBidirectionalStreamInput{ModelId: aws.String(s.cfg.Model)})
	if err != nil {
		cancel()
		return err
	}
	stream := out.GetStream()

	s.mu.Lock()
	s.stream = stream
	s.connCtx = connCtx
	s.cancel = cancel
	s.promptName = uuid.NewString()
	s.audioContent = uuid.NewString()
	s.mu.Unlock()

	if err := s.handshake(); err != nil {
		cancel()
		_ = stream.Close()
		return err
	}
	s.ready.Store(true)

	s.wg.Add(1)
	go s.readLoop(stream)
	return nil
}

func (c Config) loadOptions() []func(*awsconfig.LoadOptions) error {
	var opts []func(*awsconfig.LoadOptions) error
	if c.Region != "" {
		opts = append(opts, awsconfig.WithRegion(c.Region))
	}
	if c.AccessKeyID != "" && c.SecretAccessKey != "" {
		opts = append(opts, awsconfig.WithCredentialsProvider(
			credentials.NewStaticCredentialsProvider(c.AccessKeyID, c.SecretAccessKey, c.SessionToken),
		))
	}
	return opts
}

// handshake sends the ordered session-setup events: sessionStart, promptStart,
// the system-prompt content block, then opens the audio input block.
func (s *Service) handshake() error {
	sysContent := uuid.NewString()
	events := []map[string]any{
		s.sessionStart(),
		s.promptStart(),
		contentStart(s.promptName, sysContent, "TEXT", "SYSTEM", false, nil),
		textInput(s.promptName, sysContent, s.cfg.Instructions),
		contentEnd(s.promptName, sysContent),
		contentStart(s.promptName, s.audioContent, "AUDIO", "USER", true, map[string]any{
			"audioInputConfiguration": audioConfig(inputSampleRate, ""),
		}),
	}
	for _, ev := range events {
		if err := s.send(ev); err != nil {
			return err
		}
	}
	return nil
}

func (s *Service) sessionStart() map[string]any {
	inference := map[string]any{}
	if s.cfg.MaxTokens > 0 {
		inference["maxTokens"] = s.cfg.MaxTokens
	}
	if s.cfg.Temperature != nil {
		inference["temperature"] = *s.cfg.Temperature
	}
	if s.cfg.TopP != nil {
		inference["topP"] = *s.cfg.TopP
	}
	return event("sessionStart", map[string]any{"inferenceConfiguration": inference})
}

func (s *Service) promptStart() map[string]any {
	return event("promptStart", map[string]any{
		keyPromptName:              s.promptName,
		"textOutputConfiguration":  map[string]any{keyMediaType: "text/plain"},
		"audioOutputConfiguration": audioConfig(outputSampleRate, s.cfg.Voice),
	})
}

// sendAudio streams a chunk of input PCM as an audioInput event.
func (s *Service) sendAudio(pcm []byte) {
	if len(pcm) == 0 || !s.ready.Load() {
		return
	}
	s.mu.Lock()
	prompt, content := s.promptName, s.audioContent
	s.mu.Unlock()
	_ = s.send(event("audioInput", map[string]any{
		keyPromptName:  prompt,
		keyContentName: content,
		"content":      base64.StdEncoding.EncodeToString(pcm),
	}))
}

// send marshals an event and writes it as one input chunk, serializing writes.
func (s *Service) send(ev map[string]any) error {
	data, err := json.Marshal(ev)
	if err != nil {
		return err
	}
	s.mu.Lock()
	stream, connCtx := s.stream, s.connCtx
	s.mu.Unlock()
	if stream == nil {
		return nil
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return stream.Writer.Send(connCtx, &types.InvokeModelWithBidirectionalStreamInputMemberChunk{
		Value: types.BidirectionalInputPayloadPart{Bytes: data},
	})
}

// disconnect sends the teardown events, closes the stream, and waits for the
// read loop. It is safe to call more than once.
func (s *Service) disconnect() {
	s.mu.Lock()
	stream, cancel := s.stream, s.cancel
	prompt, content := s.promptName, s.audioContent
	s.stream, s.cancel, s.connCtx = nil, nil, nil
	s.mu.Unlock()
	if stream == nil {
		return
	}
	s.ready.Store(false)

	// Best-effort orderly teardown before closing the stream.
	_ = s.sendOn(stream, contentEnd(prompt, content))
	_ = s.sendOn(stream, event("promptEnd", map[string]any{keyPromptName: prompt}))
	_ = s.sendOn(stream, event("sessionEnd", map[string]any{}))

	if cancel != nil {
		cancel()
	}
	_ = stream.Close()
	s.wg.Wait()
}

// sendOn writes an event on a specific stream during teardown, without the
// connection-context guard send uses.
func (s *Service) sendOn(
	stream *bedrockruntime.InvokeModelWithBidirectionalStreamEventStream, ev map[string]any,
) error {
	data, err := json.Marshal(ev)
	if err != nil {
		return err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return stream.Writer.Send(context.Background(), &types.InvokeModelWithBidirectionalStreamInputMemberChunk{
		Value: types.BidirectionalInputPayloadPart{Bytes: data},
	})
}

// readLoop reads model output events until the stream ends.
func (s *Service) readLoop(stream *bedrockruntime.InvokeModelWithBidirectionalStreamEventStream) {
	defer s.wg.Done()
	for ev := range stream.Reader.Events() {
		chunk, ok := ev.(*types.InvokeModelWithBidirectionalStreamOutputMemberChunk)
		if !ok {
			continue // error union members are surfaced via Reader.Err below
		}
		var env outputEnvelope
		if json.Unmarshal(chunk.Value.Bytes, &env) != nil {
			continue
		}
		s.handle(env.Event)
	}
	if err := stream.Reader.Err(); err != nil {
		s.mu.Lock()
		ctx := s.connCtx
		s.mu.Unlock()
		if ctx != nil && ctx.Err() == nil {
			slog.Debug("nova sonic read ended", "err", err)
		}
	}
}

// handle maps one Nova Sonic output event onto downstream pipeline frames.
func (s *Service) handle(ev outputEvent) {
	ctx := s.eventCtx()
	switch {
	case ev.AudioOutput != nil:
		if pcm, err := base64.StdEncoding.DecodeString(ev.AudioOutput.Content); err == nil && len(pcm) > 0 {
			s.setSpeaking(ctx, true)
			_ = s.PushFrame(ctx, frames.NewTTSAudioRawFrame(pcm, outputSampleRate, 1), processor.Downstream)
		}
	case ev.ContentStart != nil:
		s.assistantSpeculative = ev.ContentStart.Role == "ASSISTANT" &&
			generationStage(ev.ContentStart.AdditionalModelFields) == "SPECULATIVE"
	case ev.TextOutput != nil:
		s.handleText(ctx, ev.TextOutput.Role, ev.TextOutput.Content)
	case ev.ContentEnd != nil:
		if ev.ContentEnd.Type == "AUDIO" {
			s.setSpeaking(ctx, false)
		}
	}
}

// handleText routes a transcript: a barge-in marker interrupts, a user transcript
// becomes a TranscriptionFrame, and a (non-speculative) assistant transcript
// becomes an LLMTextFrame.
func (s *Service) handleText(ctx context.Context, role, content string) {
	var probe struct {
		Interrupted bool `json:"interrupted"`
	}
	if json.Unmarshal([]byte(content), &probe) == nil && probe.Interrupted {
		s.setSpeaking(ctx, false)
		_ = s.PushFrame(ctx, frames.NewUserStartedSpeakingFrame(), processor.Downstream)
		_ = s.PushFrame(ctx, frames.NewInterruptionFrame(), processor.Downstream)
		return
	}
	switch role {
	case "USER":
		_ = s.PushFrame(ctx, frames.NewTranscriptionFrame(content, "", ""), processor.Downstream)
	case "ASSISTANT":
		if !s.assistantSpeculative {
			_ = s.PushFrame(ctx, frames.NewLLMTextFrame(content), processor.Downstream)
		}
	}
}

// eventCtx returns the connection context for pushing frames from the read loop.
func (s *Service) eventCtx() context.Context {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.connCtx != nil {
		return s.connCtx
	}
	return context.Background()
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

// generationStage extracts generationStage from contentStart's stringified
// additionalModelFields JSON.
func generationStage(additionalModelFields string) string {
	if additionalModelFields == "" {
		return ""
	}
	var f struct {
		GenerationStage string `json:"generationStage"` //nolint:tagliatelle // Nova Sonic wire field
	}
	_ = json.Unmarshal([]byte(additionalModelFields), &f)
	return f.GenerationStage
}

// --- event builders (outbound; maps, so no struct-tag casing concerns) ---

func event(name string, body map[string]any) map[string]any {
	return map[string]any{"event": map[string]any{name: body}}
}

func audioConfig(sampleRate int, voice string) map[string]any {
	cfg := map[string]any{
		keyMediaType:      "audio/lpcm",
		"sampleRateHertz": sampleRate,
		"sampleSizeBits":  16,
		"channelCount":    1,
		"audioType":       "SPEECH",
		"encoding":        "base64",
	}
	if voice != "" {
		cfg["voiceId"] = voice
	}
	return cfg
}

func contentStart(prompt, content, typ, role string, interactive bool, extra map[string]any) map[string]any {
	body := map[string]any{
		keyPromptName:  prompt,
		keyContentName: content,
		"type":         typ,
		"role":         role,
		"interactive":  interactive,
	}
	if typ == "TEXT" {
		body["textInputConfiguration"] = map[string]any{keyMediaType: "text/plain"}
	}
	maps.Copy(body, extra)
	return event("contentStart", body)
}

func textInput(prompt, content, text string) map[string]any {
	return event("textInput", map[string]any{
		keyPromptName:  prompt,
		keyContentName: content,
		"content":      text,
	})
}

func contentEnd(prompt, content string) map[string]any {
	return event("contentEnd", map[string]any{
		keyPromptName:  prompt,
		keyContentName: content,
	})
}

// --- inbound event parsing ---

// The JSON field names below are Nova Sonic's wire protocol (camelCase), so the
// snake_case house style does not apply.

type outputEnvelope struct {
	Event outputEvent `json:"event"`
}

type outputEvent struct {
	AudioOutput *struct {
		Content string `json:"content"`
	} `json:"audioOutput"` //nolint:tagliatelle // Nova Sonic wire field
	TextOutput *struct {
		Content string `json:"content"`
		Role    string `json:"role"`
	} `json:"textOutput"` //nolint:tagliatelle // Nova Sonic wire field
	ContentStart *struct {
		Type                  string `json:"type"`
		Role                  string `json:"role"`
		AdditionalModelFields string `json:"additionalModelFields"` //nolint:tagliatelle // Nova Sonic wire field
	} `json:"contentStart"` //nolint:tagliatelle // Nova Sonic wire field
	ContentEnd *struct {
		Type       string `json:"type"`
		StopReason string `json:"stopReason"` //nolint:tagliatelle // Nova Sonic wire field
	} `json:"contentEnd"` //nolint:tagliatelle // Nova Sonic wire field
}
