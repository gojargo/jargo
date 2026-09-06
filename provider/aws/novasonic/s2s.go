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
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
	"github.com/google/uuid"
)

// Service is the Nova Sonic speech-to-speech processor.
type Service struct {
	*llm.Base
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

	// convo is the conversation this session is part of, as the pipeline last
	// reported it. The model generates continuously and so never re-reads it,
	// but a tool call is answered into it and its results are read back out of
	// it here. Guarded by mu.
	convo *frames.LLMContext
	// tools are the tools the session was opened with. Nova Sonic takes them in
	// the prompt start and nowhere else, so a conversation bringing tools to a
	// session that has none is applied by opening another. Guarded by mu.
	tools []frames.Tool
	// sentResults names the tool calls whose result has already gone to the
	// model, so a conversation reported again does not send one twice. Guarded
	// by mu.
	sentResults map[string]bool
}

// New builds a Nova Sonic service.
func New(cfg Config) *Service {
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.Voice == "" {
		cfg.Voice = defaultVoice
	}
	s := &Service{cfg: cfg, sentResults: map[string]bool{}}
	// A model service, so it keeps the tool registry and the machinery that runs
	// what the model calls. It generates continuously rather than being run by a
	// conversation arriving, which is what the option says.
	s.Base = llm.New("NovaSonic", s, llm.WithContinuousGeneration())
	s.SetModel(cfg.Model)
	return s
}

// Generate is never called: this service generates continuously from the audio
// it is sent, and says so with llm.WithContinuousGeneration, so the base never
// asks it to answer a conversation. It exists because the base identifies the
// service it belongs to by the generator it was built with.
func (s *Service) Generate(context.Context, *frames.LLMContext, llm.Emit) error {
	return errNotGenerator
}

// ProcessFrame opens the session on StartFrame, forwards input audio to the
// model, and tears the session down when the pipeline ends.
func (s *Service) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if audio, ok := f.(*frames.InputAudioRawFrame); ok && dir == processor.Downstream {
		// The model consumes the audio, so it does not travel on. Only the
		// processor bookkeeping runs: the LLM base would forward it.
		if err := s.Base.Base.ProcessFrame(ctx, f, dir); err != nil {
			return err
		}
		s.sendAudio(audio.Audio)
		return nil
	}
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.StartFrame:
		if err := s.connect(ctx); err != nil {
			s.PushError(ctx, "nova sonic connect failed", err, true)
		}
		return nil
	case *frames.LLMContextFrame:
		// The conversation the session is part of: the toolset it advertises, and
		// the tool results it has gained since it was last reported. The base
		// holds a conversation back from a service generating continuously, so
		// this is where it travels on.
		if fr.Context != nil {
			s.handleContext(ctx, fr.Context)
		}
		return s.PushFrame(ctx, f, dir)
	case *frames.EndFrame, *frames.CancelFrame:
		s.disconnect()
		return nil
	default:
		return nil
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
	body := map[string]any{
		keyPromptName:              s.promptName,
		"textOutputConfiguration":  map[string]any{keyMediaType: mediaTypeText},
		"audioOutputConfiguration": audioConfig(outputSampleRate, s.cfg.Voice),
	}
	s.mu.Lock()
	tools := s.tools
	s.mu.Unlock()
	if specs := toolSpecs(tools); len(specs) > 0 {
		body["toolUseOutputConfiguration"] = map[string]any{keyMediaType: "application/json"}
		body["toolConfiguration"] = map[string]any{"tools": specs}
	}
	return event("promptStart", body)
}

// toolSpecs renders the tools the way Nova Sonic declares them: each carries its
// JSON Schema as a string rather than as an object.
func toolSpecs(tools []frames.Tool) []map[string]any {
	specs := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		spec := map[string]any{"name": t.Name}
		if t.Description != "" {
			spec["description"] = t.Description
		}
		schema := t.Parameters
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		spec["inputSchema"] = map[string]any{"json": string(schema)}
		specs = append(specs, map[string]any{"toolSpec": spec})
	}
	return specs
}

// handleContext takes the conversation the pipeline reported.
//
// The model generates continuously and never re-reads it, so a conversation
// arriving means two things and only two: the toolset it advertises, and the
// results of the tool calls the model asked for.
//
// The toolset is only settled once. Nova Sonic takes it in the prompt start and
// offers no way to change it afterwards, so a conversation bringing tools to a
// session opened without them is applied by opening another session with them.
func (s *Service) handleContext(ctx context.Context, convo *frames.LLMContext) {
	s.mu.Lock()
	first := s.convo == nil
	s.convo = convo
	had := len(s.tools) > 0
	s.mu.Unlock()

	tools := convo.Tools()
	if first && !had && len(tools) > 0 {
		s.mu.Lock()
		s.tools = tools
		s.mu.Unlock()
		s.reopenWithTools(ctx)
	}

	// The results already in the first conversation were produced before this
	// session existed, so they are recorded as known rather than sent: the model
	// never asked for them and has nothing to do with them.
	s.processCompletedCalls(ctx, convo, !first)
}

// reopenWithTools opens the session again so its prompt start declares the
// tools. What the model has heard so far is lost with the session, which at this
// point is nothing: the conversation has only just reached the service.
func (s *Service) reopenWithTools(ctx context.Context) {
	slog.InfoContext(ctx, "reopening the session to declare its tools", "service", s.Name())
	s.disconnect()
	if err := s.connect(ctx); err != nil {
		s.PushError(ctx, "nova sonic reconnect failed", err, true)
	}
}

// processCompletedCalls finds the tool results the model has not been given yet
// and, when send is set, hands each to the session.
//
// The results are read out of the conversation rather than taken from the
// handler directly, because that is where a call's result lands however it was
// produced: a handler that answered at once, one that answered later, or one the
// application answered for.
func (s *Service) processCompletedCalls(ctx context.Context, convo *frames.LLMContext, send bool) {
	for _, m := range convo.Messages() {
		for _, r := range m.ToolResults {
			if s.resultSeen(r.ID) {
				continue
			}
			// A call still running has a placeholder standing in for its result.
			// It is not an answer, so it is neither sent nor recorded: the real
			// one has still to come.
			if r.Content == frames.ToolResultInProgress {
				continue
			}
			if async, ok := frames.ParseAsyncToolMessage(m); ok && !s.asyncResultSendable(ctx, async) {
				s.recordResult(r.ID)
				continue
			}
			s.recordResult(r.ID)
			if send {
				s.sendToolResult(ctx, r.ID, r.Content)
			}
		}
	}
}

// asyncResultSendable reports whether an asynchronous tool's message carries a
// result this session can take. Only the final one does: the session has no way
// to receive a result in parts, and the marker that opens the call says nothing
// the model does not already know, having made the call itself.
func (s *Service) asyncResultSendable(ctx context.Context, m frames.AsyncToolMessage) bool {
	switch m.Kind {
	case frames.AsyncToolFinal:
		return true
	case frames.AsyncToolIntermediate:
		msg := "nova sonic takes no streamed result from an asynchronous tool"
		slog.ErrorContext(ctx, msg, "function", m.ToolCallID)
		s.PushError(ctx, msg, nil, false)
		return false
	default:
		return false
	}
}

// resultSeen reports whether the model has already been given this call's
// result.
func (s *Service) resultSeen(toolCallID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.sentResults[toolCallID]
}

// recordResult marks a call's result as settled, so a conversation reported
// again does not send it twice.
func (s *Service) recordResult(toolCallID string) {
	s.mu.Lock()
	s.sentResults[toolCallID] = true
	s.mu.Unlock()
}

// sendToolResult hands one call's result to the session. Nova Sonic takes it as
// a content block of its own rather than as a single message: the block is
// opened naming the call it answers, the result is sent, and the block is
// closed.
func (s *Service) sendToolResult(ctx context.Context, toolCallID, result string) {
	s.mu.Lock()
	prompt := s.promptName
	s.mu.Unlock()
	if prompt == "" {
		return
	}

	content := uuid.NewString()
	slog.DebugContext(ctx, "sending a tool result to the session",
		"service", s.Name(), "tool_call_id", toolCallID)

	for _, ev := range toolResultEvents(prompt, content, toolCallID, result) {
		if err := s.send(ev); err != nil {
			slog.ErrorContext(ctx, "sending a tool result failed", "service", s.Name(), "err", err)
			return
		}
	}
}

// toolResultEvents is the block a tool result is sent as: it is opened naming
// the call it answers, the result is sent, and it is closed. Nova Sonic takes a
// result this way rather than as a single message.
func toolResultEvents(prompt, content, toolCallID, result string) []map[string]any {
	return []map[string]any{
		contentStart(prompt, content, "TOOL", "TOOL", false, map[string]any{
			"toolResultInputConfiguration": map[string]any{
				"toolUseId":              toolCallID,
				"type":                   "TEXT",
				"textInputConfiguration": map[string]any{keyMediaType: mediaTypeText},
			},
		}),
		event("toolResult", map[string]any{
			keyPromptName:  prompt,
			keyContentName: content,
			keyContent:     result,
		}),
		contentEnd(prompt, content),
	}
}

// runToolCall runs the call the model asked for.
func (s *Service) runToolCall(ctx context.Context, use *toolUse) {
	if use == nil || use.ToolUseID == "" {
		return
	}
	s.mu.Lock()
	convo := s.convo
	s.mu.Unlock()
	if convo == nil {
		// A call's result is written into the conversation and read back out of
		// it, so without one there is nowhere for the answer to go.
		slog.ErrorContext(ctx, "the model called a tool before any conversation reached the service",
			"service", s.Name(), "function", use.ToolName)
		return
	}

	args := json.RawMessage(use.Content)
	if len(args) == 0 {
		args = json.RawMessage("{}")
	}
	call := frames.ToolCall{ID: use.ToolUseID, Name: use.ToolName, Args: args}
	if err := s.RunFunctionCalls(ctx, convo, []frames.ToolCall{call}); err != nil {
		slog.ErrorContext(ctx, "running a tool call failed", "service", s.Name(),
			"function", use.ToolName, "err", err)
	}
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
		keyContent:     base64.StdEncoding.EncodeToString(pcm),
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
	case ev.ToolUse != nil:
		s.runToolCall(ctx, ev.ToolUse)
	case ev.UsageEvent != nil:
		s.handleUsage(ctx, ev.UsageEvent.Details.Delta)
	}
}

// handleUsage reports the tokens one usage event accounts for. The delta is
// reported rather than the running total, so usage stays incremental per event
// as the other realtime services report it.
func (s *Service) handleUsage(ctx context.Context, delta usageDelta) {
	// The service splits each direction into speech and text, so the prompt and
	// completion counts are the two added together.
	prompt := delta.Input.SpeechTokens + delta.Input.TextTokens
	completion := delta.Output.SpeechTokens + delta.Output.TextTokens
	if prompt == 0 && completion == 0 {
		return
	}
	_ = s.PushTokenUsage(ctx, frames.LLMTokenUsage{
		PromptTokens:      prompt,
		CompletionTokens:  completion,
		TotalTokens:       prompt + completion,
		InputAudioTokens:  new(delta.Input.SpeechTokens),
		OutputAudioTokens: new(delta.Output.SpeechTokens),
		InputTextTokens:   new(delta.Input.TextTokens),
		OutputTextTokens:  new(delta.Output.TextTokens),
	})
}

// handleText routes a transcript: a barge-in marker interrupts, a user transcript
// becomes a TranscriptionFrame, and a (non-speculative) assistant transcript
// becomes an LLMTextFrame.
func (s *Service) handleText(ctx context.Context, role, content string) {
	var probe struct {
		Interrupted bool `json:"interrupted"`
	}
	if json.Unmarshal([]byte(content), &probe) == nil && probe.Interrupted {
		// The bot was cut off, so it gives the floor up and the pipeline is
		// interrupted. No user-speaking frame goes with it: this service reports
		// a barge-in but never a turn starting or ending, so a start emitted
		// here would have no stop to match it, and everything keyed off those
		// frames (turn tracking, the idle watch, mute strategies) would be left
		// believing the user is still speaking. A pipeline that needs turn
		// boundaries runs its own detection alongside this service.
		s.setSpeaking(ctx, false)
		// Broadcast, not pushed on: the aggregators on either side of this
		// service both act on an interruption, and the one keeping the user's
		// turn sits upstream of it.
		_ = s.BroadcastInterruption(ctx)
		return
	}
	switch role {
	case "USER":
		// The user's words go upstream, where the user aggregator is: everything
		// downstream of this service is the reply to them.
		_ = s.PushFrame(ctx, frames.NewTranscriptionFrame(content, "", ""), processor.Upstream)
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
		body["textInputConfiguration"] = map[string]any{keyMediaType: mediaTypeText}
	}
	maps.Copy(body, extra)
	return event("contentStart", body)
}

func textInput(prompt, content, text string) map[string]any {
	return event("textInput", map[string]any{
		keyPromptName:  prompt,
		keyContentName: content,
		keyContent:     text,
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
	ToolUse    *toolUse `json:"toolUse"` //nolint:tagliatelle // Nova Sonic wire field
	UsageEvent *struct {
		Details struct {
			// Delta is what this event adds; Details also carries a running
			// total, which is left alone so usage stays incremental.
			Delta usageDelta `json:"delta"`
		} `json:"details"`
	} `json:"usageEvent"` //nolint:tagliatelle // Nova Sonic wire field
}

// toolUse is the model asking for a function to be called. The arguments arrive
// as the JSON text the model wrote, not as a decoded object.
type toolUse struct {
	ToolName  string `json:"toolName"`  //nolint:tagliatelle // Nova Sonic wire field
	ToolUseID string `json:"toolUseId"` //nolint:tagliatelle // Nova Sonic wire field
	Content   string `json:"content"`
}

// usageDelta is the token accounting one usage event adds, split by direction
// and, within each, by modality.
type usageDelta struct {
	Input  usageTokens `json:"input"`
	Output usageTokens `json:"output"`
}

// usageTokens is one direction's accounting for a usage event.
type usageTokens struct {
	SpeechTokens int64 `json:"speechTokens"` //nolint:tagliatelle // Nova Sonic wire field
	TextTokens   int64 `json:"textTokens"`   //nolint:tagliatelle // Nova Sonic wire field
}

// CanGenerateMetrics reports that this service times the conversation and reports
// the result, so the pipeline counts it when it collects the processors that
// report metrics.
func (s *Service) CanGenerateMetrics() bool { return true }

// ServiceMetadataFrame describes this service to the pipeline as a realtime one.
// The user aggregator reads that and stops holding turns open for transcripts,
// which this service is not answering from, and writes the user's message when
// the answer starts rather than when the turn ends, because the transcript for
// what the user said arrives late.
func (s *Service) ServiceMetadataFrame() frames.ServiceMetadata {
	f := frames.NewLLMServiceMetadataFrame(s.Name())
	f.Realtime = true
	return f
}
