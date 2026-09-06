package realtime

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/wsutil"
)

// Service is the xAI Realtime speech-to-speech processor.
type Service struct {
	*llm.Base
	cfg Config

	mu      sync.Mutex
	conn    *wsutil.Conn
	connCtx context.Context
	cancel  context.CancelFunc
	writeMu sync.Mutex
	wg      sync.WaitGroup

	// established reports whether the server has opened the conversation. xAI
	// only accepts session configuration from that point on, so a tool change
	// arriving earlier is folded into the initial session update instead.
	established bool

	// sessionReady reports whether the server has acknowledged the session
	// configuration. The sample rate the session runs at is settled at that
	// point, so audio sent earlier would be read at the wrong rate.
	sessionReady bool
	// loggedAudioDrop records that audio arriving before the session was ready
	// has been reported once, so a talking caller does not fill the log.
	loggedAudioDrop bool
	// pendingResponse records that a response was asked for before the session
	// was ready, to be asked for again once it is.
	pendingResponse bool
	// audioResp is the assistant audio item currently being played, which an
	// interruption truncates on the server so the conversation the model keeps
	// holds what the caller actually heard. Nil when nothing is playing.
	audioResp *audioResponse

	// tools is the function-calling configuration currently advertised to the
	// session. The model generates continuously, so it does not re-read the
	// context between turns: every change must be pushed to it with a
	// session.update. It is guarded by mu.
	tools []frames.Tool

	// convo is the conversation this session is part of, as the pipeline last
	// reported it. The model generates continuously and so never re-reads it,
	// but a tool call is answered into it and its results are read back out of
	// it here. Guarded by mu.
	convo *frames.LLMContext
	// pendingCalls are the tool calls the model has announced but not finished
	// naming the arguments for, by call id. Guarded by mu.
	pendingCalls map[string]bool
	// sentResults names the tool calls whose result has already gone to the
	// model, so a conversation reported again does not send one twice. Guarded
	// by mu.
	sentResults map[string]bool
}

// audioResponse tracks the assistant audio item the session is playing: which
// item it is, when it started, and how much audio has arrived. An interruption
// reads it to say how much of the item the caller heard.
type audioResponse struct {
	itemID       string
	contentIndex int
	started      time.Time
	total        int
}

// New builds an xAI Realtime service.
func New(cfg Config) *Service {
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.Voice == "" {
		cfg.Voice = defaultVoice
	}
	s := &Service{
		cfg:          cfg,
		pendingCalls: map[string]bool{},
		sentResults:  map[string]bool{},
	}
	// A model service, so it keeps the tool registry and the machinery that runs
	// what the model calls. It generates continuously rather than being run by a
	// conversation arriving, which is what the option says.
	s.Base = llm.New("XAIRealtime", s, llm.WithContinuousGeneration())
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

// ProcessFrame opens the session on StartFrame, forwards input audio up to the
// model, and tears the session down when the pipeline ends.
//
// Every frame but the input audio is forwarded by the LLM base, which handles it
// first, so the work done here is the session's side of it and nothing is pushed
// a second time. The exception is the conversation, which the base holds back
// from a service that generates continuously.
func (s *Service) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if audio, ok := f.(*frames.InputAudioRawFrame); ok && dir == processor.Downstream {
		// The model consumes the audio, so it does not travel on. Only the
		// processor bookkeeping runs: the LLM base would forward it.
		if err := s.Base.Base.ProcessFrame(ctx, f, dir); err != nil {
			return err
		}
		s.sendAudio(ctx, audio.Audio)
		return nil
	}
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.StartFrame:
		if err := s.connect(ctx); err != nil {
			s.PushError(ctx, "xai realtime connect failed", err, true)
		}
		return nil
	case *frames.InterruptionFrame:
		s.handleInterruption(ctx)
		return nil
	case *frames.UserStoppedSpeakingFrame:
		// Under manual turn detection the pipeline decides where the turn ends,
		// so the input buffer is committed and a response asked for here. With
		// server VAD on the session does both for itself.
		s.handleUserStoppedSpeaking(ctx)
		return nil
	case *frames.LLMContextFrame:
		// The conversation the session is part of: the toolset it advertises, and
		// the tool results it has gained since it was last reported.
		if fr.Context != nil {
			s.handleContext(ctx, fr.Context)
		}
		return s.PushFrame(ctx, f, dir)
	case *frames.LLMSetToolsFrame:
		// The toolset changed mid-conversation. A text LLM would pick this up on
		// its next run; this model is generating continuously, so tell it now.
		// The handlers the new tools carry are registered here for the same
		// reason: the base does that when a conversation arrives, and this
		// service does not get one per turn.
		s.syncTools(fr.Tools)
		s.SyncToolHandlers(ctx, fr.Tools)
		return nil
	case *frames.EndFrame, *frames.CancelFrame:
		s.disconnect()
		return nil
	default:
		// A tool-choice change reaches no session control: the model always
		// decides for itself whether to call something.
		return nil
	}
}

// Cleanup tears down the session and stops the read loop.
func (s *Service) Cleanup(ctx context.Context) error {
	s.disconnect()
	return s.Base.Cleanup(ctx)
}

// connect dials the Realtime WebSocket and starts the read loop. The session is
// configured once the server opens the conversation, not here.
func (s *Service) connect(ctx context.Context) error {
	header := http.Header{}
	header.Set("Authorization", "Bearer "+s.cfg.APIKey)

	endpoint := s.cfg.BaseURL + "?model=" + url.QueryEscape(s.cfg.Model)
	conn, err := wsutil.Dial(ctx, endpoint, header, readLimit)
	if err != nil {
		return err
	}

	connCtx, cancel := context.WithCancel(context.Background())
	s.mu.Lock()
	s.conn = conn
	s.connCtx = connCtx
	s.cancel = cancel
	s.established = false
	s.mu.Unlock()

	s.wg.Add(1)
	go s.readLoop(conn, connCtx)
	return nil
}

// sessionUpdateMsg configures the session.
type sessionUpdateMsg struct {
	Type    string         `json:"type"`
	Session map[string]any `json:"session"`
}

// audioAppendMsg appends a chunk of input PCM to the model's input buffer.
type audioAppendMsg struct {
	Type  string `json:"type"`
	Audio string `json:"audio"`
}

// audioFormat renders the session's audio configuration. xAI nests the format
// under an input/output audio object rather than naming each direction's format
// at the top level, and the transcript of the user's audio is configured
// alongside the input format.
func (s *Service) audioFormat() map[string]any {
	format := func() map[string]any {
		return map[string]any{keyType: pcmFormat, "rate": s.cfg.sampleRate()}
	}
	input := map[string]any{"format": format()}
	if t := s.cfg.Transcription; t != nil {
		transcription := map[string]any{}
		if t.Model != "" {
			transcription["model"] = t.Model
		}
		if t.LanguageHint != "" {
			transcription["language_hint"] = t.LanguageHint
		}
		if len(t.Keyterms) > 0 {
			transcription["keyterms"] = t.Keyterms
		}
		if len(transcription) > 0 {
			input["transcription"] = transcription
		}
	}
	return map[string]any{"input": input, "output": map[string]any{"format": format()}}
}

// turnDetection renders the session's turn-detection configuration: xAI's
// server-side VAD with whatever tuning the config carries, or nothing at all
// when the pipeline detects the turns itself.
func (s *Service) turnDetection() any {
	if !s.cfg.serverVAD() {
		return nil
	}
	td := map[string]any{keyType: "server_vad"}
	v := s.cfg.VAD
	if v == nil {
		return td
	}
	if v.Threshold != 0 {
		td["threshold"] = v.Threshold
	}
	if v.SilenceMS != 0 {
		td["silence_duration_ms"] = v.SilenceMS
	}
	if v.PrefixPaddingMS != 0 {
		td["prefix_padding_ms"] = v.PrefixPaddingMS
	}
	if v.IdleTimeoutMS != 0 {
		td["idle_timeout_ms"] = v.IdleTimeoutMS
	}
	return td
}

// sessionUpdate is the session configuration message. The model is not part of
// it: xAI selects the model on the handshake.
func (s *Service) sessionUpdate() sessionUpdateMsg {
	session := map[string]any{
		"voice": s.cfg.Voice,
		"audio": s.audioFormat(),
	}
	if s.cfg.Instructions != "" {
		session["instructions"] = s.cfg.Instructions
	}
	session["turn_detection"] = s.turnDetection()
	if s.cfg.Reasoning != "" {
		session["reasoning"] = map[string]any{"effort": s.cfg.Reasoning}
	}
	if s.cfg.Resumption {
		session["resumption"] = map[string]any{"enabled": true}
	}
	if len(s.cfg.Replace) > 0 {
		session["replace"] = s.cfg.Replace
	}
	if tools := s.toolSpecs(s.currentTools()); tools != nil {
		session["tools"] = tools
	}
	return sessionUpdateMsg{Type: "session.update", Session: session}
}

// toolSpecs renders the tool list: xAI's built-in search tools from the config,
// then the function tools the pipeline advertises. It is nil when there are
// none, so a session without tools is configured without the field.
func (s *Service) toolSpecs(tools []frames.Tool) []map[string]any {
	specs := make([]map[string]any, 0, len(tools)+3)
	if s.cfg.WebSearch {
		specs = append(specs, map[string]any{keyType: "web_search"})
	}
	if s.cfg.XSearch {
		spec := map[string]any{keyType: "x_search"}
		if len(s.cfg.XSearchHandles) > 0 {
			spec["allowed_x_handles"] = s.cfg.XSearchHandles
		}
		specs = append(specs, spec)
	}
	if fs := s.cfg.FileSearch; fs != nil {
		spec := map[string]any{keyType: "file_search", "vector_store_ids": fs.VectorStoreIDs}
		if fs.MaxResults > 0 {
			spec["max_num_results"] = fs.MaxResults
		}
		specs = append(specs, spec)
	}
	for _, t := range tools {
		spec := map[string]any{keyType: "function", "name": t.Name}
		if t.Description != "" {
			spec["description"] = t.Description
		}
		if len(t.Parameters) > 0 {
			spec["parameters"] = t.Parameters
		}
		specs = append(specs, spec)
	}
	if len(specs) == 0 {
		return nil
	}
	return specs
}

// handleContext takes the conversation the pipeline reported.
//
// The model generates continuously and never re-reads it, so a conversation
// arriving means two things and only two: the toolset it advertises may have
// changed, and it may have gained the result of a tool call the model asked for.
// Both are pushed to the session here.
func (s *Service) handleContext(ctx context.Context, convo *frames.LLMContext) {
	s.mu.Lock()
	first := s.convo == nil
	s.convo = convo
	s.mu.Unlock()

	s.syncTools(convo.Tools())

	// The results already in the first conversation were produced before this
	// session existed, so they are recorded as known rather than sent: the model
	// never asked for them and has nothing to do with them.
	s.processCompletedCalls(ctx, convo, !first)
}

// processCompletedCalls finds the tool results the model has not been given yet
// and, when send is set, hands each to the session and asks for a reply.
//
// The results are read out of the conversation rather than taken from the
// handler directly, because that is where a call's result lands however it was
// produced: a handler that answered at once, one that answered later, or one the
// application answered for.
func (s *Service) processCompletedCalls(ctx context.Context, convo *frames.LLMContext, send bool) {
	sent := false
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
				sent = true
			}
		}
	}
	if sent {
		// The model has been told what its calls returned, so it can speak the
		// answer. It is generating from audio otherwise, and nothing else would
		// prompt it.
		s.createResponse(ctx)
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
		msg := "xai realtime takes no streamed result from an asynchronous tool"
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

// sendToolResult hands one call's result to the session.
func (s *Service) sendToolResult(ctx context.Context, toolCallID, result string) {
	slog.DebugContext(ctx, "sending a tool result to the realtime session",
		"service", s.Name(), "tool_call_id", toolCallID)
	if err := s.send(map[string]any{
		keyType: "conversation.item.create",
		"item": map[string]any{
			keyType:   "function_call_output",
			"call_id": toolCallID,
			"output":  result,
		},
	}); err != nil {
		slog.ErrorContext(ctx, "sending a tool result failed", "service", s.Name(), "err", err)
	}
}

// createResponse asks the model to speak. The session generates from audio on
// its own, so this is for the turns nothing spoken prompted: the reply to a tool
// result.
//
// A request made before the server has acknowledged the session is held until it
// has, because the session it would run against is not settled yet.
func (s *Service) createResponse(ctx context.Context) {
	s.mu.Lock()
	ready := s.sessionReady
	if !ready {
		s.pendingResponse = true
	}
	s.mu.Unlock()
	if !ready {
		return
	}
	if err := s.send(map[string]any{keyType: "response.create"}); err != nil {
		slog.ErrorContext(ctx, "asking for a response failed", "service", s.Name(), "err", err)
	}
}

// handleInterruption stops the model on the wire when the pipeline reports an
// interruption.
//
// The response is always canceled, so the model stops speaking promptly. The
// input buffer is only cleared under manual turn detection, which owns what goes
// into it: with server VAD on the buffer holds the speech that interrupted, and
// clearing it would throw the user's words away.
func (s *Service) handleInterruption(ctx context.Context) {
	if !s.cfg.serverVAD() {
		if err := s.send(map[string]any{keyType: "input_audio_buffer.clear"}); err != nil {
			slog.DebugContext(ctx, "clearing the input buffer failed", "service", s.Name(), "err", err)
		}
	}
	if err := s.send(map[string]any{keyType: "response.cancel"}); err != nil {
		slog.DebugContext(ctx, "canceling the response failed", "service", s.Name(), "err", err)
	}
	s.truncateAudioResponse(ctx)
}

// handleUserStoppedSpeaking closes a turn the pipeline detected itself. With
// server VAD on the session commits and answers on its own, so this does
// nothing.
func (s *Service) handleUserStoppedSpeaking(ctx context.Context) {
	if s.cfg.serverVAD() {
		return
	}
	if err := s.send(map[string]any{keyType: "input_audio_buffer.commit"}); err != nil {
		slog.ErrorContext(ctx, "committing the input buffer failed", "service", s.Name(), "err", err)
		return
	}
	s.createResponse(ctx)
}

// trackAudio records the assistant audio item being played, so an interruption
// can say how much of it the caller heard. The first chunk opens the item and
// starts its clock; the rest only add to its size.
func (s *Service) trackAudio(ev serverEvent, n int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.audioResp == nil {
		s.audioResp = &audioResponse{
			itemID:       ev.ItemID,
			contentIndex: ev.ContentIndex,
			started:      time.Now(),
		}
	}
	s.audioResp.total += n
}

// truncateAudioResponse cuts the item being played back to what the caller
// actually heard, so the conversation the model keeps matches it. Without this
// the model believes it said a whole answer the user cut off after a word.
//
// The cut is the shorter of the time since playback started and the audio that
// arrived, since either can be the smaller: a response can arrive faster than it
// plays, and playback can outlast a response that stopped arriving.
func (s *Service) truncateAudioResponse(ctx context.Context) {
	s.mu.Lock()
	resp := s.audioResp
	s.audioResp = nil
	s.mu.Unlock()
	if resp == nil {
		return
	}
	// 16-bit mono PCM, so two bytes to the sample.
	audioMS := int64(resp.total) * 1000 / int64(2*s.cfg.sampleRate())
	elapsedMS := time.Since(resp.started).Milliseconds()
	endMS := max(0, min(elapsedMS, audioMS))
	slog.DebugContext(ctx, "truncating the interrupted response", "service", s.Name(),
		"item_id", resp.itemID, "audio_ms", audioMS, "elapsed_ms", elapsedMS, "end_ms", endMS)
	if err := s.send(map[string]any{
		keyType:         "conversation.item.truncate",
		"item_id":       resp.itemID,
		"content_index": resp.contentIndex,
		"audio_end_ms":  endMS,
	}); err != nil {
		// Non-fatal: the response has been canceled either way, and only the
		// model's own record of the turn is left overstated.
		slog.WarnContext(ctx, "truncating the interrupted response failed",
			"service", s.Name(), "err", err)
	}
}

// ForceMessage speaks text verbatim, without the model writing it. The session
// plays it and reports the response itself, so nothing else has to be asked for.
func (s *Service) ForceMessage(ctx context.Context, text string) error {
	slog.DebugContext(ctx, "speaking a forced message", "service", s.Name())
	return s.send(map[string]any{
		keyType: "conversation.item.create",
		"item": map[string]any{
			keyType: "force_message",
			"role":  "assistant",
			"content": []map[string]any{
				{keyType: "text", "text": text},
			},
		},
	})
}

// DeleteConversationItem removes an item from the conversation the session
// keeps, by the id the session gave it.
func (s *Service) DeleteConversationItem(ctx context.Context, itemID string) error {
	slog.DebugContext(ctx, "deleting a conversation item", "service", s.Name(), "item_id", itemID)
	return s.send(map[string]any{keyType: "conversation.item.delete", "item_id": itemID})
}

// handleToolEvent takes the events that make up a tool call, reporting whether
// it took this one. They are handled apart from the rest because they are one
// exchange rather than several unrelated events.
func (s *Service) handleToolEvent(ctx context.Context, ev serverEvent) bool {
	switch ev.Type {
	case "conversation.item.added":
		s.trackToolCall(ev.Item)
		return true
	case "response.function_call_arguments.done":
		s.runToolCall(ctx, ev)
		return true
	default:
		return false
	}
}

// trackToolCall records a tool call the model has announced. The call is
// announced before its arguments finish arriving, so it has to be known to be
// expected when they do.
func (s *Service) trackToolCall(item *conversationItem) {
	if item == nil || item.Type != "function_call" || item.CallID == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.pendingCalls[item.CallID] {
		slog.Warn("the realtime session announced a tool call it had already announced",
			"service", s.Name(), "tool_call_id", item.CallID)
		return
	}
	s.pendingCalls[item.CallID] = true
}

// runToolCall runs a call whose arguments the model has finished writing. The
// session names the function on this event, so only the call has to be tracked.
//
// It runs on the arguments rather than waiting for the response to complete,
// because a response that only calls a tool may never report itself done.
func (s *Service) runToolCall(ctx context.Context, ev serverEvent) {
	s.mu.Lock()
	tracked := s.pendingCalls[ev.CallID]
	// Taken out first, so a repeated event cannot run the same call twice.
	delete(s.pendingCalls, ev.CallID)
	convo := s.convo
	s.mu.Unlock()

	if !tracked {
		slog.WarnContext(ctx, "the realtime session finished a tool call it never announced",
			"service", s.Name(), "tool_call_id", ev.CallID)
		return
	}
	if convo == nil {
		// A call's result is written into the conversation and read back out of
		// it, so without one there is nowhere for the answer to go.
		slog.ErrorContext(ctx, "the model called a tool before any conversation reached the service",
			"service", s.Name(), "function", ev.Name)
		return
	}

	args := json.RawMessage(ev.Arguments)
	if len(args) == 0 {
		args = json.RawMessage("{}")
	}
	call := frames.ToolCall{ID: ev.CallID, Name: ev.Name, Args: args}
	if err := s.RunFunctionCalls(ctx, convo, []frames.ToolCall{call}); err != nil {
		slog.ErrorContext(ctx, "running a tool call failed", "service", s.Name(),
			"function", ev.Name, "err", err)
	}
}

// currentTools returns the function tools currently advertised to the session.
func (s *Service) currentTools() []frames.Tool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.tools
}

// syncTools records the function-calling configuration and, when it differs from
// what the live session was told, pushes a session.update so the continuously
// running model picks the change up. Before the conversation opens it only
// records: the initial session update carries whatever has been recorded by then.
func (s *Service) syncTools(tools []frames.Tool) {
	s.mu.Lock()
	if sameTools(s.tools, tools) {
		s.mu.Unlock()
		return
	}
	s.tools = tools
	live := s.established
	s.mu.Unlock()

	if !live {
		return
	}
	specs := s.toolSpecs(tools)
	if specs == nil {
		// Clearing the toolset still has to reach the model.
		specs = []map[string]any{}
	}
	if err := s.send(sessionUpdateMsg{
		Type:    "session.update",
		Session: map[string]any{"tools": specs},
	}); err != nil {
		slog.Warn("xai realtime tool update failed", "err", err)
	}
}

// sameTools reports whether two toolsets are equivalent for session purposes.
func sameTools(a, b []frames.Tool) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Name != b[i].Name ||
			a[i].Description != b[i].Description ||
			!bytes.Equal(a[i].Parameters, b[i].Parameters) {
			return false
		}
	}
	return true
}

// sendAudio appends a chunk of input PCM to the model's input buffer.
//
// Audio arriving before the server has acknowledged the session is dropped: the
// rate the session reads it at is only settled then. The drop is reported once
// per session, since a caller already talking produces a chunk every few
// milliseconds.
func (s *Service) sendAudio(ctx context.Context, pcm []byte) {
	if len(pcm) == 0 {
		return
	}
	s.mu.Lock()
	ready, logged := s.sessionReady, s.loggedAudioDrop
	if !ready {
		s.loggedAudioDrop = true
	}
	s.mu.Unlock()
	if !ready {
		if !logged {
			slog.DebugContext(ctx, "dropping user audio: the realtime session is not ready yet",
				"service", s.Name())
		}
		return
	}
	_ = s.send(audioAppendMsg{
		Type:  "input_audio_buffer.append",
		Audio: base64.StdEncoding.EncodeToString(pcm),
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
		return errNotConnected
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return conn.Write(connCtx, websocket.MessageText, data)
}

// disconnect cancels the session context, closes the socket, and waits for the
// read loop to exit. It is safe to call more than once.
func (s *Service) disconnect() {
	s.mu.Lock()
	conn, cancel := s.conn, s.cancel
	s.conn, s.cancel, s.connCtx = nil, nil, nil
	s.established = false
	s.sessionReady = false
	s.loggedAudioDrop = false
	s.pendingResponse = false
	s.audioResp = nil
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	if conn != nil {
		_ = conn.Close(websocket.StatusNormalClosure, "")
	}
	s.wg.Wait()
}

// serverEvent is the subset of Realtime server events the service handles. The
// delta field carries base64 PCM for audio events and plain text for transcript
// events. Token accounting arrives on response.done, either at the top level or
// nested in the response.
type serverEvent struct {
	Type       string          `json:"type"`
	Delta      string          `json:"delta"`
	Transcript string          `json:"transcript"`
	Usage      *usage          `json:"usage"`
	Response   *responseObject `json:"response"`
	// CallID, Name and Arguments carry a tool call the model finished naming.
	// The arguments arrive as the JSON text the model wrote, not as a decoded
	// object.
	CallID    string `json:"call_id"` //nolint:tagliatelle // xAI wire field
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	// Item is the conversation item an item event is about. A tool call is
	// announced as one of these before its arguments are finished.
	Item *conversationItem `json:"item"`
	// ItemID and ContentIndex name the piece of the conversation an event is
	// about, which an interruption needs to truncate the audio being played.
	ItemID       string `json:"item_id"` //nolint:tagliatelle // xAI wire field
	ContentIndex int    `json:"content_index"`
	Error        struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

// conversationItem is the subset of a conversation item we read: enough to
// recognize a tool call.
type conversationItem struct {
	Type   string `json:"type"`
	Name   string `json:"name"`
	CallID string `json:"call_id"` //nolint:tagliatelle // xAI wire field
}

// responseObject is the completed-response payload on a response.done event.
type responseObject struct {
	Status string `json:"status"`
	Usage  *usage `json:"usage"`
}

// usage is the per-response token accounting. xAI reports totals only, with no
// per-modality breakdown, so the audio and text token fields stay zero.
type usage struct {
	TotalTokens  int64 `json:"total_tokens"`  //nolint:tagliatelle // xAI wire field
	InputTokens  int64 `json:"input_tokens"`  //nolint:tagliatelle // xAI wire field
	OutputTokens int64 `json:"output_tokens"` //nolint:tagliatelle // xAI wire field
}

// tokenUsage converts the wire accounting into the framework's usage shape.
func (u usage) tokenUsage() frames.LLMTokenUsage {
	return frames.LLMTokenUsage{
		PromptTokens:     u.InputTokens,
		CompletionTokens: u.OutputTokens,
		TotalTokens:      u.TotalTokens,
	}
}

// readLoop reads server events until the connection is closed or canceled.
func (s *Service) readLoop(conn *wsutil.Conn, connCtx context.Context) {
	defer s.wg.Done()
	for {
		_, data, err := conn.Read(connCtx)
		if err != nil {
			if connCtx.Err() == nil {
				slog.Debug("xai realtime read ended", "err", err)
			}
			return
		}
		var ev serverEvent
		if json.Unmarshal(data, &ev) != nil {
			continue
		}
		if stop := s.handleEvent(connCtx, ev); stop {
			return
		}
	}
}

// handleEvent maps a server event onto pipeline frames, reporting whether the
// session has failed in a way that ends the read loop.
func (s *Service) handleEvent(ctx context.Context, ev serverEvent) bool {
	switch {
	case s.handleToolEvent(ctx, ev):
		return false
	case s.handleSessionEvent(ctx, ev):
		return false
	case s.handleResponseEvent(ctx, ev):
		return false
	case s.handleTranscriptEvent(ctx, ev):
		return false
	case ev.Type == "error":
		return s.handleServerError(ctx, ev)
	}
	return false
}

// handleSessionEvent takes the events about the session itself: it opening,
// being configured, and what it hears of the user. It reports whether it took
// the event.
func (s *Service) handleSessionEvent(ctx context.Context, ev serverEvent) bool {
	switch ev.Type {
	case "conversation.created":
		s.configureSession(ctx)
	case "session.updated":
		s.sessionUpdated(ctx)
	case "input_audio_buffer.speech_started":
		s.speechStarted(ctx)
	case "input_audio_buffer.speech_stopped":
		s.speechStopped(ctx)
	default:
		return false
	}
	return true
}

// handleResponseEvent takes the events that make up the model's answer: its
// audio, the transcript of that audio, the same answer in text from a session
// answering in that modality, and the end of the response. It reports whether it
// took the event.
func (s *Service) handleResponseEvent(ctx context.Context, ev serverEvent) bool {
	switch ev.Type {
	case "response.created":
		_ = s.PushFrame(ctx, frames.NewBotStartedSpeakingFrame(), processor.Downstream)
	case "response.output_audio.delta":
		if pcm, err := base64.StdEncoding.DecodeString(ev.Delta); err == nil && len(pcm) > 0 {
			s.trackAudio(ev, len(pcm))
			_ = s.PushFrame(ctx, frames.NewTTSAudioRawFrame(pcm, s.cfg.sampleRate(), 1), processor.Downstream)
		}
	case "response.output_audio_transcript.delta", "response.output_text.delta", "response.text.delta":
		if ev.Delta != "" {
			_ = s.PushFrame(ctx, frames.NewLLMTextFrame(ev.Delta), processor.Downstream)
		}
	case "response.done":
		s.finishResponse(ctx, ev)
	default:
		return false
	}
	return true
}

// handleTranscriptEvent takes what the session heard the user say, reporting
// whether it took the event.
//
// A transcript travels upstream, where the user aggregator is: it is the user's
// message, and everything downstream of this service is the reply to it. The
// updated event carries the turn so far and the completed one the whole of it.
func (s *Service) handleTranscriptEvent(ctx context.Context, ev serverEvent) bool {
	text := strings.TrimSpace(ev.Transcript)
	switch ev.Type {
	case "conversation.item.input_audio_transcription.updated":
		if text != "" {
			_ = s.PushFrame(ctx, frames.NewInterimTranscriptionFrame(text, "", ""), processor.Upstream)
		}
	case "conversation.item.input_audio_transcription.completed":
		if text != "" {
			_ = s.PushFrame(ctx, frames.NewTranscriptionFrame(text, "", ""), processor.Upstream)
		}
	default:
		return false
	}
	return true
}

// handleServerError reports an error event, saying whether it ends the session.
//
// A canceled response that had already finished, and a truncation of an item
// that is no longer playing, are races the interruption path runs into by
// design: they say the model had already stopped, which is what was wanted. They
// are logged and the session carries on. Anything else is reported and ends the
// read loop, since the session cannot be relied on afterwards.
func (s *Service) handleServerError(ctx context.Context, ev serverEvent) bool {
	if benignSessionError(ev) {
		slog.DebugContext(ctx, "the realtime session reported a benign error",
			"service", s.Name(), "code", ev.Error.Code, "message", ev.Error.Message)
		return false
	}
	s.PushError(ctx, "xai realtime error: "+ev.Error.Message,
		fmt.Errorf("%w: %s", errServer, ev.Error.Message), false)
	return true
}

// benignSessionError reports whether an error event is one of the races the
// interruption path expects. xAI's codes for these differ from the ones the
// Realtime protocol documents elsewhere, so the message is matched as well.
func benignSessionError(ev serverEvent) bool {
	switch ev.Error.Code {
	case "response_cancel_not_active", "conversation_already_has_active_response":
		return true
	}
	msg := strings.ToLower(ev.Error.Message)
	return strings.Contains(msg, "no active response") ||
		strings.Contains(msg, "already has an active response") ||
		strings.Contains(msg, "truncat")
}

// sessionUpdated records that the server has taken the session configuration,
// which is the point from which it reads audio at the configured rate. A
// response asked for while it was still settling is asked for now.
func (s *Service) sessionUpdated(ctx context.Context) {
	s.mu.Lock()
	s.sessionReady = true
	pending := s.pendingResponse
	s.pendingResponse = false
	s.mu.Unlock()
	if pending {
		s.createResponse(ctx)
	}
}

// speechStarted takes the session's report that the user has begun speaking.
//
// The boundary is proposed rather than announced: the external turn strategies
// the service recommends decide the turn from it and broadcast the interruption,
// which comes back here and stops the model on the wire. Under manual turn
// detection the pipeline owns the boundary and the session's opinion is ignored.
func (s *Service) speechStarted(ctx context.Context) {
	if !s.cfg.serverVAD() {
		return
	}
	// The model has stopped speaking as far as the caller is concerned, so its
	// record of the turn is cut back to what was heard before the interruption
	// the strategies are about to broadcast arrives.
	s.truncateAudioResponse(ctx)
	_ = s.Broadcast(ctx, func() frames.Frame { return frames.NewProposedUserStartedSpeakingFrame() })
}

// speechStopped takes the session's report that the user has stopped speaking,
// proposing the end of the turn for the strategies to resolve.
func (s *Service) speechStopped(ctx context.Context) {
	if !s.cfg.serverVAD() {
		return
	}
	_ = s.Broadcast(ctx, func() frames.Frame { return frames.NewProposedUserStoppedSpeakingFrame() })
}

// configureSession sends the session configuration once the server has opened
// the conversation, which is the point from which xAI accepts it. Later tool
// changes push their own updates.
func (s *Service) configureSession(ctx context.Context) {
	s.mu.Lock()
	s.established = true
	s.mu.Unlock()
	if err := s.send(s.sessionUpdate()); err != nil {
		s.PushError(ctx, "xai realtime session update failed", err, false)
	}
}

// finishResponse closes out a completed response: it reports the token
// accounting, ends the bot's turn, and surfaces a response the model failed to
// produce.
func (s *Service) finishResponse(ctx context.Context, ev serverEvent) {
	s.reportUsage(ctx, ev)
	// The response is over, so there is no longer an item an interruption would
	// cut back.
	s.mu.Lock()
	s.audioResp = nil
	s.mu.Unlock()
	_ = s.PushFrame(ctx, frames.NewBotStoppedSpeakingFrame(), processor.Downstream)
	if ev.Response != nil && ev.Response.Status == "failed" {
		s.PushError(ctx, "xai realtime response failed", errServer, false)
	}
}

// reportUsage forwards the token accounting on a response.done event to metrics
// and telemetry, when usage metrics are enabled. xAI reports it at the top level
// on some responses and inside the response object on others.
func (s *Service) reportUsage(ctx context.Context, ev serverEvent) {
	if !s.UsageMetricsEnabled() {
		return
	}
	u := ev.Usage
	if u == nil && ev.Response != nil {
		u = ev.Response.Usage
	}
	if u == nil || u.TotalTokens == 0 {
		return
	}
	_ = s.PushTokenUsage(ctx, u.tokenUsage())
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
//
// A session detecting the turns recommends the external strategies that resolve
// the boundaries it proposes. Without them a proposal reaches nothing and no
// turn is ever opened or closed.
func (s *Service) ServiceMetadataFrame() frames.ServiceMetadata {
	f := frames.NewLLMServiceMetadataFrame(s.Name())
	f.Realtime = true
	if s.cfg.serverVAD() {
		f.UserTurnStrategies = turns.ExternalStrategies(turns.ExternalStrategiesConfig{})
	}
	return f
}
