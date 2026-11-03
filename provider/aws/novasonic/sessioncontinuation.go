package novasonic

import (
	"context"
	"log/slog"
	"slices"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/google/uuid"
)

// Session continuation rotates Nova Sonic sessions before the AWS time limit
// (about eight minutes), preserving the conversation with no gap the user can
// hear. Past a threshold, the next time the assistant starts speaking the next
// session is opened in the background and set up, the user's audio is buffered,
// and once the assistant has finished the conversation history and the buffered
// audio are handed to the new session, which takes over. It follows the AWS
// reference architecture, the session-continuation pattern of
// github.com/aws-samples/amazon-nova-samples (speech-to-speech,
// amazon-nova-2-sonic, repeatable-patterns).

// The limits on the history handed to the next session.
const (
	maxHistoryMessageBytes = 50 * 1024
	maxHistoryTotalBytes   = 200 * 1024
	// nextSessionTimeout is how long a prepared next session may wait unused
	// before it is replaced by a fresh one.
	nextSessionTimeout = 30 * time.Second
	// monitorInterval is how often the session's age is checked.
	monitorInterval = time.Second
	// The bounds on closing a session that has been handed over.
	oldSessionInputCloseTimeout = 2 * time.Second
	oldSessionReceiveTimeout    = 5 * time.Second
)

// SessionContinuationParams configures how sessions are rotated before the
// AWS time limit.
type SessionContinuationParams struct {
	// Enabled turns session continuation on.
	Enabled bool
	// TransitionThreshold is how long into a session to start looking for a
	// moment to hand over: the next time the assistant starts speaking.
	TransitionThreshold time.Duration
	// AudioBufferDuration is how much of the user's audio is buffered during the
	// handover, to be replayed into the new session so none is lost.
	AudioBufferDuration time.Duration
	// AudioStartTimeout is how long to wait for the assistant to start speaking
	// once the threshold is reached before handing over anyway. Zero waits
	// indefinitely.
	AudioStartTimeout time.Duration
}

// DefaultSessionContinuationParams turns session continuation on, looking to
// hand over six minutes into a session, buffering three seconds of audio, and
// handing over anyway after eighty seconds without the assistant speaking.
func DefaultSessionContinuationParams() SessionContinuationParams {
	return SessionContinuationParams{
		Enabled:             true,
		TransitionThreshold: 360 * time.Second,
		AudioBufferDuration: 3 * time.Second,
		AudioStartTimeout:   80 * time.Second,
	}
}

// sessionSender is the protocol a session continuation drives, addressing the
// next session as well as the current one. The service implements it.
type sessionSender interface {
	createClient(ctx context.Context) (sessionClient, error)
	openStream(ctx context.Context, client sessionClient) (sessionStream, error)
	sessionStartEvent() map[string]any
	sendPromptStart(ctx context.Context, tools []map[string]any, prompt string, stream sessionStream) error
	sendText(ctx context.Context, text, role, prompt string, stream sessionStream, interactive bool) error
	sendAudioInputStart(ctx context.Context, prompt, content string, stream sessionStream) error
	sendAudio(ctx context.Context, audio []byte, prompt, content string, stream sessionStream) error
	audioConfig() AudioConfig
	setupParams(ctx context.Context) (string, []map[string]any)
}

// nextSession is a session opened and set up ahead of the handover.
type nextSession struct {
	client           sessionClient
	stream           sessionStream
	promptName       string
	audioContentName string
}

// historyMessage is one message of the conversation as the sessions heard it.
type historyMessage struct {
	role string
	text string
}

// sessionContinuation holds the state of rotating sessions. The service calls
// it at the points of the protocol that decide a handover; each returns whether
// the handover should run now.
type sessionContinuation struct {
	params SessionContinuationParams
	sender sessionSender
	// handoff sets the handover going in the background, swapping the next
	// session in; the service sets it.
	handoff func()
	// now and tick are the clock and the monitor's interval, replaced in tests.
	now  func() time.Time
	tick time.Duration

	mu                   sync.Mutex
	audioBuffer          [][]byte
	audioBufferMaxBytes  int
	next                 *nextSession
	buffering            bool
	waitingForAudioStart bool
	waitingForCompletion bool
	handoffInProgress    bool
	audioStartWaitTime   time.Time
	nextSessionCreatedAt time.Time
	monitorCancel        context.CancelFunc
	monitorDone          chan struct{}
	connectedTime        time.Time
	wantsConnection      bool
	speculativeTextCount int
	finalTextCount       int
	bargeInDetected      bool
	conversationHistory  []historyMessage
}

// newSessionContinuation builds the helper, sizing the audio buffer from the
// sender's audio.
func newSessionContinuation(params SessionContinuationParams, sender sessionSender) *sessionContinuation {
	ac := sender.audioConfig()
	return &sessionContinuation{
		params: params,
		sender: sender,
		now:    time.Now,
		tick:   monitorInterval,
		audioBufferMaxBytes: int(params.AudioBufferDuration.Seconds() *
			float64(ac.InputSampleRate) * float64(ac.InputSampleSize) / 8 * float64(ac.InputChannelCount)),
	}
}

// isBuffering reports whether the user's audio is being buffered for the
// handover.
func (c *sessionContinuation) isBuffering() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.buffering
}

// inHandoff reports whether a handover is under way.
func (c *sessionContinuation) inHandoff() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.handoffInProgress
}

// setConnected records that a session has finished connecting, and starts the
// per-session counts over.
func (c *sessionContinuation) setConnected(at time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.connectedTime = at
	c.wantsConnection = true
	c.speculativeTextCount = 0
	c.finalTextCount = 0
	c.bargeInDetected = false
}

// setDisconnected records that the session has disconnected.
func (c *sessionContinuation) setDisconnected() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.wantsConnection = false
	c.connectedTime = time.Time{}
}

// seedHistory adds a message the session was seeded with to the history.
func (c *sessionContinuation) seedHistory(role, text string) {
	if text == "" {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.conversationHistory = append(c.conversationHistory, historyMessage{role: role, text: text})
}

// startMonitor starts watching the session's age.
func (c *sessionContinuation) startMonitor() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.params.Enabled || c.monitorCancel != nil {
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	c.monitorCancel, c.monitorDone = cancel, done
	go func() {
		defer close(done)
		c.monitorLoop(ctx)
	}()
}

// stopMonitor stops watching the session's age.
func (c *sessionContinuation) stopMonitor() {
	c.mu.Lock()
	cancel, done := c.monitorCancel, c.monitorDone
	c.monitorCancel, c.monitorDone = nil, nil
	c.mu.Unlock()
	if cancel == nil {
		return
	}
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
	}
}

// onAudioInput buffers a chunk of the user's audio during the handover, keeping
// the most recent within the buffer's duration.
func (c *sessionContinuation) onAudioInput(audio []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.buffering {
		return
	}
	c.audioBuffer = append(c.audioBuffer, append([]byte(nil), audio...))
	total := 0
	for _, chunk := range c.audioBuffer {
		total += len(chunk)
	}
	for total > c.audioBufferMaxBytes && len(c.audioBuffer) > 0 {
		total -= len(c.audioBuffer[0])
		c.audioBuffer = c.audioBuffer[1:]
	}
}

// onAssistantAudioStarted is called when the assistant starts an audio block.
// Past the threshold, it starts buffering the user's audio and prepares the
// next session. It reports whether the handover was set going.
func (c *sessionContinuation) onAssistantAudioStarted(ctx context.Context) bool {
	c.mu.Lock()
	if !c.waitingForAudioStart || c.handoffInProgress {
		c.mu.Unlock()
		return false
	}
	c.waitingForAudioStart = false
	c.audioStartWaitTime = time.Time{}
	c.buffering = true
	c.waitingForCompletion = true
	hasNext := c.next != nil
	c.mu.Unlock()

	slog.InfoContext(ctx, "session continuation: assistant audio started, "+
		"buffering user audio and creating the next session")
	if !hasNext {
		if err := c.prepareNextSession(ctx); err != nil {
			slog.ErrorContext(ctx, "session continuation: failed to prepare the next session", "err", err)
			c.mu.Lock()
			c.buffering = false
			c.waitingForCompletion = false
			c.mu.Unlock()
			return false
		}
	}
	return true
}

// onTextOutput counts the assistant's speculative and final text, whatever the
// state of the handover.
func (c *sessionContinuation) onTextOutput(role, stage string) {
	if role != string(roleAssistant) {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	switch stage {
	case stageSpeculative:
		c.speculativeTextCount++
	case stageFinal:
		c.finalTextCount++
	}
}

// onContentEndAssistantFinalText adds the assistant's final text to the history
// and reports whether it completes the response being waited on: as much final
// text as speculative.
func (c *sessionContinuation) onContentEndAssistantFinalText(text string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if text != "" {
		c.conversationHistory = append(c.conversationHistory, historyMessage{role: string(roleAssistant), text: text})
	}
	if c.waitingForCompletion && c.speculativeTextCount > 0 && c.finalTextCount > 0 &&
		c.finalTextCount >= c.speculativeTextCount && !c.handoffInProgress {
		slog.Info("session continuation: completion signal, the text pairs matched",
			"final", c.finalTextCount, "speculative", c.speculativeTextCount)
		c.waitingForCompletion = false
		return true
	}
	return false
}

// onContentEndTextInterrupted records a barge-in, which completes the response
// being waited on.
func (c *sessionContinuation) onContentEndTextInterrupted() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.bargeInDetected = true
	if c.waitingForCompletion && !c.handoffInProgress {
		slog.Info("session continuation: completion signal, text interrupted (barge-in)")
		c.waitingForCompletion = false
		return true
	}
	return false
}

// onContentEndUserFinalText adds the user's final text to the history. After a
// barge-in, the final text the interrupted response still owed will never come,
// so the counts are reconciled.
func (c *sessionContinuation) onContentEndUserFinalText(text string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if text != "" {
		c.conversationHistory = append(c.conversationHistory, historyMessage{role: string(roleUser), text: text})
	}
	if c.bargeInDetected && c.speculativeTextCount > c.finalTextCount {
		slog.Info("session continuation: the user spoke after a barge-in, reconciling the text counts",
			"final", c.finalTextCount, "speculative", c.speculativeTextCount)
		c.finalTextCount = c.speculativeTextCount
	}
}

// onUserContentStarted is called when the user starts a content block. During a
// handover it is a barge-in, and one before the assistant has said anything
// completes the handover at once.
func (c *sessionContinuation) onUserContentStarted() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.waitingForCompletion && c.next != nil {
		c.bargeInDetected = true
	}
	if c.waitingForCompletion && !c.handoffInProgress && c.next != nil && c.finalTextCount == 0 {
		slog.Info("session continuation: the user spoke during a forced transition " +
			"(no assistant response yet), completing the handover now")
		c.waitingForCompletion = false
		return true
	}
	return false
}

// onCompletionEnd is the fallback completion signal.
func (c *sessionContinuation) onCompletionEnd() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.waitingForCompletion && !c.handoffInProgress {
		slog.Info("session continuation: completion signal, completion end (fallback)")
		c.waitingForCompletion = false
		return true
	}
	return false
}

// executeHandoff hands the conversation to the next session: the recent
// history, the audio block opened, and the buffered audio. promote swaps the
// next session in; it runs before the handover is marked over, so no user audio
// slips between the two sessions. It reports whether the handover went through.
func (c *sessionContinuation) executeHandoff(ctx context.Context, promote func(*nextSession)) bool {
	c.mu.Lock()
	if c.handoffInProgress {
		c.mu.Unlock()
		return false
	}
	c.handoffInProgress = true
	ns := c.next
	history := append([]historyMessage(nil), c.conversationHistory...)
	c.mu.Unlock()

	defer func() {
		c.mu.Lock()
		c.handoffInProgress = false
		c.waitingForAudioStart = false
		c.waitingForCompletion = false
		c.audioStartWaitTime = time.Time{}
		c.nextSessionCreatedAt = time.Time{}
		c.mu.Unlock()
	}()

	if ns == nil {
		slog.WarnContext(ctx, "session continuation: no next session available for the handover")
		return false
	}
	slog.InfoContext(ctx, "session continuation: executing the handover")

	prepared, totalBytes := trimHistory(history)
	if len(prepared) > 0 {
		slog.InfoContext(ctx, "session continuation: sending the history to the new session",
			"messages", len(prepared), "bytes", totalBytes)
	}
	for _, m := range prepared {
		if err := c.sender.sendText(ctx, m.text, m.role, ns.promptName, ns.stream, false); err != nil {
			return c.failHandoff(ctx, err)
		}
	}
	if err := c.sender.sendAudioInputStart(ctx, ns.promptName, ns.audioContentName, ns.stream); err != nil {
		return c.failHandoff(ctx, err)
	}

	c.mu.Lock()
	buffered := c.audioBuffer
	c.mu.Unlock()
	slog.InfoContext(ctx, "session continuation: sending the buffered audio to the new session",
		"chunks", len(buffered))
	for _, chunk := range buffered {
		if err := c.sender.sendAudio(ctx, chunk, ns.promptName, ns.audioContentName, ns.stream); err != nil {
			return c.failHandoff(ctx, err)
		}
	}

	slog.InfoContext(ctx, "session continuation: promoting the new session")
	c.mu.Lock()
	c.next = nil
	c.buffering = false
	c.audioBuffer = nil
	c.mu.Unlock()
	promote(ns)
	return true
}

// failHandoff gives up on a handover that failed, closing the next session.
func (c *sessionContinuation) failHandoff(ctx context.Context, err error) bool {
	slog.ErrorContext(ctx, "session continuation: handover error", "err", err)
	c.cleanupNextSession()
	return false
}

// trimHistory is the history to hand over: the most recent messages, each cut to
// the per-message limit, within the total limit, starting with the user's.
func trimHistory(history []historyMessage) ([]historyMessage, int) {
	var prepared []historyMessage
	total := 0
	for _, m := range slices.Backward(history) {
		text := m.text
		if len(text) > maxHistoryMessageBytes {
			text = text[:maxHistoryMessageBytes]
			for len(text) > 0 && !utf8.ValidString(text) {
				text = text[:len(text)-1]
			}
		}
		if total+len(text) > maxHistoryTotalBytes {
			break
		}
		total += len(text)
		prepared = append(prepared, historyMessage{role: m.role, text: text})
	}
	// Oldest first.
	for i, j := 0, len(prepared)-1; i < j; i, j = i+1, j-1 {
		prepared[i], prepared[j] = prepared[j], prepared[i]
	}
	for len(prepared) > 0 && prepared[0].role != string(roleUser) {
		prepared = prepared[1:]
	}
	return prepared, total
}

// closeOldSession closes a session that has been handed over, in the
// background: the audio block, the prompt and the session are closed, then the
// stream's input, and its receive loop is given time to finish.
func (c *sessionContinuation) closeOldSession(ctx context.Context, stream sessionStream,
	receiveDone <-chan struct{}, prompt, audioContent string,
) {
	if stream != nil && prompt != "" {
		if audioContent != "" {
			_ = sendEvent(ctx, stream, contentEnd(prompt, audioContent))
		}
		_ = sendEvent(ctx, stream, promptEnd(prompt))
		_ = sendEvent(ctx, stream, sessionEnd())
	}
	if stream != nil {
		closed := make(chan struct{})
		go func() {
			_ = stream.CloseInput()
			close(closed)
		}()
		select {
		case <-closed:
		case <-time.After(oldSessionInputCloseTimeout):
		}
	}
	if receiveDone != nil {
		select {
		case <-receiveDone:
		case <-time.After(oldSessionReceiveTimeout):
		}
	}
	// The stream is released once its receive loop is done with it.
	if stream != nil {
		_ = stream.Close()
	}
	slog.DebugContext(ctx, "session continuation: old session closed")
}

// cleanupNextSession closes the prepared next session, if there is one.
func (c *sessionContinuation) cleanupNextSession() {
	c.mu.Lock()
	ns := c.next
	c.next = nil
	c.buffering = false
	c.audioBuffer = nil
	c.nextSessionCreatedAt = time.Time{}
	c.mu.Unlock()
	if ns != nil && ns.stream != nil {
		_ = ns.stream.Close()
	}
}

// monitorLoop checks the session's age every tick: past the threshold it starts
// waiting for the assistant to speak, hands over anyway when it does not within
// the timeout, and replaces a prepared session left unused too long.
func (c *sessionContinuation) monitorLoop(ctx context.Context) {
	ticker := time.NewTicker(c.tick)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
		c.mu.Lock()
		wants := c.wantsConnection
		c.mu.Unlock()
		if !wants {
			return
		}
		c.checkSession(ctx)
	}
}

// checkSession is one tick of the monitor.
func (c *sessionContinuation) checkSession(ctx context.Context) {
	c.mu.Lock()
	if c.connectedTime.IsZero() || c.handoffInProgress {
		c.mu.Unlock()
		return
	}
	now := c.now()
	c.startWaitingIfDueLocked(ctx, now)
	forced := c.forceIfTimedOutLocked(ctx, now)
	hasNext := c.next != nil
	c.mu.Unlock()

	if forced {
		c.forceTransition(ctx, hasNext)
	}
	c.replaceStaleNextSession(ctx)
}

// startWaitingIfDueLocked starts waiting for the assistant to speak once the
// session reaches the threshold. The caller holds c.mu.
func (c *sessionContinuation) startWaitingIfDueLocked(ctx context.Context, now time.Time) {
	age := now.Sub(c.connectedTime)
	if age < c.params.TransitionThreshold || c.waitingForAudioStart || c.next != nil || c.waitingForCompletion {
		return
	}
	slog.InfoContext(ctx, "session continuation: the session reached the threshold, "+
		"waiting for the assistant to speak", "age", age, "threshold", c.params.TransitionThreshold)
	c.waitingForAudioStart = true
	c.audioStartWaitTime = now
}

// forceIfTimedOutLocked reports whether the assistant has not spoken within the
// timeout, setting the forced transition going. The caller holds c.mu.
func (c *sessionContinuation) forceIfTimedOutLocked(ctx context.Context, now time.Time) bool {
	timeout := c.params.AudioStartTimeout
	if !c.waitingForAudioStart || c.audioStartWaitTime.IsZero() || timeout <= 0 ||
		now.Sub(c.audioStartWaitTime) <= timeout {
		return false
	}
	slog.InfoContext(ctx, "session continuation: the assistant did not speak in time, forcing the transition",
		"timeout", timeout)
	c.waitingForAudioStart = false
	c.audioStartWaitTime = time.Time{}
	c.buffering = true
	c.waitingForCompletion = false
	return true
}

// forceTransition prepares the next session if there is none and hands over
// through the service, which swaps the new session in.
func (c *sessionContinuation) forceTransition(ctx context.Context, hasNext bool) {
	if !hasNext {
		if err := c.prepareNextSession(ctx); err != nil {
			slog.ErrorContext(ctx, "session continuation: the forced transition failed", "err", err)
			c.mu.Lock()
			c.buffering = false
			c.mu.Unlock()
			return
		}
	}
	c.handoff()
}

// replaceStaleNextSession replaces a prepared session left unused too long.
func (c *sessionContinuation) replaceStaleNextSession(ctx context.Context) {
	c.mu.Lock()
	stale := c.next != nil && !c.nextSessionCreatedAt.IsZero() && !c.handoffInProgress &&
		c.now().Sub(c.nextSessionCreatedAt) > nextSessionTimeout
	c.mu.Unlock()
	if !stale {
		return
	}
	slog.WarnContext(ctx, "session continuation: the next session was idle too long, recreating it",
		"timeout", nextSessionTimeout)
	c.cleanupNextSession()
	if err := c.prepareNextSession(ctx); err != nil {
		slog.ErrorContext(ctx, "session continuation: failed to recreate the next session", "err", err)
	}
}

// prepareNextSession opens the next session and sets it up as far as the system
// instruction. The history and the audio wait for the handover.
func (c *sessionContinuation) prepareNextSession(ctx context.Context) error {
	prompt := uuid.NewString()
	audioContent := uuid.NewString()
	client, err := c.sender.createClient(ctx)
	if err != nil {
		return err
	}
	stream, err := c.sender.openStream(ctx, client)
	if err != nil {
		return err
	}
	ns := &nextSession{client: client, stream: stream, promptName: prompt, audioContentName: audioContent}
	c.mu.Lock()
	c.next = ns
	c.nextSessionCreatedAt = c.now()
	c.mu.Unlock()

	if err := sendEvent(ctx, stream, c.sender.sessionStartEvent()); err != nil {
		return err
	}
	instruction, tools := c.sender.setupParams(ctx)
	if err := c.sender.sendPromptStart(ctx, tools, prompt, stream); err != nil {
		return err
	}
	if instruction != "" {
		if err := c.sender.sendText(ctx, instruction, string(roleSystem), prompt, stream, false); err != nil {
			return err
		}
	}
	slog.DebugContext(ctx, "session continuation: next session prepared", "prompt", prompt[:8])
	return nil
}
