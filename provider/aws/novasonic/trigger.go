package novasonic

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/binary"
	"log/slog"
	"strings"
	"time"
)

// The original Nova Sonic model does not start speaking on a text prompt: it
// starts when it hears the user. The assistant response trigger plays it a
// recording of the user saying "ready", with an instruction to start speaking
// when it hears that. Nova 2 Sonic and later answer a conversation that ends
// with a user message, so they need none.

// AwaitTriggerAssistantResponseInstruction is the instruction to add to the
// system instruction of the original Nova Sonic model when the conversation is
// started with TriggerAssistantResponse.
const AwaitTriggerAssistantResponseInstruction = "Start speaking when you hear the user say 'ready', " +
	"but don't consider that 'ready' to be a meaningful part of the conversation other than as a " +
	"trigger for you to start speaking."

const (
	// triggerChunkDuration is how much audio the trigger sends at a time, what an
	// input transport would deliver.
	triggerChunkDuration = 20 * time.Millisecond
	// maxBlankAudioDuration is how much silence leads the trigger on a session
	// just connected: the model does not quite hear the first audio a session
	// is sent.
	maxBlankAudioDuration = 500 * time.Millisecond
)

//go:embed ready.wav
var readyWAV []byte

// readyAudio is the PCM of the trigger recording.
func readyAudio() []byte { return wavData(readyWAV) }

// wavData is the sample data of a WAV file, or nil when it has none.
func wavData(wav []byte) []byte {
	const headerSize = 12
	if len(wav) < headerSize || string(wav[0:4]) != "RIFF" || string(wav[8:12]) != "WAVE" {
		return nil
	}
	for rest := wav[headerSize:]; len(rest) >= 8; {
		id := string(rest[0:4])
		size := int(binary.LittleEndian.Uint32(rest[4:8]))
		body := rest[8:]
		if size > len(body) {
			size = len(body)
		}
		if id == "data" {
			return bytes.Clone(body[:size])
		}
		// Chunks are padded to an even size.
		next := size + size%2
		if next > len(body) {
			return nil
		}
		rest = body[next:]
	}
	return nil
}

// normalizedTrigger reports whether a transcript is the trigger's.
func normalizedTrigger(text string) bool { return strings.ToLower(strings.TrimSpace(text)) == "ready" }

// TriggerAssistantResponse starts the original Nova Sonic model speaking by
// playing it the user saying "ready". It is sent once the session is set up,
// right away when it already is. Nova 2 Sonic and later need none: a
// conversation ending with a user message, run with an LLMRunFrame, starts them.
func (s *Service) TriggerAssistantResponse(ctx context.Context) {
	if !assistantResponseTriggerNeeded(s.model()) {
		slog.WarnContext(ctx, "the assistant response trigger is not needed for this model; skipping. "+
			"An LLMRunFrame should be enough to prompt the assistant to respond, "+
			"assuming the conversation ends in a user message", "model", s.model())
		return
	}
	s.mu.Lock()
	if s.triggeringAssistantResponse {
		s.mu.Unlock()
		return
	}
	s.triggeringAssistantResponse = true
	connected := !s.connectedTime.IsZero()
	s.mu.Unlock()
	if connected {
		s.sendAssistantResponseTrigger(ctx)
	}
}

// sendAssistantResponseTrigger plays the trigger recording to the session, in
// chunks at the pace of real audio, led by silence on a session just
// connected. The recording is transcribed into the conversation, alongside the
// instruction to wait for it.
func (s *Service) sendAssistantResponseTrigger(ctx context.Context) {
	s.mu.Lock()
	connectedTime := s.connectedTime
	if connectedTime.IsZero() {
		s.mu.Unlock()
		return
	}
	s.waitingForTriggerTranscription = true
	s.mu.Unlock()
	defer func() {
		// Cleared even when cut short, as by the pipeline stopping.
		s.mu.Lock()
		s.triggeringAssistantResponse = false
		s.mu.Unlock()
	}()
	slog.DebugContext(ctx, "sending the assistant response trigger", "service", s.Name())

	ac := s.audio
	chunkSize := int(triggerChunkDuration.Seconds() * float64(ac.InputSampleRate) *
		float64(ac.InputChannelCount) * float64(ac.InputSampleSize) / 8)
	if chunkSize <= 0 {
		return
	}

	if since := time.Since(connectedTime); since < maxBlankAudioDuration {
		blank := maxBlankAudioDuration - since
		slog.DebugContext(ctx, "leading the assistant response trigger with silence", "service", s.Name(),
			"duration", blank)
		silence := make([]byte, chunkSize)
		for range int(blank / triggerChunkDuration) {
			s.sendUserAudioEvent(ctx, silence)
			if !sleep(ctx, triggerChunkDuration) {
				return
			}
		}
	}
	for i := 0; i < len(s.triggerAudio); i += chunkSize {
		s.sendUserAudioEvent(ctx, s.triggerAudio[i:min(i+chunkSize, len(s.triggerAudio))])
		if !sleep(ctx, triggerChunkDuration) {
			return
		}
	}
}

// sleep waits d, reporting false when ctx ends first.
func sleep(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-ctx.Done():
		return false
	}
}
