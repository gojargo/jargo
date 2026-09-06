package realtime_test

import (
	"encoding/base64"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/xai/realtime"
)

// ready is the event script a session answers a connection with: the server
// opens the conversation, then acknowledges the configuration the service sends
// back, which is the point from which the session can be spoken to.
func ready(extra ...string) []string {
	return append([]string{
		`{"type":"conversation.created"}`,
		`{"type":"session.updated"}`,
	}, extra...)
}

// playing scripts a session that has started speaking, so there is an assistant
// audio item an interruption can cut back.
func playing() []string {
	audio := base64.StdEncoding.EncodeToString(make([]byte, 48000)) // 1s at 24 kHz
	return ready(
		`{"type":"response.created"}`,
		`{"type":"response.output_audio.delta","item_id":"item-audio","content_index":0,"delta":"`+audio+`"}`,
	)
}

// TestServerVADInterruptionCancelsTheResponse checks an interruption stops the
// model on the wire. Without it the session goes on speaking into a call where
// the caller has already been told the bot stopped.
func TestServerVADInterruptionCancelsTheResponse(t *testing.T) {
	srv := newFakeRealtime(t, ready())
	task, done, _ := run(t, realtime.Config{APIKey: "k", BaseURL: srv.wsURL()})
	srv.awaitMessage(t, "session.update")

	task.QueueFrame(frames.NewInterruptionFrame())
	srv.awaitMessage(t, "response.cancel")

	// The buffer holds the speech that interrupted, so clearing it would throw
	// away the words the user is in the middle of saying.
	for _, m := range srv.messages() {
		if m["type"] == "input_audio_buffer.clear" {
			t.Error("the input buffer was cleared while xAI was detecting the turns")
		}
	}

	task.StopWhenDone()
	<-done
}

// TestManualTurnInterruptionClearsAndCancels checks that with the pipeline
// detecting turns, an interruption also drops what has been appended: nothing
// else owns that buffer.
func TestManualTurnInterruptionClearsAndCancels(t *testing.T) {
	serverVAD := false
	srv := newFakeRealtime(t, ready())
	task, done, _ := run(t, realtime.Config{APIKey: "k", BaseURL: srv.wsURL(), ServerVAD: &serverVAD})
	srv.awaitMessage(t, "session.update")

	task.QueueFrame(frames.NewInterruptionFrame())
	srv.awaitMessage(t, "input_audio_buffer.clear")
	srv.awaitMessage(t, "response.cancel")

	task.StopWhenDone()
	<-done
}

// TestInterruptionTruncatesTheAudioBeingPlayed checks the session's record of
// the turn is cut back to what the caller heard. Left whole, the model believes
// it said an answer the user cut off after a word, and answers what it never
// said next.
func TestInterruptionTruncatesTheAudioBeingPlayed(t *testing.T) {
	srv := newFakeRealtime(t, playing())
	task, done, collected := run(t, realtime.Config{APIKey: "k", BaseURL: srv.wsURL()})
	srv.awaitMessage(t, "session.update")
	waitFor(t, func() bool { return hasFrame[*frames.TTSAudioRawFrame](collected()) })

	task.QueueFrame(frames.NewInterruptionFrame())
	truncate := srv.awaitMessage(t, "conversation.item.truncate")

	if truncate["item_id"] != "item-audio" {
		t.Errorf("item_id = %v, want the item being played", truncate["item_id"])
	}
	if truncate["content_index"] != float64(0) {
		t.Errorf("content_index = %v, want 0", truncate["content_index"])
	}
	// A second of audio arrived, so the cut is however much of it had played:
	// somewhere between nothing and the whole second.
	end, ok := truncate["audio_end_ms"].(float64)
	if !ok || end < 0 || end > 1000 {
		t.Errorf("audio_end_ms = %v, want the played part of one second", truncate["audio_end_ms"])
	}

	task.StopWhenDone()
	<-done
}

// TestSpeechStartedTruncatesTheAudioBeingPlayed checks the cut is made where the
// session reports the user speaking, not only where the pipeline reports the
// interruption: the strategies decide the turn from the proposal, so the
// interruption arrives afterwards.
func TestSpeechStartedTruncatesTheAudioBeingPlayed(t *testing.T) {
	srv := newFakeRealtime(t, append(playing(), `{"type":"input_audio_buffer.speech_started"}`))
	task, done, _ := run(t, realtime.Config{APIKey: "k", BaseURL: srv.wsURL()})

	truncate := srv.awaitMessage(t, "conversation.item.truncate")
	if truncate["item_id"] != "item-audio" {
		t.Errorf("item_id = %v, want the item being played", truncate["item_id"])
	}

	task.StopWhenDone()
	<-done
}

// TestUserAudioWaitsForTheSessionToBeReady checks audio is dropped until the
// server has acknowledged the configuration. The rate the session reads it at is
// only settled then, so audio sent earlier is heard at the wrong speed.
func TestUserAudioWaitsForTheSessionToBeReady(t *testing.T) {
	srv := newFakeRealtime(t, []string{`{"type":"conversation.created"}`})
	task, done, _ := run(t, realtime.Config{APIKey: "k", BaseURL: srv.wsURL()})
	srv.awaitMessage(t, "session.update")

	task.QueueFrame(frames.NewInputAudioRawFrame([]byte{1, 2}, 24000, 1))
	time.Sleep(200 * time.Millisecond)
	for _, m := range srv.messages() {
		if m["type"] == "input_audio_buffer.append" {
			t.Fatal("audio reached the session before it was ready")
		}
	}

	task.StopWhenDone()
	<-done
}

// TestManualTurnCommitsAndAsksForAResponse checks that with the pipeline
// detecting turns, the end of the user's turn commits what was appended and asks
// the model to answer. Nothing else would: the session is not listening for the
// boundary itself.
func TestManualTurnCommitsAndAsksForAResponse(t *testing.T) {
	serverVAD := false
	srv := newFakeRealtime(t, ready())
	task, done, _ := run(t, realtime.Config{APIKey: "k", BaseURL: srv.wsURL(), ServerVAD: &serverVAD})
	srv.awaitMessage(t, "session.update")

	task.QueueFrame(frames.NewUserStoppedSpeakingFrame())
	srv.awaitMessage(t, "input_audio_buffer.commit")
	srv.awaitMessage(t, "response.create")

	task.StopWhenDone()
	<-done
}

// TestServerVADIgnoresPipelineTurnFrames checks the session is left to its own
// boundaries when it is the one detecting them: committing on the pipeline's
// turn frame would cut the turn in two.
func TestServerVADIgnoresPipelineTurnFrames(t *testing.T) {
	srv := newFakeRealtime(t, ready())
	task, done, _ := run(t, realtime.Config{APIKey: "k", BaseURL: srv.wsURL()})
	srv.awaitMessage(t, "session.update")

	task.QueueFrame(frames.NewUserStoppedSpeakingFrame())
	time.Sleep(200 * time.Millisecond)
	for _, m := range srv.messages() {
		if m["type"] == "input_audio_buffer.commit" {
			t.Error("the input buffer was committed while xAI was detecting the turns")
		}
	}

	task.StopWhenDone()
	<-done
}

// TestForceMessageSpeaksVerbatim checks the item shape xAI plays without the
// model writing it.
func TestForceMessageSpeaksVerbatim(t *testing.T) {
	srv := newFakeRealtime(t, ready())
	svc := realtime.New(realtime.Config{APIKey: "k", BaseURL: srv.wsURL()})
	task, done := runService(t, svc)
	srv.awaitMessage(t, "session.update")

	if err := svc.ForceMessage(t.Context(), "This call is being recorded."); err != nil {
		t.Fatalf("ForceMessage: %v", err)
	}
	created := srv.awaitMessage(t, "conversation.item.create")

	item, _ := created["item"].(map[string]any)
	if item["type"] != "force_message" {
		t.Errorf("item type = %v, want force_message", item["type"])
	}
	if item["role"] != "assistant" {
		t.Errorf("item role = %v, want assistant", item["role"])
	}
	content, _ := item["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("item content = %v, want the one text part", item["content"])
	}
	part, _ := content[0].(map[string]any)
	if part["text"] != "This call is being recorded." {
		t.Errorf("item text = %v, want the text given", part["text"])
	}

	task.StopWhenDone()
	<-done
}

// TestDeleteConversationItem checks an item is removed by the id the session
// gave it.
func TestDeleteConversationItem(t *testing.T) {
	srv := newFakeRealtime(t, ready())
	svc := realtime.New(realtime.Config{APIKey: "k", BaseURL: srv.wsURL()})
	task, done := runService(t, svc)
	srv.awaitMessage(t, "session.update")

	if err := svc.DeleteConversationItem(t.Context(), "item-1"); err != nil {
		t.Fatalf("DeleteConversationItem: %v", err)
	}
	deleted := srv.awaitMessage(t, "conversation.item.delete")
	if deleted["item_id"] != "item-1" {
		t.Errorf("item_id = %v, want item-1", deleted["item_id"])
	}

	task.StopWhenDone()
	<-done
}

// TestBenignErrorsLeaveTheSessionRunning checks the races the interruption path
// runs into by design are not reported as failures. Canceling a response that
// has already finished says the model had stopped, which is what was asked for.
func TestBenignErrorsLeaveTheSessionRunning(t *testing.T) {
	srv := newFakeRealtime(t, ready(
		`{"type":"error","error":{"message":"Cancellation failed: no active response found"}}`,
		`{"type":"response.created"}`,
	))
	task, done, collected := run(t, realtime.Config{APIKey: "k", BaseURL: srv.wsURL()})

	// The session carried on: an event after the error still reached the
	// pipeline, which it would not have had the read loop stopped.
	waitFor(t, func() bool { return hasFrame[*frames.BotStartedSpeakingFrame](collected()) })
	if hasFrame[*frames.ErrorFrame](collected()) {
		t.Error("a cancel race was reported as an error")
	}

	task.StopWhenDone()
	<-done
}

// TestServerErrorIsReported checks a real failure reaches the pipeline.
func TestServerErrorIsReported(t *testing.T) {
	srv := newFakeRealtime(t, ready(
		`{"type":"error","error":{"code":"invalid_api_key","message":"invalid api key"}}`,
	))
	task, done, collected := run(t, realtime.Config{APIKey: "k", BaseURL: srv.wsURL()})

	waitFor(t, func() bool { return hasFrame[*frames.ErrorFrame](collected()) })

	task.StopWhenDone()
	<-done
}

// TestStartFrameTravelsOnce checks the service forwards a frame once. The LLM
// base forwards what it handles, so a service pushing again would deliver two of
// every frame, and describe itself to the pipeline twice with them.
func TestStartFrameTravelsOnce(t *testing.T) {
	srv := newFakeRealtime(t, ready())
	// Only what reaches the end of the pipeline: a broadcast builds a frame per
	// direction, so watching both ends would count the metadata twice.
	task, done, collected := runCollecting(t, realtime.Config{APIKey: "k", BaseURL: srv.wsURL()}, false)
	srv.awaitMessage(t, "session.update")

	task.StopWhenDone()
	<-done

	starts, ends, metadata := 0, 0, 0
	for _, f := range collected() {
		switch f.(type) {
		case *frames.StartFrame:
			starts++
		case *frames.EndFrame:
			ends++
		case *frames.LLMServiceMetadataFrame:
			metadata++
		}
	}
	if starts != 1 {
		t.Errorf("StartFrames reaching downstream = %d, want 1", starts)
	}
	if ends != 1 {
		t.Errorf("EndFrames reaching downstream = %d, want 1", ends)
	}
	if metadata != 1 {
		t.Errorf("service metadata frames = %d, want 1", metadata)
	}
}

// TestUserTranscriptTravelsUpstream checks the user's words go towards the user
// aggregator, which sits before this service. Pushed downstream they would reach
// the assistant aggregator and the output transport instead, and the
// conversation would hold no record of what the user said.
func TestUserTranscriptTravelsUpstream(t *testing.T) {
	script := ready(
		`{"type":"conversation.item.input_audio_transcription.completed","transcript":"hi there"}`,
	)

	// Downstream only: nothing the user said should arrive here.
	srv := newFakeRealtime(t, script)
	task, done, collected := runCollecting(t, realtime.Config{APIKey: "k", BaseURL: srv.wsURL()}, false)
	srv.awaitMessage(t, "session.update")
	time.Sleep(200 * time.Millisecond)
	for _, f := range collected() {
		if _, ok := f.(*frames.TranscriptionFrame); ok {
			t.Error("the user's transcript traveled downstream, away from the user aggregator")
		}
	}
	task.StopWhenDone()
	<-done

	// Watching both ends, it does arrive.
	srv2 := newFakeRealtime(t, script)
	task2, done2, collected2 := run(t, realtime.Config{APIKey: "k", BaseURL: srv2.wsURL()})
	waitFor(t, func() bool { return hasFrame[*frames.TranscriptionFrame](collected2()) })
	task2.StopWhenDone()
	<-done2
}
