package realtime

import (
	"slices"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
)

func TestTypedTextUsesHistoryAndStandardOutputWithoutReceipts(t *testing.T) {
	h := converse(t, harnessOptions{})
	h.queue(appendUser("Hello Prism", true))
	request := h.peer.nextReply()
	if _, ok := request["response"].(object)["prism_input_text"]; ok {
		t.Error("the reply carries typed input")
	}
	if texts := h.peer.historyTexts(); texts[len(texts)-1] != "Hello Prism" {
		t.Errorf("history = %v, want the appended message last", texts)
	}
	h.answer(request, "Hello caller", 960)
	until(t, "the reply in the conversation", func() bool { return h.hasMessage(frames.RoleAssistant, "Hello caller") })
	until(t, "audio written", func() bool { return h.output.written() > 0 })
	if n := h.countMessage(frames.RoleUser, "Hello Prism"); n != 1 {
		t.Errorf("the appended message is in the conversation %d times, want once", n)
	}
	if len(h.peer.sent("prism.playback.drained")) > 0 {
		t.Error("a playback receipt was sent")
	}
	if !slices.ContainsFunc(captured[*frames.LLMTextFrame](h.down), func(f *frames.LLMTextFrame) bool {
		return f.Text == "Hello caller"
	}) {
		t.Error("the reply's text never reached the aggregator")
	}
	h.queue(appendUser("Again", true))
	h.answer(nil, "Second reply", 480)
	until(t, "the second reply", func() bool { return h.hasMessage(frames.RoleAssistant, "Second reply") })
	// Each append writes only its own messages.
	if texts := h.peer.createdTexts(); !slices.Equal(texts, []string{"Hello Prism", "Again"}) {
		t.Errorf("created = %v", texts)
	}
}

func TestInitialHistoryImportStartsOneReply(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.SetMessages([]frames.Message{
		{Role: frames.RoleSystem, Text: "Be brief"},
		{Role: frames.RoleDeveloper, Text: "Speak English"},
		{Role: frames.RoleUser, Text: "Earlier turn"},
	})
	h := converse(t, harnessOptions{convo: convo})
	h.peer.mu.Lock()
	var got [][2]any
	for _, item := range h.peer.history {
		got = append(got, [2]any{item["role"], item["text"]})
	}
	h.peer.mu.Unlock()
	want := [][2]any{{"system", "Be brief"}, {"system", "Speak English"}, {"user", "Earlier turn"}}
	if !slices.Equal(got, want) {
		t.Errorf("history = %v, want %v", got, want)
	}
	if n := len(h.peer.sent(kindResponseNew)); n != 1 {
		t.Errorf("%d replies requested, want 1", n)
	}
}

func TestVoiceTranscriptAndReplyAreRecordedOnceWithoutReimport(t *testing.T) {
	h := converse(t, harnessOptions{})
	h.userSpeaks()
	h.peer.emit("conversation.item.input_audio_transcription.completed",
		object{"item_id": "speech-1", "transcript": "Voice question"})
	ref := h.peer.start(nil, []string{"speech-1"})
	h.peer.say(ref, "Voice answer")
	h.peer.complete(ref)
	until(t, "the reply in the conversation", func() bool { return h.hasMessage(frames.RoleAssistant, "Voice answer") })
	if n := h.countMessage(frames.RoleUser, "Voice question"); n != 1 {
		t.Errorf("the transcript is in the conversation %d times, want once", n)
	}
	if n := len(captured[*frames.TranscriptionFrame](h.down)); n != 1 {
		t.Errorf("%d transcriptions, want 1", n)
	}
	time.Sleep(50 * time.Millisecond)
	if n := len(h.peer.sent(kindResponseNew)); n != 1 {
		t.Errorf("%d replies requested, want 1", n)
	}
	if n := len(h.peer.sent(kindItemCreate)); n != 0 {
		t.Errorf("%d items created, want none", n)
	}
}
