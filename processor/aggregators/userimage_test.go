package aggregators_test

import (
	"context"
	"encoding/base64"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
)

// userImage is a one-pixel image from the user, asked to go into the
// conversation.
func userImage() *frames.UserImageRawFrame {
	f := frames.NewUserImageRawFrame("u1", []byte{0, 0, 0}, 1, 1, "RGB")
	f.AppendToContext = true
	return f
}

func TestUserImageFrameRunLLMFalseDoesNotRun(t *testing.T) {
	convo := frames.NewLLMContext("")
	task, count, stop := assistantUpstream(t, convo)

	f := userImage()
	f.RunLLM = new(false)
	task.QueueFrame(f)
	time.Sleep(200 * time.Millisecond)
	stop()

	if got := count(); got != 0 {
		t.Errorf("generation ran %d times, want 0", got)
	}
	if msgs := convo.Messages(); len(msgs) != 1 {
		t.Errorf("messages %+v, want the image", msgs)
	}
}

func TestUserImageFrameRunLLMDefaultRuns(t *testing.T) {
	convo := frames.NewLLMContext("")
	task, count, stop := assistantUpstream(t, convo)

	task.QueueFrame(userImage())
	awaitCount(t, count, 1, "an image appended to the conversation")
	stop()

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Role != frames.RoleUser || len(msgs[0].Content) != 1 ||
		msgs[0].Content[0].Type != frames.ContentPartImageURL {
		t.Errorf("messages %+v, want one user image message", msgs)
	}
}

func TestUserImageNotForTheContextIsLeftOut(t *testing.T) {
	convo := frames.NewLLMContext("")
	task, count, stop := assistantUpstream(t, convo)

	f := userImage()
	f.AppendToContext = false
	task.QueueFrame(f)
	time.Sleep(200 * time.Millisecond)
	stop()

	if got, msgs := count(), convo.Messages(); got != 0 || len(msgs) != 0 {
		t.Errorf("generation ran %d times with messages %+v, want neither", got, msgs)
	}
}

// TestUserImageForAToolCallLandsAfterItsResult covers a tool that asks to see the
// user: the image answering it is held until the call's result arrives, the call
// is told the image has been taken so it can complete, and the image goes into
// the conversation after the result and runs the model.
func TestUserImageForAToolCallLandsAfterItsResult(t *testing.T) {
	convo := frames.NewLLMContext("")
	task, count, stop := assistantUpstream(t, convo)

	task.QueueFrame(frames.NewFunctionCallsStartedFrame([]frames.ToolCall{{ID: "c1", Name: "get_image"}}))
	task.QueueFrame(inProgress("c1", "get_image", nil))

	var mu sync.Mutex
	var results []string
	called := make(chan struct{}, 1)
	request := frames.NewUserImageRequestFrame("u1")
	request.FunctionName, request.ToolCallID = "get_image", "c1"
	request.AppendToContext, request.Text = true, "What is in front of the user?"
	request.ResultCallback = func(_ context.Context, result string, _ *frames.FunctionCallResultProperties) error {
		mu.Lock()
		results = append(results, result)
		mu.Unlock()
		called <- struct{}{}
		return nil
	}
	time.Sleep(100 * time.Millisecond)

	image := userImage()
	image.Text, image.Request = request.Text, request
	task.QueueFrame(image)

	select {
	case <-called:
	case <-time.After(3 * time.Second):
		t.Fatal("the tool call was not told the image was taken")
	}
	if got := count(); got != 0 {
		t.Errorf("generation ran %d times before the call's result, want 0", got)
	}

	// What the callback reports: a call that completed with no result.
	task.QueueFrame(frames.NewFunctionCallResultFrame("c1", "get_image", nil, ""))
	awaitCount(t, count, 1, "the image after the call's result")
	stop()

	mu.Lock()
	defer mu.Unlock()
	if len(results) != 1 || results[0] != "" {
		t.Errorf("callback results %q, want one empty result", results)
	}
	msgs := convo.Messages()
	if len(msgs) != 3 {
		t.Fatalf("messages %+v, want the call, its result and the image", msgs)
	}
	if len(msgs[1].ToolResults) != 1 || msgs[1].ToolResults[0].Content != frames.ToolResultCompleted {
		t.Errorf("result %+v, want the call completed", msgs[1])
	}
	if msgs[2].StandardText() != "What is in front of the user?" || len(msgs[2].Content) != 2 {
		t.Errorf("image message %+v, want the text and the image", msgs[2])
	}
}

// TestUserImageMidReplyLandsAfterWhatWasSaid checks an image arriving while the
// bot is replying goes in after what the bot has said so far.
func TestUserImageMidReplyLandsAfterWhatWasSaid(t *testing.T) {
	convo := frames.NewLLMContext("")
	runAssistant(t, convo, func(task *pipeline.Worker) {
		task.QueueFrame(frames.NewLLMFullResponseStartFrame())
		task.QueueFrame(frames.NewLLMTextFrame("Hello there."))
		time.Sleep(100 * time.Millisecond)
		f := userImage()
		f.RunLLM = new(false)
		task.QueueFrame(f)
		time.Sleep(100 * time.Millisecond)
		task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	})

	msgs := convo.Messages()
	if len(msgs) != 2 || msgs[0].Text != "Hello there." || len(msgs[1].Content) != 1 {
		t.Errorf("messages %+v, want the reply then the image", msgs)
	}
}

func TestAssistantImageGoesInAsTheAssistants(t *testing.T) {
	convo := frames.NewLLMContext("")
	f := frames.NewAssistantImageRawFrame([]byte{0, 0, 0}, 1, 1, "RGB")
	f.OriginalData, f.OriginalMIMEType = []byte("png bytes"), "image/png"
	drainAssistant(t, convo, f)

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Role != frames.RoleAssistant || len(msgs[0].Content) != 1 {
		t.Fatalf("messages %+v, want one assistant image", msgs)
	}
	want := "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("png bytes"))
	if got := msgs[0].Content[0].ImageURL; got != want {
		t.Errorf("url %q, want the original image %q", got, want)
	}
}
