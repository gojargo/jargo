package gemini

import (
	"testing"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
)

const tinyPNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

func TestComplexMessageContentPreserved(t *testing.T) {
	c := frames.NewLLMContext("You are a helpful assistant that can analyze images and audio.")
	c.SetMessages([]frames.Message{
		frames.CreateImageURLMessage(frames.RoleUser, "data:image/jpeg;base64,"+tinyPNG, "What's in this image?"),
		{Role: frames.RoleAssistant, Content: []frames.ContentPart{
			{Type: frames.ContentPartText, Text: "Let me analyze this step by step:"},
			{Type: frames.ContentPartText, Text: "1. First, I'll examine the visual elements"},
			{Type: frames.ContentPartText, Text: "2. Then I'll provide my conclusions"},
		}},
	})
	params, err := (&Adapter{}).LLMInvocationParams(c, adapter.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if params.SystemInstruction != "You are a helpful assistant that can analyze images and audio." {
		t.Errorf("system instruction %q", params.SystemInstruction)
	}
	// A conversation ending on the model has the system prompt follow it as the
	// user, so only the first two contents are the messages.
	if len(params.Contents) < 2 {
		t.Fatalf("contents %v, want the two messages", params.Contents)
	}

	user := params.Contents[0]
	parts := contentParts(user)
	if user[keyRole] != roleUser || len(parts) != 2 {
		t.Fatalf("user content %v, want text and an image", user)
	}
	if parts[0][keyText] != "What's in this image?" {
		t.Errorf("first part %v, want the text", parts[0])
	}
	blob, _ := parts[1][keyInlineData].(map[string]any)
	if blob[keyMIMEType] != "image/jpeg" || blob[keyData] != tinyPNG {
		t.Errorf("second part %v, want the inline image", parts[1])
	}

	model := params.Contents[1]
	if parts := contentParts(model); model[keyRole] != roleModel || len(parts) != 3 {
		t.Errorf("model content %v, want three text parts", model)
	}
}

func TestImageURLBecomesFileData(t *testing.T) {
	c := frames.NewLLMContext("")
	c.SetMessages([]frames.Message{
		frames.CreateImageURLMessage(frames.RoleUser, "gs://bucket/a.jpg", ""),
	})
	params, err := (&Adapter{}).LLMInvocationParams(c, adapter.Options{})
	if err != nil {
		t.Fatal(err)
	}
	parts := contentParts(params.Contents[0])
	file, _ := parts[0][keyFileData].(map[string]any)
	if file[keyFileURI] != "gs://bucket/a.jpg" {
		t.Errorf("part %v, want file data for the URL", parts[0])
	}
}

func TestLoggingElidesInlineImageData(t *testing.T) {
	c := frames.NewLLMContext("")
	c.SetMessages([]frames.Message{
		frames.CreateImageURLMessage(frames.RoleUser, "data:image/png;base64,"+tinyPNG, "look"),
	})
	logged := (&Adapter{}).MessagesForLogging(c)
	parts := contentParts(logged[0])
	blob, _ := parts[1][keyInlineData].(map[string]any)
	if blob[keyData] != "..." {
		t.Errorf("logged part %v, want its data elided", parts[1])
	}
	// The conversation keeps its image.
	if got := c.Messages()[0].Content[1].ImageURL; got != "data:image/png;base64,"+tinyPNG {
		t.Errorf("conversation url %q after logging", got)
	}
}
