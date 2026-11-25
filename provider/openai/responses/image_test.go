package responses

import (
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm/adapter"
)

// encodedParts renders an input item's content parts as they go on the wire.
func encodedParts(t *testing.T, item InputItem) []map[string]any {
	t.Helper()
	raw, err := json.Marshal(item)
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Content []map[string]any `json:"content"`
	}
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	return got.Content
}

func inputFor(t *testing.T, m frames.Message) []InputItem {
	t.Helper()
	c := frames.NewLLMContext("")
	c.SetMessages([]frames.Message{m})
	params, err := (&Adapter{}).LLMInvocationParams(c, adapter.Options{})
	if err != nil {
		t.Fatal(err)
	}
	return params.Input
}

func TestMultimodalImageConversion(t *testing.T) {
	input := inputFor(t, frames.CreateImageURLMessage(frames.RoleUser, "data:image/jpeg;base64,abc123", "Describe this:"))
	content := encodedParts(t, input[0])
	if len(content) != 2 {
		t.Fatalf("content %v, want 2 parts", content)
	}
	if content[0]["type"] != "input_text" || content[0]["text"] != "Describe this:" {
		t.Errorf("first part %v, want input_text", content[0])
	}
	if content[1]["type"] != "input_image" || content[1]["image_url"] != "data:image/jpeg;base64,abc123" ||
		content[1]["detail"] != "auto" {
		t.Errorf("second part %v, want input_image with detail auto", content[1])
	}
}

func TestMultimodalImageWithDetail(t *testing.T) {
	input := inputFor(t, frames.Message{Role: frames.RoleUser, Content: []frames.ContentPart{
		{Type: frames.ContentPartImageURL, ImageURL: "https://example.com/img.png", Detail: "high"},
	}})
	if content := encodedParts(t, input[0]); content[0]["detail"] != "high" {
		t.Errorf("part %v, want detail high", content[0])
	}
}

func TestLoggingElidesImageData(t *testing.T) {
	c := frames.NewLLMContext("")
	c.SetMessages([]frames.Message{
		frames.CreateImageURLMessage(frames.RoleUser, "data:image/png;base64,iVBORw==", "look"),
	})
	logged := (&Adapter{}).MessagesForLogging(c)
	content, _ := logged[0]["content"].([]any)
	image, _ := content[1].(map[string]any)
	if image["image_url"] != "data:image/..." {
		t.Errorf("logged part %v, want the placeholder", image)
	}
	if got := c.Messages()[0].Content[1].ImageURL; got != "data:image/png;base64,iVBORw==" {
		t.Errorf("conversation url %q after logging", got)
	}
}
