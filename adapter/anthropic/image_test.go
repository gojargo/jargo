package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
)

// encodedContent renders a message's content blocks as they go on the wire.
func encodedContent(t *testing.T, m any) []map[string]any {
	t.Helper()
	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	var msg struct {
		Content []map[string]any `json:"content"`
	}
	if err := json.Unmarshal(raw, &msg); err != nil {
		t.Fatal(err)
	}
	return msg.Content
}

func TestComplexMessageContentPreserved(t *testing.T) {
	c := frames.NewLLMContext("")
	c.SetMessages([]frames.Message{
		{Role: frames.RoleUser, Content: []frames.ContentPart{
			{Type: frames.ContentPartText, Text: "What do you see in this image?"},
			{Type: frames.ContentPartImageURL, ImageURL: "data:image/jpeg;base64,fake_image_data"},
			{Type: frames.ContentPartText, Text: "Please describe it in detail."},
		}},
		{Role: frames.RoleAssistant, Text: "I can see the image clearly."},
	})
	params, err := (&Adapter{}).LLMInvocationParams(c, adapter.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if len(params.Messages) != 2 {
		t.Fatalf("messages %+v, want 2", params.Messages)
	}
	content := encodedContent(t, params.Messages[0])
	if len(content) != 3 {
		t.Fatalf("content %v, want 3 blocks", content)
	}
	// A single image moves ahead of the text, as Anthropic recommends.
	source, _ := content[0]["source"].(map[string]any)
	if content[0]["type"] != "image" || source["type"] != "base64" ||
		source["media_type"] != "image/jpeg" || source["data"] != "fake_image_data" {
		t.Errorf("first block %v, want the base64 image", content[0])
	}
	if content[1]["type"] != "text" || content[1]["text"] != "What do you see in this image?" {
		t.Errorf("second block %v, want the first text", content[1])
	}
	if content[2]["type"] != "text" || content[2]["text"] != "Please describe it in detail." {
		t.Errorf("third block %v, want the second text", content[2])
	}
}

func TestImageURLBecomesAURLSource(t *testing.T) {
	c := frames.NewLLMContext("")
	c.SetMessages([]frames.Message{
		frames.CreateImageURLMessage(frames.RoleUser, "https://example.com/a.jpg", ""),
	})
	params, err := (&Adapter{}).LLMInvocationParams(c, adapter.Options{})
	if err != nil {
		t.Fatal(err)
	}
	content := encodedContent(t, params.Messages[0])
	source, _ := content[0]["source"].(map[string]any)
	if content[0]["type"] != "image" || source["type"] != "url" || source["url"] != "https://example.com/a.jpg" {
		t.Errorf("block %v, want a URL image", content[0])
	}
}

func TestUnsupportedImageURLIsLeftOut(t *testing.T) {
	c := frames.NewLLMContext("")
	c.SetMessages([]frames.Message{
		frames.CreateImageURLMessage(frames.RoleUser, "ftp://example.com/a.jpg", "look"),
	})
	params, err := (&Adapter{}).LLMInvocationParams(c, adapter.Options{})
	if err != nil {
		t.Fatal(err)
	}
	content := encodedContent(t, params.Messages[0])
	if len(content) != 1 || content[0]["type"] != "text" {
		t.Errorf("content %v, want only the text", content)
	}
}

func TestSeveralImagesKeepTheirOrder(t *testing.T) {
	c := frames.NewLLMContext("")
	c.SetMessages([]frames.Message{{Role: frames.RoleUser, Content: []frames.ContentPart{
		{Type: frames.ContentPartText, Text: "Compare"},
		{Type: frames.ContentPartImageURL, ImageURL: "data:image/png;base64,AAAA"},
		{Type: frames.ContentPartImageURL, ImageURL: "data:image/png;base64,BBBB"},
	}}})
	params, err := (&Adapter{}).LLMInvocationParams(c, adapter.Options{})
	if err != nil {
		t.Fatal(err)
	}
	content := encodedContent(t, params.Messages[0])
	if len(content) != 3 || content[0]["type"] != "text" {
		t.Errorf("content %v, want the text first with two images", content)
	}
}

func TestLoggingElidesImageData(t *testing.T) {
	c := frames.NewLLMContext("")
	c.SetMessages([]frames.Message{
		frames.CreateImageURLMessage(frames.RoleUser, "data:image/png;base64,iVBORw==", "look"),
	})
	logged := (&Adapter{}).MessagesForLogging(c)
	content, _ := logged[0]["content"].([]any)
	image, _ := content[0].(map[string]any)
	source, _ := image["source"].(map[string]any)
	if source["data"] != "..." {
		t.Errorf("logged image %v, want its data elided", image)
	}
}
