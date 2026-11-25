package chat

import (
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm/adapter"
)

// imageConvo is a conversation whose one user message holds text and an image.
func imageConvo(url string) *frames.LLMContext {
	c := frames.NewLLMContext("You are a helpful assistant that can analyze images.")
	c.SetMessages([]frames.Message{frames.CreateImageURLMessage(frames.RoleUser, url, "What's in this image?")})
	return c
}

// encoded renders a message as it goes on the wire.
func encoded(t *testing.T, m Message) map[string]any {
	t.Helper()
	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestComplexMessageContentPreserved(t *testing.T) {
	url := "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
	params, err := (&Adapter{}).LLMInvocationParams(imageConvo(url), adapter.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if len(params.Messages) != 2 {
		t.Fatalf("messages %+v, want the system prompt and the image message", params.Messages)
	}
	content, _ := encoded(t, params.Messages[1])["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("content %v, want two parts", content)
	}
	text, _ := content[0].(map[string]any)
	image, _ := content[1].(map[string]any)
	if text["type"] != "text" || text["text"] != "What's in this image?" {
		t.Errorf("first part %v, want the text", text)
	}
	imageURL, _ := image["image_url"].(map[string]any)
	if image["type"] != "image_url" || imageURL["url"] != url {
		t.Errorf("second part %v, want the image", image)
	}
}

func TestImageDetailIsKept(t *testing.T) {
	c := frames.NewLLMContext("")
	c.SetMessages([]frames.Message{{Role: frames.RoleUser, Content: []frames.ContentPart{
		{Type: frames.ContentPartImageURL, ImageURL: "https://example.com/img.png", Detail: "high"},
	}}})
	params, err := (&Adapter{}).LLMInvocationParams(c, adapter.Options{})
	if err != nil {
		t.Fatal(err)
	}
	content, _ := encoded(t, params.Messages[0])["content"].([]any)
	image, _ := content[0].(map[string]any)
	if imageURL, _ := image["image_url"].(map[string]any); imageURL["detail"] != "high" {
		t.Errorf("part %v, want detail high", image)
	}
}

func loggedImageURLs(t *testing.T, c *frames.LLMContext) []any {
	t.Helper()
	var urls []any
	for _, m := range (&Adapter{}).MessagesForLogging(c) {
		parts, _ := m["content"].([]any)
		for _, p := range parts {
			part, _ := p.(map[string]any)
			if image, ok := part["image_url"].(map[string]any); ok {
				urls = append(urls, image["url"])
			}
		}
	}
	return urls
}

func TestLoggingElidesBase64ImageURL(t *testing.T) {
	urls := loggedImageURLs(t, imageConvo("data:image/jpeg;base64,/9j/4AAQSkZJRg=="))
	if len(urls) != 1 || urls[0] != "data:image/..." {
		t.Errorf("logged urls %v, want the placeholder", urls)
	}
}

func TestLoggingPreservesHTTPImageURL(t *testing.T) {
	urls := loggedImageURLs(t, imageConvo("https://example.com/image.jpg"))
	if len(urls) != 1 || urls[0] != "https://example.com/image.jpg" {
		t.Errorf("logged urls %v, want the web URL", urls)
	}
}

func TestLoggingElidesMultipleImages(t *testing.T) {
	c := frames.NewLLMContext("")
	c.SetMessages([]frames.Message{{Role: frames.RoleUser, Content: []frames.ContentPart{
		{Type: frames.ContentPartImageURL, ImageURL: "data:image/jpeg;base64,AAAA"},
		{Type: frames.ContentPartImageURL, ImageURL: "data:image/png;base64,BBBB"},
		{Type: frames.ContentPartImageURL, ImageURL: "data:image/gif;base64,CCCC"},
	}}})
	urls := loggedImageURLs(t, c)
	if len(urls) != 3 {
		t.Fatalf("logged urls %v, want three", urls)
	}
	for _, u := range urls {
		if u != "data:image/..." {
			t.Errorf("logged url %v, want the placeholder", u)
		}
	}
}

func TestLoggingDoesNotMutateTheConversation(t *testing.T) {
	url := "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
	c := imageConvo(url)
	loggedImageURLs(t, c)
	if got := c.Messages()[0].Content[1].ImageURL; got != url {
		t.Errorf("conversation url %q after logging, want %q", got, url)
	}
}
