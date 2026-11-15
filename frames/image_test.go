package frames_test

import (
	"bytes"
	"encoding/base64"
	"image/jpeg"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
)

func TestEncodedImageKeepsItsBytesAndType(t *testing.T) {
	png := []byte("\x89PNG fake")
	msg, err := frames.CreateImageMessage("", "image/png", 2, 2, png, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(msg.Content) != 1 || msg.Content[0].Type != frames.ContentPartImageURL {
		t.Fatalf("content %+v, want one image part", msg.Content)
	}
	want := "data:image/png;base64," + base64.StdEncoding.EncodeToString(png)
	if got := msg.Content[0].ImageURL; got != want {
		t.Errorf("url %q, want %q", got, want)
	}
	if msg.Role != frames.RoleUser {
		t.Errorf("role %q, want user", msg.Role)
	}
}

func TestPixelsWithAlphaAreEncodedAsAJPEG(t *testing.T) {
	pixels := bytes.Repeat([]byte{255, 0, 0, 128}, 4*4)
	msg, err := frames.CreateImageMessage("", "RGBA", 4, 4, pixels, "")
	if err != nil {
		t.Fatal(err)
	}
	const prefix = "data:image/jpeg;base64,"
	url := msg.Content[0].ImageURL
	if !strings.HasPrefix(url, prefix) {
		t.Fatalf("url %q, want a JPEG data URL", url)
	}
	data, err := base64.StdEncoding.DecodeString(url[len(prefix):])
	if err != nil {
		t.Fatal(err)
	}
	img, err := jpeg.Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if b := img.Bounds(); b.Dx() != 4 || b.Dy() != 4 {
		t.Errorf("decoded size %v, want 4x4", b)
	}
}

func TestImageMessageCarriesItsTextFirst(t *testing.T) {
	msg := frames.CreateImageURLMessage(frames.RoleAssistant, "https://example.com/a.jpg", "Describe this image")
	want := []frames.ContentPart{
		{Type: frames.ContentPartText, Text: "Describe this image"},
		{Type: frames.ContentPartImageURL, ImageURL: "https://example.com/a.jpg"},
	}
	if msg.Role != frames.RoleAssistant || len(msg.Content) != 2 ||
		msg.Content[0] != want[0] || msg.Content[1] != want[1] {
		t.Errorf("message %+v, want %+v", msg, want)
	}
	if got := msg.StandardText(); got != "Describe this image" {
		t.Errorf("standard text %q, want the text part", got)
	}
}

func TestAddImageFrameMessageAppendsToContext(t *testing.T) {
	c := frames.NewLLMContext("")
	if err := c.AddImageFrameMessage("image/png", 1, 1, []byte("png"), "look", ""); err != nil {
		t.Fatal(err)
	}
	msgs := c.Messages()
	if len(msgs) != 1 || msgs[0].Role != frames.RoleUser || len(msgs[0].Content) != 2 {
		t.Fatalf("messages %+v, want one user image message with its text", msgs)
	}
	// The context hands out copies, so editing one leaves the conversation alone.
	msgs[0].Content[0].Text = "changed"
	if c.Messages()[0].Content[0].Text != "look" {
		t.Error("editing a returned message changed the conversation")
	}
}

func TestAddImageFrameMessageRejectsBadPixels(t *testing.T) {
	c := frames.NewLLMContext("")
	if err := c.AddImageFrameMessage("RGB", 4, 4, []byte{1, 2, 3}, "", ""); err == nil {
		t.Error("pixels short of the size were accepted")
	}
	if len(c.Messages()) != 0 {
		t.Error("a failed image was added to the conversation")
	}
}
