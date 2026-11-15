// Package evalws is the bot's end of an eval: the eval serializer, the
// connection's flags, and the eval transport over a WebSocket the bot accepted.
//
// The eval harness drives a bot as an RTVI client. Serializer speaks RTVI and
// adds the eval's own messages on top: eval-connect, eval-configure,
// eval-context, eval-cancel and eval-image from the harness, and eval-bot-audio
// and eval-bot-image from the bot when the harness asks for them.
//
// The package depends on nothing but the WebSocket transport and RTVI, so a bot
// (and the development runner hosting it) can serve the harness without pulling
// in the harness itself.
package evalws

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	_ "image/gif"  // registers the GIF decoder for user images
	_ "image/jpeg" // registers the JPEG decoder for user images
	_ "image/png"  // registers the PNG decoder for user images
	"log/slog"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/rtviws"
)

const (
	// ConnectMessageType is the client-message `t` of the first message the
	// harness sends on every connection, carrying the connection's flags
	// (ConnectionFlags). On a WebSocket the bot accepted itself, it is how the
	// runner's CreateTransport recognizes an eval and how the flags reach the
	// bot, since only the session body survives the hops in between. The
	// serializer consumes it.
	ConnectMessageType = "eval-connect"

	// ContextMessageType is the client-message `t` the serializer intercepts and
	// turns into an LLMMessagesUpdateFrame instead of forwarding it to the RTVI
	// processor. It keeps per-eval context seeding out of the bot.
	ContextMessageType = "eval-context"

	// ConfigureMessageType is the client-message `t` the serializer intercepts
	// and turns into an rtvi.ConfigureObserverFrame. This is the trust boundary
	// for raising the function-call report level: only the eval transport
	// understands it, so a production RTVI serializer cannot be elevated by a
	// remote client.
	ConfigureMessageType = "eval-configure"

	// CancelMessageType is the client-message `t` the serializer intercepts and
	// turns into a CancelWorkerFrame, so the harness can end a scenario by
	// tearing down the bot's pipeline gracefully (closing its service
	// connections) instead of the orchestrator having to kill the process.
	CancelMessageType = "eval-cancel"

	// ImageMessageType is the client-message `t` registering an image (the
	// base64-encoded bytes of a PNG, JPEG or similar, plus its MIME `format`) on
	// the serializer. The eval input transport hands it back when the bot asks
	// for a user image, so a vision bot can be driven without a real camera.
	ImageMessageType = "eval-image"

	// BotAudioType is the outbound message carrying a chunk of the bot's
	// synthesized audio (base64), sent only when the harness asked for it. It is
	// sent under the RTVI label so the harness reader sees it; the harness
	// transcribes the audio itself.
	BotAudioType = "eval-bot-audio"

	// BotImageType is the outbound message describing an image the bot output
	// (its size and color format, not its pixels), sent only when the harness
	// asked for it.
	BotImageType = "eval-bot-image"

	// STTSampleRate is the rate the harness resamples the bot's audio to before
	// its own VAD and STT see it. 16 kHz is what Silero and the local STT models
	// expect; the harness configures its input transport at this rate to match.
	STTSampleRate = 16000
)

// ConnectionFlags is what one eval connection asks of the bot, sent in the
// eval-connect message.
type ConnectionFlags struct {
	// SkipTTS silences the bot's speech for the connection (text mode).
	SkipTTS bool `json:"skip_tts"`
	// CaptureBotAudio forwards the bot's synthesized audio to the harness.
	CaptureBotAudio bool `json:"capture_bot_audio"`
	// CaptureBotImages reports the images the bot outputs to the harness.
	CaptureBotImages bool `json:"capture_bot_images"`
}

// ConnectMessage is the eval-connect message carrying flags, as JSON text.
func ConnectMessage(flags ConnectionFlags) string {
	data, _ := json.Marshal(rtvi.Message{ //nolint:errchkjson // the message is plain data
		Label: rtvi.MessageLabel,
		Type:  rtvi.TypeClientMessage,
		ID:    ConnectMessageType,
		Data:  map[string]any{"t": ConnectMessageType, "d": flags},
	})
	return string(data)
}

// ParseConnect returns the flags of an eval-connect message, or nil for any
// other message. message is a decoded JSON message, as encoding/json decodes it
// into an any. A flag the message leaves out, or does not give as a boolean, is
// off.
func ParseConnect(message any) *ConnectionFlags {
	m, ok := message.(map[string]any)
	if !ok || m["label"] != rtvi.MessageLabel {
		return nil
	}
	if m["type"] != rtvi.TypeClientMessage {
		return nil
	}
	data, ok := m["data"].(map[string]any)
	if !ok || data["t"] != ConnectMessageType {
		return nil
	}
	payload, _ := data["d"].(map[string]any)
	flags := &ConnectionFlags{}
	flags.SkipTTS, _ = payload["skip_tts"].(bool)
	flags.CaptureBotAudio, _ = payload["capture_bot_audio"].(bool)
	flags.CaptureBotImages, _ = payload["capture_bot_images"].(bool)
	return flags
}

// userImage is an image the harness registered: its still-encoded bytes and
// its MIME type.
type userImage struct {
	data   []byte
	format string
}

// Serializer bridges RTVI messages and frames on the bot's side of an eval. The
// bot pipeline needs an rtvi.Processor and an rtvi.Observer.
//
// Unlike the plain RTVI serializer, every RTVI message the harness sends other
// than the eval's own reaches the RTVI processor as it came, raw-audio
// included: the processor decodes it and pushes the audio into the input
// transport.
type Serializer struct {
	// The RTVI serializer writes RTVI messages to the wire rather than dropping
	// them: putting them there, so the harness can observe semantic events, is
	// the whole point.
	*rtviws.Serializer

	// captureAudio is off by default. The eval transport turns it on per
	// connection, from its CaptureBotAudio flag, so the bot's audio is not
	// shipped over the wire unless something asserts on it.
	captureAudio atomic.Bool
	// captureImages is off by default like the audio. The eval transport turns
	// it on per connection, from its CaptureBotImages flag, so a bot animating
	// its video output does not flood the harness.
	captureImages atomic.Bool

	// mu guards userImage, which the read loop sets and the input reads.
	mu sync.Mutex
	// userImage is the most recent image the harness registered, served back
	// when the bot asks for a user image. See ImageMessageType.
	userImage *userImage
}

var _ wsserver.Serializer = (*Serializer)(nil)

// NewSerializer builds the eval serializer.
func NewSerializer() *Serializer {
	return &Serializer{Serializer: rtviws.New()}
}

// SetCaptureAudio turns forwarding the bot's synthesized audio to the harness
// on or off.
func (s *Serializer) SetCaptureAudio(capture bool) { s.captureAudio.Store(capture) }

// SetCaptureImages turns reporting the images the bot outputs to the harness on
// or off.
func (s *Serializer) SetCaptureImages(capture bool) { s.captureImages.Store(capture) }

// UserImage returns the image registered for the current turn, as its encoded
// bytes and MIME type, and whether there is one.
func (s *Serializer) UserImage() (data []byte, format string, ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.userImage == nil {
		return nil, "", false
	}
	return s.userImage.data, s.userImage.format, true
}

// UserImageFrame returns the registered image as the frame answering request,
// or nil when no image is registered.
//
// The harness sends the image encoded, but a real camera transport pushes raw
// frames, so it is decoded to raw RGB. The LLM context encodes raw frames again
// anyway, and consumers that read the pixels directly (a local vision model,
// say) need the raw pixels and the real size.
func (s *Serializer) UserImageFrame(request *frames.UserImageRequestFrame) (*frames.UserImageRawFrame, error) {
	data, _, ok := s.UserImage()
	if !ok {
		return nil, nil //nolint:nilnil // no image is registered
	}
	decoded, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("evalws: decode the user image: %w", err)
	}
	pixels, width, height := rgbPixels(decoded)
	f := frames.NewUserImageRawFrame(request.UserID, pixels, width, height, "RGB")
	f.Text = request.Text
	f.AppendToContext = request.AppendToContext
	f.Request = request
	return f, nil
}

// rgbPixels converts an image to packed 8-bit RGB, row by row, and returns the
// pixels with the image's width and height. An alpha channel is dropped rather
// than blended in.
func rgbPixels(img image.Image) (pixels []byte, width, height int) {
	bounds := img.Bounds()
	width, height = bounds.Dx(), bounds.Dy()
	pixels = make([]byte, 0, width*height*3)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			c, _ := color.NRGBAModel.Convert(img.At(x, y)).(color.NRGBA)
			pixels = append(pixels, c.R, c.G, c.B)
		}
	}
	return pixels, width, height
}

// botAudioData is the payload of an eval-bot-audio message.
type botAudioData struct {
	Audio string `json:"audio"`
	//nolint:tagliatelle // RTVI wire field, camelCase in the protocol
	SampleRate int `json:"sampleRate"`
}

// botImageData is the payload of an eval-bot-image message.
type botImageData struct {
	Width  int    `json:"width"`
	Height int    `json:"height"`
	Format string `json:"format"`
}

// Serialize encodes an outbound frame for the harness. Only RTVI server
// messages go out, plus the bot's audio and the size and format of its images
// when the harness asked for them.
func (s *Serializer) Serialize(f frames.Frame) (wsserver.Message, error) {
	switch fr := f.(type) {
	case frames.OutputTransportMessage:
		return s.Serializer.Serialize(f)
	case frames.OutputAudioFrame:
		if !s.captureAudio.Load() {
			return wsserver.Message{}, nil
		}
		a := fr.AudioData()
		return wsserver.TextMessage(json.Marshal(rtvi.Message{
			Label: rtvi.MessageLabel,
			Type:  BotAudioType,
			Data: botAudioData{
				Audio:      base64.StdEncoding.EncodeToString(a.Audio),
				SampleRate: a.SampleRate,
			},
		}))
	case frames.OutputImageFrame:
		if !s.captureImages.Load() {
			return wsserver.Message{}, nil
		}
		img := fr.ImageData()
		return wsserver.TextMessage(json.Marshal(rtvi.Message{
			Label: rtvi.MessageLabel,
			Type:  BotImageType,
			Data:  botImageData{Width: img.Width, Height: img.Height, Format: img.Format},
		}))
	}
	return wsserver.Message{}, nil
}

// Deserialize decodes an inbound JSON RTVI message into a frame: an
// LLMMessagesUpdateFrame for eval-context, an rtvi.ConfigureObserverFrame for
// eval-configure, a CancelWorkerFrame for eval-cancel, and an
// InputTransportMessageFrame wrapping any other RTVI message. eval-connect and
// eval-image are consumed, and a payload that is not an RTVI message is
// dropped.
func (s *Serializer) Deserialize(data []byte) (frames.Frame, error) {
	var decoded any
	if err := json.Unmarshal(data, &decoded); err != nil {
		slog.Warn("evalws: dropping non-JSON message", "err", err)
		return nil, nil //nolint:nilnil,nilerr // dropping a non-JSON message is intentional
	}

	message, ok := decoded.(map[string]any)
	if !ok || message["label"] != rtvi.MessageLabel {
		slog.Warn("evalws: ignoring non-RTVI message", "message", string(data))
		return nil, nil //nolint:nilnil // dropping a non-RTVI message is intentional
	}

	// The transport already applied the connection's flags: from this message
	// when it was built, or from the URL when it is the server.
	if ParseConnect(message) != nil {
		return nil, nil //nolint:nilnil // eval-connect is consumed
	}

	if f, matched := s.maybeContextFrame(message); matched {
		return f, nil
	}

	if f, matched := s.maybeConfigureFrame(message); matched {
		return f, nil
	}

	if f := s.maybeCancelFrame(message); f != nil {
		return f, nil
	}

	if s.maybeStoreImage(message) {
		return nil, nil //nolint:nilnil // eval-image is consumed
	}

	// The transport may reuse its read buffer, and the frame outlives this call,
	// so hand the RTVI processor its own copy.
	msg := make([]byte, len(data))
	copy(msg, data)
	return frames.NewInputTransportMessageFrame(msg), nil
}

// evalPayload returns the `d` of a client-message whose `t` is t, and whether
// message is one.
func evalPayload(message map[string]any, t string) (any, bool) {
	if message["type"] != rtvi.TypeClientMessage {
		return nil, false
	}
	data, ok := message["data"].(map[string]any)
	if !ok || data["t"] != t {
		return nil, false
	}
	return data["d"], true
}

// maybeContextFrame returns the LLMMessagesUpdateFrame for an eval-context
// message, and whether message is one. The messages are LLM messages in the
// universal role and content shape; a context whose messages cannot be read is
// dropped, with a nil frame.
func (s *Serializer) maybeContextFrame(message map[string]any) (frames.Frame, bool) {
	d, ok := evalPayload(message, ContextMessageType)
	if !ok {
		return nil, false
	}
	payload, _ := d.(map[string]any)
	raw, _ := payload["messages"].([]any)
	messages, err := contextMessages(raw)
	if err != nil {
		slog.Warn("evalws: dropping an eval-context whose messages cannot be read", "err", err)
		return nil, true
	}
	f := frames.NewLLMMessagesUpdateFrame(messages)
	// Seed the context, do not trigger a response.
	f.RunLLM = false
	return f, true
}

// errContextMessage is an eval-context message that is not an LLM message.
var errContextMessage = errors.New("not an LLM message with a role")

// contextMessages reads the messages of an eval-context. Each is a mapping with
// a role and its content, as text or as a list of text parts; an assistant's
// may carry tool_calls, and a tool message the tool_call_id it answers.
func contextMessages(raw []any) ([]frames.Message, error) {
	messages := make([]frames.Message, 0, len(raw))
	for i, item := range raw {
		m, ok := item.(map[string]any)
		role, _ := m["role"].(string)
		if !ok || role == "" {
			return nil, fmt.Errorf("message %d: %w", i, errContextMessage)
		}
		if role == "tool" {
			id, _ := m["tool_call_id"].(string)
			messages = append(messages, frames.Message{
				Role:        frames.RoleUser,
				ToolResults: []frames.ToolResult{{ID: id, Content: contentText(m["content"])}},
			})
			continue
		}
		msg := frames.Message{Role: frames.Role(role), ToolCalls: toolCalls(m["tool_calls"])}
		if list, ok := m["content"].([]any); ok {
			msg.Content = contentParts(list)
		} else {
			msg.Text, _ = m["content"].(string)
		}
		messages = append(messages, msg)
	}
	return messages, nil
}

// contentParts reads a list of content parts: text, and images by URL.
func contentParts(list []any) []frames.ContentPart {
	parts := make([]frames.ContentPart, 0, len(list))
	for _, p := range list {
		part, _ := p.(map[string]any)
		switch part["type"] {
		case string(frames.ContentPartText):
			text, _ := part["text"].(string)
			parts = append(parts, frames.ContentPart{Type: frames.ContentPartText, Text: text})
		case string(frames.ContentPartImageURL):
			image, _ := part["image_url"].(map[string]any)
			url, _ := image["url"].(string)
			detail, _ := image["detail"].(string)
			parts = append(parts, frames.ContentPart{Type: frames.ContentPartImageURL, ImageURL: url, Detail: detail})
		}
	}
	return parts
}

// contentText is the text of a content: the string itself, or its text parts
// joined.
func contentText(content any) string {
	switch c := content.(type) {
	case string:
		return c
	case []any:
		var texts []string
		for _, part := range contentParts(c) {
			if part.Type == frames.ContentPartText {
				texts = append(texts, part.Text)
			}
		}
		return strings.Join(texts, "\n")
	default:
		return ""
	}
}

// toolCalls reads an assistant message's tool_calls, each a function call with
// its id, its name and its arguments as JSON text.
func toolCalls(raw any) []frames.ToolCall {
	list, _ := raw.([]any)
	var calls []frames.ToolCall
	for _, item := range list {
		c, _ := item.(map[string]any)
		id, _ := c["id"].(string)
		fn, _ := c["function"].(map[string]any)
		name, _ := fn["name"].(string)
		var args json.RawMessage
		switch a := fn["arguments"].(type) {
		case string:
			args = json.RawMessage(a)
		case nil:
		default:
			args, _ = json.Marshal(a) //nolint:errchkjson // decoded JSON encodes again
		}
		calls = append(calls, frames.ToolCall{ID: id, Name: name, Args: args})
	}
	return calls
}

// maybeCancelFrame returns the frame for an eval-cancel message, or nil for any
// other message. It cancels the whole pipeline so the bot exits on its own.
func (s *Serializer) maybeCancelFrame(message map[string]any) frames.Frame {
	if _, ok := evalPayload(message, CancelMessageType); !ok {
		return nil
	}
	return frames.NewCancelWorkerFrame()
}

// maybeStoreImage keeps the image of an eval-image message for the input
// transport to serve, and reports whether message was one (and so consumed).
func (s *Serializer) maybeStoreImage(message map[string]any) bool {
	d, ok := evalPayload(message, ImageMessageType)
	if !ok {
		return false
	}
	payload, _ := d.(map[string]any)
	encoded, _ := payload["image"].(string)
	if encoded == "" {
		return true
	}
	format, _ := payload["format"].(string)
	if format == "" {
		format = "image/jpeg"
	}
	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		slog.Warn("evalws: dropping an eval-image that is not base64", "err", err)
		return true
	}
	s.mu.Lock()
	s.userImage = &userImage{data: data, format: format}
	s.mu.Unlock()
	return true
}

// reportLevels are the function-call report levels an eval-configure may ask
// for.
var reportLevels = map[rtvi.FunctionCallReportLevel]bool{ //nolint:gochecknoglobals // a fixed set
	rtvi.ReportDisabled: true, rtvi.ReportNone: true, rtvi.ReportName: true,
	rtvi.ReportArguments: true, rtvi.ReportFull: true,
}

// maybeConfigureFrame returns the rtvi.ConfigureObserverFrame for an
// eval-configure message, and whether message is one. A report level that is
// not one of the levels drops the message, with a nil frame.
func (s *Serializer) maybeConfigureFrame(message map[string]any) (frames.Frame, bool) {
	d, ok := evalPayload(message, ConfigureMessageType)
	if !ok {
		return nil, false
	}
	payload, _ := d.(map[string]any)

	var reportLevel map[string]rtvi.FunctionCallReportLevel
	if levels, ok := payload["function_call_report_level"].(map[string]any); ok {
		// Values arrive as strings ("none", "name", "full"); coerce them to the
		// level type.
		reportLevel = make(map[string]rtvi.FunctionCallReportLevel, len(levels))
		for name, v := range levels {
			str, _ := v.(string)
			level := rtvi.FunctionCallReportLevel(str)
			if !reportLevels[level] {
				slog.Warn("evalws: dropping an eval-configure with an unknown report level",
					"function", name, "level", v)
				return nil, true
			}
			reportLevel[name] = level
		}
	}
	return rtvi.NewConfigureObserverFrame(
		reportLevel,
		optionalBool(payload, "vad_user_speaking"),
		optionalBool(payload, "llm_markers"),
	), true
}

// optionalBool reads key of payload as a setting: nil when it is absent or
// null, which leaves the observer's setting alone, and otherwise whether the
// value is truthy.
func optionalBool(payload map[string]any, key string) *bool {
	v, ok := payload[key]
	if !ok || v == nil {
		return nil
	}
	b := truthy(v)
	return &b
}

// truthy reports whether a decoded JSON value counts as true: a true boolean,
// a non-zero number, or a non-empty string, array or object.
func truthy(v any) bool {
	switch x := v.(type) {
	case bool:
		return x
	case float64:
		return x != 0
	case string:
		return x != ""
	case []any:
		return len(x) > 0
	case map[string]any:
		return len(x) > 0
	default:
		return v != nil
	}
}
