package novasonic

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"sync"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/google/uuid"
)

// The Nova Sonic wire protocol: a session is a stream of JSON events, each an
// object naming one event. The client opens the session (sessionStart), opens a
// prompt declaring the output and the tools (promptStart), and then sends
// content blocks, each opened with contentStart, filled, and closed with
// contentEnd: the system instruction and history as text, the user's audio as
// one long audio block, tool results. The model answers in content blocks of its
// own.

const (
	keyPromptName  = "promptName"
	keyContentName = "contentName"
	keyMediaType   = "mediaType"
	keyContent     = "content"
	mediaTypeText  = "text/plain"
	contentTypeTxt = "TEXT"
)

// sessionClient opens sessions. It stands in for the Bedrock runtime client, so
// a test can open sessions of its own.
type sessionClient interface {
	// OpenStream opens a bidirectional stream on model.
	OpenStream(ctx context.Context, model string) (sessionStream, error)
}

// sessionStream is one session's bidirectional stream.
type sessionStream interface {
	// Send writes one event.
	Send(ctx context.Context, event []byte) error
	// Recv reads the next event the model sent. It returns nil, nil when the
	// stream has ended without an error.
	Recv() ([]byte, error)
	// CloseInput closes the client's side of the stream.
	CloseInput() error
	// Close closes the whole stream.
	Close() error
}

// bedrockClient opens sessions on Bedrock.
type bedrockClient struct{ client *bedrockruntime.Client }

// newBedrockClient builds a Bedrock runtime client from the configured region
// and credentials, falling back to the default AWS chain for what is not set.
func newBedrockClient(ctx context.Context, cfg Config) (sessionClient, error) {
	var opts []func(*awsconfig.LoadOptions) error
	if cfg.Region != "" {
		opts = append(opts, awsconfig.WithRegion(cfg.Region))
	}
	if cfg.AccessKeyID != "" && cfg.SecretAccessKey != "" {
		opts = append(opts, awsconfig.WithCredentialsProvider(
			credentials.NewStaticCredentialsProvider(cfg.AccessKeyID, cfg.SecretAccessKey, cfg.SessionToken),
		))
	}
	awsCfg, err := awsconfig.LoadDefaultConfig(ctx, opts...)
	if err != nil {
		return nil, err
	}
	return &bedrockClient{client: bedrockruntime.NewFromConfig(awsCfg)}, nil
}

// OpenStream implements sessionClient. The stream lives until it is closed,
// not as long as ctx.
func (c *bedrockClient) OpenStream(ctx context.Context, model string) (sessionStream, error) {
	streamCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	out, err := c.client.InvokeModelWithBidirectionalStream(streamCtx,
		&bedrockruntime.InvokeModelWithBidirectionalStreamInput{ModelId: aws.String(model)})
	if err != nil {
		cancel()
		return nil, err
	}
	return &bedrockStream{stream: out.GetStream(), ctx: streamCtx, cancel: cancel}, nil
}

// errStreamReported reports the stream sending something other than an event.
//
//nolint:gochecknoglobals // sentinel error
var errStreamReported = errors.New("AWS Bedrock stream reported")

// bedrockStream is a session on Bedrock's bidirectional stream.
type bedrockStream struct {
	stream  *bedrockruntime.InvokeModelWithBidirectionalStreamEventStream
	ctx     context.Context //nolint:containedctx // the stream outlives the call that opened it
	cancel  context.CancelFunc
	writeMu sync.Mutex
}

// Send implements sessionStream.
func (b *bedrockStream) Send(_ context.Context, event []byte) error {
	b.writeMu.Lock()
	defer b.writeMu.Unlock()
	return b.stream.Writer.Send(b.ctx, &types.InvokeModelWithBidirectionalStreamInputMemberChunk{
		Value: types.BidirectionalInputPayloadPart{Bytes: event},
	})
}

// Recv implements sessionStream. Anything the stream reports other than an
// event is an error.
func (b *bedrockStream) Recv() ([]byte, error) {
	ev, ok := <-b.stream.Reader.Events()
	if !ok {
		return nil, b.stream.Reader.Err()
	}
	chunk, ok := ev.(*types.InvokeModelWithBidirectionalStreamOutputMemberChunk)
	if !ok {
		return nil, fmt.Errorf("%w: %v", errStreamReported, ev)
	}
	return chunk.Value.Bytes, nil
}

// CloseInput implements sessionStream.
func (b *bedrockStream) CloseInput() error {
	b.writeMu.Lock()
	defer b.writeMu.Unlock()
	return b.stream.Writer.Close()
}

// Close implements sessionStream.
func (b *bedrockStream) Close() error {
	err := b.stream.Close()
	b.cancel()
	return err
}

// --- the events the client sends ---

// event wraps one event's body.
func event(name string, body map[string]any) map[string]any {
	return map[string]any{"event": map[string]any{name: body}}
}

// sendEvent writes an event on stream. A missing stream sends nothing.
func sendEvent(ctx context.Context, stream sessionStream, ev map[string]any) error {
	if stream == nil {
		return nil
	}
	data, err := json.Marshal(ev)
	if err != nil {
		return err
	}
	return stream.Send(ctx, data)
}

// sessionStartEvent opens a session with the inference settings, and the
// endpointing sensitivity when one is set.
func (s *Service) sessionStartEvent() map[string]any {
	body := map[string]any{
		"inferenceConfiguration": map[string]any{
			"maxTokens":   optValue(s.live.MaxTokens.Value()),
			"topP":        optValue(s.live.TopP.Value()),
			"temperature": optValue(s.live.Temperature.Value()),
		},
	}
	if sensitivity := s.live.EndpointingSensitivity.Or(""); sensitivity != "" {
		body["turnDetectionConfiguration"] = map[string]any{"endpointingSensitivity": sensitivity}
	}
	return event("sessionStart", body)
}

// optValue is a setting's value, or nil (sent as null) when it has none.
func optValue[T any](v T, given bool) any {
	if !given {
		return nil
	}
	return v
}

// promptStartEvent opens a prompt declaring the output the model produces and,
// when there are any, the tools it may call.
func (s *Service) promptStartEvent(tools []map[string]any, prompt string) map[string]any {
	ac := s.audio
	body := map[string]any{
		keyPromptName:             prompt,
		"textOutputConfiguration": map[string]any{keyMediaType: mediaTypeText},
		"audioOutputConfiguration": map[string]any{
			keyMediaType:      "audio/lpcm",
			"sampleRateHertz": ac.OutputSampleRate,
			"sampleSizeBits":  ac.OutputSampleSize,
			"channelCount":    ac.OutputChannelCount,
			"voiceId":         s.live.Voice.Or(""),
			"encoding":        "base64",
			"audioType":       "SPEECH",
		},
	}
	if len(tools) > 0 {
		body["toolUseOutputConfiguration"] = map[string]any{keyMediaType: "application/json"}
		body["toolConfiguration"] = map[string]any{"tools": tools}
	}
	return event("promptStart", body)
}

// contentStart opens a content block.
func contentStart(prompt, content, typ, role string, interactive bool, extra map[string]any) map[string]any {
	body := map[string]any{
		keyPromptName:  prompt,
		keyContentName: content,
		"type":         typ,
		"interactive":  interactive,
		"role":         role,
	}
	maps.Copy(body, extra)
	return event("contentStart", body)
}

// contentEnd closes a content block.
func contentEnd(prompt, content string) map[string]any {
	return event("contentEnd", map[string]any{keyPromptName: prompt, keyContentName: content})
}

// promptEnd closes the prompt.
func promptEnd(prompt string) map[string]any {
	return event("promptEnd", map[string]any{keyPromptName: prompt})
}

// sessionEnd closes the session.
func sessionEnd() map[string]any { return event("sessionEnd", map[string]any{}) }

// textEvents are the block a text is sent as: opened, filled and closed.
func textEvents(text, role, prompt string, interactive bool) []map[string]any {
	content := uuid.NewString()
	return []map[string]any{
		contentStart(prompt, content, contentTypeTxt, role, interactive, map[string]any{
			"textInputConfiguration": map[string]any{keyMediaType: mediaTypeText},
		}),
		event("textInput", map[string]any{keyPromptName: prompt, keyContentName: content, keyContent: text}),
		contentEnd(prompt, content),
	}
}

// audioInputStartEvent opens the block the user's audio is streamed in.
func (s *Service) audioInputStartEvent(prompt, content string) map[string]any {
	ac := s.audio
	return contentStart(prompt, content, "AUDIO", "USER", true, map[string]any{
		"audioInputConfiguration": map[string]any{
			keyMediaType:      "audio/lpcm",
			"sampleRateHertz": ac.InputSampleRate,
			"sampleSizeBits":  ac.InputSampleSize,
			"channelCount":    ac.InputChannelCount,
			"audioType":       "SPEECH",
			"encoding":        "base64",
		},
	})
}

// audioInputEvent is one chunk of the user's audio.
func audioInputEvent(audio []byte, prompt, content string) map[string]any {
	return event("audioInput", map[string]any{
		keyPromptName:  prompt,
		keyContentName: content,
		keyContent:     base64.StdEncoding.EncodeToString(audio),
	})
}

// toolResultEvents are the block a tool result is sent as: it is opened naming
// the call it answers, the result is sent, and it is closed.
func toolResultEvents(prompt, toolCallID, result string) []map[string]any {
	content := uuid.NewString()
	return []map[string]any{
		contentStart(prompt, content, "TOOL", "TOOL", false, map[string]any{
			"toolResultInputConfiguration": map[string]any{
				"toolUseId":              toolCallID,
				"type":                   contentTypeTxt,
				"textInputConfiguration": map[string]any{keyMediaType: mediaTypeText},
			},
		}),
		event("toolResult", map[string]any{keyPromptName: prompt, keyContentName: content, keyContent: result}),
		contentEnd(prompt, content),
	}
}

// --- the session sender: the protocol on an explicit stream and prompt, so
// session continuation can address the next session as well as the current ---

// sendText sends a text block to stream.
func (s *Service) sendText(ctx context.Context, text, role, prompt string, stream sessionStream,
	interactive bool,
) error {
	if text == "" || stream == nil || prompt == "" {
		return nil
	}
	for _, ev := range textEvents(text, role, prompt, interactive) {
		if err := sendEvent(ctx, stream, ev); err != nil {
			return err
		}
	}
	return nil
}

// sendPromptStart opens a prompt on stream.
func (s *Service) sendPromptStart(ctx context.Context, tools []map[string]any, prompt string,
	stream sessionStream,
) error {
	return sendEvent(ctx, stream, s.promptStartEvent(tools, prompt))
}

// sendAudioInputStart opens the audio block on stream.
func (s *Service) sendAudioInputStart(ctx context.Context, prompt, content string, stream sessionStream) error {
	return sendEvent(ctx, stream, s.audioInputStartEvent(prompt, content))
}

// sendAudio sends a chunk of the user's audio on stream.
func (s *Service) sendAudio(ctx context.Context, audio []byte, prompt, content string, stream sessionStream) error {
	return sendEvent(ctx, stream, audioInputEvent(audio, prompt, content))
}

// openStream opens a session on client.
func (s *Service) openStream(ctx context.Context, client sessionClient) (sessionStream, error) {
	return client.OpenStream(ctx, s.model())
}

// createClient builds a client to open sessions with.
func (s *Service) createClient(ctx context.Context) (sessionClient, error) {
	return s.newClient(ctx, s.cfg)
}

// audioConfig is the audio the sessions take in and send back.
func (s *Service) audioConfig() AudioConfig { return s.audio }
