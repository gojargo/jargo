package evalws

import (
	"context"
	"log/slog"
	"net/http"
	"net/url"
	"strings"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport/wsserver"
)

// The query parameters a harness connecting to an eval server sets on the
// connection URL, one per flag. SkipTTSQueryParam, CaptureAudioQueryParam and
// CaptureImagesQueryParam carry the ConnectionFlags of the same names.
// TriggerDisconnectQueryParam asks the server to fire the bot's client
// disconnected event when the connection ends.
const (
	SkipTTSQueryParam           = "skip_tts"
	CaptureAudioQueryParam      = "capture_bot_audio"
	CaptureImagesQueryParam     = "capture_bot_images"
	TriggerDisconnectQueryParam = "trigger_disconnect"
)

// queryString is the query string of the connection's URL: the parsed URL's
// when there is one, and otherwise the raw request target's.
func queryString(r *http.Request) string {
	if r == nil {
		return ""
	}
	if r.URL != nil {
		return r.URL.RawQuery
	}
	u, err := url.ParseRequestURI(r.RequestURI)
	if err != nil {
		return ""
	}
	return u.RawQuery
}

// QueryFlag reports whether the client's connection URL set the boolean query
// parameter name: to 1, true or yes, in any case.
func QueryFlag(r *http.Request, name string) bool {
	values, _ := url.ParseQuery(queryString(r))
	v := values[name]
	if len(v) == 0 {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(v[0])) {
	case "1", "true", "yes":
		return true
	default:
		return false
	}
}

// TransportParams configures the eval transport, so a bot's runner transport
// parameters name it as such: the WebSocket transport's parameters and the
// serializer.
type TransportParams struct {
	wsserver.Params
	// Serializer is the wire format. Nil uses a NewSerializer.
	Serializer wsserver.Serializer
}

// InputTransport is the eval transport's input: it serves the harness's image.
//
// A vision bot asks for the user's camera image; under eval there is no camera,
// so the image the harness registered for the turn is served instead. With no
// image registered, the request is answered with an error, so the tool call
// that made it still finishes.
type InputTransport struct {
	*wsserver.InputTransport

	params TransportParams
	// pushFrame sends a frame downstream.
	pushFrame func(ctx context.Context, f frames.Frame) error
}

func newInputTransport(t *wsserver.Transport, params TransportParams) *InputTransport {
	in := &InputTransport{params: params}
	in.InputTransport = wsserver.NewInputTransport(t, in)
	in.pushFrame = func(ctx context.Context, f frames.Frame) error {
		return in.PushFrame(ctx, f, processor.Downstream)
	}
	return in
}

// ProcessFrame serves image requests; otherwise it behaves like the WebSocket
// input.
func (in *InputTransport) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := in.InputTransport.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if request, ok := f.(*frames.UserImageRequestFrame); ok {
		return in.serveUserImage(ctx, request)
	}
	return nil
}

func (in *InputTransport) serveUserImage(ctx context.Context, request *frames.UserImageRequestFrame) error {
	var image *frames.UserImageRawFrame
	if s, ok := in.params.Serializer.(*Serializer); ok {
		var err error
		if image, err = s.UserImageFrame(request); err != nil {
			return err
		}
	}
	if image == nil {
		in.AnswerImageRequests(ctx, []*frames.UserImageRequestFrame{request}, "No user image is registered.")
		return nil
	}
	return in.pushFrame(ctx, image)
}

// OutputTransport is the eval transport's output: it reports the bot's images
// to the harness.
//
// An image the bot outputs is handed to the serializer as it arrives, whether
// or not the bot enabled video output; the serializer sends it only when the
// harness asked.
type OutputTransport struct {
	*wsserver.OutputTransport
}

func newOutputTransport(t *wsserver.Transport) *OutputTransport {
	out := &OutputTransport{}
	out.OutputTransport = wsserver.NewOutputTransport(t, out)
	return out
}

// ProcessFrame reports output images; otherwise it behaves like the WebSocket
// output.
func (out *OutputTransport) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := out.OutputTransport.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(frames.OutputImageFrame); ok {
		if _, err := out.WriteFrame(ctx, f); err != nil {
			return err
		}
	}
	return nil
}

// Transport is the eval transport over a WebSocket the bot accepted itself.
//
// On a hosted deployment a session's WebSocket arrives already accepted, and
// the hops in between pass on only the session body, never the URL's query. The
// harness therefore sends its connection's flags as its first message,
// eval-connect, which the runner's CreateTransport reads before building this
// transport with them. SkipTTS is applied before the client connected event
// fires, so a greeting made there is silent too.
//
// The connection is the session, so the client disconnected event fires
// whenever the harness disconnects, letting the bot end with it.
type Transport struct {
	*wsserver.Transport

	params TransportParams
	flags  ConnectionFlags
	input  *InputTransport
	output *OutputTransport
}

// NewTransport builds the eval transport on conn, a WebSocket already accepted
// with its eval-connect message read, for the connection flags says. pending
// are messages already read off the socket, as for wsserver.NewTransport; a
// replayed eval-connect among them is consumed by the serializer.
func NewTransport(conn *websocket.Conn, params TransportParams, flags ConnectionFlags, pending ...[]byte) *Transport {
	if params.Serializer == nil {
		params.Serializer = NewSerializer()
	}
	ws := wsserver.NewTransport(conn, params.Serializer, params.Params, pending...)
	t := &Transport{
		Transport: ws,
		params:    params,
		flags:     flags,
		input:     newInputTransport(ws, params),
		output:    newOutputTransport(ws),
	}
	ws.SetInput(t.input.InputTransport)
	ws.SetOutput(t.output.OutputTransport)

	if s, ok := params.Serializer.(*Serializer); ok {
		s.SetCaptureAudio(flags.CaptureBotAudio)
		s.SetCaptureImages(flags.CaptureBotImages)
	}

	// Added before any handler the bot adds, and the handlers of one event run
	// in the order they were added, so the output is configured before a
	// greeting the bot makes on connect.
	t.Events().Add(wsserver.EventClientConnected, func(ctx context.Context, _ any, _ ...any) {
		t.onClientConnected(ctx)
	})
	return t
}

// Params returns the parameters the transport was built with, its serializer
// filled in.
func (t *Transport) Params() TransportParams { return t.params }

// onClientConnected configures the bot's output for the connection, ahead of
// the bot's own handlers (configuration before any greeting).
func (t *Transport) onClientConnected(ctx context.Context) {
	skipTTS := t.flags.SkipTTS
	slog.Debug("evalws: configuring eval LLM output", "skip_tts", skipTTS)
	if err := t.input.pushFrame(ctx, frames.NewLLMConfigureOutputFrame(skipTTS)); err != nil {
		slog.Warn("evalws: could not configure the LLM output", "err", err)
	}
}
