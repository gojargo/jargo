// Package ttsstream holds the loops the HTTP text-to-speech providers share:
// reading a raw PCM response body in chunks and turning the chunks into audio
// frames.
package ttsstream

import (
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/gojargo/jargo/errclass"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/tts"
)

// errStatus is returned when a provider responds with a non-200 status.
//
//nolint:gochecknoglobals // sentinel error
var errStatus = errors.New("ttsstream: unexpected status")

// readChunk is the size of each audio read from an HTTP stream.
const readChunk = 4096

// PCMYielder adapts a frame yield to a callback taking raw 16-bit mono PCM, for
// a provider whose output is audio and nothing else.
func PCMYielder(yield func(f frames.Frame) error, rate int) func(pcm []byte) error {
	return func(pcm []byte) error {
		if len(pcm) == 0 {
			return nil
		}
		return yield(frames.NewTTSAudioRawFrame(pcm, rate, 1))
	}
}

// Response issues req and streams the raw-PCM response body to emit in
// chunks. It is the shared body-reading loop for HTTP TTS providers. Pair it
// with PCMYielder to turn the chunks into frames.
func Response(client *http.Client, req *http.Request, emit func(pcm []byte) error) error {
	return streamResponse(client, req, nil, emit)
}

// Speech is Response for a request synthesizing text: once the
// provider accepts the request, text is reported as usage (see
// tts.StartUsageMetrics), through the context req was built with.
func Speech(client *http.Client, req *http.Request, text string, emit func(pcm []byte) error) error {
	return streamResponse(client, req, &text, emit)
}

func streamResponse(client *http.Client, req *http.Request, text *string, emit func(pcm []byte) error) error {
	resp, err := client.Do(req) //nolint:gosec // request target is the service's configured endpoint
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return errclass.NewHTTPStatusError(resp.StatusCode, fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg))
	}
	if text != nil {
		tts.StartUsageMetrics(req.Context(), *text)
	}
	buf := make([]byte, readChunk)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			chunk := make([]byte, n)
			copy(chunk, buf[:n])
			if perr := emit(chunk); perr != nil {
				return perr
			}
		}
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
	}
}
