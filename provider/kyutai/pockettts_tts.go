package kyutai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/tts"
	errs "github.com/gojargo/jargo/utils/errors"
)

// NewPocketTTS builds a Pocket TTS service against a local pocket-tts server.
func NewPocketTTS(cfg PocketTTSConfig) *tts.Base {
	if cfg.SampleRate == 0 {
		cfg.SampleRate = pocketSampleRate
	}
	cfg.BaseURL = strings.TrimSuffix(cfg.BaseURL, "/")
	return tts.New("PocketTTS", &pocketSynthesizer{cfg: cfg, http: &http.Client{}})
}

type pocketSynthesizer struct {
	cfg  PocketTTSConfig
	http *http.Client
}

// SampleRate reports the rate the service expects the server to generate at.
func (s *pocketSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// RunTTS asks the server for one sentence and streams the audio downstream as it
// is generated. The response is a WAV whose header goes out before the samples
// exist, so the header is consumed and everything after it is pushed on as it
// arrives rather than collected first.
func (s *pocketSynthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	body, contentType, err := s.requestBody(text)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.BaseURL+pocketTTSPath, body)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", contentType)
	// Reported as the request is made, ahead of any answer.
	tts.StartUsageMetrics(ctx, text)

	resp, err := s.http.Do(req) //nolint:gosec // request target is the configured local endpoint
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return errs.NewHTTPStatusError(resp.StatusCode, fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg))
	}

	rate, pcm, err := pcmStream(resp.Body)
	if err != nil {
		return err
	}
	if rate != s.cfg.SampleRate {
		slog.Warn("pocket-tts server generated at a different rate than configured",
			"server", rate, "configured", s.cfg.SampleRate)
	}
	trim := newPaddingTrimmer(tts.PCMYielder(yield, rate), rate)
	if err := streamWAV(pcm, trim.write); err != nil {
		return err
	}
	return trim.close()
}

// serverPadding is the silence `pocket-tts serve` appends to every response
// after the last generated sample, "to ensure proper playback" of a file. In a
// stream it is a pause after every sentence the model never produced, and the
// bot's turn is reported over that much later.
const serverPadding = 200 * time.Millisecond

// paddingTrimmer passes audio on except the server's padding, so what reaches
// the pipeline is what the model generated. It holds back trailing zero bytes,
// up to the length of the padding, until either more audio follows (they were
// part of the speech, and go out ahead of it) or the stream ends (they were the
// padding, and are dropped).
type paddingTrimmer struct {
	emit func(pcm []byte) error
	// held is the run of zero bytes at the end of what has arrived and not been
	// passed on; max is the most of it that can be padding.
	held, max int
}

func newPaddingTrimmer(emit func(pcm []byte) error, rate int) *paddingTrimmer {
	return &paddingTrimmer{emit: emit, max: int(int64(rate) * 2 * int64(serverPadding) / int64(time.Second))}
}

// write takes the next chunk of the stream.
func (p *paddingTrimmer) write(chunk []byte) error {
	last := lastNonZero(chunk)
	if last < 0 {
		p.held += len(chunk)
		return p.release(p.held - p.max)
	}
	if err := p.release(p.held); err != nil {
		return err
	}
	if err := p.emit(chunk[:last+1]); err != nil {
		return err
	}
	p.held = len(chunk) - last - 1
	return p.release(p.held - p.max)
}

// close ends the stream, dropping the padding. An odd run of zeros ends on the
// high byte of the last sample, which goes out so the samples stay whole.
func (p *paddingTrimmer) close() error {
	return p.release(p.held % 2)
}

// release passes n of the held zero bytes on.
func (p *paddingTrimmer) release(n int) error {
	if n <= 0 {
		return nil
	}
	p.held -= n
	return p.emit(make([]byte, n))
}

// lastNonZero is the index of the last byte of b that is not zero, or -1.
func lastNonZero(b []byte) int {
	for i, v := range slices.Backward(b) {
		if v != 0 {
			return i
		}
	}
	return -1
}

// requestBody builds the multipart form the server expects: the text to speak,
// and the voice when one is configured. Leaving the voice out is what asks the
// server for the default voice of the language it was started with.
func (s *pocketSynthesizer) requestBody(text string) (io.Reader, string, error) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	if err := w.WriteField("text", text); err != nil {
		return nil, "", err
	}
	if s.cfg.Voice != "" {
		if err := w.WriteField("voice_url", s.cfg.Voice); err != nil {
			return nil, "", err
		}
	}
	if err := w.Close(); err != nil {
		return nil, "", err
	}
	return &buf, w.FormDataContentType(), nil
}

// stream pushes the PCM downstream in chunks until the audio ends.
func streamWAV(r io.Reader, emit func(pcm []byte) error) error {
	buf := make([]byte, readChunk)
	for {
		n, err := r.Read(buf)
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

// pcmStream consumes the RIFF/WAVE header from r, returning the sample rate it
// declares and a reader positioned at the first sample.
//
// The chunks are walked rather than a fixed header skipped, so anything sitting
// between the format and the samples is stepped over. The data chunk's declared
// length is ignored: the header is written before the audio exists, so it
// carries a placeholder, and the samples run to the end of the response.
func pcmStream(r io.Reader) (int, io.Reader, error) {
	br := bufio.NewReader(r)

	var riff [12]byte
	if _, err := io.ReadFull(br, riff[:]); err != nil {
		return 0, nil, errBadWAV
	}
	if string(riff[0:4]) != "RIFF" || string(riff[8:12]) != "WAVE" {
		return 0, nil, errBadWAV
	}

	rate := 0
	for {
		var head [8]byte
		if _, err := io.ReadFull(br, head[:]); err != nil {
			return 0, nil, errBadWAV
		}
		id := string(head[0:4])
		size := int64(binary.LittleEndian.Uint32(head[4:8]))

		if id == "data" {
			if rate == 0 {
				return 0, nil, errBadWAV
			}
			return rate, br, nil
		}

		payload := make([]byte, size)
		if _, err := io.ReadFull(br, payload); err != nil {
			return 0, nil, errBadWAV
		}
		if id == "fmt " && len(payload) >= 8 {
			rate = int(binary.LittleEndian.Uint32(payload[4:8]))
		}
		if size%2 == 1 {
			// Chunks are word-aligned, so an odd one is followed by a pad byte.
			if _, err := br.Discard(1); err != nil {
				return 0, nil, errBadWAV
			}
		}
	}
}
