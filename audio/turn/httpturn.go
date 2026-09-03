package turn

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"
)

// errHTTPStatus is returned when the remote analyzer answers with a status other
// than 200.
//
//nolint:gochecknoglobals // sentinel error
var errHTTPStatus = errors.New("turn: the end-of-turn service returned an error")

// errNoURL is returned when the analyzer is built without an endpoint to ask.
//
//nolint:gochecknoglobals // sentinel error
var errNoURL = errors.New("turn: the end-of-turn service URL is required")

// httpErrorBodyLimit bounds how much of a failed response is read for the log.
const httpErrorBodyLimit = 4096

// HTTPConfig configures an HTTPSmartTurn analyzer.
type HTTPConfig struct {
	// URL is the endpoint the audio is posted to. Required.
	URL string
	// Headers are sent on every request, on top of the content type. It is where
	// an authorization header belongs.
	Headers map[string]string
	// Client makes the requests; nil uses a client of its own. The request
	// deadline comes from the analysis parameters rather than from the client,
	// so a client configured with its own timeout has whichever is shorter apply.
	Client *http.Client
	// Params configures the analysis; the zero value uses the defaults.
	Params *Params
}

// HTTPSmartTurn is an end-of-turn Analyzer that asks a remote service for the
// prediction, for a deployment that runs the model somewhere other than in the
// bot. It buffers and segments the turn's audio exactly as the local analyzer
// does, and posts that segment for scoring.
//
// A request is given the silence window to answer in, since an answer arriving
// later is an answer about a turn that is already over. A request that times
// out, and any other failure, is reported as an unfinished turn: a service that
// is down leaves every turn to the stop-seconds safety net rather than cutting
// the user off mid-sentence.
type HTTPSmartTurn struct {
	*smartTurnBase
	cfg HTTPConfig
}

// NewHTTPSmartTurn builds an analyzer backed by a remote scoring service.
func NewHTTPSmartTurn(cfg HTTPConfig) (*HTTPSmartTurn, error) {
	if cfg.URL == "" {
		return nil, errNoURL
	}
	params := DefaultParams()
	if cfg.Params != nil {
		params = *cfg.Params
	}
	if cfg.Client == nil {
		cfg.Client = &http.Client{}
	}
	s := &HTTPSmartTurn{cfg: cfg}
	s.smartTurnBase = newSmartTurnBase(s, params)
	return s, nil
}

// Close releases nothing: the analyzer holds no session of its own, and the
// client it posts with belongs to whoever configured it.
func (s *HTTPSmartTurn) Close() error { return nil }

// httpPrediction is the answer the service returns.
type httpPrediction struct {
	// Prediction is 1 for a complete turn and 0 for an unfinished one.
	Prediction int `json:"prediction"`
	// Probability is the model's confidence that the turn is complete, in [0,1].
	Probability float64 `json:"probability"`
}

// predictEndpoint posts the segment for scoring and reads the verdict back. A
// failure is reported as an unfinished turn rather than as an error, so a
// service that is down leaves the turn to the silence safety net.
func (s *HTTPSmartTurn) predictEndpoint(audio []float32) (bool, float64, error) {
	complete, prob, err := s.ask(audio)
	if err != nil {
		slog.Error("the end-of-turn prediction failed", "err", err)
		return false, 0, nil
	}
	return complete, prob, nil
}

// ask makes the request and reads the verdict.
func (s *HTTPSmartTurn) ask(audio []float32) (bool, float64, error) {
	body := encodeNPY(audio)

	// The silence the analyzer already treats as an ending is all the time there
	// is: an answer arriving later is an answer to a turn that is over.
	timeout := time.Duration(s.params.StopSecs * float64(time.Second))
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.URL, bytes.NewReader(body))
	if err != nil {
		return false, 0, err
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	for k, v := range s.cfg.Headers {
		req.Header.Set(k, v)
	}

	resp, err := s.cfg.Client.Do(req)
	if err != nil {
		return false, 0, err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, httpErrorBodyLimit))
		return false, 0, fmt.Errorf("%w: %d: %s", errHTTPStatus, resp.StatusCode, msg)
	}

	var out httpPrediction
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return false, 0, fmt.Errorf("turn: reading the end-of-turn verdict: %w", err)
	}
	return out.Prediction == 1, out.Probability, nil
}

// npyMagic opens a NumPy array file, which is the format the service reads.
const npyMagic = "\x93NUMPY"

// npyAlign is the boundary a NumPy header is padded out to.
const npyAlign = 64

// encodeNPY serializes a float32 slice as a version 1.0 NumPy array file, the
// shape the remote service loads its input from.
func encodeNPY(audio []float32) []byte {
	header := fmt.Sprintf("{'descr': '<f4', 'fortran_order': False, 'shape': (%d,), }", len(audio))
	// The header runs to a 64-byte boundary counting the magic, the version and
	// the length field, and ends in a newline.
	hlen := len(header) + 1
	pad := npyAlign - ((len(npyMagic) + 2 + 2 + hlen) % npyAlign)

	var b bytes.Buffer
	b.Grow(len(npyMagic) + 4 + hlen + pad + len(audio)*4)
	b.WriteString(npyMagic)
	b.WriteByte(1) // major version
	b.WriteByte(0) // minor version
	_ = binary.Write(&b, binary.LittleEndian, uint16(hlen+pad))
	b.WriteString(header)
	b.Write(bytes.Repeat([]byte{' '}, pad))
	b.WriteByte('\n')
	for _, f := range audio {
		_ = binary.Write(&b, binary.LittleEndian, f)
	}
	return b.Bytes()
}

var _ Analyzer = (*HTTPSmartTurn)(nil)
