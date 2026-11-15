package rime

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/rime/internal/rimepb"
	errs "github.com/gojargo/jargo/utils/errors"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// Tests of the client against a loopback gRPC peer: the wire format, the
// protocol checks and the status handling are the real ones, and only the
// connection is redirected. The channel is plaintext, and no credential of
// Rime's is used.

// peerFailure ends a call with a gRPC status.
type peerFailure struct {
	code   codes.Code
	detail string
}

// peerCall is one stream the peer was opened with.
type peerCall struct {
	requestID     string
	config        *rimepb.StreamingConfig
	authorization string

	mu    sync.Mutex
	audio [][]byte

	receivedAudio chan struct{}
	eof           chan struct{}
	closed        chan struct{}
	// responses are what the peer sends: a response, a peerFailure, or nil to
	// end the call.
	responses chan any
	text      string
	revision  uint64
}

// pcm is everything the call was sent.
func (c *peerCall) pcm() []byte {
	c.mu.Lock()
	defer c.mu.Unlock()
	return bytes.Join(c.audio, nil)
}

func resolved(tag string) *rimepb.ResolvedLanguage {
	return &rimepb.ResolvedLanguage{Tag: tag, Source: rimepb.LanguageSource_LANGUAGE_SOURCE_SELECTED}
}

// update sends a transcript update: an appended fragment, or a new hypothesis.
func (c *peerCall) update(text string, appending bool) {
	if appending {
		c.text += text
		c.responses <- &rimepb.StreamingTranscriptionResponse{
			Payload: &rimepb.StreamingTranscriptionResponse_Delta{Delta: &rimepb.TranscriptionDelta{Text: text}},
		}
		return
	}
	c.text = text
	c.revision++
	c.responses <- &rimepb.StreamingTranscriptionResponse{
		Payload: &rimepb.StreamingTranscriptionResponse_Hypothesis{
			Hypothesis: &rimepb.TranscriptionHypothesis{Text: text, Revision: c.revision},
		},
	}
}

// finish sends the final transcript, the text so far unless text overrides
// it, and ends the call.
func (c *peerCall) finish(text *string) {
	final := c.text
	if text != nil {
		final = *text
	}
	c.responses <- &rimepb.StreamingTranscriptionResponse{
		Payload: &rimepb.StreamingTranscriptionResponse_Done{Done: &rimepb.TranscriptionDone{
			Text: final, Revision: c.revision, Language: resolved(c.config.GetLanguage()),
		}},
	}
	c.responses <- nil
}

// peer serves streaming recognition with responses the test controls.
type peer struct {
	rimepb.UnimplementedSpeechToTextServer

	mu       sync.Mutex
	calls    []*peerCall
	requests chan *peerCall
}

func (p *peer) callCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.calls)
}

func (p *peer) TranscribeStreaming(
	srv grpc.BidiStreamingServer[rimepb.StreamingTranscriptionRequest, rimepb.StreamingTranscriptionResponse],
) error {
	first, err := srv.Recv()
	if err != nil {
		return err
	}
	if first.GetConfig() == nil {
		return status.Error(codes.InvalidArgument, "Configuration must arrive first")
	}
	md, _ := metadata.FromIncomingContext(srv.Context())
	p.mu.Lock()
	call := &peerCall{
		requestID:     fmt.Sprintf("peer-%d", len(p.calls)),
		config:        first.GetConfig(),
		authorization: strings.Join(md.Get("authorization"), ","),
		receivedAudio: make(chan struct{}),
		eof:           make(chan struct{}),
		closed:        make(chan struct{}),
		responses:     make(chan any, 16),
	}
	p.calls = append(p.calls, call)
	p.mu.Unlock()
	defer close(call.closed)
	if err := srv.SendHeader(metadata.Pairs(requestIDHeader, call.requestID)); err != nil {
		return err
	}
	p.requests <- call
	if err := srv.Send(&rimepb.StreamingTranscriptionResponse{
		Payload: &rimepb.StreamingTranscriptionResponse_Accepted{Accepted: &rimepb.StreamingAccepted{
			OutputContract: call.config.GetOutputContract(), Language: resolved(call.config.GetLanguage()),
		}},
	}); err != nil {
		return err
	}

	go func() {
		var once sync.Once
		for {
			req, err := srv.Recv()
			if errors.Is(err, io.EOF) {
				close(call.eof)
				return
			}
			if err != nil {
				return
			}
			call.mu.Lock()
			call.audio = append(call.audio, req.GetAudio())
			call.mu.Unlock()
			once.Do(func() { close(call.receivedAudio) })
		}
	}()

	for {
		select {
		case r := <-call.responses:
			switch v := r.(type) {
			case peerFailure:
				return status.Error(v.code, v.detail)
			case *rimepb.StreamingTranscriptionResponse:
				if err := srv.Send(v); err != nil {
					return err
				}
			case nil:
				return nil
			}
		case <-srv.Context().Done():
			return srv.Context().Err()
		}
	}
}

// next waits for the next call.
func (p *peer) next(t *testing.T) *peerCall {
	t.Helper()
	select {
	case c := <-p.requests:
		return c
	case <-time.After(waitTimeout):
		t.Fatal("no call reached the peer")
		return nil
	}
}

// peerServer starts a peer, and returns it with a factory for clients
// connected to it.
func peerServer(t *testing.T) (*peer, func(string) (transcriber, error)) {
	t.Helper()
	p := &peer{requests: make(chan *peerCall, 16)}
	lis, err := (&net.ListenConfig{}).Listen(t.Context(), "tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server := grpc.NewServer()
	rimepb.RegisterSpeechToTextServer(server, p)
	go func() { _ = server.Serve(lis) }()
	t.Cleanup(server.Stop)
	return p, func(apiKey string) (transcriber, error) {
		conn, err := grpc.NewClient(lis.Addr().String(),
			grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithDisableRetry())
		if err != nil {
			return nil, err
		}
		return newSTTClient(apiKey, conn)
	}
}

// startSpeech starts a stretch of speech and waits for its call to carry
// audio.
func startSpeech(t *testing.T, r *sttRun, p *peer) *peerCall {
	t.Helper()
	r.send(vadStart(), audioFrame(pcm(1), 16000))
	call := p.next(t)
	waitClosed(t, "the call carried no audio", call.receivedAudio)
	return call
}

// Ported from upstream (test_streaming_config_pcm_and_transcript_contract).
func TestSTTStreamingConfigPCMAndTranscriptContract(t *testing.T) {
	for _, contract := range []string{contractRevised, contractAppendFragments} {
		for _, rate := range sttSupportedSampleRates {
			t.Run(fmt.Sprintf("%s/%d", contract, rate), func(t *testing.T) {
				p, newClient := peerServer(t)
				r := runSTTWith(t, STTConfig{
					APIKey:         "local-peer",
					SampleRate:     rate,
					Language:       "es-MX",
					Mode:           modeVerbatim,
					ContextTerms:   []string{"jargo", "Rime"},
					OutputContract: contract,
				}, newClient)
				r.waitConnected(t)
				samples := make([]byte, 0, rate/10*2)
				for range rate / 10 {
					samples = binary.LittleEndian.AppendUint16(samples, 1000)
				}
				r.send(vadStart(), frames.NewUserAudioRawFrame("speaker", samples, rate, 1))
				call := p.next(t)
				waitClosed(t, "the call carried no audio", call.receivedAudio)

				cfg := call.config
				if cfg.GetLanguage() != "es-MX" || cfg.GetMode() != rimepb.TranscriptionMode_TRANSCRIPTION_MODE_VERBATIM ||
					!slices.Equal(cfg.GetContextTerms(), []string{"jargo", "Rime"}) ||
					cfg.GetOutputContract() != outputContractProto(contract) {
					t.Errorf("config = %v, want the configured settings", cfg)
				}
				if call.authorization != "Bearer local-peer" {
					t.Errorf("authorization = %q, want the Bearer key", call.authorization)
				}
				select {
				case <-call.eof:
					t.Fatal("the input ended before the speech did")
				default:
				}
				appending := contract == contractAppendFragments
				call.update("hola ", appending)
				r.waitInterims(t, 1)
				if appending {
					call.update("mundo", true)
				} else {
					call.update("hola mundo", false)
				}
				r.waitInterims(t, 2)
				r.send(vadStop(0))
				waitClosed(t, "the input never ended", call.eof)
				call.finish(nil)
				final := r.waitTranscripts(t, 1)[0]
				if final.Text != "hola mundo" || !final.Finalized {
					t.Errorf("final = %q (finalized %t), want hola mundo finalized", final.Text, final.Finalized)
				}
				if final.UserID != "speaker" {
					t.Errorf("user = %q, want speaker", final.UserID)
				}
				if got := r.interims(); !slices.Equal(got, []string{"hola ", "hola mundo"}) {
					t.Errorf("interims = %q, want them cumulative", got)
				}

				wire := call.pcm()
				n := len(wire) / 2
				if n < 1599 || n > 1601 {
					t.Errorf("the peer was sent %d samples, want 100 ms at 16 kHz", n)
				}
				var sum float64
				for i := 100; i < n-100; i++ {
					sum += float64(int16(binary.LittleEndian.Uint16(wire[2*i:])))
				}
				if mean := sum / float64(n-200); math.Abs(mean-1000) > 1 {
					t.Errorf("mean sample = %v, want 1000", mean)
				}
			})
		}
	}
}

// Ported from upstream (test_later_reply_waits_for_the_earlier_final).
func TestSTTLaterReplyWaitsForTheEarlierFinal(t *testing.T) {
	p, newClient := peerServer(t)
	r := runSTTWith(t, STTConfig{APIKey: "local-peer"}, newClient)
	r.waitConnected(t)
	var calls []*peerCall
	for range 2 {
		call := startSpeech(t, r, p)
		r.send(vadStop(0))
		waitClosed(t, "the input never ended", call.eof)
		calls = append(calls, call)
	}
	first, second := calls[0], calls[1]
	second.update("second", false)
	second.finish(nil)
	waitClosed(t, "the second call never closed", second.closed)
	time.Sleep(50 * time.Millisecond)
	if n := len(r.transcripts()); n != 0 {
		t.Fatalf("transcripts = %d, want the later final held back", n)
	}
	first.update("first", false)
	first.finish(nil)
	finals := r.waitTranscripts(t, 2)
	if got := texts(finals); !slices.Equal(got, []string{"first", "second"}) {
		t.Errorf("transcripts = %q, want them in speech order", got)
	}
	if got := finalized(finals); !slices.Equal(got, []bool{false, true}) {
		t.Errorf("finalized = %v, want only the latest", got)
	}
}

// Ported from upstream (test_grpc_failure_category_and_request_id).
func TestSTTGRPCFailureCategoryAndRequestID(t *testing.T) {
	cases := []struct {
		code     codes.Code
		category errs.Category
		usable   bool
	}{
		{codes.Unauthenticated, errs.Authentication, false},
		{codes.PermissionDenied, errs.Authorization, false},
		{codes.InvalidArgument, errs.InvalidRequest, false},
		{codes.ResourceExhausted, errs.Unknown, true},
		{codes.Unavailable, errs.Connectivity, true},
		{codes.DeadlineExceeded, errs.Connectivity, true},
		{codes.Internal, errs.Unknown, true},
	}
	for _, c := range cases {
		t.Run(c.code.String(), func(t *testing.T) {
			p, newClient := peerServer(t)
			r := runSTTWith(t, STTConfig{APIKey: "local-peer"}, newClient)
			r.waitConnected(t)
			call := startSpeech(t, r, p)
			call.update("unfinished", false)
			r.waitInterims(t, 1)
			call.responses <- peerFailure{code: c.code, detail: "Private recognition hints"}
			ef := r.waitErrors(t, 1)[0]
			if ef.Category != c.category || !strings.Contains(ef.Error, call.requestID) {
				t.Errorf("error = %q (%v), want %v naming %s", ef.Error, ef.Category, c.category, call.requestID)
			}
			if strings.Contains(ef.Error, "Private recognition hints") {
				t.Errorf("error = %q, want the details left out", ef.Error)
			}
			if r.c.svc.Usable() != c.usable {
				t.Errorf("usable = %t, want %t", r.c.svc.Usable(), c.usable)
			}
			if n := len(r.transcripts()); n != 0 {
				t.Errorf("transcripts = %d, want none", n)
			}
			waitClosed(t, "the call never closed", call.closed)
			if n := p.callCount(); n != 1 {
				t.Errorf("calls = %d, want one", n)
			}
		})
	}
}

// Ported from upstream
// (test_invalid_completion_reports_an_error_without_a_final).
func TestSTTInvalidCompletionReportsAnErrorWithoutAFinal(t *testing.T) {
	for _, malformed := range []string{"missing_final", "mismatched_final"} {
		t.Run(malformed, func(t *testing.T) {
			p, newClient := peerServer(t)
			r := runSTTWith(t, STTConfig{APIKey: "local-peer"}, newClient)
			r.waitConnected(t)
			call := startSpeech(t, r, p)
			call.update("partial", false)
			r.waitInterims(t, 1)
			r.send(vadStop(0))
			waitClosed(t, "the input never ended", call.eof)
			if malformed == "missing_final" {
				call.responses <- nil
			} else {
				different := "different"
				call.finish(&different)
			}
			ef := r.waitErrors(t, 1)[0]
			if !strings.Contains(ef.Error, "RimeStreamError") {
				t.Errorf("error = %q, want a stream error", ef.Error)
			}
			if n := len(r.transcripts()); n != 0 {
				t.Errorf("transcripts = %d, want none", n)
			}
		})
	}
}

// Audio at another rate than 16 kHz is converted by linear interpolation with
// no delay: a constant signal stays constant from its first sample, and the
// length scales with the rate.
func TestSTTRateConversionKeepsPaceWithNoDelay(t *testing.T) {
	for _, rate := range []int{8000, 24000, 48000} {
		c := newRateConverter(rate, 16000)
		var in []byte
		for range rate / 100 {
			in = binary.LittleEndian.AppendUint16(in, uint16(0xFC18)) // -1000
		}
		out := c.process(in[:len(in)/2])
		out = append(out, c.process(in[len(in)/2:])...)
		if n := len(out) / 2; n < 159 || n > 161 {
			t.Errorf("%d Hz: %d samples out, want 10 ms at 16 kHz", rate, n)
		}
		for i := 0; i+1 < len(out); i += 2 {
			if v := int16(binary.LittleEndian.Uint16(out[i:])); v != -1000 {
				t.Fatalf("%d Hz: sample %d = %d, want -1000", rate, i/2, v)
			}
		}
	}
}

// Ported from upstream (test_cleanup_closes_the_sdk_client).
func TestSTTCleanupClosesTheClient(t *testing.T) {
	_, newClient := peerServer(t)
	client, err := newClient("local-peer")
	if err != nil {
		t.Fatal(err)
	}
	if err := client.close(); err != nil {
		t.Fatal(err)
	}
	q := newAudioQueue()
	q.put(make([]byte, 320))
	q.close()
	_, err = client.stream(context.Background(), streamOptions{language: "en", sampleRate: 16000}, q,
		func(transcriptUpdate) {})
	re := asRimeError(err)
	if re == nil || re.kind != kindInput || !strings.Contains(re.message, "closed") {
		t.Errorf("stream after close = %v, want an input error naming the closed client", err)
	}
}
