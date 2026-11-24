package rime

import (
	"context"
	"crypto/tls"
	"errors"
	"io"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gojargo/jargo/provider/rime/internal/rimepb"
	"github.com/gojargo/jargo/provider/rime/internal/rimepcm"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// The transport and resource bounds of Rime's streaming recognition.
const (
	// sttTarget is Rime's speech-to-text gRPC endpoint.
	sttTarget = "stt.api.rime.ai:443"
	// sttConnectionTimeout bounds waiting for the connection to be ready.
	sttConnectionTimeout = 10 * time.Second
	// sttAcceptanceTimeout bounds waiting for Rime to accept a stream's
	// configuration.
	sttAcceptanceTimeout = 10 * time.Second
	// sttCompletionTimeout bounds waiting for the final transcript once the
	// audio has ended.
	sttCompletionTimeout = 120 * time.Second
	// sttReceiveBytes is the largest response message accepted.
	sttReceiveBytes = 1048576 + 65536
	// sttSendBytes is the largest request message sent.
	sttSendBytes = 16*1024*1024 + 65536
	// sttTranscriptBytes bounds the transcript of one stream.
	sttTranscriptBytes = 1048576
	// sttSourceBytes bounds how much audio is converted at once, however large
	// the chunk the audio arrives in.
	sttSourceBytes = 16384
	// sttMessageBytes is the most audio one request message carries.
	sttMessageBytes = 65536
	// sttWireRate is the rate Rime takes audio at: every stream's audio is
	// converted to it.
	sttWireRate = rimepcm.WireRate
	// requestIDHeader names the metadata that carries Rime's request identity.
	requestIDHeader = "x-request-id"
	// errorKindHeader names the metadata that carries Rime's structured reason
	// for a refusal.
	errorKindHeader = "x-rime-error-kind"
	// kindInsufficientCredits is the structured reason for refusing an account
	// that has run out of credits.
	kindInsufficientCredits = "insufficient_credits"
)

// The values the stream configuration names.
const (
	modeWritten             = "written"
	modeVerbatim            = "verbatim"
	contractRevised         = "revised_hypotheses"
	contractAppendFragments = "append_fragments"
)

// errorKind is what a Rime failure means, the way Rime's own client sorts its
// failures.
type errorKind int

const (
	kindStream errorKind = iota
	kindAuthentication
	kindPermission
	kindBilling
	kindInput
	kindResourceLimit
	kindUnavailable
	kindTimeout
	kindAudioFormat
	kindCancelled
)

// String names the kind as Rime's own client names the failure, which is what
// an error report names it by.
func (k errorKind) String() string {
	switch k {
	case kindAuthentication:
		return "RimeAuthenticationError"
	case kindPermission:
		return "RimePermissionError"
	case kindBilling:
		return "RimeBillingError"
	case kindInput:
		return "RimeInputError"
	case kindResourceLimit:
		return "RimeResourceLimitError"
	case kindUnavailable:
		return "RimeUnavailableError"
	case kindTimeout:
		return "RimeTimeoutError"
	case kindAudioFormat:
		return "RimeAudioFormatError"
	case kindCancelled:
		return "RimeCancelledError"
	case kindStream:
		return "RimeStreamError"
	default:
		return "RimeStreamError"
	}
}

// rimeError is a failure of a Rime request, with the identity Rime gave the
// request when it is known.
type rimeError struct {
	kind      errorKind
	message   string
	requestID string
}

func newRimeError(kind errorKind, message string) *rimeError {
	return &rimeError{kind: kind, message: message}
}

func (e *rimeError) Error() string { return e.message }

// asRimeError is err as a Rime failure, or nil when it is not one.
func asRimeError(err error) *rimeError {
	if re, ok := errors.AsType[*rimeError](err); ok {
		return re
	}
	return nil
}

// serverKind is the structured reason for a refusal the headers carry, else the
// one the trailers carry.
func serverKind(headers, trailers metadata.MD) string {
	if v := headers.Get(errorKindHeader); len(v) > 0 && v[0] != "" {
		return v[0]
	}
	if v := trailers.Get(errorKindHeader); len(v) > 0 {
		return v[0]
	}
	return ""
}

// rpcError is the failure a gRPC status means. serverKind is Rime's structured
// reason for it, which tells a refusal for exhausted credits from any other
// denied permission.
func rpcError(code codes.Code, requestID, details, serverKind string) *rimeError {
	kind := kindStream
	switch code {
	case codes.Unauthenticated:
		kind = kindAuthentication
	case codes.PermissionDenied:
		kind = kindPermission
	case codes.InvalidArgument:
		kind = kindInput
	case codes.ResourceExhausted:
		kind = kindResourceLimit
	case codes.Unavailable:
		kind = kindUnavailable
	case codes.DeadlineExceeded:
		kind = kindTimeout
	case codes.Canceled:
		kind = kindCancelled
	default:
		// Every other status is a stream failure.
	}
	if code == codes.PermissionDenied && serverKind == kindInsufficientCredits {
		kind = kindBilling
	}
	message := details
	if strings.TrimSpace(message) == "" {
		message = "Rime operation failed: " + strings.ToUpper(code.String())
	}
	return &rimeError{kind: kind, message: message, requestID: requestID}
}

// requestID is the request identity the headers carry, else the one already
// known, else the one the trailers carry.
func requestID(headers, trailers metadata.MD, previous string) string {
	if v := headers.Get(requestIDHeader); len(v) > 0 && v[0] != "" {
		return v[0]
	}
	if previous != "" {
		return previous
	}
	if v := trailers.Get(requestIDHeader); len(v) > 0 {
		return v[0]
	}
	return ""
}

// updateKind tells the transcript updates of a stream apart.
type updateKind int

const (
	// updatePartial is a replacement snapshot: the whole transcript so far.
	updatePartial updateKind = iota
	// updateDelta is a permanent fragment to append to the transcript.
	updateDelta
)

// transcriptUpdate is an interim transcript update.
type transcriptUpdate struct {
	kind updateKind
	text string
}

// transcriptFinal is the completed transcript of a stream.
type transcriptFinal struct {
	text string
	// language is the BCP-47 tag Rime resolved the language to.
	language string
}

// streamOptions configures one recognition stream.
type streamOptions struct {
	language       string
	mode           string
	contextTerms   []string
	outputContract string
	// sampleRate is the rate of the audio the stream is given, one of 8, 16,
	// 24 or 48 kHz. It is converted to 16 kHz on the way out.
	sampleRate int
}

// audioSource is where a stream reads its audio from, until it ends.
type audioSource interface {
	// next returns the next chunk, or false once the audio has ended.
	next(ctx context.Context) ([]byte, bool, error)
}

// transcriber is what a session streams its segments on: Rime's client.
type transcriber interface {
	// stream transcribes one utterance (see sttClient.stream).
	stream(ctx context.Context, opts streamOptions, source audioSource,
		onUpdate func(transcriptUpdate)) (transcriptFinal, error)
	// close releases the client.
	close() error
}

// sttClient is a connection to Rime's speech recognition, which any number of
// streams share.
type sttClient struct {
	apiKey string
	conn   *grpc.ClientConn

	mu     sync.Mutex
	closed bool
}

// dialSTT opens the connection to Rime's endpoint. It does no I/O: the
// connection is established by the first stream.
func dialSTT() (*grpc.ClientConn, error) {
	creds := credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12})
	return grpc.NewClient(sttTarget, grpc.WithTransportCredentials(creds), grpc.WithDisableRetry())
}

// newRimeSTTClient creates a client on Rime's endpoint, authenticating with
// apiKey.
func newRimeSTTClient(apiKey string) (transcriber, error) {
	conn, err := dialSTT()
	if err != nil {
		return nil, err
	}
	client, err := newSTTClient(apiKey, conn)
	if err != nil {
		_ = conn.Close()
		return nil, err
	}
	return client, nil
}

// newSTTClient builds a client authenticating with apiKey over conn.
func newSTTClient(apiKey string, conn *grpc.ClientConn) (*sttClient, error) {
	if strings.TrimSpace(apiKey) == "" {
		return nil, newRimeError(kindAuthentication, "Provide an API key")
	}
	return &sttClient{apiKey: apiKey, conn: conn}, nil
}

// close closes the connection. A stream started afterwards fails.
func (c *sttClient) close() error {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil
	}
	c.closed = true
	c.mu.Unlock()
	return c.conn.Close()
}

func (c *sttClient) checkOpen() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return newRimeError(kindInput, "The Rime client is closed")
	}
	return nil
}

// ready waits for the connection to be ready, for as long as
// sttConnectionTimeout allows.
func (c *sttClient) ready(ctx context.Context) error {
	waitCtx, cancel := context.WithTimeout(ctx, sttConnectionTimeout)
	defer cancel()
	c.conn.Connect()
	for {
		state := c.conn.GetState()
		switch state {
		case connectivity.Ready:
			return nil
		case connectivity.Shutdown:
			return newRimeError(kindInput, "The Rime client is closed")
		default:
			// The other states are waited out.
		}
		if !c.conn.WaitForStateChange(waitCtx, state) {
			if ctx.Err() != nil {
				return newRimeError(kindCancelled, "Transcription canceled")
			}
			return newRimeError(kindTimeout, "STT connection establishment timed out")
		}
	}
}

// stream transcribes one utterance: it sends the audio source holds until the
// source ends, which asks Rime for the final transcript, and hands each interim
// update to onUpdate as it arrives. It returns the final transcript, or the
// first failure, which is a *rimeError. Canceling ctx cancels the stream.
func (c *sttClient) stream(
	ctx context.Context, opts streamOptions, source audioSource, onUpdate func(transcriptUpdate),
) (transcriptFinal, error) {
	if err := c.checkOpen(); err != nil {
		return transcriptFinal{}, err
	}
	if err := c.ready(ctx); err != nil {
		return transcriptFinal{}, err
	}
	call := &transcriptionCall{opts: opts, source: source, onUpdate: onUpdate, accepted: make(chan struct{})}
	return call.run(ctx, c)
}

// transcriptionCall is one recognition stream: the audio going out, and the
// transcript coming back.
type transcriptionCall struct {
	opts     streamOptions
	source   audioSource
	onUpdate func(transcriptUpdate)

	stream rimepb.SpeechToText_TranscribeStreamingClient
	cancel context.CancelCauseFunc
	// accepted is closed once Rime accepts the configuration, which is when
	// the audio starts going out.
	accepted  chan struct{}
	inputDone atomic.Bool

	mu        sync.Mutex
	requestID string
	timers    []*time.Timer
}

// run makes the call and waits for both of its halves. The first failure is
// the one reported, whichever half met it.
func (t *transcriptionCall) run(ctx context.Context, c *sttClient) (transcriptFinal, error) {
	callCtx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	t.cancel = cancel
	defer t.stopTimers()

	md := metadata.Pairs("authorization", "Bearer "+c.apiKey)
	stream, err := rimepb.NewSpeechToTextClient(c.conn).TranscribeStreaming(
		metadata.NewOutgoingContext(callCtx, md),
		grpc.MaxCallRecvMsgSize(sttReceiveBytes),
		grpc.MaxCallSendMsgSize(sttSendBytes),
	)
	if err != nil {
		return transcriptFinal{}, t.failure(callCtx, t.statusError(err, nil))
	}
	t.stream = stream
	admission := t.deadline(sttAcceptanceTimeout, "Transcription acceptance timed out")
	if err := stream.Send(t.configRequest()); err != nil && !errors.Is(err, io.EOF) {
		return transcriptFinal{}, t.failure(callCtx, t.statusError(err, nil))
	}

	var wg sync.WaitGroup
	wg.Go(func() {
		if err := t.produce(callCtx); err != nil {
			cancel(err)
		}
	})
	final, readErr := t.read(callCtx, admission)
	if readErr != nil {
		cancel(readErr)
	}
	wg.Wait()
	if readErr != nil {
		return transcriptFinal{}, t.failure(callCtx, readErr)
	}
	return final, nil
}

// failure is the first failure the call met, with the request identity
// attached when the failure carries none.
func (t *transcriptionCall) failure(callCtx context.Context, fallback error) error {
	err := context.Cause(callCtx)
	if err == nil {
		err = fallback
	}
	re := asRimeError(err)
	switch {
	case re != nil:
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		re = newRimeError(kindCancelled, "Transcription canceled")
	default:
		re = newRimeError(kindStream, "Transcription failed")
	}
	if re.requestID == "" {
		re.requestID = t.currentRequestID()
	}
	return re
}

// configRequest opens the stream with the recognition settings.
func (t *transcriptionCall) configRequest() *rimepb.StreamingTranscriptionRequest {
	mode := rimepb.TranscriptionMode_TRANSCRIPTION_MODE_WRITTEN
	if t.opts.mode == modeVerbatim {
		mode = rimepb.TranscriptionMode_TRANSCRIPTION_MODE_VERBATIM
	}
	return &rimepb.StreamingTranscriptionRequest{
		Payload: &rimepb.StreamingTranscriptionRequest_Config{Config: &rimepb.StreamingConfig{
			Language:       &t.opts.language,
			Mode:           mode,
			ContextTerms:   t.opts.contextTerms,
			OutputContract: outputContractProto(t.opts.outputContract),
		}},
	}
}

// outputContractProto is the wire value of an output contract.
func outputContractProto(contract string) rimepb.StreamingOutputContract {
	if contract == contractAppendFragments {
		return rimepb.StreamingOutputContract_STREAMING_OUTPUT_CONTRACT_APPEND_FRAGMENTS
	}
	return rimepb.StreamingOutputContract_STREAMING_OUTPUT_CONTRACT_REVISED_HYPOTHESES
}

// deadline fails the call with a timeout once d has passed, unless the timer
// is stopped first.
func (t *transcriptionCall) deadline(d time.Duration, message string) *time.Timer {
	timer := time.AfterFunc(d, func() { t.cancel(newRimeError(kindTimeout, message)) })
	t.mu.Lock()
	t.timers = append(t.timers, timer)
	t.mu.Unlock()
	return timer
}

func (t *transcriptionCall) stopTimers() {
	t.mu.Lock()
	defer t.mu.Unlock()
	for _, timer := range t.timers {
		timer.Stop()
	}
}

func (t *transcriptionCall) currentRequestID() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.requestID
}

func (t *transcriptionCall) setRequestID(id string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.requestID = id
}

// statusError is the failure a gRPC error means, with the request identity
// the call's metadata carries.
func (t *transcriptionCall) statusError(err error, trailers metadata.MD) *rimeError {
	var headers metadata.MD
	if t.stream != nil {
		headers, _ = t.stream.Header()
		if trailers == nil {
			trailers = t.stream.Trailer()
		}
	}
	id := requestID(headers, trailers, t.currentRequestID())
	t.setRequestID(id)
	st := status.Convert(err)
	return rpcError(st.Code(), id, st.Message(), serverKind(headers, trailers))
}

// produce sends the audio once Rime has accepted the stream, and ends the
// input when the source ends, which asks Rime for the final transcript.
func (t *transcriptionCall) produce(ctx context.Context) error {
	select {
	case <-t.accepted:
	case <-ctx.Done():
		return nil
	}
	input, err := newInputAudio(t.opts.sampleRate)
	if err != nil {
		return err
	}
	for {
		data, ok, err := t.source.next(ctx)
		if err != nil {
			return sourceFailure(ctx)
		}
		if !ok {
			break
		}
		for _, part := range input.feed(data) {
			if !t.sendAudio(part) {
				// The call has ended, and how is what the read reports.
				return nil
			}
		}
	}
	if err := input.finish(); err != nil {
		return err
	}
	t.inputDone.Store(true)
	t.deadline(sttCompletionTimeout, "Transcription completion timed out")
	if err := t.stream.CloseSend(); err != nil {
		return t.statusError(err, nil)
	}
	return nil
}

// sendAudio sends one message of audio, and reports whether the call took it.
func (t *transcriptionCall) sendAudio(part []byte) bool {
	return t.stream.Send(&rimepb.StreamingTranscriptionRequest{
		Payload: &rimepb.StreamingTranscriptionRequest_Audio{Audio: part},
	}) == nil
}

// sourceFailure is what a failed read of the audio source means: the failure
// that ended the call, when the read failed because the call is over.
func sourceFailure(ctx context.Context) error {
	if cause := context.Cause(ctx); cause != nil {
		return cause
	}
	return newRimeError(kindInput, "The audio source failed")
}

// read takes the responses until the call ends, handing each interim update on,
// and returns the final transcript.
func (t *transcriptionCall) read(ctx context.Context, admission *time.Timer) (transcriptFinal, error) {
	if headers, err := t.stream.Header(); err == nil {
		t.setRequestID(requestID(headers, nil, ""))
	}
	state := newTranscriptState(sttTranscriptBytes, t.opts.outputContract)
	accepted := false
	for {
		message, err := t.stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			if cause := context.Cause(ctx); cause != nil {
				return transcriptFinal{}, cause
			}
			return transcriptFinal{}, t.statusError(err, t.stream.Trailer())
		}
		update, hasUpdate, err := state.accept(message, t.inputDone.Load())
		if err != nil {
			return transcriptFinal{}, err
		}
		if state.language != "" && !accepted {
			accepted = true
			close(t.accepted)
			admission.Stop()
		}
		if hasUpdate {
			t.onUpdate(update)
		}
	}
	t.setRequestID(requestID(nil, t.stream.Trailer(), t.currentRequestID()))
	return state.finish()
}

// transcriptState checks the responses of one stream against the output
// contract it selected, and builds the transcript from them.
type transcriptState struct {
	textLimit int
	contract  string
	language  string
	revision  uint64
	text      string
	final     *transcriptFinal
}

func newTranscriptState(textLimit int, contract string) *transcriptState {
	return &transcriptState{textLimit: textLimit, contract: contract}
}

// errTextLimit is the failure for a transcript longer than the client takes.
func errTextLimit() error {
	return newRimeError(kindResourceLimit, "The transcript exceeds the client text limit")
}

// accept takes one response, and returns the interim update it carries, and
// whether it carries one.
func (s *transcriptState) accept(
	message *rimepb.StreamingTranscriptionResponse, inputDone bool,
) (transcriptUpdate, bool, error) {
	if message.GetPayload() == nil {
		return transcriptUpdate{}, false, nil
	}
	if s.final != nil {
		return transcriptUpdate{}, false, newRimeError(kindStream, "Received a message after transcription completion")
	}
	if accepted := message.GetAccepted(); accepted != nil {
		return transcriptUpdate{}, false, s.acceptAccepted(accepted)
	}
	if s.language == "" {
		return transcriptUpdate{}, false, newRimeError(kindStream, "Received a transcript before acceptance")
	}
	appending := s.contract == contractAppendFragments
	if delta := message.GetDelta(); delta != nil && appending {
		if len(s.text)+len(delta.GetText()) > s.textLimit {
			return transcriptUpdate{}, false, errTextLimit()
		}
		s.text += delta.GetText()
		return transcriptUpdate{kind: updateDelta, text: delta.GetText()}, true, nil
	}
	hypothesis := message.GetHypothesis()
	done := message.GetDone()
	if done == nil && (hypothesis == nil || appending) {
		return transcriptUpdate{}, false, newRimeError(kindStream, "Received an unexpected transcription response")
	}
	if hypothesis != nil {
		if err := s.acceptHypothesis(hypothesis); err != nil {
			return transcriptUpdate{}, false, err
		}
		return transcriptUpdate{kind: updatePartial, text: hypothesis.GetText()}, true, nil
	}
	return transcriptUpdate{}, false, s.acceptDone(done, inputDone)
}

// acceptAccepted takes Rime's confirmation of the stream's configuration.
func (s *transcriptState) acceptAccepted(accepted *rimepb.StreamingAccepted) error {
	if s.language != "" {
		return newRimeError(kindStream, "Received duplicate transcription acceptance")
	}
	if accepted.GetOutputContract() != outputContractProto(s.contract) {
		return newRimeError(kindStream, "The service accepted a different transcript contract")
	}
	if accepted.GetLanguage().GetTag() == "" || accepted.GetLanguage().GetSource() == 0 {
		return newRimeError(kindStream, "The service did not confirm a resolved language")
	}
	s.language = accepted.GetLanguage().GetTag()
	return nil
}

// acceptHypothesis takes a replacement hypothesis.
func (s *transcriptState) acceptHypothesis(hypothesis *rimepb.TranscriptionHypothesis) error {
	if len(hypothesis.GetText()) > s.textLimit {
		return errTextLimit()
	}
	revision := hypothesis.GetRevision()
	if revision <= s.revision || (s.revision == 0 && revision != 1) {
		return newRimeError(kindStream, "Transcript revisions are not increasing from one")
	}
	s.revision, s.text = revision, hypothesis.GetText()
	return nil
}

// acceptDone takes the final transcript, which has to match what came before
// it.
func (s *transcriptState) acceptDone(done *rimepb.TranscriptionDone, inputDone bool) error {
	if len(done.GetText()) > s.textLimit {
		return errTextLimit()
	}
	if !inputDone {
		return newRimeError(kindStream, "The service completed before input finished")
	}
	if done.GetRevision() != s.revision || done.GetText() != s.text {
		return newRimeError(kindStream, "Final transcript does not match prior updates")
	}
	if done.GetLanguage().GetTag() != s.language || done.GetLanguage().GetSource() == 0 {
		return newRimeError(kindStream, "Final language does not match the accepted language")
	}
	s.final = &transcriptFinal{text: done.GetText(), language: done.GetLanguage().GetTag()}
	return nil
}

// finish is the final transcript, once the call has ended.
func (s *transcriptState) finish() (transcriptFinal, error) {
	if s.final == nil {
		return transcriptFinal{}, newRimeError(kindStream, "The service ended without a final transcript")
	}
	return *s.final, nil
}

// inputAudio prepares one utterance's audio as Rime takes it: whole 16-bit
// frames, at 16 kHz, in messages no larger than Rime accepts. A frame split
// across two chunks is carried over to the next; nothing is padded or cut.
type inputAudio struct {
	converter *rimepcm.RateConverter
	pending   []byte
}

// newInputAudio prepares mono audio at sampleRate.
func newInputAudio(sampleRate int) (*inputAudio, error) {
	in := &inputAudio{}
	switch sampleRate {
	case 8000, 24000, 48000:
		in.converter = rimepcm.NewRateConverter(sampleRate, sttWireRate)
	case sttWireRate:
	default:
		return nil, newRimeError(kindAudioFormat, "Use PCM16 at 8, 16, 24 or 48 kHz")
	}
	return in, nil
}

// feed converts one chunk, and returns the messages that carry it.
func (in *inputAudio) feed(data []byte) [][]byte {
	var out [][]byte
	for offset := 0; offset < len(data); offset += sttSourceBytes {
		part := slices.Concat(in.pending, data[offset:min(offset+sttSourceBytes, len(data))])
		complete := len(part) - len(part)%2
		in.pending = append([]byte(nil), part[complete:]...)
		if complete == 0 {
			continue
		}
		converted := part[:complete]
		if in.converter != nil {
			converted = in.converter.Process(converted)
		}
		for start := 0; start < len(converted); start += sttMessageBytes {
			out = append(out, converted[start:min(start+sttMessageBytes, len(converted))])
		}
	}
	return out
}

// finish ends the input. Audio that ends partway through a frame is invalid.
func (in *inputAudio) finish() error {
	if len(in.pending) > 0 {
		return newRimeError(kindAudioFormat, "The audio source ended with an incomplete PCM16 frame")
	}
	return nil
}
