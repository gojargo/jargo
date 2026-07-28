package nvidia

import (
	"context"
	"crypto/tls"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/provider/nvidia/internal/rivapb"
	"github.com/gojargo/jargo/service/stt"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
)

// segmentedTTFSP99 is the time-to-final-segment P99 latency reported downstream.
// A batch model transcribes only once the utterance is complete, so it trails
// the streaming service.
const segmentedTTFSP99 = 1500 * time.Millisecond

// SegmentedSTTConfig configures the NVIDIA Riva batch speech-to-text service.
// It transcribes one complete utterance per call against Riva's offline models,
// so it needs a turn detector upstream to delimit each segment.
type SegmentedSTTConfig struct {
	// Server is the Riva gRPC endpoint; empty uses NVIDIA's hosted endpoint. For
	// a local NIM this is typically "localhost:50051".
	Server string
	// APIKey is the NVIDIA API key, sent as an "authorization: Bearer" header.
	// Required by the hosted endpoint; omit for an unauthenticated local NIM.
	APIKey string
	// FunctionID is the NVIDIA Cloud Function id, sent as a "function-id" header.
	// Required by the hosted endpoint to select the model; omit for a local NIM.
	FunctionID string
	// Model selects a served model by name; empty lets the server choose.
	Model string
	// Language of the audio, mapped to a Riva BCP-47 code; the zero value uses US
	// English.
	Language language.Language
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
	// UseSSL wraps the gRPC connection in TLS; nil defaults to true. Set to a
	// pointer to false for a plaintext local NIM.
	UseSSL *bool
	// AutomaticPunctuation adds punctuation to results; nil defaults to true.
	AutomaticPunctuation *bool
	// VerbatimTranscripts disables inverse text normalization; nil defaults to
	// true.
	VerbatimTranscripts *bool
	// ProfanityFilter masks profanities in the transcript.
	ProfanityFilter bool
	// MaxAlternatives caps the hypotheses returned per result; 0 uses 1.
	MaxAlternatives int
	// AudioChannelCount is the input channel count; 0 uses 1 (mono).
	AudioChannelCount int
	// CustomConfiguration passes request-level options to the model pipeline,
	// keyed by option name.
	CustomConfiguration map[string]string
}

// Validate reports whether the configuration is usable.
func (c SegmentedSTTConfig) Validate() error { return validate.Struct(c) }

// withSegmentedDefaults fills unset fields with their defaults.
func (c SegmentedSTTConfig) withSegmentedDefaults() SegmentedSTTConfig {
	if c.Server == "" {
		c.Server = defaultSTTServer
	}
	if c.Language == "" {
		c.Language = language.EnglishUS
	}
	if c.MaxAlternatives == 0 {
		c.MaxAlternatives = 1
	}
	if c.AudioChannelCount == 0 {
		c.AudioChannelCount = 1
	}
	return c
}

// NewSegmentedSTT builds an NVIDIA Riva batch speech-to-text service. It
// transcribes each utterance in one call once a turn detector upstream has
// delimited it, unlike NewSTT which streams continuously. Use it with Riva's
// offline models, which are more accurate but produce no interim transcripts.
func NewSegmentedSTT(cfg SegmentedSTTConfig) *stt.SegmentService {
	cfg = cfg.withSegmentedDefaults()
	return stt.NewSegment("NvidiaSegmentedSTT", &segmentedTranscriber{cfg: cfg}, cfg.SampleRate)
}

type segmentedTranscriber struct {
	cfg SegmentedSTTConfig

	// mu guards the lazily dialed connection, reused across utterances so a
	// segment does not pay for a fresh TLS handshake.
	mu   sync.Mutex
	conn *grpc.ClientConn
}

// Metadata reports the Riva batch latency to downstream processors.
func (t *segmentedTranscriber) Metadata() stt.Metadata {
	return stt.Metadata{
		RecommendedUserTurns: frames.UserTurnUnspecified,
		TTFSP99:              segmentedTTFSP99,
		Model:                t.cfg.Model,
	}
}

// client returns the shared gRPC connection, dialing it on first use.
func (t *segmentedTranscriber) client() (*grpc.ClientConn, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.conn != nil {
		return t.conn, nil
	}
	creds := credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12})
	if !boolOrTrue(t.cfg.UseSSL) {
		creds = insecure.NewCredentials()
	}
	conn, err := grpc.NewClient(t.cfg.Server, grpc.WithTransportCredentials(creds))
	if err != nil {
		return nil, err
	}
	t.conn = conn
	return conn, nil
}

// Close releases the shared connection.
func (t *segmentedTranscriber) Close() error {
	t.mu.Lock()
	conn := t.conn
	t.conn = nil
	t.mu.Unlock()
	if conn == nil {
		return nil
	}
	return conn.Close()
}

// recognitionConfig builds the configuration for one batch request.
func (t *segmentedTranscriber) recognitionConfig(sampleRate int) *rivapb.RecognitionConfig {
	return &rivapb.RecognitionConfig{
		Encoding:                   rivapb.AudioEncoding_LINEAR_PCM,
		SampleRateHertz:            int32(sampleRate), //nolint:gosec // a sample rate is small and positive
		LanguageCode:               nvidiaLanguage(t.cfg.Language),
		MaxAlternatives:            int32(t.cfg.MaxAlternatives),   //nolint:gosec // a small bounded count
		AudioChannelCount:          int32(t.cfg.AudioChannelCount), //nolint:gosec // a small bounded count
		ProfanityFilter:            t.cfg.ProfanityFilter,
		EnableAutomaticPunctuation: boolOrTrue(t.cfg.AutomaticPunctuation),
		Model:                      t.cfg.Model,
		VerbatimTranscripts:        boolOrTrue(t.cfg.VerbatimTranscripts),
		CustomConfiguration:        t.cfg.CustomConfiguration,
	}
}

// Transcribe sends one complete utterance and returns its transcript.
func (t *segmentedTranscriber) Transcribe(ctx context.Context, audio []byte, sampleRate int) (string, error) {
	conn, err := t.client()
	if err != nil {
		return "", err
	}

	md := metadata.MD{}
	if t.cfg.FunctionID != "" {
		md.Set("function-id", t.cfg.FunctionID)
	}
	if t.cfg.APIKey != "" {
		md.Set("authorization", "Bearer "+t.cfg.APIKey)
	}

	resp, err := rivapb.NewRivaSpeechRecognitionClient(conn).Recognize(
		metadata.NewOutgoingContext(ctx, md),
		&rivapb.RecognizeRequest{Config: t.recognitionConfig(sampleRate), Audio: audio},
	)
	if err != nil {
		return "", err
	}

	// A batch response splits long audio across results, so they are joined back
	// into the utterance's transcript.
	var text string
	for _, r := range resp.GetResults() {
		alts := r.GetAlternatives()
		if len(alts) == 0 {
			continue
		}
		if transcript := alts[0].GetTranscript(); transcript != "" {
			if text != "" {
				text += " "
			}
			text += transcript
		}
	}
	return text, nil
}
