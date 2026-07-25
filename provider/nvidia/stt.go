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

const (
	// defaultSTTServer is NVIDIA's hosted Cloud Function ASR endpoint. Point at a
	// locally deployed Riva/NIM (for example a parakeet model) with its address
	// and UseSSL set to false.
	defaultSTTServer = "grpc.nvcf.nvidia.com:443"
	// sttTTFSP99 is the time-to-final-segment P99 latency reported downstream.
	sttTTFSP99 = time.Second
)

// Endpointing tunes Riva's server-side start/end-of-utterance detector. Every
// field is optional; a nil field keeps the server's built-in default. The
// history fields are window sizes in milliseconds and the threshold fields are
// fractions in [0, 1].
type Endpointing struct {
	StartHistory     *int32
	StartThreshold   *float32
	StopHistory      *int32
	StopThreshold    *float32
	StopHistoryEOU   *int32
	StopThresholdEOU *float32
}

// STTConfig configures the NVIDIA Riva streaming speech-to-text service. The
// same service talks to NVIDIA's hosted ASR endpoint and to a locally deployed
// Riva/NIM model (such as parakeet); the deployment is selected entirely through
// Server, UseSSL, and the auth fields.
type STTConfig struct {
	// Server is the Riva gRPC endpoint; empty uses NVIDIA's hosted endpoint. For
	// a local NIM this is typically "localhost:50051".
	Server string
	// APIKey is the NVIDIA API key, sent as an "authorization: Bearer" header.
	// Required by the hosted endpoint; omit for an unauthenticated local NIM.
	APIKey string
	// FunctionID is the NVIDIA Cloud Function id, sent as a "function-id" header.
	// Required by the hosted endpoint to select the model; omit for a local NIM.
	FunctionID string
	// Model selects a served model by name; empty lets the server choose. A local
	// Riva build usually serves one model, so this can stay empty.
	Model string
	// Language of the audio, mapped to a Riva BCP-47 code; the zero value uses US
	// English.
	Language language.Language
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int

	// UseSSL wraps the gRPC connection in TLS; nil defaults to true. Set to a
	// pointer to false for a plaintext local NIM.
	UseSSL *bool
	// InterimResults requests tentative partial transcripts; nil defaults to true.
	InterimResults *bool
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
	// CustomConfiguration passes request-level options to the model pipeline
	// (for example "enable_vad_endpointing:true"), keyed by option name.
	CustomConfiguration map[string]string
	// Endpointing tunes the server-side utterance detector; nil uses the server
	// defaults.
	Endpointing *Endpointing
}

// Validate reports whether the configuration is usable.
func (c STTConfig) Validate() error { return validate.Struct(c) }

// NewSTT builds an NVIDIA Riva streaming speech-to-text service. A finalized
// result (Riva's is_final, driven by server-side endpointing) marks the end of
// the user's turn.
func NewSTT(cfg STTConfig) *stt.StreamService {
	cfg = cfg.withDefaults()
	return stt.NewStream("NvidiaSTT", &sttConnector{cfg: cfg}, cfg.SampleRate)
}

// withDefaults fills unset fields with their defaults.
func (c STTConfig) withDefaults() STTConfig {
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

// nvidiaBaseLangs promotes the common base language codes to the region-qualified
// codes Riva expects. Already-qualified or unmapped codes pass through unchanged.
//
//nolint:gochecknoglobals // static lookup table
var nvidiaBaseLangs = map[language.Language]string{
	language.English:    "en-US",
	language.French:     "fr-FR",
	language.German:     "de-DE",
	language.Spanish:    "es-ES",
	language.Italian:    "it-IT",
	language.Portuguese: "pt-PT",
	language.Dutch:      "nl-NL",
	language.Russian:    "ru-RU",
}

// nvidiaLanguage maps a Language to a Riva ASR BCP-47 code.
func nvidiaLanguage(l language.Language) string {
	if code, ok := nvidiaBaseLangs[l]; ok {
		return code
	}
	return l.Code()
}

// boolOrTrue returns *p, or true when p is nil.
func boolOrTrue(p *bool) bool { return p == nil || *p }

type sttConnector struct {
	cfg STTConfig
}

// Metadata reports the Riva ASR time-to-final-segment latency to downstream
// processors.
func (c *sttConnector) Metadata() stt.Metadata {
	return stt.Metadata{
		RecommendedUserTurns: frames.UserTurnUnspecified,
		TTFSP99:              sttTTFSP99,
		Model:                c.cfg.Model,
	}
}

// Connect dials the Riva gRPC endpoint, opens a StreamingRecognize stream, and
// sends the opening configuration message.
func (c *sttConnector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	creds := credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12})
	if !boolOrTrue(c.cfg.UseSSL) {
		creds = insecure.NewCredentials()
	}
	conn, err := grpc.NewClient(c.cfg.Server, grpc.WithTransportCredentials(creds))
	if err != nil {
		return nil, err
	}

	md := metadata.MD{}
	if c.cfg.FunctionID != "" {
		md.Set("function-id", c.cfg.FunctionID)
	}
	if c.cfg.APIKey != "" {
		md.Set("authorization", "Bearer "+c.cfg.APIKey)
	}
	streamCtx := metadata.NewOutgoingContext(ctx, md)

	rs, err := rivapb.NewRivaSpeechRecognitionClient(conn).StreamingRecognize(streamCtx)
	if err != nil {
		_ = conn.Close()
		return nil, err
	}
	if err := rs.Send(c.cfg.streamingConfig(sampleRate)); err != nil {
		_ = conn.Close()
		return nil, err
	}
	return &sttStream{conn: conn, rs: rs}, nil
}

// streamingConfig builds the opening StreamingRecognize request for sampleRate.
func (c *STTConfig) streamingConfig(sampleRate int) *rivapb.StreamingRecognizeRequest {
	rc := &rivapb.RecognitionConfig{
		Encoding:                   rivapb.AudioEncoding_LINEAR_PCM,
		SampleRateHertz:            int32(sampleRate),
		LanguageCode:               nvidiaLanguage(c.Language),
		MaxAlternatives:            int32(c.MaxAlternatives),
		ProfanityFilter:            c.ProfanityFilter,
		AudioChannelCount:          int32(c.AudioChannelCount),
		EnableAutomaticPunctuation: boolOrTrue(c.AutomaticPunctuation),
		Model:                      c.Model,
		VerbatimTranscripts:        boolOrTrue(c.VerbatimTranscripts),
		CustomConfiguration:        c.CustomConfiguration,
	}
	if e := c.Endpointing; e != nil {
		rc.EndpointingConfig = &rivapb.EndpointingConfig{
			StartHistory:     e.StartHistory,
			StartThreshold:   e.StartThreshold,
			StopHistory:      e.StopHistory,
			StopThreshold:    e.StopThreshold,
			StopHistoryEou:   e.StopHistoryEOU,
			StopThresholdEou: e.StopThresholdEOU,
		}
	}
	return &rivapb.StreamingRecognizeRequest{
		StreamingRequest: &rivapb.StreamingRecognizeRequest_StreamingConfig{
			StreamingConfig: &rivapb.StreamingRecognitionConfig{
				Config:         rc,
				InterimResults: boolOrTrue(c.InterimResults),
			},
		},
	}
}

type sttStream struct {
	conn    *grpc.ClientConn
	rs      rivapb.RivaSpeechRecognition_StreamingRecognizeClient
	writeMu sync.Mutex
}

// Send writes a chunk of PCM audio as an audio_content message.
func (s *sttStream) Send(audio []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.rs.Send(&rivapb.StreamingRecognizeRequest{
		StreamingRequest: &rivapb.StreamingRecognizeRequest_AudioContent{AudioContent: audio},
	})
}

// Recv reads the next batch of results, skipping empty responses. A finalized
// result carries Riva's is_final as the end-of-turn signal.
func (s *sttStream) Recv() ([]stt.Result, error) {
	for {
		resp, err := s.rs.Recv()
		if err != nil {
			return nil, err
		}
		var out []stt.Result
		for _, r := range resp.GetResults() {
			alts := r.GetAlternatives()
			if len(alts) == 0 {
				continue
			}
			text := alts[0].GetTranscript()
			if text == "" {
				continue
			}
			lang := ""
			if codes := alts[0].GetLanguageCode(); len(codes) > 0 {
				lang = codes[0]
			}
			out = append(out, stt.Result{
				Text:      text,
				Final:     r.GetIsFinal(),
				EndOfTurn: r.GetIsFinal(),
				Language:  lang,
			})
		}
		if len(out) > 0 {
			return out, nil
		}
	}
}

// Close half-closes the send direction and tears down the gRPC connection.
func (s *sttStream) Close() error {
	s.writeMu.Lock()
	_ = s.rs.CloseSend()
	s.writeMu.Unlock()
	return s.conn.Close()
}
