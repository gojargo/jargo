package nvidia

import (
	"context"
	"crypto/tls"
	"errors"
	"io"
	"sort"
	"strings"
	"sync"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/provider/nvidia/internal/rivapb"
	"github.com/gojargo/jargo/service/tts"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
)

const (
	// defaultTTSServer is NVIDIA's hosted Cloud Function TTS endpoint. Point at a
	// locally deployed Riva/NIM with its address and UseSSL set to false.
	defaultTTSServer = "grpc.nvcf.nvidia.com:443"
	// defaultTTSFunctionID selects NVIDIA's hosted multilingual Magpie model.
	defaultTTSFunctionID = "877104f7-e885-42b9-8de8-f6e4c6303969"
	// defaultTTSModel names the model the hosted function serves. It labels the
	// metrics and traces; the function id is what actually selects it.
	defaultTTSModel = "magpie-tts-multilingual"
	// defaultTTSVoice is a multilingual Magpie voice.
	defaultTTSVoice = "Magpie-Multilingual.EN-US.Aria"
	// defaultTTSSampleRate is the PCM rate requested from the server.
	defaultTTSSampleRate = 24000
	// defaultZeroShotQuality is how many decoder passes a zero-shot prompt gets
	// when unset.
	defaultZeroShotQuality = 20
	// maxTTSChunkLen bounds one synthesis request's text. Models cap the request
	// length, so a long sentence is split across several requests on the same
	// stream.
	maxTTSChunkLen = 200
)

// ZeroShot supplies the audio prompt a zero-shot model clones its voice from.
// Access to NVIDIA's hosted zero-shot models needs approval.
type ZeroShot struct {
	// AudioPrompt is the prompt audio. NVIDIA recommends 16-bit mono, 22.05 kHz
	// or higher, and 3 to 10 seconds long. Required.
	AudioPrompt []byte `validate:"required,min=1"`
	// Encoding of the prompt, either "pcm" (the default) or "oggopus".
	Encoding string `validate:"omitempty,oneof=pcm oggopus"`
	// SampleRate of the prompt; 0 lets the server assume 22050.
	SampleRate int
	// Quality is how many times the prompt passes through the decoder, 1 to 40;
	// 0 uses 20.
	Quality int `validate:"omitempty,min=1,max=40"`
	// Transcript of the prompt audio; empty sends none.
	Transcript string
}

// TTSConfig configures the NVIDIA Riva streaming text-to-speech service. The
// same service talks to NVIDIA's hosted TTS endpoint and to a locally deployed
// Riva/NIM model; the deployment is selected entirely through Server, UseSSL,
// and the auth fields.
type TTSConfig struct {
	// Server is the Riva gRPC endpoint; empty uses NVIDIA's hosted endpoint. For
	// a local NIM this is typically "localhost:50051".
	Server string
	// APIKey is the NVIDIA API key, sent as an "authorization: Bearer" header.
	// Required by the hosted endpoint; omit for an unauthenticated local NIM.
	APIKey string
	// FunctionID is the NVIDIA Cloud Function id, sent as a "function-id" header.
	// It selects the model on the hosted endpoint; empty uses the multilingual
	// Magpie function. Set it to "-" to send no function id, as a local NIM
	// expects.
	FunctionID string
	// Model names the served model. It labels the metrics and traces and is what
	// a cost-tracking backend prices against; empty uses the hosted default. It
	// does not select the model, FunctionID does.
	Model string
	// Voice is the voice name; empty uses a multilingual Magpie voice.
	Voice string
	// Language for synthesis, mapped to a Riva BCP-47 code; the zero value uses
	// US English.
	Language language.Language
	// SampleRate is the PCM rate requested from the server and emitted
	// downstream; 0 uses 24 kHz. Rates below 8 kHz do not produce usable audio.
	SampleRate int
	// UseSSL wraps the gRPC connection in TLS; nil defaults to true. Set to a
	// pointer to false for a plaintext local NIM.
	UseSSL *bool
	// CustomDictionary maps a written form to its IPA pronunciation, for example
	// {"NVIDIA": "ɛn.vɪ.diː.ʌ"}; empty sends none.
	CustomDictionary map[string]string
	// CustomConfiguration passes model-specific options the schema does not name
	// (for example "exaggeration_factor"), keyed by option name.
	CustomConfiguration map[string]string
	// ZeroShot supplies the voice-cloning prompt a zero-shot model needs; nil
	// omits it, which is what every other model expects.
	ZeroShot *ZeroShot `validate:"omitempty"`
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }

// NewTTS builds an NVIDIA Riva streaming text-to-speech service. Each sentence
// opens a SynthesizeOnline stream, sends the text (split when the model's
// request-length limit requires it), and streams the generated audio downstream.
func NewTTS(cfg TTSConfig) *tts.Base {
	cfg = cfg.withTTSDefaults()
	return tts.New("NvidiaTTS", &ttsSynthesizer{cfg: cfg})
}

// withTTSDefaults fills unset fields with their defaults.
func (c TTSConfig) withTTSDefaults() TTSConfig {
	if c.Server == "" {
		c.Server = defaultTTSServer
	}
	if c.FunctionID == "" {
		c.FunctionID = defaultTTSFunctionID
	}
	if c.Model == "" {
		c.Model = defaultTTSModel
	}
	if c.Voice == "" {
		c.Voice = defaultTTSVoice
	}
	if c.Language == "" {
		c.Language = language.EnglishUS
	}
	if c.SampleRate == 0 {
		c.SampleRate = defaultTTSSampleRate
	}
	return c
}

// customDictionary renders the pronunciation dictionary as Riva's
// comma-separated list of grapheme/phoneme pairs, each pair separated by two
// spaces. Entries are sorted so the request is stable across runs.
func (c TTSConfig) customDictionary() string {
	if len(c.CustomDictionary) == 0 {
		return ""
	}
	graphemes := make([]string, 0, len(c.CustomDictionary))
	for g := range c.CustomDictionary {
		graphemes = append(graphemes, g)
	}
	sort.Strings(graphemes)
	entries := make([]string, 0, len(graphemes))
	for _, g := range graphemes {
		entries = append(entries, g+"  "+c.CustomDictionary[g])
	}
	return strings.Join(entries, ",")
}

// zeroShotData renders the voice-cloning prompt, or nil when there is none.
// Validate constrains the encoding, so an unrecognized one falls back to PCM
// rather than failing a synthesis that is already under way.
func (c TTSConfig) zeroShotData() *rivapb.ZeroShotData {
	z := c.ZeroShot
	if z == nil {
		return nil
	}
	encoding := rivapb.AudioEncoding_LINEAR_PCM
	if z.Encoding == "oggopus" {
		encoding = rivapb.AudioEncoding_OGGOPUS
	}
	quality := z.Quality
	if quality == 0 {
		quality = defaultZeroShotQuality
	}
	return &rivapb.ZeroShotData{
		AudioPrompt:  z.AudioPrompt,
		SampleRateHz: int32(z.SampleRate),
		Encoding:     encoding,
		Quality:      int32(quality),
		Transcript:   z.Transcript,
	}
}

type ttsSynthesizer struct {
	cfg TTSConfig

	// mu guards the lazily dialed connection, which is reused across syntheses
	// so a sentence does not pay for a fresh TLS handshake.
	mu   sync.Mutex
	conn *grpc.ClientConn
}

// SampleRate reports the requested PCM output rate.
func (s *ttsSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// Metadata reports the Riva model and voice synthesis is billed against.
func (s *ttsSynthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.cfg.Model, VoiceID: s.cfg.Voice}
}

// client returns the shared gRPC connection, dialing it on first use.
// grpc.NewClient does no I/O, so the connection is established by the first RPC.
func (s *ttsSynthesizer) client() (*grpc.ClientConn, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.conn != nil {
		return s.conn, nil
	}
	creds := credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12})
	if !boolOrTrue(s.cfg.UseSSL) {
		creds = insecure.NewCredentials()
	}
	conn, err := grpc.NewClient(s.cfg.Server, grpc.WithTransportCredentials(creds))
	if err != nil {
		return nil, err
	}
	s.conn = conn
	return conn, nil
}

// Close releases the shared connection, implementing tts.Closer.
func (s *ttsSynthesizer) Close() error {
	s.mu.Lock()
	conn := s.conn
	s.conn = nil
	s.mu.Unlock()
	if conn == nil {
		return nil
	}
	return conn.Close()
}

// outgoingContext attaches the credentials the hosted endpoint expects. A local
// NIM needs neither, so both headers are omitted when unset.
func (s *ttsSynthesizer) outgoingContext(ctx context.Context) context.Context {
	md := metadata.MD{}
	if s.cfg.FunctionID != "" && s.cfg.FunctionID != "-" {
		md.Set("function-id", s.cfg.FunctionID)
	}
	if s.cfg.APIKey != "" {
		md.Set("authorization", "Bearer "+s.cfg.APIKey)
	}
	return metadata.NewOutgoingContext(ctx, md)
}

// request builds the synthesis request for one span of text.
func (s *ttsSynthesizer) request(text string, zeroShot *rivapb.ZeroShotData) *rivapb.SynthesizeSpeechRequest {
	return &rivapb.SynthesizeSpeechRequest{
		Text:         text,
		LanguageCode: nvidiaLanguage(s.cfg.Language),
		// The pipeline carries linear PCM, so the output encoding is fixed.
		Encoding:            rivapb.AudioEncoding_LINEAR_PCM,
		SampleRateHz:        int32(s.cfg.SampleRate),
		VoiceName:           s.cfg.Voice,
		ZeroShotData:        zeroShot,
		CustomDictionary:    s.cfg.customDictionary(),
		CustomConfiguration: s.cfg.CustomConfiguration,
	}
}

// Synthesize opens a synthesis stream, sends the sentence, and streams the
// generated audio downstream until the server closes the stream.
func (s *ttsSynthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	chunks := chunkText(text, maxTTSChunkLen)
	if len(chunks) == 0 {
		return nil
	}
	conn, err := s.client()
	if err != nil {
		return err
	}
	zeroShot := s.cfg.zeroShotData()

	stream, err := rivapb.NewRivaSpeechSynthesisClient(conn).SynthesizeOnline(s.outgoingContext(ctx))
	if err != nil {
		return err
	}
	for _, chunk := range chunks {
		if err := stream.Send(s.request(chunk, zeroShot)); err != nil {
			return err
		}
	}
	// Half-close so the server knows the sentence is complete and finishes.
	if err := stream.CloseSend(); err != nil {
		return err
	}

	for {
		resp, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		if audio := resp.GetAudio(); len(audio) > 0 {
			if err := emit(audio); err != nil {
				return err
			}
		}
	}
}

// chunkText splits text into spans of at most width bytes, breaking on
// whitespace where it can and mid-word only when a single word is longer than
// width. It mirrors the request-length limit the models impose.
func chunkText(text string, width int) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	if len(text) <= width {
		return []string{text}
	}

	var chunks []string
	var current strings.Builder
	flush := func() {
		if current.Len() > 0 {
			chunks = append(chunks, current.String())
			current.Reset()
		}
	}
	for word := range strings.FieldsSeq(text) {
		// A word longer than a whole chunk has to be split mid-word.
		for len(word) > width {
			flush()
			chunks = append(chunks, word[:width])
			word = word[width:]
		}
		if current.Len() > 0 && current.Len()+1+len(word) > width {
			flush()
		}
		if current.Len() > 0 {
			current.WriteByte(' ')
		}
		current.WriteString(word)
	}
	flush()
	return chunks
}
