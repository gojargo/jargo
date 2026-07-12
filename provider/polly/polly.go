// Package polly is a text-to-speech service backed by Amazon Polly's
// SynthesizeSpeech API. It requests signed 16-bit mono PCM at 8 or 16 kHz
// (Polly caps PCM at 16 kHz) and streams the returned audio downstream.
// Credentials and region come from the standard AWS chain (environment, shared
// config, or an IAM role) unless set explicitly.
package polly

import (
	"context"
	"errors"
	"io"
	"strconv"
	"strings"
	"sync"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/polly"
	"github.com/aws/aws-sdk-go-v2/service/polly/types"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/tts"
)

const (
	defaultVoice      = "Joanna"
	defaultLanguage   = "en-US"
	defaultSampleRate = 16000
	// readChunk is the size of each audio read from the synthesis stream.
	readChunk = 8192
)

// Config configures the Amazon Polly TTS service.
type Config struct {
	// Region is the AWS region (e.g. us-east-1); empty uses the default chain
	// (AWS_REGION, shared config).
	Region string
	// AccessKeyID and SecretAccessKey set static credentials; leave both empty to
	// use the default AWS credential chain (environment, shared config, IAM role).
	AccessKeyID     string
	SecretAccessKey string
	// SessionToken is the optional session token for temporary credentials.
	SessionToken string
	// Voice is the Polly voice id (e.g. Joanna, Matthew); empty uses a default.
	Voice string
	// Engine selects the synthesis engine; empty leaves Polly's default in place.
	Engine string `validate:"omitempty,oneof=standard neural long-form generative"`
	// Language is the BCP-47 language tag applied to the synthesized text
	// (e.g. en-US, fr-FR); empty uses en-US.
	Language string
	// SampleRate is the PCM rate requested from Polly and emitted downstream;
	// Polly supports 8000 or 16000 for PCM. 0 uses 16 kHz.
	SampleRate int `validate:"omitempty,oneof=8000 16000"`
	// Pitch adjusts voice pitch (standard engine only); empty omits it.
	Pitch string
	// Rate adjusts speech rate; empty omits it.
	Rate string
	// Volume adjusts voice volume; empty omits it.
	Volume string
	// LexiconNames lists pronunciation lexicons to apply during synthesis.
	LexiconNames []string
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// loadOptions builds the AWS config load options from the static fields. An
// empty Region or empty credentials fall back to the default AWS chain.
func (c Config) loadOptions() []func(*awsconfig.LoadOptions) error {
	var opts []func(*awsconfig.LoadOptions) error
	if c.Region != "" {
		opts = append(opts, awsconfig.WithRegion(c.Region))
	}
	if c.AccessKeyID != "" && c.SecretAccessKey != "" {
		opts = append(opts, awsconfig.WithCredentialsProvider(
			credentials.NewStaticCredentialsProvider(c.AccessKeyID, c.SecretAccessKey, c.SessionToken),
		))
	}
	return opts
}

// NewTTS builds an Amazon Polly TTS service.
func NewTTS(cfg Config) *tts.Base {
	if cfg.Voice == "" {
		cfg.Voice = defaultVoice
	}
	if cfg.Language == "" {
		cfg.Language = defaultLanguage
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	return tts.New("PollyTTS", &synthesizer{cfg: cfg})
}

type synthesizer struct {
	cfg Config

	mu     sync.Mutex
	client *polly.Client
}

// SampleRate reports the requested PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// Synthesize requests speech for text and streams the raw PCM downstream.
func (s *synthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	client, err := s.pollyClient(ctx)
	if err != nil {
		return err
	}
	out, err := client.SynthesizeSpeech(ctx, s.cfg.request(text))
	if err != nil {
		return err
	}
	defer func() { _ = out.AudioStream.Close() }()

	buf := make([]byte, readChunk)
	for {
		n, err := out.AudioStream.Read(buf)
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

// pollyClient lazily loads AWS config (which may read the environment, shared
// config files or instance metadata) and caches a client.
func (s *synthesizer) pollyClient(ctx context.Context) (*polly.Client, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.client != nil {
		return s.client, nil
	}
	awsCfg, err := awsconfig.LoadDefaultConfig(ctx, s.cfg.loadOptions()...)
	if err != nil {
		return nil, err
	}
	s.client = polly.NewFromConfig(awsCfg)
	return s.client, nil
}

// request builds the SynthesizeSpeech input for text.
func (c Config) request(text string) *polly.SynthesizeSpeechInput {
	in := &polly.SynthesizeSpeechInput{
		Text:         aws.String(c.ssml(text)),
		TextType:     types.TextTypeSsml,
		OutputFormat: types.OutputFormatPcm,
		VoiceId:      types.VoiceId(c.Voice),
		SampleRate:   aws.String(strconv.Itoa(c.SampleRate)),
	}
	if c.Engine != "" {
		in.Engine = types.Engine(c.Engine)
	}
	if len(c.LexiconNames) > 0 {
		in.LexiconNames = c.LexiconNames
	}
	return in
}

// ssml wraps text in an SSML document carrying the language and, when set, the
// prosody controls. Pitch applies only to the standard engine.
func (c Config) ssml(text string) string {
	var b strings.Builder
	b.WriteString("<speak><lang xml:lang='")
	b.WriteString(c.Language)
	b.WriteString("'>")

	var prosody []string
	if c.Engine == "standard" && c.Pitch != "" {
		prosody = append(prosody, "pitch='"+c.Pitch+"'")
	}
	if c.Rate != "" {
		prosody = append(prosody, "rate='"+c.Rate+"'")
	}
	if c.Volume != "" {
		prosody = append(prosody, "volume='"+c.Volume+"'")
	}
	if len(prosody) > 0 {
		b.WriteString("<prosody ")
		b.WriteString(strings.Join(prosody, " "))
		b.WriteString(">")
	}
	b.WriteString(text)
	if len(prosody) > 0 {
		b.WriteString("</prosody>")
	}
	b.WriteString("</lang></speak>")
	return b.String()
}
