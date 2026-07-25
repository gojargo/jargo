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
	"github.com/gojargo/jargo/service/tts"
)

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
