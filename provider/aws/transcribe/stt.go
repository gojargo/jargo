package transcribe

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/transcribestreaming"
	"github.com/aws/aws-sdk-go-v2/service/transcribestreaming/types"
	"github.com/gojargo/jargo/errclass"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
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

// NewSTT builds an Amazon Transcribe streaming STT service.
func NewSTT(cfg Config) *stt.StreamService {
	return stt.NewStream("TranscribeSTT", newConnector(cfg), cfg.SampleRate)
}

// newConnector builds the connector a service opens its sessions with, its
// settings store taken from cfg.
func newConnector(cfg Config) *connector {
	return &connector{cfg: cfg, live: newSTTSettings(cfg)}
}

type connector struct {
	cfg Config
	// live is the settings store updates are merged into. The service holds it
	// steady while a session opens from it.
	live *STTSettings

	mu     sync.Mutex
	client *transcribestreaming.Client
}

// Metadata reports Transcribe's turn strategy and latency to downstream
// processors. It finalizes per segment, so it leaves turn detection to the
// pipeline rather than recommending external turns.
func (c *connector) Metadata() stt.Metadata {
	return stt.Metadata{
		TTFSP99: stt.TTFSP99Or(c.cfg.TTFSP99, stt.AWSTranscribeTTFSP99),
	}
}

// Settings is the configuration a caller may change while the pipeline runs.
func (c *connector) Settings() any { return c.live }

// PrepareSettings refuses an update whose language identification settings
// conflict once merged into the settings in force, whole, before any of it is
// applied.
func (c *connector) PrepareSettings(delta any) error {
	copied, err := settings.Copy(c.live)
	if err != nil {
		return err
	}
	candidate, _ := copied.(*STTSettings)
	if _, err := settings.Apply(candidate, delta); err != nil {
		return err
	}
	if err := validateLanguageIdentification(candidate); err != nil {
		return fmt.Errorf("%w: %w", errSettingsRejected, err)
	}
	return nil
}

// UpdateSettings asks for the session to be reopened whenever anything changed.
// Transcribe binds all of it to a connection, so a change reaches it only by
// opening another.
func (c *connector) UpdateSettings(context.Context, settings.Changed) (bool, error) {
	return true, nil
}

// ClassifyError says what a Transcribe failure means: a refused settings update
// is the application's mistake.
func (c *connector) ClassifyError(err error) errclass.Category {
	if errors.Is(err, errSettingsRejected) {
		return errclass.Application
	}
	return errclass.Unset
}

// SendSampleRate is the rate Transcribe is sent audio at, implementing
// stt.SendRater. Transcribe takes 8 or 16 kHz, so audio at either goes as it is,
// and audio at any other rate is resampled to 16 kHz.
func (c *connector) SendSampleRate(sampleRate int) int {
	if sampleRate == 8000 || sampleRate == 16000 {
		return sampleRate
	}
	return fallbackSampleRate
}

// Connect opens a streaming transcription session for audio at sampleRate,
// which is the rate SendSampleRate gave, with the settings as they stand.
func (c *connector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	in, err := c.request(sampleRate)
	if err != nil {
		return nil, err
	}
	client, err := c.transcribeClient(ctx)
	if err != nil {
		return nil, err
	}
	out, err := client.StartStreamTranscription(ctx, in)
	if err != nil {
		return nil, err
	}
	es := out.GetStream()
	return &stream{
		ctx:      ctx,
		es:       es,
		events:   es.Events(),
		lang:     c.live.Language.Or(""),
		identify: in.LanguageCode == "",
	}, nil
}

// transcribeClient lazily loads AWS config (which may read the environment,
// shared config files or instance metadata) and caches a client.
func (c *connector) transcribeClient(ctx context.Context) (*transcribestreaming.Client, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.client != nil {
		return c.client, nil
	}
	awsCfg, err := awsconfig.LoadDefaultConfig(ctx, c.cfg.loadOptions()...)
	if err != nil {
		return nil, err
	}
	c.client = transcribestreaming.NewFromConfig(awsCfg)
	return c.client, nil
}

// request builds the StartStreamTranscription input for the given sample rate
// from the settings in force.
func (c *connector) request(sampleRate int) (*transcribestreaming.StartStreamTranscriptionInput, error) {
	in := &transcribestreaming.StartStreamTranscriptionInput{
		MediaEncoding:        types.MediaEncodingPcm,
		MediaSampleRateHertz: aws.Int32(int32(sampleRate)),
	}
	if err := c.setLanguage(in); err != nil {
		return nil, err
	}
	if c.cfg.PartialResultsStabilization == nil || *c.cfg.PartialResultsStabilization {
		in.EnablePartialResultsStabilization = true
		in.PartialResultsStability = types.PartialResultsStability(c.live.PartialResultsStability.Or(defaultStability))
	}
	if c.cfg.VocabularyName != "" {
		in.VocabularyName = aws.String(c.cfg.VocabularyName)
	}
	if c.cfg.VocabularyFilterName != "" {
		in.VocabularyFilterName = aws.String(c.cfg.VocabularyFilterName)
	}
	return in, nil
}

// setLanguage sets the configured language on in, or the language
// identification that replaces it when language options are set. Transcribe
// rejects a language code alongside identification.
func (c *connector) setLanguage(in *transcribestreaming.StartStreamTranscriptionInput) error {
	options, _ := c.live.LanguageOptions.Value()
	if len(options) == 0 {
		code := c.live.Language.Or("")
		if code == "" {
			return fmt.Errorf("%w: %q", errUnsupportedLanguage, code)
		}
		in.LanguageCode = types.LanguageCode(code)
		return nil
	}
	multiple := c.live.IdentifyMultipleLanguages.Or(false)
	in.IdentifyLanguage = !multiple
	in.IdentifyMultipleLanguages = multiple
	in.LanguageOptions = aws.String(strings.Join(languageOptionCodes(options), ","))
	if preferred := c.live.PreferredLanguage.Or(""); preferred != "" {
		in.PreferredLanguage = types.LanguageCode(preferred)
	}
	return nil
}

type stream struct {
	ctx    context.Context
	es     *transcribestreaming.StartStreamTranscriptionEventStream
	events <-chan types.TranscriptResultStream
	// lang is the language the session was opened to transcribe.
	lang string
	// identify reports whether the session identifies the language, so each
	// result carries the language Transcribe identified rather than lang.
	identify bool
}

// Send writes a chunk of PCM audio as an AudioEvent.
func (s *stream) Send(audio []byte) error {
	if len(audio) == 0 {
		return nil
	}
	return s.es.Send(s.ctx, &types.AudioStreamMemberAudioEvent{
		Value: types.AudioEvent{AudioChunk: audio},
	})
}

// Recv reads the next transcript event and maps its results. It returns io.EOF
// when the event stream ends, or the stream's error if one occurred.
func (s *stream) Recv() ([]stt.Result, error) {
	for {
		ev, ok := <-s.events
		if !ok {
			if err := s.es.Err(); err != nil {
				return nil, err
			}
			return nil, io.EOF
		}
		te, ok := ev.(*types.TranscriptResultStreamMemberTranscriptEvent)
		if !ok || te.Value.Transcript == nil {
			continue
		}
		if results := s.results(te.Value.Transcript.Results); len(results) > 0 {
			return results, nil
		}
	}
}

// results maps a Transcribe transcript event onto an STT result. A non-partial
// segment is final and marks the end of the current segment's transcription.
//
// One event describes one stretch of speech: its first segment, read through the
// service's own best alternative for it. The transcript is labeled with the
// language Transcribe identified for the segment when the session identifies
// the language, and otherwise with the language the session was opened with,
// which is the language the service was asked to transcribe.
func (s *stream) results(segments []types.Result) []stt.Result {
	if len(segments) == 0 {
		return nil
	}
	r := segments[0]
	if len(r.Alternatives) == 0 || r.Alternatives[0].Transcript == nil {
		return nil
	}
	text := *r.Alternatives[0].Transcript
	if text == "" {
		return nil
	}
	final := !r.IsPartial
	return []stt.Result{{
		Text:      text,
		Final:     final,
		EndOfTurn: final,
		Language:  s.resultLanguage(r),
	}}
}

// resultLanguage is the language of a result: the one Transcribe identified,
// or the configured one.
func (s *stream) resultLanguage(r types.Result) string {
	if s.identify {
		return string(r.LanguageCode)
	}
	return s.lang
}

// Close tears down the event stream.
func (s *stream) Close() error {
	return s.es.Close()
}
