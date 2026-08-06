package transcribe

import (
	"context"
	"io"
	"sync"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/transcribestreaming"
	"github.com/aws/aws-sdk-go-v2/service/transcribestreaming/types"
	"github.com/gojargo/jargo/frames"
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
	if cfg.Language == "" {
		cfg.Language = defaultLanguage
	}
	if cfg.PartialResultsStability == "" {
		cfg.PartialResultsStability = defaultStability
	}
	return stt.NewStream("TranscribeSTT", &connector{cfg: cfg}, cfg.SampleRate)
}

type connector struct {
	cfg Config

	mu     sync.Mutex
	client *transcribestreaming.Client
}

// Metadata reports Transcribe's turn strategy and latency to downstream
// processors. It finalizes per segment, so it leaves turn detection to the
// pipeline rather than recommending external turns.
func (c *connector) Metadata() stt.Metadata {
	return stt.Metadata{
		RecommendedUserTurns: frames.UserTurnUnspecified,
		TTFSP99:              ttfsP99,
	}
}

// Connect opens a streaming transcription session at the resolved sample rate.
func (c *connector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	client, err := c.transcribeClient(ctx)
	if err != nil {
		return nil, err
	}
	rate := c.cfg.SampleRate
	if rate == 0 {
		rate = sampleRate
	}
	if rate != 8000 && rate != 16000 {
		rate = fallbackSampleRate
	}
	out, err := client.StartStreamTranscription(ctx, c.cfg.request(rate))
	if err != nil {
		return nil, err
	}
	es := out.GetStream()
	return &stream{ctx: ctx, es: es, events: es.Events(), lang: c.cfg.Language}, nil
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

// request builds the StartStreamTranscription input for the given sample rate.
func (c Config) request(sampleRate int) *transcribestreaming.StartStreamTranscriptionInput {
	in := &transcribestreaming.StartStreamTranscriptionInput{
		LanguageCode:         types.LanguageCode(c.Language),
		MediaEncoding:        types.MediaEncodingPcm,
		MediaSampleRateHertz: aws.Int32(int32(sampleRate)),
	}
	if c.PartialResultsStabilization == nil || *c.PartialResultsStabilization {
		in.EnablePartialResultsStabilization = true
		in.PartialResultsStability = types.PartialResultsStability(c.PartialResultsStability)
	}
	if c.VocabularyName != "" {
		in.VocabularyName = aws.String(c.VocabularyName)
	}
	if c.VocabularyFilterName != "" {
		in.VocabularyFilterName = aws.String(c.VocabularyFilterName)
	}
	return in
}

type stream struct {
	ctx    context.Context
	es     *transcribestreaming.StartStreamTranscriptionEventStream
	events <-chan types.TranscriptResultStream
	lang   string
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
// language the session was opened with, which is the language the service was
// asked to transcribe.
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
		Language:  s.lang,
	}}
}

// Close tears down the event stream.
func (s *stream) Close() error {
	return s.es.Close()
}
