package speech

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"slices"
	"sync"
	"time"

	speechapi "cloud.google.com/go/speech/apiv2"
	"cloud.google.com/go/speech/apiv2/speechpb"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
	"google.golang.org/api/option"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// StreamingLimit is how long one streaming recognition call is fed audio before
// it is replaced by another. Google ends a streaming call after five minutes,
// so the call is half-closed at four, its last results are read, and the next
// call carries on with the audio that arrived meanwhile.
const StreamingLimit = 4 * time.Minute

const (
	// defaultSTTModel is the recognition model used when none is configured.
	defaultSTTModel = "latest_long"
	// defaultSTTLanguageCode is the language recognized when none is given.
	defaultSTTLanguageCode = "en-US"
	// reconnectDelay is how long a failed streaming call is waited out before
	// the next one is opened.
	reconnectDelay = time.Second
)

// errNoProject is returned when no project could be found to address the
// recognizer by.
//
//nolint:gochecknoglobals // sentinel error
var errNoProject = errors.New("google stt: project ID not found in credentials")

// STTConfig configures the Google Cloud Speech-to-Text V2 streaming service.
// It is sent 16-bit mono PCM at the pipeline's input rate, or at SampleRate.
type STTConfig struct {
	// Credentials authorize the requests. Required.
	Credentials Credentials
	// ProjectID is the Google Cloud project the recognizer belongs to. Empty
	// takes the project_id of the service-account key; it is required with a
	// token source, which names no project.
	ProjectID string
	// Location is the Google Cloud location (e.g. "global", "us-central1",
	// "us"); empty uses "global". Any other location is served by its regional
	// endpoint.
	Location string
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int

	// Languages are the languages to recognize, the first being the primary
	// one; empty recognizes US English.
	Languages []language.Language
	// Model is the recognition model; empty uses "latest_long".
	Model string
	// UseSeparateRecognitionPerChannel processes each audio channel separately.
	UseSeparateRecognitionPerChannel bool
	// EnableAutomaticPunctuation adds punctuation to transcripts; nil defaults
	// to true.
	EnableAutomaticPunctuation *bool
	// EnableSpokenPunctuation includes spoken punctuation in the transcript.
	EnableSpokenPunctuation bool
	// EnableSpokenEmojis includes spoken emojis in the transcript.
	EnableSpokenEmojis bool
	// ProfanityFilter filters profanity from the transcript.
	ProfanityFilter bool
	// EnableWordTimeOffsets includes timing information for each word.
	EnableWordTimeOffsets bool
	// EnableWordConfidence includes confidence scores for each word.
	EnableWordConfidence bool
	// EnableInterimResults streams partial recognition results; nil defaults
	// to true.
	EnableInterimResults *bool
	// EnableVoiceActivityEvents detects voice activity in the audio.
	EnableVoiceActivityEvents bool
	// Adaptation biases recognition toward specific words: a
	// *speechpb.SpeechAdaptation, or the equivalent map. Each "phrase_sets"
	// entry is either the resource name of a phrase set or an inline one:
	//
	//	map[string]any{"phrase_sets": []any{
	//		"projects/my-project/locations/global/phraseSets/catalog",
	//		map[string]any{"phrases": []any{map[string]any{"value": "jargo", "boost": 15.0}}},
	//	}}
	//
	// Referenced phrase sets must live in the same location as the service.
	// The telephony model rejects adaptation and transcribes without it;
	// support otherwise varies by model and language.
	Adaptation any
	// DenoiserConfig is background-noise removal: a *speechpb.DenoiserConfig,
	// or the equivalent map such as map[string]any{"denoise_audio": true}. Only
	// the Chirp 3 models denoise.
	DenoiserConfig any

	// TTFSP99 overrides the measured transcript latency the turn strategies
	// size their wait by; nil uses stt.GoogleTTFSP99, and zero means no wait.
	TTFSP99 *time.Duration
}

// Validate reports whether the configuration is usable: there are credentials,
// a project to address the recognizer by, and an adaptation and denoiser
// config Google's API can represent.
func (c STTConfig) Validate() error {
	if err := validate.Struct(c); err != nil {
		return err
	}
	if err := c.Credentials.Validate(); err != nil {
		return err
	}
	if _, err := c.projectID(); err != nil {
		return err
	}
	if c.Adaptation != nil {
		if _, err := normalizeSpeechAdaptation(c.Adaptation); err != nil {
			return err
		}
	}
	if c.DenoiserConfig != nil {
		if _, err := denoiserConfig(c.DenoiserConfig); err != nil {
			return err
		}
	}
	return nil
}

// projectID is the project the recognizer is addressed by: the configured one,
// or the one the service-account key names.
func (c STTConfig) projectID() (string, error) {
	if c.ProjectID != "" {
		return c.ProjectID, nil
	}
	project, err := c.Credentials.ProjectID()
	if err != nil {
		return "", err
	}
	if project == "" {
		return "", errNoProject
	}
	return project, nil
}

// STTSettings is the part of the Speech-to-Text configuration that can change
// while the pipeline runs. Google is told all of it when a streaming call
// opens, so a change to any of it reopens the session.
type STTSettings struct {
	settings.STT

	// Languages are the languages to recognize, the first being the primary
	// one. An update giving a language replaces them with that language.
	Languages settings.Opt[[]language.Language] `settings:"languages"`
	// UseSeparateRecognitionPerChannel processes each audio channel separately.
	UseSeparateRecognitionPerChannel settings.Opt[bool] `settings:"use_separate_recognition_per_channel"`
	// EnableAutomaticPunctuation adds punctuation to transcripts.
	EnableAutomaticPunctuation settings.Opt[bool] `settings:"enable_automatic_punctuation"`
	// EnableSpokenPunctuation includes spoken punctuation in the transcript.
	EnableSpokenPunctuation settings.Opt[bool] `settings:"enable_spoken_punctuation"`
	// EnableSpokenEmojis includes spoken emojis in the transcript.
	EnableSpokenEmojis settings.Opt[bool] `settings:"enable_spoken_emojis"`
	// ProfanityFilter filters profanity from the transcript.
	ProfanityFilter settings.Opt[bool] `settings:"profanity_filter"`
	// EnableWordTimeOffsets includes timing information for each word.
	EnableWordTimeOffsets settings.Opt[bool] `settings:"enable_word_time_offsets"`
	// EnableWordConfidence includes confidence scores for each word.
	EnableWordConfidence settings.Opt[bool] `settings:"enable_word_confidence"`
	// EnableInterimResults streams partial recognition results.
	EnableInterimResults settings.Opt[bool] `settings:"enable_interim_results"`
	// EnableVoiceActivityEvents detects voice activity in the audio.
	EnableVoiceActivityEvents settings.Opt[bool] `settings:"enable_voice_activity_events"`
	// Adaptation biases recognition toward specific words. See
	// STTConfig.Adaptation.
	Adaptation settings.Opt[any] `settings:"adaptation"`
	// DenoiserConfig is background-noise removal. See
	// STTConfig.DenoiserConfig.
	DenoiserConfig settings.Opt[any] `settings:"denoiser_config"`
}

// ApplyUpdate merges an update, implementing settings.Updater. Google reads the
// languages when a call opens, not the language, so a language the update
// gives replaces the languages rather than being stored.
func (s *STTSettings) ApplyUpdate(delta any) (settings.Changed, error) {
	value, move := settings.Get(delta, "language")
	code, _ := value.(string)
	move = move && code != ""
	kept := s.Language
	changed, err := settings.ApplyFields(s, delta)
	if err != nil || !move {
		return changed, err
	}
	s.Language = kept
	delete(changed, "language")
	languages := []language.Language{language.Language(code)}
	if old, ok := s.Languages.Value(); !ok || !slices.Equal(old, languages) {
		changed["languages"] = s.Languages.AsAny()
		s.Languages = settings.Set(languages)
	}
	return changed, nil
}

// newSTTSettings is the starting state, taken from what the service was built
// with.
func newSTTSettings(cfg STTConfig) *STTSettings {
	s := &STTSettings{}
	s.Model = settings.Set(cfg.Model)
	s.Language = settings.Cleared[string]()
	languages := cfg.Languages
	if len(languages) == 0 {
		languages = []language.Language{language.EnglishUS}
	}
	s.Languages = settings.Set(slices.Clone(languages))
	s.UseSeparateRecognitionPerChannel = settings.Set(cfg.UseSeparateRecognitionPerChannel)
	s.EnableAutomaticPunctuation = settings.Set(cfg.EnableAutomaticPunctuation == nil || *cfg.EnableAutomaticPunctuation)
	s.EnableSpokenPunctuation = settings.Set(cfg.EnableSpokenPunctuation)
	s.EnableSpokenEmojis = settings.Set(cfg.EnableSpokenEmojis)
	s.ProfanityFilter = settings.Set(cfg.ProfanityFilter)
	s.EnableWordTimeOffsets = settings.Set(cfg.EnableWordTimeOffsets)
	s.EnableWordConfidence = settings.Set(cfg.EnableWordConfidence)
	s.EnableInterimResults = settings.Set(cfg.EnableInterimResults == nil || *cfg.EnableInterimResults)
	s.EnableVoiceActivityEvents = settings.Set(cfg.EnableVoiceActivityEvents)
	s.Adaptation = settings.Cleared[any]()
	if cfg.Adaptation != nil {
		// Normalized on the way in where it can be. One that cannot is kept
		// as it was given: Validate reports it, and a session is not opened
		// with it.
		adaptation := cfg.Adaptation
		if normalized, err := normalizeSpeechAdaptation(adaptation); err == nil {
			adaptation = normalized
		}
		s.Adaptation = settings.Set(adaptation)
	}
	s.DenoiserConfig = settings.Cleared[any]()
	if cfg.DenoiserConfig != nil {
		s.DenoiserConfig = settings.Set(cfg.DenoiserConfig)
	}
	return s
}

// STTService is the Google Cloud speech-to-text service NewSTT builds.
type STTService struct {
	*stt.StreamService
}

// NewSTT builds a Google Cloud Speech-to-Text V2 streaming service.
//
// Audio is streamed to Google over one streaming recognition call, which is
// replaced before Google's limit on a call's length (see StreamingLimit). A
// call that fails is reported and replaced a second later. A final result
// marks the end of the user's turn.
func NewSTT(cfg STTConfig) *STTService {
	s := &STTService{}
	s.StreamService = buildSTT(cfg, stt.WithSelf(s))
	return s
}

// buildSTT builds the service NewSTT returns, registering the
// service opts names as the processor the pipeline runs.
func buildSTT(cfg STTConfig, opts ...stt.Option) *stt.StreamService {
	svc, _ := newSTT(cfg, nil, opts...)
	return svc
}

// newSTT builds the service and the connector driving it, its clients built
// with clientOptions when that is not nil.
func newSTT(
	cfg STTConfig, clientOptions func() []option.ClientOption, opts ...stt.Option,
) (*stt.StreamService, *sttConnector) {
	if cfg.Location == "" {
		cfg.Location = defaultLocation
	}
	if cfg.Model == "" {
		cfg.Model = defaultSTTModel
	}
	c := &sttConnector{
		cfg:            cfg,
		live:           newSTTSettings(cfg),
		streamingLimit: StreamingLimit,
		clientOptions:  clientOptions,
	}
	svc := stt.NewStream("GoogleSTT", c, cfg.SampleRate, opts...)
	c.svc = svc
	return svc, c
}

// sttConnector opens streaming recognition sessions.
type sttConnector struct {
	cfg  STTConfig
	live *STTSettings
	// svc is the service this connector drives, for reporting a failed call.
	// It is nil for a connector built outside NewSTT.
	svc *stt.StreamService
	// streamingLimit is how long a call is fed before it is replaced.
	streamingLimit time.Duration
	// clientOptions replace the endpoint and authorization a client is built
	// with. Tests point the client at a local server this way.
	clientOptions func() []option.ClientOption
}

// Metadata reports the transcript latency the turn strategies size their wait
// by, and the model recognition is billed against.
func (c *sttConnector) Metadata() stt.Metadata {
	model, _ := c.live.Model.Value()
	return stt.Metadata{
		TTFSP99: stt.TTFSP99Or(c.cfg.TTFSP99, stt.GoogleTTFSP99),
		Model:   model,
	}
}

// Settings is the configuration a caller may change while the pipeline runs.
func (c *sttConnector) Settings() any { return c.live }

// PrepareSettings normalizes the adaptation an update gives, implementing
// stt.SettingsPreparer, so one Google's API cannot represent is refused before
// anything is stored.
func (c *sttConnector) PrepareSettings(delta any) error {
	adaptation, ok := settings.Get(delta, "adaptation")
	if !ok {
		return nil
	}
	normalized, err := normalizeSpeechAdaptation(adaptation)
	if err != nil {
		return err
	}
	return settings.SetNamed(delta, "adaptation", normalized)
}

// UpdateSettings asks for the session to be reopened whenever anything changed.
// Google is told all of it when a streaming call opens, so a change reaches it
// only by opening another.
func (c *sttConnector) UpdateSettings(context.Context, settings.Changed) (bool, error) {
	slog.Debug("Reconnecting stream due to configuration changes")
	return true, nil
}

// languageCodes resolves the languages to the codes Google names them by,
// falling back to US English.
func (c *sttConnector) languageCodes() []string {
	languages, _ := c.live.Languages.Value()
	if len(languages) == 0 {
		return []string{defaultSTTLanguageCode}
	}
	codes := make([]string, 0, len(languages))
	for _, l := range languages {
		codes = append(codes, languageToService(l))
	}
	return codes
}

// languageToService names a language the way Speech-to-Text V2 does, falling
// back to US English for one that names nothing.
func languageToService(l language.Language) string {
	if code := languageToGoogleSTTLanguage(l); code != "" {
		return code
	}
	return defaultSTTLanguageCode
}

// speechAdaptation is the adaptation to send, or nil for none.
func (c *sttConnector) speechAdaptation() (*speechpb.SpeechAdaptation, error) {
	adaptation, ok := c.live.Adaptation.Value()
	if !ok || adaptation == nil {
		return nil, nil //nolint:nilnil // no adaptation configured
	}
	return normalizeSpeechAdaptation(adaptation)
}

// denoiserConfig is the denoiser config to send, or nil for none.
func (c *sttConnector) denoiserConfig() (*speechpb.DenoiserConfig, error) {
	config, ok := c.live.DenoiserConfig.Value()
	if !ok || config == nil {
		return nil, nil //nolint:nilnil // no denoiser configured
	}
	return denoiserConfig(config)
}

// streamingConfig builds the configuration a streaming call opens with, for
// audio at sampleRate.
func (c *sttConnector) streamingConfig(sampleRate int) (*speechpb.StreamingRecognitionConfig, error) {
	model, _ := c.live.Model.Value()
	recognitionConfig := &speechpb.RecognitionConfig{
		DecodingConfig: &speechpb.RecognitionConfig_ExplicitDecodingConfig{
			ExplicitDecodingConfig: &speechpb.ExplicitDecodingConfig{
				Encoding:          speechpb.ExplicitDecodingConfig_LINEAR16,
				SampleRateHertz:   int32(sampleRate), //nolint:gosec // G115: an audio sample rate
				AudioChannelCount: 1,
			},
		},
		LanguageCodes: c.languageCodes(),
		Model:         model,
		Features: &speechpb.RecognitionFeatures{
			EnableAutomaticPunctuation: c.live.EnableAutomaticPunctuation.Or(false),
			EnableSpokenPunctuation:    c.live.EnableSpokenPunctuation.Or(false),
			EnableSpokenEmojis:         c.live.EnableSpokenEmojis.Or(false),
			ProfanityFilter:            c.live.ProfanityFilter.Or(false),
			EnableWordTimeOffsets:      c.live.EnableWordTimeOffsets.Or(false),
			EnableWordConfidence:       c.live.EnableWordConfidence.Or(false),
		},
	}

	denoiser, err := c.denoiserConfig()
	if err != nil {
		return nil, err
	}
	if denoiser != nil {
		recognitionConfig.DenoiserConfig = denoiser
	}

	adaptation, err := c.speechAdaptation()
	if err != nil {
		return nil, err
	}
	if adaptation != nil {
		if modelSupportsAdaptation(model) {
			recognitionConfig.Adaptation = adaptation
		} else {
			slog.Warn("Google STT model rejects adaptation; transcribing without it", "model", model)
		}
	}

	return &speechpb.StreamingRecognitionConfig{
		Config: recognitionConfig,
		StreamingFeatures: &speechpb.StreamingRecognitionFeatures{
			EnableVoiceActivityEvents: c.live.EnableVoiceActivityEvents.Or(false),
			InterimResults:            c.live.EnableInterimResults.Or(false),
		},
	}, nil
}

// newClient builds the client a session streams on, authorized with the
// configured credentials and addressed to the configured location's endpoint.
func (c *sttConnector) newClient(ctx context.Context) (*speechapi.Client, error) {
	var opts []option.ClientOption
	if c.clientOptions != nil {
		opts = c.clientOptions()
	} else {
		resolved, err := c.cfg.Credentials.Resolve(ctx)
		if err != nil {
			return nil, err
		}
		opts = []option.ClientOption{option.WithTokenSource(resolved.TokenSource)}
		if c.cfg.Location != defaultLocation {
			opts = append(opts, option.WithEndpoint(c.cfg.Location+"-speech.googleapis.com:443"))
		}
	}
	return speechapi.NewClient(ctx, opts...)
}

// Connect builds the streaming configuration from the settings as they stand
// and starts the task that streams the audio Send queues to Google. A setting
// Google's API cannot represent, or a project that cannot be found, will not
// change on a retry, so either is reported as permanent.
func (c *sttConnector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	slog.Debug("Connecting to Google Speech-to-Text")

	project, err := c.cfg.projectID()
	if err != nil {
		return nil, stt.Permanent(err)
	}
	config, err := c.streamingConfig(sampleRate)
	if err != nil {
		return nil, stt.Permanent(err)
	}
	client, err := c.newClient(ctx)
	if err != nil {
		return nil, err
	}

	streamCtx, cancel := context.WithCancel(ctx)
	s := &sttStream{
		connector:  c,
		client:     client,
		config:     config,
		recognizer: fmt.Sprintf("projects/%s/locations/%s/recognizers/_", project, c.cfg.Location),
		language:   c.languageCodes()[0],
		limit:      c.streamingLimit,
		ctx:        streamCtx,
		cancel:     cancel,
		queue:      newRequestQueue(),
		results:    make(chan []stt.Result),
		done:       make(chan struct{}),
	}
	s.setStreamStartTime(time.Now())
	go s.streamAudio()
	return s, nil
}

// pushError reports a failed call on the pipeline. It is not fatal: the call
// is replaced.
func (c *sttConnector) pushError(ctx context.Context, err error) {
	msg := fmt.Sprintf("Unknown error occurred: %v", err)
	if c.svc == nil {
		slog.Error(msg)
		return
	}
	c.svc.PushError(ctx, msg, err, false)
}

// recognizeStream is the response side of a streaming recognition call.
type recognizeStream interface {
	Recv() (*speechpb.StreamingRecognizeResponse, error)
}

// sttStream is one streaming recognition session. Audio is queued by Send and
// streamed to Google by a task of its own, over as many streaming calls as the
// session lasts: one is replaced when it reaches the streaming limit, and when
// it fails.
type sttStream struct {
	connector  *sttConnector
	client     *speechapi.Client
	config     *speechpb.StreamingRecognitionConfig
	recognizer string
	// language is the primary language, which the transcripts are labeled with.
	language string
	limit    time.Duration

	ctx    context.Context
	cancel context.CancelFunc
	queue  *requestQueue
	// results carries what a call recognized to Recv.
	results chan []stt.Result
	// done is closed when the streaming task ends, and err says why.
	done chan struct{}
	err  error

	mu sync.Mutex
	// streamStartTime is when the current call was opened, which is what the
	// streaming limit is measured from.
	streamStartTime time.Time
}

// Send queues a chunk of PCM for the streaming task to send.
func (s *sttStream) Send(audio []byte) error {
	s.queue.put(slices.Clone(audio))
	return nil
}

// Recv returns the next batch of results. It returns io.EOF when the streaming
// task ended because Google ended a call before the limit, and the context's
// error when the session was closed.
func (s *sttStream) Recv() ([]stt.Result, error) {
	select {
	case r := <-s.results:
		return r, nil
	case <-s.done:
		return nil, s.err
	}
}

// Close stops the streaming task, which ends the call it is on, and releases
// the client.
func (s *sttStream) Close() error {
	s.cancel()
	<-s.done
	return s.client.Close()
}

func (s *sttStream) setStreamStartTime(t time.Time) {
	s.mu.Lock()
	s.streamStartTime = t
	s.mu.Unlock()
}

// limitReached reports whether the current call has been open past the
// streaming limit.
func (s *sttStream) limitReached() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return time.Since(s.streamStartTime) > s.limit
}

// streamAudio runs streaming calls for as long as the session lasts. A call
// opens once there is audio to send. One that ended at the streaming limit is
// followed by the next at once; one that failed is reported and followed by
// the next a second later; one Google ended on its own ends the session.
func (s *sttStream) streamAudio() {
	defer close(s.done)
	for {
		if !s.queue.wait(s.ctx) {
			s.err = s.ctx.Err()
			return
		}

		err := s.recognize()
		if s.ctx.Err() != nil {
			s.err = s.ctx.Err()
			return
		}

		if err == nil {
			if s.limitReached() {
				slog.Debug("Reconnecting stream after timeout")
				s.setStreamStartTime(time.Now())
				continue
			}
			// Normal stream end.
			s.err = io.EOF
			return
		}

		s.connector.pushError(s.ctx, err)
		select {
		case <-s.ctx.Done():
			s.err = s.ctx.Err()
			return
		case <-time.After(reconnectDelay):
		}
		s.setStreamStartTime(time.Now())
	}
}

// recognize runs one streaming call: the requests go out on a task of their
// own while the responses are read here, until Google ends the call.
func (s *sttStream) recognize() error {
	callCtx, cancel := context.WithCancel(s.ctx)
	defer cancel()
	call, err := s.client.StreamingRecognize(callCtx)
	if err != nil {
		return err
	}
	var requests sync.WaitGroup
	requests.Go(func() { s.requestGenerator(callCtx, call) })
	err = s.processResponses(call)
	// The responses have ended, so the requests have nowhere to go.
	cancel()
	requests.Wait()
	return err
}

// requestGenerator sends the configuration, then the queued audio, until the
// call reaches the streaming limit. The chunk that found the limit reached is
// put back in the queue for the next call, and the call is half-closed so
// Google returns what it still holds and ends it.
func (s *sttStream) requestGenerator(ctx context.Context, call speechpb.Speech_StreamingRecognizeClient) {
	slog.Debug("Using recognizer path", "recognizer", s.recognizer)

	// Send initial config.
	err := call.Send(&speechpb.StreamingRecognizeRequest{
		Recognizer:       s.recognizer,
		StreamingRequest: &speechpb.StreamingRecognizeRequest_StreamingConfig{StreamingConfig: s.config},
	})
	if err != nil {
		// The call has failed, which its responses report.
		return
	}

	for {
		audio, ok := s.queue.get(ctx)
		if !ok {
			return
		}

		if s.limitReached() {
			slog.Debug("Streaming limit reached, initiating graceful reconnection")
			// Put the current audio chunk back in the queue.
			s.queue.put(audio)
			_ = call.CloseSend()
			return
		}

		err := call.Send(&speechpb.StreamingRecognizeRequest{
			StreamingRequest: &speechpb.StreamingRecognizeRequest_Audio{Audio: audio},
		})
		if err != nil {
			return
		}
	}
}

// processResponses reads a call's responses until Google ends it, handing
// what it recognized to Recv. Every result is handed on, including those that
// arrive past the streaming limit: the final for the audio in flight is
// produced by the half-close at the limit, so it arrives after it.
//
// A call Google aborts for inactivity, which it does after ten seconds with no
// audio, is not unexpected and is only logged; the session replaces it all the
// same.
func (s *sttStream) processResponses(call recognizeStream) error {
	for {
		response, err := call.Recv()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			if s.ctx.Err() != nil {
				return err
			}
			if status.Code(err) == codes.Aborted {
				slog.Debug("Google STT stream aborted due to inactivity (no audio input), reconnecting automatically")
				return err
			}
			s.connector.pushError(s.ctx, err)
			return err
		}

		results := s.responseResults(response)
		if len(results) == 0 {
			continue
		}
		select {
		case s.results <- results:
		case <-s.ctx.Done():
			return s.ctx.Err()
		}
	}
}

// responseResults turns a response into transcription results. A result with
// no alternative, or an empty transcript, says nothing. A final result ends the
// user's turn.
func (s *sttStream) responseResults(response *speechpb.StreamingRecognizeResponse) []stt.Result {
	var out []stt.Result
	for _, result := range response.GetResults() {
		alternatives := result.GetAlternatives()
		if len(alternatives) == 0 {
			continue
		}
		transcript := alternatives[0].GetTranscript()
		if transcript == "" {
			continue
		}
		final := result.GetIsFinal()
		out = append(out, stt.Result{
			Text:      transcript,
			Final:     final,
			EndOfTurn: final,
			Language:  s.language,
		})
	}
	return out
}

// requestQueue is the audio waiting to be sent, unbounded so queuing never
// blocks the pipeline.
type requestQueue struct {
	mu    sync.Mutex
	items [][]byte
	// ready is signaled when an item is added.
	ready chan struct{}
}

func newRequestQueue() *requestQueue {
	return &requestQueue{ready: make(chan struct{}, 1)}
}

// put adds a chunk at the back of the queue.
func (q *requestQueue) put(audio []byte) {
	q.mu.Lock()
	q.items = append(q.items, audio)
	q.mu.Unlock()
	select {
	case q.ready <- struct{}{}:
	default:
	}
}

// get takes the chunk at the front of the queue, waiting for one. It reports
// false when ctx ends first.
func (q *requestQueue) get(ctx context.Context) ([]byte, bool) {
	for {
		q.mu.Lock()
		if len(q.items) > 0 {
			audio := q.items[0]
			q.items[0] = nil
			q.items = q.items[1:]
			q.mu.Unlock()
			return audio, true
		}
		q.mu.Unlock()
		select {
		case <-q.ready:
		case <-ctx.Done():
			return nil, false
		}
	}
}

// wait blocks until the queue holds a chunk, without taking it. It reports
// false when ctx ends first.
func (q *requestQueue) wait(ctx context.Context) bool {
	for {
		q.mu.Lock()
		waiting := len(q.items) > 0
		q.mu.Unlock()
		if waiting {
			return true
		}
		select {
		case <-q.ready:
		case <-ctx.Done():
			return false
		}
	}
}
