package speech

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"sync"

	ttsapi "cloud.google.com/go/texttospeech/apiv1"
	"cloud.google.com/go/texttospeech/apiv1/texttospeechpb"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/ttsstream"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/provider/google/internal/gcloud"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
	"google.golang.org/api/option"
)

const (
	// defaultVoice is the voice used when none is configured.
	defaultVoice = "en-US-Chirp3-HD-Charon"
	// defaultTTSLanguageCode is the language spoken when none is configured.
	defaultTTSLanguageCode = "en-US"
	// defaultTTSSampleRate is the PCM rate requested when none is configured.
	defaultTTSSampleRate = 24000
	// minSpeakingRate and maxSpeakingRate bound the speaking rate Google
	// accepts.
	minSpeakingRate = 0.25
	maxSpeakingRate = 2.0
)

// TTSConfig configures the Google Cloud Text-to-Speech streaming service. It
// emits 16-bit mono PCM at the configured rate.
//
// Only Chirp 3 HD and Journey voices synthesize over the streaming API. The
// gemini package's Text-to-Speech service, which synthesizes each sentence in
// one request, takes the others.
type TTSConfig struct {
	// Credentials authorize the requests. Required.
	Credentials Credentials
	// Location is the Google Cloud location of a regional endpoint (e.g.
	// "us-central1"); empty uses the global endpoint.
	Location string
	// Voice is the voice name (e.g. "en-US-Chirp3-HD-Charon"); empty uses that
	// one.
	Voice string
	// Language is the language spoken, mapped to a Google language code; the
	// zero value uses US English.
	Language language.Language
	// SpeakingRate is the speaking rate, in the range [0.25, 2.0]; nil leaves
	// it to Google.
	SpeakingRate *float64
	// VoiceCloningKey selects a Chirp 3 custom voice; when set, it is used in
	// place of Voice.
	VoiceCloningKey string
	// SampleRate is the PCM rate requested and emitted downstream; 0 uses
	// 24 kHz.
	SampleRate int
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error {
	if err := validate.Struct(c); err != nil {
		return err
	}
	return c.Credentials.Validate()
}

// TTSSettings is the part of the synthesis configuration that can change while
// the pipeline runs. Each sentence is synthesized on a streaming call of its
// own, so a change takes effect on the next one.
type TTSSettings struct {
	settings.TTS

	// SpeakingRate is the speaking rate, in the range [0.25, 2.0]. No value
	// leaves it to Google.
	SpeakingRate settings.Opt[float64] `settings:"speaking_rate"`
}

// ApplyUpdate merges an update, implementing settings.Updater. A speaking
// rate outside the range Google accepts is refused with a warning: it stays as
// it was, while the rest of the update still applies.
func (s *TTSSettings) ApplyUpdate(delta any) (settings.Changed, error) {
	return applySpeakingRate(s, &s.SpeakingRate, delta)
}

// applySpeakingRate merges delta into store, keeping speakingRate as it was
// when the update gives one outside the range Google accepts.
func applySpeakingRate(store any, speakingRate *settings.Opt[float64], delta any) (settings.Changed, error) {
	kept := *speakingRate
	changed, err := settings.ApplyFields(store, delta)
	if err != nil || !changed.Has("speaking_rate") {
		return changed, err
	}
	rate, ok := speakingRate.Value()
	if !ok || (rate >= minSpeakingRate && rate <= maxSpeakingRate) {
		// No value leaves the rate to Google, which needs no check.
		return changed, nil
	}
	slog.Warn("Invalid speaking_rate value, must be between 0.25 and 2.0", "speaking_rate", rate)
	*speakingRate = kept
	delete(changed, "speaking_rate")
	return changed, nil
}

// NewTTS builds a Google Cloud Text-to-Speech streaming service. Each sentence
// is synthesized on a streaming call, and its audio is passed on in half-second
// frames as it arrives.
func NewTTS(cfg TTSConfig) *tts.Base {
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultTTSSampleRate
	}
	return tts.New("GoogleTTS", &ttsSynthesizer{cfg: cfg, live: newTTSSettings(cfg)})
}

// newTTSSettings is the starting state, taken from what the service was built
// with.
func newTTSSettings(cfg TTSConfig) *TTSSettings {
	s := &TTSSettings{}
	s.Model = settings.Cleared[string]()
	s.Voice = settings.Set(defaultVoice)
	if cfg.Voice != "" {
		s.Voice = settings.Set(cfg.Voice)
	}
	s.Language = settings.Set(defaultTTSLanguageCode)
	if cfg.Language != "" {
		s.Language = settings.Set(gcloud.TTSLanguage(cfg.Language))
	}
	s.SpeakingRate = settings.Cleared[float64]()
	if cfg.SpeakingRate != nil {
		s.SpeakingRate = settings.Set(*cfg.SpeakingRate)
	}
	return s
}

// synthesizeStream is a streaming synthesis call.
type synthesizeStream = texttospeechpb.TextToSpeech_StreamingSynthesizeClient

type ttsSynthesizer struct {
	cfg  TTSConfig
	live *TTSSettings

	// clientOptions replace the endpoint and authorization the client is built
	// with. Tests point the client at a local server this way.
	clientOptions func() []option.ClientOption

	// mu guards the client, built on first use and reused across syntheses so
	// a sentence does not pay for a fresh connection.
	mu     sync.Mutex
	client *ttsapi.Client
}

// SampleRate reports the requested PCM output rate.
func (s *ttsSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *ttsSynthesizer) Settings() any { return s.live }

// ServiceLanguage names a language the way Cloud Text-to-Speech does,
// implementing tts.LanguageNamer.
func (s *ttsSynthesizer) ServiceLanguage(l language.Language) string {
	return gcloud.TTSLanguage(l)
}

// Metadata reports the voice synthesis is billed against.
func (s *ttsSynthesizer) Metadata() tts.Metadata {
	voice, _ := s.live.Voice.Value()
	return tts.Metadata{VoiceID: voice}
}

// Close releases the client, implementing tts.Closer.
func (s *ttsSynthesizer) Close() error {
	s.mu.Lock()
	client := s.client
	s.client = nil
	s.mu.Unlock()
	if client == nil {
		return nil
	}
	return client.Close()
}

// textToSpeechClient returns the shared client, building it on first use. The
// token source it is authorized with outlives the call that built it, so it is
// not tied to that call's context.
func (s *ttsSynthesizer) textToSpeechClient(ctx context.Context) (*ttsapi.Client, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.client != nil {
		return s.client, nil
	}
	ctx = context.WithoutCancel(ctx)
	var opts []option.ClientOption
	if s.clientOptions != nil {
		opts = s.clientOptions()
	} else {
		resolved, err := s.cfg.Credentials.Resolve(ctx)
		if err != nil {
			return nil, err
		}
		opts = []option.ClientOption{option.WithTokenSource(resolved.TokenSource)}
		if s.cfg.Location != "" {
			opts = append(opts, option.WithEndpoint(s.cfg.Location+"-texttospeech.googleapis.com:443"))
		}
	}
	client, err := ttsapi.NewClient(ctx, opts...)
	if err != nil {
		return nil, err
	}
	s.client = client
	return client, nil
}

// streamingConfig builds the configuration a streaming call opens with: the
// voice, or the cloned voice when a cloning key is configured, and raw PCM at
// the configured rate.
func (s *ttsSynthesizer) streamingConfig() *texttospeechpb.StreamingSynthesizeConfig {
	languageCode, _ := s.live.Language.Value()
	voice := &texttospeechpb.VoiceSelectionParams{LanguageCode: languageCode}
	if s.cfg.VoiceCloningKey != "" {
		voice.VoiceClone = &texttospeechpb.VoiceCloneParams{VoiceCloningKey: s.cfg.VoiceCloningKey}
	} else {
		voice.Name, _ = s.live.Voice.Value()
	}
	speakingRate, _ := s.live.SpeakingRate.Value()
	return &texttospeechpb.StreamingSynthesizeConfig{
		Voice: voice,
		StreamingAudioConfig: &texttospeechpb.StreamingAudioConfig{
			AudioEncoding:   texttospeechpb.AudioEncoding_PCM,
			SampleRateHertz: int32(s.cfg.SampleRate), //nolint:gosec // G115: an audio sample rate
			SpeakingRate:    speakingRate,
		},
	}
}

// RunTTS synthesizes text on a streaming call and passes its audio on as it
// arrives.
func (s *ttsSynthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	return s.streamTTS(ctx, s.streamingConfig(), text, yield)
}

// streamTTS opens a streaming call, sends the configuration and then the text,
// and passes the audio on in frames of tts.ChunkSize, the last one holding
// whatever is left.
func (s *ttsSynthesizer) streamTTS(
	ctx context.Context, config *texttospeechpb.StreamingSynthesizeConfig, text string,
	yield func(f frames.Frame) error,
) error {
	client, err := s.textToSpeechClient(ctx)
	if err != nil {
		return err
	}
	call, err := client.StreamingSynthesize(ctx)
	if err != nil {
		return err
	}
	if err := s.sendRequests(call, config, text); err != nil {
		return err
	}
	tts.StartUsageMetrics(ctx, text)

	emit := ttsstream.PCMYielder(yield, s.SampleRate())
	chunkSize := tts.ChunkSize(s.SampleRate())
	var audioBuffer []byte
	for {
		response, err := call.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return err
		}
		chunk := response.GetAudioContent()
		if len(chunk) == 0 {
			continue
		}
		audioBuffer = append(audioBuffer, chunk...)
		for len(audioBuffer) >= chunkSize {
			piece := audioBuffer[:chunkSize:chunkSize]
			audioBuffer = audioBuffer[chunkSize:]
			if err := emit(piece); err != nil {
				return err
			}
		}
	}
	return emit(audioBuffer)
}

// sendRequests writes the configuration and the text, then half-closes the
// call so Google synthesizes it and ends the call. A call Google has already
// ended takes nothing more, and why it ended is read from its responses.
func (s *ttsSynthesizer) sendRequests(
	call synthesizeStream, config *texttospeechpb.StreamingSynthesizeConfig, text string,
) error {
	err := s.send(call, config, text)
	if errors.Is(err, io.EOF) {
		return nil
	}
	return err
}

func (s *ttsSynthesizer) send(
	call synthesizeStream, config *texttospeechpb.StreamingSynthesizeConfig, text string,
) error {
	err := call.Send(&texttospeechpb.StreamingSynthesizeRequest{
		StreamingRequest: &texttospeechpb.StreamingSynthesizeRequest_StreamingConfig{StreamingConfig: config},
	})
	if err != nil {
		return err
	}
	err = call.Send(&texttospeechpb.StreamingSynthesizeRequest{
		StreamingRequest: &texttospeechpb.StreamingSynthesizeRequest_Input{
			Input: &texttospeechpb.StreamingSynthesisInput{
				InputSource: &texttospeechpb.StreamingSynthesisInput_Text{Text: text},
			},
		},
	})
	if err != nil {
		return err
	}
	return call.CloseSend()
}
