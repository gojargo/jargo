package speech

import (
	"bytes"
	"testing"
	"time"

	"cloud.google.com/go/texttospeech/apiv1/texttospeechpb"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// newTestSynthesizer builds a synthesizer for cfg talking to server.
func newTestSynthesizer(t *testing.T, cfg TTSConfig, server *fakeServer) *ttsSynthesizer {
	t.Helper()
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultTTSSampleRate
	}
	cfg.Credentials = testCredentials()
	s := &ttsSynthesizer{cfg: cfg, live: newTTSSettings(cfg), clientOptions: server.clientOptions(t)}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

// synthesizeFrames runs one synthesis and returns the audio frames it yielded.
func synthesizeFrames(t *testing.T, s *ttsSynthesizer, text string) ([]*frames.TTSAudioRawFrame, error) {
	t.Helper()
	var out []*frames.TTSAudioRawFrame
	err := s.RunTTS(t.Context(), text, "ctx-1", func(f frames.Frame) error {
		audio, ok := f.(*frames.TTSAudioRawFrame)
		if !ok {
			t.Errorf("yielded %T, want a TTSAudioRawFrame", f)
			return nil
		}
		out = append(out, audio)
		return nil
	})
	return out, err
}

// nextSynthesis waits for the server to finish a call and returns it.
func nextSynthesis(t *testing.T, f *fakeTextToSpeech) *synthesizeCall {
	t.Helper()
	select {
	case c := <-f.calls:
		return c
	case <-time.After(3 * time.Second):
		t.Fatal("no synthesis call arrived")
		return nil
	}
}

// TestStreamingSynthesisRequests checks a call opens with the voice, the
// language and raw PCM at the configured rate, then sends the text.
func TestStreamingSynthesisRequests(t *testing.T) {
	fake, server := newFakeTextToSpeech(t, [][]byte{{1, 2}}, nil)
	rate := 1.25
	s := newTestSynthesizer(t, TTSConfig{
		Voice: "en-GB-Chirp3-HD-Kore", Language: language.EnglishGB, SpeakingRate: &rate, SampleRate: 16000,
	}, server)

	if _, err := synthesizeFrames(t, s, "hello there"); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	call := nextSynthesis(t, fake)
	if len(call.requests) != 2 {
		t.Fatalf("requests = %v, want the configuration and the text", call.requests)
	}

	config := call.requests[0].GetStreamingConfig()
	if config.GetVoice().GetName() != "en-GB-Chirp3-HD-Kore" || config.GetVoice().GetLanguageCode() != "en-GB" {
		t.Errorf("voice = %v", config.GetVoice())
	}
	audio := config.GetStreamingAudioConfig()
	if audio.GetAudioEncoding() != texttospeechpb.AudioEncoding_PCM || audio.GetSampleRateHertz() != 16000 ||
		audio.GetSpeakingRate() != rate {
		t.Errorf("audio config = %v, want PCM at 16000 and the speaking rate", audio)
	}
	if got := call.requests[1].GetInput().GetText(); got != "hello there" {
		t.Errorf("text = %q, want the text to speak", got)
	}
}

// TestVoiceCloningKeyReplacesTheVoice checks a cloning key selects the cloned
// voice in place of the named one.
func TestVoiceCloningKeyReplacesTheVoice(t *testing.T) {
	fake, server := newFakeTextToSpeech(t, nil, nil)
	s := newTestSynthesizer(t, TTSConfig{VoiceCloningKey: "clone-key"}, server)

	if _, err := synthesizeFrames(t, s, "hi"); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	voice := nextSynthesis(t, fake).requests[0].GetStreamingConfig().GetVoice()
	if voice.GetVoiceClone().GetVoiceCloningKey() != "clone-key" || voice.GetName() != "" {
		t.Errorf("voice = %v, want the cloned voice and no name", voice)
	}
	if voice.GetLanguageCode() != "en-US" {
		t.Errorf("language = %q, want the default en-US", voice.GetLanguageCode())
	}
}

// TestAudioIsChunked checks the audio is passed on in frames of the chunk size
// whatever sizes it arrives in, the last frame holding what is left.
func TestAudioIsChunked(t *testing.T) {
	const rate = 16000
	chunk := tts.ChunkSize(rate)
	pcm := bytes.Repeat([]byte{1, 2, 3, 4}, (chunk*2+100)/4)
	parts := [][]byte{pcm[:chunk/3], pcm[chunk/3 : chunk+50], {}, pcm[chunk+50:]}
	_, server := newFakeTextToSpeech(t, parts, nil)
	s := newTestSynthesizer(t, TTSConfig{SampleRate: rate}, server)

	got, err := synthesizeFrames(t, s, "hi")
	if err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	var joined []byte
	for i, f := range got {
		if f.SampleRate != rate {
			t.Errorf("frame %d rate = %d, want %d", i, f.SampleRate, rate)
		}
		if i < len(got)-1 && len(f.Audio) != chunk {
			t.Errorf("frame %d holds %d bytes, want %d", i, len(f.Audio), chunk)
		}
		joined = append(joined, f.Audio...)
	}
	if len(got) != 3 {
		t.Errorf("frames = %d, want two full chunks and the rest", len(got))
	}
	if !bytes.Equal(joined, pcm) {
		t.Error("the audio passed on is not the audio Google sent")
	}
}

// TestSynthesisErrorIsReturned checks a call Google fails reports its error.
func TestSynthesisErrorIsReturned(t *testing.T) {
	_, server := newFakeTextToSpeech(t, nil, status.Error(codes.InvalidArgument, "voice not found"))
	s := newTestSynthesizer(t, TTSConfig{}, server)

	_, err := synthesizeFrames(t, s, "hi")
	if status.Code(err) != codes.InvalidArgument {
		t.Errorf("RunTTS = %v, want Google's error", err)
	}
}

// TestTTSSpeakingRateOutsideTheRangeIsRefused checks an update giving a
// speaking rate Google would reject leaves the rate as it was.
func TestTTSSpeakingRateOutsideTheRangeIsRefused(t *testing.T) {
	store := newTTSSettings(TTSConfig{})

	changed, err := settings.Apply(store, &TTSSettings{SpeakingRate: settings.Set(0.1)})
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if changed.Has("speaking_rate") {
		t.Error("an out-of-range speaking rate was reported as changed")
	}
	if _, ok := store.SpeakingRate.Value(); ok {
		t.Error("an out-of-range speaking rate was stored")
	}

	changed, err = settings.Apply(store, &TTSSettings{SpeakingRate: settings.Set(2.0)})
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if rate, _ := store.SpeakingRate.Value(); !changed.Has("speaking_rate") || rate != 2.0 {
		t.Errorf("speaking rate = %v, want an in-range rate stored", rate)
	}
}

// TestTTSLanguageIsNamedTheGoogleWay checks the language is named the way
// Cloud Text-to-Speech names it, from the configuration and from an update.
func TestTTSLanguageIsNamedTheGoogleWay(t *testing.T) {
	s := &ttsSynthesizer{live: newTTSSettings(TTSConfig{Language: language.Portuguese})}
	if got, _ := s.live.Language.Value(); got != "pt-BR" {
		t.Errorf("language = %q, want pt-BR", got)
	}
	if got := s.ServiceLanguage(language.Chinese); got != "cmn-CN" {
		t.Errorf("ServiceLanguage(zh) = %q, want cmn-CN", got)
	}
}
