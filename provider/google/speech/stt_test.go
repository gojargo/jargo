package speech

import (
	"bytes"
	"context"
	"errors"
	"io"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"cloud.google.com/go/speech/apiv2/speechpb"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
	"golang.org/x/oauth2"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// testCredentials authorize nothing; the fake server does not ask.
func testCredentials() Credentials {
	return Credentials{TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "t"})}
}

// responses is the response side of a call that answers with a fixed list.
type responses struct {
	list []*speechpb.StreamingRecognizeResponse
	end  error
}

func (r *responses) Recv() (*speechpb.StreamingRecognizeResponse, error) {
	if len(r.list) == 0 {
		if r.end != nil {
			return nil, r.end
		}
		return nil, io.EOF
	}
	next := r.list[0]
	r.list = r.list[1:]
	return next, nil
}

// response is a response carrying one result.
func response(transcript string, final bool) *speechpb.StreamingRecognizeResponse {
	return &speechpb.StreamingRecognizeResponse{Results: []*speechpb.StreamingRecognitionResult{{
		Alternatives: []*speechpb.SpeechRecognitionAlternative{{Transcript: transcript}},
		IsFinal:      final,
	}}}
}

// bareStream is a session whose current call opened elapsed ago, with nothing
// streaming: its responses are handed to it directly.
func bareStream(t *testing.T, elapsed time.Duration) *sttStream {
	t.Helper()
	ctx, cancel := context.WithCancel(t.Context())
	t.Cleanup(cancel)
	s := &sttStream{
		connector: &sttConnector{},
		language:  "en-US",
		limit:     StreamingLimit,
		ctx:       ctx,
		cancel:    cancel,
		queue:     newRequestQueue(),
		results:   make(chan []stt.Result, 8),
		done:      make(chan struct{}),
	}
	s.setStreamStartTime(time.Now().Add(-elapsed))
	return s
}

// drained is every batch of results the session handed on.
func drained(s *sttStream) []stt.Result {
	var out []stt.Result
	for {
		select {
		case r := <-s.results:
			out = append(out, r...)
		default:
			return out
		}
	}
}

// Ported from upstream (test_google_final_result_emits_finalized_transcription_frame).
// An interim goes out as an interim, and a final as a transcript that ends the
// turn, labeled with the primary language.
func TestGoogleFinalResultEmitsFinalizedTranscriptionFrame(t *testing.T) {
	s := bareStream(t, 0)

	err := s.processResponses(&responses{list: []*speechpb.StreamingRecognizeResponse{
		response("hel", false),
		response("hello", true),
	}})
	if err != nil {
		t.Fatalf("processResponses: %v", err)
	}

	got := drained(s)
	want := []stt.Result{
		{Text: "hel", Language: "en-US"},
		{Text: "hello", Final: true, EndOfTurn: true, Language: "en-US"},
	}
	if !slices.Equal(got, want) {
		t.Errorf("results = %+v, want %+v", got, want)
	}
}

// Ported from upstream (test_google_results_past_the_streaming_limit_reach_the_pipeline).
// Everything Google delivers is handed on, including past the streaming limit:
// the final is produced by the call half-closing at the limit, so it arrives
// after the limit and behind any interim still in flight.
func TestGoogleResultsPastTheStreamingLimitReachThePipeline(t *testing.T) {
	s := bareStream(t, StreamingLimit+time.Millisecond)

	err := s.processResponses(&responses{list: []*speechpb.StreamingRecognizeResponse{
		response("hel", false),
		response("hello there", true),
	}})
	if err != nil {
		t.Fatalf("processResponses: %v", err)
	}

	got := drained(s)
	if len(got) != 2 || got[0].Text != "hel" || got[1].Text != "hello there" {
		t.Fatalf("results = %+v, want the interim and then the final", got)
	}
	if !got[1].EndOfTurn {
		t.Error("the final does not end the turn")
	}
}

// TestEmptyResultsAreSkipped checks a result with no alternative, or an empty
// transcript, is not handed on.
func TestEmptyResultsAreSkipped(t *testing.T) {
	s := bareStream(t, 0)

	err := s.processResponses(&responses{list: []*speechpb.StreamingRecognizeResponse{
		{},
		{Results: []*speechpb.StreamingRecognitionResult{{IsFinal: true}}},
		response("", true),
	}})
	if err != nil {
		t.Fatalf("processResponses: %v", err)
	}
	if got := drained(s); len(got) != 0 {
		t.Errorf("results = %+v, want none", got)
	}
}

// TestAbortedCallIsReturned checks a call Google aborts for inactivity ends the
// read with the abort, so the session replaces the call.
func TestAbortedCallIsReturned(t *testing.T) {
	s := bareStream(t, 0)
	aborted := status.Error(codes.Aborted, "stream timed out")

	err := s.processResponses(&responses{end: aborted})

	if status.Code(err) != codes.Aborted {
		t.Errorf("processResponses = %v, want the abort", err)
	}
}

// Ported from upstream (test_normalize_speech_adaptation_accepts_native_message).
func TestNormalizeSpeechAdaptationAcceptsNativeMessage(t *testing.T) {
	adaptation := &speechpb.SpeechAdaptation{}

	normalized, err := normalizeSpeechAdaptation(adaptation)
	if err != nil {
		t.Fatalf("normalizeSpeechAdaptation: %v", err)
	}
	if normalized != adaptation {
		t.Error("a message was not returned as it was given")
	}
}

// Ported from upstream (test_normalize_speech_adaptation_accepts_single_phrase_set_string).
func TestNormalizeSpeechAdaptationAcceptsSinglePhraseSetString(t *testing.T) {
	phraseSet := "projects/test/locations/global/phraseSets/support-terms"

	normalized, err := normalizeSpeechAdaptation(map[string]any{"phrase_sets": phraseSet})
	if err != nil {
		t.Fatalf("normalizeSpeechAdaptation: %v", err)
	}
	if len(normalized.GetPhraseSets()) != 1 {
		t.Fatalf("phrase sets = %v, want one", normalized.GetPhraseSets())
	}
	if got := normalized.GetPhraseSets()[0].GetPhraseSet(); got != phraseSet {
		t.Errorf("phrase_set = %q, want %q", got, phraseSet)
	}
}

// Ported from upstream (test_normalize_speech_adaptation_accepts_single_inline_phrase_set).
func TestNormalizeSpeechAdaptationAcceptsSingleInlinePhraseSet(t *testing.T) {
	normalized, err := normalizeSpeechAdaptation(map[string]any{
		"phrase_sets": map[string]any{"phrases": []any{map[string]any{"value": "jargo"}}},
	})
	if err != nil {
		t.Fatalf("normalizeSpeechAdaptation: %v", err)
	}
	if len(normalized.GetPhraseSets()) != 1 {
		t.Fatalf("phrase sets = %v, want one", normalized.GetPhraseSets())
	}
	if got := normalized.GetPhraseSets()[0].GetInlinePhraseSet().GetPhrases()[0].GetValue(); got != "jargo" {
		t.Errorf("phrase value = %q, want jargo", got)
	}
}

// Ported from upstream (test_normalize_speech_adaptation_converts_string_and_inline_phrase_sets).
func TestNormalizeSpeechAdaptationConvertsStringAndInlinePhraseSets(t *testing.T) {
	normalized, err := normalizeSpeechAdaptation(map[string]any{
		"phrase_sets": []any{
			"projects/test/locations/global/phraseSets/catalog",
			map[string]any{
				"phrases": []any{
					map[string]any{"value": "jargo", "boost": 15.0},
					map[string]any{"value": "voice pipeline"},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("normalizeSpeechAdaptation: %v", err)
	}

	sets := normalized.GetPhraseSets()
	if got := sets[0].GetPhraseSet(); got != "projects/test/locations/global/phraseSets/catalog" {
		t.Errorf("phrase_set = %q", got)
	}
	phrases := sets[1].GetInlinePhraseSet().GetPhrases()
	if phrases[0].GetValue() != "jargo" || phrases[0].GetBoost() != 15.0 {
		t.Errorf("first phrase = %v, want jargo boosted 15", phrases[0])
	}
	if phrases[1].GetValue() != "voice pipeline" {
		t.Errorf("second phrase = %v, want voice pipeline", phrases[1])
	}
}

// Ported from upstream (test_normalize_speech_adaptation_rejects_invalid_phrase_set_entries).
func TestNormalizeSpeechAdaptationRejectsInvalidPhraseSetEntries(t *testing.T) {
	_, err := normalizeSpeechAdaptation(map[string]any{"phrase_sets": []any{123}})
	if !errors.Is(err, errInvalidPhraseSet) {
		t.Fatalf("normalizeSpeechAdaptation = %v, want errInvalidPhraseSet", err)
	}
	if !strings.Contains(err.Error(), "invalid Google SpeechAdaptation phrase_set entry") {
		t.Errorf("error = %q", err)
	}
}

// invalidAdaptation is a phrase set whose phrases are strings rather than
// objects, which Google's API cannot represent.
func invalidAdaptation() map[string]any {
	return map[string]any{"phrase_sets": []any{map[string]any{"phrases": []any{"hello"}}}}
}

// Ported from upstream (test_google_stt_rejects_invalid_adaptation_during_initialization).
// The service is built without failing; the configuration is refused when it
// is validated, which the service does when the pipeline starts.
func TestGoogleSTTRejectsInvalidAdaptationDuringInitialization(t *testing.T) {
	cfg := STTConfig{Credentials: testCredentials(), ProjectID: "p", Adaptation: invalidAdaptation()}

	if err := cfg.Validate(); err == nil {
		t.Fatal("Validate accepted an adaptation Google's API cannot represent")
	}
}

// Ported from upstream (test_google_stt_rejects_invalid_runtime_adaptation_before_commit).
// An update carrying an adaptation Google's API cannot represent is refused
// before anything is stored.
func TestGoogleSTTRejectsInvalidRuntimeAdaptationBeforeCommit(t *testing.T) {
	r := runGoogleSTT(t, STTConfig{})
	defer r.stop(t)

	r.task.QueueFrame(frames.NewSTTUpdateSettingsFrame(&STTSettings{
		Adaptation: settings.Set[any](invalidAdaptation()),
	}))

	waitFor(t, "the refused update was not reported", func() bool { return len(r.reportedErrors()) > 0 })
	live, _ := r.svc.CurrentSettings().(*STTSettings)
	if adaptation, ok := live.Adaptation.Value(); ok {
		t.Errorf("adaptation = %v, want none stored", adaptation)
	}
}

// connectedRecognitionConfig builds the configuration a call opens with for a
// service built with adaptation, model and denoiser.
func connectedRecognitionConfig(t *testing.T, adaptation any, model string, denoiser any) *speechpb.RecognitionConfig {
	t.Helper()
	if model == "" {
		model = defaultSTTModel
	}
	cfg := STTConfig{Model: model, Adaptation: adaptation, DenoiserConfig: denoiser}
	c := &sttConnector{cfg: cfg, live: newSTTSettings(cfg)}
	config, err := c.streamingConfig(16000)
	if err != nil {
		t.Fatalf("streamingConfig: %v", err)
	}
	return config.GetConfig()
}

// Ported from upstream (test_google_connect_sends_adaptation_for_supporting_models).
func TestGoogleConnectSendsAdaptationForSupportingModels(t *testing.T) {
	phraseSet := "projects/test/locations/global/phraseSets/catalog"
	for _, model := range []string{"latest_long", "short", "chirp_2", "telephony_short"} {
		config := connectedRecognitionConfig(t, map[string]any{"phrase_sets": []any{phraseSet}}, model, nil)

		if config.GetModel() != model {
			t.Errorf("%s: model = %q", model, config.GetModel())
		}
		if got := config.GetAdaptation().GetPhraseSets(); len(got) != 1 || got[0].GetPhraseSet() != phraseSet {
			t.Errorf("%s: phrase sets = %v, want %q", model, got, phraseSet)
		}
	}
}

// Ported from upstream (test_google_connect_omits_adaptation_for_the_telephony_model).
func TestGoogleConnectOmitsAdaptationForTheTelephonyModel(t *testing.T) {
	phraseSet := "projects/test/locations/global/phraseSets/catalog"
	for _, model := range []string{"telephony", "TELEPHONY"} {
		config := connectedRecognitionConfig(t, map[string]any{"phrase_sets": []any{phraseSet}}, model, nil)

		if config.GetAdaptation() != nil {
			t.Errorf("%s: adaptation = %v, want none", model, config.GetAdaptation())
		}
	}
}

// Ported from upstream (test_google_connect_sends_denoiser_config).
func TestGoogleConnectSendsDenoiserConfig(t *testing.T) {
	config := connectedRecognitionConfig(t, nil, "chirp_3", map[string]any{"denoise_audio": true})

	if !config.GetDenoiserConfig().GetDenoiseAudio() {
		t.Errorf("denoiser config = %v, want denoise_audio set", config.GetDenoiserConfig())
	}
}

// Ported from upstream (test_google_connect_leaves_denoiser_config_unset_when_not_configured).
func TestGoogleConnectLeavesDenoiserConfigUnsetWhenNotConfigured(t *testing.T) {
	if config := connectedRecognitionConfig(t, nil, "", nil); config.GetDenoiserConfig() != nil {
		t.Errorf("denoiser config = %v, want none", config.GetDenoiserConfig())
	}
}

// Ported from upstream (test_google_connect_leaves_adaptation_unset_when_not_configured).
func TestGoogleConnectLeavesAdaptationUnsetWhenNotConfigured(t *testing.T) {
	if config := connectedRecognitionConfig(t, nil, "", nil); config.GetAdaptation() != nil {
		t.Errorf("adaptation = %v, want none", config.GetAdaptation())
	}
}

// TestRecognitionConfigDefaults checks the configuration a call opens with by
// default: LINEAR16 mono at the session's rate, US English, latest_long, with
// automatic punctuation and interim results on.
func TestRecognitionConfigDefaults(t *testing.T) {
	cfg := STTConfig{Model: defaultSTTModel}
	c := &sttConnector{cfg: cfg, live: newSTTSettings(cfg)}
	config, err := c.streamingConfig(8000)
	if err != nil {
		t.Fatalf("streamingConfig: %v", err)
	}

	want := &speechpb.StreamingRecognitionConfig{
		Config: &speechpb.RecognitionConfig{
			DecodingConfig: &speechpb.RecognitionConfig_ExplicitDecodingConfig{
				ExplicitDecodingConfig: &speechpb.ExplicitDecodingConfig{
					Encoding:          speechpb.ExplicitDecodingConfig_LINEAR16,
					SampleRateHertz:   8000,
					AudioChannelCount: 1,
				},
			},
			LanguageCodes: []string{"en-US"},
			Model:         "latest_long",
			Features:      &speechpb.RecognitionFeatures{EnableAutomaticPunctuation: true},
		},
		StreamingFeatures: &speechpb.StreamingRecognitionFeatures{InterimResults: true},
	}
	if !proto.Equal(config, want) {
		t.Errorf("config = %v, want %v", config, want)
	}
}

// TestLanguageUpdateReplacesTheLanguages checks a language an update gives
// replaces the languages, which are what a call opens with, rather than being
// stored on its own.
func TestLanguageUpdateReplacesTheLanguages(t *testing.T) {
	store := newSTTSettings(STTConfig{Languages: []language.Language{language.EnglishUS, language.French}})

	changed, err := settings.Apply(store, &settings.STT{Language: settings.Set("de-DE")})
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if !changed.Has("languages") || changed.Has("language") {
		t.Errorf("changed = %v, want languages and not language", changed)
	}
	if got, _ := store.Languages.Value(); !slices.Equal(got, []language.Language{language.GermanDE}) {
		t.Errorf("languages = %v, want [de-DE]", got)
	}
	if _, ok := store.Language.Value(); ok {
		t.Error("the language was stored on its own")
	}
	c := &sttConnector{live: store}
	if got := c.languageCodes(); !slices.Equal(got, []string{"de-DE"}) {
		t.Errorf("language codes = %v, want [de-DE]", got)
	}
}

// TestLanguageCodesResolveThroughTheTable checks the languages are named the
// way Speech-to-Text V2 names them.
func TestLanguageCodesResolveThroughTheTable(t *testing.T) {
	c := &sttConnector{live: newSTTSettings(STTConfig{
		Languages: []language.Language{language.Chinese, language.Hebrew, language.Portuguese},
	})}
	if got := c.languageCodes(); !slices.Equal(got, []string{"cmn-Hans-CN", "iw-IL", "pt-PT"}) {
		t.Errorf("language codes = %v", got)
	}
}

// TestSTTConfigValidate checks credentials and a project are required, and the
// project is read from a service-account key when not given.
func TestSTTConfigValidate(t *testing.T) {
	if err := (STTConfig{ProjectID: "p"}).Validate(); !errors.Is(err, ErrNoCredentials) {
		t.Errorf("Validate() without credentials = %v, want ErrNoCredentials", err)
	}
	if err := (STTConfig{Credentials: testCredentials()}).Validate(); !errors.Is(err, errNoProject) {
		t.Errorf("Validate() with a token source and no project = %v, want errNoProject", err)
	}
	key := Credentials{JSON: []byte(`{"type":"service_account","project_id":"from-key"}`)}
	if err := (STTConfig{Credentials: key}).Validate(); err != nil {
		t.Errorf("Validate() with a key naming its project = %v, want nil", err)
	}
	if project, _ := (STTConfig{Credentials: key}).projectID(); project != "from-key" {
		t.Errorf("project = %q, want the key's", project)
	}
	noProject := STTConfig{Credentials: Credentials{JSON: []byte(`{"type":"service_account"}`)}}
	if err := noProject.Validate(); !errors.Is(err, errNoProject) {
		t.Errorf("Validate() with a key naming no project = %v, want errNoProject", err)
	}
}

// dialSTT opens a session against the fake server for a service built with cfg.
func dialSTT(t *testing.T, cfg STTConfig, limit time.Duration) (*sttStream, *fakeSpeech) {
	t.Helper()
	fake, server := newFakeSpeech(t)
	cfg.Credentials = testCredentials()
	if cfg.ProjectID == "" {
		cfg.ProjectID = "proj"
	}
	_, c := newSTT(cfg, server.clientOptions(t))
	c.svc = nil
	if limit > 0 {
		c.streamingLimit = limit
	}
	session, err := c.Connect(t.Context(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	t.Cleanup(func() { _ = session.Close() })
	s, ok := session.(*sttStream)
	if !ok {
		t.Fatalf("Connect returned a %T", session)
	}
	return s, fake
}

// recv reads the next batch of results, failing the test if none comes.
func recv(t *testing.T, s *sttStream) []stt.Result {
	t.Helper()
	type batch struct {
		results []stt.Result
		err     error
	}
	got := make(chan batch, 1)
	go func() {
		r, err := s.Recv()
		got <- batch{r, err}
	}()
	select {
	case b := <-got:
		if b.err != nil {
			t.Fatalf("Recv: %v", b.err)
		}
		return b.results
	case <-time.After(3 * time.Second):
		t.Fatal("no results arrived")
		return nil
	}
}

// TestStreamingCallOpensWithTheRecognizerAndConfig checks a call opens once
// there is audio, with the recognizer path and the configuration first and the
// audio after it, and that what Google answers comes back through Recv.
func TestStreamingCallOpensWithTheRecognizerAndConfig(t *testing.T) {
	s, fake := dialSTT(t, STTConfig{Location: "us", Languages: []language.Language{language.French}}, 0)

	chunk := bytes.Repeat([]byte{1, 2}, 160)
	if err := s.Send(chunk); err != nil {
		t.Fatalf("Send: %v", err)
	}
	call := fake.nextCall(t)

	first := call.nextRequest(t)
	if first.GetRecognizer() != "projects/proj/locations/us/recognizers/_" {
		t.Errorf("recognizer = %q", first.GetRecognizer())
	}
	config := first.GetStreamingConfig().GetConfig()
	if config.GetExplicitDecodingConfig().GetSampleRateHertz() != 16000 {
		t.Errorf("sample rate = %d, want the session's 16000", config.GetExplicitDecodingConfig().GetSampleRateHertz())
	}
	if !slices.Equal(config.GetLanguageCodes(), []string{"fr-FR"}) {
		t.Errorf("language codes = %v, want [fr-FR]", config.GetLanguageCodes())
	}
	if got := call.nextRequest(t).GetAudio(); !bytes.Equal(got, chunk) {
		t.Errorf("audio = % x, want the chunk sent", got)
	}

	call.respond <- response("bonjour", true)
	results := recv(t, s)
	want := []stt.Result{{Text: "bonjour", Final: true, EndOfTurn: true, Language: "fr-FR"}}
	if !slices.Equal(results, want) {
		t.Errorf("results = %+v, want %+v", results, want)
	}
}

// TestNoCallOpensWithoutAudio checks the session waits for audio before it
// opens a call, so an idle session is not aborted by Google for inactivity.
func TestNoCallOpensWithoutAudio(t *testing.T) {
	_, fake := dialSTT(t, STTConfig{}, 0)

	select {
	case <-fake.calls:
		t.Fatal("a call was opened with no audio to send")
	case <-time.After(100 * time.Millisecond):
	}
}

// TestStreamingLimitReplacesTheCall checks a call fed past the streaming limit
// is half-closed, its last results still come back, and the next call opens
// with the configuration and carries on with the chunk that found the limit.
func TestStreamingLimitReplacesTheCall(t *testing.T) {
	s, fake := dialSTT(t, STTConfig{}, 200*time.Millisecond)

	if err := s.Send([]byte{1, 1}); err != nil {
		t.Fatalf("Send: %v", err)
	}
	first := fake.nextCall(t)
	first.nextRequest(t) // the configuration
	if got := first.nextRequest(t).GetAudio(); !bytes.Equal(got, []byte{1, 1}) {
		t.Fatalf("audio = % x, want the first chunk", got)
	}

	time.Sleep(250 * time.Millisecond)
	if err := s.Send([]byte{2, 2}); err != nil {
		t.Fatalf("Send: %v", err)
	}
	select {
	case <-first.halfClosed:
	case <-time.After(3 * time.Second):
		t.Fatal("the call was not half-closed at the limit")
	}

	// The final for the audio in flight comes after the half-close.
	first.respond <- response("hello", true)
	if results := recv(t, s); len(results) != 1 || results[0].Text != "hello" {
		t.Fatalf("results = %+v, want the final from the closing call", results)
	}
	first.end <- nil

	second := fake.nextCall(t)
	if second.nextRequest(t).GetStreamingConfig() == nil {
		t.Error("the next call did not open with the configuration")
	}
	if got := second.nextRequest(t).GetAudio(); !bytes.Equal(got, []byte{2, 2}) {
		t.Errorf("audio = % x, want the chunk that found the limit", got)
	}
}

// TestFailedCallIsReplaced checks a call that fails is followed by another,
// carrying on with the audio.
func TestFailedCallIsReplaced(t *testing.T) {
	s, fake := dialSTT(t, STTConfig{}, 0)

	if err := s.Send([]byte{1, 1}); err != nil {
		t.Fatalf("Send: %v", err)
	}
	first := fake.nextCall(t)
	first.nextRequest(t)
	first.nextRequest(t)
	first.end <- status.Error(codes.Unavailable, "gone")

	// Sent once the failure has ended the call, inside the delay before the
	// next one; a chunk the failing call had already taken is lost with it.
	time.Sleep(200 * time.Millisecond)
	if err := s.Send([]byte{2, 2}); err != nil {
		t.Fatalf("Send: %v", err)
	}
	second := fake.nextCall(t)
	if second.nextRequest(t).GetStreamingConfig() == nil {
		t.Error("the next call did not open with the configuration")
	}
	if got := second.nextRequest(t).GetAudio(); !bytes.Equal(got, []byte{2, 2}) {
		t.Errorf("audio = % x, want the chunk sent after the failure", got)
	}
}

// TestCallGoogleEndsEndsTheSession checks a call Google ends before the limit
// ends the session with io.EOF, which the service reconnects from.
func TestCallGoogleEndsEndsTheSession(t *testing.T) {
	s, fake := dialSTT(t, STTConfig{}, 0)

	if err := s.Send([]byte{1, 1}); err != nil {
		t.Fatalf("Send: %v", err)
	}
	call := fake.nextCall(t)
	call.nextRequest(t)
	call.end <- nil

	done := make(chan error, 1)
	go func() {
		_, err := s.Recv()
		done <- err
	}()
	select {
	case err := <-done:
		if !errors.Is(err, io.EOF) {
			t.Errorf("Recv = %v, want io.EOF", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the session did not end")
	}
}

// sttRun is a pipeline holding the service, against the fake server.
type sttRun struct {
	svc  *stt.StreamService
	fake *fakeSpeech
	task *pipeline.Worker
	done chan error

	mu     sync.Mutex
	down   []frames.Frame
	errors []*frames.ErrorFrame
}

// runGoogleSTT starts a pipeline holding a service built with cfg.
func runGoogleSTT(t *testing.T, cfg STTConfig) *sttRun {
	t.Helper()
	fake, server := newFakeSpeech(t)
	cfg.Credentials = testCredentials()
	cfg.ProjectID = "proj"
	cfg.SampleRate = 16000
	svc, _ := newSTT(cfg, server.clientOptions(t))
	r := &sttRun{svc: svc, fake: fake, done: make(chan error, 1)}
	r.task = pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		IdleTimeout:             -1,
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	})
	event.On(&r.task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		switch f.(type) {
		case *frames.TranscriptionFrame, *frames.InterimTranscriptionFrame:
			r.mu.Lock()
			r.down = append(r.down, f)
			r.mu.Unlock()
		}
	})
	event.On(&r.task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		if ef, ok := f.(*frames.ErrorFrame); ok {
			r.mu.Lock()
			r.errors = append(r.errors, ef)
			r.mu.Unlock()
		}
	})
	go func() { r.done <- r.task.Run(context.Background()) }()
	waitFor(t, "the service never held its session", svc.Connected)
	return r
}

func (r *sttRun) transcripts() []frames.Frame {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.down)
}

func (r *sttRun) reportedErrors() []*frames.ErrorFrame {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.errors)
}

func (r *sttRun) stop(t *testing.T) {
	t.Helper()
	r.task.StopWhenDone()
	select {
	case <-r.done:
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not finish")
	}
}

// TestTranscriptsReachThePipeline checks the input audio is streamed to Google
// and its answers come out of the service as an interim and then a finalized
// transcript.
func TestTranscriptsReachThePipeline(t *testing.T) {
	r := runGoogleSTT(t, STTConfig{})
	defer r.stop(t)

	r.task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 640), 16000, 1))
	call := r.fake.nextCall(t)
	call.nextRequest(t)
	if len(call.nextRequest(t).GetAudio()) != 640 {
		t.Error("the input audio did not reach the call")
	}

	call.respond <- response("hel", false)
	call.respond <- response("hello", true)
	waitFor(t, "the transcripts never reached the pipeline", func() bool { return len(r.transcripts()) == 2 })

	got := r.transcripts()
	interim, ok := got[0].(*frames.InterimTranscriptionFrame)
	if !ok || interim.Text != "hel" {
		t.Errorf("first frame = %#v, want the interim", got[0])
	}
	final, ok := got[1].(*frames.TranscriptionFrame)
	if !ok || final.Text != "hello" || !final.Finalized || final.Language != "en-US" {
		t.Errorf("second frame = %#v, want the finalized transcript in en-US", got[1])
	}
}
