package deepgram

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"maps"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"reflect"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/gojargo/jargo/errclass"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
)

// segmentWAV stands in for a WAV segment where the request body is checked.
var segmentWAV = []byte("RIFF....WAVEfmt ") //nolint:gochecknoglobals // test fixture

// segmentPCM is a segment of 16-bit mono PCM, as the service hands it over.
var segmentPCM = make([]byte, 3200) //nolint:gochecknoglobals // test fixture

// listenResult is a pre-recorded response body with one channel and one
// alternative, with channel's fields added to the channel.
func listenResult(transcript string, channel map[string]any) map[string]any {
	ch := map[string]any{
		"alternatives": []any{map[string]any{"transcript": transcript, "confidence": 0.99}},
	}
	maps.Copy(ch, channel)
	return map[string]any{
		"metadata": map[string]any{"request_id": "req-1", "duration": 1.0, "channels": 1.0},
		"results":  map[string]any{"channels": []any{ch}},
	}
}

// firstAlternative is the alternative of a listenResult, for a test to add to.
func firstAlternative(body map[string]any) map[string]any {
	channels, _ := body["results"].(map[string]any)["channels"].([]any)
	alternatives, _ := channels[0].(map[string]any)["alternatives"].([]any)
	alternative, _ := alternatives[0].(map[string]any)
	return alternative
}

// listenRequest is what one request to the test server carried.
type listenRequest struct {
	header http.Header
	query  [][2]string
	body   []byte
}

// prerecordedServer serves /v1/listen, answering every request with status and
// body (listenResult("Hello there.", nil) when body is nil), and records what
// the last request carried.
type prerecordedServer struct {
	*httptest.Server
	mu   sync.Mutex
	last listenRequest
}

func newPrerecordedServer(t *testing.T, status int, body any) *prerecordedServer {
	t.Helper()
	if body == nil {
		body = listenResult("Hello there.", nil)
	}
	s := &prerecordedServer{}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != listenPath || r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		data, _ := io.ReadAll(r.Body)
		s.mu.Lock()
		s.last = listenRequest{header: r.Header.Clone(), query: orderedQuery(r.URL.RawQuery), body: data}
		s.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(body)
	}))
	t.Cleanup(s.Close)
	return s
}

func (s *prerecordedServer) request() listenRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.last
}

// orderedQuery decodes a raw query into its name and value pairs, in order.
func orderedQuery(raw string) [][2]string {
	var out [][2]string
	for part := range strings.SplitSeq(raw, "&") {
		if part == "" {
			continue
		}
		name, value, _ := strings.Cut(part, "=")
		name, _ = url.QueryUnescape(name)
		value, _ = url.QueryUnescape(value)
		out = append(out, [2]string{name, value})
	}
	return out
}

// queryValuesOf is the values sent for name, in order.
func queryValuesOf(query [][2]string, name string) []string {
	var out []string
	for _, p := range query {
		if p[0] == name {
			out = append(out, p[1])
		}
	}
	return out
}

// queryNames is the names of the parameters sent.
func queryNames(query [][2]string) []string {
	out := make([]string, 0, len(query))
	for _, p := range query {
		out = append(out, p[0])
	}
	return out
}

// segmentedConfig is a configuration pointed at baseURL.
func segmentedConfig(baseURL string, s *SegmentedSTTSettings) SegmentedSTTConfig {
	return SegmentedSTTConfig{APIKey: "test-key", BaseURL: baseURL, SampleRate: 16000, Settings: s}
}

// segmentRun is what transcribing one segment produced: the frames the run
// yielded, and the errors among them.
type segmentRun struct {
	frames []frames.Frame
	errors []*frames.ErrorFrame
}

// transcriptions is the transcription frames the run yielded.
func (r segmentRun) transcriptions() []*frames.TranscriptionFrame {
	var out []*frames.TranscriptionFrame
	for _, f := range r.frames {
		if tf, ok := f.(*frames.TranscriptionFrame); ok {
			out = append(out, tf)
		}
	}
	return out
}

// runOnce transcribes one segment through svc's run and returns what it
// yielded.
func runOnce(t *testing.T, svc *stt.SegmentService) segmentRun {
	t.Helper()
	var run segmentRun
	for f := range svc.RunSTT()(t.Context(), segmentPCM, 16000) {
		run.frames = append(run.frames, f)
		if ef, ok := f.(*frames.ErrorFrame); ok {
			run.errors = append(run.errors, ef)
		}
	}
	return run
}

// runSegmented builds a service against baseURL and transcribes one segment.
func runSegmented(t *testing.T, cfg SegmentedSTTConfig) segmentRun {
	t.Helper()
	return runOnce(t, NewSegmentedSTT(cfg))
}

// runAndReport runs svc in a pipeline over one segment the VAD cut out, and
// returns the transcripts that came out of it and the errors it reported.
//
// Errors are classified as the service pushes them, not as its run yields
// them, so a test reading the category or the service's usability has to go
// through the service the way the pipeline does.
func runAndReport(t *testing.T, svc *stt.SegmentService) ([]*frames.TranscriptionFrame, []*frames.ErrorFrame) {
	t.Helper()
	var (
		mu       sync.Mutex
		got      []*frames.TranscriptionFrame
		reported []*frames.ErrorFrame
	)
	event.On(svc.Events(), processor.EventError, func(_ context.Context, ef *frames.ErrorFrame) {
		mu.Lock()
		reported = append(reported, ef)
		mu.Unlock()
	})
	noRTVI := false
	worker := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		EnableRTVI:              &noRTVI,
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	event.On(&worker.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if tf, ok := f.(*frames.TranscriptionFrame); ok {
			mu.Lock()
			got = append(got, tf)
			mu.Unlock()
		}
	})
	done := make(chan error, 1)
	go func() { done <- worker.Run(context.Background()) }()
	worker.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0, time.Time{}))
	worker.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 3200), 16000, 1))
	worker.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0, time.Time{}))
	worker.StopWhenDone()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not stop")
	}
	mu.Lock()
	defer mu.Unlock()
	return got, reported
}

// reportedError is the one error running svc over a segment reported, failing
// the test otherwise.
func reportedError(t *testing.T, svc *stt.SegmentService) *frames.ErrorFrame {
	t.Helper()
	got, reported := runAndReport(t, svc)
	if len(reported) != 1 || len(got) != 0 {
		t.Fatalf("reported %v and transcribed %v, want one error and no transcript", reported, got)
	}
	return reported[0]
}

// currentSegmented is the settings svc holds.
func currentSegmented(t *testing.T, svc *stt.SegmentService) *SegmentedSTTSettings {
	t.Helper()
	s, ok := svc.CurrentSettings().(*SegmentedSTTSettings)
	if !ok {
		t.Fatalf("settings = %T, want *SegmentedSTTSettings", svc.CurrentSettings())
	}
	return s
}

// updateSettings sends svc a settings update, as the pipeline would.
func updateSettings(t *testing.T, svc *stt.SegmentService, delta any) {
	t.Helper()
	if err := svc.ProcessFrame(t.Context(), frames.NewSTTUpdateSettingsFrame(delta), processor.Downstream); err != nil {
		t.Fatalf("update settings: %v", err)
	}
}

//
// Settings
//

func TestSegmentedDefaultsUseNova3AndEnglish(t *testing.T) {
	t.Parallel()
	s := currentSegmented(t, NewSegmentedSTT(SegmentedSTTConfig{APIKey: "k"}))

	if got := s.Model.Or(""); got != "nova-3-general" {
		t.Errorf("model = %q, want nova-3-general", got)
	}
	if got := s.Language.Or(""); got != "en" {
		t.Errorf("language = %q, want en", got)
	}
	if !s.Punctuate.Or(false) {
		t.Error("punctuate is off, want it on")
	}
}

func TestSegmentedTTFSLatencyDefaultsToTheServiceConstant(t *testing.T) {
	t.Parallel()
	svc := NewSegmentedSTT(SegmentedSTTConfig{APIKey: "k"})

	md, ok := svc.ServiceMetadataFrame().(*frames.STTMetadataFrame)
	if !ok {
		t.Fatalf("metadata = %T, want *frames.STTMetadataFrame", svc.ServiceMetadataFrame())
	}
	if md.TTFSP99Latency != stt.DefaultTTFSP99 {
		t.Errorf("TTFS P99 = %v, want %v", md.TTFSP99Latency, stt.DefaultTTFSP99)
	}
}

func TestSegmentedName(t *testing.T) {
	t.Parallel()
	// processor.New appends a per-instance counter.
	if name := NewSegmentedSTT(SegmentedSTTConfig{APIKey: "k"}).Name(); !strings.HasPrefix(name, "DeepgramSegmentedSTT") {
		t.Errorf("service name = %q, want the DeepgramSegmentedSTT label", name)
	}
}

func TestSegmentedConfigRequiresAnAPIKey(t *testing.T) {
	t.Parallel()
	if err := (SegmentedSTTConfig{}).Validate(); err == nil {
		t.Error("a configuration without an API key validated")
	}
	if err := (SegmentedSTTConfig{APIKey: "k"}).Validate(); err != nil {
		t.Errorf("validate: %v", err)
	}
}

//
// Request construction
//

func TestSegmentedTranscribePostsTheWAVSegmentWithTheKey(t *testing.T) {
	t.Parallel()
	srv := newPrerecordedServer(t, http.StatusOK, nil)
	tr := newSegmentedTranscriber(segmentedConfig(srv.URL, nil))

	body, err := tr.transcribe(t.Context(), segmentWAV)
	if err != nil {
		t.Fatalf("transcribe: %v", err)
	}

	var result map[string]any
	if err := json.Unmarshal(body, &result); err != nil {
		t.Fatalf("decoding the result: %v", err)
	}
	if !reflect.DeepEqual(result, listenResult("Hello there.", nil)) {
		t.Errorf("result = %v, want the server's", result)
	}
	got := srv.request()
	if a := got.header.Get("Authorization"); a != "Token test-key" {
		t.Errorf("Authorization = %q, want Token test-key", a)
	}
	if ct := got.header.Get("Content-Type"); ct != "audio/wav" {
		t.Errorf("Content-Type = %q, want audio/wav", ct)
	}
	if string(got.body) != string(segmentWAV) {
		t.Errorf("body = %q, want the segment", got.body)
	}
	want := [][2]string{
		{"model", "nova-3-general"},
		{"language", "en"},
		{"detect_entities", "false"},
		{"dictation", "false"},
		{"numerals", "false"},
		{"punctuate", "true"},
		{"smart_format", "false"},
		{"detect_language", "false"},
		{"filler_words", "false"},
	}
	if !reflect.DeepEqual(got.query, want) {
		t.Errorf("query = %v, want %v", got.query, want)
	}
}

func TestSegmentedSettingsAreSentAsQueryParameters(t *testing.T) {
	t.Parallel()
	srv := newPrerecordedServer(t, http.StatusOK, nil)
	runSegmented(t, segmentedConfig(srv.URL, &SegmentedSTTSettings{
		Model:       settings.Set("nova-3"),
		Language:    settings.Set(string(language.Spanish)),
		Punctuate:   settings.Set(false),
		SmartFormat: settings.Set(true),
		Keyterm:     settings.Set([]string{"jargo", "Daily"}),
		Version:     settings.Set("2025-07-31.0"),
	}))

	query := srv.request().query
	for _, p := range [][2]string{
		{"model", "nova-3"},
		{"language", "es"},
		{"punctuate", "false"},
		{"smart_format", "true"},
		{"version", "2025-07-31.0"},
	} {
		if !slices.Contains(query, p) {
			t.Errorf("query %v lacks %v", query, p)
		}
	}
	// A list is sent as a repeated parameter.
	if got := queryValuesOf(query, "keyterm"); !slices.Equal(got, []string{"jargo", "Daily"}) {
		t.Errorf("keyterm = %v, want [jargo Daily]", got)
	}
}

func TestSegmentedSettingsSetToNoneAreLeftOut(t *testing.T) {
	t.Parallel()
	srv := newPrerecordedServer(t, http.StatusOK, nil)
	s := &SegmentedSTTSettings{}
	s.Language = settings.Cleared[string]()
	s.Keyterm = settings.Cleared[[]string]()
	s.Version = settings.Cleared[string]()
	runSegmented(t, segmentedConfig(srv.URL, s))

	for _, name := range queryNames(srv.request().query) {
		if name == "language" || name == "keyterm" || name == "version" {
			t.Errorf("%s was sent", name)
		}
	}
}

func TestSegmentedSpeakerFormatIsNotSent(t *testing.T) {
	t.Parallel()
	srv := newPrerecordedServer(t, http.StatusOK, nil)
	s := &SegmentedSTTSettings{}
	s.SpeakerFormat = settings.Set("{speaker}: {text}")
	runSegmented(t, segmentedConfig(srv.URL, s))

	if slices.Contains(queryNames(srv.request().query), "speaker_format") {
		t.Error("speaker_format was sent")
	}
}

func TestSegmentedExtraAndInitOptionsAreSent(t *testing.T) {
	t.Parallel()
	srv := newPrerecordedServer(t, http.StatusOK, nil)
	optOut := true
	s := &SegmentedSTTSettings{}
	s.Extra = map[string]any{"utt_split": 0.5, "paragraphs": true}
	cfg := segmentedConfig(srv.URL, s)
	cfg.Tag = []string{"support", "eu"}
	cfg.MipOptOut = &optOut
	runSegmented(t, cfg)

	query := srv.request().query
	if got := queryValuesOf(query, "tag"); !slices.Equal(got, []string{"support", "eu"}) {
		t.Errorf("tag = %v, want [support eu]", got)
	}
	for _, p := range [][2]string{{"mip_opt_out", "true"}, {"utt_split", "0.5"}, {"paragraphs", "true"}} {
		if !slices.Contains(query, p) {
			t.Errorf("query %v lacks %v", query, p)
		}
	}
}

func TestSegmentedAnExtraKeyNamingASettingSetsIt(t *testing.T) {
	t.Parallel()
	srv := newPrerecordedServer(t, http.StatusOK, nil)
	s := &SegmentedSTTSettings{}
	s.Extra = map[string]any{"smart_format": true}
	svc := NewSegmentedSTT(segmentedConfig(srv.URL, s))
	runOnce(t, svc)

	store := currentSegmented(t, svc)
	if !store.SmartFormat.Or(false) {
		t.Error("smart_format is off, want the extra key to have set it")
	}
	if _, ok := store.Extra["smart_format"]; ok {
		t.Error("smart_format is still in extra")
	}
	if got := queryValuesOf(srv.request().query, "smart_format"); !slices.Equal(got, []string{"true"}) {
		t.Errorf("smart_format = %v, want [true]", got)
	}
	// The caller's settings are left as they were given.
	if _, ok := s.Extra["smart_format"]; !ok {
		t.Error("the caller's extra was changed")
	}
}

func TestSegmentedASettingsUpdateAppliesToTheNextRequest(t *testing.T) {
	t.Parallel()
	srv := newPrerecordedServer(t, http.StatusOK, nil)
	svc := NewSegmentedSTT(segmentedConfig(srv.URL, nil))

	delta := &SegmentedSTTSettings{}
	delta.SmartFormat = settings.Set(true)
	updateSettings(t, svc, delta)
	runOnce(t, svc)

	if !slices.Contains(srv.request().query, [2]string{"smart_format", "true"}) {
		t.Errorf("query = %v, want smart_format=true", srv.request().query)
	}
}

func TestSegmentedAnExtraKeyNamingASettingUpdatesItAtRuntime(t *testing.T) {
	t.Parallel()
	srv := newPrerecordedServer(t, http.StatusOK, nil)
	svc := NewSegmentedSTT(segmentedConfig(srv.URL, nil))

	delta := &SegmentedSTTSettings{}
	delta.Extra = map[string]any{"smart_format": true}
	updateSettings(t, svc, delta)
	runOnce(t, svc)

	store := currentSegmented(t, svc)
	if !store.SmartFormat.Or(false) {
		t.Error("smart_format is off, want the extra key to have set it")
	}
	if _, ok := store.Extra["smart_format"]; ok {
		t.Error("smart_format is still in extra")
	}
	if got := queryValuesOf(srv.request().query, "smart_format"); !slices.Equal(got, []string{"true"}) {
		t.Errorf("smart_format = %v, want [true]", got)
	}
}

//
// Transcription
//

func TestSegmentedRunYieldsATranscriptionFrame(t *testing.T) {
	t.Parallel()
	srv := newPrerecordedServer(t, http.StatusOK, nil)
	run := runSegmented(t, segmentedConfig(srv.URL, nil))

	if len(run.frames) != 1 {
		t.Fatalf("frames = %v, want one transcription", run.frames)
	}
	tf, ok := run.frames[0].(*frames.TranscriptionFrame)
	if !ok {
		t.Fatalf("frame = %T, want a transcription", run.frames[0])
	}
	if tf.Text != "Hello there." {
		t.Errorf("text = %q, want Hello there.", tf.Text)
	}
	if tf.Language != "en" {
		t.Errorf("language = %q, want en", tf.Language)
	}
	if !reflect.DeepEqual(tf.Result, listenResult("Hello there.", nil)) {
		t.Errorf("result = %v, want the server's", tf.Result)
	}
}

func TestSegmentedRunYieldsNothingForAnEmptyTranscript(t *testing.T) {
	t.Parallel()
	srv := newPrerecordedServer(t, http.StatusOK, listenResult("", nil))
	if run := runSegmented(t, segmentedConfig(srv.URL, nil)); len(run.frames) != 0 || len(run.errors) != 0 {
		t.Errorf("run = %+v, want nothing", run)
	}
}

func TestSegmentedRunYieldsNothingForAResultWithoutChannels(t *testing.T) {
	t.Parallel()
	// Deepgram's answer to a WAV with no samples.
	body := map[string]any{
		"metadata": map[string]any{"duration": 0.0, "channels": 0},
		"results":  map[string]any{"channels": []any{}},
	}
	srv := newPrerecordedServer(t, http.StatusOK, body)
	if run := runSegmented(t, segmentedConfig(srv.URL, nil)); len(run.frames) != 0 || len(run.errors) != 0 {
		t.Errorf("run = %+v, want nothing", run)
	}
}

func TestSegmentedADetectedLanguageIsReported(t *testing.T) {
	t.Parallel()
	srv := newPrerecordedServer(t, http.StatusOK, listenResult("Hola.", map[string]any{"detected_language": "es"}))
	run := runSegmented(t, segmentedConfig(srv.URL, nil))

	if got := run.transcriptions(); len(got) != 1 || got[0].Language != string(language.Spanish) {
		t.Errorf("transcriptions = %v, want one in es", got)
	}
}

func TestSegmentedAMultilingualResultReportsItsFirstLanguage(t *testing.T) {
	t.Parallel()
	body := listenResult("Hola.", nil)
	firstAlternative(body)["languages"] = []any{"es", "en"}
	srv := newPrerecordedServer(t, http.StatusOK, body)
	run := runSegmented(t, segmentedConfig(srv.URL, nil))

	if got := run.transcriptions(); len(got) != 1 || got[0].Language != string(language.Spanish) {
		t.Errorf("transcriptions = %v, want one in es", got)
	}
}

func TestSegmentedAnUnknownDetectedLanguageIsReportedAsIs(t *testing.T) {
	t.Parallel()
	srv := newPrerecordedServer(t, http.StatusOK, listenResult("Hello.", map[string]any{"detected_language": "xx"}))
	run := runSegmented(t, segmentedConfig(srv.URL, nil))

	if got := run.transcriptions(); len(got) != 1 || got[0].Language != "xx" {
		t.Errorf("transcriptions = %v, want one in xx", got)
	}
}

//
// Diarization
//

// diarizedResult is a result whose alternative carries words.
func diarizedResult(transcript string, words []any) map[string]any {
	body := listenResult(transcript, nil)
	firstAlternative(body)["words"] = words
	return body
}

// twoSpeakers is a result with two speakers, the first saying two words.
func twoSpeakers() map[string]any {
	return diarizedResult("Hi there. Hello.", []any{
		map[string]any{"word": "hi", "punctuated_word": "Hi", "speaker": 0},
		map[string]any{"word": "there", "punctuated_word": "there.", "speaker": 0},
		map[string]any{"word": "hello", "punctuated_word": "Hello.", "speaker": 1},
	})
}

// diarized is settings that turn diarization on.
func diarized() *SegmentedSTTSettings {
	s := &SegmentedSTTSettings{}
	s.Diarize = settings.Set(true)
	return s
}

// speakerTexts is each transcription's speaker label ("" for none) and text.
func speakerTexts(tfs []*frames.TranscriptionFrame) [][2]string {
	out := make([][2]string, 0, len(tfs))
	for _, tf := range tfs {
		label := ""
		if tf.Speaker != nil {
			label = tf.Speaker.Label
		}
		out = append(out, [2]string{label, tf.Text})
	}
	return out
}

func TestSegmentedDiarizationIsRequestedByModelAlone(t *testing.T) {
	t.Parallel()
	srv := newPrerecordedServer(t, http.StatusOK, nil)
	runSegmented(t, segmentedConfig(srv.URL, diarized()))

	query := srv.request().query
	// Deepgram refuses diarize alongside diarize_model.
	if !slices.Contains(query, [2]string{"diarize_model", "latest"}) {
		t.Errorf("query = %v, want diarize_model=latest", query)
	}
	if slices.Contains(queryNames(query), "diarize") {
		t.Error("diarize was sent")
	}
}

func TestSegmentedNoDiarizationParameterIsSentWhenDiarizationIsOff(t *testing.T) {
	t.Parallel()
	srv := newPrerecordedServer(t, http.StatusOK, nil)
	s := &SegmentedSTTSettings{}
	s.DiarizeModel = settings.Set("v2")
	runSegmented(t, segmentedConfig(srv.URL, s))

	for _, name := range queryNames(srv.request().query) {
		if name == "diarize" || name == "diarize_model" {
			t.Errorf("%s was sent", name)
		}
	}
}

func TestSegmentedAPinnedDiarizeModelIsSent(t *testing.T) {
	t.Parallel()
	srv := newPrerecordedServer(t, http.StatusOK, nil)
	s := diarized()
	s.DiarizeModel = settings.Set("v2")
	runSegmented(t, segmentedConfig(srv.URL, s))

	if got := queryValuesOf(srv.request().query, "diarize_model"); !slices.Equal(got, []string{"v2"}) {
		t.Errorf("diarize_model = %v, want [v2]", got)
	}
}

func TestSegmentedADiarizeModelInExtraIsTheOneSent(t *testing.T) {
	t.Parallel()
	srv := newPrerecordedServer(t, http.StatusOK, nil)
	s := diarized()
	s.Extra = map[string]any{"diarize_model": "v1"}
	runSegmented(t, segmentedConfig(srv.URL, s))

	if got := queryValuesOf(srv.request().query, "diarize_model"); !slices.Equal(got, []string{"v1"}) {
		t.Errorf("diarize_model = %v, want [v1]", got)
	}
}

func TestSegmentedADiarizedResultYieldsOneFramePerSpeakerRun(t *testing.T) {
	t.Parallel()
	srv := newPrerecordedServer(t, http.StatusOK, twoSpeakers())
	s := diarized()
	s.SpeakerFormat = settings.Set("{speaker}: {text}")
	tfs := runSegmented(t, segmentedConfig(srv.URL, s)).transcriptions()

	if len(tfs) != 2 {
		t.Fatalf("transcriptions = %v, want two", tfs)
	}
	want := []struct {
		speaker frames.TranscriptionSpeaker
		text    string
	}{
		{frames.TranscriptionSpeaker{Label: "0", Number: 1}, "0: Hi there."},
		{frames.TranscriptionSpeaker{Label: "1", Number: 2}, "1: Hello."},
	}
	for i, w := range want {
		if tfs[i].Speaker == nil || *tfs[i].Speaker != w.speaker || tfs[i].Text != w.text {
			t.Errorf("transcription %d = %v, %q, want %v, %q", i, tfs[i].Speaker, tfs[i].Text, w.speaker, w.text)
		}
		if tfs[i].Language != "en" {
			t.Errorf("transcription %d language = %q, want en", i, tfs[i].Language)
		}
	}
}

func TestSegmentedUnpunctuatedWordsAreSplitByTheirPlainText(t *testing.T) {
	t.Parallel()
	body := diarizedResult("hi there hello", []any{
		map[string]any{"word": "hi", "speaker": 0},
		map[string]any{"word": "there", "speaker": 0},
		map[string]any{"word": "hello", "speaker": 1},
	})
	srv := newPrerecordedServer(t, http.StatusOK, body)
	s := diarized()
	s.Punctuate = settings.Set(false)
	tfs := runSegmented(t, segmentedConfig(srv.URL, s)).transcriptions()

	if got, want := speakerTexts(tfs), [][2]string{{"0", "hi there"}, {"1", "hello"}}; !reflect.DeepEqual(got, want) {
		t.Errorf("runs = %v, want %v", got, want)
	}
}

func TestSegmentedWordsWithoutSpeakersYieldOneUnlabelledFrame(t *testing.T) {
	t.Parallel()
	body := diarizedResult("Hi there.", []any{map[string]any{"word": "hi"}, map[string]any{"word": "there"}})
	srv := newPrerecordedServer(t, http.StatusOK, body)
	tfs := runSegmented(t, segmentedConfig(srv.URL, diarized())).transcriptions()

	if len(tfs) != 1 || tfs[0].Speaker != nil || tfs[0].Text != "Hi there." {
		t.Errorf("runs = %v, want one unlabelled Hi there.", speakerTexts(tfs))
	}
}

func TestSegmentedSpeakersAreIgnoredWhenDiarizationIsOff(t *testing.T) {
	t.Parallel()
	srv := newPrerecordedServer(t, http.StatusOK, twoSpeakers())
	tfs := runSegmented(t, segmentedConfig(srv.URL, nil)).transcriptions()

	if len(tfs) != 1 || tfs[0].Speaker != nil || tfs[0].Text != "Hi there. Hello." {
		t.Errorf("runs = %v, want one unlabelled Hi there. Hello.", speakerTexts(tfs))
	}
}

func TestSegmentedOnlyTheLastSpeakerRunOfASegmentIsFinalized(t *testing.T) {
	t.Parallel()
	srv := newPrerecordedServer(t, http.StatusOK, twoSpeakers())
	svc := NewSegmentedSTT(segmentedConfig(srv.URL, diarized()))

	got, _ := runAndReport(t, svc)

	type finalized struct {
		text      string
		finalized bool
	}
	var runs []finalized
	for _, tf := range got {
		runs = append(runs, finalized{tf.Text, tf.Finalized})
	}
	// A finalized transcript ends the user's turn, so it waits for every run.
	if want := []finalized{{"Hi there.", false}, {"Hello.", true}}; !reflect.DeepEqual(runs, want) {
		t.Errorf("transcriptions = %v, want %v", runs, want)
	}
}

//
// Errors
//

// onlyError is the one error a run reported, failing the test otherwise.
func onlyError(t *testing.T, run segmentRun) *frames.ErrorFrame {
	t.Helper()
	if len(run.errors) != 1 || len(run.transcriptions()) != 0 {
		t.Fatalf("run = %+v, want one error and no transcript", run)
	}
	return run.errors[0]
}

func TestSegmentedRunNamesTheErrorCodeMessageAndRequest(t *testing.T) {
	t.Parallel()
	body := map[string]any{
		"err_code":   "Bad Request",
		"err_msg":    "No such model/language/tier combination found.",
		"request_id": "req-42",
	}
	srv := newPrerecordedServer(t, http.StatusBadRequest, body)
	ef := onlyError(t, runSegmented(t, segmentedConfig(srv.URL, nil)))

	msg := ef.Error
	if !strings.Contains(msg, "400 Bad Request - No such model/language/tier combination found.") {
		t.Errorf("error = %q, want the status, code and message", msg)
	}
	if !strings.Contains(msg, "req-42") {
		t.Errorf("error = %q, want the request id", msg)
	}
}

func TestSegmentedAnErrorDoesNotCarryTheRequestSettings(t *testing.T) {
	t.Parallel()
	body := map[string]any{"err_code": "Bad Request", "err_msg": "Invalid keyterm.", "request_id": "r"}
	srv := newPrerecordedServer(t, http.StatusBadRequest, body)
	s := &SegmentedSTTSettings{}
	s.Keyterm = settings.Set([]string{"Jane Doe"})
	s.Replace = settings.Set([]string{"foo:bar"})
	ef := onlyError(t, runSegmented(t, segmentedConfig(srv.URL, s)))

	// The error text reaches the client and the logs; the query it was sent
	// with stays on the error's request URL.
	for _, text := range []string{ef.Error, ef.Err.Error()} {
		if strings.Contains(text, "Jane") || strings.Contains(text, "foo:bar") {
			t.Errorf("error %q carries the request settings", text)
		}
	}
	se, ok := errors.AsType[*statusError](ef.Err)
	if !ok {
		t.Fatalf("error = %T, want it to carry the response status", ef.Err)
	}
	if !strings.Contains(se.requestURL, "Jane+Doe") {
		t.Errorf("request URL = %q, want the query it was sent with", se.requestURL)
	}
}

// hangingTransport is a transport whose every connect hangs, so a request times
// out without leaving the host. A connect gives up after connectTimeout with a
// dial timeout, or never when it is zero.
func hangingTransport(connectTimeout time.Duration) *http.Transport {
	return &http.Transport{DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
		var timeout <-chan time.Time
		if connectTimeout > 0 {
			timeout = time.After(connectTimeout)
		}
		select {
		case <-timeout:
			return nil, &net.OpError{Op: "dial", Net: network, Err: os.ErrDeadlineExceeded}
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}}
}

// runTimedOut transcribes one segment with client, whose connect hangs past its
// timeout, and returns the error yielded and the name of the service that
// yielded it.
func runTimedOut(t *testing.T, client *http.Client) (*frames.ErrorFrame, string) {
	t.Helper()
	s := &SegmentedSTTSettings{}
	s.Keyterm = settings.Set([]string{"Jane Doe"})
	s.Replace = settings.Set([]string{"foo:bar"})
	// An IP address, so nothing is resolved before the connect.
	cfg := segmentedConfig("http://127.0.0.1:9", s)
	cfg.HTTPClient = client
	svc := NewSegmentedSTT(cfg)
	return onlyError(t, runOnce(t, svc)), svc.Name()
}

func TestSegmentedAConnectionTimeoutDoesNotCarryTheRequestSettings(t *testing.T) {
	t.Parallel()
	ef, service := runTimedOut(t, &http.Client{Transport: hangingTransport(300 * time.Millisecond)})

	if got, want := ef.Error, service+" transcription error: Post: dial tcp: i/o timeout"; got != want {
		t.Errorf("error = %q, want %q", got, want)
	}
	// The failure underneath is the request, URL and all.
	ue, ok := errors.AsType[*url.Error](ef.Err)
	if !ok || !ue.Timeout() {
		t.Fatalf("error = %v, want a timed-out request underneath", ef.Err)
	}
	if !strings.Contains(ue.URL, "Jane+Doe") {
		t.Errorf("request URL = %q, want the query it was sent with", ue.URL)
	}
}

func TestSegmentedATotalTimeoutIsReportedByItsKind(t *testing.T) {
	t.Parallel()
	ef, service := runTimedOut(t, &http.Client{Timeout: 300 * time.Millisecond, Transport: hangingTransport(0)})

	msg := ef.Error
	if !strings.HasPrefix(msg, service+" transcription error: Post: ") ||
		!strings.Contains(msg, "Client.Timeout exceeded") {
		t.Errorf("error = %q, want the client timeout named", msg)
	}
	if strings.Contains(msg, "Jane") || strings.Contains(msg, "127.0.0.1") {
		t.Errorf("error = %q carries the request URL", msg)
	}
}

func TestSegmentedARejectedRequestLeavesTheServiceUnusable(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		status   int
		category errclass.Category
	}{
		{http.StatusBadRequest, errclass.InvalidRequest},
		{http.StatusUnauthorized, errclass.Authentication},
		{http.StatusForbidden, errclass.Authorization},
	} {
		t.Run(http.StatusText(tc.status), func(t *testing.T) {
			t.Parallel()
			body := map[string]any{"err_code": "REJECTED", "err_msg": "Rejected.", "request_id": "r"}
			srv := newPrerecordedServer(t, tc.status, body)
			svc := NewSegmentedSTT(segmentedConfig(srv.URL, nil))
			ef := reportedError(t, svc)

			if ef.Category != tc.category {
				t.Errorf("category = %q, want %q", ef.Category, tc.category)
			}
			// A rejected key or setting stays rejected, so the service stops
			// being handed work.
			if svc.Usable() {
				t.Error("the service is still usable")
			}
		})
	}
}

func TestSegmentedASettingsUpdateMakesARejectedServiceUsableAgain(t *testing.T) {
	t.Parallel()
	body := map[string]any{"err_code": "INSUFFICIENT_PERMISSIONS", "err_msg": "No access.", "request_id": "r"}
	srv := newPrerecordedServer(t, http.StatusForbidden, body)
	svc := NewSegmentedSTT(segmentedConfig(srv.URL, nil))
	reportedError(t, svc)

	delta := &SegmentedSTTSettings{}
	delta.Model = settings.Set("nova-2")
	updateSettings(t, svc, delta)

	if !svc.Usable() {
		t.Error("the service is still unusable after its settings changed")
	}
}

func TestSegmentedARateLimitLeavesTheServiceUsable(t *testing.T) {
	t.Parallel()
	body := map[string]any{"err_code": "TOO_MANY_REQUESTS", "err_msg": "Too many requests.", "request_id": "r"}
	srv := newPrerecordedServer(t, http.StatusTooManyRequests, body)
	svc := NewSegmentedSTT(segmentedConfig(srv.URL, nil))
	ef := reportedError(t, svc)

	if ef.Category != errclass.RateLimit {
		t.Errorf("category = %q, want %q", ef.Category, errclass.RateLimit)
	}
	if !svc.Usable() {
		t.Error("the service is unusable after a rate limit")
	}
}

func TestSegmentedAServerErrorLeavesTheServiceUsable(t *testing.T) {
	t.Parallel()
	srv := newPrerecordedServer(t, http.StatusServiceUnavailable, map[string]any{})
	svc := NewSegmentedSTT(segmentedConfig(srv.URL, nil))
	ef := reportedError(t, svc)

	if ef.Category != errclass.Server {
		t.Errorf("category = %q, want %q", ef.Category, errclass.Server)
	}
	if !svc.Usable() {
		t.Error("the service is unusable after a server error")
	}
}

func TestSegmentedACanceledRequestLeavesTheClientUsable(t *testing.T) {
	t.Parallel()
	received := make(chan struct{})
	release := make(chan struct{})
	var requests atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := requests.Add(1)
		_, _ = io.ReadAll(r.Body)
		if n == 1 {
			close(received)
			<-release
		}
		_ = json.NewEncoder(w).Encode(listenResult("Hello there.", nil))
	}))
	t.Cleanup(srv.Close)
	tr := newSegmentedTranscriber(segmentedConfig(srv.URL, nil))

	ctx, cancel := context.WithCancel(t.Context())
	failed := make(chan error, 1)
	go func() {
		_, err := tr.transcribe(ctx, segmentWAV)
		failed <- err
	}()
	<-received
	cancel()
	if err := <-failed; !errors.Is(err, context.Canceled) {
		t.Errorf("canceled request error = %v, want a cancellation", err)
	}
	close(release)

	body, err := tr.transcribe(t.Context(), segmentWAV)
	if err != nil {
		t.Fatalf("transcribe after a cancellation: %v", err)
	}
	var result map[string]any
	if err := json.Unmarshal(body, &result); err != nil || !reflect.DeepEqual(result, listenResult("Hello there.", nil)) {
		t.Errorf("result = %v (%v), want the server's", result, err)
	}
	if got := requests.Load(); got != 2 {
		t.Errorf("requests = %d, want 2", got)
	}
}

// droppingServer drops the connection, without answering, for its first drops
// requests, as Deepgram closes an idle keep-alive connection a request was sent
// on, and counts the requests it gets.
func droppingServer(t *testing.T, drops int32) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var requests atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := requests.Add(1)
		_, _ = io.ReadAll(r.Body)
		if n <= drops {
			hj, ok := w.(http.Hijacker)
			if !ok {
				t.Error("the test server cannot hijack its connections")
				return
			}
			conn, _, err := hj.Hijack()
			if err == nil {
				_ = conn.Close()
			}
			return
		}
		_ = json.NewEncoder(w).Encode(listenResult("Hello there.", nil))
	}))
	t.Cleanup(srv.Close)
	return srv, &requests
}

func TestSegmentedARequestOnADroppedConnectionIsSentAgain(t *testing.T) {
	t.Parallel()
	srv, requests := droppingServer(t, 1)
	run := runSegmented(t, segmentedConfig(srv.URL, nil))

	if got := speakerTexts(run.transcriptions()); !reflect.DeepEqual(got, [][2]string{{"", "Hello there."}}) {
		t.Errorf("transcriptions = %v, want Hello there.", got)
	}
	if len(run.errors) != 0 {
		t.Errorf("errors = %v, want none", run.errors)
	}
	if got := requests.Load(); got != 2 {
		t.Errorf("requests = %d, want 2", got)
	}
}

func TestSegmentedARequestDroppedTwiceIsReported(t *testing.T) {
	t.Parallel()
	srv, requests := droppingServer(t, 2)
	ef := onlyError(t, runSegmented(t, segmentedConfig(srv.URL, nil)))

	if !errors.Is(ef.Err, io.EOF) && !errors.Is(ef.Err, io.ErrUnexpectedEOF) {
		t.Errorf("error = %v, want the dropped connection", ef.Err)
	}
	if got := requests.Load(); got != 2 {
		t.Errorf("requests = %d, want 2", got)
	}
}

func TestSegmentedAFailureToConnectIsNotRetried(t *testing.T) {
	t.Parallel()
	var connects atomic.Int32
	client := &http.Client{Transport: &http.Transport{
		DialContext: func(_ context.Context, network, _ string) (net.Conn, error) {
			connects.Add(1)
			return nil, &net.OpError{Op: "dial", Net: network, Err: syscall.ECONNREFUSED}
		},
	}}
	cfg := segmentedConfig("http://deepgram.invalid", nil)
	cfg.HTTPClient = client
	onlyError(t, runSegmented(t, cfg))

	if got := connects.Load(); got != 1 {
		t.Errorf("connects = %d, want 1", got)
	}
}
