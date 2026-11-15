package speechmatics

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/utils/events"
)

// Ported from upstream (test_resolve_model_*). The model and the deprecated
// operating_point alias resolve into the one model the session is opened with.
func TestResolveModel(t *testing.T) {
	cases := []struct {
		name, model, operatingPoint, want string
	}{
		// test_resolve_model_prefers_model_over_none_operating_point
		{"model alone", "linden-1", "", "linden-1"},
		// test_resolve_model_operating_point_only_warns_and_is_used
		{"operating point alone", "", "linden-1", "linden-1"},
		// test_resolve_model_defaults_when_neither_given
		{"neither", "", "", defaultModel},
		{"both the same", "linden-2", "linden-2", "linden-2"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := resolveModel(tc.model, tc.operatingPoint)
			if err != nil || got != tc.want {
				t.Errorf("resolveModel(%q, %q) = %q, %v; want %q", tc.model, tc.operatingPoint, got, err, tc.want)
			}
		})
	}
}

// Ported from upstream (test_resolve_model_conflicting_values_raise). Two
// different values is a caller error, not a silent pick.
func TestResolveModelConflictingValues(t *testing.T) {
	if _, err := resolveModel("linden-1", "some-other-model"); !errors.Is(err, errModelConflict) {
		t.Errorf("err = %v, want the conflict reported", err)
	}
	cfg := Config{APIKey: "k", Model: "linden-1", OperatingPoint: "some-other-model"}
	if err := cfg.Validate(); !errors.Is(err, errModelConflict) {
		t.Errorf("Validate = %v, want the conflict reported", err)
	}
}

// Ported from upstream (test_default_turn_detection_mode). The default drives
// turn events and endpointing, so it is pinned.
func TestDefaultTurnDetectionMode(t *testing.T) {
	if got := conn("").live.TurnDetectionMode.Or(""); got != TurnDetectionExternal {
		t.Errorf("turn detection mode = %q, want %q", got, TurnDetectionExternal)
	}
	if defaultTurnDetectionMode != TurnDetectionExternal {
		t.Errorf("default = %q, want external", defaultTurnDetectionMode)
	}
}

// Ported from upstream (test_operating_point_resolved_into_model). The
// deprecated operating point is resolved into the model at construction.
func TestOperatingPointResolvedIntoModel(t *testing.T) {
	c := conn("", func(c *Config) { c.OperatingPoint = "linden-1" })
	if got := c.live.Model.Or(""); got != "linden-1" {
		t.Errorf("model = %q, want linden-1", got)
	}
}

// Ported from upstream (test_diarization_config_built_when_enabled and
// test_no_diarization_leaves_config_empty).
func TestDiarizationConfig(t *testing.T) {
	tc := conn("", func(c *Config) {
		c.EnableDiarization = new(true)
		c.MaxSpeakers = new(2)
	}).live.transcriptionConfig()
	if tc["diarization"] != "speaker" {
		t.Errorf("diarization = %v, want speaker", tc["diarization"])
	}
	dc, _ := tc["speaker_diarization_config"].(map[string]any)
	if !reflect.DeepEqual(dc, map[string]any{"max_speakers": 2}) {
		t.Errorf("speaker_diarization_config = %v, want max_speakers alone", dc)
	}

	// Diarization off sends neither, whatever knob is set.
	tc = conn("", func(c *Config) { c.MaxSpeakers = new(2) }).live.transcriptionConfig()
	if _, ok := tc["diarization"]; ok {
		t.Errorf("diarization = %v, want it omitted", tc["diarization"])
	}
	if _, ok := tc["speaker_diarization_config"]; ok {
		t.Errorf("speaker_diarization_config = %v, want it omitted", tc["speaker_diarization_config"])
	}

	// Diarization on with no knob sends no empty config.
	tc = conn("", func(c *Config) { c.EnableDiarization = new(true) }).live.transcriptionConfig()
	if _, ok := tc["speaker_diarization_config"]; ok {
		t.Errorf("speaker_diarization_config = %v, want no empty config", tc["speaker_diarization_config"])
	}
}

// Ported from upstream (test_punctuation_overrides_reach_the_wire_config).
func TestPunctuationOverridesReachTheWireConfig(t *testing.T) {
	overrides := map[string]any{"permitted_marks": []string{".", ","}}
	tc := conn("", func(c *Config) { c.PunctuationOverrides = overrides }).live.transcriptionConfig()
	if !reflect.DeepEqual(tc["punctuation_overrides"], overrides) {
		t.Errorf("punctuation_overrides = %v, want %v", tc["punctuation_overrides"], overrides)
	}
	if _, ok := conn("").live.transcriptionConfig()["punctuation_overrides"]; ok {
		t.Error("punctuation_overrides sent when not set")
	}
}

// Ported from upstream (test_emit_sentences_reaches_the_wire_config_only_when_set).
// Unset means the service default, so the key stays off the wire.
func TestEmitSentencesReachesTheWireConfigOnlyWhenSet(t *testing.T) {
	if _, ok := conn("").live.transcriptionConfig()["emit_sentences"]; ok {
		t.Error("emit_sentences sent when not set")
	}
	tc := conn("", func(c *Config) { c.EmitSentences = new(true) }).live.transcriptionConfig()
	if tc["emit_sentences"] != true {
		t.Errorf("emit_sentences = %v, want true", tc["emit_sentences"])
	}
}

// Ported from upstream (test_service_closes_turns_true_for_vad and
// test_service_closes_turns_false_for_external).
func TestServiceClosesTurns(t *testing.T) {
	vad := conn("", func(c *Config) { c.TurnDetectionMode = TurnDetectionVAD })
	if !vad.live.serviceClosesTurns() {
		t.Error("VAD mode does not close turns")
	}
	external := conn("", func(c *Config) { c.TurnDetectionMode = TurnDetectionExternal })
	if external.live.serviceClosesTurns() {
		t.Error("external mode closes turns")
	}
}

// Ported from upstream (test_metadata_frame_routes_should_interrupt_to_external_strategies
// and test_metadata_frame_recommends_no_strategies_in_external_mode).
func TestMetadataRecommendsStrategiesWhenTheServiceClosesTurns(t *testing.T) {
	c := conn("", func(c *Config) {
		c.TurnDetectionMode = TurnDetectionVAD
		c.ShouldInterrupt = new(false)
	})
	strategies, ok := c.Metadata().UserTurnStrategies.(turns.UserTurnStrategies)
	if !ok {
		t.Fatalf("strategies = %T, want the external ones", c.Metadata().UserTurnStrategies)
	}
	enabled, external := strategies.ExternalInterruptions()
	if !external || enabled {
		t.Errorf("external = %v, interruptions = %v; want external strategies without interruptions", external, enabled)
	}

	if s := conn("").Metadata().UserTurnStrategies; s != nil {
		t.Errorf("strategies = %v, want none in external mode", s)
	}
}

// Ported from upstream (test_locale_maps_regional_variant,
// test_locale_none_without_regional_variant and
// test_unsupported_regional_variant_constructs_and_falls_back).
func TestOutputLocale(t *testing.T) {
	if got := speechmaticsLocale("en", language.EnglishGB); got != "en-GB" {
		t.Errorf("locale = %q, want en-GB", got)
	}
	if got := speechmaticsLocale("en", language.English); got != "" {
		t.Errorf("locale = %q, want none", got)
	}
	tc := conn("", func(c *Config) { c.Language = language.EnglishIN }).live.transcriptionConfig()
	if tc["language"] != "en" {
		t.Errorf("language = %v, want en", tc["language"])
	}
	if _, ok := tc["output_locale"]; ok {
		t.Errorf("output_locale = %v, want it omitted", tc["output_locale"])
	}
}

// applyUpdate merges an update into the connector's settings and asks it
// whether the session must be reopened, as the service does.
func applyUpdate(t *testing.T, c *connector, delta *Settings) (settings.Changed, bool) {
	t.Helper()
	changed, err := settings.Apply(c.live, delta)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	reopen, err := c.UpdateSettings(context.Background(), changed)
	if err != nil {
		t.Fatalf("UpdateSettings: %v", err)
	}
	return changed, reopen
}

// Ported from upstream (test_update_speaker_format_does_not_reconnect). The
// speaker format is applied locally.
func TestUpdateSpeakerFormatDoesNotReconnect(t *testing.T) {
	changed, reopen := applyUpdate(t, conn(""), &Settings{SpeakerFormat: settings.Set("{speaker}: {text}")})
	if !changed.Has("speaker_format") {
		t.Errorf("changed = %v, want speaker_format", changed)
	}
	if reopen {
		t.Error("a speaker format change reopens the session")
	}
}

// Ported from upstream (test_update_settings_operating_point_reresolves_into_model).
// The deprecated operating_point is resolved into the model the session is
// built from, and the session is reopened.
func TestUpdateOperatingPointReresolvesIntoModel(t *testing.T) {
	c := conn("")
	changed, reopen := applyUpdate(t, c, &Settings{OperatingPoint: settings.Set("linden-2")})
	if got := c.live.Model.Or(""); got != "linden-2" {
		t.Errorf("model = %q, want linden-2", got)
	}
	if got := c.live.transcriptionConfig()["model"]; got != "linden-2" {
		t.Errorf("wire model = %v, want linden-2", got)
	}
	if !changed.Has("model") {
		t.Errorf("changed = %v, want the model reported changed", changed)
	}
	if !reopen {
		t.Error("the session is not reopened")
	}
}

// Ported from upstream (test_update_settings_operating_point_does_not_clash_with_resolved_model).
// A lone operating_point update wins on its own instead of clashing with the
// model already resolved.
func TestUpdateOperatingPointDoesNotClashWithResolvedModel(t *testing.T) {
	c := conn("", func(c *Config) { c.Model = "linden-1" })
	applyUpdate(t, c, &Settings{OperatingPoint: settings.Set("linden-2")})
	if got := c.live.Model.Or(""); got != "linden-2" {
		t.Errorf("model = %q, want linden-2", got)
	}
}

// Ported from upstream (test_update_settings_model_reresolves_into_model).
func TestUpdateModelReresolvesIntoModel(t *testing.T) {
	c := conn("")
	delta := &Settings{}
	delta.Model = settings.Set("linden-2")
	_, reopen := applyUpdate(t, c, delta)
	if got := c.live.transcriptionConfig()["model"]; got != "linden-2" {
		t.Errorf("wire model = %v, want linden-2", got)
	}
	if !reopen {
		t.Error("the session is not reopened")
	}
}

// Ported from upstream (test_update_settings_unrelated_field_leaves_model_untouched).
func TestUpdateUnrelatedFieldLeavesModelUntouched(t *testing.T) {
	c := conn("")
	changed, reopen := applyUpdate(t, c, &Settings{Domain: settings.Set("finance")})
	if got := c.live.Model.Or(""); got != defaultModel {
		t.Errorf("model = %q, want %q", got, defaultModel)
	}
	if changed.Has("model") {
		t.Errorf("changed = %v, want the model left alone", changed)
	}
	if got := c.live.Domain.Or(""); got != "finance" {
		t.Errorf("domain = %q, want finance", got)
	}
	if !reopen {
		t.Error("a domain change does not reopen the session")
	}
}

// Ported from upstream (test_open_connection_unsupported_sample_rate_is_permanent).
// A rate the service does not accept fails every attempt the same way, so it is
// permanent, and nothing is dialed.
func TestConnectUnsupportedSampleRateIsPermanent(t *testing.T) {
	var dialed atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { dialed.Add(1) }))
	t.Cleanup(srv.Close)

	_, err := conn("ws"+strings.TrimPrefix(srv.URL, "http")).Connect(context.Background(), 24000)
	if !errors.Is(err, stt.ErrPermanent) {
		t.Errorf("err = %v, want a permanent failure", err)
	}
	if n := dialed.Load(); n != 0 {
		t.Errorf("dials = %d, want none", n)
	}
}

// Ported from upstream (test_open_connection_supported_sample_rate_connects).
// Both rates the service accepts reach the audio format unchanged.
func TestConnectSupportedSampleRateConnects(t *testing.T) {
	for _, rate := range supportedSampleRates {
		endpoint, seen := sttServer(t, "", nil)
		s, err := conn(endpoint).Connect(context.Background(), rate)
		if err != nil {
			t.Fatalf("Connect(%d): %v", rate, err)
		}
		format, _ := seen().start["audio_format"].(map[string]any)
		if format["sample_rate"] != float64(rate) {
			t.Errorf("sample_rate = %v, want %d", format["sample_rate"], rate)
		}
		_ = s.Close()
	}
}

// Ported from upstream (test_open_connection_auth_rejection_is_permanent). A
// handshake rejected for its credentials stops the session for good rather
// than being retried.
func TestAnAuthRejectionIsPermanent(t *testing.T) {
	var dialed atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		dialed.Add(1)
		http.Error(w, "unauthorized", http.StatusUnauthorized)
	}))
	t.Cleanup(srv.Close)
	r := runService(t, "ws"+strings.TrimPrefix(srv.URL, "http"))

	waitFor(t, "the service to become unusable", func() bool { return !r.svc.Usable() })
	if len(r.reported()) == 0 {
		t.Fatal("the rejection was not reported")
	}
	if !r.reported()[0].Category.IsPermanent() {
		t.Errorf("category = %v, want a permanent one", r.reported()[0].Category)
	}
	time.Sleep(50 * time.Millisecond)
	if n := dialed.Load(); n != 1 {
		t.Errorf("dials = %d, want the rejected one alone", n)
	}
}

// Ported from upstream (test_send_message_without_connection_raises and
// test_send_message_propagates_send_failure). Sending with no session open
// fails, and so does a send the session refuses.
func TestSendMessage(t *testing.T) {
	svc := NewSTT(Config{APIKey: "k"})
	if err := SendMessage(svc, "GetSpeakers", nil); !errors.Is(err, errNotConnected) {
		t.Errorf("err = %v, want it to report no session", err)
	}

	endpoint, seen := sttServer(t, "", nil)
	c, ok := svc.Connector().(*connector)
	if !ok {
		t.Fatal("not a Speechmatics connector")
	}
	c.cfg.URL = endpoint
	s, err := c.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	if err := SendMessage(svc, "GetSpeakers", map[string]any{"final": true}); err != nil {
		t.Fatalf("SendMessage: %v", err)
	}
	select {
	case m := <-seen().messages:
		if m["message"] != "GetSpeakers" || m["final"] != true {
			t.Errorf("message = %v, want GetSpeakers with its fields", m)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the message never arrived")
	}

	// A session closed under the caller refuses the send.
	inner := c.current()
	_ = inner.conn.CloseNow()
	if err := SendMessage(svc, "GetSpeakers", nil); err == nil {
		t.Error("a send on a closed session succeeded")
	}
	_ = s.Close()
	if err := SendMessage(svc, "GetSpeakers", nil); !errors.Is(err, errNotConnected) {
		t.Errorf("err = %v, want no session once it closed", err)
	}
}

// transcriptRun runs the service against a fake endpoint that sends reply, and
// collects the transcripts that reach the end of the pipeline.
type transcriptRun struct {
	mu     sync.Mutex
	frames []frames.Frame
}

func (r *transcriptRun) got() []frames.Frame {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]frames.Frame(nil), r.frames...)
}

func runTranscripts(t *testing.T, cfg Config, reply []map[string]any, before ...frames.Frame) *transcriptRun {
	t.Helper()
	endpoint, _ := sttServer(t, "", reply)
	cfg.APIKey, cfg.URL, cfg.SampleRate = "test-key", endpoint, 16000
	svc := NewSTT(cfg)
	r := &transcriptRun{}
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		IdleTimeout:             -1,
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		switch f.(type) {
		case *frames.TranscriptionFrame, *frames.InterimTranscriptionFrame,
			*frames.ProposedUserStartedSpeakingFrame, *frames.ProposedUserStoppedSpeakingFrame:
			r.mu.Lock()
			r.frames = append(r.frames, f)
			r.mu.Unlock()
		}
	})
	for _, f := range before {
		task.QueueFrame(f)
	}
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	t.Cleanup(func() {
		task.StopWhenDone()
		select {
		case <-done:
		case <-time.After(10 * time.Second):
			t.Error("the pipeline did not shut down")
		}
	})
	return r
}

// Ported from upstream (test_segment_to_frame_final_vs_interim_type,
// test_segment_to_frame_carries_speaker_and_keeps_user_id,
// test_segment_to_frame_without_speaker_has_no_label and
// test_segment_to_frame_applies_speaker_format). A final segment is a
// TranscriptionFrame and a partial one an interim; the speaker goes on the
// frame, the user stays the transport participant, and the speaker format
// labels the text only when there is a speaker.
func TestSegmentsBecomeTranscripts(t *testing.T) {
	r := runTranscripts(t,
		Config{EnableDiarization: new(true), SpeakerFormat: "{speaker}: {text}"},
		[]map[string]any{
			{"message": awaitAudio},
			segment(msgAddPartialSegment, "hi", ""),
			segment(msgAddSegment, "hi", "S1"),
		},
		frames.NewUserAudioRawFrame("session-user", make([]byte, 320), 16000, 1),
	)
	waitFor(t, "two transcripts", func() bool { return len(r.got()) == 2 })

	got := r.got()
	interim, ok := got[0].(*frames.InterimTranscriptionFrame)
	if !ok {
		t.Fatalf("first frame = %T, want an interim", got[0])
	}
	if interim.Text != "hi" || interim.Speaker != nil {
		t.Errorf("interim = %q by %v, want the text unlabeled with no speaker", interim.Text, interim.Speaker)
	}
	final, ok := got[1].(*frames.TranscriptionFrame)
	if !ok {
		t.Fatalf("second frame = %T, want a transcription", got[1])
	}
	if final.Text != "S1: hi" {
		t.Errorf("text = %q, want the speaker format applied", final.Text)
	}
	if final.Speaker == nil || final.Speaker.Label != "S1" {
		t.Errorf("speaker = %v, want S1", final.Speaker)
	}
	if final.UserID != "session-user" {
		t.Errorf("user = %q, want the transport participant", final.UserID)
	}
}

// With Speechmatics closing turns, its turn events reach the pipeline as
// proposals for the turn strategies to resolve, around the transcript.
func TestTurnEventsBecomeProposals(t *testing.T) {
	r := runTranscripts(t,
		Config{TurnDetectionMode: TurnDetectionVAD},
		[]map[string]any{
			{"message": msgStartOfTurn},
			segment(msgAddSegment, "hello", ""),
			{"message": msgEndOfTurn},
		},
	)
	waitFor(t, "the turn", func() bool { return len(r.got()) == 3 })

	var kinds []string
	for _, f := range r.got() {
		kinds = append(kinds, f.Name())
	}
	if !strings.HasPrefix(kinds[0], "ProposedUserStartedSpeakingFrame") ||
		!strings.HasPrefix(kinds[1], "TranscriptionFrame") ||
		!strings.HasPrefix(kinds[2], "ProposedUserStoppedSpeakingFrame") {
		t.Errorf("frames = %v, want start, transcript, stop", kinds)
	}
}

// A speakers result is reported through the event, with diarization on.
func TestSpeakersResultIsReported(t *testing.T) {
	endpoint, _ := sttServer(t, "", []map[string]any{
		{"message": msgSpeakersResult, "speakers": []any{map[string]any{"label": "S1"}}},
	})
	svc := NewSTT(Config{APIKey: "k", URL: endpoint, SampleRate: 16000, EnableDiarization: new(true)})
	got := make(chan map[string]any, 1)
	events.On(svc.Events(), EventSpeakersResult, func(_ context.Context, msg map[string]any) { got <- msg })
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{IdleTimeout: -1})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	defer func() {
		task.StopWhenDone()
		<-done
	}()

	select {
	case msg := <-got:
		if msg["message"] != msgSpeakersResult {
			t.Errorf("message = %v, want the SpeakersResult", msg)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the speakers result was never reported")
	}
}

// The resolved URL selects the Agent STT endpoint and names the application,
// keeping what was already on it.
func TestResolveURL(t *testing.T) {
	cases := map[string]string{
		"wss://eu2.rt.speechmatics.com/v2":           "wss://eu2.rt.speechmatics.com/v2/agent?sm-app=jargo",
		"wss://eu2.rt.speechmatics.com/v2/agent":     "wss://eu2.rt.speechmatics.com/v2/agent?sm-app=jargo",
		"wss://host/v2/?x=1":                         "wss://host/v2/agent?sm-app=jargo&x=1",
		"wss://host/v2/agent?sm-app=other&region=eu": "wss://host/v2/agent?region=eu&sm-app=jargo",
	}
	for in, want := range cases {
		if got := resolveURL(in); got != want {
			t.Errorf("resolveURL(%q) = %q, want %q", in, got, want)
		}
	}
}

var _ processor.Processor = (*stt.StreamService)(nil)
