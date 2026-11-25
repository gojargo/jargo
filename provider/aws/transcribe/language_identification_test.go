package transcribe

import (
	"context"
	"errors"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/transcribestreaming"
	"github.com/aws/aws-sdk-go-v2/service/transcribestreaming/types"
	"github.com/gojargo/jargo/errclass"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
)

// languageRequest is the request c opens a session with at 16 kHz.
func languageRequest(t *testing.T, c *connector) *transcribestreaming.StartStreamTranscriptionInput {
	t.Helper()
	in, err := c.request(16000)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	return in
}

// Ported from upstream (test_configured_language_is_sent_without_language_options).
func TestConfiguredLanguageIsSentWithoutLanguageOptions(t *testing.T) {
	in := languageRequest(t, newConnector(Config{Language: "es-ES"}))

	if in.LanguageCode != "es-ES" {
		t.Errorf("language code = %q, want es-ES", in.LanguageCode)
	}
	if in.IdentifyLanguage || in.IdentifyMultipleLanguages || in.LanguageOptions != nil || in.PreferredLanguage != "" {
		t.Errorf("identification requested without language options: %+v", in)
	}
}

// Ported from upstream (test_language_options_replace_the_configured_language).
func TestLanguageOptionsReplaceTheConfiguredLanguage(t *testing.T) {
	tests := []struct {
		name         string
		cfg          Config
		wantMultiple bool
		wantOptions  string
		wantPrefer   types.LanguageCode
	}{
		{
			name:        "the dominant language",
			cfg:         Config{LanguageOptions: []string{"en-US", "es-ES"}},
			wantOptions: "en-US,es-ES",
		},
		{
			name: "each segment's language, with a preferred one",
			cfg: Config{
				LanguageOptions:           []string{"en-US", "es-US"},
				IdentifyMultipleLanguages: true,
				PreferredLanguage:         "es-US",
			},
			wantMultiple: true,
			wantOptions:  "en-US,es-US",
			wantPrefer:   "es-US",
		},
		{
			// Transcribe rejects a repeated code, so each is listed once.
			name:        "a repeated code",
			cfg:         Config{LanguageOptions: []string{"en-US", "en-US", "fr-FR"}},
			wantOptions: "en-US,fr-FR",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.cfg.Language = "de-DE"
			in := languageRequest(t, newConnector(tt.cfg))

			if in.LanguageCode != "" {
				t.Errorf("language code = %q, want none alongside identification", in.LanguageCode)
			}
			if in.IdentifyLanguage != !tt.wantMultiple || in.IdentifyMultipleLanguages != tt.wantMultiple {
				t.Errorf("identify language = %v, identify multiple = %v, want multiple %v",
					in.IdentifyLanguage, in.IdentifyMultipleLanguages, tt.wantMultiple)
			}
			if got := aws.ToString(in.LanguageOptions); got != tt.wantOptions {
				t.Errorf("language options = %q, want %q", got, tt.wantOptions)
			}
			if in.PreferredLanguage != tt.wantPrefer {
				t.Errorf("preferred language = %q, want %q", in.PreferredLanguage, tt.wantPrefer)
			}
		})
	}
}

// Ported from upstream (test_conflicting_settings_fail_at_construction).
func TestConflictingSettingsAreRefusedByValidate(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
		want error
	}{
		{"multiple languages without options", Config{IdentifyMultipleLanguages: true}, errIdentifyMultipleWithoutOptions},
		{"a preferred language without options", Config{PreferredLanguage: "en-US"}, errPreferredWithoutOptions},
		{"one option", Config{LanguageOptions: []string{"en-US"}}, errTooFewLanguageOptions},
		{"one option repeated", Config{LanguageOptions: []string{"en-US", "en-US"}}, errTooFewLanguageOptions},
		{
			"a preferred language that is not an option",
			Config{LanguageOptions: []string{"en-US", "es-ES"}, PreferredLanguage: "fr-FR"},
			errPreferredNotAnOption,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.cfg.Validate(); !errors.Is(err, tt.want) {
				t.Errorf("Validate() = %v, want %v", err, tt.want)
			}
		})
	}
	valid := Config{LanguageOptions: []string{"en-US", "es-ES"}, PreferredLanguage: "es-ES"}
	if err := valid.Validate(); err != nil {
		t.Errorf("Validate() = %v, want the settings accepted", err)
	}
}

// settingsRun is a Transcribe service running in a pipeline, its sessions
// recorded rather than opened on AWS.
type settingsRun struct {
	conn *recordingConnector
	svc  *stt.StreamService
	task *pipeline.Worker

	mu     sync.Mutex
	errors []*frames.ErrorFrame
}

// runSettings starts a Transcribe service built from cfg and waits for its
// first session.
func runSettings(t *testing.T, cfg Config) *settingsRun {
	t.Helper()
	r := &settingsRun{conn: &recordingConnector{connector: newConnector(cfg), sent: make(chan []byte, 4)}}
	r.svc = stt.NewStream("TranscribeSTT", r.conn, 16000)
	event.On(r.svc.Events(), processor.EventError, func(_ context.Context, ef *frames.ErrorFrame) {
		r.mu.Lock()
		r.errors = append(r.errors, ef)
		r.mu.Unlock()
	})
	r.task = pipeline.NewWorker(pipeline.New(r.svc), pipeline.WorkerConfig{})
	done := make(chan error, 1)
	go func() { done <- r.task.Run(context.Background()) }()
	t.Cleanup(func() {
		r.task.StopWhenDone()
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Error("the pipeline did not shut down")
		}
	})
	r.waitSessions(t, 1)
	return r
}

// update sends the service a settings update.
func (r *settingsRun) update(delta *STTSettings) {
	r.task.QueueFrame(frames.NewSTTUpdateSettingsFrame(delta))
}

// waitSessions waits until n sessions have been opened.
func (r *settingsRun) waitSessions(t *testing.T, n int) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for len(r.conn.opened()) < n {
		if time.Now().After(deadline) {
			t.Fatalf("%d sessions opened, want %d", len(r.conn.opened()), n)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// waitErrors waits until n errors have been reported, and returns them.
func (r *settingsRun) waitErrors(t *testing.T, n int) []*frames.ErrorFrame {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for {
		r.mu.Lock()
		got := slices.Clone(r.errors)
		r.mu.Unlock()
		if len(got) >= n {
			return got
		}
		if time.Now().After(deadline) {
			t.Fatalf("%d errors reported, want %d", len(got), n)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// request is the request the next session opens with.
func (r *settingsRun) request(t *testing.T) *transcribestreaming.StartStreamTranscriptionInput {
	t.Helper()
	// Taking the lock the recording takes orders this read after the update
	// the last session was opened for.
	r.conn.mu.Lock()
	defer r.conn.mu.Unlock()
	in, err := r.conn.connector.request(16000)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	return in
}

// Ported from upstream
// (test_runtime_update_turns_language_identification_on_and_reconnects).
func TestRuntimeUpdateTurnsLanguageIdentificationOnAndReopens(t *testing.T) {
	r := runSettings(t, Config{})

	r.update(&STTSettings{
		LanguageOptions:   settings.Set([]string{"en-US", "es-ES"}),
		PreferredLanguage: settings.Set("es-ES"),
	})
	r.waitSessions(t, 2)

	in := r.request(t)
	if got := aws.ToString(in.LanguageOptions); got != "en-US,es-ES" {
		t.Errorf("language options = %q, want en-US,es-ES", got)
	}
	if in.PreferredLanguage != "es-ES" || in.LanguageCode != "" {
		t.Errorf("preferred = %q, language code = %q, want es-ES and none", in.PreferredLanguage, in.LanguageCode)
	}
}

// Ported from upstream (test_runtime_update_is_checked_against_the_settings_it_joins).
func TestRuntimeUpdateIsCheckedAgainstTheSettingsItJoins(t *testing.T) {
	r := runSettings(t, Config{LanguageOptions: []string{"en-US", "es-ES"}})

	r.update(&STTSettings{PreferredLanguage: settings.Set("es-ES")})
	r.waitSessions(t, 2)

	if got := r.request(t).PreferredLanguage; got != "es-ES" {
		t.Errorf("preferred language = %q, want es-ES", got)
	}
}

// Ported from upstream (test_conflicting_runtime_update_is_rejected_whole).
func TestConflictingRuntimeUpdateIsRejectedWhole(t *testing.T) {
	r := runSettings(t, Config{LanguageOptions: []string{"en-US", "es-ES"}, PreferredLanguage: "es-ES"})

	// Dropping the options while the preferred language is still set conflicts,
	// so the stability change in the same update is not applied either.
	r.update(&STTSettings{
		LanguageOptions:         settings.Cleared[[]string](),
		PartialResultsStability: settings.Set("low"),
	})

	got := r.waitErrors(t, 1)
	if len(got) != 1 {
		t.Fatalf("%d errors reported, want 1", len(got))
	}
	ef := got[0]
	if ef.Fatal || ef.Category != errclass.Application ||
		!strings.Contains(ef.Err.Error(), "preferred_language requires") {
		t.Errorf("error = %v (%v, fatal %v), want a non-fatal application error naming preferred_language",
			ef.Err, ef.Category, ef.Fatal)
	}
	if n := len(r.conn.opened()); n != 1 {
		t.Errorf("%d sessions opened, want the first kept", n)
	}
	in := r.request(t)
	if got := aws.ToString(in.LanguageOptions); got != "en-US,es-ES" {
		t.Errorf("language options = %q, want them kept", got)
	}
	if in.PartialResultsStability != types.PartialResultsStabilityHigh {
		t.Errorf("stability = %q, want high kept", in.PartialResultsStability)
	}
}

// Ported from upstream (test_runtime_update_turns_language_identification_off).
func TestRuntimeUpdateTurnsLanguageIdentificationOff(t *testing.T) {
	r := runSettings(t, Config{
		Language:          "fr-FR",
		LanguageOptions:   []string{"en-US", "es-ES"},
		PreferredLanguage: "es-ES",
	})

	r.update(&STTSettings{
		LanguageOptions:   settings.Cleared[[]string](),
		PreferredLanguage: settings.Cleared[string](),
	})
	r.waitSessions(t, 2)

	in := r.request(t)
	if in.LanguageCode != "fr-FR" || in.LanguageOptions != nil || in.IdentifyLanguage {
		t.Errorf("language code = %q, options = %v, identify = %v, want fr-FR alone",
			in.LanguageCode, aws.ToString(in.LanguageOptions), in.IdentifyLanguage)
	}
}

// Ported from upstream (test_frames_carry_the_identified_language).
func TestResultsCarryTheIdentifiedLanguage(t *testing.T) {
	s := &stream{lang: defaultLanguage, identify: true}

	var got []string
	for _, r := range []types.Result{segment("Hola", true, "es-US"), segment("Hola.", false, "es-US")} {
		for _, res := range s.results([]types.Result{r}) {
			got = append(got, res.Language)
		}
	}
	if !slices.Equal(got, []string{"es-US", "es-US"}) {
		t.Errorf("languages = %v, want es-US on the interim and the final", got)
	}
}

// Ported from upstream
// (test_frames_carry_the_configured_language_without_identification).
func TestResultsCarryTheConfiguredLanguageWithoutIdentification(t *testing.T) {
	s := &stream{lang: "fr-FR"}

	got := s.results([]types.Result{segment("Bonjour.", false, "")})
	if len(got) != 1 || got[0].Language != "fr-FR" {
		t.Errorf("results = %+v, want one in fr-FR", got)
	}
}
