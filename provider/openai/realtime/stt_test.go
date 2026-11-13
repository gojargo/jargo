package realtime

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/provider/openai/realtime/events"
)

// TestSTTConfigValidate pins which fields the provider requires.
func TestSTTConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: STTConfig{}, Valid: false},
		{Name: "API key only", Cfg: STTConfig{APIKey: "k"}, Valid: true},
		{Name: "supported noise reduction", Cfg: STTConfig{APIKey: "k", NoiseReduction: "near_field"}, Valid: true},
		{Name: "unsupported noise reduction", Cfg: STTConfig{APIKey: "k", NoiseReduction: "loud"}, Valid: false},
	})
}

// TestNewSTT checks the constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNewSTT(t *testing.T) {
	providertest.Service(t, "OpenAIRealtimeSTT", NewSTT(STTConfig{APIKey: "k"}))
}

// TestSTTShouldInterruptRidesOnRecommendedStrategies checks ShouldInterrupt
// configures the strategies the service recommends; the service never
// broadcasts the interruption itself.
func TestSTTShouldInterruptRidesOnRecommendedStrategies(t *testing.T) {
	for _, shouldInterrupt := range []bool{true, false} {
		c := &sttConnector{cfg: STTConfig{
			APIKey:          "k",
			Model:           "gpt-transcribe",
			TurnDetection:   &events.TurnDetection{Type: events.TurnDetectionServerVAD},
			ShouldInterrupt: new(shouldInterrupt),
		}}
		got, ok := c.Metadata().UserTurnStrategies.(turns.UserTurnStrategies)
		if !ok {
			t.Fatalf("UserTurnStrategies = %T, want external turn strategies", c.Metadata().UserTurnStrategies)
		}
		enabled, external := got.ExternalInterruptions()
		if !external {
			t.Error("the recommended strategies are not the external ones")
		}
		if enabled != shouldInterrupt {
			t.Errorf("interruptions enabled = %v, want %v", enabled, shouldInterrupt)
		}
	}
}

// TestSTTServerDefaultsRecommendStrategies checks an empty turn detection, which
// leaves the session's own default standing, gets the recommendation: that
// default detects turns, just as an explicit configuration does.
func TestSTTServerDefaultsRecommendStrategies(t *testing.T) {
	c := &sttConnector{cfg: STTConfig{APIKey: "k", TurnDetection: &events.TurnDetection{}}}
	if _, ok := c.Metadata().UserTurnStrategies.(turns.UserTurnStrategies); !ok {
		t.Fatalf("UserTurnStrategies = %T, want external turn strategies", c.Metadata().UserTurnStrategies)
	}
}

// TestSTTLocalVADModeRecommendsNoStrategies checks that with turn detection off,
// the default, the service recommends nothing: the server reports no
// boundaries to propose.
func TestSTTLocalVADModeRecommendsNoStrategies(t *testing.T) {
	for _, td := range []*events.TurnDetection{nil, events.NoTurnDetection()} {
		c := &sttConnector{cfg: STTConfig{APIKey: "k", Model: defaultSTTModel, TurnDetection: td}}
		if got := c.Metadata().UserTurnStrategies; got != nil {
			t.Errorf("TurnDetection %v: UserTurnStrategies = %v, want none", td, got)
		}
	}
}

// sessionInput renders a session update to JSON, as it goes on the wire, and
// returns its audio input configuration.
func sessionInput(t *testing.T, c *sttConnector) map[string]any {
	t.Helper()
	b, err := json.Marshal(c.sessionUpdate())
	if err != nil {
		t.Fatalf("marshal session update: %v", err)
	}
	var msg map[string]any
	if err := json.Unmarshal(b, &msg); err != nil {
		t.Fatalf("unmarshal session update: %v", err)
	}
	session, _ := msg["session"].(map[string]any)
	audio, _ := session[keyAudio].(map[string]any)
	input, _ := audio["input"].(map[string]any)
	return input
}

// TestSTTSessionUpdate checks the transcription session configuration: the
// session type, the audio format, and the server-side turn detection that
// delimits each utterance.
func TestSTTSessionUpdate(t *testing.T) {
	t.Run("turn detection", func(t *testing.T) {
		for _, tc := range []struct {
			name    string
			td      *events.TurnDetection
			present bool
			want    any
		}{
			{"off by default", nil, true, nil},
			{"turned off", events.NoTurnDetection(), true, nil},
			{"session default", &events.TurnDetection{}, false, nil},
			{
				"configured",
				&events.TurnDetection{Type: events.TurnDetectionServerVAD, SilenceDurationMs: new(700)},
				true,
				map[string]any{"type": "server_vad", "silence_duration_ms": float64(700)},
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				input := sessionInput(t, &sttConnector{cfg: STTConfig{APIKey: "k", TurnDetection: tc.td}})
				got, present := input["turn_detection"]
				if present != tc.present {
					t.Fatalf("turn_detection present = %v, want %v", present, tc.present)
				}
				if !reflect.DeepEqual(got, tc.want) {
					t.Errorf("turn_detection = %v, want %v", got, tc.want)
				}
			})
		}
	})

	t.Run("defaults", func(t *testing.T) {
		c := &sttConnector{cfg: STTConfig{APIKey: "k", Model: defaultSTTModel}}
		msg := c.sessionUpdate()
		if msg[keyType] != msgSessionUpdate {
			t.Errorf("type = %v, want %q", msg[keyType], msgSessionUpdate)
		}
		session, _ := msg["session"].(map[string]any)
		if session[keyType] != "transcription" {
			t.Errorf("session type = %v, want transcription", session[keyType])
		}
		audio, _ := session[keyAudio].(map[string]any)
		input, _ := audio["input"].(map[string]any)

		format, _ := input["format"].(map[string]any)
		if format[keyType] != "audio/pcm" || format["rate"] != sttSampleRate {
			t.Errorf("format = %v, want PCM at %d", format, sttSampleRate)
		}
		transcription, _ := input["transcription"].(map[string]any)
		if transcription["model"] != defaultSTTModel {
			t.Errorf("model = %v, want %q", transcription["model"], defaultSTTModel)
		}
		if _, ok := transcription["language"]; ok {
			t.Error("language is present when unset, want it omitted")
		}
		if _, ok := transcription["prompt"]; ok {
			t.Error("prompt is present when unset, want it omitted")
		}
		if _, ok := transcription["delay"]; ok {
			t.Error("delay is present when unset, want it omitted")
		}
		if turn, ok := input["turn_detection"]; !ok || turn != nil {
			t.Errorf("turn_detection = %v (present %v), want null so the local VAD delimits", turn, ok)
		}
		if _, ok := input["noise_reduction"]; ok {
			t.Error("noise_reduction is present when unset, want it omitted")
		}
	})

	t.Run("optional settings", func(t *testing.T) {
		c := &sttConnector{cfg: STTConfig{
			APIKey:         "k",
			Model:          "gpt-4o-transcribe",
			Language:       language.FrenchCA,
			Prompt:         "product names",
			Delay:          "minimal",
			NoiseReduction: "far_field",
		}}
		session, _ := c.sessionUpdate()["session"].(map[string]any)
		audio, _ := session[keyAudio].(map[string]any)
		input, _ := audio["input"].(map[string]any)

		transcription, _ := input["transcription"].(map[string]any)
		if transcription["language"] != "fr" {
			t.Errorf("language = %v, want the base code fr", transcription["language"])
		}
		if transcription["prompt"] != "product names" {
			t.Errorf("prompt = %v, want the configured prompt", transcription["prompt"])
		}
		if transcription["delay"] != "minimal" {
			t.Errorf("delay = %v, want the configured wait", transcription["delay"])
		}
		nr, _ := input["noise_reduction"].(map[string]any)
		if nr[keyType] != "far_field" {
			t.Errorf("noise_reduction = %v, want far_field", nr)
		}
	})
}

// sttServer starts a fake transcription endpoint replaying scripted events and
// recording what the client sent.
func sttServer(
	t *testing.T, events []map[string]any,
) (endpoint string, sent <-chan map[string]any, query func() url.Values) {
	t.Helper()
	messages := make(chan map[string]any, 8)
	queryCh := make(chan url.Values, 1)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case queryCh <- r.URL.Query():
		default:
		}
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		// The client configures the session before any audio flows.
		_, data, err := c.Read(ctx)
		if err != nil {
			return
		}
		var setup map[string]any
		if json.Unmarshal(data, &setup) == nil {
			messages <- setup
		}
		for _, ev := range events {
			b, _ := json.Marshal(ev)
			if c.Write(ctx, websocket.MessageText, b) != nil {
				return
			}
		}
		for {
			_, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			var msg map[string]any
			if json.Unmarshal(data, &msg) == nil {
				messages <- msg
			}
		}
	}))
	t.Cleanup(srv.Close)

	return "ws" + strings.TrimPrefix(srv.URL, "http"), messages, func() url.Values {
		select {
		case q := <-queryCh:
			return q
		default:
			return nil
		}
	}
}

// TestSTTRecv checks the interim and final mapping. The API streams interims as
// fragments, so they are accumulated into the running transcript rather than
// emitted piecemeal.
func TestSTTRecv(t *testing.T) {
	endpoint, sent, query := sttServer(t, []map[string]any{
		{"type": sttEventDelta, "delta": "hello"},
		{"type": sttEventDelta, "delta": " there"},
		{"type": sttEventCompleted, "transcript": "Hello there."},
	})

	conn := &sttConnector{cfg: STTConfig{
		APIKey:   "test-key",
		BaseURL:  endpoint,
		Model:    defaultSTTModel,
		Language: language.English,
	}}
	stream, err := conn.Connect(context.Background(), sttSampleRate)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	if q := query(); q.Get("intent") != "transcription" {
		t.Errorf("intent = %q, want transcription so the model only transcribes", q.Get("intent"))
	}
	if setup := <-sent; setup[keyType] != msgSessionUpdate {
		t.Errorf("first message = %v, want the session configuration", setup)
	}

	want := []struct {
		text  string
		final bool
	}{
		{"hello", false},
		{"hello there", false}, // accumulated, not the raw fragment
		{"Hello there.", true},
	}
	for i, w := range want {
		res, rerr := stream.Recv()
		if rerr != nil {
			t.Fatalf("Recv %d: %v", i, rerr)
		}
		if len(res) != 1 || res[0].Text != w.text || res[0].Final != w.final {
			t.Fatalf("result %d = %+v, want text %q final=%v", i, res, w.text, w.final)
		}
		if res[0].Language != "en" {
			t.Errorf("result %d language = %q, want en", i, res[0].Language)
		}
	}

	if err := stream.Send([]byte{1, 2, 3, 4}); err != nil {
		t.Fatalf("Send: %v", err)
	}
	audio := <-sent
	if audio[keyType] != "input_audio_buffer.append" {
		t.Errorf("audio message type = %v, want the buffer append", audio[keyType])
	}
	if got, _ := audio[keyAudio].(string); got != base64.StdEncoding.EncodeToString([]byte{1, 2, 3, 4}) {
		t.Errorf("audio = %q, want the base64 PCM", got)
	}
}

// TestSTTPartialResetsOnFinal checks the accumulator is cleared between
// utterances, so the next one does not inherit the previous one's text.
func TestSTTPartialResetsOnFinal(t *testing.T) {
	endpoint, _, _ := sttServer(t, []map[string]any{
		{"type": sttEventDelta, "delta": "first"},
		{"type": sttEventCompleted, "transcript": "First."},
		{"type": sttEventDelta, "delta": "second"},
	})

	conn := &sttConnector{cfg: STTConfig{APIKey: "k", BaseURL: endpoint, Model: defaultSTTModel}}
	stream, err := conn.Connect(context.Background(), sttSampleRate)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	for range 2 {
		if _, err := stream.Recv(); err != nil {
			t.Fatalf("Recv: %v", err)
		}
	}
	res, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if len(res) != 1 || res[0].Text != "second" {
		t.Errorf("result = %+v, want only the new utterance's text", res)
	}
}

// TestSTTServerError surfaces a session failure.
func TestSTTServerError(t *testing.T) {
	endpoint, _, _ := sttServer(t, []map[string]any{
		{"type": sttEventError, "error": map[string]any{"message": "invalid model"}},
	})
	conn := &sttConnector{cfg: STTConfig{APIKey: "k", BaseURL: endpoint, Model: defaultSTTModel}}
	stream, err := conn.Connect(context.Background(), sttSampleRate)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	if _, err := stream.Recv(); err == nil {
		t.Fatal("Recv() = nil error, want the server error surfaced")
	} else if !strings.Contains(err.Error(), "invalid model") {
		t.Errorf("error = %v, want it to carry the server message", err)
	}
}

// TestSTTIsSentAudioAt24kHz checks the service is asked for audio at the 24 kHz
// the session is configured for, whatever rate the pipeline runs at, so the
// audio it appends is at that rate as it stands.
func TestSTTIsSentAudioAt24kHz(t *testing.T) {
	conn := &sttConnector{cfg: STTConfig{APIKey: "k", Model: defaultSTTModel}}
	for _, rate := range []int{8000, 16000, 24000, 48000} {
		if got := conn.SendSampleRate(rate); got != sttSampleRate {
			t.Errorf("at %d Hz the service sends audio at %d Hz, want %d", rate, got, sttSampleRate)
		}
	}

	endpoint, sent, _ := sttServer(t, nil)
	conn.cfg.BaseURL = endpoint
	stream, err := conn.Connect(context.Background(), sttSampleRate)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()
	<-sent // the session configuration

	chunk := bytes.Repeat([]byte{1, 2}, 480)
	if err := stream.Send(chunk); err != nil {
		t.Fatalf("Send: %v", err)
	}
	select {
	case msg := <-sent:
		b64, _ := msg[keyAudio].(string)
		pcm, derr := base64.StdEncoding.DecodeString(b64)
		if derr != nil {
			t.Fatalf("decode audio: %v", derr)
		}
		if !bytes.Equal(pcm, chunk) {
			t.Errorf("appended %d bytes, want the %d sent as they are", len(pcm), len(chunk))
		}
	case <-time.After(2 * time.Second):
		t.Fatal("no audio was appended")
	}
}

// TestSTTFinalizeCommitsInLocalVADMode checks the local VAD's stop commits the
// audio buffer when the server-side VAD is off, and does nothing when it is on,
// since the server then commits the buffer itself.
func TestSTTFinalizeCommitsInLocalVADMode(t *testing.T) {
	for _, tc := range []struct {
		name   string
		td     *events.TurnDetection
		commit bool
	}{
		{"local VAD", nil, true},
		{"server VAD", &events.TurnDetection{Type: events.TurnDetectionServerVAD}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			endpoint, sent, _ := sttServer(t, nil)
			conn := &sttConnector{cfg: STTConfig{APIKey: "k", BaseURL: endpoint, TurnDetection: tc.td}}
			stream, err := conn.Connect(context.Background(), sttSampleRate)
			if err != nil {
				t.Fatalf("Connect: %v", err)
			}
			defer func() { _ = stream.Close() }()
			<-sent // the session configuration

			fin, ok := stream.(interface{ Finalize() error })
			if !ok {
				t.Fatal("the stream does not implement stt.Finalizer")
			}
			if err := fin.Finalize(); err != nil {
				t.Fatalf("Finalize: %v", err)
			}
			// Audio after the finalize marks where the stream is by then.
			if err := stream.Send([]byte{1, 2}); err != nil {
				t.Fatalf("Send: %v", err)
			}
			first := <-sent
			if committed := first[keyType] == "input_audio_buffer.commit"; committed != tc.commit {
				t.Errorf("first message after Finalize = %v, want commit %v", first[keyType], tc.commit)
			}
		})
	}
}

// TestSTTEndOfTurnFollowsTheMode checks a completed transcript ends the turn
// only when the server-side VAD delimited it; with the local VAD the
// pipeline's turn strategies decide.
func TestSTTEndOfTurnFollowsTheMode(t *testing.T) {
	for _, tc := range []struct {
		name string
		td   *events.TurnDetection
		eot  bool
	}{
		{"local VAD", nil, false},
		{"server VAD", &events.TurnDetection{Type: events.TurnDetectionServerVAD}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			endpoint, _, _ := sttServer(t, []map[string]any{
				{"type": sttEventCompleted, "transcript": "Hello."},
			})
			conn := &sttConnector{cfg: STTConfig{APIKey: "k", BaseURL: endpoint, TurnDetection: tc.td}}
			stream, err := conn.Connect(context.Background(), sttSampleRate)
			if err != nil {
				t.Fatalf("Connect: %v", err)
			}
			defer func() { _ = stream.Close() }()
			res, err := stream.Recv()
			if err != nil {
				t.Fatalf("Recv: %v", err)
			}
			if len(res) != 1 || !res[0].Final || res[0].EndOfTurn != tc.eot {
				t.Errorf("result = %+v, want final with EndOfTurn %v", res, tc.eot)
			}
		})
	}
}

// TestNewSTTOmitsUnsupportedPrompt checks a prompt is dropped for the default
// model, which rejects it, and kept for a model that takes one.
func TestNewSTTOmitsUnsupportedPrompt(t *testing.T) {
	if got := omitUnsupportedPrompt(STTConfig{Model: defaultSTTModel, Prompt: "names"}); got.Prompt != "" {
		t.Errorf("prompt = %q for %s, want it omitted", got.Prompt, defaultSTTModel)
	}
	if got := omitUnsupportedPrompt(STTConfig{Model: "gpt-transcribe", Prompt: "names"}); got.Prompt != "names" {
		t.Errorf("prompt = %q for gpt-transcribe, want it kept", got.Prompt)
	}
}
