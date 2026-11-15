package wsclient

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline/capabilities"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport/wsclient/rtviclient"
)

// The RTVI-specific parts added on top of the WebSocket client transport: the
// client-ready handshake message and bot-ready detection.

// connectedSession is a session to srv that reports nothing, standing in for
// the one a transport would have connected.
func connectedSession(t *testing.T, srv *testServer) *Session {
	t.Helper()
	s := NewSession(srv.url, Params{}, Callbacks{}, "Test")
	if err := s.Connect(t.Context()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Disconnect(context.Background()) })
	return s
}

func botReadyMessage(t *testing.T, data any) []byte {
	t.Helper()
	raw, err := json.Marshal(map[string]any{"label": rtvi.MessageLabel, "type": "bot-ready", "id": "1", "data": data})
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestDefaultsToRTVIClientSerializer(t *testing.T) {
	tr := NewRTVITransport("ws://localhost:0", Params{})

	if _, ok := tr.params.Serializer.(*rtviclient.Serializer); !ok {
		t.Errorf("serializer = %T, want *rtviclient.Serializer", tr.params.Serializer)
	}
}

// TestRawAudioCarriesBarePCM checks the audio write path sends the PCM as given:
// no WAV header on the wire.
func TestRawAudioCarriesBarePCM(t *testing.T) {
	srv := newTestServer(t, nil)
	pcm := bytes.Repeat(func() []byte {
		b := make([]byte, 256)
		for i := range b {
			b[i] = byte(i)
		}
		return b
	}(), 4)
	out := NewRTVITransport("ws://localhost:0", Params{}).outputTransport()
	out.session = connectedSession(t, srv)
	setUpOutput(t, out, 16000)

	written, err := out.WriteAudio(t.Context(), frames.NewOutputAudioRawFrame(pcm, 16000, 1))
	if err != nil || !written {
		t.Fatalf("WriteAudio = %v, %v; want true", written, err)
	}

	var sent struct {
		Type string            `json:"type"`
		Data rtvi.RawAudioData `json:"data"`
	}
	if err := json.Unmarshal(srv.conn(t, 0).next(t), &sent); err != nil {
		t.Fatal(err)
	}
	if sent.Type != "raw-audio" {
		t.Errorf("type = %q, want raw-audio", sent.Type)
	}
	if got, err := base64.StdEncoding.DecodeString(sent.Data.Base64Audio); err != nil || !bytes.Equal(got, pcm) {
		t.Errorf("audio is not the bare PCM given (%v)", err)
	}
	if sent.Data.SampleRate != 16000 || sent.Data.NumChannels != 1 {
		t.Errorf("(sampleRate, numChannels) = (%d, %d), want (16000, 1)", sent.Data.SampleRate, sent.Data.NumChannels)
	}
}

func TestSendClientReadyMessage(t *testing.T) {
	srv := newTestServer(t, nil)
	tr := NewRTVITransport("ws://localhost:0", Params{})
	tr.session = connectedSession(t, srv)

	tr.sendClientReady(t.Context())

	conn := srv.conn(t, 0)
	var sent struct {
		Label string                     `json:"label"`
		Type  string                     `json:"type"`
		Data  map[string]json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(conn.next(t), &sent); err != nil {
		t.Fatal(err)
	}
	if sent.Label != rtvi.MessageLabel || sent.Type != "client-ready" {
		t.Errorf("(label, type) = (%q, %q), want (%q, client-ready)", sent.Label, sent.Type, rtvi.MessageLabel)
	}
	var version string
	if err := json.Unmarshal(sent.Data["version"], &version); err != nil || version != rtvi.ProtocolVersion {
		t.Errorf("version = %q (%v), want %q", version, err, rtvi.ProtocolVersion)
	}
	var about map[string]any
	if err := json.Unmarshal(sent.Data["about"], &about); err != nil {
		t.Fatalf("about: %v", err)
	}
	if _, ok := about["library"]; !ok {
		t.Errorf("about = %v, want a library", about)
	}
	select {
	case m := <-conn.received:
		t.Errorf("a second message was sent: %s", m)
	default:
	}
}

func TestIsBotReady(t *testing.T) {
	notReady := []byte(`{"label": "rtvi-ai", "type": "bot-llm-text"}`)
	noLabel := []byte(`{"type": "bot-ready"}`)

	if !isBotReady(botReadyMessage(t, map[string]any{"version": "2.0.0"})) {
		t.Error("bot-ready not recognized")
	}
	for name, message := range map[string][]byte{
		"not ready": notReady, "no label": noLabel, "not json": []byte("not json"),
	} {
		if isBotReady(message) {
			t.Errorf("%s: recognized as bot-ready", name)
		}
	}
}

func TestCapabilities(t *testing.T) {
	reported := botCapabilities(botReadyMessage(t, map[string]any{
		"version": "2.0.0", "capabilities": map[string]any{"audio_in": false, "audio_out": true},
	}))

	want := &capabilities.BotCapabilities{AudioIn: new(false), AudioOut: new(true)}
	if !reflect.DeepEqual(reported, want) {
		t.Errorf("capabilities = %+v, want %+v", reported, want)
	}
	for name, data := range map[string]any{
		"none reported": map[string]any{"version": "1.0.0"},
		"no data":       nil,
		"unreadable":    map[string]any{"capabilities": map[string]any{"audio_in": "maybe"}},
	} {
		if got := botCapabilities(botReadyMessage(t, data)); got != nil {
			t.Errorf("%s: capabilities = %+v, want nil", name, got)
		}
	}
}

func TestBotReadyFlagSetOnMessage(t *testing.T) {
	tr := NewRTVITransport("ws://localhost:0", Params{})
	if tr.BotReady() {
		t.Fatal("bot ready before bot-ready")
	}

	tr.onMessage(t.Context(), nil, botReadyMessage(t, map[string]any{"version": "2.0.0"}))

	if !tr.BotReady() {
		t.Error("bot not ready after bot-ready")
	}
}
