package runner

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/livekit/protocol/auth"
)

// telephonySocket serves a WebSocket whose client sends messages and then
// closes, and returns the server end wrapped as runner arguments.
func telephonySocket(t *testing.T, messages ...string) *WebSocketRunnerArguments {
	t.Helper()
	accepted := make(chan *websocket.Conn, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			t.Errorf("accept: %v", err)
			return
		}
		accepted <- c
		<-r.Context().Done()
	}))
	t.Cleanup(srv.Close)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	client, resp, err := websocket.Dial(ctx, "ws"+strings.TrimPrefix(srv.URL, "http"), nil)
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	for _, m := range messages {
		if err := client.Write(ctx, websocket.MessageText, []byte(m)); err != nil {
			t.Fatalf("write: %v", err)
		}
	}
	// Closed abruptly: a graceful close would wait on a server not reading yet.
	_ = client.CloseNow()

	server := <-accepted
	t.Cleanup(func() { _ = server.CloseNow() })
	return &WebSocketRunnerArguments{WebSocket: server}
}

func parse(t *testing.T, messages ...string) (string, *CallData, error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return ParseTelephonyWebSocket(ctx, telephonySocket(t, messages...))
}

func TestParseTelephonyNoMessages(t *testing.T) {
	if _, _, err := parse(t); !errors.Is(err, ErrHandshakeClosed) {
		t.Fatalf("err = %v, want ErrHandshakeClosed", err)
	}
}

func TestParseTelephonyOneMessageContinues(t *testing.T) {
	typ, data, err := parse(t,
		`{"event":"start","start":{"streamSid":"MZ123","callSid":"CA123","customParameters":{"user_id":"test_user"}}}`)
	if err != nil {
		t.Fatal(err)
	}
	if typ != "twilio" || data.StreamID != "MZ123" || data.CallID != "CA123" {
		t.Fatalf("got %s %+v", typ, data)
	}
}

func TestParseTelephonyTwoMessages(t *testing.T) {
	typ, data, err := parse(t, `{"event":"connected"}`,
		`{"event":"start","start":{"streamSid":"MZ456","callSid":"CA456","customParameters":{}}}`)
	if err != nil {
		t.Fatal(err)
	}
	if typ != "twilio" || data.StreamID != "MZ456" || data.CallID != "CA456" {
		t.Fatalf("got %s %+v", typ, data)
	}
}

// TestParseTwilioPromotesFromAndTo covers Twilio carrying the numbers as TwiML
// stream parameters: they are promoted so call data reads the same across
// providers, and the raw parameters stay in Body.
func TestParseTwilioPromotesFromAndTo(t *testing.T) {
	_, data, err := parse(t, `{"event":"start","start":{"streamSid":"MZ789","callSid":"CA789",`+
		`"customParameters":{"from_number":"+1555","to_number":"+1666"}}}`)
	if err != nil {
		t.Fatal(err)
	}
	if data.FromNumber != "+1555" || data.ToNumber != "+1666" || data.Body["to_number"] != "+1666" {
		t.Fatalf("call data = %+v", data)
	}
}

func TestParseTelnyx(t *testing.T) {
	typ, data, err := parse(t, `{"stream_id":"stream_123","start":{"call_control_id":"cc_123",`+
		`"media_format":{"encoding":"PCMU"},"from":"+15551234567","to":"+15559876543"}}`)
	if err != nil {
		t.Fatal(err)
	}
	if typ != "telnyx" || data.StreamID != "stream_123" || data.CallID != "cc_123" ||
		data.OutboundEncoding != "PCMU" || data.FromNumber != "+15551234567" || data.ToNumber != "+15559876543" {
		t.Fatalf("got %s %+v", typ, data)
	}
}

func TestParsePlivo(t *testing.T) {
	typ, data, err := parse(t, `{"start":{"streamId":"stream_456","callId":"call_456"}}`)
	if err != nil {
		t.Fatal(err)
	}
	if typ != "plivo" || data.StreamID != "stream_456" || data.CallID != "call_456" {
		t.Fatalf("got %s %+v", typ, data)
	}
}

func TestParseExotel(t *testing.T) {
	typ, data, err := parse(t, `{"event":"connected"}`, `{"event":"start","start":{"stream_sid":"s_789",`+
		`"call_sid":"c_789","account_sid":"acc_789","from":"+1","to":"+2","custom_parameters":{"k":"v"}}}`)
	if err != nil {
		t.Fatal(err)
	}
	if typ != "exotel" || data.StreamID != "s_789" || data.CallID != "c_789" || data.AccountSID != "acc_789" ||
		data.FromNumber != "+1" || data.ToNumber != "+2" {
		t.Fatalf("got %s %+v", typ, data)
	}
	if cp, ok := data.CustomParameters.(map[string]any); !ok || cp["k"] != "v" {
		t.Fatalf("custom parameters = %v", data.CustomParameters)
	}
}

// TestParseIsIdempotent covers parsing twice: the messages are gone from the
// socket after the first parse, so the second returns the first's result.
func TestParseIsIdempotent(t *testing.T) {
	args := telephonySocket(t, `{"event":"start","start":{"streamSid":"MZ1","callSid":"CA1"}}`)
	ctx := context.Background()
	typ1, data1, err1 := ParseTelephonyWebSocket(ctx, args)
	typ2, data2, err2 := ParseTelephonyWebSocket(ctx, args)
	if err1 != nil || err2 != nil || typ1 != typ2 || data1 != data2 {
		t.Fatalf("second parse differs: %s %v %v / %s %v %v", typ1, data1, err1, typ2, data2, err2)
	}
}

// TestCreateTransportExposesCallData covers a telephony session: the provider
// is detected, the bot's arguments carry it and the call data, and the
// messages the parse read are handed to the transport.
func TestCreateTransportExposesCallData(t *testing.T) {
	args := telephonySocket(t, `{"event":"connected"}`,
		`{"event":"start","start":{"streamSid":"MZ9","callSid":"CA9","customParameters":{"to_number":"+1666"}}}`)
	tr, err := CreateTransport(context.Background(), args, TransportParams{
		Twilio: func() WebSocketParams { return WebSocketParams{Params: wsserver.DefaultParams()} },
	})
	if err != nil {
		t.Fatal(err)
	}
	if tr == nil {
		t.Fatal("no transport")
	}
	if args.TransportType != "twilio" || args.CallData == nil || args.CallData.CallID != "CA9" ||
		args.CallData.ToNumber != "+1666" {
		t.Fatalf("arguments = %s %+v", args.TransportType, args.CallData)
	}
	if !args.consumedTaken || len(args.consumed) != 2 {
		t.Fatalf("the transport was not handed the %d messages the parse read", len(args.consumed))
	}
}

// TestParseEvalConnectDetectedFromFirstMessage covers an eval harness: its
// eval-connect is recognized from the first message, and the call data's body
// carries its flags.
func TestParseEvalConnectDetectedFromFirstMessage(t *testing.T) {
	flags := evalws.ConnectionFlags{SkipTTS: true, CaptureBotAudio: true}
	typ, data, err := parse(t, evalws.ConnectMessage(flags))
	if err != nil {
		t.Fatal(err)
	}
	if typ != "eval" || evalFlags(data.Body) != flags {
		t.Fatalf("got %s %+v, want eval with %+v", typ, data, flags)
	}
}

// TestParseEvalConnectLeavesTheNextMessageUnread covers the harness's
// client-ready, which follows eval-connect and must reach the bot.
func TestParseEvalConnectLeavesTheNextMessageUnread(t *testing.T) {
	args := telephonySocket(t, evalws.ConnectMessage(evalws.ConnectionFlags{}), `{"type":"client-ready"}`)
	if _, _, err := ParseTelephonyWebSocket(context.Background(), args); err != nil {
		t.Fatal(err)
	}
	if len(args.consumed) != 1 {
		t.Fatalf("the parse read %d messages, want only the eval-connect", len(args.consumed))
	}
}

func TestCreateTransportEvalConnectBuildsTheEvalTransportWithItsFlags(t *testing.T) {
	flags := evalws.ConnectionFlags{SkipTTS: true, CaptureBotAudio: true}
	args := telephonySocket(t, evalws.ConnectMessage(flags))

	tr, err := CreateTransport(context.Background(), args, TransportParams{
		Eval: func() evalws.TransportParams { return evalws.TransportParams{Params: wsserver.DefaultParams()} },
	})
	if err != nil {
		t.Fatal(err)
	}

	evalTr, ok := tr.(*evalws.Transport)
	if !ok {
		t.Fatalf("transport = %T, want the eval transport", tr)
	}
	if args.TransportType != "eval" {
		t.Fatalf("transport type = %q, want eval", args.TransportType)
	}
	ser, ok := evalTr.Params().Serializer.(*evalws.Serializer)
	if !ok {
		t.Fatalf("serializer = %T, want the eval serializer", evalTr.Params().Serializer)
	}
	// The bot's audio is captured: it goes out to the harness.
	out, err := ser.Serialize(frames.NewOutputAudioRawFrame([]byte{0, 0}, 16000, 1))
	if err != nil || out.Empty() {
		t.Fatalf("bot audio = %q, %v, want it captured", out.Data, err)
	}
}

func TestCreateTransportEvalConnectWithoutEvalParamsFails(t *testing.T) {
	args := telephonySocket(t, evalws.ConnectMessage(evalws.ConnectionFlags{}))
	_, err := CreateTransport(context.Background(), args, TransportParams{
		Twilio: func() WebSocketParams { return WebSocketParams{Params: wsserver.DefaultParams()} },
	})
	if !errors.Is(err, ErrMissingTransportParams) || !strings.Contains(err.Error(), `"eval"`) {
		t.Fatalf("err = %v, want it to name the missing eval params", err)
	}
}

func TestCreateTransportNeedsParams(t *testing.T) {
	args := telephonySocket(t, `{"start":{"streamId":"s","callId":"c"}}`)
	_, err := CreateTransport(context.Background(), args, TransportParams{})
	if err == nil || !strings.Contains(err.Error(), "TransportParams.Plivo") {
		t.Fatalf("err = %v, want it to name the missing params", err)
	}
	if _, err := CreateTransport(context.Background(), &WebRTCRunnerArguments{}, TransportParams{}); err == nil {
		t.Fatal("WebRTC without params accepted")
	}
}

func TestCreateTransportRejectsUnknownArguments(t *testing.T) {
	_, err := CreateTransport(context.Background(), &BaseArguments{}, TransportParams{
		WebRTC: transport.DefaultParams,
	})
	if err == nil {
		t.Fatal("plain arguments accepted")
	}
}

func TestLiveKitCredentials(t *testing.T) {
	t.Setenv("LIVEKIT_URL", "")
	t.Setenv("LIVEKIT_API_KEY", "")
	t.Setenv("LIVEKIT_API_SECRET", "")
	if _, _, _, err := LiveKitCredentials(); !errors.Is(err, ErrLiveKitNotConfigured) {
		t.Fatalf("err = %v, want ErrLiveKitNotConfigured", err)
	}
	t.Setenv("LIVEKIT_URL", "wss://lk.example.com")
	t.Setenv("LIVEKIT_API_KEY", "key")
	t.Setenv("LIVEKIT_API_SECRET", "secret")
	url, key, secret, err := LiveKitCredentials()
	if err != nil || url != "wss://lk.example.com" || key != "key" || secret != "secret" {
		t.Fatalf("got %q %q %q %v", url, key, secret, err)
	}
}

// TestGenerateSessionTokensSuffixesIdentities covers sessions sharing a room:
// each token's identity carries the session, and only the bot's is an agent.
func TestGenerateSessionTokensSuffixesIdentities(t *testing.T) {
	const secret = "a-secret-long-enough-for-hmac-signing"
	agent, user, err := GenerateSessionTokens("room", "abcdef1234567890", "key", secret)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		token, identity string
		agent           bool
	}{{agent, "jargo Agent-abcdef12", true}, {user, "User-abcdef12", false}} {
		v, err := auth.ParseAPIToken(tc.token)
		if err != nil {
			t.Fatal(err)
		}
		_, claims, err := v.Verify(secret)
		if err != nil {
			t.Fatal(err)
		}
		if claims.Identity != tc.identity || claims.Video.Room != "room" || !claims.Video.RoomJoin ||
			claims.Video.Agent != tc.agent {
			t.Fatalf("claims = %+v, video = %+v", claims, claims.Video)
		}
	}
}
