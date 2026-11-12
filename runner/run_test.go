package runner

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/pion/webrtc/v4"
)

// testRunner serves the runner's routes for args, recording the sessions the
// bot is started for.
type testRunner struct {
	srv      *httptest.Server
	s        *server
	mu       sync.Mutex
	sessions []Arguments
	started  chan Arguments
}

func newTestRunner(t *testing.T, args *Args) *testRunner {
	t.Helper()
	if args.Host == "" {
		args.Host = "localhost"
	}
	if args.Port == 0 {
		args.Port = 7860
	}
	if args.WSAuth == "" {
		args.WSAuth = "none"
	}
	tr := &testRunner{started: make(chan Arguments, 8)}
	ctx, cancel := context.WithCancel(context.Background())
	// The bot lasts as long as a real session would, until the runner shuts
	// down: a session is forgotten once its bot returns.
	tr.s = newServer(ctx, args, func(ctx context.Context, a Arguments) error {
		tr.mu.Lock()
		tr.sessions = append(tr.sessions, a)
		tr.mu.Unlock()
		tr.started <- a
		<-ctx.Done()
		return ctx.Err()
	})
	tr.srv = httptest.NewServer(tr.s.routes(http.NewServeMux()))
	t.Cleanup(func() {
		cancel()
		tr.srv.Close()
		tr.s.shutdown()
		tr.s.sessions.Wait()
	})
	return tr
}

func (tr *testRunner) do(t *testing.T, method, path string, body any) (int, map[string]any) {
	t.Helper()
	var r io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		r = bytes.NewReader(b)
	}
	req, _ := http.NewRequestWithContext(t.Context(), method, tr.srv.URL+path, r)
	status, _, data := send(t, req)
	var m map[string]any
	_ = json.Unmarshal(data, &m)
	return status, m
}

// send sends req and returns the response's status, headers and body.
func send(t *testing.T, req *http.Request) (int, http.Header, []byte) {
	t.Helper()
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()
	data, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, resp.Header, data
}

// str reads a string field of a JSON response.
func str(m map[string]any, key string) string {
	v, _ := m[key].(string)
	return v
}

func (tr *testRunner) start(t *testing.T, body map[string]any) map[string]any {
	t.Helper()
	status, m := tr.do(t, http.MethodPost, "/start", body)
	if status != http.StatusOK {
		t.Fatalf("/start = %d %v", status, m)
	}
	return m
}

func (tr *testRunner) waitSession(t *testing.T) Arguments {
	t.Helper()
	select {
	case a := <-tr.started:
		return a
	case <-time.After(10 * time.Second):
		t.Fatal("the bot was not started")
		return nil
	}
}

// dialWS opens a WebSocket on the runner, reporting the HTTP status of a
// refused handshake.
func (tr *testRunner) dialWS(t *testing.T, path string, header http.Header) (*websocket.Conn, int) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	c, resp, err := websocket.Dial(ctx, "ws"+strings.TrimPrefix(tr.srv.URL, "http")+path,
		&websocket.DialOptions{HTTPHeader: header})
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		if resp != nil {
			return nil, resp.StatusCode
		}
		t.Fatalf("dial %s: %v", path, err)
	}
	t.Cleanup(func() { _ = c.CloseNow() })
	return c, http.StatusSwitchingProtocols
}

// --- WebSocket session tokens ---

func TestWSTokenHasTwoParts(t *testing.T) {
	if parts := strings.Split(newWSAuth().generate(), "."); len(parts) != 2 {
		t.Fatalf("token has %d parts, want payload.signature", len(parts))
	}
}

func TestWSTokenIsUnique(t *testing.T) {
	a := newWSAuth()
	first, second := a.generate(), a.generate()
	if first == second {
		t.Fatal("two tokens are the same")
	}
}

func TestWSTokenIsSingleUse(t *testing.T) {
	a := newWSAuth()
	token := a.generate()
	if !a.verifyAndConsume(token) {
		t.Fatal("fresh token rejected")
	}
	if a.verifyAndConsume(token) {
		t.Fatal("replayed token accepted")
	}
}

func TestWSTokenExpires(t *testing.T) {
	a := newWSAuth()
	now := time.Unix(1_000_000_000, 0)
	a.now = func() time.Time { return now }
	token := a.generate()
	now = now.Add(301 * time.Second)
	if a.verifyAndConsume(token) {
		t.Fatal("expired token accepted")
	}
}

func TestWSTokenRejectsTampering(t *testing.T) {
	a := newWSAuth()
	token := a.generate()
	i := strings.LastIndex(token, ".")
	if a.verifyAndConsume(token[:i] + ".badsignature") {
		t.Fatal("tampered signature accepted")
	}
	forged := base64.RawURLEncoding.EncodeToString([]byte(`{"exp":9999999999}`))
	if a.verifyAndConsume(forged + "." + token[i+1:]) {
		t.Fatal("tampered payload accepted")
	}
	if a.verifyAndConsume("nodothere") || a.verifyAndConsume("!!!.invalidsig") {
		t.Fatal("malformed token accepted")
	}
}

func TestExtractWSToken(t *testing.T) {
	for _, tc := range []struct {
		header, query, want string
	}{
		{"Bearer mytoken", "", "mytoken"},
		{"BEARER mytoken", "", "mytoken"},
		{"", "qptoken", "qptoken"},
		{"Bearer headertoken", "qptoken", "headertoken"},
		{"", "", ""},
	} {
		r := httptest.NewRequestWithContext(t.Context(), http.MethodGet, "/ws?token="+tc.query, nil)
		if tc.header != "" {
			r.Header.Set("Authorization", tc.header)
		}
		if got := extractWSToken(r); got != tc.want {
			t.Errorf("header %q query %q: got %q, want %q", tc.header, tc.query, got, tc.want)
		}
	}
}

// --- /start ---

func TestStartWebSocketToken(t *testing.T) {
	off := newTestRunner(t, &Args{})
	if m := off.start(t, map[string]any{"transport": "websocket"}); m["token"] != nil {
		t.Fatalf("token = %v with auth off, want null", m["token"])
	}
	on := newTestRunner(t, &Args{WSAuth: "token"})
	m := on.start(t, map[string]any{"transport": "websocket"})
	if !strings.Contains(str(m, "token"), ".") || !strings.HasSuffix(str(m, "wsUrl"), "/ws-client") {
		t.Fatalf("start = %v", m)
	}
}

func TestStartTelephonyToken(t *testing.T) {
	off := newTestRunner(t, &Args{})
	if m := off.start(t, map[string]any{"transport": "twilio"}); has(m, "token") {
		t.Fatalf("start = %v, want no token with auth off", m)
	}
	on := newTestRunner(t, &Args{WSAuth: "token"})
	m := on.start(t, map[string]any{"transport": "twilio"})
	if str(m, "token") == "" || m["wsUrl"] != "ws://localhost:7860/ws" {
		t.Fatalf("start = %v", m)
	}
}

func TestStartRejectsADisabledTransport(t *testing.T) {
	tr := newTestRunner(t, &Args{})
	status, m := tr.do(t, http.MethodPost, "/start", map[string]any{"transport": "daily"})
	if status != http.StatusBadRequest || !strings.Contains(str(m, "detail"), "disabled") {
		t.Fatalf("/start daily = %d %v", status, m)
	}
	status, _ = tr.do(t, http.MethodPost, "/start", map[string]any{"transport": "carrier-pigeon"})
	if status != http.StatusBadRequest {
		t.Fatalf("/start unknown = %d", status)
	}
	if len(tr.sessions) != 0 {
		t.Fatal("a bot was started")
	}
}

func TestStartRejectsAnotherTransportWhenRestricted(t *testing.T) {
	tr := newTestRunner(t, &Args{Transport: "webrtc"})
	status, m := tr.do(t, http.MethodPost, "/start", map[string]any{"transport": "websocket"})
	if status != http.StatusBadRequest || !strings.Contains(str(m, "detail"), "-t webrtc") {
		t.Fatalf("/start = %d %v", status, m)
	}
	_, served := tr.do(t, http.MethodGet, "/status", nil)
	if !reflect.DeepEqual(served["transports"], []any{"webrtc"}) {
		t.Fatalf("/status = %v", served)
	}
}

func TestStartICEConfig(t *testing.T) {
	plain := newTestRunner(t, &Args{})
	if m := plain.start(t, map[string]any{}); has(m, "iceConfig") {
		t.Fatalf("iceConfig without configuration or request flag: %v", m)
	}
	m := plain.start(t, map[string]any{"enableDefaultIceServers": true})
	want := map[string]any{"iceServers": []any{map[string]any{"urls": []any{"stun:stun.l.google.com:19302"}}}}
	if !reflect.DeepEqual(m["iceConfig"], want) {
		t.Fatalf("fallback iceConfig = %v", m["iceConfig"])
	}

	turn := map[string]any{"urls": "turn:turn.example.com:3478", "username": "u", "credential": "p"}
	configured := newTestRunner(t, &Args{ICEServers: []map[string]any{turn}})
	for _, body := range []map[string]any{{}, {"enableDefaultIceServers": true}} {
		m := configured.start(t, body)
		want := map[string]any{"iceServers": []any{turn}}
		if !reflect.DeepEqual(m["iceConfig"], want) {
			t.Fatalf("configured iceConfig = %v, want %v", m["iceConfig"], want)
		}
	}
}

// --- WebRTC ---

func pionOffer(t *testing.T) (*webrtc.PeerConnection, webrtc.SessionDescription) {
	t.Helper()
	m := &webrtc.MediaEngine{}
	if err := m.RegisterDefaultCodecs(); err != nil {
		t.Fatal(err)
	}
	pc, err := webrtc.NewAPI(webrtc.WithMediaEngine(m)).NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = pc.Close() })
	if _, err := pc.AddTransceiverFromKind(webrtc.RTPCodecTypeAudio); err != nil {
		t.Fatal(err)
	}
	offer, err := pc.CreateOffer(nil)
	if err != nil {
		t.Fatal(err)
	}
	gathered := webrtc.GatheringCompletePromise(pc)
	if err := pc.SetLocalDescription(offer); err != nil {
		t.Fatal(err)
	}
	<-gathered
	return pc, *pc.LocalDescription()
}

// TestStartThenOfferThroughTheSessionProxy covers the browser flow: /start
// registers a session, the offer goes through /sessions/{id}/api/offer, the bot
// starts on the connection with the body /start carried, and trickled
// candidates reach the connection through the same proxy.
func TestStartThenOfferThroughTheSessionProxy(t *testing.T) {
	tr := newTestRunner(t, &Args{})
	sessionID := str(tr.start(t, map[string]any{"body": map[string]any{"user": "ada"}}), "sessionId")

	_, offer := pionOffer(t)
	status, answer := tr.do(t, http.MethodPost, "/sessions/"+sessionID+"/api/offer",
		map[string]any{"sdp": offer.SDP, "type": "offer"})
	if status != http.StatusOK || answer["type"] != "answer" || answer["pc_id"] == "" {
		t.Fatalf("offer = %d %v", status, answer)
	}

	a, ok := tr.waitSession(t).(*WebRTCRunnerArguments)
	if !ok || a.WebRTCConnection == nil {
		t.Fatalf("bot arguments = %T", a)
	}
	if a.SessionID != sessionID || string(a.Body) != `{"user":"ada"}` {
		t.Fatalf("session %q body %s", a.SessionID, a.Body)
	}

	status, m := tr.do(t, http.MethodPatch, "/sessions/"+sessionID+"/api/offer", map[string]any{
		"pc_id":      answer["pc_id"],
		"candidates": []map[string]any{{"candidate": "", "sdp_mid": "0", "sdp_mline_index": 0}},
	})
	if status != http.StatusOK || m["status"] != "success" {
		t.Fatalf("patch = %d %v", status, m)
	}
}

func TestSessionProxyRejectsAnUnknownSession(t *testing.T) {
	tr := newTestRunner(t, &Args{})
	status, _ := tr.do(t, http.MethodPost, "/sessions/nope/api/offer", map[string]any{"sdp": "x", "type": "offer"})
	if status != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", status)
	}
}

func TestDirectOfferMintsASessionID(t *testing.T) {
	tr := newTestRunner(t, &Args{})
	_, offer := pionOffer(t)
	status, _ := tr.do(t, http.MethodPost, "/api/offer", map[string]any{"sdp": offer.SDP, "type": "offer"})
	if status != http.StatusOK {
		t.Fatalf("offer = %d", status)
	}
	if a := tr.waitSession(t); a.Base().SessionID == "" {
		t.Fatal("no session id")
	}
	status, m := tr.do(t, http.MethodPatch, "/api/offer", map[string]any{"pc_id": "unknown", "candidates": []any{}})
	if status != http.StatusNotFound {
		t.Fatalf("patch for an unknown pc_id = %d %v", status, m)
	}
}

func TestMalformedOfferIs422(t *testing.T) {
	tr := newTestRunner(t, &Args{})
	req, _ := http.NewRequestWithContext(t.Context(), http.MethodPost, tr.srv.URL+"/api/offer",
		strings.NewReader("{not json"))
	if status, _, _ := send(t, req); status != http.StatusUnprocessableEntity {
		t.Fatalf("status = %d, want 422", status)
	}
}

// --- the browser client ---

// TestServesThePrebuiltClient covers the default frontend: / sends the browser
// to /client/, which serves the prebuilt client.
func TestServesThePrebuiltClient(t *testing.T) {
	tr := newTestRunner(t, &Args{})
	client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	}}
	req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, tr.srv.URL+"/", nil)
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusTemporaryRedirect || resp.Header.Get("Location") != "/client/" {
		t.Fatalf("/ = %d %s", resp.StatusCode, resp.Header.Get("Location"))
	}
	req, _ = http.NewRequestWithContext(t.Context(), http.MethodGet, tr.srv.URL+"/client/", nil)
	status, _, body := send(t, req)
	if status != http.StatusOK || !strings.Contains(string(body), "<html") {
		t.Fatalf("/client/ = %d", status)
	}
}

// --- downloads ---

func TestDownloads(t *testing.T) {
	none := newTestRunner(t, &Args{})
	if status, _ := none.do(t, http.MethodGet, "/files/a.txt", nil); status != http.StatusNotFound {
		t.Fatalf("download without a folder = %d, want 404", status)
	}

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("hello"), 0o600); err != nil {
		t.Fatal(err)
	}
	tr := newTestRunner(t, &Args{DownloadsFolder: dir})
	req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, tr.srv.URL+"/files/a.txt", nil)
	status, header, data := send(t, req)
	if status != http.StatusOK || string(data) != "hello" ||
		!strings.Contains(header.Get("Content-Disposition"), "a.txt") {
		t.Fatalf("download = %d %q %v", status, data, header)
	}
	if status, _ := tr.do(t, http.MethodGet, "/files/missing.txt", nil); status != http.StatusNotFound {
		t.Fatalf("missing file = %d, want 404", status)
	}
	if _, ok := resolveDownloadPath(dir, "../outside"); ok {
		t.Fatal("a path leaving the folder resolved")
	}
}

// --- WebSocket routes ---

func TestPlainWebSocketAuth(t *testing.T) {
	tr := newTestRunner(t, &Args{WSAuth: "token"})
	if _, status := tr.dialWS(t, "/ws-client", nil); status != http.StatusForbidden {
		t.Fatalf("no token = %d, want 403", status)
	}
	if _, status := tr.dialWS(t, "/ws-client?token=bad.token", nil); status != http.StatusForbidden {
		t.Fatalf("invalid token = %d, want 403", status)
	}

	token := tr.s.auth.generate()
	if _, status := tr.dialWS(t, "/ws-client?token="+token, nil); status != http.StatusSwitchingProtocols {
		t.Fatalf("valid query token = %d", status)
	}
	if a, ok := tr.waitSession(t).(*WebSocketRunnerArguments); !ok || a.TransportType != "websocket" {
		t.Fatalf("bot arguments = %T", a)
	}
	if _, status := tr.dialWS(t, "/ws-client/"+token, nil); status != http.StatusForbidden {
		t.Fatalf("replayed token = %d, want 403", status)
	}
	if _, status := tr.dialWS(t, "/ws-client/"+tr.s.auth.generate(), nil); status != 101 {
		t.Fatalf("valid path token = %d", status)
	}
}

func TestWebSocketsAllowAnyConnectionWhenAuthIsOff(t *testing.T) {
	tr := newTestRunner(t, &Args{})
	for _, path := range []string{"/ws-client", "/ws"} {
		if _, status := tr.dialWS(t, path, nil); status != http.StatusSwitchingProtocols {
			t.Fatalf("%s = %d", path, status)
		}
	}
}

func TestTelephonyWebSocketAuth(t *testing.T) {
	tr := newTestRunner(t, &Args{WSAuth: "token"})
	if _, status := tr.dialWS(t, "/ws", nil); status != http.StatusForbidden {
		t.Fatalf("no token = %d, want 403", status)
	}
	if _, status := tr.dialWS(t, "/ws/"+tr.s.auth.generate(), nil); status != 101 {
		t.Fatalf("valid path token = %d", status)
	}
	if a, ok := tr.waitSession(t).(*WebSocketRunnerArguments); !ok || a.TransportType != "" {
		t.Fatalf("bot arguments = %T, want a telephony session left to detect", a)
	}
}

func TestWebSocketOrigins(t *testing.T) {
	tr := newTestRunner(t, &Args{AllowedOrigins: []string{"https://allowed.example.com"}})
	for _, path := range []string{"/ws-client", "/ws"} {
		bad := http.Header{"Origin": {"https://evil.example.com"}}
		if _, status := tr.dialWS(t, path, bad); status != http.StatusForbidden {
			t.Fatalf("%s disallowed origin = %d, want 403", path, status)
		}
		good := http.Header{"Origin": {"https://allowed.example.com"}}
		if _, status := tr.dialWS(t, path, good); status != http.StatusSwitchingProtocols {
			t.Fatalf("%s allowed origin = %d", path, status)
		}
	}
	if _, status := tr.dialWS(t, "/ws-client", nil); status != http.StatusForbidden {
		t.Fatalf("missing origin with origins configured = %d, want 403", status)
	}
}

func TestTelephonyBotGetsTheRunnerBody(t *testing.T) {
	file := filepath.Join(t.TempDir(), "body.yaml")
	if err := os.WriteFile(file, []byte("image: cat.png\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	tr := newTestRunner(t, &Args{RunnerBody: file})
	tr.dialWS(t, "/ws", nil)
	if body := string(tr.waitSession(t).Base().Body); body != `{"image":"cat.png"}` {
		t.Fatalf("body = %s", body)
	}

	// test_telephony_bot_without_a_runner_body: no body at all, not an empty
	// one.
	plain := newTestRunner(t, &Args{})
	plain.dialWS(t, "/ws", nil)
	if body := plain.waitSession(t).Base().Body; body != nil {
		t.Fatalf("body without a runner body = %s, want none", body)
	}
}

func TestTelephonyWebhookXML(t *testing.T) {
	tr := newTestRunner(t, &Args{Transport: "twilio", Proxy: "bot.example.com"})
	req, _ := http.NewRequestWithContext(t.Context(), http.MethodPost, tr.srv.URL+"/", nil)
	_, header, body := send(t, req)
	if header.Get("Content-Type") != "application/xml" ||
		!strings.Contains(string(body), `<Stream url="wss://bot.example.com/ws">`) {
		t.Fatalf("webhook = %s %s", header.Get("Content-Type"), body)
	}
}

// --- CORS ---

func TestCORS(t *testing.T) {
	preflight := func(url, origin string) (int, http.Header) {
		req, _ := http.NewRequestWithContext(t.Context(), http.MethodOptions, url+"/start", nil)
		req.Header.Set("Origin", origin)
		req.Header.Set("Access-Control-Request-Method", "POST")
		status, header, _ := send(t, req)
		return status, header
	}

	tr := newTestRunner(t, &Args{})
	status, header := preflight(tr.srv.URL, "http://localhost:3000")
	if status != http.StatusOK || header.Get("Access-Control-Allow-Origin") != "http://localhost:3000" {
		t.Fatalf("preflight = %d %v", status, header)
	}

	restricted := newTestRunner(t, &Args{AllowedOrigins: []string{"https://app.example.com"}})
	if status, _ := preflight(restricted.srv.URL, "https://evil.example.com"); status != http.StatusBadRequest {
		t.Fatalf("preflight from a disallowed origin = %d, want 400", status)
	}
}

// --- ICE servers ---

func TestParseICEServers(t *testing.T) {
	ok := []struct {
		name   string
		env    string
		tokens []string
		want   []map[string]any
	}{
		{"empty", "", nil, nil},
		{
			"cli bare urls", "",
			[]string{"stun:stun.l.google.com:19302", "turn:turn.example.com:3478"},
			[]map[string]any{{"urls": "stun:stun.l.google.com:19302"}, {"urls": "turn:turn.example.com:3478"}},
		},
		{
			"cli json object", "",
			[]string{`{"urls": "turn:t:3478", "username": "u", "credential": "p"}`},
			[]map[string]any{{"urls": "turn:t:3478", "username": "u", "credential": "p"}},
		},
		{
			"env commas", "stun:stun.l.google.com:19302, turn:turn.example.com:3478", nil,
			[]map[string]any{{"urls": "stun:stun.l.google.com:19302"}, {"urls": "turn:turn.example.com:3478"}},
		},
		{
			"env json array", `[{"urls": ["turn:t:3478"], "username": "u", "credential": "p"}, "stun:s:19302"]`, nil,
			[]map[string]any{{"urls": []any{"turn:t:3478"}, "username": "u", "credential": "p"}, {"urls": "stun:s:19302"}},
		},
		{
			"null credentials dropped", `{"urls": "stun:s:3478", "username": null}`, nil,
			[]map[string]any{{"urls": "stun:s:3478"}},
		},
	}
	for _, tc := range ok {
		got, err := parseICEServers(tc.env, tc.tokens)
		if err != nil || !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s: got %v %v, want %v", tc.name, got, err, tc.want)
		}
	}
	for _, env := range []string{
		`{"username": "u", "credential": "p"}`,
		`{"urls": "turn:t:3478", "user": "u"}`,
		`{"urls": `,
		`[{"urls": "turn:t:3478"`,
		`[7]`,
	} {
		if _, err := parseICEServers(env, nil); err == nil {
			t.Errorf("%s: accepted", env)
		}
	}
}

// --- startup output ---

func TestStartupMessage(t *testing.T) {
	var all bytes.Buffer
	printStartupMessage(&all, &Args{Host: "localhost", Port: 7860})
	if !strings.Contains(all.String(), "Open: http://localhost:7860") ||
		!strings.Contains(all.String(), "Enabled transports:") {
		t.Fatalf("all transports:\n%s", all.String())
	}

	var tel bytes.Buffer
	printStartupMessage(&tel, &Args{Host: "localhost", Port: 7860, Transport: "twilio", Proxy: "p.example.com"})
	for _, want := range []string{
		"(Twilio)", "XML webhook: http://localhost:7860/", "WebSocket:   ws://localhost:7860/ws",
	} {
		if !strings.Contains(tel.String(), want) {
			t.Fatalf("telephony message lacks %q:\n%s", want, tel.String())
		}
	}
}

func TestResolveFlags(t *testing.T) {
	t.Setenv("JARGO_ICE_SERVERS", "")
	t.Setenv("JARGO_ALLOWED_ORIGINS", "https://a.example.com, https://b.example.com")
	f := &rawFlags{args: Args{Host: "localhost", WSAuth: "none", Proxy: "https://bot.ngrok.io/"}}
	args, err := f.resolve()
	if err != nil {
		t.Fatal(err)
	}
	if args.Proxy != "bot.ngrok.io" ||
		!reflect.DeepEqual(args.AllowedOrigins, []string{"https://a.example.com", "https://b.example.com"}) {
		t.Fatalf("args = %+v", args)
	}
	for _, bad := range []rawFlags{
		{args: Args{Host: "localhost", WSAuth: "none", Transport: "daily"}},
		{args: Args{Host: "localhost", WSAuth: "maybe"}},
		{args: Args{Host: "localhost", WSAuth: "none", ESP32: true}},
	} {
		if _, err := bad.resolve(); err == nil {
			t.Errorf("%+v accepted", bad.args)
		}
	}
}

// TestRunServesUntilCanceled covers the entry point: it parses its flags,
// serves, and shuts down when its context ends.
func TestRunServesUntilCanceled(t *testing.T) {
	var lc net.ListenConfig
	l, err := lc.Listen(t.Context(), "tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := l.Addr().(*net.TCPAddr).Port //nolint:forcetypeassert // a TCP listener's address
	_ = l.Close()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	fs := flag.NewFlagSet("bot", flag.ContinueOnError)
	go func() {
		done <- Run(ctx, fs, []string{"--host", "127.0.0.1", "--port", strconv.Itoa(port), "-t", "webrtc"},
			func(context.Context, Arguments) error { return nil })
	}()

	url := fmt.Sprintf("http://127.0.0.1:%d/status", port)
	deadline := time.Now().Add(5 * time.Second)
	for {
		req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, url, nil)
		resp, err := http.DefaultClient.Do(req)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("/status = %d", resp.StatusCode)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the runner never served: %v", err)
		}
		time.Sleep(20 * time.Millisecond)
	}
	if RunnerPort() != port || RunnerHost() != "127.0.0.1" {
		t.Fatalf("runner host/port = %s/%d", RunnerHost(), RunnerPort())
	}

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Run = %v", err)
		}
	case <-time.After(15 * time.Second):
		t.Fatal("Run did not return after cancel")
	}
}
