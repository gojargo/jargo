package whatsapp

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/transport/rtc"
	"github.com/pion/webrtc/v4"
)

func TestValidate(t *testing.T) {
	full := Config{Token: "t", PhoneNumberID: "p", AppSecret: "s", VerifyToken: "v"}
	if err := full.Validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	if err := (Config{Token: "t"}).Validate(); err == nil {
		t.Fatal("incomplete config should be rejected")
	}
}

func TestFilterSDP(t *testing.T) {
	in := strings.Join([]string{
		"v=0",
		"a=fingerprint:sha-256 AA:BB",
		"a=fingerprint:sha-1 CC:DD",
		"a=fingerprint:sha-512 EE:FF",
		"a=setup:actpass",
	}, "\n")
	out := filterSDP(in)
	if !strings.Contains(out, "a=fingerprint:sha-256 AA:BB") {
		t.Error("sha-256 fingerprint was dropped")
	}
	if strings.Contains(out, "sha-1") || strings.Contains(out, "sha-512") {
		t.Errorf("non-sha-256 fingerprint kept: %q", out)
	}
	if !strings.Contains(out, "a=setup:actpass") {
		t.Error("non-fingerprint line dropped")
	}
}

func TestValidSignature(t *testing.T) {
	c := &Client{cfg: Config{AppSecret: "topsecret"}}
	body := []byte(`{"hello":"world"}`)
	if !c.validSignature(body, sign("topsecret", body)) {
		t.Error("valid signature rejected")
	}
	if c.validSignature(body, sign("wrong", body)) {
		t.Error("signature with wrong secret accepted")
	}
	if c.validSignature(body, "") {
		t.Error("missing signature accepted")
	}
}

func TestVerifyChallenge(t *testing.T) {
	c := mustClient(t, "")
	rec := httptest.NewRecorder()
	req := httptest.NewRequestWithContext(context.Background(), http.MethodGet,
		"/webhook?hub.mode=subscribe&hub.verify_token=vt&hub.challenge=42", nil)
	c.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || rec.Body.String() != "42" {
		t.Fatalf("verify: code=%d body=%q, want 200 and 42", rec.Code, rec.Body.String())
	}

	rec = httptest.NewRecorder()
	bad := httptest.NewRequestWithContext(context.Background(), http.MethodGet,
		"/webhook?hub.mode=subscribe&hub.verify_token=nope&hub.challenge=42", nil)
	c.Handler().ServeHTTP(rec, bad)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("wrong verify token: code=%d, want 403", rec.Code)
	}
}

func TestWebhookRejectsBadSignature(t *testing.T) {
	c := mustClient(t, "")
	body := []byte(`{"object":"whatsapp_business_account","entry":[]}`)
	req := httptest.NewRequestWithContext(context.Background(), http.MethodPost,
		"/webhook", strings.NewReader(string(body)))
	req.Header.Set("X-Hub-Signature-256", sign("wrongsecret", body))
	rec := httptest.NewRecorder()
	c.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("bad signature: code=%d, want 403", rec.Code)
	}
}

func TestWebhookConnectAcceptsCall(t *testing.T) {
	var mu sync.Mutex
	var actions []string
	graph := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var b map[string]any
		_ = json.NewDecoder(r.Body).Decode(&b)
		mu.Lock()
		actions = append(actions, fmt.Sprint(b["action"]))
		mu.Unlock()
		_, _ = w.Write([]byte(`{"success":true}`))
	}))
	defer graph.Close()

	c := mustClient(t, graph.URL+"/")
	got := make(chan *Call, 1)
	c.OnCall(func(call *Call) { got <- call })

	body := connectBody("call-1", "15551230000", makeOffer(t))
	post(t, c, body)

	select {
	case call := <-got:
		if call.ID != "call-1" || call.From != "15551230000" {
			t.Fatalf("unexpected call: %+v", call)
		}
		if call.Connection == nil {
			t.Fatal("nil connection")
		}
		_ = call.Connection.Close()
	case <-time.After(5 * time.Second):
		t.Fatal("OnCall was not invoked")
	}

	mu.Lock()
	defer mu.Unlock()
	if len(actions) != 2 || actions[0] != "pre_accept" || actions[1] != "accept" {
		t.Fatalf("graph actions = %v, want [pre_accept accept]", actions)
	}
}

func TestWebhookTerminateClosesCall(t *testing.T) {
	c := mustClient(t, "")
	conn, err := rtc.NewConnection(rtc.WithICEServers())
	if err != nil {
		t.Fatalf("new connection: %v", err)
	}
	c.store("call-9", conn)

	post(t, c, terminateBody("call-9"))

	select {
	case <-conn.Done():
	case <-time.After(3 * time.Second):
		t.Fatal("terminate did not close the connection")
	}
}

// --- helpers ---

func mustClient(t *testing.T, baseURL string) *Client {
	t.Helper()
	c, err := NewClient(Config{
		Token: "token", PhoneNumberID: "123", AppSecret: "secret",
		VerifyToken: "vt", BaseURL: baseURL,
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	return c
}

func sign(secret string, body []byte) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(body)
	return "sha256=" + hex.EncodeToString(mac.Sum(nil))
}

func post(t *testing.T, c *Client, body []byte) {
	t.Helper()
	req := httptest.NewRequestWithContext(context.Background(), http.MethodPost,
		"/webhook", strings.NewReader(string(body)))
	req.Header.Set("X-Hub-Signature-256", sign("secret", body))
	rec := httptest.NewRecorder()
	c.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("webhook POST: code=%d, want 200", rec.Code)
	}
}

func makeOffer(t *testing.T) string {
	t.Helper()
	pc, err := webrtc.NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		t.Fatalf("new peer connection: %v", err)
	}
	defer func() { _ = pc.Close() }()
	if _, err := pc.AddTransceiverFromKind(webrtc.RTPCodecTypeAudio); err != nil {
		t.Fatalf("add transceiver: %v", err)
	}
	offer, err := pc.CreateOffer(nil)
	if err != nil {
		t.Fatalf("create offer: %v", err)
	}
	gather := webrtc.GatheringCompletePromise(pc)
	if err := pc.SetLocalDescription(offer); err != nil {
		t.Fatalf("set local description: %v", err)
	}
	<-gather
	return pc.LocalDescription().SDP
}

func connectBody(callID, from, sdp string) []byte {
	env := map[string]any{
		"object": "whatsapp_business_account",
		"entry": []map[string]any{{
			"changes": []map[string]any{{
				"field": "calls",
				"value": map[string]any{
					"calls": []map[string]any{{
						"id":      callID,
						"from":    from,
						"event":   "connect",
						"session": map[string]any{"sdp": sdp, "sdp_type": "offer"},
					}},
				},
			}},
		}},
	}
	b, _ := json.Marshal(env)
	return b
}

func terminateBody(callID string) []byte {
	env := map[string]any{
		"object": "whatsapp_business_account",
		"entry": []map[string]any{{
			"changes": []map[string]any{{
				"field": "calls",
				"value": map[string]any{
					"calls": []map[string]any{{
						"id":     callID,
						"event":  "terminate",
						"status": "COMPLETED",
					}},
				},
			}},
		}},
	}
	b, _ := json.Marshal(env)
	return b
}
