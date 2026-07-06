// Package whatsapp answers inbound WhatsApp Business calls and bridges their
// audio into a jargo pipeline. It runs a webhook endpoint for the WhatsApp Cloud
// (Graph) API: it verifies the webhook, answers a "connect" event by building a
// WebRTC answer with the Pion transport (transport/pionrtc) and accepting the
// call over the Graph API, and closes the media on a "terminate" event.
//
// The media transport is the standard Pion WebRTC transport — the connected
// *pionrtc.Connection handed to the OnCall callback is wrapped with
// pionrtc.NewTransport to build the pipeline, exactly like the /offer examples.
package whatsapp

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"sync"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/transport/pionrtc"
	"github.com/pion/webrtc/v4"
)

// maxWebhookBody bounds a webhook request body.
const maxWebhookBody = 1 << 20

// Config configures a WhatsApp calling client.
type Config struct {
	// Token is the WhatsApp access token (Bearer) for the Graph API. Required.
	Token string `validate:"required"`
	// PhoneNumberID is the business phone number id whose calls are handled. Required.
	PhoneNumberID string `validate:"required"`
	// AppSecret verifies the X-Hub-Signature-256 on webhook deliveries. Required.
	AppSecret string `validate:"required"`
	// VerifyToken is checked against hub.verify_token on webhook verification. Required.
	VerifyToken string `validate:"required"`
	// BaseURL overrides the Graph API base; empty uses the hosted endpoint.
	BaseURL string
	// ICEServers configures WebRTC connectivity; empty uses a public STUN server.
	ICEServers []webrtc.ICEServer
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// Call is an accepted WhatsApp call whose media is ready to bridge. Wrap
// Connection with pionrtc.NewTransport to build the pipeline.
type Call struct {
	ID         string
	From       string
	Connection *pionrtc.Connection

	client *Client
}

// Hangup terminates the call over the Graph API and closes the media.
func (c *Call) Hangup(ctx context.Context) error {
	_ = c.Connection.Close()
	c.client.remove(c.ID)
	return c.client.api.terminate(ctx, c.ID)
}

// Client answers inbound WhatsApp calls via a webhook and the Graph API.
type Client struct {
	cfg    Config
	api    *api
	onCall func(*Call)

	mu    sync.Mutex
	calls map[string]*pionrtc.Connection
}

// NewClient builds a WhatsApp calling client.
func NewClient(cfg Config) (*Client, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	return &Client{
		cfg:   cfg,
		api:   &api{token: cfg.Token, phoneNumberID: cfg.PhoneNumberID, baseURL: cfg.BaseURL, http: &http.Client{}},
		calls: map[string]*pionrtc.Connection{},
	}, nil
}

// OnCall registers the handler invoked for each accepted inbound call. The
// handler runs on its own goroutine and owns the call until Hangup.
func (c *Client) OnCall(h func(*Call)) { c.onCall = h }

// Handler returns the HTTP handler for the WhatsApp webhook: GET verifies the
// subscription, POST delivers call events.
func (c *Client) Handler() http.Handler { return http.HandlerFunc(c.serveHTTP) }

func (c *Client) serveHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		c.verify(w, r)
	case http.MethodPost:
		c.webhook(w, r)
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

// verify handles the webhook verification handshake, echoing hub.challenge (a
// number) when the mode and verify token match.
func (c *Client) verify(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	if q.Get("hub.mode") != "subscribe" || q.Get("hub.verify_token") != c.cfg.VerifyToken {
		http.Error(w, "forbidden", http.StatusForbidden)
		return
	}
	challenge, err := strconv.Atoi(q.Get("hub.challenge"))
	if err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	_, _ = io.WriteString(w, strconv.Itoa(challenge))
}

// webhook validates the signature, parses the event, acks, and dispatches.
func (c *Client) webhook(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, maxWebhookBody))
	if err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	if !c.validSignature(body, r.Header.Get("X-Hub-Signature-256")) {
		http.Error(w, "invalid signature", http.StatusForbidden)
		return
	}
	var req webhookRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	// Ack promptly; call handling does network I/O on its own goroutines.
	w.WriteHeader(http.StatusOK)
	for _, entry := range req.Entry {
		for _, change := range entry.Changes {
			for _, call := range change.Value.Calls {
				switch call.Event {
				case "connect":
					go c.handleConnect(call)
				case actionTerminate:
					c.handleTerminate(call.ID)
				}
			}
		}
	}
}

// validSignature reports whether header carries a valid HMAC-SHA256 of body
// keyed by the app secret.
func (c *Client) validSignature(body []byte, header string) bool {
	if header == "" {
		return false
	}
	mac := hmac.New(sha256.New, []byte(c.cfg.AppSecret))
	mac.Write(body)
	expected := hex.EncodeToString(mac.Sum(nil))
	received := strings.TrimPrefix(header, "sha256=")
	return hmac.Equal([]byte(expected), []byte(received))
}

// handleConnect answers the offer, accepts the call, and invokes OnCall.
func (c *Client) handleConnect(call webhookCall) {
	ctx := context.Background()
	conn, err := pionrtc.NewConnection(pionrtc.WithICEServers(c.cfg.ICEServers...))
	if err != nil {
		slog.Error("whatsapp: new connection", "call", call.ID, "err", err)
		return
	}
	offer := webrtc.SessionDescription{Type: webrtc.SDPTypeOffer, SDP: call.Session.SDP}
	answer, err := conn.Answer(offer)
	if err != nil {
		slog.Error("whatsapp: answer offer", "call", call.ID, "err", err)
		_ = conn.Close()
		_ = c.api.reject(ctx, call.ID)
		return
	}

	sdp := filterSDP(answer.SDP)
	if err := c.api.answer(ctx, call.ID, "pre_accept", sdp, call.From); err != nil {
		slog.Error("whatsapp: pre_accept", "call", call.ID, "err", err)
		_ = conn.Close()
		return
	}
	if err := c.api.answer(ctx, call.ID, "accept", sdp, call.From); err != nil {
		slog.Error("whatsapp: accept", "call", call.ID, "err", err)
		_ = conn.Close()
		return
	}

	c.store(call.ID, conn)
	if c.onCall != nil {
		c.onCall(&Call{ID: call.ID, From: call.From, Connection: conn, client: c})
	}
}

func (c *Client) handleTerminate(callID string) {
	c.mu.Lock()
	conn := c.calls[callID]
	delete(c.calls, callID)
	c.mu.Unlock()
	if conn != nil {
		_ = conn.Close()
	}
}

func (c *Client) store(id string, conn *pionrtc.Connection) {
	c.mu.Lock()
	c.calls[id] = conn
	c.mu.Unlock()
	// Drop the call from the map once its media ends.
	go func() {
		<-conn.Done()
		c.remove(id)
	}()
}

func (c *Client) remove(id string) {
	c.mu.Lock()
	delete(c.calls, id)
	c.mu.Unlock()
}

// filterSDP drops non-SHA-256 DTLS fingerprints, which WhatsApp rejects.
func filterSDP(sdp string) string {
	lines := strings.Split(sdp, "\n")
	kept := lines[:0]
	for _, line := range lines {
		t := strings.TrimRight(line, "\r")
		if strings.HasPrefix(t, "a=fingerprint:") && !strings.HasPrefix(t, "a=fingerprint:sha-256") {
			continue
		}
		kept = append(kept, line)
	}
	return strings.Join(kept, "\n")
}

// webhookRequest is the subset of a WhatsApp calling webhook we read. The
// connect/terminate union is flattened: a connect carries Session, a terminate
// carries Status; both carry Event, ID and From.
type webhookRequest struct {
	Object string `json:"object"`
	Entry  []struct {
		Changes []struct {
			Field string `json:"field"`
			Value struct {
				Calls []webhookCall `json:"calls"`
			} `json:"value"`
		} `json:"changes"`
	} `json:"entry"`
}

type webhookCall struct {
	ID      string `json:"id"`
	From    string `json:"from"`
	Event   string `json:"event"`
	Status  string `json:"status,omitempty"`
	Session struct {
		SDP     string `json:"sdp"`
		SDPType string `json:"sdp_type"`
	} `json:"session"`
}
