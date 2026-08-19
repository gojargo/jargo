package whatsapp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	errs "github.com/gojargo/jargo/utils/errors"
)

// Sentinel errors for the Graph API calls.
//
//nolint:gochecknoglobals // sentinel errors
var (
	errStatus   = errors.New("whatsapp: unexpected status")
	errProtocol = errors.New("whatsapp: call action not successful")
)

// defaultBaseURL is the WhatsApp Cloud (Graph) API base for calling.
const defaultBaseURL = "https://graph.facebook.com/v23.0/"

// Shared request fields and the terminate action (also matched as a webhook event).
const (
	keyProduct = "messaging_product"
	keyAction  = "action"
	keyCallID  = "call_id"
	product    = "whatsapp"

	actionTerminate = "terminate"
)

// api is a minimal WhatsApp Cloud API client for the calling endpoint.
type api struct {
	token         string
	phoneNumberID string
	baseURL       string
	http          *http.Client
}

func (a *api) callsURL() string { return a.baseURL + a.phoneNumberID + "/calls" }

// answer sends a "pre_accept" or "accept" with the SDP answer. WhatsApp requires
// a call to be pre-accepted before it is accepted.
func (a *api) answer(ctx context.Context, callID, action, sdp, to string) error {
	body, err := a.post(ctx, map[string]any{
		keyProduct: product,
		"to":       to,
		keyAction:  action,
		keyCallID:  callID,
		"session":  map[string]any{"sdp": sdp, "sdp_type": "answer"},
	})
	if err != nil {
		return err
	}
	var out struct {
		Success bool `json:"success"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		return err
	}
	if !out.Success {
		return fmt.Errorf("%w: %s: %s", errProtocol, action, body)
	}
	return nil
}

// reject declines an incoming call that has not been accepted.
func (a *api) reject(ctx context.Context, callID string) error {
	_, err := a.post(ctx, map[string]any{
		keyProduct: product,
		keyAction:  "reject",
		keyCallID:  callID,
	})
	return err
}

// terminate ends an active call.
func (a *api) terminate(ctx context.Context, callID string) error {
	_, err := a.post(ctx, map[string]any{
		keyProduct: product,
		keyAction:  actionTerminate,
		keyCallID:  callID,
	})
	return err
}

func (a *api) post(ctx context.Context, payload map[string]any) ([]byte, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, a.callsURL(), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+a.token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := a.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<16))
	if resp.StatusCode != http.StatusOK {
		return nil, errs.NewHTTPStatusError(resp.StatusCode, fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, respBody))
	}
	return respBody, nil
}
