package smallest

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/coder/websocket"
)

func TestSTTValidate(t *testing.T) {
	if err := (STTConfig{APIKey: "k"}).Validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	if err := (STTConfig{}).Validate(); err == nil {
		t.Fatal("config without APIKey should be rejected")
	}
}

func TestSTTConnectAndRecv(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()
		if _, _, err := c.Read(ctx); err != nil { // consume the binary audio
			return
		}
		interim, _ := json.Marshal(map[string]any{"is_final": false, "transcript": "hel"})
		_ = c.Write(ctx, websocket.MessageText, interim)
		final, _ := json.Marshal(map[string]any{"is_final": true, "transcript": "hello", "language": "en"})
		_ = c.Write(ctx, websocket.MessageText, final)
	}))
	defer srv.Close()

	conn := &connector{cfg: STTConfig{APIKey: "k", BaseURL: wsURL(srv.URL), Encoding: defaultSTTEncoding}}
	stream, err := conn.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	if err := stream.Send([]byte{1, 2, 3, 4}); err != nil {
		t.Fatalf("Send: %v", err)
	}

	res, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv interim: %v", err)
	}
	if len(res) != 1 || res[0].Text != "hel" || res[0].Final {
		t.Fatalf("interim result = %+v", res)
	}

	res, err = stream.Recv()
	if err != nil {
		t.Fatalf("Recv final: %v", err)
	}
	if len(res) != 1 || res[0].Text != "hello" || !res[0].Final || !res[0].EndOfTurn || res[0].Language != "en" {
		t.Fatalf("final result = %+v", res)
	}
}
