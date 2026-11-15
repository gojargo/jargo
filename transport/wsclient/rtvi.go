package wsclient

import (
	"context"
	"encoding/json"
	"sync"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/pipeline/capabilities"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport/wsclient/rtviclient"
	"github.com/gojargo/jargo/transport/wsserver"
)

// EventBotReady fires when the bot answers the client-ready handshake with
// bot-ready. It carries nothing:
//
//	events.OnSignal(tr.Events(), wsclient.EventBotReady, func(ctx context.Context) { … })
const EventBotReady = "on_bot_ready"

// RTVITransport is a WebSocket client transport that speaks RTVI as a client to
// a bot.
//
// It connects to a bot that runs an RTVI server transport (the eval transport,
// say) and exchanges frames with it: the bot's RTVI server messages arrive as
// pipeline frames, and outgoing frames (audio, RTVI client messages built by the
// caller) are sent to the bot. It is the client-side counterpart to a bot
// running an RTVI server transport.
//
// It adds the client-ready and bot-ready handshake to Transport, and its
// serializer defaults to rtviclient.Serializer. It raises EventConnected,
// EventDisconnected and EventBotReady.
type RTVITransport struct {
	*Transport

	mu              sync.Mutex
	botReady        bool
	botCapabilities *capabilities.BotCapabilities
	// opening is what goes out first on every connection, ahead of the
	// connected event and the handshake.
	opening []wsserver.Message
}

// NewRTVITransport builds an RTVI client transport that connects to the bot's
// RTVI server transport at uri. The serializer defaults to rtviclient.Serializer
// when params does not set one.
func NewRTVITransport(uri string, params Params) *RTVITransport {
	if params.Serializer == nil {
		params.Serializer = rtviclient.New()
	}
	// An RTVI raw-audio message carries bare PCM with its own sample rate and
	// channel count, so a WAV header is never valid on this wire.
	params.AddWAVHeader = new(false)

	r := &RTVITransport{Transport: newTransport("RTVIClientTransport", uri, params)}
	r.self = r
	r.events.Register(EventBotReady, false)
	return r
}

// BotReady reports whether the bot has answered the handshake with bot-ready.
func (r *RTVITransport) BotReady() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.botReady
}

// BotCapabilities is what the bot reported it does in bot-ready. It is nil
// before bot-ready, and when the bot reported nothing readable, as bots
// predating the capabilities field do.
func (r *RTVITransport) BotCapabilities() *capabilities.BotCapabilities {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.botCapabilities
}

// SetOpeningMessages sets the messages sent first on every connection, ahead
// of the connected event and the client-ready handshake. A transport built on
// this one opens the session with messages of its own this way, such as the
// eval harness's eval-connect. Set them before the pipeline runs.
func (r *RTVITransport) SetOpeningMessages(messages ...wsserver.Message) {
	r.mu.Lock()
	r.opening = append([]wsserver.Message(nil), messages...)
	r.mu.Unlock()
}

// onConnected sends the opening messages, fires the connected event and starts
// the RTVI handshake.
func (r *RTVITransport) onConnected(ctx context.Context, conn *websocket.Conn) {
	r.mu.Lock()
	opening := r.opening
	r.mu.Unlock()
	for _, message := range opening {
		r.session.Send(ctx, message)
	}
	r.Transport.onConnected(ctx, conn)
	r.sendClientReady(ctx)
}

// onMessage watches for bot-ready to complete the handshake, then dispatches
// the message as usual.
func (r *RTVITransport) onMessage(ctx context.Context, conn *websocket.Conn, message []byte) {
	r.mu.Lock()
	ready := !r.botReady && isBotReady(message)
	if ready {
		r.botReady = true
		r.botCapabilities = botCapabilities(message)
	}
	r.mu.Unlock()
	if ready {
		r.events.Call(ctx, EventBotReady, r)
	}
	r.Transport.onMessage(ctx, conn, message)
}

// sendClientReady sends the RTVI client-ready message that starts the
// handshake.
func (r *RTVITransport) sendClientReady(ctx context.Context) {
	message := rtvi.Message{
		Label: rtvi.MessageLabel,
		Type:  rtvi.TypeClientReady,
		ID:    rtvi.TypeClientReady,
		Data: rtvi.ClientReadyData{
			Version: rtvi.ProtocolVersion,
			About:   rtvi.AboutClientData{Library: rtvi.LibraryName},
		},
	}
	data, err := json.Marshal(message)
	if err != nil {
		return
	}
	r.session.Send(ctx, wsserver.Message{Data: data})
}

// isBotReady reports whether message is an RTVI bot-ready message.
func isBotReady(message []byte) bool {
	var data struct {
		Label string `json:"label"`
		Type  string `json:"type"`
	}
	if err := json.Unmarshal(message, &data); err != nil {
		return false
	}
	return data.Label == rtvi.MessageLabel && data.Type == rtvi.TypeBotReady
}

// botCapabilities returns the capabilities a bot-ready message reports, or nil
// if it reports none.
func botCapabilities(botReady []byte) *capabilities.BotCapabilities {
	var message struct {
		Data json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(botReady, &message); err != nil {
		return nil
	}
	var data map[string]json.RawMessage
	if err := json.Unmarshal(message.Data, &data); err != nil {
		return nil
	}
	raw, ok := data["capabilities"]
	if !ok || string(raw) == "null" {
		return nil
	}
	var caps capabilities.BotCapabilities
	if err := json.Unmarshal(raw, &caps); err != nil {
		return nil
	}
	return &caps
}
