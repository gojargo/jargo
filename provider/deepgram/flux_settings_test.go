package deepgram

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/service/settings"
	errs "github.com/gojargo/jargo/utils/errors"
)

// Flux takes a settings change three different ways, and which way depends on
// the field. These tests cover that division, the serialization Flux requires of
// the messages carrying a change, and the failures it reports in its own
// protocol rather than as an HTTP status.

// fluxFakeServer is a Flux endpoint that confirms the session, records every
// text message the client sends, and can be told to answer.
type fluxFakeServer struct {
	*httptest.Server
	mu       sync.Mutex
	received []map[string]any
	// confirm reports whether the session is confirmed on open. A server that
	// does not confirm is how an endpoint rejecting a connection setting
	// behaves: it goes silent rather than refusing.
	confirm  bool
	toClient chan []byte
}

func newFluxFakeServer(t *testing.T, confirm bool) *fluxFakeServer {
	t.Helper()
	f := &fluxFakeServer{confirm: confirm, toClient: make(chan []byte, 8)}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		if f.confirm {
			connected, _ := json.Marshal(map[string]any{"type": fluxMsgConnected})
			_ = c.Write(ctx, websocket.MessageText, connected)
		}

		go func() {
			for {
				select {
				case <-ctx.Done():
					return
				case payload := <-f.toClient:
					if c.Write(ctx, websocket.MessageText, payload) != nil {
						return
					}
				}
			}
		}()

		for {
			kind, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			if kind != websocket.MessageText {
				continue
			}
			var m map[string]any
			if json.Unmarshal(data, &m) != nil {
				continue
			}
			f.mu.Lock()
			f.received = append(f.received, m)
			f.mu.Unlock()
		}
	}))
	t.Cleanup(f.Close)
	return f
}

// send makes the server answer with one message.
func (f *fluxFakeServer) send(t *testing.T, m map[string]any) {
	t.Helper()
	payload, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	f.toClient <- payload
}

// configures is every Configure message the client has sent so far.
func (f *fluxFakeServer) configures() []map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []map[string]any
	for _, m := range f.received {
		if m["type"] == "Configure" {
			out = append(out, m)
		}
	}
	return out
}

// waitForConfigures blocks until the client has sent n Configure messages, or
// fails once it is clear it will not.
func (f *fluxFakeServer) waitForConfigures(t *testing.T, n int) []map[string]any {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		got := f.configures()
		if len(got) >= n {
			return got
		}
		if time.Now().After(deadline) {
			t.Fatalf("the client sent %d configure messages, want %d", len(got), n)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// fluxSession opens a session against a confirming server.
func fluxSession(t *testing.T) (*fluxFakeServer, *fluxConnector, *fluxStream) {
	t.Helper()
	srv := newFluxFakeServer(t, true)
	c := newFluxConnector(FluxConfig{
		APIKey:    "k",
		ListenURL: wsURL(srv.URL),
		Model:     defaultFluxModel,
	})
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := c.Connect(ctx, 16000)
	if err != nil {
		cancel()
		t.Fatalf("connect: %v", err)
	}
	t.Cleanup(func() { cancel(); _ = stream.Close() })
	flux, ok := stream.(*fluxStream)
	if !ok {
		t.Fatalf("connect returned a %T, want a flux session", stream)
	}
	return srv, c, flux
}

// update applies a settings delta the way the service does, then hands the
// connector what changed.
func update(t *testing.T, c *fluxConnector, delta *FluxSettings) bool {
	t.Helper()
	changed, err := settings.Apply(c.live, delta)
	if err != nil {
		t.Fatalf("apply: %v", err)
	}
	reopen, err := c.UpdateSettings(context.Background(), changed)
	if err != nil {
		t.Fatalf("update settings: %v", err)
	}
	return reopen
}

// With nothing in flight a Configure goes out at once.
func TestFluxConfigureSendsImmediatelyWhenIdle(t *testing.T) {
	srv, c, stream := fluxSession(t)

	update(t, c, &FluxSettings{EOTThreshold: settings.Set(0.8)})

	got := srv.waitForConfigures(t, 1)
	thresholds, _ := got[0]["thresholds"].(map[string]any)
	if thresholds["eot_threshold"] != 0.8 {
		t.Errorf("configure = %v, want eot_threshold 0.8", got[0])
	}
	stream.cfgMu.Lock()
	inFlight := stream.configureInFlight
	stream.cfgMu.Unlock()
	if !inFlight {
		t.Error("the configure message was not recorded as awaiting its ack")
	}
}

// Flux caps the un-acked Configure messages, so a burst is coalesced into one
// follow-up rather than replayed message by message. The follow-up carries the
// latest value of each field, not the value in force when it was queued.
func TestFluxConfigureCoalescesBurstWhileInFlight(t *testing.T) {
	srv, c, stream := fluxSession(t)

	update(t, c, &FluxSettings{EOTThreshold: settings.Set(0.1)})
	srv.waitForConfigures(t, 1)

	// These arrive while the first is still in flight.
	update(t, c, &FluxSettings{EagerEOTThreshold: settings.Set(0.2)})
	update(t, c, &FluxSettings{EagerEOTThreshold: settings.Set(0.9)})

	if got := srv.configures(); len(got) != 1 {
		t.Fatalf("sent %d configure messages while one was in flight, want 1", len(got))
	}

	// Acking the first flushes the coalesced update.
	srv.send(t, map[string]any{"type": fluxMsgConfigureSuccess})
	go func() { _, _ = stream.Recv() }()

	got := srv.waitForConfigures(t, 2)
	thresholds, _ := got[1]["thresholds"].(map[string]any)
	if thresholds["eager_eot_threshold"] != 0.9 {
		t.Errorf("flushed configure = %v, want eager_eot_threshold 0.9", got[1])
	}
	if _, ok := thresholds["eot_threshold"]; ok {
		t.Errorf("flushed configure = %v, want only the coalesced field", got[1])
	}
}

// A rejection still acks, so the coalesced update is not stranded behind it.
func TestFluxConfigureFailureFlushesPending(t *testing.T) {
	srv, c, stream := fluxSession(t)

	update(t, c, &FluxSettings{EOTThreshold: settings.Set(0.5)})
	srv.waitForConfigures(t, 1)
	update(t, c, &FluxSettings{EagerEOTThreshold: settings.Set(0.4)})

	srv.send(t, map[string]any{
		"type": fluxMsgConfigureFailure, "error_code": "bad", "description": "nope",
	})
	go func() { _, _ = stream.Recv() }()

	srv.waitForConfigures(t, 2)
}

// A Configure whose ack never arrives must not block later updates for good.
func TestFluxConfigureSupersedesStaleInFlight(t *testing.T) {
	srv, c, stream := fluxSession(t)

	update(t, c, &FluxSettings{EOTThreshold: settings.Set(0.5)})
	srv.waitForConfigures(t, 1)

	// Age the in-flight message past the point it is still trusted.
	stream.cfgMu.Lock()
	stream.configureSentAt = time.Now().Add(-2 * fluxConfigureAckTimeout)
	stream.cfgMu.Unlock()

	update(t, c, &FluxSettings{EagerEOTThreshold: settings.Set(0.4)})

	srv.waitForConfigures(t, 2)
	stream.cfgMu.Lock()
	pending := stream.configurePending
	stream.cfgMu.Unlock()
	if pending != nil {
		t.Errorf("pending = %v, want nothing left waiting", pending)
	}
}

// An ack with nothing in flight is what a stray or duplicate one looks like, and
// it must not send anything.
func TestFluxStrayAckIsIgnored(t *testing.T) {
	srv, _, stream := fluxSession(t)

	stream.onConfigureAcked()

	if got := srv.configures(); len(got) != 0 {
		t.Errorf("a stray ack sent %d configure messages, want none", len(got))
	}
	stream.cfgMu.Lock()
	defer stream.cfgMu.Unlock()
	if stream.configureInFlight || stream.configurePending != nil {
		t.Error("a stray ack left configure state behind")
	}
}

// The fields Flux only reads from the connection URL are applied by opening
// another connection, not by sending anything.
func TestFluxConnectionFieldReopensTheSession(t *testing.T) {
	srv, c, _ := fluxSession(t)

	if !update(t, c, &FluxSettings{Numerals: settings.Set(true)}) {
		t.Error("a connection-only field did not ask for a new session")
	}
	if got := srv.configures(); len(got) != 0 {
		t.Errorf("a connection-only field sent %d configure messages, want none", len(got))
	}
}

// The fields a live connection takes reach it without the connection dropping.
func TestFluxConfigureFieldDoesNotReopenTheSession(t *testing.T) {
	srv, c, _ := fluxSession(t)

	if update(t, c, &FluxSettings{EOTThreshold: settings.Set(0.9)}) {
		t.Error("a configure-able field asked for a new session")
	}
	srv.waitForConfigures(t, 1)
}

// The confidence floor is applied to results as they arrive, so a change to it
// needs neither a message nor a new session.
func TestFluxConfidenceFloorAppliesLocally(t *testing.T) {
	srv, c, stream := fluxSession(t)

	if update(t, c, &FluxSettings{MinConfidence: settings.Set(0.7)}) {
		t.Error("the confidence floor asked for a new session")
	}
	if got := srv.configures(); len(got) != 0 {
		t.Errorf("the confidence floor sent %d configure messages, want none", len(got))
	}
	floor := stream.minConfidence()
	if floor == nil || *floor != 0.7 {
		t.Errorf("floor = %v, want 0.7", floor)
	}
}

// Every declared setting belongs to exactly one of the three ways a change
// reaches Flux, or is one Flux has no parameter for at all. A field added
// without being classified would otherwise only show up as a log line at
// runtime.
func TestFluxEverySettingIsClassified(t *testing.T) {
	// Flux has no language parameter; language_hints covers multilingual input.
	unsupported := []string{"language"}
	classified := slices.Concat(fluxConfigureFields, fluxConnectionFields, fluxLocalFields, unsupported)

	for _, name := range fluxSettingNames() {
		if !slices.Contains(classified, name) {
			t.Errorf("setting %q is not classified", name)
		}
	}
	for _, name := range classified {
		if !slices.Contains(fluxSettingNames(), name) {
			t.Errorf("classified field %q is not a declared setting", name)
		}
	}
}

// A field belongs to one bucket only: two would make which way a change reaches
// Flux depend on the order the buckets happen to be checked in.
func TestFluxNoSettingIsClassifiedTwoWays(t *testing.T) {
	seen := map[string]string{}
	for bucket, fields := range map[string][]string{
		"configure": fluxConfigureFields, "connection": fluxConnectionFields, "local": fluxLocalFields,
	} {
		for _, f := range fields {
			if other, dup := seen[f]; dup {
				t.Errorf("setting %q is in both %s and %s", f, other, bucket)
			}
			seen[f] = bucket
		}
	}
}

// fluxSettingNames is every settings name FluxSettings declares, embedded ones
// included.
func fluxSettingNames() []string {
	var walk func(t reflect.Type) []string
	walk = func(t reflect.Type) []string {
		var out []string
		for f := range t.Fields() {
			if f.Anonymous {
				out = append(out, walk(f.Type)...)
				continue
			}
			if tag := f.Tag.Get("settings"); tag != "" && tag != "-" {
				out = append(out, tag)
			}
		}
		return out
	}
	return walk(reflect.TypeFor[FluxSettings]())
}

// A fatal error reports the code and the description Flux names, which is the
// only place it says what went wrong.
func TestFluxFatalErrorReportsCodeAndDescription(t *testing.T) {
	srv, _, stream := fluxSession(t)

	srv.send(t, map[string]any{
		"type":        fluxMsgError,
		"code":        "UNPARSABLE_CLIENT_MESSAGE",
		"description": "Could not deserialize last text message",
	})

	_, err := stream.Recv()
	if err == nil {
		t.Fatal("a fatal error did not end the session")
	}
	if !errors.Is(err, errFluxServer) {
		t.Errorf("err = %v, want a server error", err)
	}
	for _, want := range []string{"UNPARSABLE_CLIENT_MESSAGE", "Could not deserialize last text message"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("err = %q, want it to name %q", err, want)
		}
	}
}

// The code travels with the error, since it is what says whether a retry could
// ever clear the cause.
func TestFluxFatalErrorCarriesTheCode(t *testing.T) {
	srv, _, stream := fluxSession(t)

	srv.send(t, map[string]any{
		"type": fluxMsgError, "code": "UNPARSABLE_CLIENT_MESSAGE", "description": "Bad message",
	})

	_, err := stream.Recv()
	var fatal *fluxFatalError
	if !errors.As(err, &fatal) {
		t.Fatalf("err = %v, want a fatal error carrying a code", err)
	}
	if fatal.code != "UNPARSABLE_CLIENT_MESSAGE" {
		t.Errorf("code = %q", fatal.code)
	}
}

// Flux reports these over the connection rather than as an HTTP status, so
// without classifying them the service would keep looking healthy.
func TestFluxErrorCodesAreClassified(t *testing.T) {
	c := newFluxConnector(FluxConfig{APIKey: "k", Model: defaultFluxModel})

	got := c.ClassifyError(&fluxFatalError{code: "UNPARSABLE_CLIENT_MESSAGE", text: "bad message"})
	if got != errs.InvalidRequest {
		t.Errorf("category = %v, want %v", got, errs.InvalidRequest)
	}
	// A permanent category is what costs the service its usability.
	if !got.IsPermanent() {
		t.Error("the category is not permanent, so the service would keep taking work")
	}
}

// A code that is not listed defers to the shared classification, leaving
// recovery to the service's own reconnect handling.
func TestFluxUnrecognizedErrorCodeFallsBack(t *testing.T) {
	c := newFluxConnector(FluxConfig{APIKey: "k", Model: defaultFluxModel})

	if got := c.ClassifyError(&fluxFatalError{code: "SOMETHING_NEW", text: "boom"}); got != errs.Unset {
		t.Errorf("category = %v, want the shared classification to decide", got)
	}
}

// An endpoint that never confirms the session fails the connection instead of
// leaving it opening forever with audio piling up behind it.
func TestFluxConnectionWaitTimesOutInsteadOfHanging(t *testing.T) {
	srv := newFluxFakeServer(t, false)
	c := newFluxConnector(FluxConfig{APIKey: "k", ListenURL: wsURL(srv.URL), Model: defaultFluxModel})
	c.connectionTimeout = 50 * time.Millisecond

	_, err := c.Connect(t.Context(), 16000)
	if err == nil {
		t.Fatal("connect returned a session the endpoint never confirmed")
	}
	if !errors.Is(err, errFluxNotConfirmed) {
		t.Errorf("err = %v, want an unconfirmed connection", err)
	}
}

// A silent endpoint means the settings were rejected, not that the network was
// slow: Flux answers a connection setting it will not accept with silence.
func TestFluxUnconfirmedConnectionIsARejectedRequest(t *testing.T) {
	c := newFluxConnector(FluxConfig{APIKey: "k", Model: defaultFluxModel})

	got := c.ClassifyError(errFluxNotConfirmed)
	if got != errs.InvalidRequest {
		t.Errorf("category = %v, want %v", got, errs.InvalidRequest)
	}
	if !got.IsPermanent() {
		t.Error("the category is not permanent, so the service would keep retrying a rejected request")
	}
}

// A confirmed connection returns without waiting the bound out.
func TestFluxConnectionWaitReturnsOnceConfirmed(t *testing.T) {
	start := time.Now()
	fluxSession(t)
	if elapsed := time.Since(start); elapsed > fluxConnectionTimeout/2 {
		t.Errorf("connecting took %s, want it to return on the confirmation", elapsed)
	}
}
