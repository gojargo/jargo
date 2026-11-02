package rtc_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/opus"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/rtc"
	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
)

// testClient is a plain Pion peer standing in for a browser client.
type testClient struct {
	pc       *webrtc.PeerConnection
	track    *webrtc.TrackLocalStaticSample
	dc       *webrtc.DataChannel
	gotAudio chan struct{}
	once     sync.Once
}

func newTestClient(t *testing.T, withDataChannel bool) *testClient {
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
	track, err := webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus, ClockRate: 48000, Channels: 2},
		"audio", "client",
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pc.AddTrack(track); err != nil {
		t.Fatal(err)
	}
	c := &testClient{pc: pc, track: track, gotAudio: make(chan struct{})}
	if withDataChannel {
		if c.dc, err = pc.CreateDataChannel("chat", nil); err != nil {
			t.Fatal(err)
		}
	}
	pc.OnTrack(func(tr *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		go func() {
			for {
				pkt, _, err := tr.ReadRTP()
				if err != nil {
					return
				}
				if len(pkt.Payload) > 0 {
					c.once.Do(func() { close(c.gotAudio) })
				}
			}
		}()
	})
	return c
}

// offer creates the client's offer, with every local candidate gathered.
func (c *testClient) offer(t *testing.T) webrtc.SessionDescription {
	t.Helper()
	offer, err := c.pc.CreateOffer(nil)
	if err != nil {
		t.Fatal(err)
	}
	gathered := webrtc.GatheringCompletePromise(c.pc)
	if err := c.pc.SetLocalDescription(offer); err != nil {
		t.Fatal(err)
	}
	<-gathered
	return *c.pc.LocalDescription()
}

// accept applies the server's answer.
func (c *testClient) accept(t *testing.T, a *rtc.Answer) {
	t.Helper()
	answer := webrtc.SessionDescription{Type: webrtc.NewSDPType(a.Type), SDP: a.SDP}
	if err := c.pc.SetRemoteDescription(answer); err != nil {
		t.Fatal(err)
	}
}

// stream sends a tone from the client until stop is closed.
func (c *testClient) stream(t *testing.T, stop <-chan struct{}) {
	t.Helper()
	enc, err := opus.NewEncoder(opus.EncoderConfig{Channels: 1})
	if err != nil {
		t.Fatal(err)
	}
	go func() {
		ticker := time.NewTicker(opus.FrameDuration)
		defer ticker.Stop()
		phase := 0.0
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				packet, err := enc.Encode(sine(&phase))
				if err != nil {
					return
				}
				_ = c.track.WriteSample(media.Sample{Data: packet, Duration: opus.FrameDuration})
			}
		}
	}()
}

func request(o webrtc.SessionDescription) rtc.Request {
	return rtc.Request{SDP: o.SDP, Type: o.Type.String()}
}

// runEcho runs the echo pipeline over conn until the test ends.
func runEcho(t *testing.T, conn *rtc.Connection) {
	t.Helper()
	params := transport.DefaultParams()
	params.AudioInSampleRate = opus.SampleRate
	params.AudioOutSampleRate = opus.SampleRate
	tr := rtc.NewTransport(conn, params)
	task := pipeline.NewWorker(pipeline.New(tr.Input(), newEcho(), tr.Output()), pipeline.WorkerConfig{
		Params: pipeline.Params{AudioInSampleRate: opus.SampleRate, AudioOutSampleRate: opus.SampleRate},
	})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { _ = task.Run(ctx); close(done) }()
	t.Cleanup(func() {
		cancel()
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Error("task did not shut down")
		}
	})
}

func waitFor(t *testing.T, ch <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(15 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
	}
}

// TestRenegotiationReusesTheConnection covers an offer carrying a known pc_id:
// it is answered on the existing connection, and no second session starts.
func TestRenegotiationReusesTheConnection(t *testing.T) {
	h := rtc.NewRequestHandler(rtc.RequestHandlerConfig{})
	t.Cleanup(func() { _ = h.Close() })
	var sessions atomic.Int32
	onConn := func(*rtc.Connection) error { sessions.Add(1); return nil }

	client := newTestClient(t, false)
	first, err := h.HandleWebRequest(request(client.offer(t)), onConn)
	if err != nil {
		t.Fatal(err)
	}
	if first.PCID == "" || first.Type != "answer" {
		t.Fatalf("answer = %+v, want a pc_id and type answer", first)
	}
	client.accept(t, first)

	req := request(client.offer(t))
	req.PCID = first.PCID
	second, err := h.HandleWebRequest(req, onConn)
	if err != nil {
		t.Fatal(err)
	}
	if second.PCID != first.PCID {
		t.Fatalf("renegotiation pc_id = %q, want %q", second.PCID, first.PCID)
	}
	if n := sessions.Load(); n != 1 {
		t.Fatalf("connection callback ran %d times, want 1", n)
	}
}

// TestRestartKeepsTheSession covers restart_pc: the client replaces its peer
// connection, the server answers on a new one under a new pc_id, and the same
// session carries on. The reader parks on the old, dead track and resumes on
// the fresh one, so audio still makes the round trip.
func TestRestartKeepsTheSession(t *testing.T) {
	h := rtc.NewRequestHandler(rtc.RequestHandlerConfig{})
	t.Cleanup(func() { _ = h.Close() })
	var sessions atomic.Int32
	onConn := func(c *rtc.Connection) error {
		sessions.Add(1)
		runEcho(t, c)
		return nil
	}

	first := newTestClient(t, false)
	a1, err := h.HandleWebRequest(request(first.offer(t)), onConn)
	if err != nil {
		t.Fatal(err)
	}
	first.accept(t, a1)
	stop1 := make(chan struct{})
	first.stream(t, stop1)
	waitFor(t, first.gotAudio, "echoed audio before the restart")
	close(stop1)

	// The client restarts after losing its network path, so its old peer
	// connection never says goodbye: closing it would end the session instead,
	// since a peer closing its side closes this one.

	second := newTestClient(t, false)
	req := request(second.offer(t))
	req.PCID = a1.PCID
	req.RestartPC = true
	a2, err := h.HandleWebRequest(req, onConn)
	if err != nil {
		t.Fatal(err)
	}
	if a2.PCID == a1.PCID {
		t.Fatal("a restart kept the old pc_id")
	}
	second.accept(t, a2)
	stop2 := make(chan struct{})
	defer close(stop2)
	second.stream(t, stop2)
	waitFor(t, second.gotAudio, "echoed audio after the restart")

	if n := sessions.Load(); n != 1 {
		t.Fatalf("connection callback ran %d times, want 1", n)
	}
}

// TestPatchRequestAddsCandidates covers trickled candidates, including the
// empty end-of-candidates markers a browser sends per media line, which must
// pass through rather than fail to parse.
func TestPatchRequestAddsCandidates(t *testing.T) {
	h := rtc.NewRequestHandler(rtc.RequestHandlerConfig{})
	t.Cleanup(func() { _ = h.Close() })
	client := newTestClient(t, true)
	a, err := h.HandleWebRequest(request(client.offer(t)), func(*rtc.Connection) error { return nil })
	if err != nil {
		t.Fatal(err)
	}

	const hostCandidate = "candidate:0 1 UDP 2121990399 192.168.1.1 35628 typ host"
	err = h.HandlePatchRequest(rtc.PatchRequest{PCID: a.PCID, Candidates: []rtc.ICECandidate{
		{Candidate: hostCandidate, SDPMid: "0", SDPMLineIndex: 0},
		{Candidate: "", SDPMid: "0", SDPMLineIndex: 0},
		{Candidate: "", SDPMid: "1", SDPMLineIndex: 1},
	}})
	if err != nil {
		t.Fatalf("patch: %v", err)
	}

	err = h.HandlePatchRequest(rtc.PatchRequest{PCID: "unknown"})
	var reqErr *rtc.RequestError
	if !errors.As(err, &reqErr) || reqErr.Status != http.StatusNotFound {
		t.Fatalf("patch for an unknown pc_id = %v, want a 404 RequestError", err)
	}
}

// TestSingleConnectionModeRefusesASecondConnection covers single mode: once a
// connection exists, an offer without its pc_id, or with another, is refused.
func TestSingleConnectionModeRefusesASecondConnection(t *testing.T) {
	h := rtc.NewRequestHandler(rtc.RequestHandlerConfig{ConnectionMode: rtc.ConnectionModeSingle})
	t.Cleanup(func() { _ = h.Close() })
	noop := func(*rtc.Connection) error { return nil }

	first := newTestClient(t, false)
	if _, err := h.HandleWebRequest(request(first.offer(t)), noop); err != nil {
		t.Fatal(err)
	}

	for _, pcID := range []string{"", "someone-else"} {
		other := newTestClient(t, false)
		req := request(other.offer(t))
		req.PCID = pcID
		_, err := h.HandleWebRequest(req, noop)
		var reqErr *rtc.RequestError
		if !errors.As(err, &reqErr) || reqErr.Status != http.StatusBadRequest {
			t.Fatalf("offer with pc_id %q = %v, want a 400 RequestError", pcID, err)
		}
	}
}

// TestRequestAcceptsEitherSpellingOfRequestData covers the custom data a client
// sends with its offer, under request_data or requestData.
func TestRequestAcceptsEitherSpellingOfRequestData(t *testing.T) {
	for _, body := range []string{
		`{"sdp":"x","type":"offer","request_data":{"a":1}}`,
		`{"sdp":"x","type":"offer","requestData":{"a":1}}`,
	} {
		var r rtc.Request
		if err := json.Unmarshal([]byte(body), &r); err != nil {
			t.Fatal(err)
		}
		if string(r.RequestData) != `{"a":1}` {
			t.Fatalf("%s: request data = %s", body, r.RequestData)
		}
	}
}

func TestSDPMunging(t *testing.T) {
	sdp := "v=0\r\n" +
		"a=fingerprint:sha-256 AA\r\n" +
		"a=fingerprint:sha-384 BB\r\n" +
		"a=fingerprint:sha-512 CC\r\n" +
		"a=candidate:1 1 udp 1 192.168.1.100 5000 typ host\r\n" +
		"a=candidate:2 1 udp 1 10.0.0.5 5000 typ host\r\n" +
		"a=candidate:3 1 udp 1 192.168.1.100 6000 typ srflx raddr 0.0.0.0 rport 0\r\n"

	got := rtc.SDPMunging(sdp, "192.168.1.100")
	want := "v=0\r\n" +
		"a=fingerprint:sha-256 AA\r\n" +
		"a=candidate:1 1 udp 1 192.168.1.100 5000 typ host\r\n"
	if got != want {
		t.Fatalf("munged SDP:\n%q\nwant:\n%q", got, want)
	}

	// Without a host only the fingerprints are cleaned up.
	if got := rtc.SDPMunging(sdp, ""); strings.Contains(got, "sha-384") || !strings.Contains(got, "10.0.0.5") {
		t.Fatalf("munged SDP without host:\n%q", got)
	}
}

// connectedPair connects a server Connection to a test client through a request
// handler and returns both, once the client's data channel is open. before runs
// on the server connection as it is created, onMessage on every message the
// client receives.
func connectedPair(
	t *testing.T, before func(*rtc.Connection), onMessage func(webrtc.DataChannelMessage),
) (*rtc.Connection, *testClient) {
	t.Helper()
	h := rtc.NewRequestHandler(rtc.RequestHandlerConfig{})
	t.Cleanup(func() { _ = h.Close() })
	client := newTestClient(t, true)
	if onMessage != nil {
		client.dc.OnMessage(onMessage)
	}
	var conn *rtc.Connection
	a, err := h.HandleWebRequest(request(client.offer(t)), func(c *rtc.Connection) error {
		conn = c
		if before != nil {
			before(c)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	open := make(chan struct{})
	client.dc.OnOpen(func() { close(open) })
	client.accept(t, a)
	waitFor(t, open, "the data channel to open")
	return conn, client
}

// TestMessagesSentBeforeTheChannelOpensAreFlushedInOrder covers messages the
// bot sends while the data channel is not open yet: they are held and arrive in
// order once it opens.
func TestMessagesSentBeforeTheChannelOpensAreFlushedInOrder(t *testing.T) {
	var mu sync.Mutex
	var got []string
	all := make(chan struct{})
	connectedPair(t, func(c *rtc.Connection) {
		for _, typ := range []string{"user-mute-started", "metrics", "bot-ready"} {
			_ = c.SendAppMessage(map[string]string{"type": typ})
		}
	}, func(msg webrtc.DataChannelMessage) {
		var m struct{ Type string }
		_ = json.Unmarshal(msg.Data, &m)
		mu.Lock()
		defer mu.Unlock()
		got = append(got, m.Type)
		if len(got) == 3 {
			close(all)
		}
	})
	waitFor(t, all, "the queued messages")
	mu.Lock()
	defer mu.Unlock()
	if strings.Join(got, ",") != "user-mute-started,metrics,bot-ready" {
		t.Fatalf("messages = %v, want them in the order sent", got)
	}
}

// TestAppMessagesWaitForConnect covers messages from the client that arrive
// before the pipeline is ready: they are held until Connect, then delivered.
func TestAppMessagesWaitForConnect(t *testing.T) {
	received := make(chan string, 4)
	conn, client := connectedPair(t, func(c *rtc.Connection) {
		c.OnMessage(func(m []byte) { received <- string(m) })
	}, nil)
	if err := client.dc.SendText(`{"type":"hello"}`); err != nil {
		t.Fatal(err)
	}
	select {
	case m := <-received:
		t.Fatalf("message %s delivered before Connect", m)
	case <-time.After(300 * time.Millisecond):
	}
	conn.Connect()
	select {
	case m := <-received:
		if m != `{"type":"hello"}` {
			t.Fatalf("message = %s", m)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("held message not delivered after Connect")
	}
}

// TestTrackStatusSignaling covers the client switching one of its tracks off
// through a signaling message, which is not an application message.
func TestTrackStatusSignaling(t *testing.T) {
	status := make(chan [2]any, 1)
	appMessages := make(chan []byte, 1)
	conn, client := connectedPair(t, func(c *rtc.Connection) {
		c.OnTrackStatus(func(i int, enabled bool) { status <- [2]any{i, enabled} })
		c.OnMessage(func(m []byte) { appMessages <- m })
	}, nil)
	conn.Connect()
	//nolint:misspell // the wire value
	msg := `{"type":"signalling","message":{"type":"trackStatus","receiver_index":1,"enabled":false}}`
	if err := client.dc.SendText(msg); err != nil {
		t.Fatal(err)
	}
	select {
	case s := <-status:
		if s[0] != 1 || s[1] != false {
			t.Fatalf("track status = %v, want [1 false]", s)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("track status not reported")
	}
	select {
	case m := <-appMessages:
		t.Fatalf("signaling message delivered as an app message: %s", m)
	case <-time.After(200 * time.Millisecond):
	}
}
