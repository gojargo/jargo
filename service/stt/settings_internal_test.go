package stt

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/settings"
)

// settingsStream stays open until the session is canceled, recording the audio
// it was given.
type settingsStream struct {
	mu    sync.Mutex
	audio [][]byte
	ctx   context.Context //nolint:containedctx // the session context, set on dial
}

func (s *settingsStream) Send(audio []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.audio = append(s.audio, append([]byte(nil), audio...))
	return nil
}

func (s *settingsStream) Recv() ([]Result, error) {
	<-s.ctx.Done()
	return nil, s.ctx.Err()
}

func (s *settingsStream) Close() error { return nil }

func (s *settingsStream) sent() [][]byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([][]byte(nil), s.audio...)
}

// settingsConnector holds settings the way a provider does, and records what it
// was told changed.
type settingsConnector struct {
	mu      sync.Mutex
	store   STTSettings
	changed []settings.Changed
	reopen  bool
	dials   int
	streams []*settingsStream

	// dialing and holdDial let a test stop a dial partway, so it can act while
	// the session is being replaced.
	dialing  chan struct{}
	holdDial chan struct{}
}

// STTSettings stands in for a provider's own settings type.
type STTSettings = settings.STT

func (c *settingsConnector) Connect(ctx context.Context, _ int) (Stream, error) {
	c.mu.Lock()
	c.dials++
	dials := c.dials
	dialing, hold := c.dialing, c.holdDial
	st := &settingsStream{ctx: ctx}
	c.streams = append(c.streams, st)
	c.mu.Unlock()

	if dials > 1 && dialing != nil {
		dialing <- struct{}{}
		<-hold
	}
	return st, nil
}

func (c *settingsConnector) Settings() any { return &c.store }

func (c *settingsConnector) UpdateSettings(_ context.Context, changed settings.Changed) (bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.changed = append(c.changed, changed)
	return c.reopen, nil
}

func (c *settingsConnector) updates() []settings.Changed {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]settings.Changed(nil), c.changed...)
}

func (c *settingsConnector) dialCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.dials
}

func runSettingsService(t *testing.T, svc *StreamService) (*pipeline.Worker, func()) {
	t.Helper()
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	return task, func() {
		task.StopWhenDone()
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Error("pipeline did not shut down")
		}
	}
}

func waitUntil(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.After(3 * time.Second)
	tick := time.NewTicker(5 * time.Millisecond)
	defer tick.Stop()
	for {
		if cond() {
			return
		}
		select {
		case <-tick.C:
		case <-deadline:
			t.Fatalf("timed out waiting for %s", what)
		}
	}
}

func TestSettingsUpdateReachesTheProvider(t *testing.T) {
	t.Parallel()

	conn := &settingsConnector{store: STTSettings{Language: settings.Set("en")}}
	svc := NewStream("FakeSTT", conn, 16000)
	task, stop := runSettingsService(t, svc)
	defer stop()
	waitUntil(t, "the session to open", func() bool { return conn.dialCount() == 1 })

	task.QueueFrame(frames.NewSTTUpdateSettingsFrame(&STTSettings{Language: settings.Set("fr")}))
	waitUntil(t, "the update to reach the provider", func() bool { return len(conn.updates()) == 1 })

	changed := conn.updates()[0]
	if !changed.Has("language") {
		t.Errorf("changed = %v, want language reported", changed)
	}
	if changed["language"] != "en" {
		t.Errorf("previous language = %v, want en", changed["language"])
	}
	conn.mu.Lock()
	got, _ := conn.store.Language.Value()
	conn.mu.Unlock()
	if got != "fr" {
		t.Errorf("stored language = %q, want fr", got)
	}
}

func TestSettingsUpdateSentAsPlainDataIsApplied(t *testing.T) {
	t.Parallel()

	conn := &settingsConnector{store: STTSettings{Language: settings.Set("en")}}
	svc := NewStream("FakeSTT", conn, 16000)
	task, stop := runSettingsService(t, svc)
	defer stop()
	waitUntil(t, "the session to open", func() bool { return conn.dialCount() == 1 })

	f := frames.NewSTTUpdateSettingsFrame(nil)
	f.Settings = map[string]any{"language": "de"}
	task.QueueFrame(f)
	waitUntil(t, "the update to reach the provider", func() bool { return len(conn.updates()) == 1 })

	conn.mu.Lock()
	got, _ := conn.store.Language.Value()
	conn.mu.Unlock()
	if got != "de" {
		t.Errorf("stored language = %q, want de", got)
	}
}

// Re-sending a service what it already holds is not a change, so the provider is
// not told to act on one.
func TestSettingsUpdateThatChangesNothingIsNotReported(t *testing.T) {
	t.Parallel()

	conn := &settingsConnector{store: STTSettings{Language: settings.Set("en")}}
	svc := NewStream("FakeSTT", conn, 16000)
	task, stop := runSettingsService(t, svc)
	defer stop()
	waitUntil(t, "the session to open", func() bool { return conn.dialCount() == 1 })

	task.QueueFrame(frames.NewSTTUpdateSettingsFrame(&STTSettings{Language: settings.Set("en")}))
	// Follow it with a real change, so the wait has something to land on and the
	// non-change has had its chance to be reported first.
	task.QueueFrame(frames.NewSTTUpdateSettingsFrame(&STTSettings{Language: settings.Set("it")}))
	waitUntil(t, "the second update", func() bool { return len(conn.updates()) == 1 })

	if got := conn.updates(); len(got) != 1 || !got[0].Has("language") || got[0]["language"] != "en" {
		t.Errorf("updates = %v, want only the change from en", got)
	}
}

// An update naming another service is left for that one.
func TestSettingsUpdateForAnotherServiceIsNotApplied(t *testing.T) {
	t.Parallel()

	conn := &settingsConnector{store: STTSettings{Language: settings.Set("en")}}
	svc := NewStream("FakeSTT", conn, 16000)
	task, stop := runSettingsService(t, svc)
	defer stop()
	waitUntil(t, "the session to open", func() bool { return conn.dialCount() == 1 })

	f := frames.NewSTTUpdateSettingsFrame(&STTSettings{Language: settings.Set("fr")})
	f.Service = otherService("SomeOtherSTT")
	task.QueueFrame(f)
	time.Sleep(100 * time.Millisecond)

	if got := conn.updates(); len(got) != 0 {
		t.Errorf("updates = %v, want none: the frame named another service", got)
	}
	conn.mu.Lock()
	got, _ := conn.store.Language.Value()
	conn.mu.Unlock()
	if got != "en" {
		t.Errorf("stored language = %q, want en", got)
	}
}

// otherService stands in for a different service the frame could name.
type otherService string

func (o otherService) Name() string { return string(o) }

// A provider that cannot take the change on a running session gets a new one.
func TestSettingsUpdateReopensTheSessionWhenTheProviderAsks(t *testing.T) {
	t.Parallel()

	conn := &settingsConnector{store: STTSettings{Language: settings.Set("en")}, reopen: true}
	svc := NewStream("FakeSTT", conn, 16000)
	task, stop := runSettingsService(t, svc)
	defer stop()
	waitUntil(t, "the session to open", func() bool { return conn.dialCount() == 1 })

	task.QueueFrame(frames.NewSTTUpdateSettingsFrame(&STTSettings{Language: settings.Set("fr")}))
	waitUntil(t, "the session to be reopened", func() bool { return conn.dialCount() == 2 })
}

// Reopening mid-sentence would drop what the user is saying, so it waits for the
// turn to end. The words lost would be the very ones the change was meant to
// transcribe better.
func TestSettingsReopenWaitsForTheUserToStopSpeaking(t *testing.T) {
	t.Parallel()

	conn := &settingsConnector{store: STTSettings{Language: settings.Set("en")}, reopen: true}
	svc := NewStream("FakeSTT", conn, 16000)
	task, stop := runSettingsService(t, svc)
	defer stop()
	waitUntil(t, "the session to open", func() bool { return conn.dialCount() == 1 })

	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.5, time.Now()))
	waitUntil(t, "the speech to register", func() bool {
		svc.mu.Lock()
		defer svc.mu.Unlock()
		return !svc.canReopen
	})

	task.QueueFrame(frames.NewSTTUpdateSettingsFrame(&STTSettings{Language: settings.Set("fr")}))
	waitUntil(t, "the update to reach the provider", func() bool { return len(conn.updates()) == 1 })
	time.Sleep(100 * time.Millisecond)

	if got := conn.dialCount(); got != 1 {
		t.Fatalf("dialed %d times, want 1: the session was replaced mid-sentence", got)
	}

	task.QueueFrame(frames.NewUserStoppedSpeakingFrame())
	waitUntil(t, "the deferred reopen", func() bool { return conn.dialCount() == 2 })
}

// Audio that arrives while the session is being replaced is held and sent on, so
// the words spoken across the gap still reach the provider. The buffer is
// cleared when the reopen starts, so only what arrives during it is replayed.
func TestAudioIsHeldAndReplayedAcrossAReopen(t *testing.T) {
	t.Parallel()

	conn := &settingsConnector{
		store:    STTSettings{Language: settings.Set("en")},
		reopen:   true,
		dialing:  make(chan struct{}, 1),
		holdDial: make(chan struct{}),
	}
	svc := NewStream("FakeSTT", conn, 16000)
	task, stop := runSettingsService(t, svc)
	defer stop()
	waitUntil(t, "the session to open", func() bool { return conn.dialCount() == 1 })

	// Ask for a reopen. The connector holds the new dial open, so the session is
	// being replaced for as long as the test needs.
	task.QueueFrame(frames.NewSTTUpdateSettingsFrame(&STTSettings{Language: settings.Set("fr")}))
	select {
	case <-conn.dialing:
	case <-time.After(3 * time.Second):
		t.Fatal("the session was never reopened")
	}

	// Audio arrives on the system path, so it lands while the frame goroutine is
	// partway through replacing the session.
	svc.send([]byte{1, 2, 3, 4})
	svc.send([]byte{5, 6, 7, 8})

	svc.mu.Lock()
	held := len(svc.held)
	svc.mu.Unlock()
	if held != 2 {
		t.Fatalf("held %d chunks, want 2: audio must not go to a session being replaced", held)
	}
	conn.mu.Lock()
	first := conn.streams[0]
	conn.mu.Unlock()
	if got := len(first.sent()); got != 0 {
		t.Errorf("the session being replaced was sent %d chunks, want 0", got)
	}

	close(conn.holdDial)

	waitUntil(t, "the held audio to reach the new session", func() bool {
		conn.mu.Lock()
		streams := append([]*settingsStream(nil), conn.streams...)
		conn.mu.Unlock()
		return len(streams) == 2 && len(streams[1].sent()) == 2
	})

	svc.mu.Lock()
	remaining := len(svc.held)
	svc.mu.Unlock()
	if remaining != 0 {
		t.Errorf("%d chunks still held after the reopen, want none", remaining)
	}
}

// plainConnector holds no settings at all, the way most providers do today.
type plainConnector struct {
	mu    sync.Mutex
	dials int
}

func (c *plainConnector) Connect(ctx context.Context, _ int) (Stream, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.dials++
	return &settingsStream{ctx: ctx}, nil
}

func (c *plainConnector) dialCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.dials
}

// A provider with no settings at all is unaffected by an update passing through.
func TestSettingsUpdateForAProviderWithNoSettings(t *testing.T) {
	t.Parallel()

	conn := &plainConnector{}
	svc := NewStream("FakeSTT", conn, 16000)
	task, stop := runSettingsService(t, svc)
	defer stop()
	waitUntil(t, "the session to open", func() bool { return conn.dialCount() == 1 })

	task.QueueFrame(frames.NewSTTUpdateSettingsFrame(&STTSettings{Language: settings.Set("fr")}))
	time.Sleep(100 * time.Millisecond)

	if got := conn.dialCount(); got != 1 {
		t.Errorf("dialed %d times, want 1: nothing should have reopened", got)
	}
}
