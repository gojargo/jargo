package speech

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/utils/events"
)

// TestRealtimeTTSConfigValidate pins which RealtimeTTSConfig fields the
// provider requires.
func TestRealtimeTTSConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing everything", Cfg: RealtimeTTSConfig{}, Valid: false},
		{Name: "key and region", Cfg: RealtimeTTSConfig{APIKey: "k", Region: "eastus"}, Valid: true},
		{
			Name:  "key and private endpoint",
			Cfg:   RealtimeTTSConfig{APIKey: "k", PrivateEndpoint: "https://my-resource.cognitiveservices.azure.com/"},
			Valid: true,
		},
		{Name: "key without region or private endpoint", Cfg: RealtimeTTSConfig{APIKey: "k"}, Valid: false},
		{Name: "region without key", Cfg: RealtimeTTSConfig{Region: "eastus"}, Valid: false},
	})
}

// TestNewRealtimeTTS checks the constructor returns a service under the label
// that identifies it in logs, metrics and traces.
func TestNewRealtimeTTS(t *testing.T) {
	providertest.Service(t, "AzureRealtimeTTS", NewRealtimeTTS(RealtimeTTSConfig{APIKey: "k", Region: "eastus"}))
}

// The synthesis callbacks are made from the goroutine reading the connection.
// They must reach a getter already waiting on an otherwise idle queue.

// fireDelay is how long the callback waits before firing, by which point the
// getter is parked.
const fireDelay = 100 * time.Millisecond

// awaitFromAnotherGoroutine parks get, fires the callback from another
// goroutine, and returns what get received, failing unless it arrived promptly.
func awaitFromAnotherGoroutine[T any](t *testing.T, get func(context.Context) (T, error), fire func()) T {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	start := time.Now()
	go func() {
		time.Sleep(fireDelay)
		fire()
	}()
	v, err := get(ctx)
	if err != nil {
		t.Fatalf("the getter was not woken: %v", err)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Fatalf("delivered after %s, want promptly", elapsed)
	}
	return v
}

func TestSynthesizingAudioWakesAnIdleGetter(t *testing.T) {
	_, s := newRealtimeTTS(RealtimeTTSConfig{APIKey: "test-key", Region: "eastus"})
	audio := []byte(strings.Repeat("\x00\x01", 256))

	item := awaitFromAnotherGoroutine(t, s.audioQueue.get, func() { s.handleSynthesizing(audio) })

	if string(item.audio) != string(audio) {
		t.Errorf("received %d bytes, want the %d sent", len(item.audio), len(audio))
	}
}

func TestCanceledErrorWakesAnIdleGetter(t *testing.T) {
	_, s := newRealtimeTTS(RealtimeTTSConfig{APIKey: "test-key", Region: "eastus"})

	item := awaitFromAnotherGoroutine(t, s.audioQueue.get, func() {
		s.handleCanceled(cancellationDetails{reason: cancellationError, errorDetails: "boom"})
	})

	if item.err == nil || !strings.Contains(item.err.Error(), "boom") {
		t.Errorf("received %+v, want the error with its details", item)
	}
}

func TestCompletionSentinelWakesAnIdleGetter(t *testing.T) {
	_, s := newRealtimeTTS(RealtimeTTSConfig{APIKey: "test-key", Region: "eastus"})

	item := awaitFromAnotherGoroutine(t, s.wordBoundaryQueue.get, func() { s.handleCompleted(synthesisResult{}) })

	if item != nil {
		t.Errorf("received %+v, want the completion sentinel", item)
	}
}

// ssmlOf builds a sentence's SSML the way each service does, for the same
// settings.
type ssmlOf func(t *testing.T, cfg RealtimeTTSConfig, text string) string

func ssmlServices() map[string]ssmlOf {
	return map[string]ssmlOf{
		"streaming": func(_ *testing.T, cfg RealtimeTTSConfig, text string) string {
			cfg.APIKey, cfg.Region = "test-key", "eastus"
			_, s := newRealtimeTTS(cfg)
			return constructSSML(text, s.live.ssmlParams())
		},
		"http": func(t *testing.T, cfg RealtimeTTSConfig, text string) string {
			t.Helper()
			s := &ttsSynthesizer{cfg: TTSConfig{
				APIKey: "test-key", Region: "eastus", Voice: cmpOr(cfg.Voice, defaultTTSVoice),
				Language: cfg.Language, ForceLocale: cfg.ForceLocale, Style: cfg.Style, Rate: cfg.Rate,
				Emphasis: cfg.Emphasis,
			}}
			return string(s.ssml(text))
		},
	}
}

func cmpOr(v, def string) string {
	if v == "" {
		return def
	}
	return v
}

// The default settings emit no <lang> element anywhere in the document.
func TestConstructSSMLDefaultHasNoLangElement(t *testing.T) {
	for name, ssml := range ssmlServices() {
		t.Run(name, func(t *testing.T) {
			got := ssml(t, RealtimeTTSConfig{}, "Hello there.")
			want := "<speak version='1.0' xml:lang='en-US' " +
				"xmlns='http://www.w3.org/2001/10/synthesis' " +
				"xmlns:mstts='http://www.w3.org/2001/mstts'>" +
				"<voice name='en-US-SaraNeural'>" +
				"<mstts:silence type='Sentenceboundary' value='20ms' />" +
				"Hello there." +
				"</voice></speak>"
			if strings.Contains(got, "<lang") {
				t.Errorf("ssml has a <lang> element:\n%s", got)
			}
			if got != want {
				t.Errorf("ssml =\n%s\nwant\n%s", got, want)
			}
		})
	}
}

// ForceLocale off explicitly matches the default byte for byte.
func TestConstructSSMLForceLocaleFalseMatchesDefault(t *testing.T) {
	for name, ssml := range ssmlServices() {
		t.Run(name, func(t *testing.T) {
			if a, b := ssml(t, RealtimeTTSConfig{}, "Hello there."),
				ssml(t, RealtimeTTSConfig{ForceLocale: false}, "Hello there."); a != b {
				t.Errorf("ssml differs:\n%s\n%s", a, b)
			}
		})
	}
}

// ForceLocale wraps the text in <lang xml:lang> inside <voice>.
func TestConstructSSMLForceLocaleWrapsTextInLangElement(t *testing.T) {
	for name, ssml := range ssmlServices() {
		t.Run(name, func(t *testing.T) {
			got := ssml(t, RealtimeTTSConfig{Language: "en-GB", ForceLocale: true}, "Hello there.")
			want := "<speak version='1.0' xml:lang='en-GB' " +
				"xmlns='http://www.w3.org/2001/10/synthesis' " +
				"xmlns:mstts='http://www.w3.org/2001/mstts'>" +
				"<voice name='en-US-SaraNeural'>" +
				"<mstts:silence type='Sentenceboundary' value='20ms' />" +
				"<lang xml:lang='en-GB'>" +
				"Hello there." +
				"</lang>" +
				"</voice></speak>"
			if got != want {
				t.Errorf("ssml =\n%s\nwant\n%s", got, want)
			}
		})
	}
}

// ForceLocale wraps the whole nested style, prosody and emphasis block, not
// just the raw text.
func TestConstructSSMLForceLocaleWrapsNestedStyleAndProsody(t *testing.T) {
	for name, ssml := range ssmlServices() {
		t.Run(name, func(t *testing.T) {
			got := ssml(t, RealtimeTTSConfig{
				Language: "en-GB", ForceLocale: true, Style: "cheerful", Rate: "slow", Emphasis: "strong",
			}, "Hello there.")
			want := "<speak version='1.0' xml:lang='en-GB' " +
				"xmlns='http://www.w3.org/2001/10/synthesis' " +
				"xmlns:mstts='http://www.w3.org/2001/mstts'>" +
				"<voice name='en-US-SaraNeural'>" +
				"<mstts:silence type='Sentenceboundary' value='20ms' />" +
				"<lang xml:lang='en-GB'>" +
				"<mstts:express-as style='cheerful'>" +
				"<prosody rate='slow'>" +
				"<emphasis level='strong'>" +
				"Hello there." +
				"</emphasis>" +
				"</prosody>" +
				"</mstts:express-as>" +
				"</lang>" +
				"</voice></speak>"
			if got != want {
				t.Errorf("ssml =\n%s\nwant\n%s", got, want)
			}
		})
	}
}

// The text is escaped, so it cannot break the markup.
func TestConstructSSMLEscapesTheText(t *testing.T) {
	got := constructSSML(`Tom & "Jerry" <it's>`, ssmlParams{Voice: "v", Language: "en-US"})
	if want := "Tom &amp; &quot;Jerry&quot; &lt;it&apos;s&gt;"; !strings.Contains(got, want) {
		t.Errorf("ssml =\n%s\nwant it to contain %s", got, want)
	}
}

func TestRealtimeTTSExpressionAdapterFollowsTheVoice(t *testing.T) {
	for _, tt := range []struct {
		voice   string
		adapter expressive.ExpressionAdapter
	}{
		{"en-US-Ava:DragonHDLatestNeural", HDExpressionAdapter{}},
		{"en-US-Ava:DragonHDOmniLatestNeural", HDExpressionAdapter{}},
		{"en-US-Tiana:DragonHDFlashLatestNeural", expressive.BaseExpressionAdapter{}},
		{"en-US-SaraNeural", expressive.BaseExpressionAdapter{}},
	} {
		_, s := newRealtimeTTS(RealtimeTTSConfig{APIKey: "key", Region: "eastus", Voice: tt.voice})
		if got := s.ExpressionAdapter(); reflect.TypeOf(got) != reflect.TypeOf(tt.adapter) {
			t.Errorf("%s: adapter %T, want %T", tt.voice, got, tt.adapter)
		}
	}
}

// Azure reports an HD voice's markers across boundaries, joined to the words
// around them; each is reported as a word of its own.
func TestMarkersAreSplitOutOfWordBoundaries(t *testing.T) {
	_, s := newRealtimeTTS(RealtimeTTSConfig{APIKey: "key", Region: "eastus", Voice: "en-US-Ava:DragonHDLatestNeural"})
	boundaries := []string{"[", "sighing] [", "sympathetic] It's", "is", "[", "slow] A"}

	var words []string
	for _, boundary := range boundaries {
		words = append(words, s.splitStyleMarkers(boundary)...)
	}

	if want := []string{"[sighing]", "[sympathetic]", "It's", "is", "[slow]", "A"}; !slices.Equal(words, want) {
		t.Errorf("words = %q, want %q", words, want)
	}
}

// queuedWords takes every word queued, stopping at the completion sentinel.
func queuedWords(t *testing.T, s *realtimeSynthesizer) []wordItem {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	var out []wordItem
	for {
		item, err := s.wordBoundaryQueue.get(ctx)
		if err != nil {
			t.Fatalf("no completion sentinel: %v", err)
		}
		if item == nil {
			return out
		}
		out = append(out, *item)
	}
}

// Punctuation is merged into the word before it, and each word is timed from
// the start of its sentence's audio.
func TestWordBoundariesMergePunctuation(t *testing.T) {
	_, s := newRealtimeTTS(RealtimeTTSConfig{APIKey: "key", Region: "eastus"})
	s.currentSentenceBaseOffset = 2
	for _, b := range []wordBoundaryEvent{
		{"Hello", 0}, {",", 4_000_000}, {"world", 5_000_000}, {".", 9_000_000},
	} {
		s.handleWordBoundary(b)
	}
	s.handleCompleted(synthesisResult{})

	got := queuedWords(t, s)
	want := []wordItem{{"Hello,", 2}, {"world.", 2.5}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("words = %+v, want %+v", got, want)
	}
}

// Characters of a language written without spaces are merged into a word until
// punctuation ends it.
func TestWordBoundariesMergeCJKCharacters(t *testing.T) {
	_, s := newRealtimeTTS(RealtimeTTSConfig{APIKey: "key", Region: "eastus", Language: "ja-JP"})
	s.cjk = true
	for _, b := range []wordBoundaryEvent{{"こん", 0}, {"にち", 1}, {"は", 2}, {"。", 3}, {"元気", 4}} {
		s.handleWordBoundary(b)
	}
	s.handleCompleted(synthesisResult{})

	var words []string
	for _, w := range queuedWords(t, s) {
		words = append(words, w.word)
	}
	if want := []string{"こんにちは。", "元気"}; !slices.Equal(words, want) {
		t.Errorf("words = %q, want %q", words, want)
	}
}

func TestSynthesisEndpoint(t *testing.T) {
	for _, tt := range []struct {
		name, region, privateEndpoint, want string
	}{
		{
			name: "a region", region: "eastus",
			want: "wss://eastus.tts.speech.microsoft.com/tts/cognitiveservices/websocket/v1",
		},
		{
			name: "a US Government region", region: "usgovarizona",
			want: "wss://usgovarizona.tts.speech.azure.us/tts/cognitiveservices/websocket/v1",
		},
		{
			name: "a China region", region: "chinaeast2",
			want: "wss://chinaeast2.tts.speech.azure.cn/tts/cognitiveservices/websocket/v1",
		},
		{
			name: "a private endpoint", privateEndpoint: "https://my-resource.cognitiveservices.azure.com/",
			want: "wss://my-resource.cognitiveservices.azure.com/tts/cognitiveservices/websocket/v1",
		},
		{
			name: "a private endpoint takes precedence", region: "eastus",
			privateEndpoint: "https://my-resource.cognitiveservices.azure.com",
			want:            "wss://my-resource.cognitiveservices.azure.com/tts/cognitiveservices/websocket/v1",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := synthesisEndpoint(tt.region, tt.privateEndpoint)
			if err != nil {
				t.Fatalf("endpoint: %v", err)
			}
			u, _ := url.Parse(got)
			if q := u.Query().Get(synthesisConnectionImplParam); q != "websocket" {
				t.Errorf("%s = %q, want websocket", synthesisConnectionImplParam, q)
			}
			u.RawQuery = ""
			if u.String() != tt.want {
				t.Errorf("endpoint = %s, want %s", u, tt.want)
			}
		})
	}
}

// fakeSynthesis is a local Azure synthesis server. For each ssml message it
// opens the turn, names the audio stream, and then does what turn says.
type fakeSynthesis struct {
	t *testing.T
	// turn answers one turn, given the request id and a way to send.
	turn func(requestID string, s *synthesisSender)

	url string

	mu          sync.Mutex
	connections int
	keys        []string
	received    []sttMessage
}

// synthesisSender sends the server's messages on one connection.
type synthesisSender struct {
	ctx  context.Context
	conn *websocket.Conn
}

func (s *synthesisSender) text(path, requestID, body string) {
	_ = s.conn.Write(s.ctx, websocket.MessageText,
		[]byte(messageHeaders(path, requestID, "application/json")+"\r\n"+body))
}

func (s *synthesisSender) audio(requestID, streamID string, pcm []byte) {
	headers := []byte("Path: audio\r\nX-RequestId: " + requestID + "\r\nX-StreamId: " + streamID + "\r\n")
	msg := make([]byte, 2+len(headers)+len(pcm))
	binary.BigEndian.PutUint16(msg, uint16(len(headers))) //nolint:gosec // a few dozen bytes
	copy(msg[2:], headers)
	copy(msg[2+len(headers):], pcm)
	_ = s.conn.Write(s.ctx, websocket.MessageBinary, msg)
}

func (s *synthesisSender) metadata(requestID string, items ...map[string]any) {
	body, _ := json.Marshal(map[string]any{"Metadata": items})
	s.text(pathAudioMetadata, requestID, string(body))
}

func wordBoundary(text string, offsetTicks int64) map[string]any {
	return map[string]any{
		"Type": metadataWordBoundary,
		"Data": map[string]any{
			"Offset": offsetTicks, "Duration": 1_000_000,
			"text": map[string]any{"Text": text, "Length": len(text), "BoundaryType": "WordBoundary"},
		},
	}
}

func sessionEnd(offsetTicks int64) map[string]any {
	return map[string]any{"Type": metadataSessionEnd, "Data": map[string]any{"Offset": offsetTicks}}
}

func newFakeSynthesis(t *testing.T, turn func(requestID string, s *synthesisSender)) *fakeSynthesis {
	t.Helper()
	f := &fakeSynthesis{t: t, turn: turn}
	srv := httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(srv.Close)
	f.url = srv.URL
	return f
}

func (f *fakeSynthesis) serve(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	f.connections++
	f.keys = append(f.keys, r.Header.Get("Ocp-Apim-Subscription-Key"))
	f.mu.Unlock()
	if r.URL.Path != synthesisPath || r.Header.Get("X-ConnectionId") == "" {
		f.t.Errorf("connected to %s with connection id %q", r.URL.Path, r.Header.Get("X-ConnectionId"))
	}
	c, err := websocket.Accept(w, r, nil)
	if err != nil {
		return
	}
	defer func() { _ = c.CloseNow() }()
	sender := &synthesisSender{ctx: r.Context(), conn: c}
	for {
		typ, data, err := c.Read(r.Context())
		if err != nil {
			return
		}
		msg, err := parseMessage(typ, data)
		if err != nil {
			f.t.Errorf("parsing a client message: %v", err)
			return
		}
		f.mu.Lock()
		f.received = append(f.received, msg)
		f.mu.Unlock()
		if msg.path != pathSSML {
			continue
		}
		requestID := msg.header("X-RequestId")
		sender.text(pathTurnStart, requestID, `{"context":{"serviceTag":"tag"}}`)
		sender.text(pathResponse, requestID, `{"context":{"serviceTag":"tag"},"audio":{"type":"inline","streamId":"S1"}}`)
		f.turn(requestID, sender)
	}
}

// paths is the path of every message received, in order.
func (f *fakeSynthesis) paths() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]string, 0, len(f.received))
	for _, m := range f.received {
		out = append(out, m.path)
	}
	return out
}

// message is the first message received on path.
func (f *fakeSynthesis) message(path string) (sttMessage, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, m := range f.received {
		if m.path == path {
			return m, true
		}
	}
	return sttMessage{}, false
}

// speakTurn answers a turn the way Azure does: word boundaries, the audio,
// and the end of the session, then turn.end.
func speakTurn(requestID string, s *synthesisSender) {
	s.metadata(requestID, wordBoundary("Hello", 500_000))
	s.metadata(requestID, wordBoundary(",", 4_000_000))
	s.audio(requestID, "S1", make([]byte, 4800))
	// Audio of another stream is not this turn's.
	s.audio(requestID, "S2", make([]byte, 4800))
	s.metadata(requestID, wordBoundary("world", 5_000_000))
	s.metadata(requestID, wordBoundary(".", 9_000_000))
	s.audio(requestID, "S1", make([]byte, 4800))
	s.metadata(requestID, sessionEnd(10_000_000))
	s.text(pathTurnEnd, requestID, "{}")
}

// step is one thing a test does to the running pipeline: queue a frame, or
// wait.
type step struct {
	frame frames.Frame
	sleep time.Duration
}

func queueFrame(f frames.Frame) step { return step{frame: f} }
func wait(d time.Duration) step      { return step{sleep: d} }

// runTest runs the service in a pipeline through the steps, ends the pipeline,
// and returns the frames that left it downstream and upstream.
func runTest(t *testing.T, svc *tts.Base, steps ...step) (down, up []frames.Frame) {
	t.Helper()
	var mu sync.Mutex
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		down = append(down, f)
		mu.Unlock()
	})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		up = append(up, f)
		mu.Unlock()
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	for _, s := range steps {
		if s.frame != nil {
			task.QueueFrame(s.frame)
		}
		if s.sleep > 0 {
			time.Sleep(s.sleep)
		}
	}
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("the pipeline did not finish")
	}
	mu.Lock()
	defer mu.Unlock()
	return slices.Clone(down), slices.Clone(up)
}

func testRealtime(f *fakeSynthesis) *tts.Base {
	return NewRealtimeTTS(RealtimeTTSConfig{APIKey: "test-key", PrivateEndpoint: f.url, SampleRate: 24000})
}

func audioBytes(fs []frames.Frame) int {
	n := 0
	for _, f := range fs {
		if a, ok := f.(*frames.TTSAudioRawFrame); ok {
			n += len(a.Audio)
		}
	}
	return n
}

func spokenWords(fs []frames.Frame) []string {
	var out []string
	for _, f := range fs {
		if w, ok := f.(*frames.TTSTextFrame); ok {
			out = append(out, w.Text)
		}
	}
	return out
}

func errorTexts(fs []frames.Frame) []string {
	var out []string
	for _, f := range fs {
		if e, ok := f.(*frames.ErrorFrame); ok {
			out = append(out, e.Error)
		}
	}
	return out
}

// A sentence is synthesized over the protocol: the audio of its stream is
// played as it arrives, and its words are reported with the punctuation merged
// in. The connection is reused for the next sentence.
func TestRealtimeTTSSpeaksOverTheSynthesisProtocol(t *testing.T) {
	f := newFakeSynthesis(t, speakTurn)

	down, _ := runTest(t, testRealtime(f),
		queueFrame(frames.NewTTSSpeakFrame("Hello, world.")), wait(300*time.Millisecond),
		queueFrame(frames.NewTTSSpeakFrame("Hello, world.")), wait(300*time.Millisecond))

	if n := audioBytes(down); n != 4*4800 {
		t.Errorf("audio = %d bytes, want %d: two turns of their own stream", n, 4*4800)
	}
	if got, want := spokenWords(down), []string{"Hello,", "world.", "Hello,", "world."}; !slices.Equal(got, want) {
		t.Errorf("words = %q, want %q", got, want)
	}
	f.mu.Lock()
	connections, keys := f.connections, slices.Clone(f.keys)
	f.mu.Unlock()
	if connections != 1 {
		t.Errorf("connections = %d, want the one reused", connections)
	}
	if !slices.Equal(keys, []string{"test-key"}) {
		t.Errorf("keys = %q, want the API key", keys)
	}
	if got := f.paths(); !slices.Equal(got[:3], []string{pathSpeechConfig, pathSynthesisContext, pathSSML}) {
		t.Errorf("messages = %q, want the connection described, then the synthesis, then the SSML", got)
	}

	msg, _ := f.message(pathSynthesisContext)
	var synthesisContext struct {
		Synthesis struct {
			Audio struct {
				MetadataOptions map[string]bool `json:"metadataOptions"` //nolint:tagliatelle // Azure wire field
				OutputFormat    string          `json:"outputFormat"`    //nolint:tagliatelle // Azure wire field
			} `json:"audio"`
		} `json:"synthesis"`
	}
	if err := json.Unmarshal(msg.body, &synthesisContext); err != nil {
		t.Fatalf("synthesis.context: %v", err)
	}
	audio := synthesisContext.Synthesis.Audio
	if audio.OutputFormat != "raw-24khz-16bit-mono-pcm" {
		t.Errorf("output format = %q", audio.OutputFormat)
	}
	for _, option := range []string{"wordBoundaryEnabled", "punctuationBoundaryEnabled", "sessionEndEnabled"} {
		if !audio.MetadataOptions[option] {
			t.Errorf("%s is off", option)
		}
	}
	ssml, _ := f.message(pathSSML)
	if ct := ssml.header("Content-Type"); ct != "application/ssml+xml" {
		t.Errorf("ssml content type = %q", ct)
	}
	if !strings.Contains(string(ssml.body), ">Hello, world.</voice>") {
		t.Errorf("ssml = %s", ssml.body)
	}
}

// The words of a turn's later sentences are timed after the earlier ones'
// audio, which the end of each session measures.
func TestRealtimeTTSTimesLaterSentencesAfterEarlierOnes(t *testing.T) {
	f := newFakeSynthesis(t, speakTurn)
	_, s := newRealtimeTTS(RealtimeTTSConfig{APIKey: "test-key", PrivateEndpoint: f.url})
	ctx := context.Background()
	s.Start(ctx)
	t.Cleanup(func() { _ = s.Close() })

	for range 2 {
		if err := s.RunTTS(ctx, "Hello, world.", "ctx-1", func(frames.Frame) error { return nil }); err != nil {
			t.Fatalf("run tts: %v", err)
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cumulativeAudioOffset != 2 {
		t.Errorf("offset after two sentences = %v, want 2s", s.cumulativeAudioOffset)
	}
}

// An interruption tells the service to stop the synthesis in progress.
func TestRealtimeTTSStopsSpeakingOnInterruption(t *testing.T) {
	f := newFakeSynthesis(t, func(requestID string, s *synthesisSender) {
		// Audio starts, and the turn never ends on its own.
		s.audio(requestID, "S1", make([]byte, 4800))
	})

	runTest(t, testRealtime(f),
		queueFrame(frames.NewLLMFullResponseStartFrame()),
		queueFrame(frames.NewLLMTextFrame("Hello there. ")),
		queueFrame(frames.NewLLMTextFrame("How are")),
		wait(200*time.Millisecond),
		queueFrame(frames.NewInterruptionFrame()),
		wait(100*time.Millisecond))

	control, ok := f.message(pathSynthesisControl)
	if !ok {
		t.Fatalf("messages = %q, want the synthesis stopped", f.paths())
	}
	ssml, _ := f.message(pathSSML)
	if control.header("X-RequestId") != ssml.header("X-RequestId") {
		t.Errorf("stopped request %q, want the one synthesizing %q",
			control.header("X-RequestId"), ssml.header("X-RequestId"))
	}
	if !strings.Contains(string(control.body), `"action":"stop"`) {
		t.Errorf("synthesis.control = %s, want a stop", control.body)
	}
}

// A synthesis the service rejects is canceled with the reason it gives, which
// is reported.
func TestRealtimeTTSReportsACanceledSynthesis(t *testing.T) {
	f := newFakeSynthesis(t, func(_ string, s *synthesisSender) {
		_ = s.conn.Close(websocket.StatusInvalidFramePayloadData, "Unsupported voice")
	})

	_, up := runTest(t, testRealtime(f), queueFrame(frames.NewTTSSpeakFrame("Hello.")), wait(300*time.Millisecond))

	errs := errorTexts(up)
	if !slices.ContainsFunc(errs, func(e string) bool {
		return strings.Contains(e, "Unsupported voice websocket error code: 1007")
	}) {
		t.Errorf("errors = %q, want the cancellation reported", errs)
	}
}
