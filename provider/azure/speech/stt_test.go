package speech

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/language"
)

// TestSTTConfigValidate pins which STTConfig fields the provider requires.
func TestSTTConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing everything", Cfg: STTConfig{}, Valid: false},
		{Name: "key and region", Cfg: STTConfig{APIKey: "k", Region: "eastus"}, Valid: true},
		{Name: "key and host", Cfg: STTConfig{APIKey: "k", Host: "wss://private.example"}, Valid: true},
		{Name: "key without region or host", Cfg: STTConfig{APIKey: "k"}, Valid: false},
		{Name: "region without key", Cfg: STTConfig{Region: "eastus"}, Valid: false},
		{Name: "supported profanity", Cfg: STTConfig{APIKey: "k", Region: "eastus", Profanity: "raw"}, Valid: true},
		{Name: "unsupported profanity", Cfg: STTConfig{APIKey: "k", Region: "eastus", Profanity: "bleep"}, Valid: false},
	})
}

// TestNewSTT checks the constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNewSTT(t *testing.T) {
	providertest.Service(t, "AzureSTT", NewSTT(STTConfig{APIKey: "k", Region: "eastus"}))
}

// TestAzureLocale checks base languages are expanded to the locale Azure wants,
// while an already-qualified or unknown code passes through.
func TestAzureLocale(t *testing.T) {
	cases := map[language.Language]string{
		"":                      "en-US", // unset falls back
		language.English:        "en-US",
		language.French:         "fr-FR",
		language.Chinese:        "zh-CN",
		language.Norwegian:      "nb-NO", // Azure names Norwegian Bokmal
		language.FrenchCA:       "fr-CA", // already qualified
		language.SpanishMX:      "es-MX",
		language.Language("xx"): "xx", // unknown passes through
	}
	for in, want := range cases {
		if got := azureLocale(in); got != want {
			t.Errorf("azureLocale(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestSTTEndpoint checks the recognition parameters, which Azure takes as query
// parameters rather than in a configuration message.
func TestSTTEndpoint(t *testing.T) {
	t.Run("defaults", func(t *testing.T) {
		c := &sttConnector{cfg: STTConfig{APIKey: "k", Region: "eastus"}}
		raw := c.endpoint()
		if !strings.HasPrefix(raw, "wss://eastus.stt.speech.microsoft.com"+sttPath+"?") {
			t.Errorf("endpoint = %q, want the region host and recognition path", raw)
		}
		q := parseQuery(t, raw)
		if got := q.Get("language"); got != "en-US" {
			t.Errorf("language = %q, want en-US", got)
		}
		if got := q.Get("format"); got != "simple" {
			t.Errorf("format = %q, want simple", got)
		}
		if q.Has("profanity") || q.Has("cid") {
			t.Errorf("optional parameters present when unset: %v", q)
		}
	})

	t.Run("host overrides the region", func(t *testing.T) {
		c := &sttConnector{cfg: STTConfig{APIKey: "k", Region: "eastus", Host: "wss://private.example/"}}
		if raw := c.endpoint(); !strings.HasPrefix(raw, "wss://private.example"+sttPath+"?") {
			t.Errorf("endpoint = %q, want the configured host with no double slash", raw)
		}
	})

	t.Run("custom model replaces the language", func(t *testing.T) {
		c := &sttConnector{cfg: STTConfig{
			APIKey:     "k",
			Region:     "eastus",
			Language:   language.French,
			EndpointID: "dep-1",
			Profanity:  "Raw",
		}}
		q := parseQuery(t, c.endpoint())
		if got := q.Get("cid"); got != "dep-1" {
			t.Errorf("cid = %q, want the deployment id", got)
		}
		if q.Has("language") {
			t.Error("language is present alongside a custom model, which derives it from the model")
		}
		if got := q.Get("profanity"); got != "raw" {
			t.Errorf("profanity = %q, want it lowercased", got)
		}
	})
}

// TestWavHeader checks the RIFF header that opens each turn describes 16-bit
// mono PCM at the session rate, with both lengths left at zero because the
// stream's length is not known up front.
func TestWavHeader(t *testing.T) {
	h := wavHeader(16000)
	if len(h) != wavHeaderLen {
		t.Fatalf("header is %d bytes, want %d", len(h), wavHeaderLen)
	}
	if string(h[0:4]) != "RIFF" || string(h[8:16]) != "WAVEfmt " || string(h[36:40]) != "data" {
		t.Errorf("chunk identifiers = %q / %q / %q", h[0:4], h[8:16], h[36:40])
	}
	if got := binary.LittleEndian.Uint32(h[4:]); got != 0 {
		t.Errorf("riff length = %d, want 0 for a stream of unknown length", got)
	}
	if got := binary.LittleEndian.Uint32(h[40:]); got != 0 {
		t.Errorf("data length = %d, want 0 for a stream of unknown length", got)
	}
	if got := binary.LittleEndian.Uint32(h[16:]); got != 16 {
		t.Errorf("fmt chunk size = %d, want 16", got)
	}
	if got := binary.LittleEndian.Uint16(h[20:]); got != pcmFormatTag {
		t.Errorf("format tag = %d, want PCM (%d)", got, pcmFormatTag)
	}
	if got := binary.LittleEndian.Uint16(h[22:]); got != 1 {
		t.Errorf("channels = %d, want mono", got)
	}
	if got := binary.LittleEndian.Uint32(h[24:]); got != 16000 {
		t.Errorf("sample rate = %d, want 16000", got)
	}
	if got := binary.LittleEndian.Uint32(h[28:]); got != 32000 {
		t.Errorf("byte rate = %d, want 32000 (16000 * 2)", got)
	}
	if got := binary.LittleEndian.Uint16(h[32:]); got != 2 {
		t.Errorf("block align = %d, want 2", got)
	}
	if got := binary.LittleEndian.Uint16(h[34:]); got != bitsPerSample {
		t.Errorf("bits per sample = %d, want %d", got, bitsPerSample)
	}
}

// TestParseMessage checks both framings Azure uses: a text message is headers,
// a blank line, then the body; a binary one prefixes the headers with their
// length.
func TestParseMessage(t *testing.T) {
	t.Run("text", func(t *testing.T) {
		raw := "X-RequestId: abc\r\nPath: speech.phrase\r\nContent-Type: application/json\r\n\r\n{\"DisplayText\":\"hi\"}"
		msg, err := parseMessage(websocket.MessageText, []byte(raw))
		if err != nil {
			t.Fatalf("parseMessage: %v", err)
		}
		if msg.path != "speech.phrase" {
			t.Errorf("path = %q, want speech.phrase", msg.path)
		}
		if string(msg.body) != `{"DisplayText":"hi"}` {
			t.Errorf("body = %q, want the JSON after the blank line", msg.body)
		}
	})

	t.Run("binary", func(t *testing.T) {
		headers := "Path: audio\r\nX-RequestId: abc\r\n"
		body := []byte{1, 2, 3}
		frame := make([]byte, 2+len(headers)+len(body))
		binary.BigEndian.PutUint16(frame, uint16(len(headers)))
		copy(frame[2:], headers)
		copy(frame[2+len(headers):], body)

		msg, err := parseMessage(websocket.MessageBinary, frame)
		if err != nil {
			t.Fatalf("parseMessage: %v", err)
		}
		if msg.path != "audio" {
			t.Errorf("path = %q, want audio", msg.path)
		}
		if string(msg.body) != "\x01\x02\x03" {
			t.Errorf("body = %q, want the payload after the headers", msg.body)
		}
	})

	t.Run("malformed", func(t *testing.T) {
		if _, err := parseMessage(websocket.MessageText, []byte("Path: x\r\nno blank line")); err == nil {
			t.Error("text message without a header separator was accepted")
		}
		if _, err := parseMessage(websocket.MessageBinary, []byte{0}); err == nil {
			t.Error("binary message too short for a header length was accepted")
		}
		short := []byte{0, 200, 'a'}
		if _, err := parseMessage(websocket.MessageBinary, short); err == nil {
			t.Error("binary message shorter than its header length was accepted")
		}
	})
}

// TestHeaderValue checks header lookup is case-insensitive, as Azure's own
// parser is.
func TestHeaderValue(t *testing.T) {
	headers := []byte("path: speech.hypothesis\r\nX-RequestId:  abc  ")
	if got := headerValue(headers, "path"); got != "speech.hypothesis" {
		t.Errorf("path = %q, want speech.hypothesis", got)
	}
	if got := headerValue(headers, "x-requestid"); got != "abc" {
		t.Errorf("x-requestid = %q, want the trimmed value", got)
	}
	if got := headerValue(headers, "absent"); got != "" {
		t.Errorf("absent header = %q, want empty", got)
	}
}

// azureSession is a fake Azure recognition endpoint. It records what the client
// sends and replays scripted messages.
type azureSession struct {
	url       string
	received  <-chan sentMessage
	handshake func() http.Header
	reply     chan<- string
}

// sentMessage is one framed message the fake endpoint received.
type sentMessage struct {
	path      string
	requestID string
	body      []byte
}

// newAzureSession starts the fake endpoint. Messages written to the returned
// reply channel are sent to the client verbatim.
func newAzureSession(t *testing.T) azureSession {
	t.Helper()
	received := make(chan sentMessage, 16)
	reply := make(chan string, 16)
	headerCh := make(chan http.Header, 1)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case headerCh <- r.Header.Clone():
		default:
		}
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		go func() {
			for msg := range reply {
				if c.Write(ctx, websocket.MessageText, []byte(msg)) != nil {
					return
				}
			}
		}()
		for {
			typ, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			msg, err := parseMessage(typ, data)
			if err != nil {
				continue
			}
			id := ""
			if typ == websocket.MessageText {
				head, _, _ := strings.Cut(string(data), "\r\n\r\n")
				id = headerValue([]byte(head), "x-requestid")
			} else if len(data) >= 2 {
				n := int(binary.BigEndian.Uint16(data))
				id = headerValue(data[2:2+n], "x-requestid")
			}
			received <- sentMessage{path: msg.path, requestID: id, body: msg.body}
		}
	}))
	t.Cleanup(func() { srv.Close(); close(reply) })

	return azureSession{
		url:      "ws" + strings.TrimPrefix(srv.URL, "http"),
		received: received,
		handshake: func() http.Header {
			select {
			case h := <-headerCh:
				return h
			default:
				return nil
			}
		},
		reply: reply,
	}
}

// textFrame renders a protocol message the way Azure sends one.
func textFrame(path, requestID, body string) string {
	return "Path: " + path + "\r\nX-RequestId: " + requestID +
		"\r\nContent-Type: application/json\r\n\r\n" + body
}

// TestSTTOpensSession checks the opening handshake: the key travels as a header,
// and the session is opened with a client description, a recognition
// configuration, and the WAV header describing the audio that follows.
func TestSTTOpensSession(t *testing.T) {
	session := newAzureSession(t)
	conn := &sttConnector{cfg: STTConfig{APIKey: "test-key", Host: session.url}}

	stream, err := conn.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	want := []string{pathSpeechConfig, pathSpeechContext, pathAudio}
	var ids []string
	for i, wantPath := range want {
		got := <-session.received
		if got.path != wantPath {
			t.Errorf("message %d path = %q, want %q", i, got.path, wantPath)
		}
		ids = append(ids, got.requestID)
	}

	// The WAV header opens the turn's audio.
	// (the third message's body, captured above)
	if ids[0] == "" {
		t.Error("messages carry no X-RequestId")
	}
	for i := range ids {
		if ids[i] != ids[0] {
			t.Errorf("message %d request id = %q, want all of a turn to share %q", i, ids[i], ids[0])
		}
	}
	if strings.Contains(ids[0], "-") {
		t.Errorf("request id = %q, want a UUID with the dashes removed", ids[0])
	}

	if h := session.handshake(); h == nil {
		t.Fatal("the service never connected")
	} else if got := h.Get("Ocp-Apim-Subscription-Key"); got != "test-key" {
		t.Errorf("subscription key header = %q, want the key", got)
	}
}

// TestSTTRecvTranscripts checks a hypothesis becomes an interim result and a
// successful phrase a finalized one that ends the turn.
func TestSTTRecvTranscripts(t *testing.T) {
	session := newAzureSession(t)
	conn := &sttConnector{cfg: STTConfig{APIKey: "k", Host: session.url, Language: language.French}}

	stream, err := conn.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	session.reply <- textFrame(pathHypothesis, "r1", `{"Text":"bonjo"}`)
	res, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv hypothesis: %v", err)
	}
	if len(res) != 1 || res[0].Text != "bonjo" || res[0].Final {
		t.Fatalf("hypothesis result = %+v, want an unfinalized \"bonjo\"", res)
	}
	if res[0].Language != "fr-FR" {
		t.Errorf("language = %q, want fr-FR", res[0].Language)
	}

	session.reply <- textFrame(pathPhrase, "r1", `{"RecognitionStatus":"Success","DisplayText":"bonjour"}`)
	res, err = stream.Recv()
	if err != nil {
		t.Fatalf("Recv phrase: %v", err)
	}
	if len(res) != 1 || res[0].Text != "bonjour" || !res[0].Final || !res[0].EndOfTurn {
		t.Fatalf("phrase result = %+v, want a finalized \"bonjour\" ending the turn", res)
	}
}

// TestSTTSkipsUnsuccessfulPhrases checks a phrase that recognized nothing
// produces no transcript rather than an empty one.
func TestSTTSkipsUnsuccessfulPhrases(t *testing.T) {
	session := newAzureSession(t)
	conn := &sttConnector{cfg: STTConfig{APIKey: "k", Host: session.url}}

	stream, err := conn.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	session.reply <- textFrame(pathPhrase, "r1", `{"RecognitionStatus":"NoMatch","DisplayText":""}`)
	session.reply <- textFrame(pathPhrase, "r1", `{"RecognitionStatus":"InitialSilenceTimeout"}`)
	session.reply <- textFrame(pathPhrase, "r1", `{"RecognitionStatus":"Success","DisplayText":"hello"}`)

	res, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if len(res) != 1 || res[0].Text != "hello" {
		t.Fatalf("result = %+v, want only the successful phrase", res)
	}
}

// TestSTTOpensNextTurn is the regression test for continuous recognition: Azure
// ends a turn after each utterance, and the client has to open the next one.
// Without this a session goes silent after the first phrase.
func TestSTTOpensNextTurn(t *testing.T) {
	session := newAzureSession(t)
	conn := &sttConnector{cfg: STTConfig{APIKey: "k", Host: session.url}}

	stream, err := conn.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	// Drain the opening three messages and remember the first turn's id.
	var firstTurn string
	for range 3 {
		firstTurn = (<-session.received).requestID
	}

	// End the turn, then check the client opens the next one.
	session.reply <- textFrame(pathTurnEnd, firstTurn, "{}")
	session.reply <- textFrame(pathHypothesis, "r2", `{"Text":"again"}`)

	res, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if len(res) != 1 || res[0].Text != "again" {
		t.Fatalf("result = %+v, want the transcript from the second turn", res)
	}

	ctx := <-session.received
	if ctx.path != pathSpeechContext {
		t.Fatalf("after turn.end the client sent %q, want a fresh %q", ctx.path, pathSpeechContext)
	}
	audio := <-session.received
	if audio.path != pathAudio {
		t.Fatalf("after turn.end the client sent %q, want a fresh WAV header on %q", audio.path, pathAudio)
	}
	if len(audio.body) != wavHeaderLen || string(audio.body[0:4]) != "RIFF" {
		t.Errorf("second turn opened with %d bytes starting %q, want a RIFF header", len(audio.body), audio.body[:4])
	}
	if ctx.requestID == firstTurn {
		t.Errorf("second turn reused request id %q, want a fresh one", firstTurn)
	}
}

// TestSTTSendAndClose checks audio reaches the service under the current turn
// and that closing signals end of stream with an empty audio message.
func TestSTTSendAndClose(t *testing.T) {
	session := newAzureSession(t)
	conn := &sttConnector{cfg: STTConfig{APIKey: "k", Host: session.url}}

	stream, err := conn.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	for range 3 {
		<-session.received
	}

	if err := stream.Send([]byte{1, 2, 3, 4}); err != nil {
		t.Fatalf("Send: %v", err)
	}
	got := <-session.received
	if got.path != pathAudio || string(got.body) != "\x01\x02\x03\x04" {
		t.Errorf("sent %q on %q, want the PCM on the audio path", got.body, got.path)
	}

	if err := stream.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	end := <-session.received
	if end.path != pathAudio || len(end.body) != 0 {
		t.Errorf("close sent %d bytes on %q, want an empty audio message", len(end.body), end.path)
	}
}

// TestSpeechConfigBody checks the client description is well-formed JSON with
// the context object Azure expects.
func TestSpeechConfigBody(t *testing.T) {
	var body struct {
		Context struct {
			System map[string]any `json:"system"`
			OS     map[string]any `json:"os"`
		} `json:"context"`
	}
	if err := json.Unmarshal([]byte(speechConfigBody()), &body); err != nil {
		t.Fatalf("speech.config is not valid JSON: %v", err)
	}
	if len(body.Context.System) == 0 || len(body.Context.OS) == 0 {
		t.Errorf("speech.config = %s, want a context with system and os", speechConfigBody())
	}
}

// parseQuery pulls the query parameters off a built endpoint URL.
func parseQuery(t *testing.T, endpoint string) url.Values {
	t.Helper()
	u, err := url.Parse(endpoint)
	if err != nil {
		t.Fatalf("endpoint %q is not a URL: %v", endpoint, err)
	}
	return u.Query()
}
