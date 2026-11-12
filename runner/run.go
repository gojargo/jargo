package runner

import (
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo-web/prebuilt"
	"github.com/gojargo/jargo/transport/rtc"
	"github.com/gojargo/jargo/transport/whatsapp"
	"github.com/gojargo/jargo/utils/security"
	"github.com/google/uuid"
	"github.com/joho/godotenv"
	"github.com/pion/webrtc/v4"
	"gopkg.in/yaml.v3"
)

// Bot runs one session. The runner calls it for every session it starts, with
// the arguments for the transport the session arrived on, and the session lasts
// as long as the call. ctx is canceled when the runner shuts down.
type Bot func(ctx context.Context, args Arguments) error

// Wire names and defaults the runner uses in several places.
const (
	defaultHost  = "localhost"
	keySessionID = "sessionId"
	keyURLs      = "urls"
	keyToken     = "token"
	wsAuthToken  = "token"
	wsAuthNone   = "none"
)

// The transports the runner can serve.
const (
	transportWebRTC    = "webrtc"
	transportLiveKit   = "livekit"
	transportWebSocket = "websocket"
	transportDaily     = "daily"
	transportMoQ       = "moq"
)

// telephonyTransports are the providers whose calls arrive on /ws.
var telephonyTransports = []string{transportTwilio, transportTelnyx, transportPlivo, transportExotel} //nolint:gochecknoglobals,lll // a fixed list

// allTransports are the transports /start accepts when none is singled out.
var allTransports = []string{ //nolint:gochecknoglobals // a fixed list
	transportWebRTC, transportLiveKit,
	transportTwilio, transportTelnyx, transportPlivo, transportExotel,
	transportWebSocket,
}

func isTelephony(t string) bool {
	return slices.Contains(telephonyTransports, t)
}

// transportRoutesEnabled reports whether a transport can run here. Daily and
// MoQ have no transport in jargo; any other name is left to the caller, which
// rejects it as unknown.
func transportRoutesEnabled(t string) bool {
	return t != transportDaily && t != transportMoQ
}

// The runner's defaults, and what it was started with.
//
//nolint:gochecknoglobals // the runner's process-wide settings, readable by bots
var (
	runnerMu              sync.Mutex
	runnerDownloadsFolder string
	runnerHost            = defaultHost
	runnerPort            = 7860
)

// RunnerDownloadsFolder returns the folder the runner serves files for download
// from, or "" when it serves none.
func RunnerDownloadsFolder() string {
	runnerMu.Lock()
	defer runnerMu.Unlock()
	return runnerDownloadsFolder
}

// RunnerHost returns the host the runner listens on.
func RunnerHost() string {
	runnerMu.Lock()
	defer runnerMu.Unlock()
	return runnerHost
}

// RunnerPort returns the port the runner listens on.
func RunnerPort() int {
	runnerMu.Lock()
	defer runnerMu.Unlock()
	return runnerPort
}

// App is the runner's HTTP router. Add routes of your own to it before calling
// Main:
//
//	runner.App.HandleFunc("GET /my-route", myHandler)
//	runner.Main(bot)
//
//nolint:gochecknoglobals // the one router Main serves, extended by the bot
var App = http.NewServeMux()

// Frontend is the browser client the runner serves at /client, with / sending
// there. It is the prebuilt client (github.com/gojargo/jargo-web/prebuilt) by
// default; set it to another handler before Main to serve your own, or to nil
// to serve none.
//
//nolint:gochecknoglobals // the client Main serves, replaceable by the bot
var Frontend = prebuilt.Handler()

// Args are the runner's parsed command-line arguments.
type Args struct {
	// Host and Port are where the server listens.
	Host string
	Port int
	// Transport restricts the server to one transport and makes it the default
	// for /start; empty serves them all.
	Transport string
	// Proxy is the public host name telephony providers reach the server on.
	Proxy string
	// DownloadsFolder holds the files served for download at /files.
	DownloadsFolder string
	// RunnerBody is a YAML or JSON file read as the body of each WebSocket
	// session.
	RunnerBody string
	// Verbose raises the logging verbosity.
	Verbose int
	// ESP32 rewrites WebRTC answers for ESP32 clients.
	ESP32 bool
	// WhatsApp serves the WhatsApp webhook.
	WhatsApp bool
	// WSAuth is "token" to require a token from /start on the WebSocket routes,
	// or "none".
	WSAuth string
	// ICEServers are the STUN and TURN servers for WebRTC, as given.
	ICEServers []map[string]any
	// AllowedOrigins limit the origins of HTTP and WebSocket requests; empty
	// allows all.
	AllowedOrigins []string
}

// iceServers converts the configured ICE servers for the WebRTC stack.
func (a *Args) iceServers() []webrtc.ICEServer {
	servers := make([]webrtc.ICEServer, 0, len(a.ICEServers))
	for _, s := range a.ICEServers {
		var server webrtc.ICEServer
		switch urls := s[keyURLs].(type) {
		case string:
			server.URLs = []string{urls}
		case []any:
			for _, u := range urls {
				if str, ok := u.(string); ok {
					server.URLs = append(server.URLs, str)
				}
			}
		case []string:
			server.URLs = urls
		}
		if u, ok := s["username"].(string); ok {
			server.Username = u
		}
		if c, ok := s["credential"].(string); ok {
			server.Credential = c
		}
		servers = append(servers, server)
	}
	return servers
}

// runnerURL is where a browser opens the runner.
func runnerURL(args *Args) string {
	return fmt.Sprintf("http://%s:%d", args.Host, args.Port)
}

// --- WebSocket session tokens ---

// wsTokenTTL is how long a WebSocket session token stays valid.
const wsTokenTTL = 300 * time.Second

type wsTokenPayload struct {
	Exp int64  `json:"exp"`
	JTI string `json:"jti"`
}

// wsAuth issues and checks the signed, single-use tokens that admit a client to
// the WebSocket routes when token auth is on.
type wsAuth struct {
	// secret is per process, so a token from one runner cannot be replayed
	// against another.
	secret []byte
	mu     sync.Mutex
	used   map[string]struct{}
	now    func() time.Time
}

func newWSAuth() *wsAuth {
	secret := make([]byte, 32)
	_, _ = rand.Read(secret)
	return &wsAuth{secret: secret, used: map[string]struct{}{}, now: time.Now}
}

func (a *wsAuth) sign(payload string) string {
	mac := hmac.New(sha256.New, a.secret)
	mac.Write([]byte(payload))
	return hex.EncodeToString(mac.Sum(nil))
}

// generate returns a token valid for wsTokenTTL: a payload of its expiry and a random
// nonce, and the payload's HMAC.
func (a *wsAuth) generate() string {
	nonce := make([]byte, 8)
	_, _ = rand.Read(nonce)
	//nolint:errchkjson // an int and a string always encode
	raw, _ := json.Marshal(wsTokenPayload{Exp: a.now().Add(wsTokenTTL).Unix(), JTI: hex.EncodeToString(nonce)})
	payload := base64.RawURLEncoding.EncodeToString(raw)
	return payload + "." + a.sign(payload)
}

// verifyAndConsume reports whether token is genuine, unexpired and unused, and
// marks it used when it is.
func (a *wsAuth) verifyAndConsume(token string) bool {
	payload, sig, ok := strings.Cut(token, ".")
	if !ok || strings.Contains(sig, ".") {
		return false
	}
	if !hmac.Equal([]byte(a.sign(payload)), []byte(sig)) {
		return false
	}
	raw, err := base64.RawURLEncoding.DecodeString(strings.TrimRight(payload, "="))
	if err != nil {
		return false
	}
	var p wsTokenPayload
	if err := json.Unmarshal(raw, &p); err != nil {
		return false
	}
	if a.now().Unix() > p.Exp {
		return false
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if _, seen := a.used[token]; seen {
		return false
	}
	a.used[token] = struct{}{}
	return true
}

// extractWSToken reads a session token from the handshake: an
// "Authorization: Bearer" header, else a token query parameter.
func extractWSToken(r *http.Request) string {
	if auth := r.Header.Get("Authorization"); len(auth) > 7 && strings.EqualFold(auth[:7], "bearer ") {
		return strings.TrimSpace(auth[7:])
	}
	return r.URL.Query().Get("token")
}

// --- the server ---

// server holds the runner's state while it serves.
type server struct {
	args *Args
	bot  Bot
	// ctx is canceled when the runner shuts down, ending every session.
	ctx context.Context //nolint:containedctx // the lifetime of every session the server starts

	mu sync.Mutex
	// activeSessions maps a session id to the body its /start carried.
	activeSessions map[string]json.RawMessage

	auth       *wsAuth
	rtcHandler *rtc.RequestHandler
	whatsapp   *whatsapp.Client
	sessions   sync.WaitGroup
}

func newServer(ctx context.Context, args *Args, bot Bot) *server {
	return &server{
		args:           args,
		bot:            bot,
		ctx:            ctx,
		activeSessions: map[string]json.RawMessage{},
		auth:           newWSAuth(),
	}
}

// runBotSession runs the bot for one session in the background. When
// sessionID is tracked, the session is forgotten once the bot returns.
func (s *server) runBotSession(args Arguments, sessionID string, tracked bool) {
	s.sessions.Go(func() {
		defer func() {
			if tracked {
				s.mu.Lock()
				delete(s.activeSessions, sessionID)
				s.mu.Unlock()
			}
		}()
		s.callBot(args, sessionID)
	})
}

// callBot runs the bot for one session and logs how it ended. A bot that panics
// is that session's failure alone: it is logged with its stack, and every other
// session the runner is serving carries on.
func (s *server) callBot(args Arguments, sessionID string) {
	defer func() {
		if r := recover(); r != nil {
			slog.Error("runner: bot session panicked",
				"session_id", sessionID, "panic", r, "stack", string(debug.Stack()))
		}
	}()
	if err := s.bot(s.ctx, args); err != nil && !errors.Is(err, context.Canceled) {
		slog.Error("runner: bot session ended with an error", "session_id", sessionID, "err", err)
	}
}

// runnerBody is the --runner-body file's contents as JSON, or nil when none
// was given. A WebSocket session has no /start to carry a body, so a bot that
// needs session data reads it from this file instead.
func (s *server) runnerBody() json.RawMessage {
	if s.args.RunnerBody == "" {
		return nil
	}
	data, err := os.ReadFile(s.args.RunnerBody) //nolint:gosec // a file the operator names on the command line
	if err != nil {
		slog.Error("runner: read runner body", "file", s.args.RunnerBody, "err", err)
		return nil
	}
	var v any
	if err = yaml.Unmarshal(data, &v); err != nil {
		slog.Error("runner: parse runner body", "file", s.args.RunnerBody, "err", err)
		return nil
	}
	body, err := json.Marshal(v)
	if err != nil {
		slog.Error("runner: encode runner body", "file", s.args.RunnerBody, "err", err)
		return nil
	}
	return body
}

// writeJSON writes v as a JSON response.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		slog.Error("runner: write response", "err", err)
	}
}

// writeDetail writes an error the way the HTTP API reports one.
func writeDetail(w http.ResponseWriter, status int, detail string) {
	writeJSON(w, status, map[string]any{"detail": detail})
}

// decodeBody decodes a JSON request body into v. A body that does not decode
// is logged with the raw body, which is what makes a malformed payload from any
// client debuggable, and answered with 422.
func decodeBody(w http.ResponseWriter, r *http.Request, v any) bool {
	body, err := io.ReadAll(r.Body)
	if err == nil {
		err = json.Unmarshal(body, v)
	}
	if err != nil {
		raw := string(body)
		if len(raw) > 5000 {
			raw = raw[:5000]
		}
		slog.Error("runner: 422 validation error", "path", r.URL.Path, "err", err)
		slog.Error("runner: raw body", "body", raw)
		writeDetail(w, http.StatusUnprocessableEntity, err.Error())
		return false
	}
	return true
}

// routes builds the server's router on top of mux.
func (s *server) routes(mux *http.ServeMux) http.Handler {
	s.setupFrontendRoutes(mux)
	s.setupWebRTCRoutes(mux)
	s.setupTelephonyRoutes(mux)
	s.setupWebSocketRoutes(mux)
	s.setupUnifiedStartRoute(mux)
	if s.args.WhatsApp {
		s.setupWhatsAppRoutes(mux)
	}
	return s.cors(mux)
}

// cors answers cross-origin requests from the allowed origins, or from any
// origin when none are configured, with credentials allowed.
func (s *server) cors(next http.Handler) http.Handler {
	allowAll := len(s.args.AllowedOrigins) == 0
	allowed := func(origin string) bool {
		if allowAll {
			return true
		}
		return slices.Contains(s.args.AllowedOrigins, origin)
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin == "" {
			next.ServeHTTP(w, r)
			return
		}
		h := w.Header()
		h.Add("Vary", "Origin")
		preflight := r.Method == http.MethodOptions && r.Header.Get("Access-Control-Request-Method") != ""
		if !allowed(origin) {
			if preflight {
				http.Error(w, "Disallowed CORS origin", http.StatusBadRequest)
				return
			}
			next.ServeHTTP(w, r)
			return
		}
		h.Set("Access-Control-Allow-Origin", origin)
		h.Set("Access-Control-Allow-Credentials", "true")
		if preflight {
			h.Set("Access-Control-Allow-Methods", "DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT")
			if req := r.Header.Get("Access-Control-Request-Headers"); req != "" {
				h.Set("Access-Control-Allow-Headers", req)
			}
			h.Set("Access-Control-Max-Age", "600")
			w.WriteHeader(http.StatusOK)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// setupFrontendRoutes serves the browser client at /client and sends / there.
func (s *server) setupFrontendRoutes(mux *http.ServeMux) {
	if Frontend == nil {
		slog.Error("runner: prebuilt frontend not available")
		return
	}
	mux.Handle("/client/", http.StripPrefix("/client", Frontend))
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/client/", http.StatusTemporaryRedirect)
	})
}

// resolveDownloadPath resolves filename inside folder, refusing a path that
// leaves it.
func resolveDownloadPath(folder, filename string) (string, bool) {
	base, err := filepath.Abs(folder)
	if err != nil {
		return "", false
	}
	path, err := filepath.Abs(filepath.Join(base, filename))
	if err != nil {
		return "", false
	}
	if resolved, linkErr := filepath.EvalSymlinks(path); linkErr == nil {
		path = resolved
	}
	if realBase, linkErr := filepath.EvalSymlinks(base); linkErr == nil {
		base = realBase
	}
	rel, err := filepath.Rel(base, path)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", false
	}
	return path, true
}

// setupWebRTCRoutes serves the WebRTC signaling routes, the session proxy and
// file downloads.
func (s *server) setupWebRTCRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /files/{filename...}", s.downloadFile)

	// The configured ICE servers are what the bot's own peer connection gathers
	// server-reflexive and relay candidates from; without them it offers host
	// candidates only.
	s.rtcHandler = rtc.NewRequestHandler(rtc.RequestHandlerConfig{
		ICEServers: s.args.iceServers(),
		ESP32Mode:  s.args.ESP32,
		Host:       s.args.Host,
	})

	mux.HandleFunc("POST /api/offer", func(w http.ResponseWriter, r *http.Request) {
		var req rtc.Request
		if !decodeBody(w, r, &req) {
			return
		}
		s.offer(w, req, r.URL.Query().Get("session_id"))
	})
	mux.HandleFunc("PATCH /api/offer", func(w http.ResponseWriter, r *http.Request) {
		var req rtc.PatchRequest
		if !decodeBody(w, r, &req) {
			return
		}
		s.iceCandidate(w, req)
	})
	mux.HandleFunc("/sessions/{session_id}/{path...}", s.proxyRequest)
}

func (s *server) downloadFile(w http.ResponseWriter, r *http.Request) {
	filename := r.PathValue("filename")
	if s.args.DownloadsFolder == "" {
		slog.Warn("runner: download requested, but no downloads folder is set up", "file", filename)
		writeDetail(w, http.StatusNotFound, "Not Found")
		return
	}
	path, ok := resolveDownloadPath(s.args.DownloadsFolder, filename)
	if !ok {
		writeDetail(w, http.StatusForbidden, "Access denied")
		return
	}
	root, err := os.OpenRoot(s.args.DownloadsFolder)
	if err != nil {
		writeDetail(w, http.StatusNotFound, "Not Found")
		return
	}
	defer func() { _ = root.Close() }()
	// Opened through the folder's root, so the file cannot be one outside it.
	f, err := root.Open(filepath.Clean(filename))
	if err != nil {
		writeDetail(w, http.StatusNotFound, "Not Found")
		return
	}
	defer func() { _ = f.Close() }()
	info, err := f.Stat()
	if err != nil || info.IsDir() {
		writeDetail(w, http.StatusNotFound, "Not Found")
		return
	}
	if ct := mime.TypeByExtension(filepath.Ext(path)); ct != "" {
		w.Header().Set("Content-Type", ct)
	}
	w.Header().Set("Content-Disposition", mime.FormatMediaType("attachment", map[string]string{"filename": info.Name()}))
	http.ServeContent(w, r, info.Name(), info.ModTime(), f)
}

// writeRequestError answers a refused signaling request.
func writeRequestError(w http.ResponseWriter, err error) {
	if reqErr, ok := errors.AsType[*rtc.RequestError](err); ok {
		writeDetail(w, reqErr.Status, reqErr.Detail)
		return
	}
	writeDetail(w, http.StatusInternalServerError, "Internal Server Error")
}

// offer answers a WebRTC offer, starting the bot for a new connection.
func (s *server) offer(w http.ResponseWriter, req rtc.Request, sessionID string) {
	// An offer through the session proxy carries its session id; a direct one
	// gets one minted, so the bot sees a stable identifier either way.
	resolved := sessionID
	if resolved == "" {
		resolved = uuid.NewString()
	}
	answer, err := s.rtcHandler.HandleWebRequest(req, func(conn *rtc.Connection) error {
		args := &WebRTCRunnerArguments{
			RunnerArguments:  newRunnerArguments(req.RequestData, resolved),
			WebRTCConnection: conn,
		}
		args.CLIArgs = s.args
		s.runBotSession(args, resolved, true)
		return nil
	})
	if err != nil {
		writeRequestError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, answer)
}

// iceCandidate adds the ICE candidates a client trickled in.
func (s *server) iceCandidate(w http.ResponseWriter, req rtc.PatchRequest) {
	slog.Debug("runner: received patch request", "pc_id", req.PCID, "candidates", len(req.Candidates))
	if err := s.rtcHandler.HandlePatchRequest(req); err != nil {
		writeRequestError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "success"})
}

// proxyRequest serves /sessions/{session_id}/..., the way a hosted deployment
// routes a session's requests: the WebRTC signaling of a session registered by
// /start, with the body /start carried as the bot's request data.
func (s *server) proxyRequest(w http.ResponseWriter, r *http.Request) {
	sessionID := r.PathValue("session_id")
	path := r.PathValue("path")
	switch r.Method {
	case http.MethodGet, http.MethodPost, http.MethodPut, http.MethodPatch, http.MethodDelete:
	default:
		writeDetail(w, http.StatusMethodNotAllowed, "Method Not Allowed")
		return
	}
	s.mu.Lock()
	activeSession, ok := s.activeSessions[sessionID]
	s.mu.Unlock()
	if !ok {
		http.Error(w, "Invalid or not-yet-ready session_id", http.StatusNotFound)
		return
	}

	if strings.HasSuffix(path, "api/offer") {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "Invalid WebRTC request", http.StatusBadRequest)
			return
		}
		switch r.Method {
		case http.MethodPost:
			var req rtc.Request
			if err := json.Unmarshal(body, &req); err != nil || req.SDP == "" || req.Type == "" {
				slog.Error("runner: failed to parse WebRTC request", "err", err)
				http.Error(w, "Invalid WebRTC request", http.StatusBadRequest)
				return
			}
			if len(req.RequestData) == 0 {
				req.RequestData = activeSession
			}
			s.offer(w, req, sessionID)
			return
		case http.MethodPatch:
			var req rtc.PatchRequest
			if err := json.Unmarshal(body, &req); err != nil || req.PCID == "" {
				slog.Error("runner: failed to parse WebRTC request", "err", err)
				http.Error(w, "Invalid WebRTC request", http.StatusBadRequest)
				return
			}
			s.iceCandidate(w, req)
			return
		}
	}
	slog.Info("runner: received request", "path", path)
	w.WriteHeader(http.StatusOK)
}

// setupUnifiedStartRoute serves POST /start, which starts a session on the
// transport the client asks for, and GET /status.
func (s *server) setupUnifiedStartRoute(mux *http.ServeMux) {
	mux.HandleFunc("GET /status", func(w http.ResponseWriter, _ *http.Request) {
		transports := allTransports
		if s.args.Transport != "" {
			transports = []string{s.args.Transport}
		}
		writeJSON(w, http.StatusOK, map[string]any{"status": "ready", "transports": transports})
	})
	mux.HandleFunc("POST /start", s.start)
}

// startRequest is the body of POST /start.
type startRequest struct {
	Transport               string          `json:"transport"`
	EnableDefaultICEServers bool            `json:"enableDefaultIceServers"` //nolint:tagliatelle // wire field
	CreateDailyRoom         bool            `json:"createDailyRoom"`         //nolint:tagliatelle // wire field
	Body                    json.RawMessage `json:"body"`
}

func (s *server) start(w http.ResponseWriter, r *http.Request) {
	var req startRequest
	if data, err := io.ReadAll(r.Body); err != nil || json.Unmarshal(data, &req) != nil {
		slog.Error("runner: failed to parse request body", "err", err)
		req = startRequest{}
	}
	slog.Debug("runner: received start request", "transport", req.Transport)
	body := req.Body
	if len(body) == 0 || string(body) == "null" {
		body = json.RawMessage("{}")
	}

	t := s.requestedTransport(req)
	if s.args.Transport != "" && t != s.args.Transport {
		writeDetail(w, http.StatusBadRequest, fmt.Sprintf(
			"Transport '%s' is not allowed. Server is configured for '%s' only (-t %s).",
			t, s.args.Transport, s.args.Transport))
		return
	}
	if !transportRoutesEnabled(t) {
		writeDetail(w, http.StatusBadRequest, fmt.Sprintf(
			"Transport '%s' is disabled in this runner environment. "+
				"Check the startup banner for enabled transports.", t))
		return
	}

	switch {
	case t == transportWebRTC:
		s.startWebRTC(w, body, req.EnableDefaultICEServers)
	case t == transportLiveKit:
		s.startLiveKit(w, body)
	case isTelephony(t):
		// The bot starts when the provider connects to /ws.
		result := map[string]any{"wsUrl": s.wsURL("/ws")}
		if s.args.WSAuth == wsAuthToken {
			result[keyToken] = s.auth.generate()
		}
		writeJSON(w, http.StatusOK, result)
	case t == transportWebSocket:
		// The bot starts when the client connects to /ws-client.
		var token any
		if s.args.WSAuth == wsAuthToken {
			token = s.auth.generate()
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"wsUrl":      s.wsURL("/ws-client"),
			keySessionID: uuid.NewString(),
			keyToken:     token,
		})
	default:
		writeDetail(w, http.StatusBadRequest, fmt.Sprintf("Unknown transport '%s'.", t))
	}
}

// requestedTransport is the transport a /start request is for: the one it
// asks for, then the legacy Daily hint, then the one the runner was restricted
// to, then WebRTC.
func (s *server) requestedTransport(req startRequest) string {
	switch {
	case req.Transport != "":
		return req.Transport
	case req.CreateDailyRoom:
		return transportDaily
	case s.args.Transport != "":
		return s.args.Transport
	default:
		return transportWebRTC
	}
}

// wsURL is the WebSocket URL of path on the runner.
func (s *server) wsURL(path string) string {
	scheme := "ws"
	if s.args.Host != defaultHost {
		scheme = "wss"
	}
	return fmt.Sprintf("%s://%s:%d%s", scheme, s.args.Host, s.args.Port, path)
}

// startWebRTC registers a WebRTC session; the bot starts when its offer
// arrives.
func (s *server) startWebRTC(w http.ResponseWriter, body json.RawMessage, enableDefaultICEServers bool) {
	sessionID := uuid.NewString()
	s.mu.Lock()
	s.activeSessions[sessionID] = body
	s.mu.Unlock()
	result := map[string]any{keySessionID: sessionID}
	// Servers configured on the runner go to the client too, so both peers
	// negotiate against the same STUN and TURN servers. They take precedence
	// over the public STUN fallback.
	switch {
	case len(s.args.ICEServers) > 0:
		result["iceConfig"] = map[string]any{"iceServers": s.args.ICEServers}
	case enableDefaultICEServers:
		result["iceConfig"] = map[string]any{"iceServers": []map[string]any{
			{keyURLs: []string{"stun:stun.l.google.com:19302"}},
		}}
	}
	writeJSON(w, http.StatusOK, result)
}

// startLiveKit starts a session in a LiveKit room: it mints tokens for the bot
// and the caller, starts the bot, and gives the caller what it needs to join.
func (s *server) startLiveKit(w http.ResponseWriter, body json.RawMessage) {
	url, apiKey, apiSecret, err := LiveKitCredentials()
	if err != nil {
		slog.Error("runner: start LiveKit session", "err", err)
		writeDetail(w, http.StatusInternalServerError, err.Error())
		return
	}
	roomName := os.Getenv("LIVEKIT_ROOM_NAME")
	if roomName == "" {
		roomName = "jargo-" + strings.ReplaceAll(uuid.NewString(), "-", "")[:8]
	}
	sessionID := uuid.NewString()
	agentToken, userToken, err := GenerateSessionTokens(roomName, sessionID, apiKey, apiSecret)
	if err != nil {
		slog.Error("runner: mint LiveKit tokens", "err", err)
		writeDetail(w, http.StatusInternalServerError, err.Error())
		return
	}
	args := &LiveKitRunnerArguments{
		RunnerArguments: newRunnerArguments(body, sessionID),
		RoomName:        roomName,
		URL:             url,
		Token:           agentToken,
	}
	args.CLIArgs = s.args
	s.mu.Lock()
	s.activeSessions[sessionID] = body
	s.mu.Unlock()
	s.runBotSession(args, sessionID, true)
	writeJSON(w, http.StatusOK, map[string]any{"url": url, "token": userToken, keySessionID: sessionID})
}

// authorizeWebSocket checks a WebSocket handshake against the token auth and
// the allowed origins, refusing it when either says no.
func (s *server) authorizeWebSocket(w http.ResponseWriter, r *http.Request, pathToken string) bool {
	if s.args.WSAuth == wsAuthToken {
		token := pathToken
		if token == "" {
			token = extractWSToken(r)
		}
		if token == "" || !s.auth.verifyAndConsume(token) {
			slog.Warn("runner: WebSocket connection rejected: invalid or missing token")
			http.Error(w, "Forbidden", http.StatusForbidden)
			return false
		}
	}
	origin := r.Header.Get("Origin")
	if !security.IsOriginAllowed(origin, s.args.AllowedOrigins) {
		slog.Warn("runner: WebSocket connection rejected: origin not allowed", "origin", origin)
		http.Error(w, "Forbidden", http.StatusForbidden)
		return false
	}
	return true
}

// acceptWebSocket upgrades an authorized handshake.
func acceptWebSocket(w http.ResponseWriter, r *http.Request) (*websocket.Conn, bool) {
	// The origin was checked against the runner's own list already.
	conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{OriginPatterns: []string{"*"}})
	if err != nil {
		slog.Error("runner: WebSocket accept", "err", err)
		return nil, false
	}
	return conn, true
}

// setupTelephonyRoutes serves the telephony WebSocket (/ws), where providers
// stream their calls, and, when one telephony provider is selected with a
// public proxy host, the webhook that tells the provider where to stream.
func (s *server) setupTelephonyRoutes(mux *http.ServeMux) {
	if isTelephony(s.args.Transport) {
		xmlTemplates := map[string]string{
			transportTwilio: fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://%s/ws"></Stream>
  </Connect>
</Response>`, s.args.Proxy),
			transportTelnyx: fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://%s/ws" bidirectionalMode="rtp"></Stream>
  </Connect>
</Response>`, s.args.Proxy),
			transportPlivo: fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">wss://%s/ws</Stream>
</Response>`, s.args.Proxy),
		}
		mux.HandleFunc("POST /{$}", func(w http.ResponseWriter, _ *http.Request) {
			if s.args.Transport == transportExotel {
				// Exotel does not use webhooks; point at where to configure it.
				slog.Debug("runner: POST for Exotel, which does not use it")
				writeJSON(w, http.StatusOK, map[string]any{
					"error":         "Exotel doesn't use POST webhooks",
					"websocket_url": fmt.Sprintf("wss://%s/ws", s.args.Proxy),
					"note":          "Configure the WebSocket URL above in your Exotel App Bazaar Voicebot Applet",
				})
				return
			}
			slog.Debug("runner: POST XML", "provider", s.args.Transport)
			xml, ok := xmlTemplates[s.args.Transport]
			if !ok {
				xml = "<Response></Response>"
			}
			w.Header().Set("Content-Type", "application/xml")
			_, _ = io.WriteString(w, xml)
		})
	}

	handle := func(w http.ResponseWriter, r *http.Request, pathToken string) {
		if !s.authorizeWebSocket(w, r, pathToken) {
			return
		}
		conn, ok := acceptWebSocket(w, r)
		if !ok {
			return
		}
		slog.Debug("runner: WebSocket connection accepted")
		args := &WebSocketRunnerArguments{
			RunnerArguments: newRunnerArguments(s.runnerBody(), uuid.NewString()),
			WebSocket:       conn,
		}
		args.CLIArgs = s.args
		s.runWebSocketBot(args)
	}
	mux.HandleFunc("GET /ws", func(w http.ResponseWriter, r *http.Request) { handle(w, r, "") })
	mux.HandleFunc("GET /ws/{token}", func(w http.ResponseWriter, r *http.Request) {
		handle(w, r, r.PathValue("token"))
	})
}

// runWebSocketBot runs the bot for a WebSocket session for as long as it lasts.
func (s *server) runWebSocketBot(args *WebSocketRunnerArguments) {
	s.sessions.Add(1)
	defer s.sessions.Done()
	defer func() { _ = args.WebSocket.CloseNow() }()
	s.callBot(args, args.SessionID)
}

// setupWebSocketRoutes serves /ws-client, for plain WebSocket clients.
func (s *server) setupWebSocketRoutes(mux *http.ServeMux) {
	handle := func(w http.ResponseWriter, r *http.Request, pathToken string) {
		if !s.authorizeWebSocket(w, r, pathToken) {
			return
		}
		conn, ok := acceptWebSocket(w, r)
		if !ok {
			return
		}
		slog.Debug("runner: plain WebSocket connection accepted")
		args := &WebSocketRunnerArguments{
			RunnerArguments: newRunnerArguments(s.runnerBody(), uuid.NewString()),
			WebSocket:       conn,
			TransportType:   transportWebSocket,
		}
		args.CLIArgs = s.args
		s.runWebSocketBot(args)
	}
	mux.HandleFunc("GET /ws-client", func(w http.ResponseWriter, r *http.Request) { handle(w, r, "") })
	mux.HandleFunc("GET /ws-client/{token}", func(w http.ResponseWriter, r *http.Request) {
		handle(w, r, r.PathValue("token"))
	})
}

// setupWhatsAppRoutes serves the WhatsApp webhook at /whatsapp, which answers
// each call and starts the bot on its connection.
func (s *server) setupWhatsAppRoutes(mux *http.ServeMux) {
	required := []string{
		"WHATSAPP_APP_SECRET",
		"WHATSAPP_PHONE_NUMBER_ID",
		"WHATSAPP_TOKEN",
		"WHATSAPP_WEBHOOK_VERIFICATION_TOKEN",
	}
	var missing []string
	for _, v := range required {
		if os.Getenv(v) == "" {
			missing = append(missing, v)
		}
	}
	if len(missing) > 0 {
		slog.Error("runner: missing required environment variables for WhatsApp transport",
			"missing", strings.Join(missing, ", "))
		return
	}
	client, err := whatsapp.NewClient(whatsapp.Config{
		Token:         os.Getenv("WHATSAPP_TOKEN"),
		PhoneNumberID: os.Getenv("WHATSAPP_PHONE_NUMBER_ID"),
		AppSecret:     os.Getenv("WHATSAPP_APP_SECRET"),
		VerifyToken:   os.Getenv("WHATSAPP_WEBHOOK_VERIFICATION_TOKEN"),
	})
	if err != nil {
		slog.Error("runner: WhatsApp client", "err", err)
		return
	}
	client.OnCall(func(call *whatsapp.Call) {
		// The call's metadata goes to the bot as its body.
		body, _ := json.Marshal(map[string]any{"id": call.ID, "from": call.From})
		args := &WebRTCRunnerArguments{
			RunnerArguments:  newRunnerArguments(body, uuid.NewString()),
			WebRTCConnection: call.Connection,
		}
		args.CLIArgs = s.args
		s.runBotSession(args, args.SessionID, false)
	})
	s.whatsapp = client
	slog.Info("runner: WhatsApp client initialized")
	mux.Handle("/whatsapp", client.Handler())
}

// shutdown ends every session and releases what the routes hold.
func (s *server) shutdown() {
	if s.rtcHandler != nil {
		_ = s.rtcHandler.Close()
	}
	if s.whatsapp != nil {
		slog.Info("runner: cleaning up WhatsApp client resources")
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		s.whatsapp.TerminateAllCalls(ctx)
		cancel()
		slog.Info("runner: WhatsApp cleanup completed")
	}
}

// --- startup output ---

// styleBannerLine renders one banner line dim cyan, unless stdout is not a
// terminal or NO_COLOR is set, so escape codes never leak into logs.
func styleBannerLine(text string) string {
	if info, err := os.Stdout.Stat(); err == nil && info.Mode()&os.ModeCharDevice != 0 && os.Getenv("NO_COLOR") == "" {
		return "\033[2;36m" + text + "\033[0m"
	}
	return text
}

// printDevRunnerBanner prints a bordered banner naming the development runner
// and linking to the deployment docs.
func printDevRunnerBanner(out io.Writer) {
	lines := []string{
		"JARGO DEVELOPMENT RUNNER",
		"",
		"Learn about running bots locally and in production:",
		"https://gojargo.github.io/jargo/deploy/",
	}
	width := 0
	for _, l := range lines {
		width = max(width, len([]rune(l)))
	}
	_, _ = fmt.Fprintln(out)
	_, _ = fmt.Fprintln(out, styleBannerLine("╭"+strings.Repeat("─", width+2)+"╮"))
	for _, l := range lines {
		pad := strings.Repeat(" ", width-len([]rune(l)))
		_, _ = fmt.Fprintln(out, styleBannerLine("│ "+l+pad+" │"))
	}
	_, _ = fmt.Fprintln(out, styleBannerLine("╰"+strings.Repeat("─", width+2)+"╯"))
}

func printSecurityStatus(out io.Writer, args *Args) {
	if args.WSAuth == wsAuthToken {
		_, _ = fmt.Fprintln(out, "   → WebSocket auth:  token (HMAC, call /start to obtain a token)")
	}
	if len(args.AllowedOrigins) > 0 {
		_, _ = fmt.Fprintf(out, "   → Allowed origins: %s\n", strings.Join(args.AllowedOrigins, ", "))
	} else {
		_, _ = fmt.Fprintln(out, "   → Allowed origins: all (no restriction)")
	}
}

// printStartupMessage prints where to reach the runner for the transport it
// serves.
func printStartupMessage(out io.Writer, args *Args) {
	p := func(format string, a ...any) { _, _ = fmt.Fprintf(out, format+"\n", a...) }
	p("")
	switch {
	case args.Transport == "":
		p("🚀 Bot ready!")
		p("   → Open: %s", runnerURL(args))
		p("   → Enabled transports: webrtc, livekit, telephony, websocket")
		printSecurityStatus(out, args)
	case args.Transport == transportWebRTC:
		switch {
		case args.ESP32:
			p("🚀 Bot ready! (ESP32 mode)")
		case args.WhatsApp:
			p("🚀 Bot ready! (WhatsApp)")
		default:
			p("🚀 Bot ready! (WebRTC)")
		}
		p("   → Open: %s", runnerURL(args))
	case args.Transport == transportLiveKit:
		p("🚀 Bot ready! (LiveKit)")
		p("   → Open: %s", runnerURL(args))
	case isTelephony(args.Transport):
		p("🚀 Bot ready! (%s)", strings.ToUpper(args.Transport[:1])+args.Transport[1:])
		p("   → Open: %s", runnerURL(args))
		if args.Proxy != "" {
			p("   → XML webhook: http://%s:%d/", args.Host, args.Port)
		}
		p("   → WebSocket:   ws://%s:%d/ws", args.Host, args.Port)
		printSecurityStatus(out, args)
	case args.Transport == transportWebSocket:
		p("🚀 Bot ready! (WebSocket)")
		p("   → Open: %s", runnerURL(args))
		scheme := "ws"
		if args.Host != defaultHost {
			scheme = "wss"
		}
		p("   → WebSocket:   %s://%s:%d/ws-client", scheme, args.Host, args.Port)
		printSecurityStatus(out, args)
	}
	p("")
}

// --- command line ---

// validateAndCleanProxy reduces a proxy given as a URL to its host name.
func validateAndCleanProxy(proxy string) string {
	if proxy == "" {
		return proxy
	}
	original := proxy
	for _, scheme := range []string{"http://", "https://"} {
		if rest, ok := strings.CutPrefix(proxy, scheme); ok {
			proxy = rest
			slog.Warn("runner: removed protocol from proxy URL; --proxy expects only the hostname "+
				"(e.g. 'mybot.ngrok.io')", "using", proxy, "given", original)
			break
		}
	}
	return strings.TrimRight(proxy, "/")
}

// ErrInvalidICEServer reports an ICE server entry that cannot be used.
var ErrInvalidICEServer = errors.New("runner: invalid ICE server")

// parseICEServers reads the ICE server configuration: the JARGO_ICE_SERVERS
// value, or the tokens given with --ice-servers. Each entry is a bare STUN or
// TURN URL, or a JSON object with urls and, for TURN, username and credential.
// An environment value is read as a JSON array when it starts with "[", as one
// JSON object when it starts with "{", and otherwise split on commas into bare
// URLs.
func parseICEServers(env string, tokens []string) ([]map[string]any, error) {
	entries, err := iceEntries(env, tokens)
	if err != nil {
		return nil, err
	}
	var servers []map[string]any
	for _, entry := range entries {
		server, err := parseICEEntry(entry)
		if err != nil {
			return nil, err
		}
		if server != nil {
			servers = append(servers, server)
		}
	}
	return servers, nil
}

// iceEntries splits the ICE server configuration into its entries: the CLI
// tokens as given, or the environment value as a JSON array, one JSON object,
// or comma-separated URLs.
func iceEntries(env string, tokens []string) ([]any, error) {
	var entries []any
	if len(tokens) > 0 {
		for _, t := range tokens {
			entries = append(entries, t)
		}
		return entries, nil
	}
	text := strings.TrimSpace(env)
	switch {
	case text == "":
		return nil, nil
	case strings.HasPrefix(text, "["):
		if err := json.Unmarshal([]byte(text), &entries); err != nil {
			return nil, fmt.Errorf("%w: could not parse %q as a JSON array: %w", ErrInvalidICEServer, text, err)
		}
	case strings.HasPrefix(text, "{"):
		entries = []any{text}
	default:
		for item := range strings.SplitSeq(text, ",") {
			if item = strings.TrimSpace(item); item != "" {
				entries = append(entries, item)
			}
		}
	}
	return entries, nil
}

// parseICEEntry reads one ICE server entry: a bare URL, or an object (or its
// JSON) with urls and optional credentials. An empty entry reads as nil.
func parseICEEntry(entry any) (map[string]any, error) {
	if text, ok := entry.(string); ok {
		text = strings.TrimSpace(text)
		if text == "" {
			return nil, nil //nolint:nilnil // an empty entry is skipped, not an error
		}
		if !strings.HasPrefix(text, "{") {
			return map[string]any{keyURLs: text}, nil
		}
		var obj map[string]any
		if err := json.Unmarshal([]byte(text), &obj); err != nil {
			return nil, fmt.Errorf("%w: could not parse %q as a JSON object: %w", ErrInvalidICEServer, text, err)
		}
		entry = obj
	}
	obj, ok := entry.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%w: expected a URL or a JSON object, got %v", ErrInvalidICEServer, entry)
	}
	for k := range obj {
		if k != keyURLs && k != "username" && k != "credential" {
			return nil, fmt.Errorf("%w: unsupported key %q in %v; supported keys are credential, urls, username",
				ErrInvalidICEServer, k, obj)
		}
	}
	if !hasURLs(obj[keyURLs]) {
		return nil, fmt.Errorf("%w: missing 'urls' in %v", ErrInvalidICEServer, obj)
	}
	server := map[string]any{}
	for k, v := range obj {
		if v != nil {
			server[k] = v
		}
	}
	return server, nil
}

// hasURLs reports whether an ICE server's urls value names any.
func hasURLs(v any) bool {
	switch urls := v.(type) {
	case string:
		return urls != ""
	case []any:
		return len(urls) > 0
	default:
		return false
	}
}

// listFlag collects a flag given any number of times.
type listFlag []string

func (l *listFlag) String() string     { return strings.Join(*l, ",") }
func (l *listFlag) Set(v string) error { *l = append(*l, v); return nil }

// countFlag counts how many times a flag is given.
type countFlag int

func (c *countFlag) String() string   { return strconv.Itoa(int(*c)) }
func (c *countFlag) Set(string) error { *c++; return nil }
func (c *countFlag) IsBoolFlag() bool { return true }

// rawFlags are the command-line values before they are checked.
type rawFlags struct {
	args           Args
	verbose        countFlag
	iceServers     listFlag
	allowedOrigins listFlag
}

// addFlags registers the runner's flags on fs. A flag with a short and a long
// name is registered under both.
func addFlags(fs *flag.FlagSet) *rawFlags {
	f := &rawFlags{}
	a := &f.args
	fs.StringVar(&a.Host, "host", defaultHost, "Host address")
	fs.IntVar(&a.Port, "port", 7860, "Port number")
	transportUsage := "Restrict the server to a single transport (webrtc, websocket, livekit, twilio, telnyx, " +
		"plivo, exotel) and set it as the default for /start. Omit to support all transports."
	fs.StringVar(&a.Transport, "transport", "", transportUsage)
	fs.StringVar(&a.Transport, "t", "", transportUsage)
	fs.StringVar(&a.Proxy, "proxy", "", "Public proxy host name")
	fs.StringVar(&a.Proxy, "x", "", "Public proxy host name")
	fs.StringVar(&a.DownloadsFolder, "downloads-folder", "", "Path to folder for files available for download")
	fs.StringVar(&a.DownloadsFolder, "f", "", "Path to folder for files available for download")
	fs.StringVar(&a.DownloadsFolder, "folder", "",
		"Path to folder for files available for download. (Deprecated: use --downloads-folder instead)")
	fs.StringVar(&a.RunnerBody, "runner-body", "",
		"Path to a YAML or JSON file with the runner args body for WebSocket sessions")
	fs.Var(&f.verbose, "verbose", "Increase logging verbosity")
	fs.Var(&f.verbose, "v", "Increase logging verbosity")
	fs.BoolVar(&a.ESP32, "esp32", false,
		"Enable SDP munging for ESP32 compatibility (requires --host with IP address)")
	fs.BoolVar(&a.WhatsApp, "whatsapp", false, "Ensure required WhatsApp environment variables are present")
	wsAuth := os.Getenv("JARGO_WEBSOCKET_AUTH")
	if wsAuth == "" {
		wsAuth = wsAuthNone
	}
	fs.StringVar(&a.WSAuth, "ws-auth", wsAuth,
		"WebSocket authentication mode, none or token. 'token' requires clients to call /start and obtain a "+
			"signed HMAC session token before connecting to /ws or /ws-client. Defaults to the "+
			"JARGO_WEBSOCKET_AUTH environment variable or 'none'.")
	fs.Var(&f.iceServers, "ice-servers",
		"A STUN or TURN server the WebRTC transport should use, as a bare URL (e.g. "+
			"stun:stun.l.google.com:19302) or, when a TURN server needs credentials, as a JSON object "+
			`(e.g. '{"urls": "turn:turn.example.com:3478", "username": "user", "credential": "pass"}'). `+
			"Repeat for several. Defaults to the JARGO_ICE_SERVERS environment variable, which takes the "+
			"same entries comma-separated or as a JSON array. Without this the bot gathers host candidates only.")
	fs.Var(&f.allowedOrigins, "allowed-origins",
		"An allowed origin for HTTP and WebSocket connections (e.g. https://example.com); repeat for several. "+
			"Omit to allow all origins. Defaults to the JARGO_ALLOWED_ORIGINS environment variable "+
			"(comma-separated).")
	return f
}

// supportedTransports are the values -t accepts.
var supportedTransports = map[string]bool{ //nolint:gochecknoglobals // a fixed set
	transportWebRTC: true, transportWebSocket: true, transportLiveKit: true,
	transportTwilio: true, transportTelnyx: true, transportPlivo: true, transportExotel: true,
}

// ErrInvalidArgs reports command-line arguments the runner cannot start with.
var ErrInvalidArgs = errors.New("runner: invalid arguments")

// resolve checks the parsed flags and completes them from the environment.
func (f *rawFlags) resolve() (*Args, error) {
	a := f.args
	a.Verbose = int(f.verbose)
	if a.Transport != "" && !supportedTransports[a.Transport] {
		return nil, fmt.Errorf("%w: transport %q is not supported; choose one of webrtc, websocket, livekit, "+
			"twilio, telnyx, plivo, exotel", ErrInvalidArgs, a.Transport)
	}
	if a.WSAuth != wsAuthNone && a.WSAuth != wsAuthToken {
		return nil, fmt.Errorf("%w: --ws-auth must be none or token, got %q", ErrInvalidArgs, a.WSAuth)
	}
	a.Proxy = validateAndCleanProxy(a.Proxy)

	servers, err := parseICEServers(os.Getenv("JARGO_ICE_SERVERS"), f.iceServers)
	if err != nil {
		return nil, fmt.Errorf("invalid ICE server configuration: %w", err)
	}
	a.ICEServers = servers

	a.AllowedOrigins = []string(f.allowedOrigins)
	if len(a.AllowedOrigins) == 0 {
		for o := range strings.SplitSeq(os.Getenv("JARGO_ALLOWED_ORIGINS"), ",") {
			if o = strings.TrimSpace(o); o != "" {
				a.AllowedOrigins = append(a.AllowedOrigins, o)
			}
		}
	}

	if a.ESP32 && a.Host == defaultHost {
		return nil, fmt.Errorf("%w: for ESP32, you need to specify `--host IP` so we can do SDP munging",
			ErrInvalidArgs)
	}
	return &a, nil
}

// Main starts the development runner and serves until it is interrupted. It
// parses the command line, loads a .env file if there is one, and starts a
// server that serves every transport at once, so a client chooses its own in
// the transport field of POST /start. With -t the server serves only that one
// and prints where to reach it.
//
// The runner runs bot for each session it starts. To add command-line flags of
// your own, register them on flag.CommandLine before calling Main; they are
// parsed with the runner's.
//
// Main exits the process when the runner cannot start.
func Main(bot Bot) {
	if err := Run(context.Background(), flag.CommandLine, os.Args[1:], bot); err != nil {
		slog.Error("runner: cannot start", "err", err)
		os.Exit(1) //nolint:forbidigo // Main is the program's entry point
	}
}

// Run is Main with an explicit flag set and arguments. It returns when ctx is
// canceled or the process is interrupted, once the server has shut down.
func Run(ctx context.Context, fs *flag.FlagSet, arguments []string, bot Bot) error {
	_ = godotenv.Overload()
	_ = os.Setenv("ENV", "local")

	raw := addFlags(fs)
	if err := fs.Parse(arguments); err != nil {
		return err
	}
	args, err := raw.resolve()
	if err != nil {
		return err
	}

	level := slog.LevelDebug
	if args.Verbose > 0 {
		level = slog.LevelDebug - 4
	}
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level})))

	printDevRunnerBanner(os.Stdout)
	printStartupMessage(os.Stdout, args)

	runnerMu.Lock()
	runnerDownloadsFolder = args.DownloadsFolder
	runnerHost = args.Host
	runnerPort = args.Port
	runnerMu.Unlock()

	ctx, stop := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)
	defer stop()
	sessionsCtx, cancelSessions := context.WithCancel(context.WithoutCancel(ctx))
	defer cancelSessions()

	s := newServer(sessionsCtx, args, bot)
	srv := &http.Server{
		Addr:              net.JoinHostPort(args.Host, strconv.Itoa(args.Port)),
		Handler:           s.routes(App),
		ReadHeaderTimeout: 10 * time.Second,
	}
	serveErr := make(chan error, 1)
	go func() { serveErr <- srv.ListenAndServe() }()

	select {
	case err := <-serveErr:
		s.shutdown()
		cancelSessions()
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	case <-ctx.Done():
	}
	slog.Info("runner: shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutdownCtx)
	s.shutdown()
	cancelSessions()
	s.sessions.Wait()
	return nil
}
