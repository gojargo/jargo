package rtc

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sync"

	"github.com/pion/webrtc/v4"
)

// Request is an SDP offer from a client. A first offer carries no PCID and
// opens a new connection; an offer carrying the PCID of an existing connection
// renegotiates it, restarting its peer connection when RestartPC is set.
type Request struct {
	SDP       string `json:"sdp"`
	Type      string `json:"type"`
	PCID      string `json:"pc_id,omitempty"`
	RestartPC bool   `json:"restart_pc,omitempty"`
	// RequestData is custom data the client sent along with the offer.
	RequestData json.RawMessage `json:"request_data,omitempty"`
}

// UnmarshalJSON accepts the custom data under either request_data or
// requestData.
func (r *Request) UnmarshalJSON(data []byte) error {
	type plain Request
	var aux struct {
		plain
		RequestDataCamel json.RawMessage `json:"requestData"` //nolint:tagliatelle // accepted from clients
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	*r = Request(aux.plain)
	if len(r.RequestData) == 0 && len(aux.RequestDataCamel) > 0 {
		r.RequestData = aux.RequestDataCamel
	}
	return nil
}

// ICECandidate is a remote ICE candidate the client trickled in after its
// offer. An empty Candidate is the end-of-candidates marker for its media line.
type ICECandidate struct {
	Candidate     string `json:"candidate"`
	SDPMid        string `json:"sdp_mid"`
	SDPMLineIndex uint16 `json:"sdp_mline_index"`
}

// PatchRequest carries ICE candidates for an existing connection.
type PatchRequest struct {
	PCID       string         `json:"pc_id"`
	Candidates []ICECandidate `json:"candidates"`
}

// ConnectionMode is how many connections a RequestHandler allows at once.
type ConnectionMode string

const (
	// ConnectionModeMultiple allows several simultaneous connections. It is the
	// default.
	ConnectionModeMultiple ConnectionMode = "multiple"
	// ConnectionModeSingle allows only one active connection.
	ConnectionModeSingle ConnectionMode = "single"
)

// RequestError is a request the handler refused, with the HTTP status that
// describes why.
type RequestError struct {
	Status int
	Detail string
}

func (e *RequestError) Error() string { return e.Detail }

// RequestHandlerConfig configures a RequestHandler.
type RequestHandlerConfig struct {
	// ICEServers are the STUN and TURN servers each connection uses. None
	// leaves a connection with host candidates only.
	ICEServers []webrtc.ICEServer
	// ESP32Mode rewrites each answer for ESP32 clients (see SDPMunging).
	ESP32Mode bool
	// Host is the address answers are filtered to in ESP32 mode.
	Host string
	// ConnectionMode defaults to ConnectionModeMultiple.
	ConnectionMode ConnectionMode
}

// RequestHandler answers the offers and ICE candidates clients send over HTTP.
// It creates a Connection for each new client, keeps it under its pc_id so
// later requests reach it, and hands each new one to a callback that runs the
// session on it.
type RequestHandler struct {
	mu  sync.Mutex
	cfg RequestHandlerConfig
	pcs map[string]*Connection
}

// NewRequestHandler returns a handler configured by cfg.
func NewRequestHandler(cfg RequestHandlerConfig) *RequestHandler {
	if cfg.ConnectionMode == "" {
		cfg.ConnectionMode = ConnectionModeMultiple
	}
	return &RequestHandler{cfg: cfg, pcs: map[string]*Connection{}}
}

// UpdateICEServers replaces the ICE servers used for new connections.
func (h *RequestHandler) UpdateICEServers(servers []webrtc.ICEServer) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.cfg.ICEServers = servers
}

// checkSingleConnectionConstraints refuses a request that would break single
// connection mode: a second connection, or an offer for a different one.
func (h *RequestHandler) checkSingleConnectionConstraints(pcID string) error {
	if h.cfg.ConnectionMode != ConnectionModeSingle || len(h.pcs) == 0 {
		return nil
	}
	var existing *Connection
	for _, c := range h.pcs {
		existing = c
		break
	}
	if pcID != "" && existing.PCID() != pcID {
		slog.Warn("rtc: connection pc_id mismatch", "existing", existing.PCID(), "received", pcID)
		return &RequestError{Status: http.StatusBadRequest, Detail: "PC ID mismatch with existing connection"}
	}
	if pcID == "" {
		slog.Warn("rtc: cannot create new connection: existing connection found but no pc_id received")
		return &RequestError{
			Status: http.StatusBadRequest,
			Detail: "Cannot create new connection with existing connection active",
		}
	}
	return nil
}

// HandleWebRequest answers an offer. An offer for a known pc_id renegotiates
// that connection; any other creates a new connection and passes it to
// onConnection, which should start the session in the background. An error from
// onConnection is logged, not returned: the client still gets its answer.
func (h *RequestHandler) HandleWebRequest(req Request, onConnection func(*Connection) error) (*Answer, error) {
	answer, err := h.handleWebRequest(req, onConnection)
	if err != nil {
		slog.Error("rtc: error processing request", "err", err, "pc_id", req.PCID)
	}
	return answer, err
}

func (h *RequestHandler) handleWebRequest(req Request, onConnection func(*Connection) error) (*Answer, error) {
	h.mu.Lock()
	if err := h.checkSingleConnectionConstraints(req.PCID); err != nil {
		h.mu.Unlock()
		return nil, err
	}
	var existing *Connection
	if req.PCID != "" {
		existing = h.pcs[req.PCID]
	}
	iceServers := h.cfg.ICEServers
	h.mu.Unlock()

	offer := webrtc.SessionDescription{Type: webrtc.NewSDPType(req.Type), SDP: req.SDP}

	conn := existing
	if conn != nil {
		slog.Info("rtc: reusing existing connection", "pc_id", req.PCID)
		if err := conn.Renegotiate(offer, req.RestartPC); err != nil {
			return nil, err
		}
	} else {
		var err error
		if conn, err = h.newConnection(offer, iceServers, onConnection); err != nil {
			return nil, err
		}
	}

	answer := conn.GetAnswer()
	if answer == nil {
		return nil, ErrNoAnswer
	}
	if h.cfg.ESP32Mode {
		answer.SDP = SDPMunging(answer.SDP, h.cfg.Host)
	}

	h.mu.Lock()
	h.pcs[answer.PCID] = conn
	h.mu.Unlock()
	return answer, nil
}

// newConnection creates a connection for a first offer, forgets it again when
// it closes, and hands it to onConnection.
func (h *RequestHandler) newConnection(
	offer webrtc.SessionDescription, iceServers []webrtc.ICEServer, onConnection func(*Connection) error,
) (*Connection, error) {
	conn, err := NewConnection(WithICEServers(iceServers...))
	if err != nil {
		return nil, err
	}
	if err := conn.Initialize(offer); err != nil {
		_ = conn.Close()
		return nil, err
	}
	conn.OnClosed(func() {
		pcID := conn.PCID()
		slog.Info("rtc: discarding peer connection", "pc_id", pcID)
		h.mu.Lock()
		delete(h.pcs, pcID)
		h.mu.Unlock()
	})
	if err := onConnection(conn); err != nil {
		slog.Error("rtc: connection callback failed", "pc_id", conn.PCID(), "err", err)
	} else {
		slog.Debug("rtc: connection callback executed", "pc_id", conn.PCID())
	}
	return conn, nil
}

// HandlePatchRequest adds the ICE candidates a client trickled in to the
// connection they belong to.
func (h *RequestHandler) HandlePatchRequest(req PatchRequest) error {
	h.mu.Lock()
	conn := h.pcs[req.PCID]
	h.mu.Unlock()
	if conn == nil {
		return &RequestError{Status: http.StatusNotFound, Detail: "Peer connection not found"}
	}
	for _, c := range req.Candidates {
		// An empty candidate passes through as the end-of-candidates marker
		// for its media line.
		mid := c.SDPMid
		index := c.SDPMLineIndex
		init := webrtc.ICECandidateInit{Candidate: c.Candidate, SDPMid: &mid, SDPMLineIndex: &index}
		if err := conn.AddICECandidate(init); err != nil {
			return fmt.Errorf("add ICE candidate: %w", err)
		}
	}
	return nil
}

// Close disconnects every connection and forgets them.
func (h *RequestHandler) Close() error {
	h.mu.Lock()
	conns := make([]*Connection, 0, len(h.pcs))
	for _, c := range h.pcs {
		conns = append(conns, c)
	}
	h.pcs = map[string]*Connection{}
	h.mu.Unlock()

	var wg sync.WaitGroup
	for _, c := range conns {
		wg.Go(func() { _ = c.Disconnect() })
	}
	wg.Wait()
	return nil
}
