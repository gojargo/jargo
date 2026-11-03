package rtc_test

import (
	"encoding/json"
	"errors"
	"net/http"

	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/rtc"
)

// A RequestHandler answers the offers and trickled ICE candidates a browser
// client posts, and hands each new connection to a callback that starts the
// session on it. The development runner serves these same routes; this is the
// way to serve them from an application's own HTTP server.
func ExampleRequestHandler() {
	handler := rtc.NewRequestHandler(rtc.RequestHandlerConfig{})
	defer func() { _ = handler.Close() }()

	startSession := func(conn *rtc.Connection) error {
		t := rtc.NewTransport(conn, transport.DefaultParams())
		go runPipeline(t) // the session outlives the request
		return nil
	}

	writeError := func(w http.ResponseWriter, err error) {
		if reqErr, ok := errors.AsType[*rtc.RequestError](err); ok {
			http.Error(w, reqErr.Detail, reqErr.Status)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}

	http.HandleFunc("POST /api/offer", func(w http.ResponseWriter, r *http.Request) {
		var req rtc.Request
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusUnprocessableEntity)
			return
		}
		answer, err := handler.HandleWebRequest(req, startSession)
		if err != nil {
			writeError(w, err)
			return
		}
		_ = json.NewEncoder(w).Encode(answer)
	})
	http.HandleFunc("PATCH /api/offer", func(w http.ResponseWriter, r *http.Request) {
		var req rtc.PatchRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusUnprocessableEntity)
			return
		}
		if err := handler.HandlePatchRequest(req); err != nil {
			writeError(w, err)
		}
	})
}

// runPipeline stands in for building and running the session's pipeline.
func runPipeline(*rtc.Transport) {}
