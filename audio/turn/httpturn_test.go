package turn

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"
	"time"
)

// numpyReference is what NumPy writes for
// np.save(np.array([0, 1, -1, 0.5], dtype=np.float32)). The remote service loads
// its input with np.load, so the bytes have to be exactly this.
//
//nolint:gochecknoglobals // reference fixture
var numpyReference = []byte{
	147, 78, 85, 77, 80, 89, 1, 0, 118, 0,
	123, 39, 100, 101, 115, 99, 114, 39, 58, 32, 39, 60, 102, 52, 39, 44, 32, 39,
	102, 111, 114, 116, 114, 97, 110, 95, 111, 114, 100, 101, 114, 39, 58, 32, 70,
	97, 108, 115, 101, 44, 32, 39, 115, 104, 97, 112, 101, 39, 58, 32, 40, 52, 44,
	41, 44, 32, 125,
	32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
	32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
	32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
	10,
	0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 128, 191, 0, 0, 0, 63,
}

// The service reads the audio with NumPy, so what is posted has to be a NumPy
// array file byte for byte.
func TestEncodeNPYMatchesNumPy(t *testing.T) {
	got := encodeNPY([]float32{0, 1, -1, 0.5})
	if !slices.Equal(got, numpyReference) {
		t.Errorf("encodeNPY produced %d bytes:\n%v\nwant %d bytes:\n%v",
			len(got), got, len(numpyReference), numpyReference)
	}
}

// The header runs to a 64-byte boundary whatever the array's length, which is
// what NumPy's reader expects.
func TestEncodeNPYHeaderIsAligned(t *testing.T) {
	for _, n := range []int{0, 1, 4, 100, 12345, 128000} {
		got := encodeNPY(make([]float32, n))
		header := len(got) - n*4
		if header%npyAlign != 0 {
			t.Errorf("n=%d: header is %d bytes, want a multiple of %d", n, header, npyAlign)
		}
		if got[header-1] != '\n' {
			t.Errorf("n=%d: the header does not end in a newline", n)
		}
	}
}

// httpTurn builds an analyzer pointed at the handler, with a turn's worth of
// speech already buffered so a verdict can be asked for.
func httpTurn(t *testing.T, h http.HandlerFunc, params *Params) *HTTPSmartTurn {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)

	a, err := NewHTTPSmartTurn(HTTPConfig{URL: srv.URL, Params: params})
	if err != nil {
		t.Fatalf("NewHTTPSmartTurn: %v", err)
	}
	a.SetSampleRate(16000)
	a.AppendAudio(make([]byte, 3200), true)
	return a
}

// The verdict the service returns is what the analyzer reports.
func TestHTTPSmartTurnReadsTheVerdict(t *testing.T) {
	var seen []byte
	a := httpTurn(t, func(w http.ResponseWriter, r *http.Request) {
		seen = make([]byte, r.ContentLength)
		_, _ = r.Body.Read(seen)
		if ct := r.Header.Get("Content-Type"); ct != "application/octet-stream" {
			t.Errorf("Content-Type = %q, want application/octet-stream", ct)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"prediction": 1, "probability": 0.93})
	}, nil)

	state, prob, err := a.AnalyzeEndOfTurn()
	if err != nil {
		t.Fatalf("AnalyzeEndOfTurn: %v", err)
	}
	if state != Complete {
		t.Errorf("state = %v, want complete", state)
	}
	if prob != 0.93 {
		t.Errorf("probability = %v, want the one the service returned", prob)
	}
	if len(seen) < len(npyMagic) || string(seen[:len(npyMagic)]) != npyMagic {
		t.Error("the posted body is not a NumPy array file")
	}
}

// An unfinished turn stays open.
func TestHTTPSmartTurnReadsAnIncompleteVerdict(t *testing.T) {
	a := httpTurn(t, func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"prediction": 0, "probability": 0.12})
	}, nil)

	state, prob, err := a.AnalyzeEndOfTurn()
	if err != nil {
		t.Fatalf("AnalyzeEndOfTurn: %v", err)
	}
	if state != Incomplete {
		t.Errorf("state = %v, want incomplete", state)
	}
	if prob != 0.12 {
		t.Errorf("probability = %v, want the one the service returned", prob)
	}
}

// A request is given the silence window to answer in, and one that outruns it is
// an unfinished turn: the answer would be about a turn that is over by then.
func TestHTTPSmartTurnTimesOut(t *testing.T) {
	params := DefaultParams()
	params.StopSecs = 0.05
	a := httpTurn(t, func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(500 * time.Millisecond)
		_ = json.NewEncoder(w).Encode(map[string]any{"prediction": 1, "probability": 0.9})
	}, &params)

	state, prob, err := a.AnalyzeEndOfTurn()
	if err != nil {
		t.Fatalf("AnalyzeEndOfTurn: %v", err)
	}
	if state != Incomplete {
		t.Errorf("state = %v, want the turn left open", state)
	}
	if prob != 0 {
		t.Errorf("probability = %v, want none for a verdict that never arrived", prob)
	}
}

// A service that is failing leaves the turn open rather than cutting the user
// off, so the conversation falls back to the stop-seconds safety net.
func TestHTTPSmartTurnLeavesTheTurnOpenOnAFailure(t *testing.T) {
	for _, tc := range []struct {
		name    string
		handler http.HandlerFunc
	}{
		{"server error", func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte("model unavailable"))
		}},
		{"unauthorized", func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusUnauthorized)
		}},
		{"not json", func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte("<html>gateway</html>"))
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := httpTurn(t, tc.handler, nil)
			state, prob, err := a.AnalyzeEndOfTurn()
			if err != nil {
				t.Fatalf("AnalyzeEndOfTurn: %v", err)
			}
			if state != Incomplete {
				t.Errorf("state = %v, want the turn left open", state)
			}
			if prob != 0 {
				t.Errorf("probability = %v, want none for a verdict that never arrived", prob)
			}
		})
	}
}

// The configured headers reach the service, which is where an authorization
// header belongs.
func TestHTTPSmartTurnSendsItsHeaders(t *testing.T) {
	got := make(chan string, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case got <- r.Header.Get("Authorization"):
		default:
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"prediction": 1, "probability": 1.0})
	}))
	t.Cleanup(srv.Close)

	a, err := NewHTTPSmartTurn(HTTPConfig{
		URL: srv.URL, Headers: map[string]string{"Authorization": "Bearer token"},
	})
	if err != nil {
		t.Fatalf("NewHTTPSmartTurn: %v", err)
	}
	a.SetSampleRate(16000)
	a.AppendAudio(make([]byte, 3200), true)
	if _, _, err := a.AnalyzeEndOfTurn(); err != nil {
		t.Fatalf("AnalyzeEndOfTurn: %v", err)
	}
	if h := <-got; h != "Bearer token" {
		t.Errorf("Authorization = %q, want the configured header", h)
	}
}

// A URL is the one thing the analyzer cannot be built without.
func TestHTTPSmartTurnNeedsAURL(t *testing.T) {
	if _, err := NewHTTPSmartTurn(HTTPConfig{}); err == nil {
		t.Error("NewHTTPSmartTurn() = nil error with no URL")
	}
}
