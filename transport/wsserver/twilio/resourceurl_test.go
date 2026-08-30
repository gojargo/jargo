package twilio

import (
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
)

// TestCallResourceURL covers the host the hang-up is addressed to: Twilio's
// global one by default, a regional edge when both halves are named, and an API
// root given outright for a Twilio-compatible or self-hosted backend.
func TestCallResourceURL(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
		want string
	}{
		{
			name: "the global host by default",
			cfg:  Config{AccountSID: "ACxxxx"},
			want: "https://api.twilio.com/2010-04-01/Accounts/ACxxxx/Calls/CAyyyy.json",
		},
		{
			name: "an edge in a region",
			cfg:  Config{AccountSID: "ACxxxx", Region: "au1", Edge: "sydney"},
			want: "https://api.sydney.au1.twilio.com/2010-04-01/Accounts/ACxxxx/Calls/CAyyyy.json",
		},
		{
			name: "a base URL replaces the host, region and edge included",
			cfg: Config{
				AccountSID: "ACxxxx",
				BaseURL:    "https://api.example.test",
				Region:     "au1",
				Edge:       "sydney",
			},
			want: "https://api.example.test/2010-04-01/Accounts/ACxxxx/Calls/CAyyyy.json",
		},
		{
			name: "a trailing slash on the base URL does not double up",
			cfg:  Config{AccountSID: "ACxxxx", BaseURL: "https://api.example.test/"},
			want: "https://api.example.test/2010-04-01/Accounts/ACxxxx/Calls/CAyyyy.json",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.cfg.callResourceURL("CAyyyy"); got != tt.want {
				t.Errorf("callResourceURL() = %q, want %q", got, tt.want)
			}
		})
	}
}

// TestValidateRefusesHalfAHost checks one half of the region-and-edge host is
// refused: the two are woven into one host name, so a lone region would address
// a host that does not exist and the hang-up would quietly fail.
func TestValidateRefusesHalfAHost(t *testing.T) {
	creds := Config{AccountSID: "ACxxxx", AuthToken: "token"}
	for _, half := range []Config{{Region: "au1"}, {Edge: "sydney"}} {
		cfg := creds
		cfg.Region, cfg.Edge = half.Region, half.Edge
		if err := cfg.Validate(); !errors.Is(err, ErrRegionEdgePair) {
			t.Errorf("Validate with Region=%q Edge=%q = %v, want ErrRegionEdgePair",
				cfg.Region, cfg.Edge, err)
		}
	}
}

// TestValidateAllowsHalfAHostUnderABaseURL checks the pairing rule does not
// apply once the host is given outright: the base URL is used as it stands, so
// there is no host being built out of halves.
func TestValidateAllowsHalfAHostUnderABaseURL(t *testing.T) {
	cfg := Config{
		AccountSID: "ACxxxx",
		AuthToken:  "token",
		BaseURL:    "https://api.example.test",
		Region:     "au1",
	}
	if err := cfg.Validate(); err != nil {
		t.Errorf("Validate: %v", err)
	}
}

// TestValidateSkipsTheHostWhenNotHangingUp checks the host is only settled when
// it is going to be used.
func TestValidateSkipsTheHostWhenNotHangingUp(t *testing.T) {
	off := false
	if err := (Config{Region: "au1", AutoHangUp: &off}).Validate(); err != nil {
		t.Errorf("Validate: %v", err)
	}
}

// recordHost captures the URL a request was addressed to and answers it without
// a network round trip, so a test can see the host the hang-up was sent to
// rather than the one a redirecting client rewrote it into.
type recordHost struct{ urls chan string }

func (r recordHost) RoundTrip(req *http.Request) (*http.Response, error) {
	r.urls <- req.URL.String()
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
		Request:    req,
	}, nil
}

// TestAutoHangUpAddressesTheConfiguredEdge checks the host reaches the wire: a
// call carried on a regional edge is ended there, not at the global host, which
// would answer for a call it does not have.
func TestAutoHangUpAddressesTheConfiguredEdge(t *testing.T) {
	urls := make(chan string, 1)
	s := New(Config{
		AccountSID: "ACxxxx",
		AuthToken:  "token",
		Region:     "au1",
		Edge:       "sydney",
		HTTPClient: &http.Client{Transport: recordHost{urls: urls}},
	})
	if _, err := s.Deserialize([]byte(startMsg)); err != nil {
		t.Fatalf("deserialize start: %v", err)
	}
	if _, err := s.Serialize(frames.NewEndFrame()); err != nil {
		t.Fatalf("serialize end: %v", err)
	}

	const want = "https://api.sydney.au1.twilio.com/2010-04-01/Accounts/ACxxxx/Calls/call-1.json"
	select {
	case got := <-urls:
		if got != want {
			t.Errorf("hang-up sent to %q, want %q", got, want)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("no hang-up request arrived")
	}
}
