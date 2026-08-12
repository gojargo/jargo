package stt_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/stt"
)

// describedConnector is a connector that describes itself with whatever
// metadata a case wants published.
type describedConnector struct {
	meta stt.Metadata
}

func (c *describedConnector) Connect(context.Context, int) (stt.Stream, error) {
	return &fakeStream{}, nil
}

func (c *describedConnector) Metadata() stt.Metadata { return c.meta }

// silentConnector describes nothing, which is every connector that has not been
// measured.
type silentConnector struct{}

func (c *silentConnector) Connect(context.Context, int) (stt.Stream, error) {
	return &fakeStream{}, nil
}

// publishedTTFS is the latency a service built on conn broadcasts at start.
func publishedTTFS(t *testing.T, conn stt.Connector) time.Duration {
	t.Helper()
	mf, ok := stt.NewStream("MeasuredSTT", conn, 16000).
		ServiceMetadataFrame().(*frames.STTMetadataFrame)
	if !ok {
		t.Fatal("ServiceMetadataFrame is not an STTMetadataFrame")
	}
	return mf.TTFSP99Latency
}

// TestPublishedTTFS covers what a service is described with, which is what a
// turn strategy sizes its wait for the closing transcript by.
func TestPublishedTTFS(t *testing.T) {
	noTTFS := false
	yesTTFS := true

	cases := []struct {
		name string
		conn stt.Connector
		want time.Duration
	}{{
		// The measurement is the point: a service that carries one is described
		// with it rather than with a wait nobody measured.
		name: "measured",
		conn: &describedConnector{meta: stt.Metadata{TTFSP99: stt.DeepgramTTFSP99}},
		want: stt.DeepgramTTFSP99,
	}, {
		// Nothing to go on, so the fallback: too short truncates the tail of an
		// utterance, and that costs more than the wait does.
		name: "described but unmeasured",
		conn: &describedConnector{meta: stt.Metadata{Model: "some-model"}},
		want: stt.DefaultTTFSP99,
	}, {
		name: "undescribed",
		conn: &silentConnector{},
		want: stt.DefaultTTFSP99,
	}, {
		// The server says where the turn ended, so there is no wait between the
		// speech ending and the transcript to sit through.
		name: "turn-based",
		conn: &describedConnector{meta: stt.Metadata{SupportsTTFS: &noTTFS}},
		want: 0,
	}, {
		// Explicitly supported is the same as saying nothing about it.
		name: "supported explicitly",
		conn: &describedConnector{meta: stt.Metadata{
			SupportsTTFS: &yesTTFS, TTFSP99: stt.SonioxTTFSP99,
		}},
		want: stt.SonioxTTFSP99,
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := publishedTTFS(t, tc.conn); got != tc.want {
				t.Errorf("published TTFS = %v, want %v", got, tc.want)
			}
		})
	}
}
