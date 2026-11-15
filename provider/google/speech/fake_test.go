package speech

import (
	"context"
	"errors"
	"io"
	"net"
	"testing"
	"time"

	"cloud.google.com/go/speech/apiv2/speechpb"
	"cloud.google.com/go/texttospeech/apiv1/texttospeechpb"
	"google.golang.org/api/option"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
)

// Fake Speech-to-Text and Text-to-Speech servers, served in process over
// bufconn, so the tests below exercise the gRPC wire the services speak.

// fakeServer serves the fake services on an in-memory listener.
type fakeServer struct {
	lis *bufconn.Listener
}

// startFakeServer serves register's services until the test ends.
func startFakeServer(t *testing.T, register func(s *grpc.Server)) *fakeServer {
	t.Helper()
	lis := bufconn.Listen(1 << 20)
	srv := grpc.NewServer()
	register(srv)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)
	return &fakeServer{lis: lis}
}

// clientOptions points a client at the server, over a connection of its own
// for every client built: closing a client closes the connection it was given.
func (f *fakeServer) clientOptions(t *testing.T) func() []option.ClientOption {
	t.Helper()
	return func() []option.ClientOption {
		conn, err := grpc.NewClient("passthrough:///bufnet",
			grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
				return f.lis.DialContext(ctx)
			}),
			grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			t.Errorf("dialing the fake server: %v", err)
			return nil
		}
		return []option.ClientOption{option.WithGRPCConn(conn)}
	}
}

// recognizeCall is one streaming recognition call as the fake server sees it.
type recognizeCall struct {
	// requests receives every request the client sent, in order.
	requests chan *speechpb.StreamingRecognizeRequest
	// halfClosed is closed when the client half-closes the call.
	halfClosed chan struct{}
	// respond sends a response to the client.
	respond chan *speechpb.StreamingRecognizeResponse
	// end ends the call with the error given, nil for a clean end.
	end chan error
}

// nextRequest waits for the next request on the call.
func (c *recognizeCall) nextRequest(t *testing.T) *speechpb.StreamingRecognizeRequest {
	t.Helper()
	select {
	case r := <-c.requests:
		return r
	case <-time.After(3 * time.Second):
		t.Fatal("no request arrived on the call")
		return nil
	}
}

// fakeSpeech is a Speech-to-Text V2 server whose calls the test drives.
type fakeSpeech struct {
	speechpb.UnimplementedSpeechServer
	calls chan *recognizeCall
}

func newFakeSpeech(t *testing.T) (*fakeSpeech, *fakeServer) {
	t.Helper()
	f := &fakeSpeech{calls: make(chan *recognizeCall, 8)}
	return f, startFakeServer(t, func(s *grpc.Server) { speechpb.RegisterSpeechServer(s, f) })
}

// nextCall waits for the client to open a call.
func (f *fakeSpeech) nextCall(t *testing.T) *recognizeCall {
	t.Helper()
	select {
	case c := <-f.calls:
		return c
	case <-time.After(3 * time.Second):
		t.Fatal("no streaming call was opened")
		return nil
	}
}

func (f *fakeSpeech) StreamingRecognize(stream speechpb.Speech_StreamingRecognizeServer) error {
	call := &recognizeCall{
		requests:   make(chan *speechpb.StreamingRecognizeRequest, 64),
		halfClosed: make(chan struct{}),
		respond:    make(chan *speechpb.StreamingRecognizeResponse, 8),
		end:        make(chan error, 1),
	}
	f.calls <- call
	go func() {
		for {
			req, err := stream.Recv()
			if err != nil {
				if errors.Is(err, io.EOF) {
					close(call.halfClosed)
				}
				return
			}
			call.requests <- req
		}
	}()
	for {
		select {
		case r := <-call.respond:
			if err := stream.Send(r); err != nil {
				return err
			}
		case err := <-call.end:
			return err
		case <-stream.Context().Done():
			return nil
		}
	}
}

// synthesizeCall is one streaming synthesis call as the fake server sees it.
type synthesizeCall struct {
	requests []*texttospeechpb.StreamingSynthesizeRequest
}

// fakeTextToSpeech is a Text-to-Speech server that answers every call with the
// same audio, in the chunks given.
type fakeTextToSpeech struct {
	texttospeechpb.UnimplementedTextToSpeechServer
	chunks [][]byte
	err    error
	calls  chan *synthesizeCall
}

func newFakeTextToSpeech(t *testing.T, chunks [][]byte, err error) (*fakeTextToSpeech, *fakeServer) {
	t.Helper()
	f := &fakeTextToSpeech{chunks: chunks, err: err, calls: make(chan *synthesizeCall, 8)}
	return f, startFakeServer(t, func(s *grpc.Server) { texttospeechpb.RegisterTextToSpeechServer(s, f) })
}

func (f *fakeTextToSpeech) StreamingSynthesize(stream texttospeechpb.TextToSpeech_StreamingSynthesizeServer) error {
	call := &synthesizeCall{}
	defer func() { f.calls <- call }()
	// The audio is sent once the client has said everything and half-closed.
	for {
		req, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return err
		}
		call.requests = append(call.requests, req)
	}
	if f.err != nil {
		return f.err
	}
	for _, chunk := range f.chunks {
		if err := stream.Send(&texttospeechpb.StreamingSynthesizeResponse{AudioContent: chunk}); err != nil {
			return err
		}
	}
	return nil
}

// waitFor polls cond until it holds, failing the test after three seconds.
func waitFor(t *testing.T, msg string, cond func() bool) {
	t.Helper()
	deadline := time.After(3 * time.Second)
	for !cond() {
		select {
		case <-deadline:
			t.Fatal(msg)
		case <-time.After(5 * time.Millisecond):
		}
	}
}
