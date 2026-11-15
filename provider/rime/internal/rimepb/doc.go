// Package rimepb holds the generated protobuf code for the subsets of Rime's API
// the services use: the gRPC client for the speech-to-text streaming RPC, and
// the WebSocket v1 envelopes of the text-to-speech API. speech_to_text.proto
// and text_to_speech.proto are the source of truth. Regenerate the *.pb.go
// files with:
//
//	go generate ./provider/rime/internal/rimepb
//
// Regeneration needs buf, protoc-gen-go, and protoc-gen-go-grpc on PATH.
package rimepb

//go:generate buf generate
