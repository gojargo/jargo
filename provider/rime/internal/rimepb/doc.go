// Package rimepb holds the generated gRPC client for the subset of Rime's
// speech-to-text API the streaming STT service uses. speech_to_text.proto is
// the source of truth. Regenerate the *.pb.go files with:
//
//	go generate ./provider/rime/internal/rimepb
//
// Regeneration needs buf, protoc-gen-go, and protoc-gen-go-grpc on PATH.
package rimepb

//go:generate buf generate
