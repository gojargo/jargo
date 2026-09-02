// Package framepb holds the generated types for the protobuf frame format a
// WebSocket client speaks. frames.proto is the source of truth; regenerate the
// *.pb.go files with:
//
//	go generate ./transport/wsserver/protobuf/internal/framepb
//
// Regeneration needs buf and protoc-gen-go on PATH.
package framepb

//go:generate buf generate
