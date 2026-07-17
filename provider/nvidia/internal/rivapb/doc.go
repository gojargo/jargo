// Package rivapb holds the generated gRPC client for the subset of the NVIDIA
// Riva speech-recognition API used by the streaming ASR service. riva.proto is
// the source of truth; regenerate the *.pb.go files with:
//
//	go generate ./provider/nvidia/internal/rivapb
//
// Regeneration needs buf, protoc-gen-go, and protoc-gen-go-grpc on PATH.
package rivapb

//go:generate buf generate
