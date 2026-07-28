// Package rivapb holds the generated gRPC clients for the subset of the NVIDIA
// Riva speech API the streaming ASR and TTS services use. The .proto files are
// the source of truth: riva.proto for recognition, riva_tts.proto for synthesis,
// and riva_audio.proto for the audio encodings they share. Regenerate the *.pb.go
// files with:
//
//	go generate ./provider/nvidia/internal/rivapb
//
// Regeneration needs buf, protoc-gen-go, and protoc-gen-go-grpc on PATH.
package rivapb

//go:generate buf generate
