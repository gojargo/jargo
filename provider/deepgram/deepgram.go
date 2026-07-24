// Package deepgram provides Deepgram's streaming speech-to-text service (over
// the live transcription WebSocket, see stt.go) and its Aura text-to-speech
// service (see tts.go).
//
// The STT service pushes InterimTranscriptionFrames and finalized
// TranscriptionFrames downstream; a finalized transcript with Deepgram's
// speech_final marks the end of the user's turn.
package deepgram

// authToken formats the shared Deepgram Authorization header value.
func authToken(apiKey string) string { return "Token " + apiKey }
