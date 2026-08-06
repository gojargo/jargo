// Package gradium provides Gradium's speech services: streaming text-to-speech
// and real-time streaming speech-to-text, both over Gradium's WebSocket API.
// The TTS side opens a session per sentence, sends the transcript, and streams
// the raw PCM audio chunks downstream; the STT side streams 80 ms audio chunks
// and surfaces recognized text as interim and finalized transcriptions.
package gradium

// Wire-protocol message keys and values shared by the STT and TTS transports.
const (
	// msgError is the server message type reporting an error.
	msgError = "error"
	// msgType is the message discriminator key.
	msgType = "type"
	// msgAudio is the base64 PCM audio field key.
	msgAudio = "audio"
	// msgText is the transcript/text field key.
	msgText = "text"
	// msgFlushed says the buffered audio has been processed.
	msgFlushed = "flushed"
	// msgEndStream is the end-of-stream message type.
	msgEndStream = "end_of_stream"
	// keyClientReqID is the client request-id field key.
	keyClientReqID = "client_req_id"
	// encPCM is the raw PCM encoding value.
	encPCM = "pcm"
)
