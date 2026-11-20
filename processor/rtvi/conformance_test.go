package rtvi_test

import (
	"reflect"
	"strings"
	"testing"

	"github.com/gojargo/jargo/processor/rtvi"
)

// The protocol's rules for what the server accepts from a client, checked
// against the processor: each client message's example is accepted, a message
// the protocol forbids is answered with an error-response, and a field or a
// message type the server does not know is ignored or refused without ending
// the session.

// examples are a valid data payload for each client message the processor
// handles, as a client of this protocol version sends it. disconnect-bot
// carries none.
//
//nolint:gochecknoglobals // fixed table, read only
var examples = map[string]string{
	rtvi.TypeClientReady:        `{"version":"2.3.0","about":{"library":"rtvi-web","library_version":"1.16.0"}}`,
	rtvi.TypeDisconnectBot:      ``,
	rtvi.TypeClientMessage:      `{"t":"get-weather","d":{"city":"Lisbon"}}`,
	rtvi.TypeSendText:           `{"content":"What's the weather like?","options":{"run_immediately":true}}`,
	rtvi.TypeDTMF:               `{"buttons":["1","#"]}`,
	rtvi.TypeFunctionCallResult: `{"function_name":"get_weather","tool_call_id":"call_123","arguments":{"city":"Lisbon"},"result":{"temperature":21}}`, //nolint:lll // one message
	rtvi.TypeRawAudio:           `{"base64Audio":"AAAAAA==","sampleRate":16000,"numChannels":1}`,
	rtvi.TypeRawAudioBatch:      `{"base64AudioBatch":["AAAAAA==","AAAAAA=="],"sampleRate":16000,"numChannels":1}`,
}

// answersBesidesBotReady are the answers to the message with id m1, leaving
// out the bot-ready that completes a handshake.
func answersBesidesBotReady(h handled) []string {
	var out []string
	for _, msgType := range h.answers["m1"] {
		if msgType != rtvi.TypeBotReady {
			out = append(out, msgType)
		}
	}
	return out
}

// TestClientMessageExamplesAreAccepted checks the server takes every client
// message as the protocol describes it, answering none with an error.
func TestClientMessageExamplesAreAccepted(t *testing.T) {
	for msgType, data := range examples {
		t.Run(msgType, func(t *testing.T) {
			h := clientSends(t, clientMessage(msgType, data))
			if got := answersBesidesBotReady(h); len(got) != 0 || h.errors != 0 {
				t.Errorf("answered %v and %d errors, want nothing", got, h.errors)
			}
		})
	}
}

// TestUnknownFieldsAreIgnored checks a field the server does not know is
// ignored: a later protocol version can add one to a client message's data.
func TestUnknownFieldsAreIgnored(t *testing.T) {
	for msgType, data := range examples {
		if data == "" {
			continue
		}
		t.Run(msgType, func(t *testing.T) {
			withUnknown := strings.TrimSuffix(data, "}") + `,"x_unknown":1}`
			h := clientSends(t, clientMessage(msgType, withUnknown))
			if got := answersBesidesBotReady(h); len(got) != 0 || h.errors != 0 {
				t.Errorf("answered %v and %d errors, want nothing", got, h.errors)
			}
		})
	}
}

// TestMessagesTheProtocolForbidsGetAnErrorResponse checks what the protocol
// requires of a client message, the server enforces: a message without the
// data it requires, missing a required field, or with a field of the wrong
// type is answered with an error-response carrying its id.
func TestMessagesTheProtocolForbidsGetAnErrorResponse(t *testing.T) {
	forbidden := []struct{ msgType, problem, data string }{
		{rtvi.TypeClientReady, "no data", ``},
		{rtvi.TypeClientReady, "no version", `{"about":{"library":"test"}}`},
		{rtvi.TypeClientReady, "version of the wrong type", `{"version":{"unexpected":true}}`},

		{rtvi.TypeClientMessage, "no data", ``},
		{rtvi.TypeClientMessage, "no t", `{"d":{"city":"Lisbon"}}`},
		{rtvi.TypeClientMessage, "t of the wrong type", `{"t":{"unexpected":true}}`},

		{rtvi.TypeSendText, "no data", ``},
		{rtvi.TypeSendText, "no content", `{"options":{"run_immediately":true}}`},
		{rtvi.TypeSendText, "content of the wrong type", `{"content":{"unexpected":true}}`},
		{rtvi.TypeSendText, "options of the wrong type", `{"content":"Hi","options":["unexpected"]}`},

		{rtvi.TypeDTMF, "no data", ``},
		{rtvi.TypeDTMF, "no buttons", `{}`},
		{rtvi.TypeDTMF, "buttons of the wrong type", `{"buttons":{"unexpected":true}}`},

		{rtvi.TypeFunctionCallResult, "no data", ``},
		{rtvi.TypeFunctionCallResult, "no function_name", `{"tool_call_id":"c","arguments":{},"result":{}}`},
		{rtvi.TypeFunctionCallResult, "no tool_call_id", `{"function_name":"f","arguments":{},"result":{}}`},
		{rtvi.TypeFunctionCallResult, "no arguments", `{"function_name":"f","tool_call_id":"c","result":{}}`},
		{rtvi.TypeFunctionCallResult, "no result", `{"function_name":"f","tool_call_id":"c","arguments":{}}`},
		{
			rtvi.TypeFunctionCallResult, "function_name of the wrong type",
			`{"function_name":{"unexpected":true},"tool_call_id":"c","arguments":{},"result":{}}`,
		},
		{
			rtvi.TypeFunctionCallResult, "tool_call_id of the wrong type",
			`{"function_name":"f","tool_call_id":{"unexpected":true},"arguments":{},"result":{}}`,
		},
		{
			rtvi.TypeFunctionCallResult, "arguments of the wrong type",
			`{"function_name":"f","tool_call_id":"c","arguments":["unexpected"],"result":{}}`,
		},
		{
			rtvi.TypeFunctionCallResult, "result of the wrong type",
			`{"function_name":"f","tool_call_id":"c","arguments":{},"result":["unexpected"]}`,
		},

		{rtvi.TypeRawAudio, "no data", ``},
		{rtvi.TypeRawAudio, "no base64Audio", `{"sampleRate":16000,"numChannels":1}`},
		{rtvi.TypeRawAudio, "no sampleRate", `{"base64Audio":"AAAAAA==","numChannels":1}`},
		{rtvi.TypeRawAudio, "no numChannels", `{"base64Audio":"AAAAAA==","sampleRate":16000}`},
		{
			rtvi.TypeRawAudio, "base64Audio of the wrong type",
			`{"base64Audio":{"unexpected":true},"sampleRate":16000,"numChannels":1}`,
		},
		{
			rtvi.TypeRawAudio, "sampleRate of the wrong type",
			`{"base64Audio":"AAAAAA==","sampleRate":{"unexpected":true},"numChannels":1}`,
		},
		{
			rtvi.TypeRawAudio, "numChannels of the wrong type",
			`{"base64Audio":"AAAAAA==","sampleRate":16000,"numChannels":{"unexpected":true}}`,
		},

		{rtvi.TypeRawAudioBatch, "no data", ``},
		{rtvi.TypeRawAudioBatch, "no base64AudioBatch", `{"sampleRate":16000,"numChannels":1}`},
		{rtvi.TypeRawAudioBatch, "no sampleRate", `{"base64AudioBatch":["AAAAAA=="],"numChannels":1}`},
		{rtvi.TypeRawAudioBatch, "no numChannels", `{"base64AudioBatch":["AAAAAA=="],"sampleRate":16000}`},
		{
			rtvi.TypeRawAudioBatch, "base64AudioBatch of the wrong type",
			`{"base64AudioBatch":{"unexpected":true},"sampleRate":16000,"numChannels":1}`,
		},

		{"not-a-message-type", "an unknown type", ``},
	}
	for _, tc := range forbidden {
		t.Run(tc.msgType+"/"+tc.problem, func(t *testing.T) {
			h := clientSends(t, clientMessage(tc.msgType, tc.data))
			if got := answersBesidesBotReady(h); !reflect.DeepEqual(got, []string{rtvi.TypeErrorResponse}) {
				t.Errorf("answered %v, want one error-response", got)
			}
		})
	}
}

// TestClientReadyIsLenientAboutTheClient checks what the client says about
// itself is advice: without it, or with one that does not describe a client,
// the session starts as with a client of an unknown library.
func TestClientReadyIsLenientAboutTheClient(t *testing.T) {
	for problem, data := range map[string]string{
		"no about":                `{"version":"2.3.0"}`,
		"about of the wrong type": `{"version":"2.3.0","about":["unexpected"]}`,
		"about without a library": `{"version":"2.3.0","about":{"platform":"iOS"}}`,
	} {
		t.Run(problem, func(t *testing.T) {
			h := clientSends(t, clientMessage(rtvi.TypeClientReady, data))
			if got := h.answers["m1"]; !reflect.DeepEqual(got, []string{rtvi.TypeBotReady}) {
				t.Errorf("answered %v, want a bot-ready alone", got)
			}
		})
	}
}

// TestClientReadyVersionCheck checks a client-ready without a version, with a
// malformed one, or of a major version the server does not speak gets an
// error-response carrying its id, and the session goes on to a bot-ready. A
// version the server speaks gets the bot-ready alone.
func TestClientReadyVersionCheck(t *testing.T) {
	for _, tc := range []struct {
		version   string
		complains bool
	}{
		{"", true},
		{"2.x", true},
		{"2.3", true},
		{"3.0.0", true},
		{rtvi.ProtocolVersion, false},
		{"1.0.0", false},
	} {
		t.Run(tc.version, func(t *testing.T) {
			data := `{}`
			if tc.version != "" {
				data = `{"version":"` + tc.version + `"}`
			}
			want := []string{rtvi.TypeBotReady}
			if tc.complains {
				want = []string{rtvi.TypeErrorResponse, rtvi.TypeBotReady}
			}
			h := clientSends(t, clientMessage(rtvi.TypeClientReady, data))
			if got := h.answers["m1"]; !reflect.DeepEqual(got, want) {
				t.Errorf("answered %v, want %v", got, want)
			}
		})
	}
}

// TestInvalidMessageGetsError checks a message labeled as RTVI that is not an
// RTVI message, such as one without an id or with data that is not an object,
// gets an error, since there is no id to answer. A message without the label
// draws nothing.
func TestInvalidMessageGetsError(t *testing.T) {
	for name, raw := range map[string]string{
		"no id":                      `{"label":"rtvi-ai","type":"send-text","data":{"content":"Hi"}}`,
		"no type":                    `{"label":"rtvi-ai","id":"m1","data":{"content":"Hi"}}`,
		"data that is not an object": `{"label":"rtvi-ai","type":"send-text","id":"m1","data":["Hi"]}`,
	} {
		t.Run(name, func(t *testing.T) {
			h := clientSends(t, raw)
			if h.errors != 1 || len(h.answers) != 0 {
				t.Errorf("sent %d errors and answered %v, want one error alone", h.errors, h.answers)
			}
		})
	}

	h := clientSends(t, `{"label":"other","type":"send-text","id":"m1"}`)
	if h.errors != 0 || len(h.answers) != 0 {
		t.Errorf("a message for another protocol drew %d errors and %v, want nothing", h.errors, h.answers)
	}
}
