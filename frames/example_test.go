package frames_test

import (
	"fmt"

	"github.com/gojargo/jargo/frames"
)

// A conversation holds the system prompt and the messages exchanged so far.
// The user aggregator adds what the user says, the assistant aggregator what
// the bot answers.
func ExampleLLMContext() {
	convo := frames.NewLLMContext("You are a friendly barista.")
	convo.AddUserMessage("A small latte, please.")
	convo.AddMessage(frames.Message{Role: frames.RoleAssistant, Text: "Coming right up."})

	fmt.Println("system:", convo.System())
	for _, m := range convo.Messages() {
		fmt.Printf("%s: %s\n", m.Role, m.Text)
	}
	// Output:
	// system: You are a friendly barista.
	// user: A small latte, please.
	// assistant: Coming right up.
}
