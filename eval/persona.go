package eval

import (
	"encoding/json"
	"strings"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
)

// The simulated caller: who they are, what they want, and how they end the
// call.
//
// A Persona is the instruction the persona LLM runs under, the LLM service
// itself, which rides in the eval pipeline, and the context it runs on, where
// the bot's turns are the user messages and its own the assistant ones. The
// context offers one tool, end_call, which the persona calls once its goal is
// achieved or clearly out of reach; its success claim is its own view, and the
// judge decides the outcome.

// EndCallFunction is the tool the persona hangs up with.
const EndCallFunction = "end_call"

// endCallSchema is the end_call tool the persona's context offers.
//
//nolint:gochecknoglobals // a fixed tool description, read only
var endCallSchema = frames.Tool{
	Name: EndCallFunction,
	Description: "Hang up the phone. This is the only way to end the call; saying goodbye " +
		"does not end it. Call it as soon as your goal is achieved, or as soon as " +
		"it is clear the assistant cannot help you achieve it. A short goodbye may " +
		"be spoken in the same turn.",
	Parameters: json.RawMessage(`{"type":"object","properties":{` +
		`"success":{"type":"boolean","description":"Whether you achieved your goal on this call."},` +
		`"reason":{"type":"string","description":"One sentence on why you are ending the call."}},` +
		`"required":["success","reason"]}`),
}

// personaInstructionTemplate is the persona LLM's system instruction, with the
// persona, the goal and the end_call tool's name to fill in.
const personaInstructionTemplate = "You are playing a person on a phone call with a voice assistant. Stay in " +
	"character throughout; never mention being simulated, a test, or an AI.\n\n" +
	"Who you are: {persona}\n\n" +
	"What you want from this call: {goal}\n\n" +
	"The assistant's words arrive as the user's messages. Reply with only what you " +
	"would say next: one short spoken turn, in the first person, in plain sentences " +
	"(no lists, markdown, or stage directions). Ask for or give one thing at a time, " +
	"as a real caller would, and do not repeat what the assistant has already " +
	"understood.\n\n" +
	"Saying goodbye does not hang up: the call ends only when you call the " +
	"{end_call} tool. As soon as your goal is achieved, or it is clear the " +
	"assistant cannot help, call {end_call} with whether you succeeded and why. You " +
	"may say one short goodbye sentence in the same turn, but you must make the call."

// PersonaLLM is the LLM service that plays the caller: a processor in the eval
// pipeline, on which the end_call tool is registered. Every jargo LLM service
// built on llm.Base is one.
type PersonaLLM interface {
	processor.Processor
	RegisterFunction(name string, h llm.FunctionCallHandler, opts ...llm.RegisterOption)
}

// Persona is the simulated caller: its instruction, its LLM, and the context it
// runs on. It is safe for concurrent use.
type Persona struct {
	description string
	goal        string
	llm         PersonaLLM
	context     *frames.LLMContext

	mu sync.Mutex
	// heard is what the persona has heard of the bot's current turn (text
	// mode).
	heard           []string
	callsInProgress int
	hungUp          bool
}

// NewPersona builds the persona: description is who the caller is, as free
// text, goal what the caller wants from the call, and service the LLM that
// plays the caller, run inside the eval pipeline.
func NewPersona(description, goal string, service PersonaLLM) *Persona {
	context := frames.NewLLMContext("")
	context.SetTools([]frames.Tool{endCallSchema})
	return &Persona{description: description, goal: goal, llm: service, context: context}
}

// Instruction is the system instruction the persona LLM runs under.
func (p *Persona) Instruction() string {
	return strings.NewReplacer(
		"{persona}", p.description, "{goal}", p.goal, "{end_call}", EndCallFunction,
	).Replace(personaInstructionTemplate)
}

// LLM is the LLM service that plays the caller, a processor in the eval
// pipeline.
func (p *Persona) LLM() PersonaLLM { return p.llm }

// Context is the context the persona runs on, with the end_call tool; empty
// until the call starts.
func (p *Persona) Context() *frames.LLMContext { return p.context }

// HungUp reports whether the persona has ended its part of the call.
func (p *Persona) HungUp() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.hungUp
}

// HangUp ends the persona's part of the call: it answers nothing more.
func (p *Persona) HangUp() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.hungUp = true
}

// Hear takes in one of the bot's events, and returns the frame that has the
// persona answer once the bot's turn is done, or nil.
//
// A response the bot gives while one of its function calls is still running is
// held and joined with the response after the call, so the persona answers the
// bot's whole turn rather than its "let me check". In audio mode the aggregator
// does this instead.
func (p *Persona) Hear(event map[string]any) *frames.LLMContextFrame {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.hungUp {
		return nil
	}
	switch event[keyType] {
	case EventFunctionCall:
		p.callsInProgress++
	case EventFunctionCallStopped:
		p.callsInProgress = max(0, p.callsInProgress-1)
	case EventLLMResponse:
		if text, _ := event[keyText].(string); text != "" {
			p.heard = append(p.heard, text)
		}
		if p.callsInProgress > 0 || len(p.heard) == 0 {
			return nil
		}
		text := strings.Join(p.heard, " ")
		p.heard = nil
		p.context.AddMessage(frames.Message{Role: frames.RoleUser, Text: text})
		return frames.NewLLMContextFrame(p.context)
	}
	return nil
}

// OnEndCall registers what happens when the persona calls end_call: handler is
// run with the call's params, whose success and reason arguments are the
// persona's own claim.
func (p *Persona) OnEndCall(handler llm.FunctionCallHandler) {
	p.llm.RegisterFunction(EndCallFunction, handler)
}
