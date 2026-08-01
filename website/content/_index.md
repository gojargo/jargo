---
title: jargo
layout: hextra-home
---

{{< hextra/hero-badge link="https://github.com/gojargo/jargo/releases" >}}
  <span>Early work in progress</span>
{{< /hextra/hero-badge >}}

<div class="hx:mt-6 hx:mb-6">
{{< hextra/hero-headline >}}
  Voice agents in Go
{{< /hextra/hero-headline >}}
</div>

<div class="hx:mb-12">
{{< hextra/hero-subtitle >}}
  A WebRTC-native, audio-first conversational-AI framework.&nbsp;<br class="hx:sm:block hx:hidden" />Audio in, streaming STT → LLM → TTS, audio out, with real turn-taking and barge-in.
{{< /hextra/hero-subtitle >}}
</div>

<div class="hx:mb-6">
{{< hextra/hero-button text="Get started" link="docs/getting-started/" >}}
</div>

<div class="hx:mt-6"></div>

{{< hextra/feature-grid >}}
  {{< hextra/feature-card
    title="One static binary"
    subtitle="The default build is cgo-free. Low, predictable memory, fast startup, and real concurrency for many simultaneous sessions."
  >}}
  {{< hextra/feature-card
    title="Standard WebRTC, self-hosted"
    subtitle="Plain Pion WebRTC: no hosted transport, no proprietary SDK, no cloud to sign up for. Ship the binary; the browser connects."
  >}}
  {{< hextra/feature-card
    title="Turn-taking that works"
    subtitle="Silero VAD and Smart Turn v3 run locally on ONNX, so the bot waits for a real end-of-turn instead of any pause, and the user can cut in."
  >}}
  {{< hextra/feature-card
    title="Pluggable providers"
    subtitle="Swap any STT, LLM or TTS behind a small interface. 50+ providers, plus single-model speech-to-speech."
  >}}
  {{< hextra/feature-card
    title="Concurrent by design"
    subtitle="Independent processors, each on its own goroutine. Interruptions are frames, so barge-in reaches every stage at once."
  >}}
  {{< hextra/feature-card
    title="Telephony included"
    subtitle="Inbound and outbound phone calls over Twilio, Telnyx, Plivo and Exotel, with DTMF and an idle watchdog."
  >}}
{{< /hextra/feature-grid >}}

<div class="hx:mt-12"></div>

```go
stt := openai.NewSTT(openai.STTConfig{APIKey: key, SampleRate: opus.SampleRate})
llm := openai.NewLLM(openai.LLMConfig{APIKey: key})
tts := openai.NewTTS(openai.TTSConfig{APIKey: key})

t := rtc.NewTransport(conn, transport.DefaultParams())
agg := aggregators.New(frames.NewLLMContext("You are a helpful voice assistant."))

task := pipeline.NewTask(pipeline.New(
    t.Input(), stt, agg.User(), llm, tts, t.Output(), agg.Assistant(),
), pipeline.TaskParams{})
task.Run(ctx)
```
