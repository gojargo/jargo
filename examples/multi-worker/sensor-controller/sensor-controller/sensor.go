package main

// The temperature sensor processors of the sensor-controller example.
//
// Two custom processors give the worker pipeline a real autonomous frame flow:
//
//   - sensorReader simulates a thermometer. It runs a tick loop that moves the
//     current temperature toward the target with a first-order lag plus
//     Gaussian noise, and pushes a sensorReadingFrame on every tick. The target
//     and the response rate can be changed, so the worker's LLM can adjust them
//     through tool calls.
//   - sensorStats consumes the readings, keeps a rolling window, and exposes the
//     current, minimum, maximum and average readings and the trend. The worker's
//     LLM reads them directly when answering the user.

import (
	"context"
	"math"
	"math/rand/v2"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// sensorReadingFrame is a single temperature reading emitted by sensorReader.
type sensorReadingFrame struct {
	frames.BaseDataFrame
	// Temperature is the reading in degrees Celsius.
	Temperature float64
	// Timestamp is the Unix time the reading was taken at.
	Timestamp float64
}

func newSensorReadingFrame(temperature, timestamp float64) *sensorReadingFrame {
	return &sensorReadingFrame{
		BaseDataFrame: frames.NewBaseDataFrame("SensorReadingFrame"),
		Temperature:   temperature,
		Timestamp:     timestamp,
	}
}

// sensorReaderConfig configures a sensorReader.
type sensorReaderConfig struct {
	// StartTemp is the initial temperature and the initial target (°C).
	StartTemp float64
	// SamplePeriod is the time between successive readings.
	SamplePeriod time.Duration
	// ResponseRate is the fraction of the gap toward the target closed each
	// tick, clamped to [0, 1].
	ResponseRate float64
	// NoiseSigma is the standard deviation of the Gaussian noise added to each
	// reading.
	NoiseSigma float64
}

// defaultSensorReaderConfig is the sensor's default configuration.
func defaultSensorReaderConfig() sensorReaderConfig {
	return sensorReaderConfig{
		StartTemp:    22.0,
		SamplePeriod: time.Second,
		ResponseRate: 0.05,
		NoiseSigma:   0.1,
	}
}

// sensorReader is a simulated temperature sensor with an adjustable target and
// response rate.
//
// Each tick, the current temperature is updated as:
//
//	current += (target - current) * response_rate + gauss(0, noise_sigma)
//
// This is a first-order lag toward the target. With a response rate of 0.05
// and a one-second tick, the current reading reaches about halfway to the
// target in about 14 seconds; with a response rate of 0.2 it converges in 5 to
// 10 seconds.
type sensorReader struct {
	*processor.Base

	samplePeriod time.Duration
	noiseSigma   float64

	mu           sync.Mutex
	current      float64
	target       float64
	responseRate float64
	cancel       context.CancelFunc
	wg           sync.WaitGroup
}

func newSensorReader(cfg sensorReaderConfig) *sensorReader {
	s := &sensorReader{
		samplePeriod: cfg.SamplePeriod,
		noiseSigma:   cfg.NoiseSigma,
		current:      cfg.StartTemp,
		target:       cfg.StartTemp,
		responseRate: clamp01(cfg.ResponseRate),
	}
	s.Base = processor.New("SensorReader", s)
	return s
}

// Current is the most recent temperature reading (°C).
func (s *sensorReader) Current() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.current
}

// Target is the temperature the sensor is drifting toward (°C).
func (s *sensorReader) Target() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.target
}

// ResponseRate is the fraction of the gap between target and current closed
// per tick.
func (s *sensorReader) ResponseRate() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.responseRate
}

// SetTarget sets a new target temperature (°C).
func (s *sensorReader) SetTarget(value float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.target = value
}

// SetResponseRate sets how aggressively the sensor approaches the target, as a
// fraction in [0, 1]; it is clamped to that range.
func (s *sensorReader) SetResponseRate(rate float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.responseRate = clamp01(rate)
}

// ProcessFrame starts the tick loop on the StartFrame, and passes every frame
// on.
func (s *sensorReader) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.StartFrame); ok {
		s.startTicking(ctx)
	}
	return s.PushFrame(ctx, f, dir)
}

// Cleanup stops the tick loop.
func (s *sensorReader) Cleanup(ctx context.Context) error {
	s.mu.Lock()
	cancel := s.cancel
	s.cancel = nil
	s.mu.Unlock()
	if cancel != nil {
		cancel()
		s.wg.Wait()
	}
	return s.Base.Cleanup(ctx)
}

// startTicking starts the tick loop, once.
func (s *sensorReader) startTicking(ctx context.Context) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cancel != nil {
		return
	}
	tickCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	s.cancel = cancel
	s.wg.Go(func() { s.tickLoop(tickCtx) })
}

// tickLoop moves the reading toward the target every sample period and pushes
// it downstream.
func (s *sensorReader) tickLoop(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-time.After(s.samplePeriod):
		}
		s.mu.Lock()
		gap := s.target - s.current
		s.current += gap*s.responseRate + rand.NormFloat64()*s.noiseSigma //nolint:gosec // simulated noise
		current := s.current
		s.mu.Unlock()
		now := float64(time.Now().UnixNano()) / float64(time.Second)
		if err := s.PushFrame(ctx, newSensorReadingFrame(current, now), processor.Downstream); err != nil {
			return
		}
	}
}

// sensorStats keeps rolling-window statistics over sensorReadingFrames.
//
// It consumes the readings as they flow downstream and exposes the rolling
// minimum, maximum, average and trend; the worker's LLM reads them directly
// when responding to the user.
type sensorStats struct {
	*processor.Base

	window int

	mu       sync.Mutex
	readings []float64
}

// newSensorStats builds the statistics over the window most recent readings.
func newSensorStats(window int) *sensorStats {
	s := &sensorStats{window: window}
	s.Base = processor.New("SensorStats", s)
	return s
}

// Current is the most recent reading, or 0 if none has been seen.
func (s *sensorStats) Current() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.readings) == 0 {
		return 0
	}
	return s.readings[len(s.readings)-1]
}

// Min is the lowest reading in the window, or 0 if none has been seen.
func (s *sensorStats) Min() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.readings) == 0 {
		return 0
	}
	lowest := s.readings[0]
	for _, r := range s.readings[1:] {
		lowest = math.Min(lowest, r)
	}
	return lowest
}

// Max is the highest reading in the window, or 0 if none has been seen.
func (s *sensorStats) Max() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.readings) == 0 {
		return 0
	}
	highest := s.readings[0]
	for _, r := range s.readings[1:] {
		highest = math.Max(highest, r)
	}
	return highest
}

// Avg is the average of the readings in the window, or 0 if none has been
// seen.
func (s *sensorStats) Avg() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.readings) == 0 {
		return 0
	}
	return sum(s.readings) / float64(len(s.readings))
}

// Trend is "rising", "falling" or "stable", from the first half of the window
// against the last half.
func (s *sensorStats) Trend() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.readings) < 4 {
		return "stable"
	}
	half := len(s.readings) / 2
	oldAvg := sum(s.readings[:half]) / float64(half)
	newAvg := sum(s.readings[half:]) / float64(len(s.readings)-half)
	diff := newAvg - oldAvg
	if math.Abs(diff) < 0.25 {
		return "stable"
	}
	if diff > 0 {
		return "rising"
	}
	return "falling"
}

// ProcessFrame records each reading, and passes every frame on.
func (s *sensorStats) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if reading, ok := f.(*sensorReadingFrame); ok {
		s.mu.Lock()
		s.readings = append(s.readings, reading.Temperature)
		if len(s.readings) > s.window {
			s.readings = s.readings[len(s.readings)-s.window:]
		}
		s.mu.Unlock()
	}
	return s.PushFrame(ctx, f, dir)
}

func sum(values []float64) float64 {
	total := 0.0
	for _, v := range values {
		total += v
	}
	return total
}

func clamp01(v float64) float64 {
	return math.Max(0, math.Min(1, v))
}
