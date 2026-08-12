package pipeline

import "github.com/gojargo/jargo/processor"

// Observer watches frames flowing through a pipeline. See processor.Observer,
// which is where the contract lives so a processor can report every handover.
type Observer = processor.Observer

// FramePushed is one frame moving from one processor to the next.
type FramePushed = processor.FramePushed

// FrameProcessed is one frame reaching a processor.
type FrameProcessed = processor.FrameProcessed

// ProcessObserver also sees a frame as it reaches a processor.
type ProcessObserver = processor.ProcessObserver

// PipelineStartedObserver also hears that the pipeline has fully started.
type PipelineStartedObserver = processor.PipelineStartedObserver
