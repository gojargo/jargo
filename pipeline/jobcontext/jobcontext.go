// Package jobcontext describes the long-running work one worker asks another
// for: how such a job ends, and the error a job carries when it is cut short.
//
// A job is requested over the worker bus and answered the same way, so its
// vocabulary lives below both the bus and the workers and is shared by them.
package jobcontext

import "errors"

// JobStatus is how a job ended.
type JobStatus string

const (
	// JobCompleted means the work finished successfully.
	JobCompleted JobStatus = "completed"
	// JobCancelled means the requester called the work off. The spelling is the
	// protocol's.
	JobCancelled JobStatus = "cancelled" //nolint:misspell // the value carried on the wire
	// JobFailed means the work did not succeed for a reason the application
	// understands, a rejected booking or an unknown customer.
	JobFailed JobStatus = "failed"
	// JobError means the work hit something unexpected while running.
	JobError JobStatus = "error"
)

// ErrJob marks a job cut short because the worker running it failed or ran out
// of time.
var ErrJob = errors.New("job stopped before it finished")
