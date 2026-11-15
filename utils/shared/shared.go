// Package shared runs the paired operations on a resource several owners share
// once each, however many owners call them.
//
// An input and an output transport share a single client, so both set it up,
// both join its room, and both tear it down. Processors are set up and cleaned
// up concurrently, so those calls overlap.
//
// Give the resource a Resource and route the paired operations through Acquire
// and Release, and the work runs once: the first owner to acquire runs the body
// while the rest wait for it, and only the last owner to release runs the undo.
package shared

import "sync"

// Resource counts the owners holding a shared resource and serializes them. Use
// one Resource per resource, so paired operations on the same object (setting a
// client up and cleaning it up, joining a room and leaving it) count their
// owners separately.
//
// The zero value is ready to use. It is safe for concurrent use.
type Resource struct {
	mu     sync.Mutex
	owners int
	// err is what acquiring returned, if it failed, so later owners fail the
	// same way.
	err error
}

// Acquire runs build for the first owner to acquire the resource.
//
// Owners that arrive while the first is still running wait for it to finish, so
// no caller continues against a half-built resource. Later owners return nil
// without running build again.
//
// A build that fails leaves the resource unbuilt, and every owner of it is told:
// the error is returned to each one that arrives later, rather than build being
// attempted again. So the owners of a resource share its verdict, and a pair of
// processors sharing a client either both come up or both fail.
//
// The lock is not reentrant, so build must not acquire or release the same
// resource.
func (r *Resource) Acquire(build func() error) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.err != nil {
		return r.err
	}
	r.owners++
	if r.owners == 1 {
		if err := build(); err != nil {
			// Nothing was built, so nothing is owned and nothing is released
			// later.
			r.owners--
			r.err = err
			return err
		}
	}
	return nil
}

// Release runs undo for the last owner to release the resource.
//
// Every other owner returns nil without running it, as does a release that no
// acquire matched.
//
// The lock is not reentrant, so undo must not acquire or release the same
// resource.
func (r *Resource) Release(undo func() error) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.owners == 0 {
		return nil
	}
	r.owners--
	if r.owners == 0 {
		return undo()
	}
	return nil
}
