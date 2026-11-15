package shared_test

import (
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/utils/shared"
)

// client stands in for a client an input and an output transport share.
type client struct {
	clientRes shared.Resource
	roomRes   shared.Resource

	mu            sync.Mutex
	setups        int
	cleanups      int
	joins         int
	leaves        int
	connected     bool
	seenConnected []bool
}

func (c *client) setup() error {
	return c.clientRes.Acquire(func() error {
		time.Sleep(50 * time.Millisecond)
		c.mu.Lock()
		defer c.mu.Unlock()
		c.setups++
		c.connected = true
		return nil
	})
}

func (c *client) cleanup() error {
	return c.clientRes.Release(func() error {
		c.mu.Lock()
		defer c.mu.Unlock()
		c.cleanups++
		c.connected = false
		return nil
	})
}

// join reports what the first owner to join gets back, and "" for the rest.
func (c *client) join() (string, error) {
	var joined string
	err := c.roomRes.Acquire(func() error {
		c.mu.Lock()
		defer c.mu.Unlock()
		c.joins++
		joined = "joined"
		return nil
	})
	return joined, err
}

func (c *client) leave() error {
	return c.roomRes.Release(func() error {
		c.mu.Lock()
		defer c.mu.Unlock()
		c.leaves++
		return nil
	})
}

// ownerSetup sets up as one of several owners, recording what it then sees.
func (c *client) ownerSetup() error {
	if err := c.setup(); err != nil {
		return err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.seenConnected = append(c.seenConnected, c.connected)
	return nil
}

var errCouldNotConnect = errors.New("could not connect")

// failingClient stands in for a shared client that cannot be built.
type failingClient struct {
	res shared.Resource

	mu       sync.Mutex
	setups   int
	cleanups int
}

func (c *failingClient) setup() error {
	return c.res.Acquire(func() error {
		time.Sleep(50 * time.Millisecond)
		c.mu.Lock()
		defer c.mu.Unlock()
		c.setups++
		return errCouldNotConnect
	})
}

func (c *failingClient) cleanup() error {
	return c.res.Release(func() error {
		c.mu.Lock()
		defer c.mu.Unlock()
		c.cleanups++
		return nil
	})
}

// gather runs every call concurrently and returns their errors in order.
func gather(calls ...func() error) []error {
	errs := make([]error, len(calls))
	var wg sync.WaitGroup
	for i, call := range calls {
		wg.Go(func() { errs[i] = call() })
	}
	wg.Wait()
	return errs
}

func mustAll(t *testing.T, errs []error) {
	t.Helper()
	for _, err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
}

func TestOnlyTheFirstOwnerRunsTheBody(t *testing.T) {
	c := &client{}

	mustAll(t, gather(c.setup, c.setup, c.setup))

	if c.setups != 1 {
		t.Errorf("setups = %d, want 1", c.setups)
	}
}

// TestLaterOwnersWaitForTheFirst checks a caller does not continue against a
// half-built resource. Processors are set up concurrently, so an owner can
// arrive while another is still setting the resource up.
func TestLaterOwnersWaitForTheFirst(t *testing.T) {
	c := &client{}

	mustAll(t, gather(c.ownerSetup, c.ownerSetup))

	if len(c.seenConnected) != 2 || !c.seenConnected[0] || !c.seenConnected[1] {
		t.Errorf("owners saw connected = %v, want [true true]", c.seenConnected)
	}
}

func TestOnlyTheLastOwnerUndoesIt(t *testing.T) {
	c := &client{}
	mustAll(t, gather(c.setup, c.setup))

	mustAll(t, []error{c.cleanup()})
	if c.cleanups != 0 {
		t.Fatal("the first owner leaving must not tear down")
	}

	mustAll(t, []error{c.cleanup()})
	if c.cleanups != 1 {
		t.Errorf("cleanups = %d, want 1", c.cleanups)
	}
	if c.connected {
		t.Error("still connected after the last owner released it")
	}
}

func TestReleaseWithoutAcquireDoesNothing(t *testing.T) {
	c := &client{}

	mustAll(t, []error{c.cleanup(), c.cleanup()})

	if c.cleanups != 0 {
		t.Errorf("cleanups = %d, want 0", c.cleanups)
	}
}

// TestResourcesAreCountedSeparately checks setup/cleanup and join/leave are
// independent on the same object.
func TestResourcesAreCountedSeparately(t *testing.T) {
	c := &client{}
	mustAll(t, gather(c.setup, c.setup))

	join := func() error { _, err := c.join(); return err }
	mustAll(t, gather(join, join))
	mustAll(t, []error{c.leave()})

	if c.joins != 1 {
		t.Errorf("joins = %d, want 1", c.joins)
	}
	if c.leaves != 0 {
		t.Error("one owner still holds the room")
	}
	if c.cleanups != 0 {
		t.Error("releasing a room must not clean up the client")
	}

	mustAll(t, []error{c.leave()})
	if c.leaves != 1 {
		t.Errorf("leaves = %d, want 1", c.leaves)
	}
}

func TestInstancesCountTheirOwnOwners(t *testing.T) {
	first := &client{}
	second := &client{}

	mustAll(t, gather(first.setup, first.setup))
	mustAll(t, []error{second.setup()})

	if first.setups != 1 || second.setups != 1 {
		t.Errorf("setups = %d and %d, want 1 each", first.setups, second.setups)
	}
}

func TestFirstOwnerGetsTheReturnValue(t *testing.T) {
	c := &client{}

	if got, err := c.join(); err != nil || got != "joined" {
		t.Errorf("first join = %q, %v; want joined", got, err)
	}
	if got, err := c.join(); err != nil || got != "" {
		t.Errorf("second join = %q, %v; want nothing", got, err)
	}
}

func TestAcquiringAgainAfterReleaseRunsTheBody(t *testing.T) {
	c := &client{}

	mustAll(t, []error{c.setup(), c.cleanup(), c.setup()})

	if c.setups != 2 {
		t.Errorf("setups = %d, want 2", c.setups)
	}
	if !c.connected {
		t.Error("not connected after acquiring again")
	}
}

// TestEveryOwnerOfAResourceThatFailedIsTold checks a sibling does not carry on
// regardless, which would run against nothing.
func TestEveryOwnerOfAResourceThatFailedIsTold(t *testing.T) {
	c := &failingClient{}

	for i := range 2 {
		if err := c.setup(); !errors.Is(err, errCouldNotConnect) {
			t.Errorf("setup %d = %v, want %v", i, err, errCouldNotConnect)
		}
	}
}

func TestABodyThatFailsIsNotAttemptedAgain(t *testing.T) {
	c := &failingClient{}

	for range 2 {
		if err := c.setup(); !errors.Is(err, errCouldNotConnect) {
			t.Fatalf("setup = %v, want %v", err, errCouldNotConnect)
		}
	}

	if c.setups != 1 {
		t.Errorf("setups = %d, want 1", c.setups)
	}
}

func TestConcurrentOwnersOfAFailedResourceAllFail(t *testing.T) {
	c := &failingClient{}

	errs := gather(c.setup, c.setup)

	if c.setups != 1 {
		t.Errorf("setups = %d, want 1", c.setups)
	}
	for i, err := range errs {
		if !errors.Is(err, errCouldNotConnect) {
			t.Errorf("owner %d = %v, want %v", i, err, errCouldNotConnect)
		}
	}
}

// TestNothingIsReleasedWhenAcquiringFailed checks the undo does not run against
// a resource that was never built.
func TestNothingIsReleasedWhenAcquiringFailed(t *testing.T) {
	c := &failingClient{}

	if err := c.setup(); !errors.Is(err, errCouldNotConnect) {
		t.Fatalf("setup = %v, want %v", err, errCouldNotConnect)
	}
	if err := c.cleanup(); err != nil {
		t.Fatal(err)
	}

	if c.cleanups != 0 {
		t.Errorf("cleanups = %d, want 0", c.cleanups)
	}
}
