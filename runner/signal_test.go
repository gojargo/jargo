//go:build unix

package runner

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"net"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/coder/websocket"
)

// signalChildEnv marks the test binary running as the runner under test.
const signalChildEnv = "JARGO_RUNNER_SIGNAL_CHILD"

// TestSecondInterruptForcesExit checks the first interrupt starts a graceful
// shutdown that waits for the sessions, and a second one ends the process even
// while a session will not wind down.
func TestSecondInterruptForcesExit(t *testing.T) {
	if os.Getenv(signalChildEnv) == "1" {
		runStuckRunner(t)
		return
	}
	cmd := exec.CommandContext(t.Context(), os.Args[0], //nolint:gosec // the test binary re-running itself
		"-test.run=^TestSecondInterruptForcesExit$", "-test.v")
	cmd.Env = append(os.Environ(), signalChildEnv+"=1")
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	ready := make(chan struct{})
	go func() {
		sc := bufio.NewScanner(stdout)
		for sc.Scan() {
			if strings.Contains(sc.Text(), "READY") {
				close(ready)
			}
		}
	}()
	select {
	case <-ready:
	case <-time.After(15 * time.Second):
		_ = cmd.Process.Kill()
		t.Fatal("the runner never became ready")
	}
	exited := make(chan struct{})
	go func() { _ = cmd.Wait(); close(exited) }()

	_ = cmd.Process.Signal(syscall.SIGINT)
	select {
	case <-exited:
		t.Fatal("the first interrupt ended the runner without waiting for its session")
	case <-time.After(500 * time.Millisecond):
	}
	_ = cmd.Process.Signal(syscall.SIGINT)
	select {
	case <-exited:
	case <-time.After(4 * time.Second):
		_ = cmd.Process.Kill()
		<-exited
		t.Error("a second interrupt did not end the runner")
	}
}

// runStuckRunner runs the runner with a session that never ends, prints READY
// once the session is up, and waits to be ended by a signal.
func runStuckRunner(t *testing.T) {
	t.Helper()
	l, err := new(net.ListenConfig).Listen(t.Context(), "tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr, ok := l.Addr().(*net.TCPAddr)
	if !ok {
		t.Fatal("not a TCP address")
	}
	port := addr.Port
	_ = l.Close()

	started := make(chan struct{})
	go func() {
		fs := flag.NewFlagSet("bot", flag.ContinueOnError)
		_ = Run(context.Background(), fs, []string{"--host", "127.0.0.1", "--port", strconv.Itoa(port)},
			func(context.Context, Arguments) error {
				close(started)
				select {} // a session that does not wind down
			})
		os.Exit(0) //nolint:forbidigo // the child process ends with the runner
	}()
	url := fmt.Sprintf("ws://127.0.0.1:%d/ws", port)
	for range 100 {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		c, resp, err := websocket.Dial(ctx, url, nil)
		cancel()
		if resp != nil && resp.Body != nil {
			_ = resp.Body.Close()
		}
		if err == nil {
			defer func() { _ = c.CloseNow() }()
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	select {
	case <-started:
	case <-time.After(10 * time.Second):
		t.Fatal("no session started")
	}
	fmt.Println("READY") //nolint:forbidigo // read by the parent test
	time.Sleep(30 * time.Second)
}
