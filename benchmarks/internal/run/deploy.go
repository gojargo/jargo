package run

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
)

// The deployment protocol. A coordinating process starts the serving processes
// and drives them over their standard input and output, one command per line,
// each answered by one line:
//
//	(on starting)        ready
//	start <n> <depth>    started     n sessions are running and ready for frames
//	round                done        one frame went through every session
//	settle               settled     freed memory was handed back to the system
//	stop                 stopped     every session ended; the process exits
//
// A server that fails answers "error <reason>" and exits. The Python side
// speaks the same protocol, so either coordinator can drive either server.
const (
	cmdStart   = "start"
	cmdRound   = "round"
	cmdSettle  = "settle"
	cmdStop    = "stop"
	ansReady   = "ready"
	ansStarted = "started"
	ansDone    = "done"
	ansSettled = "settled"
	ansStopped = "stopped"
)

var (
	// errProtocol is returned when a server answers something unexpected.
	errProtocol = errors.New("deployment protocol")
	// errServer is returned when a server reports a failure.
	errServer = errors.New("server failed")
)

// Serve runs a serving process of a deployment: it holds the sessions the
// coordinator asks for, on commands read from in, and answers on out.
func Serve(ctx context.Context, in io.Reader, out io.Writer) error {
	sv := &serving{w: bufio.NewWriter(out)}
	releaseMemory()
	if err := sv.answer(ansReady); err != nil {
		return err
	}
	sc := bufio.NewScanner(in)
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) == 0 {
			continue
		}
		reply, err := sv.handle(ctx, fields)
		if err != nil {
			_ = sv.answer("error " + strings.ReplaceAll(err.Error(), "\n", " "))
			return err
		}
		if err := sv.answer(reply); err != nil {
			return err
		}
		if reply == ansStopped {
			return nil
		}
	}
	return sc.Err()
}

// serving is what a serving process holds between commands.
type serving struct {
	w       *bufio.Writer
	group   []*Session
	arrived chan struct{}
}

// answer writes one line of reply.
func (sv *serving) answer(line string) error {
	if _, err := sv.w.WriteString(line + "\n"); err != nil {
		return err
	}
	return sv.w.Flush()
}

// handle carries out one command and returns the reply to it.
func (sv *serving) handle(ctx context.Context, fields []string) (string, error) {
	switch fields[0] {
	case cmdStart:
		n, depth, err := startArgs(fields)
		if err != nil {
			return "", err
		}
		sv.arrived = make(chan struct{}, n)
		sv.group = make([]*Session, n)
		for i := range sv.group {
			sv.group[i] = NewSession(depth, sv.arrived)
		}
		for _, s := range sv.group {
			s.Start(ctx)
		}
		for _, s := range sv.group {
			if err := s.WaitStarted(ctx); err != nil {
				return "", abandon(err, sv.group...)
			}
		}
		return ansStarted, nil
	case cmdRound:
		if err := sendRound(ctx, sv.group, sv.arrived); err != nil {
			return "", abandon(err, sv.group...)
		}
		return ansDone, nil
	case cmdSettle:
		releaseMemory()
		return ansSettled, nil
	case cmdStop:
		if err := StopAll(ctx, sv.group); err != nil {
			return "", err
		}
		return ansStopped, nil
	default:
		return "", fmt.Errorf("%w: unknown command %q", errProtocol, fields[0])
	}
}

// startArgs parses "start <n> <depth>".
func startArgs(fields []string) (n, depth int, err error) {
	if len(fields) != 3 {
		return 0, 0, fmt.Errorf("%w: want start <n> <depth>", errProtocol)
	}
	n, err = strconv.Atoi(fields[1])
	if err != nil || n < 0 {
		return 0, 0, fmt.Errorf("%w: bad session count %q", errProtocol, fields[1])
	}
	depth, err = strconv.Atoi(fields[2])
	if err != nil || depth < 1 {
		return 0, 0, fmt.Errorf("%w: bad depth %q", errProtocol, fields[2])
	}
	return n, depth, nil
}

// sendRound sends one frame into every session of group and waits for all of
// them to come out.
func sendRound(ctx context.Context, group []*Session, arrived <-chan struct{}) error {
	for _, s := range group {
		s.Send(frames.NewTextFrame(text))
	}
	for range group {
		if err := Wait(ctx, arrived); err != nil {
			return err
		}
	}
	return nil
}

// server is a serving process as the coordinator sees it.
type server struct {
	cmd *exec.Cmd
	in  io.WriteCloser
	out *bufio.Scanner
}

// call sends a command and waits for its answer.
func (s *server) call(command, want string) error {
	if command != "" {
		if _, err := io.WriteString(s.in, command+"\n"); err != nil {
			return fmt.Errorf("send %q: %w", command, err)
		}
	}
	if !s.out.Scan() {
		if err := s.out.Err(); err != nil {
			return fmt.Errorf("await %q: %w", want, err)
		}
		return fmt.Errorf("%w: server exited awaiting %q", errProtocol, want)
	}
	got := s.out.Text()
	if reason, failed := strings.CutPrefix(got, "error "); failed {
		return fmt.Errorf("%w: %s", errServer, reason)
	}
	if got != want {
		return fmt.Errorf("%w: got %q, want %q", errProtocol, got, want)
	}
	return nil
}

// Deployment measures count sessions of depth processors served by processes
// serving processes, each started with argv and speaking the protocol above.
// The sessions are split as evenly as the processes allow. It reports the
// memory of the serving processes idle (baseline-B) and with every session
// running (B/deployment), the memory each session adds (B/session), the time
// until every session was ready (startup-ns/session), and rounds of one frame
// into every session (ns/round, ns/frame, frames/s), after warmup rounds.
//
// Memory is each process's proportional set size, read from
// /proc/<pid>/smaps_rollup after it has handed its freed memory back to the
// system. Pages shared between the processes, such as a shared library's, are
// split between them rather than counted once per process, so the total is what
// the deployment as a whole holds. The coordinator is not counted.
func Deployment(ctx context.Context, argv []string, processes, depth, count, warmup, rounds int) (_ Result, err error) {
	group, err := startServers(ctx, argv, processes)
	defer func() {
		// A sample that failed may have left a server mid-command, so they are
		// killed rather than waited for.
		stopServers(group, err != nil)
		if err != nil && errors.Is(ctx.Err(), context.DeadlineExceeded) {
			err = ErrTimeout
		}
	}()
	if err != nil {
		return Result{}, err
	}
	baseline, err := deploymentPSS(group)
	if err != nil {
		return Result{}, err
	}

	start := time.Now()
	err = all(group, func(i int, s *server) error {
		return s.call(fmt.Sprintf("%s %d %d", cmdStart, share(count, processes, i), depth), ansStarted)
	})
	if err != nil {
		return Result{}, err
	}
	startup := time.Since(start)

	round := func() error {
		return all(group, func(_ int, s *server) error { return s.call(cmdRound, ansDone) })
	}
	for range warmup {
		if err = round(); err != nil {
			return Result{}, err
		}
	}
	start = time.Now()
	for range rounds {
		if err = round(); err != nil {
			return Result{}, err
		}
	}
	elapsed := time.Since(start)

	if err = all(group, func(_ int, s *server) error { return s.call(cmdSettle, ansSettled) }); err != nil {
		return Result{}, err
	}
	total, err := deploymentPSS(group)
	if err != nil {
		return Result{}, err
	}
	if err = all(group, func(_ int, s *server) error { return s.call(cmdStop, ansStopped) }); err != nil {
		return Result{}, err
	}

	moved := float64(rounds) * float64(count)
	return Result{
		Iterations: rounds,
		Metrics: []Metric{
			{float64(elapsed.Nanoseconds()) / float64(rounds), "ns/round"},
			{float64(elapsed.Nanoseconds()) / moved, unitFrame},
			{moved / elapsed.Seconds(), unitFramesPerSec},
			{float64(startup.Nanoseconds()) / float64(count), "startup-ns/session"},
			{float64(baseline), "baseline-B"},
			{float64(total), "B/deployment"},
			{float64(total-baseline) / float64(count), "B/session"},
		},
	}, nil
}

// share is how many of count sessions the i-th of processes serves: an even
// split, the first ones taking one more where it does not divide.
func share(count, processes, i int) int {
	n := count / processes
	if i < count%processes {
		n++
	}
	return n
}

// startServers starts the serving processes and waits for each to be ready.
// The servers it managed to start are returned even when it fails, so they can
// be stopped.
func startServers(ctx context.Context, argv []string, processes int) ([]*server, error) {
	group := make([]*server, 0, processes)
	for range processes {
		cmd := exec.CommandContext(ctx, argv[0], argv[1:]...) //nolint:gosec // the benchmark's own executable
		cmd.Stderr = os.Stderr
		in, err := cmd.StdinPipe()
		if err != nil {
			return group, err
		}
		out, err := cmd.StdoutPipe()
		if err != nil {
			return group, err
		}
		if err := cmd.Start(); err != nil {
			return group, fmt.Errorf("start server: %w", err)
		}
		group = append(group, &server{cmd: cmd, in: in, out: bufio.NewScanner(out)})
	}
	return group, all(group, func(_ int, s *server) error { return s.call("", ansReady) })
}

// stopServers closes the servers' input, kills them when told to, and waits
// for them to exit. One that is already gone is simply reaped.
func stopServers(group []*server, kill bool) {
	for _, s := range group {
		_ = s.in.Close()
		if kill {
			_ = s.cmd.Process.Kill()
		}
	}
	for _, s := range group {
		_ = s.cmd.Wait()
	}
}

// all runs f on every server at once and returns the first error.
func all(group []*server, f func(i int, s *server) error) error {
	errs := make([]error, len(group))
	var wg sync.WaitGroup
	for i, s := range group {
		wg.Go(func() { errs[i] = f(i, s) })
	}
	wg.Wait()
	return errors.Join(errs...)
}

// deploymentPSS sums the servers' proportional set sizes.
func deploymentPSS(group []*server) (int64, error) {
	var total int64
	for _, s := range group {
		pss, err := proportionalBytes(s.cmd.Process.Pid)
		if err != nil {
			return 0, err
		}
		total += pss
	}
	return total, nil
}

// proportionalBytes reads a process's proportional set size from
// /proc/<pid>/smaps_rollup.
func proportionalBytes(pid int) (int64, error) {
	path := fmt.Sprintf("/proc/%d/smaps_rollup", pid)
	data, err := os.ReadFile(path) //nolint:gosec // the pid of a process this one started
	if err != nil {
		return 0, fmt.Errorf("read proportional set size: %w", err)
	}
	sc := bufio.NewScanner(bytes.NewReader(data))
	for sc.Scan() {
		rest, ok := bytes.CutPrefix(sc.Bytes(), []byte("Pss:"))
		if !ok {
			continue
		}
		// The value is in kibibytes: "Pss:     12345 kB".
		fields := bytes.Fields(rest)
		if len(fields) == 0 {
			break
		}
		kib, err := strconv.ParseInt(string(fields[0]), 10, 64)
		if err != nil {
			return 0, fmt.Errorf("parse proportional set size: %w", err)
		}
		return kib * 1024, nil
	}
	return 0, fmt.Errorf("%w: no Pss line in smaps_rollup", errProtocol)
}
