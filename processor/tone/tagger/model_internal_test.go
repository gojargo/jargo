package tagger

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/gojargo/jargo/internal/hfhub"
)

// modelWithConfig is a model with the config.json raw and nothing loaded.
func modelWithConfig(t *testing.T, raw string) *Model {
	t.Helper()
	m := &Model{layout: defaultLayout()}
	if err := json.Unmarshal([]byte(raw), &m.config); err != nil {
		t.Fatal(err)
	}
	return m
}

var errStop = errors.New("stop after the load")

// The release's folder for a size holds config.json, tokenizer.json and the
// ONNX file config.json names. They are asked for by name, so the folder is
// asked for in two steps: its config.json, then every file in it.
func TestEachSizeDownloadsItsOwnFolder(t *testing.T) {
	root := t.TempDir()
	folder := filepath.Join(root, "tagger", "32m")
	if err := os.MkdirAll(folder, 0o750); err != nil {
		t.Fatal(err)
	}
	config := []byte(`{"onnx": "model.int8.onnx"}`)
	if err := os.WriteFile(filepath.Join(folder, "config.json"), config, 0o600); err != nil {
		t.Fatal(err)
	}
	var asked []hfhub.Config
	var dir string
	oldDownload, oldLoad := download, loadFolder
	t.Cleanup(func() { download, loadFolder = oldDownload, oldLoad })
	download = func(_ context.Context, cfg hfhub.Config) (string, error) {
		asked = append(asked, cfg)
		return root, nil
	}
	loadFolder = func(d string, _ int) (*Model, error) {
		dir = d
		return nil, errStop
	}

	if _, err := DownloadModel(context.Background(), Size32M, Revision, 1, ""); !errors.Is(err, errStop) {
		t.Fatalf("err = %v", err)
	}

	want := [][]string{
		{"tagger/32m/config.json"},
		{"tagger/32m/config.json", "tagger/32m/tokenizer.json", "tagger/32m/model.int8.onnx"},
	}
	if len(asked) != len(want) {
		t.Fatalf("asked %+v", asked)
	}
	for i, cfg := range asked {
		if cfg.Repo != "pipecat-ai/tone-models" || cfg.Revision != "tagger-rc1" || !slices.Equal(cfg.Files, want[i]) {
			t.Errorf("asked %+v, want the files %q", cfg, want[i])
		}
	}
	if dir != folder {
		t.Errorf("loaded %q, want %q", dir, folder)
	}
}

func TestABreathKindGivenTwiceKeepsItsFirstPlaceAndLastShare(t *testing.T) {
	// Python: json.loads('{"a": 1, "b": 2, "a": 3}') == {"a": 3, "b": 2}.
	m := modelWithConfig(t, `{"breath_kinds": {"a": 1, "b": 2, "a": 3}}`)
	if want := []BreathKind{{"a", 3}, {"b", 2}}; !slices.Equal(m.BreathKinds(), want) {
		t.Errorf("breath kinds %v, want %v", m.BreathKinds(), want)
	}
	if got := modelWithConfig(t, `{"breath_kinds": null}`).BreathKinds(); len(got) != 0 {
		t.Errorf("breath kinds %v, want none", got)
	}
}

func TestABreathLimitComesFromTheConfig(t *testing.T) {
	if n, ok := modelWithConfig(t, `{"breaths_per_reply": 1}`).BreathsPerReply(); !ok || n != 1 {
		t.Errorf("limit %d, %v", n, ok)
	}
	if _, ok := modelWithConfig(t, `{}`).BreathsPerReply(); ok {
		t.Error("a limit without one in the config")
	}
}

func TestALayoutReadsOverTheDefaultsAndRefusesWhatItDoesNotKnow(t *testing.T) {
	l, err := parseLayout(json.RawMessage(`{"previous_tags": true, "max_tokens": 200}`))
	if err != nil {
		t.Fatal(err)
	}
	want := layout{ContextTurns: 1, Earlier: 1, State: true, MaxTokens: 200, PreviousTags: true}
	if l != want {
		t.Errorf("layout %+v, want %+v", l, want)
	}
	if _, err := parseLayout(json.RawMessage(`{"turns": 2}`)); err == nil {
		t.Error("an unknown setting was accepted")
	}
}

func TestRangesAreWidenedToWholeWords(t *testing.T) {
	sentence := []rune("Your code is A B 7.")
	// Tokens: "Your", " code", " is", " A", " B", " 7", ".", with the code
	// begun on " A" and carried on to " 7", and "co" alone begun mid-word.
	offsets := []offset{
		{0, 4, true},
		{4, 7, true},
		{7, 9, true},
		{9, 12, true},
		{12, 14, true},
		{14, 16, true},
		{16, 18, true},
		{18, 19, true},
		{0, 0, false},
	}
	classes := []int{outside, begin, outside, outside, begin, inside, inside, outside, inside}
	got, err := ranges(classes, offsets, sentence)
	if err != nil {
		t.Fatal(err)
	}
	if want := [][2]int{{5, 9}, {13, 18}}; !reflect.DeepEqual(got, want) {
		t.Errorf("ranges %v, want %v", got, want)
	}
	if _, err := ranges([]int{begin}, []offset{{30, 40, true}}, sentence); !errors.Is(err, errOffset) {
		t.Errorf("err = %v, want errOffset", err)
	}
}

func TestTheLikeliestAllowedClassIsChosenTheFirstOfEqualOnes(t *testing.T) {
	classes := []string{"none", "calm", "warm", "sad"}
	allowed := func(c string) bool { return c != "warm" }
	for _, c := range []struct {
		logits []float64
		want   string
	}{
		{[]float64{0, 1, 9, 1}, "calm"},
		{[]float64{2, 1, 9, 1}, ""},
		{[]float64{0, 3, 9, 3}, "calm"},
	} {
		if got, err := choose(c.logits, classes, allowed); err != nil || got != c.want {
			t.Errorf("choose(%v) = %q, %v, want %q", c.logits, got, err, c.want)
		}
	}
}
