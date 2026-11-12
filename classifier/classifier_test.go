package classifier_test

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/frames"
)

func question() []classifier.Named[classifier.YesNoQuestion] {
	return []classifier.Named[classifier.YesNoQuestion]{
		{Name: "answer", Question: classifier.YesNoQuestion{Instructions: "is this red?"}},
	}
}

func anyQuestion() []classifier.Named[classifier.Question] {
	return []classifier.Named[classifier.Question]{
		{Name: "answer", Question: classifier.YesNoQuestion{Instructions: "is this red?"}},
	}
}

//nolint:gochecknoglobals // test fixture
var pdf = classifier.File{ContentType: "application/pdf", Data: []byte("%PDF-1.4")}

// recorder answers every question with probability and records what it was
// given. The capabilities are the base's unless a fake overrides them.
type recorder struct {
	*classifier.Base
	probability float64
	inputs      [][]classifier.Input
	states      []any
}

func (r *recorder) Model() string { return "" }

func (r *recorder) AskQuestions(
	_ context.Context, state any, questions []classifier.Named[classifier.Question], inputs []classifier.Input,
) (map[string]classifier.Result, *frames.LLMTokenUsage, error) {
	r.inputs = append(r.inputs, inputs)
	r.states = append(r.states, state)
	results := map[string]classifier.Result{}
	for _, q := range questions {
		results[q.Name] = classifier.YesNoResult{Probability: r.probability}
	}
	return results, nil, nil
}

// textClassifier takes no inputs.
type textClassifier struct{ recorder }

func newTextClassifier() *textClassifier {
	c := &textClassifier{recorder{probability: 0.5}}
	c.Base = classifier.NewBase("TextClassifier", "", c)
	return c
}

// imageClassifier sees images.
type imageClassifier struct{ recorder }

func (*imageClassifier) SupportsImages() bool { return true }

func newImageClassifier() *imageClassifier {
	c := &imageClassifier{recorder{probability: 0.9}}
	c.Base = classifier.NewBase("ImageClassifier", "", c)
	return c
}

// anyInputClassifier sees images, hears clips and reads PDFs.
type anyInputClassifier struct{ recorder }

func (*anyInputClassifier) SupportsImages() bool { return true }
func (*anyInputClassifier) SupportsAudio() bool  { return true }
func (*anyInputClassifier) FileTypes() []string  { return []string{"application/pdf"} }

func newAnyInputClassifier() *anyInputClassifier {
	c := &anyInputClassifier{recorder{probability: 0.6}}
	c.Base = classifier.NewBase("AnyInputClassifier", "", c)
	return c
}

// audioClassifier hears clips of 1 to 3 seconds.
type audioClassifier struct {
	recorder
	sampleRate int
}

func (*audioClassifier) SupportsAudio() bool     { return true }
func (*audioClassifier) MinAudio() time.Duration { return time.Second }
func (*audioClassifier) MaxAudio() time.Duration { return 3 * time.Second }
func (c *audioClassifier) AudioSampleRate() int  { return c.sampleRate }
func (c *audioClassifier) clips() []classifier.Input {
	var out []classifier.Input
	for _, in := range c.inputs {
		var first classifier.Input
		if len(in) > 0 {
			first = in[0]
		}
		out = append(out, first)
	}
	return out
}

func newAudioClassifier(sampleRate int) *audioClassifier {
	c := &audioClassifier{sampleRate: sampleRate}
	c.probability = 0.7
	c.Base = classifier.NewBase("AudioClassifier", "", c)
	return c
}

func encoded(t *testing.T, encode func(*bytes.Buffer, image.Image) error) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 8, 8))
	for y := range 8 {
		for x := range 8 {
			img.Set(x, y, color.RGBA{R: 0xff, A: 0xff})
		}
	}
	var buf bytes.Buffer
	if err := encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func jpegBytes(t *testing.T) []byte {
	t.Helper()
	return encoded(t, func(b *bytes.Buffer, img image.Image) error { return jpeg.Encode(b, img, nil) })
}

func jpegImage(t *testing.T) classifier.Image {
	t.Helper()
	img, err := classifier.NewImage(jpegBytes(t), classifier.ImageJPEG)
	if err != nil {
		t.Fatal(err)
	}
	return img
}

func clip(secs float64, sampleRate int) classifier.Audio {
	return classifier.Audio{Audio: make([]byte, int(secs*float64(sampleRate))*2), SampleRate: sampleRate}
}

func wantClassifierError(t *testing.T, err error, match string) {
	t.Helper()
	if !errors.Is(err, classifier.ErrClassifier) || !strings.Contains(err.Error(), match) {
		t.Fatalf("err = %v, want a classifier error matching %q", err, match)
	}
}

func TestAClassifierThatCannotSeeImagesRefusesThem(t *testing.T) {
	c := newTextClassifier()

	if c.SupportsImages() {
		t.Fatal("a text classifier sees images")
	}
	_, err := c.YesNo(t.Context(), "a frame", question(), jpegImage(t))
	wantClassifierError(t, err, "cannot see images")
	if len(c.inputs) != 0 {
		t.Errorf("calls = %d, want 0", len(c.inputs))
	}
}

func TestAClassifierThatCannotSeeImagesStillAnswersText(t *testing.T) {
	results, err := newTextClassifier().YesNo(t.Context(), "hello", question())
	if err != nil {
		t.Fatal(err)
	}
	if results["answer"].Probability != 0.5 {
		t.Errorf("probability = %v, want 0.5", results["answer"].Probability)
	}
}

func TestImagesReachAClassifierThatSeesThem(t *testing.T) {
	c := newImageClassifier()
	img := jpegImage(t)

	if _, err := c.YesNo(t.Context(), "a frame", question(), img); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Ask(t.Context(), "a frame", anyQuestion(), img, img); err != nil {
		t.Fatal(err)
	}
	if _, err := c.YesNo(t.Context(), "no frame", question()); err != nil {
		t.Fatal(err)
	}

	want := [][]classifier.Input{{img}, {img, img}, {}}
	if !reflect.DeepEqual(c.inputs, want) {
		t.Errorf("inputs = %v, want %v", c.inputs, want)
	}
}

func TestAClassifierThatTakesNoFilesRefusesThem(t *testing.T) {
	c := newTextClassifier()

	if len(c.FileTypes()) != 0 {
		t.Fatalf("file types = %v, want none", c.FileTypes())
	}
	_, err := c.YesNo(t.Context(), "an invoice", question(), pdf)
	wantClassifierError(t, err, "doesn't take application/pdf files")
	if len(c.inputs) != 0 {
		t.Errorf("calls = %d, want 0", len(c.inputs))
	}
}

func TestAFileOfATypeTheClassifierDoesntTakeIsRefused(t *testing.T) {
	c := newAnyInputClassifier()
	video, err := classifier.NewFile("video/mp4", nil, "https://example.com/clip.mp4")
	if err != nil {
		t.Fatal(err)
	}

	_, err = c.YesNo(t.Context(), "a clip", question(), video)
	wantClassifierError(t, err, "doesn't take video/mp4 files")
	if len(c.inputs) != 0 {
		t.Errorf("inputs = %v, want none", c.inputs)
	}
}

func TestAFileOfATypeTheClassifierTakesReachesIt(t *testing.T) {
	c := newAnyInputClassifier()

	if _, err := c.YesNo(t.Context(), "an invoice", question(), pdf); err != nil {
		t.Fatal(err)
	}

	if want := [][]classifier.Input{{pdf}}; !reflect.DeepEqual(c.inputs, want) {
		t.Errorf("inputs = %v, want %v", c.inputs, want)
	}
}

func TestAFileHasDataOrAURLButNotBoth(t *testing.T) {
	if _, err := classifier.NewFile("application/pdf", nil, ""); !errors.Is(err, classifier.ErrInvalidInput) {
		t.Errorf("neither: err = %v", err)
	}
	if _, err := classifier.NewFile("application/pdf", []byte("x"), "https://example.com"); !errors.Is(
		err, classifier.ErrInvalidInput) {
		t.Errorf("both: err = %v", err)
	}
}

func TestAFileFromAFrameOfBytesDecodesItsDataURL(t *testing.T) {
	file, err := classifier.FileFromFrame(frames.FileRawData{
		File:     "data:application/pdf;base64," + base64.StdEncoding.EncodeToString([]byte("%PDF-1.4")),
		Type:     frames.FileSourceBytes,
		Filename: "invoice.pdf",
		Format:   "application/pdf",
	})
	if err != nil {
		t.Fatal(err)
	}

	if file.ContentType != "application/pdf" || string(file.Data) != "%PDF-1.4" || file.URL != "" {
		t.Errorf("file = %+v", file)
	}
}

func TestAFileFromAFrameOfAURLKeepsTheURL(t *testing.T) {
	file, err := classifier.FileFromFrame(frames.FileRawData{
		File: "https://example.com/invoice.pdf", Type: frames.FileSourceURL, Format: "application/pdf",
	})
	if err != nil {
		t.Fatal(err)
	}

	if file.URL != "https://example.com/invoice.pdf" || file.Data != nil {
		t.Errorf("file = %+v", file)
	}
}

func TestInputsOfEveryKindReachTheClassifierInOrder(t *testing.T) {
	c := newAnyInputClassifier()
	img, audio := jpegImage(t), clip(1, 16000)

	if _, err := c.Ask(t.Context(), "a ticket", anyQuestion(), audio, pdf, img); err != nil {
		t.Fatal(err)
	}

	if want := [][]classifier.Input{{audio, pdf, img}}; !reflect.DeepEqual(c.inputs, want) {
		t.Errorf("inputs = %v, want %v", c.inputs, want)
	}
}

func TestAClassifierWithNoInputsGetsAnEmptyList(t *testing.T) {
	c := newAnyInputClassifier()

	if _, err := c.Ask(t.Context(), "a ticket", anyQuestion()); err != nil {
		t.Fatal(err)
	}

	if len(c.inputs) != 1 || c.inputs[0] == nil || len(c.inputs[0]) != 0 {
		t.Errorf("inputs = %#v, want one empty list", c.inputs)
	}
}

func TestAClassifierThatCannotHearRefusesAClip(t *testing.T) {
	c := newTextClassifier()

	if c.SupportsAudio() {
		t.Fatal("a text classifier hears")
	}
	_, err := c.YesNo(t.Context(), "a caller", question(), clip(2, 16000))
	wantClassifierError(t, err, "cannot hear audio")
	if len(c.inputs) != 0 {
		t.Errorf("calls = %d, want 0", len(c.inputs))
	}
}

func TestAClipReachesAClassifierThatHearsWithItsState(t *testing.T) {
	c := newAudioClassifier(0)
	audio := clip(2, 16000)

	results, err := c.YesNo(t.Context(), "a caller", question(), audio)
	if err != nil {
		t.Fatal(err)
	}

	if want := map[string]classifier.YesNoResult{"answer": {Probability: 0.7}}; !reflect.DeepEqual(results, want) {
		t.Errorf("results = %v, want %v", results, want)
	}
	if want := []classifier.Input{audio}; !reflect.DeepEqual(c.clips(), want) {
		t.Errorf("clips = %v, want %v", c.clips(), want)
	}
	if want := []any{"a caller"}; !reflect.DeepEqual(c.states, want) {
		t.Errorf("states = %v, want %v", c.states, want)
	}
}

func TestAClassifierThatHearsStillAnswersWithoutAClip(t *testing.T) {
	c := newAudioClassifier(0)

	if _, err := c.Ask(t.Context(), "a caller", anyQuestion()); err != nil {
		t.Fatal(err)
	}

	if want := []classifier.Input{nil}; !reflect.DeepEqual(c.clips(), want) {
		t.Errorf("clips = %v, want %v", c.clips(), want)
	}
}

func TestAClipOfALengthTheClassifierDoesntAcceptIsRefused(t *testing.T) {
	for _, secs := range []float64{0.5, 3.5} {
		c := newAudioClassifier(0)

		_, err := c.Ask(t.Context(), "a caller", anyQuestion(), clip(secs, 16000))
		wantClassifierError(t, err, "of audio")
		if len(c.inputs) != 0 {
			t.Errorf("%vs: clips = %v, want none", secs, c.inputs)
		}
	}
}

func TestAClipIsResampledToTheClassifiersSampleRate(t *testing.T) {
	c := newAudioClassifier(16000)

	if _, err := c.Ask(t.Context(), "a caller", anyQuestion(), clip(2, 8000)); err != nil {
		t.Fatal(err)
	}

	clips := c.clips()
	if len(clips) != 1 {
		t.Fatalf("clips = %d, want 1", len(clips))
	}
	got, _ := clips[0].(classifier.Audio)
	if got.SampleRate != 16000 || math.Abs(got.Duration().Seconds()-2) > 0.05 {
		t.Errorf("clip at %d Hz for %v, want 16000 Hz for about 2s", got.SampleRate, got.Duration())
	}
}

func TestAClipKeepsItsOwnRateWhenTheClassifierTakesAny(t *testing.T) {
	c := newAudioClassifier(0)
	audio := clip(2, 8000)

	if _, err := c.Ask(t.Context(), "a caller", anyQuestion(), audio); err != nil {
		t.Fatal(err)
	}

	if want := []classifier.Input{audio}; !reflect.DeepEqual(c.clips(), want) {
		t.Errorf("clips = %v, want %v", c.clips(), want)
	}
}

func TestAClassifierAnswersBestWithItsShortestClipByDefault(t *testing.T) {
	if got := newAudioClassifier(0).PreferredAudio(); got != time.Second {
		t.Errorf("preferred = %v, want 1s", got)
	}
}

func TestAClipKnowsHowLongItIs(t *testing.T) {
	if got := clip(2.5, 16000).Duration(); got != 2500*time.Millisecond {
		t.Errorf("duration = %v, want 2.5s", got)
	}
}

// webp is a 1x1 lossless WebP image.
//
//nolint:gochecknoglobals // test fixture
var webp = []byte("RIFF\x1a\x00\x00\x00WEBPVP8L\x0d\x00\x00\x00" +
	"\x2f\x00\x00\x00\x10\x07\x10\x11\x11\x88\x88\xfe\x07\x00")

func TestTheTypeIsReadFromTheBytes(t *testing.T) {
	for want, data := range map[string][]byte{
		classifier.ImagePNG:  encoded(t, func(b *bytes.Buffer, img image.Image) error { return png.Encode(b, img) }),
		classifier.ImageJPEG: jpegBytes(t),
		classifier.ImageWebP: webp,
	} {
		img, err := classifier.NewImage(data, "")
		if err != nil {
			t.Fatalf("%s: %v", want, err)
		}
		if img.ContentType != want {
			t.Errorf("content type = %q, want %q", img.ContentType, want)
		}
	}
}

func TestAGivenTypeIsKept(t *testing.T) {
	img, err := classifier.NewImage([]byte("png bytes"), classifier.ImagePNG)
	if err != nil {
		t.Fatal(err)
	}
	if img.ContentType != classifier.ImagePNG {
		t.Errorf("content type = %q", img.ContentType)
	}
}

func TestAnotherImageTypeIsAnError(t *testing.T) {
	data := encoded(t, func(b *bytes.Buffer, img image.Image) error { return gif.Encode(b, img, nil) })

	_, err := classifier.NewImage(data, "")
	if !errors.Is(err, classifier.ErrInvalidInput) || !strings.Contains(err.Error(), "PNG, JPEG or WebP, got image/gif") {
		t.Errorf("err = %v", err)
	}
}

func TestBytesThatAreNotAnImageAreAnError(t *testing.T) {
	_, err := classifier.NewImage([]byte("not an image"), "")
	if !errors.Is(err, classifier.ErrInvalidInput) || !strings.Contains(err.Error(), "not an image") {
		t.Errorf("err = %v", err)
	}
}

func TestRawPixelsAreEncodedAsAJPEG(t *testing.T) {
	pixels := bytes.Repeat([]byte{0xff, 0, 0}, 16*9)

	img, err := classifier.ImageFromFrame(frames.ImageRawData{Image: pixels, Width: 16, Height: 9, Format: "RGB"})
	if err != nil {
		t.Fatal(err)
	}

	if img.ContentType != classifier.ImageJPEG {
		t.Errorf("content type = %q", img.ContentType)
	}
	cfg, format, err := image.DecodeConfig(bytes.NewReader(img.Data))
	if err != nil || format != "jpeg" || cfg.Width != 16 || cfg.Height != 9 {
		t.Errorf("decoded %s %dx%d, err %v; want jpeg 16x9", format, cfg.Width, cfg.Height, err)
	}
}

func TestPixelsWithAlphaAreEncodedWithoutIt(t *testing.T) {
	pixels := bytes.Repeat([]byte{0xff, 0, 0, 128}, 4*4)

	img, err := classifier.ImageFromFrame(frames.ImageRawData{Image: pixels, Width: 4, Height: 4, Format: "RGBA"})
	if err != nil {
		t.Fatal(err)
	}

	decoded, err := jpeg.Decode(bytes.NewReader(img.Data))
	if err != nil {
		t.Fatal(err)
	}
	// A JPEG has no alpha: every pixel decodes opaque.
	if _, _, _, a := decoded.At(0, 0).RGBA(); a != 0xffff {
		t.Errorf("alpha = %#x, want opaque", a)
	}
}

func TestAnEncodedFrameKeepsItsBytes(t *testing.T) {
	data := encoded(t, func(b *bytes.Buffer, img image.Image) error { return png.Encode(b, img) })

	img, err := classifier.ImageFromFrame(frames.ImageRawData{Image: data, Width: 8, Height: 8, Format: "image/png"})
	if err != nil {
		t.Fatal(err)
	}

	if img.ContentType != classifier.ImagePNG || !bytes.Equal(img.Data, data) {
		t.Errorf("image = %s, %d bytes", img.ContentType, len(img.Data))
	}
}

func TestAnEncodedFrameOfAnotherTypeIsAnError(t *testing.T) {
	_, err := classifier.ImageFromFrame(frames.ImageRawData{
		Image: []byte("gif bytes"), Width: 4, Height: 4, Format: "image/gif",
	})
	if !errors.Is(err, classifier.ErrInvalidInput) {
		t.Errorf("err = %v", err)
	}
}
