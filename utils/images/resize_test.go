package images_test

import (
	"bytes"
	"errors"
	"testing"

	"github.com/gojargo/jargo/utils/images"
)

func TestResizeKeepsAUniformColor(t *testing.T) {
	for _, size := range [][2]int{{8, 4}, {1, 1}, {3, 7}} {
		src := bytes.Repeat([]byte{10, 200, 30}, 4*2)
		out, err := images.Resize(src, 4, 2, "RGB", size[0], size[1])
		if err != nil {
			t.Fatal(err)
		}
		want := bytes.Repeat([]byte{10, 200, 30}, size[0]*size[1])
		if !bytes.Equal(out, want) {
			t.Errorf("%dx%d: got %v, want a uniform image", size[0], size[1], out)
		}
	}
}

func TestResizeKeepsThePixelFormat(t *testing.T) {
	out, err := images.Resize(make([]byte, 2*2*4), 2, 2, "RGBA", 3, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3*5*4 {
		t.Errorf("%d bytes, want %d", len(out), 3*5*4)
	}
}

func TestResizeRejectsWhatItCannotRead(t *testing.T) {
	if _, err := images.Resize([]byte("png"), 2, 2, "image/png", 4, 4); !errors.Is(err, images.ErrPixelFormat) {
		t.Errorf("encoded image: err %v, want ErrPixelFormat", err)
	}
	if _, err := images.Resize([]byte{1, 2}, 2, 2, "RGB", 4, 4); !errors.Is(err, images.ErrPixelSize) {
		t.Errorf("short pixels: err %v, want ErrPixelSize", err)
	}
}
