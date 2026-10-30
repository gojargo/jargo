package tagger

import (
	"crypto/sha512"
	"errors"
	"math"
	"math/bits"
)

// The kinds of breath are drawn as the reference implementation draws them,
// with Python's random.Random seeded by the call so far, so a call draws the
// same kinds wherever it is replayed: the Mersenne Twister (MT19937), seeded
// from a text by the text's UTF-8 bytes followed by their SHA-512, read as one
// big-endian integer.

var (
	errNoWeights      = errors.New("tagger: the number of weights does not match the population")
	errNonPositive    = errors.New("tagger: total of weights must be greater than zero")
	errInfiniteWeight = errors.New("tagger: total of weights must be finite")
)

const (
	mtN       = 624
	mtM       = 397
	mtMatrixA = 0x9908b0df
	mtUpper   = 0x80000000
	mtLower   = 0x7fffffff
)

// draw is a Mersenne Twister, as Python's random.Random.
type draw struct {
	mt    [mtN]uint32
	index int
}

// newDraw seeds a generator with a text, as random.Random(text) does.
func newDraw(seed string) *draw {
	b := []byte(seed)
	sum := sha512.Sum512(b)
	b = append(b, sum[:]...)
	// The integer's 32-bit words, least significant first.
	for len(b) > 0 && b[0] == 0 {
		b = b[1:]
	}
	key := []uint32{0}
	if len(b) > 0 {
		bitLen := 8*len(b) - bits.LeadingZeros8(b[0])
		key = make([]uint32, (bitLen-1)/32+1)
		for i := range key {
			for k := range 4 {
				if at := len(b) - 1 - (4*i + k); at >= 0 {
					key[i] |= uint32(b[at]) << (8 * k)
				}
			}
		}
	}
	d := &draw{}
	d.initByArray(key)
	return d
}

func (d *draw) initGenrand(s uint32) {
	d.mt[0] = s
	for i := 1; i < mtN; i++ {
		d.mt[i] = 1812433253*(d.mt[i-1]^(d.mt[i-1]>>30)) + uint32(i) //nolint:gosec // i < 624
	}
	d.index = mtN
}

func (d *draw) initByArray(key []uint32) {
	d.initGenrand(19650218)
	i, j := 1, 0
	for k := max(mtN, len(key)); k > 0; k-- {
		d.mt[i] = (d.mt[i] ^ ((d.mt[i-1] ^ (d.mt[i-1] >> 30)) * 1664525)) + key[j] + uint32(j) //nolint:gosec // j < len(key)
		i++
		j++
		if i >= mtN {
			d.mt[0] = d.mt[mtN-1]
			i = 1
		}
		if j >= len(key) {
			j = 0
		}
	}
	for k := mtN - 1; k > 0; k-- {
		d.mt[i] = (d.mt[i] ^ ((d.mt[i-1] ^ (d.mt[i-1] >> 30)) * 1566083941)) - uint32(i) //nolint:gosec // i < 624
		i++
		if i >= mtN {
			d.mt[0] = d.mt[mtN-1]
			i = 1
		}
	}
	d.mt[0] = 0x80000000
}

func (d *draw) uint32() uint32 {
	mag := [2]uint32{0, mtMatrixA}
	if d.index >= mtN {
		var kk int
		for kk = 0; kk < mtN-mtM; kk++ {
			y := (d.mt[kk] & mtUpper) | (d.mt[kk+1] & mtLower)
			d.mt[kk] = d.mt[kk+mtM] ^ (y >> 1) ^ mag[y&1]
		}
		for ; kk < mtN-1; kk++ {
			y := (d.mt[kk] & mtUpper) | (d.mt[kk+1] & mtLower)
			d.mt[kk] = d.mt[kk+(mtM-mtN)] ^ (y >> 1) ^ mag[y&1]
		}
		y := (d.mt[mtN-1] & mtUpper) | (d.mt[0] & mtLower)
		d.mt[mtN-1] = d.mt[mtM-1] ^ (y >> 1) ^ mag[y&1]
		d.index = 0
	}
	y := d.mt[d.index]
	d.index++
	y ^= y >> 11
	y ^= (y << 7) & 0x9d2c5680
	y ^= (y << 15) & 0xefc60000
	y ^= y >> 18
	return y
}

// random is random.random(): a float in [0, 1) from 53 random bits.
func (d *draw) random() float64 {
	a, b := d.uint32()>>5, d.uint32()>>6
	return (float64(a)*67108864.0 + float64(b)) * (1.0 / 9007199254740992.0)
}

// choice is random.choices(population, weights)[0]: one of population, each
// as likely as its weight.
func (d *draw) choice(population []string, weights []float64) (string, error) {
	if len(weights) != len(population) || len(population) == 0 {
		return "", errNoWeights
	}
	cumulative := make([]float64, len(weights))
	var total float64
	for i, w := range weights {
		if i == 0 {
			total = w
		} else {
			total += w
		}
		cumulative[i] = total
	}
	if total <= 0 {
		return "", errNonPositive
	}
	if math.IsInf(total, 0) || math.IsNaN(total) {
		return "", errInfiniteWeight
	}
	x := d.random() * total
	// bisect_right(cumulative, x, 0, len - 1)
	lo, hi := 0, len(population)-1
	for lo < hi {
		mid := (lo + hi) / 2
		if x < cumulative[mid] {
			hi = mid
		} else {
			lo = mid + 1
		}
	}
	return population[lo], nil
}
