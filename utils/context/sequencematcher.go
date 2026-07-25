package context

// opcode describes one aligned block between two token sequences a and b: tag is
// one of "equal", "replace", "delete", "insert"; a[i1:i2] aligns with b[j1:j2].
type opcode struct {
	tag            string
	i1, i2, j1, j2 int
}

// match is a matching block: a[a0:a0+size] == b[b0:b0+size].
type match struct {
	a, b, size int
}

// getOpcodes computes the alignment between token sequences a and b as a list of
// opcodes, using the longest-contiguous-matching-block diff (the same algorithm
// as a stdlib-style sequence matcher with junk detection disabled). Each opcode
// later becomes an aligned text segment.
func getOpcodes(a, b []string) []opcode {
	blocks := matchingBlocks(a, b)
	var codes []opcode
	i, j := 0, 0
	for _, bl := range blocks {
		tag := ""
		switch {
		case i < bl.a && j < bl.b:
			tag = "replace"
		case i < bl.a:
			tag = "delete"
		case j < bl.b:
			tag = "insert"
		}
		if tag != "" {
			codes = append(codes, opcode{tag, i, bl.a, j, bl.b})
		}
		i, j = bl.a+bl.size, bl.b+bl.size
		if bl.size > 0 {
			codes = append(codes, opcode{"equal", bl.a, i, bl.b, j})
		}
	}
	return codes
}

// matchingBlocks returns the list of maximal matching blocks that describe how a
// and b align, terminated by a zero-size sentinel block at (len(a), len(b), 0).
func matchingBlocks(a, b []string) []match {
	// b2j maps each element of b to the sorted list of indices where it occurs.
	b2j := make(map[string][]int, len(b))
	for j, s := range b {
		b2j[s] = append(b2j[s], j)
	}

	type span struct{ alo, ahi, blo, bhi int }
	queue := []span{{0, len(a), 0, len(b)}}
	var matches []match
	for len(queue) > 0 {
		s := queue[len(queue)-1]
		queue = queue[:len(queue)-1]
		m := findLongestMatch(a, b, b2j, s.alo, s.ahi, s.blo, s.bhi)
		if m.size > 0 {
			matches = append(matches, m)
			if s.alo < m.a && s.blo < m.b {
				queue = append(queue, span{s.alo, m.a, s.blo, m.b})
			}
			if m.a+m.size < s.ahi && m.b+m.size < s.bhi {
				queue = append(queue, span{m.a + m.size, s.ahi, m.b + m.size, s.bhi})
			}
		}
	}
	sortMatches(matches)

	// Collapse adjacent equal blocks into one.
	var nonAdjacent []match
	i1, j1, k1 := 0, 0, 0
	for _, m := range matches {
		if i1+k1 == m.a && j1+k1 == m.b {
			k1 += m.size
			continue
		}
		if k1 > 0 {
			nonAdjacent = append(nonAdjacent, match{i1, j1, k1})
		}
		i1, j1, k1 = m.a, m.b, m.size
	}
	if k1 > 0 {
		nonAdjacent = append(nonAdjacent, match{i1, j1, k1})
	}
	nonAdjacent = append(nonAdjacent, match{len(a), len(b), 0})
	return nonAdjacent
}

// findLongestMatch finds the longest matching block a[i:i+k] == b[j:j+k] within
// the windows [alo,ahi) and [blo,bhi). Ties resolve to the earliest i, then the
// earliest j.
func findLongestMatch(a, b []string, b2j map[string][]int, alo, ahi, blo, bhi int) match {
	besti, bestj, bestsize := alo, blo, 0
	j2len := map[int]int{}
	for i := alo; i < ahi; i++ {
		newj2len := map[int]int{}
		for _, j := range b2j[a[i]] {
			if j < blo {
				continue
			}
			if j >= bhi {
				break
			}
			k := j2len[j-1] + 1
			newj2len[j] = k
			if k > bestsize {
				besti, bestj, bestsize = i-k+1, j-k+1, k
			}
		}
		j2len = newj2len
	}
	// Extend over matching elements on both ends (a no-op without junk, kept for
	// fidelity with the reference algorithm).
	for besti > alo && bestj > blo && a[besti-1] == b[bestj-1] {
		besti, bestj, bestsize = besti-1, bestj-1, bestsize+1
	}
	for besti+bestsize < ahi && bestj+bestsize < bhi && a[besti+bestsize] == b[bestj+bestsize] {
		bestsize++
	}
	return match{besti, bestj, bestsize}
}

// sortMatches sorts matches by (a, b) ascending. The set is small, so an
// insertion sort keeps the dependency surface minimal and the order stable.
func sortMatches(m []match) {
	for i := 1; i < len(m); i++ {
		v := m[i]
		j := i - 1
		for j >= 0 && (m[j].a > v.a || (m[j].a == v.a && m[j].b > v.b)) {
			m[j+1] = m[j]
			j--
		}
		m[j+1] = v
	}
}
