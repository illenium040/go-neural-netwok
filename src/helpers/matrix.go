package helpers

import (
	"gonum.org/v1/gonum/mat"
	"math/rand"
	"time"
)

func FillWithRandomFloats(matrix ...*mat.Dense) {
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	for _, m := range matrix {
		rawMatrix := m.RawMatrix().Data
		for i := range rawMatrix {
			rawMatrix[i] = randGen.Float64()
		}
	}
}
