package nnetwork

import (
	"errors"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"math"
	"neural-network/src/helpers"
)

type NetworkConfig struct {
	InputNeurons      int
	OutputNeurons     int
	HiddenNeurons     int
	HiddenLayresCount int
	NumEpochs         int
	LearningRate      float64
}

type Network struct {
	config  NetworkConfig
	wHidden []*mat.Dense
	bHidden []*mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

func NewNeuralNet(config NetworkConfig) Network {
	return Network{
		config: config,
	}
}

func (n *Network) Init() {
	for i := 0; i < n.config.HiddenLayresCount; i++ {
		var neuronsCount = n.config.HiddenNeurons
		if i == 0 {
			neuronsCount = n.config.InputNeurons
		}

		n.wHidden = append(n.wHidden, mat.NewDense(neuronsCount, n.config.HiddenNeurons, nil))
		n.bHidden = append(n.bHidden, mat.NewDense(1, n.config.HiddenNeurons, nil))

		helpers.FillWithRandomFloats(n.wHidden[i], n.bHidden[i])
	}

	n.wOut = mat.NewDense(n.config.HiddenNeurons, n.config.OutputNeurons, nil)
	n.bOut = mat.NewDense(1, n.config.OutputNeurons, nil)

	helpers.FillWithRandomFloats(n.wOut, n.bOut)
}

func (n *Network) Propagate(x *mat.Dense) error {
	applySigmoid := func(_, _ int, v float64) float64 {
		return Sigmoid(v)
	}

	for i := 0; i < n.config.HiddenLayresCount; i++ {
		hiddenLayerInput := new(mat.Dense)
		if i == 0 {
			hiddenLayerInput.Mul(x, n.wHidden[i])
		} else {
			hiddenLayerInput.Mul(n.bHidden[i-1], n.wHidden[i])
		}
		hiddenLayerInput.Apply(func(_, col int, v float64) float64 {
			return v + n.bHidden[i].At(0, col)
		}, hiddenLayerInput)
		n.bHidden[i].Apply(applySigmoid, hiddenLayerInput)
	}

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(n.bHidden[n.config.HiddenLayresCount-1], n.wOut)
	outputLayerInput.Apply(func(_, col int, v float64) float64 {
		return v + n.bOut.At(0, col)
	}, outputLayerInput)
	n.bOut.Apply(applySigmoid, outputLayerInput)

	return nil
}

func (n *Network) Backpropagate(x, y *mat.Dense) error {
	applySigmoidPrime := func(_, _ int, v float64) float64 { return SigmoidPrime(v) }

	networkError := new(mat.Dense)
	networkError.Sub(y, n.bOut)

	var nError = new(mat.Dense)
	for i := n.config.HiddenLayresCount; i >= 0; i-- {
		slopeLayer := new(mat.Dense)
		dOutput := new(mat.Dense)
		if i == n.config.HiddenLayresCount {
			slopeLayer.Apply(applySigmoidPrime, n.bOut)
			dOutput.MulElem(networkError, slopeLayer)
			nError.Mul(dOutput, n.wOut.T())

			wOutAdj := new(mat.Dense)
			wOutAdj.Mul(n.bHidden[i-1].T(), dOutput)
			wOutAdj.Scale(n.config.LearningRate, wOutAdj)
			n.wOut.Add(n.wOut, wOutAdj)

			dOutput.Scale(n.config.LearningRate, dOutput)
			n.bOut.Add(n.bOut, dOutput)
		} else {
			slopeLayer.Apply(applySigmoidPrime, n.bHidden[i])
			dOutput.MulElem(nError, slopeLayer)

			wHiddenAdj := new(mat.Dense)
			if i == 0 {
				wHiddenAdj.Mul(x.T(), dOutput)
			} else {
				wHiddenAdj.Mul(n.bHidden[i-1].T(), dOutput)
			}

			wHiddenAdj.Scale(n.config.LearningRate, wHiddenAdj)
			n.wHidden[i].Add(n.wHidden[i], wHiddenAdj)

			dOutput.Scale(n.config.LearningRate, dOutput)
			n.bHidden[i].Add(n.bHidden[i], dOutput)
		}
	}
	return nil
}

func (n *Network) Train(x, y *mat.Dense) error {
	rowCount, _ := x.Caps()
	for i := 0; i < n.config.NumEpochs; i++ {
		for j := 0; j < rowCount; j++ {
			dx := mat.NewDense(1, 4, mat.Row(nil, j, x))
			err := n.Propagate(dx)
			if err != nil {
				return err
			}

			dy := mat.NewDense(1, 3, mat.Row(nil, j, y))
			err = n.Backpropagate(dx, dy)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

func (n *Network) Predict(x *mat.Dense) (*mat.Dense, error) {
	if n.wHidden == nil || n.wOut == nil {
		return nil, errors.New("the supplied weights are empty")
	}
	if n.bHidden == nil || n.bOut == nil {
		return nil, errors.New("the supplied biases are empty")
	}

	// Определяем выход сети.
	output := new(mat.Dense)

	applySigmoid := func(_, _ int, v float64) float64 {
		return Sigmoid(v)
	}

	var layers = make([]*mat.Dense, n.config.HiddenLayresCount, n.config.HiddenLayresCount)
	for i := 0; i < n.config.HiddenLayresCount; i++ {
		layers[i] = new(mat.Dense)
		if i == 0 {
			layers[i].Mul(x, n.wHidden[i])
		} else {
			layers[i].Mul(layers[i-1], n.wHidden[i])
		}
		layers[i].Apply(func(_, col int, v float64) float64 {
			return v + n.bHidden[i].At(0, col)
		}, layers[i])
		layers[i].Apply(applySigmoid, layers[i])
	}

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(layers[n.config.HiddenLayresCount-1], n.wOut)
	outputLayerInput.Apply(func(_, col int, v float64) float64 {
		return v + n.bOut.At(0, col)
	}, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)

	return output, nil
}

func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {
	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
}

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func SigmoidPrime(x float64) float64 {
	return x * (1.0 - x)
}
