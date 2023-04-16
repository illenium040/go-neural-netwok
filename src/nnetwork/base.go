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

	// Данные проходят по всем скрытым слоям
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

	//var nErrors = make([]*mat.Dense, n.config.HiddenLayresCount+1, n.config.HiddenLayresCount+1)
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

	// Проверяем, представляет ли значение neuralNet
	// обученную модель.
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
	// Данные проходят по всем скрытым слоям
	//hiddenLayerInput := new(mat.Dense)
	//for i := 0; i < n.config.HiddenLayresCount; i++ {
	//	if i == 0 {
	//		hiddenLayerInput.Mul(x, n.wHidden[i])
	//	} else {
	//		hiddenLayerInput.Mul(n.bHidden[i-1], n.wHidden[i])
	//	}
	//	hiddenLayerInput.Apply(func(_, col int, v float64) float64 {
	//		return v + n.bHidden[i].At(0, col)
	//	}, hiddenLayerInput)
	//	hiddenLayerInput.Apply(applySigmoid, hiddenLayerInput)
	//}
	//
	//outputLayerInput := new(mat.Dense)
	//outputLayerInput.Mul(hiddenLayerInput, n.wOut)
	//outputLayerInput.Apply(func(_, col int, v float64) float64 {
	//	return v + n.bOut.At(0, col)
	//}, outputLayerInput)
	//output.Apply(applySigmoid, outputLayerInput)

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

//func (nn *NeuralNet) backpropagate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {
//	for i := 0; i < nn.config.NumEpochs; i++ {
//
//		// Завершаем процесс прямого распространения.
//		hiddenLayerInput := new(mat.Dense)
//		hiddenLayerInput.Mul(x, wHidden)
//		addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
//		hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)
//
//		hiddenLayerActivations := new(mat.Dense)
//		applySigmoid := func(_, _ int, v float64) float64 { return Sigmoid(v) }
//		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)
//
//		outputLayerInput := new(mat.Dense)
//		outputLayerInput.Mul(hiddenLayerActivations, wOut)
//		addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
//		outputLayerInput.Apply(addBOut, outputLayerInput)
//		output.Apply(applySigmoid, outputLayerInput)
//
//		// Завершаем обратное расространение.
//		networkError := new(mat.Dense)
//		networkError.Sub(y, output)
//
//		slopeOutputLayer := new(mat.Dense)
//		applySigmoidPrime := func(_, _ int, v float64) float64 { return SigmoidPrime(v) }
//		slopeOutputLayer.Apply(applySigmoidPrime, output)
//		slopeHiddenLayer := new(mat.Dense)
//		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)
//
//		dOutput := new(mat.Dense)
//		dOutput.MulElem(networkError, slopeOutputLayer)
//		errorAtHiddenLayer := new(mat.Dense)
//		errorAtHiddenLayer.Mul(dOutput, wOut.T())
//
//		dHiddenLayer := new(mat.Dense)
//		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)
//
//		// Регулируем параметры.
//		wOutAdj := new(mat.Dense)
//		wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
//		wOutAdj.Scale(nn.config.LearningRate, wOutAdj)
//		wOut.Add(wOut, wOutAdj)
//
//		bOutAdj, err := sumAlongAxis(0, dOutput)
//		if err != nil {
//			return err
//		}
//		bOutAdj.Scale(nn.config.LearningRate, bOutAdj)
//		bOut.Add(bOut, bOutAdj)
//
//		wHiddenAdj := new(mat.Dense)
//		wHiddenAdj.Mul(x.T(), dHiddenLayer)
//		wHiddenAdj.Scale(nn.config.LearningRate, wHiddenAdj)
//		wHidden.Add(wHidden, wHiddenAdj)
//
//		bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
//		if err != nil {
//			return err
//		}
//		bHiddenAdj.Scale(nn.config.LearningRate, bHiddenAdj)
//		bHidden.Add(bHidden, bHiddenAdj)
//	}
//
//	return nil
//}
//
//func (nn *NeuralNet) train(x, y *mat.Dense) error {
//	randSource := rand.NewSource(time.Now().UnixNano())
//	randGen := rand.New(randSource)
//
//	wHidden := mat.NewDense(nn.config.InputNeurons, nn.config.HiddenNeurons, nil)
//	bHidden := mat.NewDense(1, nn.config.HiddenNeurons, nil)
//	wOut := mat.NewDense(nn.config.HiddenNeurons, nn.config.OutputNeurons, nil)
//	bOut := mat.NewDense(1, nn.config.OutputNeurons, nil)
//
//	wHiddenRaw := wHidden.RawMatrix().Data
//	bHiddenRaw := bHidden.RawMatrix().Data
//	wOutRaw := wOut.RawMatrix().Data
//	bOutRaw := bOut.RawMatrix().Data
//
//	for _, param := range [][]float64{
//		wHiddenRaw,
//		bHiddenRaw,
//		wOutRaw,
//		bOutRaw,
//	} {
//		for i := range param {
//			param[i] = randGen.Float64()
//		}
//	}
//
//	// Определяем выход сети.
//	output := new(mat.Dense)
//
//	// Используем обратное распространение для регулировки весов и смещений.
//	if err := nn.backpropagate(x, y, wHidden, bHidden, wOut, bOut, output); err != nil {
//		return err
//	}
//
//	// Определяем обученную сеть.
//	nn.wHidden = wHidden
//	nn.bHidden = bHidden
//	nn.wOut = wOut
//	nn.bOut = bOut
//
//	return nil
//}
//
//func (nn *NeuralNet) predict(x *mat.Dense) (*mat.Dense, error) {
//
//	// Проверяем, представляет ли значение neuralNet
//	// обученную модель.
//	if nn.wHidden == nil || nn.wOut == nil {
//		return nil, errors.New("the supplied weights are empty")
//	}
//	if nn.bHidden == nil || nn.bOut == nil {
//		return nil, errors.New("the supplied biases are empty")
//	}
//
//	// Определяем выход сети.
//	output := new(mat.Dense)
//
//	// Завершаем процесс прямого распространения.
//	hiddenLayerInput := new(mat.Dense)
//	hiddenLayerInput.Mul(x, nn.wHidden)
//	addBHidden := func(_, col int, v float64) float64 { return v + nn.bHidden.At(0, col) }
//	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)
//
//	hiddenLayerActivations := new(mat.Dense)
//	applySigmoid := func(_, _ int, v float64) float64 { return Sigmoid(v) }
//	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)
//
//	outputLayerInput := new(mat.Dense)
//	outputLayerInput.Mul(hiddenLayerActivations, nn.wOut)
//	addBOut := func(_, col int, v float64) float64 { return v + nn.bOut.At(0, col) }
//	outputLayerInput.Apply(addBOut, outputLayerInput)
//	output.Apply(applySigmoid, outputLayerInput)
//
//	return output, nil
//}

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func SigmoidPrime(x float64) float64 {
	return x * (1.0 - x)
}
