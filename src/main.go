package main

import (
	"encoding/csv"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"log"
	"neural-network/src/nnetwork"
	"os"
	"strconv"
)

func main() {
	f, err := os.Open("./data/train.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 7

	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	inputsData := make([]float64, 4*len(rawCSVData))
	labelsData := make([]float64, 3*len(rawCSVData))

	var inputsIndex int
	var labelsIndex int

	// Последовательно помещаем строки в inputsData.
	for idx, record := range rawCSVData {

		// Пропускаем строку заголовков.
		if idx == 0 {
			continue
		}

		// Проходимся в цикле по столбцам.
		for i, val := range record {

			// Преобразуем значение в float.
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			// Добавляем значение в labelsData при необходимости.
			if i == 4 || i == 5 || i == 6 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			// Добавляем значение в inputsData.
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}

	// Формируем матрицы.
	inputs := mat.NewDense(len(rawCSVData), 4, inputsData)
	labels := mat.NewDense(len(rawCSVData), 3, labelsData)

	// Определяем архитектуру нашей сети
	// и параметры обучения.
	config := nnetwork.NetworkConfig{
		InputNeurons:      4,
		OutputNeurons:     3,
		HiddenNeurons:     3,
		NumEpochs:         10000,
		LearningRate:      0.3,
		HiddenLayresCount: 5,
	}

	fmt.Println("Обучаем...")

	// Обучаем сеть
	network := nnetwork.NewNeuralNet(config)
	network.Init()

	err = network.Train(inputs, labels)
	if err != nil {
		log.Fatal(err)
	}

	tf, err := os.Open("./data/test.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer tf.Close()

	reader = csv.NewReader(tf)
	reader.FieldsPerRecord = 7

	rawCSVData, err = reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	inputsDataT := make([]float64, 4*len(rawCSVData))
	labelsDataT := make([]float64, 3*len(rawCSVData))

	inputsIndex = 0
	labelsIndex = 0

	for idx, record := range rawCSVData {
		if idx == 0 {
			continue
		}

		// Проходимся в цикле по столбцам.
		for i, val := range record {

			// Преобразуем значение в float.
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			// Добавляем значение в labelsData при необходимости.
			if i == 4 || i == 5 || i == 6 {
				labelsDataT[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			// Добавляем значение в inputsData.
			inputsDataT[inputsIndex] = parsedVal
			inputsIndex++
		}
	}

	testInputs := mat.NewDense(len(rawCSVData), 4, inputsDataT)
	testLabels := mat.NewDense(len(rawCSVData), 3, labelsDataT)

	rowCounts, _ := testInputs.Caps()
	for i := 0; i < rowCounts; i++ {
		dence := mat.NewDense(1, 4, mat.Row(nil, i, testInputs))
		answer := mat.NewDense(1, 3, mat.Row(nil, i, testLabels))
		predictions, err := network.Predict(dence)
		if err != nil {
			log.Fatal(err)
		}
		fa := mat.Formatted(predictions, mat.Prefix("    "), mat.Squeeze())
		fa1 := mat.Formatted(answer, mat.Prefix("    "), mat.Squeeze())
		fmt.Println(i+1, fa, "answer:", fa1)
	}
}
