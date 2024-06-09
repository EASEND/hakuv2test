go
package main

import (
    "bufio"
    "encoding/json"
    "fmt"
    "github.com/bwmarrin/discordgo"
    "github.com/nlpodyssey/spago/pkg/mat32"
    "github.com/nlpodyssey/spago/pkg/ml/nn"
    "github.com/nlpodyssey/spago/pkg/ml/nn/activations"
    "github.com/nlpodyssey/spago/pkg/ml/nn/initializers"
    "github.com/nlpodyssey/spago/pkg/ml/nn/layers"
    "github.com/nlpodyssey/spago/pkg/ml/nn/losses"
    "github.com/nlpodyssey/spago/pkg/ml/optimizers"
    "github.com/nlpodyssey/spago/pkg/ml/regimes"
    "github.com/nlpodyssey/spago/pkg/utils/files"
    "github.com/nlpodyssey/spago/pkg/utils/logs"
    "github.com/nlpodyssey/spago/pkg/utils/serialization"
    "io/ioutil"
    "log"
    "math/rand"
    "os"
    "strings"
    "time"
    "google"
    "windows"
)

type Model struct {
    Embedding  *nn.Embedding
    Encoder    *nn.Sequential
    Decoder    *nn.Sequential
    Linear     *nn.Linear
    Activation  *nn.Activation
}

func NewModel(vocabSize, embeddingSize, hiddenSize int) *Model {
    encoderLayers := []nn.Layer{
        layers.NewLSTM(embeddingSize, hiddenSize, false),
        layers.NewLSTM(hiddenSize, hiddenSize, false),
    }

    decoderLayers := []nn.Layer{
        layers.NewLSTM(embeddingSize, hiddenSize, false),
        layers.NewLSTM(hiddenSize, hiddenSize, false),
    }

    return &Model{
        Embedding: nn.NewEmbedding(vocabSize, embeddingSize, initializers.NewXavier()),
        Encoder:   nn.NewSequential(encoderLayers...),
        Decoder:   nn.NewSequential(decoderLayers...),
        Linear:    nn.NewLinear(hiddenSize, vocabSize, initializers.NewXavier()),
        Activation: nn.NewActivation(activations.Softmax),
    }
}


// Битие секторов, инструкция - 
func (m *Model) Forward(xs *mat32.Matrix, hxs, cxs *mat32.Matrix) (*mat32.Matrix, *mat32.Matrix, *mat32.Matrix) {
    xs = m.Embedding.Forward(xs)
    xs, hxs, cxs = m.Encoder.Forward(xs, hxs, cxs)
    xs, hxs, cxs = m.Decoder.Forward(xs, hxs, cxs)
    xs = m.Linear.Forward(xs)
    xs = m.Activation.Forward(xs)
    return xs, hxs, cxs
}

func (m *Model) Sample(xs *mat32.Matrix, hxs, cxs *mat32.Matrix, maxLength int) []int {
    sequence := []int{}
    for i := 0; i < maxLength; i++ {
        xs, hxs, cxs = m.Forward(xs, hxs, cxs)
        probs := xs.RawVector().Data
        nextIndex := rand.Intn(len(probs))
        for j := range probs {
            if j == nextIndex {
                break
            }
            nextIndex += int(probs[j])
        }
        sequence = append(sequence, nextIndex)
        xs = mat32.NewVector(len(probs), nil)
        xs.SetVec(i+1, probs[nextIndex])
    }
    return sequence
}

// Битие секторов, инструкция - 

func Train(model *Model, trainData [][][]int, labels [][]int, epochs int) {
    criterion := nn.NewCriterion(losses.CrossEntropy)
    optimizer := optimizers.NewAdam(model.Parameters(), 0.001)
    regime := regimes.NewExponentialLR(optimizer, 0.95, 1)

    for e := 1; e <= epochs; e++ {
        for _, xss := range trainData {
            xs := mat32.NewMatrix(1, len(xss[0]), nil)
            copy(xs.RawMatrix().Data, xss[0])

            hxs := mat32.NewMatrix(1, model.Encoder.Layers()[0].(*nn.LSTM).HiddenSize, nil)
            cxs := mat32.NewMatrix(1, model.Encoder.Layers()[0].(*nn.LSTM).HiddenSize, nil)

            optimizer.ZeroGrad()
            yss := []*mat32.Matrix{}
            for i, xs := range xss[1:] {
                copy(xs.RawMatrix().Data, xs)
                ys, hxs, cxs := model.Forward(xs, hxs, cxs)
                yss = append(yss, ys)
                if i < len(xss[1:])-1 {
                    xs = mat32.NewMatrix(1, len(xs), nil)
                }
            }
            loss := criterion.Forward(yss, mat32.NewMatrixFromRows(labels[1:]...))
            loss.Backward()
            regime.Step()

            fmt.Printf("Работает %d, Ошибка: %f\n", e, loss.Data())
        }
    }
}

func LoadVocab() (map[string]int, map[int]string) {
    file, err := os.Open("hakuv2.duri")
    if err != nil {
        log.Fatal(err)
    }
    defer file.Close()

    var vocab map[string]int
    err = json.NewDecoder(file).Decode(&vocab)
    if err != nil {
        log.Fatal(err)
    }

    reverseVocab := make(map[int]string)
    for k, v := range vocab {
        reverseVocab[v] = k
    }

    return vocab, reverseVocab
}

func LoadData(vocab map[string]int) ([][][]int, [][]int) {
    files, err := ioutil.ReadDir("data")
    if err != nil {
        log.Fatal(err)
    }

    trainData := [][][]int{}
    labels := [][]int{}

    for _, file := range files {
        if file.IsDir() {
            continue
        }

        lines, err := files.ReadLines("data/" + file.Name())
        if err != nil {
            log.Fatal(err)
        }

        input := []int{}
        for _, word := range strings.Split(lines[0], " ") {
            input = append(input, vocab[word])
        }

        output := [][]int{}
        for i := 1; i < len(lines); i++ {
            sequence := []int{}
            for _, word := range strings.Split(lines[i], " ") {
                sequence = append(sequence, vocab[word])
            }
            output = append(output, sequence)
        }

        label := []int{}
        for _, word := range strings.Split(file.Name(), ".") {
            label = append(label, vocab[word])
        }

        trainData = append(trainData, [][]int{input, output})
        labels = append(labels, label)
    }

    return trainData, labels
}


// Проксима
func main() {
    rand.Seed(time.Now().UnixNano())

    logs.SetDefault()
    serialization.SetDefaultProtocol(serialization.Gob)

    dg, err := discordgo.New("Бот")
    if err != nil {
        panic(err)
    }

    vocab, reverseVocab := LoadVocab()
    trainData, labels := LoadData(vocab)
    model := NewModel(len(vocab), 300, 512)
    Train(model, trainData, labels, 10)

    dg.AddHandler(func(s *discordgo.Session, m *discordgo.MessageCreate) {
        if m.Author.ID == s.State.User.ID {
            return
        }

        input := []int{}
        for _, word := range strings.Split(strings.ToLower(m.Content), " ") {
            if val, ok := vocab[word]; ok {
                input = append(input, val)
            }
        }

        xs := mat32.NewMatrix(1, len(input), nil)
        copy(xs.RawMatrix().Data, input)

        hxs := mat32.NewMatrix(1, model.Encoder.Layers()[0].(*nn.LSTM).HiddenSize, nil)
        cxs := mat32.NewMatrix(1, model.Encoder.Layers()[0].(*nn.LSTM).HiddenSize, nil)

        sequence := model.Sample(xs, hxs, cxs, 20)
        output := []string{}
        for _, index := range sequence {
            if val, ok := reverseVocab[index]; ok {
                output = append(output, val)
            }
        }

        s.ChannelMessageSend(m.ChannelID, strings.Join(output, " "))
    })

    err = dg.Open()
    if err != nil {
        panic(err)
    }

    <-make(chan struct{})
}
// Если не работает то запустите от имени админа