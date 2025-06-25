import NetworkMatrix from './matrix.js';
import { sigmoid, dsigmoid, relu, drelu, softmax, dsoftmax_softmaxCrossEntropy } from './activations.js';

export class Layer {
  constructor(inSize, outSize, activation = 'sigmoid') {
    this.weights = NetworkMatrix.random(outSize, inSize);
    this.biases = NetworkMatrix.random(outSize, 1);
    this.activation = activation;
  }

  activate(z) {
    if (this.activation === 'sigmoid') return sigmoid(z);
    if (this.activation === 'relu') return relu(z);
    if (this.activation === 'softmax') return softmax(z);
    throw new Error('Unknown activation');
  }

  activateDerivative(a) {
    if (this.activation === 'sigmoid') return dsigmoid(a);
    if (this.activation === 'relu') return drelu(a);
    throw new Error('No derivative');
  }
}

export class Network {
  constructor(sizes) {
    this.layers = [];
    for (let i = 1; i < sizes.length; i++) {
      const act = i === sizes.length - 1 ? 'softmax' : 'sigmoid';
      this.layers.push(new Layer(sizes[i - 1], sizes[i], act));
    }
  }

  feedforward(inputArr) {
    let activation = NetworkMatrix.fromArray(inputArr);
    const activations = [activation];
    const zs = [];
    for (const layer of this.layers) {
      const z = NetworkMatrix.dot(layer.weights, activation).add(layer.biases);
      zs.push(z);
      activation = layer.activate(z);
      activations.push(activation);
    }
    return { activations, zs };
  }

  backprop(x, y) {
    const nablaW = this.layers.map(l => NetworkMatrix.zeros(l.weights.rows, l.weights.cols));
    const nablaB = this.layers.map(l => NetworkMatrix.zeros(l.biases.rows, 1));
    const { activations, zs } = this.feedforward(x);

    // output error
    let delta = dsoftmax_softmaxCrossEntropy(
      activations[activations.length - 1],
      NetworkMatrix.fromArray(y)
    );
    nablaB[nablaB.length - 1] = delta;
    nablaW[nablaW.length - 1] = NetworkMatrix.dot(
      delta,
      NetworkMatrix.transpose(activations[activations.length - 2])
    );

    // backprop hidden layers
    for (let l = 2; l <= this.layers.length; l++) {
      const layerIdx = this.layers.length - l;
      const sp = this.layers[layerIdx].activateDerivative(activations[activations.length - l]);
      delta = NetworkMatrix.dot(
        NetworkMatrix.transpose(this.layers[layerIdx + 1].weights),
        delta
      ).multiply(sp);
      nablaB[layerIdx] = delta;
      nablaW[layerIdx] = NetworkMatrix.dot(
        delta,
        NetworkMatrix.transpose(activations[activations.length - l - 1])
      );
    }

    return { nablaW, nablaB };
  }

  updateMiniBatch(miniBatch, eta) {
    const nablaW = this.layers.map(l => NetworkMatrix.zeros(l.weights.rows, l.weights.cols));
    const nablaB = this.layers.map(l => NetworkMatrix.zeros(l.biases.rows, 1));

    for (const [x, y] of miniBatch) {
      const { nablaW: dW, nablaB: dB } = this.backprop(x, y);
      for (let i = 0; i < this.layers.length; i++) {
        nablaW[i] = nablaW[i].add(dW[i]);
        nablaB[i] = nablaB[i].add(dB[i]);
      }
    }

    const m = miniBatch.length;
    for (let i = 0; i < this.layers.length; i++) {
      this.layers[i].weights = this.layers[i].weights.subtract(nablaW[i].multiply(eta / m));
      this.layers[i].biases = this.layers[i].biases.subtract(nablaB[i].multiply(eta / m));
    }
  }

  SGD(trainData, epochs, batchSize, eta, testData = null) {
    for (let e = 0; e < epochs; e++) {
      trainData.sort(() => Math.random() - 0.5);
      const batches = [];
      for (let k = 0; k < trainData.length; k += batchSize) {
        batches.push(trainData.slice(k, k + batchSize));
      }
      for (const batch of batches) this.updateMiniBatch(batch, eta);
      if (testData) {
        console.log(
          `Epoch ${e}: ${this.evaluate(testData)} / ${testData.length}`
        );
      } else console.log(`Epoch ${e} complete`);
    }
  }

  evaluate(testData) {
    let correct = 0;
    for (const [x, y] of testData) {
      const { activations } = this.feedforward(x);
      const out = activations[activations.length - 1].toArray();
      if (out.indexOf(Math.max(...out)) === y.indexOf(Math.max(...y))) correct++;
    }
    return correct;
  }

  predict(inputArr) {
    const { activations } = this.feedforward(inputArr);
    return activations[activations.length - 1].toArray();
  }
}

export default Network;
