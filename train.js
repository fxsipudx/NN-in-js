// train.js
// Node script to train the neural network on MNIST JSON data and export weights

import fs from 'fs/promises';
import Network from './network.js';

async function loadData(path) {
  const raw = await fs.readFile(path, 'utf8');
  const items = JSON.parse(raw);
  return items.map(obj => {
    // normalize to [0,1]
    const xArr = obj.image.map(v => v / 255);
    if (xArr.length !== 784) {
      throw new Error(`Unexpected input length: ${xArr.length}`);
    }
    // one-hot encode
    const yArr = Array(10).fill(0);
    yArr[obj.label] = 1;
    return [xArr, yArr];
  });
}

async function main() {
  console.log('Loading data...');
  const trainData = await loadData('./data/mnist_handwritten_train.json');
  const testData = await loadData('./data/mnist_handwritten_test.json');

  // Format for SGD: pairs of [inputArr, targetArr]
  const net = new Network([784, 128, 64, 10]);
  const epochs = 20;
  const batchSize = 16;
  const learningRate = 0.05;

  console.log('Training...');
  net.SGD(trainData, epochs, batchSize, learningRate, testData);

  console.log('Exporting weights...');
  const exportObj = net.layers.map(layer => ({
    weights: layer.weights.toArray(),
    biases: layer.biases.toArray(),
    rows: layer.weights.rows,
    cols: layer.weights.cols,
  }));

  await fs.writeFile('weights.json', JSON.stringify(exportObj));
  console.log('Done. Weights saved to weights.json');
}

main().catch(err => console.error(err));