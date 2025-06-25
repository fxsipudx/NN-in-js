// activations.js
// Activation functions and their derivatives for neural network layers.

import Matrix from './matrix.js';

// Sigmoid activation: 1 / (1 + e^-x)
export function sigmoid(mat) {
  return mat.map(x => 1 / (1 + Math.exp(-x)));
}

// Derivative of sigmoid when input is already sigmoid output: s * (1 - s)
export function dsigmoid(mat) {
  return mat.map(s => s * (1 - s));
}

// ReLU activation: max(0, x)
export function relu(mat) {
  return mat.map(x => (x > 0 ? x : 0));
}

// Derivative of ReLU: 1 if x > 0, else 0
export function drelu(mat) {
  return mat.map(x => (x > 0 ? 1 : 0));
}

// Softmax activation for a column vector
export function softmax(mat) {
  const values = mat.toArray();
  const maxVal = Math.max(...values);
  const exps = values.map(v => Math.exp(v - maxVal));
  const sum = exps.reduce((acc, v) => acc + v, 0);
  const result = exps.map(v => v / sum);
  return Matrix.fromArray(result);
}

// Note: For training with cross-entropy loss, combine softmax and loss derivative directly
export function dsoftmax_softmaxCrossEntropy(output, target) {
  // output: softmax probabilities (Matrix), target: one-hot Matrix
  // derivative of loss L = -sum(t * log(o)) w.r.t. inputs: o - t
  return output.subtract(target);
}
