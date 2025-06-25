// Activation functions and their derivatives
const Activation = {
  // Sigmoid: smooth S-curve, outputs 0-1
  sigmoid: {
    func: (x) => 1 / (1 + Math.exp(-x)),
    derivative: (x) => x * (1 - x) // derivative of sigmoid(x) where x is already sigmoid output
  },
  
  // ReLU: simple and effective, outputs 0 or x
  relu: {
    func: (x) => Math.max(0, x),
    derivative: (x) => x > 0 ? 1 : 0
  },
  
  // Tanh: similar to sigmoid but -1 to 1
  tanh: {
    func: (x) => Math.tanh(x),
    derivative: (x) => 1 - (x * x) // derivative where x is already tanh output
  }
};

// Neural Network class
class NeuralNetwork {
  constructor(inputNodes, hiddenNodes, outputNodes, learningRate = 0.1) {
    this.inputNodes = inputNodes;
    this.hiddenNodes = hiddenNodes;
    this.outputNodes = outputNodes;
    this.learningRate = learningRate;
    
    // Initialize weights with random values
    // weights_ih: input to hidden layer weights
    this.weights_ih = Matrix.random(this.hiddenNodes, this.inputNodes);
    // weights_ho: hidden to output layer weights
    this.weights_ho = Matrix.random(this.outputNodes, this.hiddenNodes);
    
    // Initialize biases
    this.bias_h = Matrix.random(this.hiddenNodes, 1);
    this.bias_o = Matrix.random(this.outputNodes, 1);
    
    // Default activation functions
    this.hiddenActivation = Activation.sigmoid;
    this.outputActivation = Activation.sigmoid;
  }
  
  // Set activation functions for different layers
  setActivation(hidden = 'sigmoid', output = 'sigmoid') {
    this.hiddenActivation = Activation[hidden];
    this.outputActivation = Activation[output];
  }
  
  // Forward pass: predict output for given input
  predict(inputArray) {
    // Convert input array to matrix
    let inputs = Matrix.fromArray(inputArray);
    
    // Calculate hidden layer
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden = hidden.add(this.bias_h);
    // Apply activation function to hidden layer
    hidden = hidden.map(this.hiddenActivation.func);
    
    // Calculate output layer
    let output = Matrix.multiply(this.weights_ho, hidden);
    output = output.add(this.bias_o);
    // Apply activation function to output layer
    output = output.map(this.outputActivation.func);
    
    // Convert back to array for easy use
    return output.toArray();
  }
  
  // Training function using backpropagation
  train(inputArray, targetArray) {
    // === Forward Pass ===
    let inputs = Matrix.fromArray(inputArray);
    
    // Calculate hidden layer
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden = hidden.add(this.bias_h);
    hidden = hidden.map(this.hiddenActivation.func);
    
    // Calculate outputs
    let outputs = Matrix.multiply(this.weights_ho, hidden);
    outputs = outputs.add(this.bias_o);
    outputs = outputs.map(this.outputActivation.func);
    
    // === Backward Pass (Backpropagation) ===
    // Convert targets to matrix
    let targets = Matrix.fromArray(targetArray);
    
    // Calculate output errors
    let outputErrors = targets.subtract(outputs);
    
    // Calculate gradients for output layer
    let gradients = outputs.map(this.outputActivation.derivative);
    gradients = gradients.multiply(outputErrors);
    gradients = gradients.multiply(this.learningRate);
    
    // Calculate deltas for weights (how much to change weights)
    let hiddenT = hidden.transpose();
    let weight_ho_deltas = Matrix.multiply(gradients, hiddenT);
    
    // Adjust weights and biases for output layer
    this.weights_ho = this.weights_ho.add(weight_ho_deltas);
    this.bias_o = this.bias_o.add(gradients);
    
    // Calculate hidden layer errors
    let who_t = this.weights_ho.transpose();
    let hiddenErrors = Matrix.multiply(who_t, outputErrors);
    
    // Calculate gradients for hidden layer
    let hiddenGradient = hidden.map(this.hiddenActivation.derivative);
    hiddenGradient = hiddenGradient.multiply(hiddenErrors);
    hiddenGradient = hiddenGradient.multiply(this.learningRate);
    
    // Calculate deltas for input->hidden weights
    let inputs_T = inputs.transpose();
    let weight_ih_deltas = Matrix.multiply(hiddenGradient, inputs_T);
    
    // Adjust weights and biases for hidden layer
    this.weights_ih = this.weights_ih.add(weight_ih_deltas);
    this.bias_h = this.bias_h.add(hiddenGradient);
  }
  
  // Calculate loss for a batch of data
  calculateLoss(data) {
    let totalLoss = 0;
    for (let i = 0; i < data.length; i++) {
      let prediction = this.predict(data[i].input);
      let target = data[i].target;
      
      // Mean squared error
      for (let j = 0; j < prediction.length; j++) {
        let error = target[j] - prediction[j];
        totalLoss += error * error;
      }
    }
    return totalLoss / data.length;
  }
  
  // Train on a batch of data
  trainBatch(data, epochs = 1) {
    const losses = [];
    
    for (let epoch = 0; epoch < epochs; epoch++) {
      // Shuffle data for each epoch
      const shuffled = [...data].sort(() => Math.random() - 0.5);
      
      // Train on each example
      for (let i = 0; i < shuffled.length; i++) {
        this.train(shuffled[i].input, shuffled[i].target);
      }
      
      // Calculate and store loss for this epoch
      const loss = this.calculateLoss(data);
      losses.push(loss);
      
      // Log progress every 100 epochs
      if (epoch % 100 === 0) {
        console.log(`Epoch ${epoch}, Loss: ${loss.toFixed(6)}`);
      }
    }
    
    return losses;
  }
  
  // Test accuracy on dataset
  test(testData) {
    let correct = 0;
    for (let i = 0; i < testData.length; i++) {
      let prediction = this.predict(testData[i].input);
      let target = testData[i].target;
      
      // Find the index of highest prediction and target
      let predictedClass = prediction.indexOf(Math.max(...prediction));
      let actualClass = target.indexOf(Math.max(...target));
      
      if (predictedClass === actualClass) {
        correct++;
      }
    }
    return correct / testData.length;
  }
  
  // Save model weights (returns object with weights and biases)
  save() {
    return {
      weights_ih: this.weights_ih.data,
      weights_ho: this.weights_ho.data,
      bias_h: this.bias_h.data,
      bias_o: this.bias_o.data,
      inputNodes: this.inputNodes,
      hiddenNodes: this.hiddenNodes,
      outputNodes: this.outputNodes,
      learningRate: this.learningRate
    };
  }
  
  // Load model weights
  load(modelData) {
    this.weights_ih = new Matrix(modelData.hiddenNodes, modelData.inputNodes, modelData.weights_ih);
    this.weights_ho = new Matrix(modelData.outputNodes, modelData.hiddenNodes, modelData.weights_ho);
    this.bias_h = new Matrix(modelData.hiddenNodes, 1, modelData.bias_h);
    this.bias_o = new Matrix(modelData.outputNodes, 1, modelData.bias_o);
    this.inputNodes = modelData.inputNodes;
    this.hiddenNodes = modelData.hiddenNodes;
    this.outputNodes = modelData.outputNodes;
    this.learningRate = modelData.learningRate;
  }
}

// Helper function to create one-hot encoded arrays
function oneHot(index, size) {
  const arr = new Array(size).fill(0);
  arr[index] = 1;
  return arr;
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { NeuralNetwork, Activation, oneHot };
}