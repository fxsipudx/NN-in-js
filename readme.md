# Neural Networks & Transformers in JavaScript

This project demonstrates machine learning models built from scratch using JavaScript, featuring both traditional neural networks and modern transformer architectures. It includes MNIST digit recognition and a complete transformer implementation.

## ✨ Features

### 🧠 **Neural Network (MNIST)**
- **Custom Neural Network**: Built without external ML libraries. Implements forward pass, backpropagation, and softmax cross-entropy loss.
- **Canvas Drawing Interface**: Draw digits on a canvas, get live predictions.
- **📊 Visualization**: Real-time 3D visualization of the network during training (using Three.js).
- **📦 Training Script**: Offline training using preprocessed MNIST JSON data.
- **💾 Weight Saving**: Saves trained weights to a JSON file (`weights.json`) for reuse.

### 🤖 **Transformer Library**
- **Complete Transformer Implementation**: Multi-head attention, positional encoding, layer normalization, and feed-forward networks.
- **Universal Compatibility**: Works in both Node.js and browser environments.
- **Modular Design**: Each component can be used independently.
- **Text Generation**: Includes greedy decoding for autoregressive tasks.
- **Configurable Architecture**: Customizable model dimensions, heads, and layers.

## 📁 Project Structure

### Core Neural Network
- `matrix.js` — Lightweight matrix math library.
- `activations.js` — Activation functions and their derivatives.
- `network.js` — Layer and Network classes implementing feedforward and backprop.
- `train.js` — Loads MNIST data, trains the model, and saves weights.
- `app.js` — Frontend logic: canvas input, prediction, and visualization.
- `index.html` — Main page with canvas and prediction UI.
- `style.css` — Basic styling.
- `weights.json` — Saved model weights (after training).

### Transformer Library
- `transformer.js` — Complete transformer implementation with all components:
  - Multi-head attention mechanisms
  - Positional encoding
  - Layer normalization
  - Feed-forward networks
  - Transformer encoder blocks
  - Text generation utilities

## 🚀 Getting Started

### Neural Network (MNIST)
1. Clone the repo:
   ```bash
   git clone https://github.com/fxsipudx/NN-in-js.git
   cd NN-in-js
   ```

2. Train the network:
   ```bash
   node train.js
   ```

3. Open `index.html` in your browser to test drawing and predictions.

### Transformer Library
The transformer library can be used in multiple ways:

#### Node.js
```javascript
const { Transformer, createCausalMask } = require('./transformer.js');

// Create a model
const model = new Transformer({
  dModel: 512,
  numHeads: 8,
  numLayers: 6,
  vocabSize: 10000
});

// Forward pass
const input = Matrix.from2DArray([[1, 2, 3, 4, 5]]); // token IDs
const mask = createCausalMask(5);
const output = model.forward(input, mask);
```

#### Browser
```html
<script src="transformer.js"></script>
<script>
  const { Transformer, Matrix } = window.Transformer;
  
  const model = new Transformer({
    dModel: 256,
    numHeads: 4,
    numLayers: 3
  });
  
  // Use the model...
</script>
```

## 🧠 Tech Stack

- **JavaScript (vanilla)** - Core implementation
- **HTML5 + Canvas** - Interactive drawing interface
- **Three.js** - 3D visualization
- **Node.js** - Training and utilities
- **Float32Array** - Efficient matrix operations

## 📚 What You Can Build

### With the Neural Network
- Handwritten digit recognition
- Custom image classification
- Pattern recognition systems

### With the Transformer Library
- **Language Models**: GPT-style text generation
- **Text Classification**: Sentiment analysis, topic classification
- **Sequence-to-Sequence**: Translation, summarization
- **Embeddings**: Text representation learning
- **Custom Architectures**: BERT-like encoders, custom attention mechanisms

## 🎯 Use Cases

- **Education**: Learn how neural networks and transformers work internally
- **Prototyping**: Quick experimentation with transformer architectures
- **Research**: Custom attention mechanisms and model variants
- **Web Applications**: Client-side ML without external dependencies

## 📌 Notes

> ⚠️ **MNIST Data**: JSON files (`mnist_handwritten_train.json`, `mnist_handwritten_test.json`) are not included due to size. You can generate or fetch these separately.

> 💡 **Performance**: This is an educational implementation. For production use, consider optimized libraries like TensorFlow.js or PyTorch.

## 🔧 Extending the Project

The modular design makes it easy to:
- Add new activation functions
- Implement different attention mechanisms
- Create custom transformer variants
- Add training loops for the transformer
- Integrate with real datasets

## 📌 Author

Shubham Jena

---

**Explore the fundamentals of modern AI!** This project bridges the gap between traditional neural networks and cutting-edge transformer architectures, all implemented in pure JavaScript for maximum understanding and flexibility.