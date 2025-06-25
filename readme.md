
# Neural Network in JavaScript — MNIST Digit Recognizer

This project demonstrates a fully functional neural network built from scratch using JavaScript to recognize handwritten digits from the MNIST dataset. It includes:

## ✨ Features

- 🧠 **Custom Neural Network**: Built without external ML libraries. Implements forward pass, backpropagation, and softmax cross-entropy loss.
- 🎨 **Canvas Drawing Interface**: Draw digits on a canvas, get live predictions.
- 📊 **Visualization**: Real-time 3D visualization of the network during training (using Three.js).
- 📦 **Training Script**: Offline training using preprocessed MNIST JSON data.
- 💾 **Weight Saving**: Saves trained weights to a JSON file (`weights.json`) for reuse.

## 📁 Project Structure

- `matrix.js` — Lightweight matrix math library.
- `activations.js` — Activation functions and their derivatives.
- `network.js` — Layer and Network classes implementing feedforward and backprop.
- `train.js` — Loads MNIST data, trains the model, and saves weights.
- `app.js` — Frontend logic: canvas input, prediction, and visualization.
- `index.html` — Main page with canvas and prediction UI.
- `style.css` — Basic styling.
- `weights.json` — Saved model weights (after training).

## 🚀 Getting Started

1. Clone the repo:
   ```
   git clone https://github.com/fxsipudx/NN-in-js.git
   cd NN-in-js
   ```

2. Install dependencies (if any).

3. Train the network:
   ```
   node train.js
   ```

4. Open `index.html` in your browser to test drawing and predictions.

> ⚠️ Note: MNIST JSON files (`mnist_handwritten_train.json`, `mnist_handwritten_test.json`) are not included due to size. You can generate or fetch these separately.

## 🧠 Tech Stack

- JavaScript (vanilla)
- HTML5 + Canvas
- Three.js (for visualization)
- Node.js (for training)

## 📌 Author

Shubham Jena 

---

Feel free to explore, extend, or visualize this network!