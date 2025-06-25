// app.js
// Browser script: load model weights, capture canvas input, preprocess, predict and display

import Network from './network.js';
import Matrix from './matrix.js';

let net;

// Load weights and biases
async function loadModel() {
  const resp = await fetch('weights.json');
  const layersData = await resp.json();
  // network sizes from weights
  const sizes = [784, ...layersData.map(l => l.biases.length)];
  net = new Network(sizes);
  // overwrite random weights with trained values
  layersData.forEach((layerObj, i) => {
    const { weights, biases, rows, cols } = layerObj;
    net.layers[i].weights = new Matrix(rows, cols, new Float32Array(weights));
    net.layers[i].biases = new Matrix(rows, 1, new Float32Array(biases));
  });
}

// Preprocess canvas to 28x28 grayscale array
function getCanvasData() {
  const canvas = document.getElementById('canvas');
  const tmp = document.createElement('canvas');
  tmp.width = 28; tmp.height = 28;
  const tctx = tmp.getContext('2d');
  tctx.drawImage(canvas, 0, 0, 28, 28);
  const img = tctx.getImageData(0, 0, 28, 28).data;
  const input = [];
  for (let i = 0; i < img.length; i += 4) {
    const r = img[i], g = img[i+1], b = img[i+2], a = img[i+3];
    // if fully transparent, assume white (255)
    const avg = (a === 0) ? 255 : (r + g + b) / 3;
    // normalize & invert
    input.push((255 - avg) / 255);
  }
  return input;
}

// Display prediction
function showPrediction(arr) {
  const pred = arr.indexOf(Math.max(...arr));
  document.getElementById('prediction').innerText = `Prediction: ${pred}`;
}

// Setup canvas drawing
function setupCanvas() {
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  ctx.lineWidth = 15;
  ctx.lineCap = 'round';
  let drawing = false;
  canvas.addEventListener('mousedown', () => drawing = true);
  canvas.addEventListener('mouseup', () => drawing = false);
  canvas.addEventListener('mousemove', e => {
    if (!drawing) return;
    const rect = canvas.getBoundingClientRect();
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
  });
  // clear on double-click
  canvas.addEventListener('dblclick', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('prediction').innerText = '';
  });
}

// Hook predict button
function setupPredict() {
  document.getElementById('predict').addEventListener('click', () => {
    const input = getCanvasData();
    const output = net.predict(input);
    showPrediction(output);
  });
}

// Initialize
window.onload = async () => {
  await loadModel();
  setupCanvas();
  setupPredict();
};