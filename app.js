// app.js – interactive MNIST demo with live-training visualisation
// -----------------------------------------------------------------------------
// 1. Draw on the canvas ➜ 28×28 grayscale vector
// 2. NN predicts / trains and we stream visual feedback:
//    • Left panel shows epoch / loss / accuracy + progress bar
//    • Right 3-D scene updates connection colours + opacity each epoch
// -----------------------------------------------------------------------------
/* global Matrix, NeuralNetwork, oneHot, THREE */

// ────────────────────────────────────────────────────────────
// Global state
// ────────────────────────────────────────────────────────────
let neuralNetwork;
const mnistData  = { train: [], test: [] };
let isTraining   = false;

// THREE.js bits
let scene, camera, renderer, networkGroup;
const neuronMeshes   = { input: [], hidden: [], output: [] };
const connectionLines = [];    // each → { line, fromLayer, toLayer, fromIndex, toIndex }
const flowParticles   = [];

// Canvas bits
let canvas, ctx;
let isDrawing = false;

// Consts
const HIDDEN_VIS   = 30;       // neurons drawn & used
const INPUT_VIS    = 20;       // only for visuals (real input=784)
const EPOCHS       = 10;       // quick demo training

// ────────────────────────────────────────────────────────────
// Boot
// ────────────────────────────────────────────────────────────
window.addEventListener('load', init);

async function init () {
  setupCanvas();
  await loadMNISTData();
  setupNeuralNetwork();
  setup3DVisualization();
  setupEventListeners();
  updateStatus('Ready – draw a digit!');
}

// ────────────────────────────────────────────────────────────
// Canvas helpers
// ────────────────────────────────────────────────────────────
function setupCanvas () {
  canvas = document.getElementById('drawingCanvas');
  ctx    = canvas.getContext('2d');
  ctx.lineCap   = 'round';
  ctx.lineJoin  = 'round';
  ctx.lineWidth = 12;
  clearCanvas();

  // mouse
  canvas.addEventListener('mousedown', startDrawing);
  canvas.addEventListener('mousemove', draw);
  canvas.addEventListener('mouseup',   stopDrawing);
  canvas.addEventListener('mouseout',  stopDrawing);
  // touch
  canvas.addEventListener('touchstart', handleTouch, { passive: false });
  canvas.addEventListener('touchmove',  handleTouch, { passive: false });
  canvas.addEventListener('touchend',   stopDrawing);
}

function startDrawing (e) {
  isDrawing = true;
  const { x, y } = canvasCoords(e);
  ctx.beginPath(); ctx.moveTo(x, y);
}
function draw (e) {
  if (!isDrawing) return;
  const { x, y } = canvasCoords(e);
  ctx.strokeStyle = 'black';
  ctx.lineTo(x, y); ctx.stroke();
}
function stopDrawing () { isDrawing = false; }
function handleTouch (e) {
  e.preventDefault();
  if (!e.touches.length) return;
  const t = e.touches[0];
  const type = e.type === 'touchstart' ? 'mousedown' : (e.type === 'touchmove' ? 'mousemove' : 'mouseup');
  canvas.dispatchEvent(new MouseEvent(type, { clientX: t.clientX, clientY: t.clientY }));
}
function canvasCoords (evt) {
  const r = canvas.getBoundingClientRect();
  return { x: evt.clientX - r.left, y: evt.clientY - r.top };
}
function clearCanvas () {
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  hidePrediction();
}

// ────────────────────────────────────────────────────────────
// Data loading
// ────────────────────────────────────────────────────────────
async function loadMNISTData () {
  updateStatus('Loading training data…');
  try {
    const trainRaw = await (await fetch('data/mnist_handwritten_train.json')).json();
    updateStatus('Loading test data…');
    const testRaw  = await (await fetch('data/mnist_handwritten_test.json')).json();

    // tiny subset for fast demos
    mnistData.train = trainRaw.slice(0, 5000).map(d => ({ input: d.image.map(p => p / 255), target: oneHot(d.label, 10) }));
    mnistData.test  =  testRaw.slice(0, 1000).map(d => ({ input: d.image.map(p => p / 255), target: oneHot(d.label, 10) }));

    updateStatus(`Data ready – ${mnistData.train.length} train / ${mnistData.test.length} test`);
    document.getElementById('trainBtn').disabled = false;
  } catch (err) {
    updateStatus('Error loading data: ' + err.message);
    console.error(err);
  }
}

// ────────────────────────────────────────────────────────────
// NN init
// ────────────────────────────────────────────────────────────
function setupNeuralNetwork () {
  // 784 ➜ 30 ➜ 10
  neuralNetwork = new NeuralNetwork(784, HIDDEN_VIS, 10, 0.15);
  neuralNetwork.setActivation('relu', 'sigmoid');
  updateStatus('Neural network initialised');
}

// ────────────────────────────────────────────────────────────
// THREE.js visual
// ────────────────────────────────────────────────────────────
function setup3DVisualization () {
  const container = document.getElementById('networkViz');
  container.innerHTML = '';

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);
  camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
  camera.position.set(0, 0, 15);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  createMatrixRain();
  createNetworkVisualization();
  scene.add(new THREE.AmbientLight(0x003300, 0.6));

  window.addEventListener('resize', onWindowResize);
  animate();
}

function createMatrixRain () {
  const count = 800;
  const pos   = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    pos[3 * i]     = (Math.random() - 0.5) * 20;
    pos[3 * i + 1] =  Math.random() * 20;
    pos[3 * i + 2] = (Math.random() - 0.5) * 20;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const mat = new THREE.PointsMaterial({ color: 0x00ff00, size: 0.07, transparent: true, opacity: 0.3 });
  const rain = new THREE.Points(geo, mat);
  rain.name = 'matrixRain'; scene.add(rain);
}

function createNetworkVisualization () {
  networkGroup = new THREE.Group(); scene.add(networkGroup);

  const spacingX = 8;
  const layers = {
    input:  { x: -spacingX, count: INPUT_VIS },
    hidden: { x:  0,        count: HIDDEN_VIS },
    output: { x:  spacingX, count: 10 }
  };

  for (const layer in layers) {
    const { x, count } = layers[layer];
    const gap = 0.8; const y0 = -((count - 1) * gap) / 2;
    neuronMeshes[layer] = [];
    for (let i = 0; i < count; i++) {
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(0.15, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0x00ff00, transparent: true, opacity: 0.25 })
      );
      sphere.position.set(x, y0 + i * gap, 0);
      networkGroup.add(sphere);
      neuronMeshes[layer].push(sphere);
    }
  }
  createConnections('input', 'hidden');
  createConnections('hidden', 'output');
}

function createConnections (from, to) {
  neuronMeshes[from].forEach((a, i) => {
    neuronMeshes[to].forEach((b, j) => {
      const geom = new THREE.BufferGeometry().setFromPoints([a.position, b.position]);
      const mat  = new THREE.LineBasicMaterial({ color: 0x666666, transparent: true, opacity: 0.05 });
      const line = new THREE.Line(geom, mat);
      networkGroup.add(line);
      connectionLines.push({ line, fromLayer: from, toLayer: to, fromIndex: i, toIndex: j });
    });
  });
}

function animate () {
  requestAnimationFrame(animate);
  if (networkGroup) networkGroup.rotation.y += 0.005;
  animateMatrixRain();
  animateFlowParticles();
  renderer.render(scene, camera);
}

function animateMatrixRain () {
  const rain = scene.getObjectByName('matrixRain'); if (!rain) return;
  const pos = rain.geometry.attributes.position.array;
  for (let i = 1; i < pos.length; i += 3) {
    pos[i] -= 0.15; if (pos[i] < -10) pos[i] = 20;
  }
  rain.geometry.attributes.position.needsUpdate = true;
}

function animateFlowParticles () {
  for (let i = flowParticles.length - 1; i >= 0; i--) {
    const p = flowParticles[i]; p.age++;
    p.mesh.position.lerp(p.target, 0.05);
    if (p.age > 60) { scene.remove(p.mesh); flowParticles.splice(i, 1); }
  }
}

function onWindowResize () {
  const c = document.getElementById('networkViz');
  camera.aspect = c.clientWidth / c.clientHeight; camera.updateProjectionMatrix();
  renderer.setSize(c.clientWidth, c.clientHeight);
}

// ────────────────────────────────────────────────────────────
// UI event wiring
// ────────────────────────────────────────────────────────────
function setupEventListeners () {
  document.getElementById('clearBtn') .addEventListener('click', clearCanvas);
  document.getElementById('predictBtn').addEventListener('click', predictDigit);
  document.getElementById('trainBtn') .addEventListener('click', trainNetwork);
}

// ────────────────────────────────────────────────────────────
// Prediction
// ────────────────────────────────────────────────────────────
function predictDigit () {
  if (!neuralNetwork || isTraining) return;
  updateStatus('Predicting…');
  const input = canvasToInputVector();
  const out   = neuralNetwork.predict(input);
  const idx   = out.indexOf(Math.max(...out));
  showPrediction(idx, out);

  // fire some visual particles along random paths to chosen output
  for (let k = 0; k < 12; k++) {
    const inI = Math.floor(Math.random() * neuronMeshes.input.length);
    const hidI= Math.floor(Math.random() * neuronMeshes.hidden.length);
    emitFlow(buildFlowPath(inI, hidI, idx));
  }
  updateStatus('Prediction complete');
}

function canvasToInputVector () {
  const tmp = document.createElement('canvas'); tmp.width = tmp.height = 28;
  const tctx = tmp.getContext('2d');
  tctx.drawImage(canvas, 0, 0, 28, 28);
  const { data } = tctx.getImageData(0, 0, 28, 28);
  const vec = new Array(784);
  for (let i = 0; i < 784; i++) {
    const v = data[i * 4] / 255; vec[i] = 1 - v; // invert: black stroke ➜ 1.0
  }
  return vec;
}

// ────────────────────────────────────────────────────────────
// Training with live stats + visual feedback
// ────────────────────────────────────────────────────────────
async function trainNetwork () {
  if (isTraining) return;
  isTraining = true; updateStatus('Training started…');
  document.getElementById('trainBtn').disabled = true;

  const progBar = document.getElementById('progressBar');
  const progFill= document.getElementById('progressFill');
  progBar.style.display = 'block';
  document.getElementById('trainingStats').style.display = 'block';

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    updateStatus(`Training epoch ${epoch + 1} / ${EPOCHS}`);
    neuralNetwork.trainBatch(mnistData.train, 1);

    // metrics
    const loss = neuralNetwork.calculateLoss(mnistData.train);
    const acc  = neuralNetwork.test(mnistData.test);

    // UI update
    document.getElementById('currentEpoch').textContent    = epoch + 1;
    document.getElementById('currentLoss').textContent     = loss.toFixed(4);
    document.getElementById('currentAccuracy').textContent = (acc * 100).toFixed(1) + '%';
    progFill.style.width = `${((epoch + 1) / EPOCHS) * 100}%`;

    // 3-D weight colouring
    updateNetworkVisual();

    await new Promise(r => setTimeout(r, 30)); // let UI breathe
  }

  progBar.style.display = 'none';
  updateStatus('Training finished – try predicting again!');
  document.getElementById('trainBtn').disabled = false;
  isTraining = false;
}

// update colour & opacity of hidden→output lines based on weights
function updateNetworkVisual () {
  connectionLines.forEach(conn => {
    if (conn.fromLayer === 'hidden' && conn.toLayer === 'output') {
      const w = neuralNetwork.weights_ho.data[conn.toIndex][conn.fromIndex];
      const absW = Math.abs(w);
      conn.line.material.opacity = THREE.MathUtils.clamp(absW * 2, 0.05, 1);
      conn.line.material.color.set(w >= 0 ? 0x00ff00 : 0xff0000);
    }
  });
}

// ────────────────────────────────────────────────────────────
// Visual signal flow helpers
// ────────────────────────────────────────────────────────────
function buildFlowPath (inputIdx, hiddenIdx, outputIdx) {
  return [
    { from: neuronMeshes.input[inputIdx],  to: neuronMeshes.hidden[hiddenIdx] },
    { from: neuronMeshes.hidden[hiddenIdx], to: neuronMeshes.output[outputIdx] }
  ];
}
function emitFlow (path) {
  path.forEach(({ from, to }, i) => {
    setTimeout(() => {
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.06, 8, 8), new THREE.MeshBasicMaterial({ color: 0x00ff00 }));
      mesh.position.copy(from.position); scene.add(mesh);
      flowParticles.push({ mesh, target: to.position.clone(), age: 0 });
    }, i * 100);
  });
}

// ────────────────────────────────────────────────────────────
// Status & prediction UI
// ────────────────────────────────────────────────────────────
function updateStatus (txt) {
  document.getElementById('statusText').textContent = txt;
}
function showPrediction (digit, conf) {
  const area = document.getElementById('predictionArea'); area.style.display = 'block';
  document.getElementById('predictedDigit').textContent = digit;
  updateConfidenceBars(conf);
}
function hidePrediction () {
  document.getElementById('predictionArea').style.display = 'none';
}
function updateConfidenceBars (conf) {
  const cont = document.getElementById('confidenceBars');
  if (!cont.children.length) {
    for (let i = 0; i < 10; i++) {
      const bar = document.createElement('div'); bar.className = 'confidence-bar';
      const fill= document.createElement('div'); fill.className = 'confidence-fill'; bar.appendChild(fill);
      const lbl = document.createElement('div'); lbl.className = 'confidence-label'; lbl.textContent = i; bar.appendChild(lbl);
      cont.appendChild(bar);
    }
  }
  Array.from(cont.children).forEach((bar, i) => {
    bar.querySelector('.confidence-fill').style.height = `${Math.max(5, conf[i] * 60)}px`;
  });
}
