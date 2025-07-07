// MATRIX UTILITIES
class Matrix {
  constructor(rows, cols, data = null) {
    this.rows = rows;
    this.cols = cols;
    this.data = data instanceof Float32Array ? data : new Float32Array(rows * cols);
  }

  static zeros(rows, cols) {
    return new Matrix(rows, cols);
  }

  static ones(rows, cols) {
    const m = new Matrix(rows, cols);
    m.data.fill(1);
    return m;
  }

  static random(rows, cols, scale = 1) {
    const m = new Matrix(rows, cols);
    for (let i = 0; i < m.data.length; i++) {
      m.data[i] = (Math.random() * 2 - 1) * scale;
    }
    return m;
  }

  static fromArray(arr) {
    const m = new Matrix(arr.length, 1);
    for (let i = 0; i < arr.length; i++) {
      m.data[i] = arr[i];
    }
    return m;
  }

  static from2DArray(arr) {
    const rows = arr.length;
    const cols = arr[0].length;
    const m = new Matrix(rows, cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        m.set(i, j, arr[i][j]);
      }
    }
    return m;
  }

  toArray() {
    return Array.from(this.data);
  }

  to2DArray() {
    const result = [];
    for (let i = 0; i < this.rows; i++) {
      const row = [];
      for (let j = 0; j < this.cols; j++) {
        row.push(this.get(i, j));
      }
      result.push(row);
    }
    return result;
  }

  _idx(i, j) {
    return i * this.cols + j;
  }

  get(i, j) {
    return this.data[this._idx(i, j)];
  }

  set(i, j, v) {
    this.data[this._idx(i, j)] = v;
  }

  map(fn) {
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        const v = this.get(i, j);
        result.set(i, j, fn(v, i, j));
      }
    }
    return result;
  }

  add(n) {
    if (n instanceof Matrix) {
      if (n.rows !== this.rows || n.cols !== this.cols) {
        throw new Error('Dimension mismatch');
      }
      return this.map((v, i, j) => v + n.get(i, j));
    }
    return this.map(v => v + n);
  }

  subtract(n) {
    if (n instanceof Matrix) {
      if (n.rows !== this.rows || n.cols !== this.cols) {
        throw new Error('Dimension mismatch');
      }
      return this.map((v, i, j) => v - n.get(i, j));
    }
    return this.map(v => v - n);
  }

  multiply(n) {
    if (n instanceof Matrix) {
      if (n.rows !== this.rows || n.cols !== this.cols) {
        throw new Error('Dimension mismatch');
      }
      return this.map((v, i, j) => v * n.get(i, j));
    }
    return this.map(v => v * n);
  }

  static transpose(mat) {
    const result = new Matrix(mat.cols, mat.rows);
    for (let i = 0; i < mat.rows; i++) {
      for (let j = 0; j < mat.cols; j++) {
        result.set(j, i, mat.get(i, j));
      }
    }
    return result;
  }

  static dot(a, b) {
    if (a.cols !== b.rows) {
      throw new Error(`Cannot multiply ${a.rows}x${a.cols} with ${b.rows}x${b.cols}`);
    }
    const result = new Matrix(a.rows, b.cols);
    for (let i = 0; i < a.rows; i++) {
      for (let j = 0; j < b.cols; j++) {
        let sum = 0;
        for (let k = 0; k < a.cols; k++) {
          sum += a.get(i, k) * b.get(k, j);
        }
        result.set(i, j, sum);
      }
    }
    return result;
  }

  clone() {
    return new Matrix(this.rows, this.cols, new Float32Array(this.data));
  }
}

// ACTIVATION FUNCTIONS
function softmax(mat) {
  const result = new Matrix(mat.rows, mat.cols);
  for (let i = 0; i < mat.rows; i++) {
    let maxVal = -Infinity;
    for (let j = 0; j < mat.cols; j++) {
      maxVal = Math.max(maxVal, mat.get(i, j));
    }
    let sum = 0;
    for (let j = 0; j < mat.cols; j++) {
      const exp = Math.exp(mat.get(i, j) - maxVal);
      result.set(i, j, exp);
      sum += exp;
    }
    for (let j = 0; j < mat.cols; j++) {
      result.set(i, j, result.get(i, j) / sum);
    }
  }
  return result;
}

function relu(mat) {
  return mat.map(x => Math.max(0, x));
}

function gelu(mat) {
  return mat.map(x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3)))));
}

// LAYER NORMALIZATION
class LayerNorm {
  constructor(size, eps = 1e-6) {
    this.size = size;
    this.eps = eps;
    this.gamma = Matrix.ones(1, size);
    this.beta = Matrix.zeros(1, size);
  }

  forward(x) {
    const batchSize = x.rows;
    const result = new Matrix(batchSize, this.size);
    
    for (let i = 0; i < batchSize; i++) {
      // Calculate mean
      let sum = 0;
      for (let j = 0; j < this.size; j++) {
        sum += x.get(i, j);
      }
      const mean = sum / this.size;
      
      // Calculate variance
      let varSum = 0;
      for (let j = 0; j < this.size; j++) {
        const diff = x.get(i, j) - mean;
        varSum += diff * diff;
      }
      const variance = varSum / this.size;
      const std = Math.sqrt(variance + this.eps);
      
      // Normalize and scale
      for (let j = 0; j < this.size; j++) {
        const normalized = (x.get(i, j) - mean) / std;
        const scaled = normalized * this.gamma.get(0, j) + this.beta.get(0, j);
        result.set(i, j, scaled);
      }
    }
    
    return result;
  }
}

// POSITIONAL ENCODING 
class PositionalEncoding {
  constructor(dModel, maxLen = 5000) {
    this.dModel = dModel;
    this.maxLen = maxLen;
    this.encodings = this._generateEncodings();
  }

  _generateEncodings() {
    const pe = new Matrix(this.maxLen, this.dModel);
    
    for (let pos = 0; pos < this.maxLen; pos++) {
      for (let i = 0; i < this.dModel; i++) {
        const angle = pos / Math.pow(10000, 2 * Math.floor(i / 2) / this.dModel);
        if (i % 2 === 0) {
          pe.set(pos, i, Math.sin(angle));
        } else {
          pe.set(pos, i, Math.cos(angle));
        }
      }
    }
    
    return pe;
  }

  forward(x) {
    const seqLen = x.rows;
    const result = new Matrix(seqLen, this.dModel);
    
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < this.dModel; j++) {
        result.set(i, j, x.get(i, j) + this.encodings.get(i, j));
      }
    }
    
    return result;
  }
}

// MULTI-HEAD ATTENTION 
class MultiHeadAttention {
  constructor(dModel, numHeads) {
    this.dModel = dModel;
    this.numHeads = numHeads;
    this.dK = Math.floor(dModel / numHeads);
    
    if (dModel % numHeads !== 0) {
      throw new Error('dModel must be divisible by numHeads');
    }
    
    // Initialize weight matrices
    const scale = Math.sqrt(1.0 / dModel);
    this.wQ = Matrix.random(dModel, dModel, scale);
    this.wK = Matrix.random(dModel, dModel, scale);
    this.wV = Matrix.random(dModel, dModel, scale);
    this.wO = Matrix.random(dModel, dModel, scale);
  }

  forward(query, key, value, mask = null) {
    const batchSize = query.rows;
    const seqLen = query.rows;
    
    // Linear transformations
    const Q = Matrix.dot(query, this.wQ);
    const K = Matrix.dot(key, this.wK);
    const V = Matrix.dot(value, this.wV);
    
    // Reshape for multi-head attention (simplified for single batch)
    const attention = this._scaledDotProductAttention(Q, K, V, mask);
    
    // Output projection
    return Matrix.dot(attention, this.wO);
  }

  _scaledDotProductAttention(Q, K, V, mask = null) {
    const seqLen = Q.rows;
    
    // Calculate attention scores: Q * K^T
    const scores = Matrix.dot(Q, Matrix.transpose(K));
    
    // Scale by sqrt(d_k)
    const scaledScores = scores.multiply(1.0 / Math.sqrt(this.dK));
    
    // Apply mask if provided
    let maskedScores = scaledScores;
    if (mask) {
      maskedScores = scaledScores.map((v, i, j) => {
        return mask.get(i, j) === 0 ? -Infinity : v;
      });
    }
    
    // Apply softmax
    const attentionWeights = softmax(maskedScores);
    
    // Apply attention weights to values
    return Matrix.dot(attentionWeights, V);
  }
}

// FEED FORWARD NETWORK 
class FeedForward {
  constructor(dModel, dFF = null) {
    this.dModel = dModel;
    this.dFF = dFF || 4 * dModel;
    
    const scale = Math.sqrt(1.0 / dModel);
    this.w1 = Matrix.random(dModel, this.dFF, scale);
    this.b1 = Matrix.zeros(1, this.dFF);
    this.w2 = Matrix.random(this.dFF, dModel, scale);
    this.b2 = Matrix.zeros(1, dModel);
  }

  forward(x) {
    // First linear transformation
    const hidden = Matrix.dot(x, this.w1);
    
    // Add bias (broadcast)
    const withBias1 = new Matrix(hidden.rows, hidden.cols);
    for (let i = 0; i < hidden.rows; i++) {
      for (let j = 0; j < hidden.cols; j++) {
        withBias1.set(i, j, hidden.get(i, j) + this.b1.get(0, j));
      }
    }
    
    // Apply activation (GELU)
    const activated = gelu(withBias1);
    
    // Second linear transformation
    const output = Matrix.dot(activated, this.w2);
    
    // Add bias (broadcast)
    const withBias2 = new Matrix(output.rows, output.cols);
    for (let i = 0; i < output.rows; i++) {
      for (let j = 0; j < output.cols; j++) {
        withBias2.set(i, j, output.get(i, j) + this.b2.get(0, j));
      }
    }
    
    return withBias2;
  }
}

// TRANSFORMER BLOCK
class TransformerBlock {
  constructor(dModel, numHeads, dFF = null) {
    this.attention = new MultiHeadAttention(dModel, numHeads);
    this.feedForward = new FeedForward(dModel, dFF);
    this.layerNorm1 = new LayerNorm(dModel);
    this.layerNorm2 = new LayerNorm(dModel);
  }

  forward(x, mask = null) {
    // Self-attention with residual connection
    const attnOutput = this.attention.forward(x, x, x, mask);
    const normed1 = this.layerNorm1.forward(x.add(attnOutput));
    
    // Feed-forward with residual connection
    const ffOutput = this.feedForward.forward(normed1);
    const normed2 = this.layerNorm2.forward(normed1.add(ffOutput));
    
    return normed2;
  }
}

// TRANSFORMER ENCODER 
class TransformerEncoder {
  constructor(dModel, numHeads, numLayers, dFF = null, maxLen = 5000) {
    this.dModel = dModel;
    this.numLayers = numLayers;
    this.positionalEncoding = new PositionalEncoding(dModel, maxLen);
    
    this.blocks = [];
    for (let i = 0; i < numLayers; i++) {
      this.blocks.push(new TransformerBlock(dModel, numHeads, dFF));
    }
  }

  forward(x, mask = null) {
    // Add positional encoding
    let output = this.positionalEncoding.forward(x);
    
    // Pass through transformer blocks
    for (const block of this.blocks) {
      output = block.forward(output, mask);
    }
    
    return output;
  }
}

// TRANSFORMER MODEL 
class Transformer {
  constructor(config) {
    const {
      dModel = 512,
      numHeads = 8,
      numLayers = 6,
      dFF = null,
      maxLen = 5000,
      vocabSize = null
    } = config;

    this.dModel = dModel;
    this.vocabSize = vocabSize;
    
    // Embedding layer (if vocab size is provided)
    if (vocabSize) {
      this.embedding = Matrix.random(vocabSize, dModel, Math.sqrt(1.0 / dModel));
    }
    
    this.encoder = new TransformerEncoder(dModel, numHeads, numLayers, dFF, maxLen);
    
    // Output projection (if vocab size is provided)
    if (vocabSize) {
      this.outputProjection = Matrix.random(dModel, vocabSize, Math.sqrt(1.0 / dModel));
    }
  }

  forward(input, mask = null) {
    let x = input;
    
    // Apply embedding if input is token indices
    if (this.vocabSize && input.cols === 1) {
      x = this._applyEmbedding(input);
    }
    
    // Pass through encoder
    const encoded = this.encoder.forward(x, mask);
    
    // Apply output projection if available
    if (this.outputProjection) {
      return Matrix.dot(encoded, this.outputProjection);
    }
    
    return encoded;
  }

  _applyEmbedding(tokenIndices) {
    const seqLen = tokenIndices.rows;
    const embedded = new Matrix(seqLen, this.dModel);
    
    for (let i = 0; i < seqLen; i++) {
      const tokenId = Math.floor(tokenIndices.get(i, 0));
      for (let j = 0; j < this.dModel; j++) {
        embedded.set(i, j, this.embedding.get(tokenId, j));
      }
    }
    
    return embedded;
  }

  // Generate text (simple greedy decoding)
  generate(prompt, maxLength = 50) {
    if (!this.vocabSize) {
      throw new Error('Vocab size must be set for text generation');
    }
    
    let sequence = Matrix.fromArray(prompt);
    
    for (let i = 0; i < maxLength; i++) {
      const output = this.forward(sequence);
      const lastOutput = new Matrix(1, this.vocabSize);
      
      // Get the last token's logits
      for (let j = 0; j < this.vocabSize; j++) {
        lastOutput.set(0, j, output.get(output.rows - 1, j));
      }
      
      // Apply softmax and get next token
      const probs = softmax(lastOutput);
      const nextToken = this._sampleFromDistribution(probs);
      
      // Add to sequence
      const newSequence = new Matrix(sequence.rows + 1, 1);
      for (let j = 0; j < sequence.rows; j++) {
        newSequence.set(j, 0, sequence.get(j, 0));
      }
      newSequence.set(sequence.rows, 0, nextToken);
      sequence = newSequence;
    }
    
    return sequence.toArray();
  }

  _sampleFromDistribution(probs) {
    const rand = Math.random();
    let cumSum = 0;
    
    for (let i = 0; i < probs.cols; i++) {
      cumSum += probs.get(0, i);
      if (rand <= cumSum) {
        return i;
      }
    }
    
    return probs.cols - 1;
  }
}

// UTILITY FUNCTIONS 
function createCausalMask(seqLen) {
  const mask = new Matrix(seqLen, seqLen);
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      mask.set(i, j, j <= i ? 1 : 0);
    }
  }
  return mask;
}

function createPaddingMask(tokens, padToken = 0) {
  const seqLen = tokens.length;
  const mask = new Matrix(seqLen, seqLen);
  
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      mask.set(i, j, tokens[j] !== padToken ? 1 : 0);
    }
  }
  
  return mask;
}

// EXPORTS 
// Universal module definition for compatibility
(function (root, factory) {
  if (typeof exports === 'object' && typeof module !== 'undefined') {
    // Node.js
    module.exports = factory();
  } else if (typeof define === 'function' && define.amd) {
    // AMD
    define([], factory);
  } else {
    // Browser globals
    root.Transformer = factory();
  }
}(typeof self !== 'undefined' ? self : this, function () {
  return {
    // Core classes
    Transformer,
    TransformerEncoder,
    TransformerBlock,
    MultiHeadAttention,
    FeedForward,
    LayerNorm,
    PositionalEncoding,
    
    // Utility classes
    Matrix,
    
    // Activation functions
    softmax,
    relu,
    gelu,
    
    // Utility functions
    createCausalMask,
    createPaddingMask
  };
}));