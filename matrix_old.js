// matrix.js
// Implements a basic Matrix class using Float32Array for performance.
// Supports common operations: creation, addition, subtraction, multiplication, transpose, dot, map.

export class Matrix {
  constructor(rows, cols, data = null) {
    this.rows = rows;
    this.cols = cols;
    this.data = data instanceof Float32Array
      ? data
      : new Float32Array(rows * cols);
  }

  // Create a matrix filled with zeros
  static zeros(rows, cols) {
    return new Matrix(rows, cols);
  }

  // Create a matrix with random values in [-1, 1]
  static random(rows, cols) {
    const m = new Matrix(rows, cols);
    for (let i = 0; i < m.data.length; i++) {
      m.data[i] = Math.random() * 2 - 1;
    }
    return m;
  }

  // Convert a flat array into a column vector
  static fromArray(arr) {
    const m = new Matrix(arr.length, 1);
    for (let i = 0; i < arr.length; i++) {
      m.data[i] = arr[i];
    }
    return m;
  }

  // Convert matrix data to a JS array
  toArray() {
    return Array.from(this.data);
  }

  // Internal index calc
  _idx(i, j) {
    return i * this.cols + j;
  }

  // Get value at (i, j)
  get(i, j) {
    return this.data[this._idx(i, j)];
  }

  // Set value at (i, j)
  set(i, j, v) {
    this.data[this._idx(i, j)] = v;
  }

  // Apply a function to every element, return new Matrix
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

  // Add matrix or scalar
  add(n) {
    if (n instanceof Matrix) {
      if (n.rows !== this.rows || n.cols !== this.cols) {
        throw new Error('Dimension mismatch');
      }
      return this.map((v, i, j) => v + n.get(i, j));
    }
    return this.map(v => v + n);
  }

  // Subtract matrix or scalar
  subtract(n) {
    if (n instanceof Matrix) {
      if (n.rows !== this.rows || n.cols !== this.cols) {
        throw new Error('Dimension mismatch');
      }
      return this.map((v, i, j) => v - n.get(i, j));
    }
    return this.map(v => v - n);
  }

  // Element-wise multiply (Hadamard) or scalar
  multiply(n) {
    if (n instanceof Matrix) {
      if (n.rows !== this.rows || n.cols !== this.cols) {
        throw new Error('Dimension mismatch');
      }
      return this.map((v, i, j) => v * n.get(i, j));
    }
    return this.map(v => v * n);
  }

  // Transpose the matrix
  static transpose(mat) {
    const result = new Matrix(mat.cols, mat.rows);
    for (let i = 0; i < mat.rows; i++) {
      for (let j = 0; j < mat.cols; j++) {
        result.set(j, i, mat.get(i, j));
      }
    }
    return result;
  }

  // Dot product: matrix multiplication
  static dot(a, b) {
    if (a.cols !== b.rows) {
      throw new Error('Dimension mismatch for dot product');
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
}

export default Matrix;