// Matrix class for neural network operations
class Matrix {
  constructor(rows, cols, data = null) {
    this.rows = rows;
    this.cols = cols;
    
    // Initialize with provided data or zeros
    if (data) {
      this.data = data;
    } else {
      this.data = [];
      for (let i = 0; i < rows; i++) {
        this.data[i] = new Array(cols).fill(0);
      }
    }
  }

  // Create matrix filled with random values between -1 and 1
  static random(rows, cols) {
    const matrix = new Matrix(rows, cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        matrix.data[i][j] = Math.random() * 2 - 1; // Random between -1 and 1
      }
    }
    return matrix;
  }

  // Create matrix from 1D array (useful for inputs/outputs)
  static fromArray(arr) {
    const matrix = new Matrix(arr.length, 1);
    for (let i = 0; i < arr.length; i++) {
      matrix.data[i][0] = arr[i];
    }
    return matrix;
  }

  // Convert matrix to 1D array
  toArray() {
    const arr = [];
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        arr.push(this.data[i][j]);
      }
    }
    return arr;
  }

  // Matrix addition - can add scalar or another matrix
  add(n) {
    if (n instanceof Matrix) {
      // Matrix addition
      if (this.rows !== n.rows || this.cols !== n.cols) {
        throw new Error('Matrix dimensions must match for addition');
      }
      const result = new Matrix(this.rows, this.cols);
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          result.data[i][j] = this.data[i][j] + n.data[i][j];
        }
      }
      return result;
    } else {
      // Scalar addition
      const result = new Matrix(this.rows, this.cols);
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          result.data[i][j] = this.data[i][j] + n;
        }
      }
      return result;
    }
  }

  // Matrix subtraction
  subtract(n) {
    if (n instanceof Matrix) {
      if (this.rows !== n.rows || this.cols !== n.cols) {
        throw new Error('Matrix dimensions must match for subtraction');
      }
      const result = new Matrix(this.rows, this.cols);
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          result.data[i][j] = this.data[i][j] - n.data[i][j];
        }
      }
      return result;
    } else {
      // Scalar subtraction
      const result = new Matrix(this.rows, this.cols);
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          result.data[i][j] = this.data[i][j] - n;
        }
      }
      return result;
    }
  }

  // Element-wise multiplication (Hadamard product)
  multiply(n) {
    if (n instanceof Matrix) {
      if (this.rows !== n.rows || this.cols !== n.cols) {
        throw new Error('Matrix dimensions must match for element-wise multiplication');
      }
      const result = new Matrix(this.rows, this.cols);
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          result.data[i][j] = this.data[i][j] * n.data[i][j];
        }
      }
      return result;
    } else {
      // Scalar multiplication
      const result = new Matrix(this.rows, this.cols);
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          result.data[i][j] = this.data[i][j] * n;
        }
      }
      return result;
    }
  }

  // Matrix multiplication (dot product)
  static multiply(a, b) {
    if (a.cols !== b.rows) {
      throw new Error('Columns of A must match rows of B for matrix multiplication');
    }
    
    const result = new Matrix(a.rows, b.cols);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
        let sum = 0;
        for (let k = 0; k < a.cols; k++) {
          sum += a.data[i][k] * b.data[k][j];
        }
        result.data[i][j] = sum;
      }
    }
    return result;
  }

  // Transpose matrix (flip rows and columns)
  transpose() {
    const result = new Matrix(this.cols, this.rows);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[j][i] = this.data[i][j];
      }
    }
    return result;
  }

  // Apply function to each element
  map(func) {
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[i][j] = func(this.data[i][j]);
      }
    }
    return result;
  }

  // Print matrix for debugging
  print() {
    console.table(this.data);
  }

  // Copy matrix
  copy() {
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[i][j] = this.data[i][j];
      }
    }
    return result;
  }
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Matrix;
}