
class WebGLCompute {
  constructor() {
    this.gl = null;
    this.isWebGLAvailable = false;
    this.programs = {};
    this.buffers = new Map();
    this.init();
  }

  init() {
    try {
      const canvas = document.createElement('canvas');
      this.gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      
      if (!this.gl) {
        console.warn('WebGL not available, falling back to CPU');
        return;
      }

      // Check for required extensions
      const ext = this.gl.getExtension('OES_texture_float');
      if (!ext) {
        console.warn('Float textures not supported, falling back to CPU');
        return;
      }

      this.isWebGLAvailable = true;
      this.setupShaders();
    } catch (e) {
      console.warn('WebGL initialization failed:', e);
      this.isWebGLAvailable = false;
    }
  }

  setupShaders() {
    const gl = this.gl;

    // Vertex shader (same for all operations)
    const vertexShaderSource = `
      attribute vec2 a_position;
      varying vec2 v_texCoord;
      void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
        v_texCoord = a_position * 0.5 + 0.5;
      }
    `;

    // Matrix multiplication shader
    const matMulFragmentShader = `
      precision mediump float;
      uniform sampler2D u_matrixA;
      uniform sampler2D u_matrixB;
      uniform float u_aRows;
      uniform float u_aCols;
      uniform float u_bCols;
      varying vec2 v_texCoord;
      
      void main() {
        float row = floor(v_texCoord.y * u_aRows);
        float col = floor(v_texCoord.x * u_bCols);
        
        float sum = 0.0;
        for (float k = 0.0; k < u_aCols; k += 1.0) {
          vec2 aCoord = vec2((k + 0.5) / u_aCols, (row + 0.5) / u_aRows);
          vec2 bCoord = vec2((col + 0.5) / u_bCols, (k + 0.5) / u_aCols);
          
          float aVal = texture2D(u_matrixA, aCoord).r;
          float bVal = texture2D(u_matrixB, bCoord).r;
          sum += aVal * bVal;
        }
        
        gl_FragColor = vec4(sum, 0.0, 0.0, 1.0);
      }
    `;

    // Element-wise operation shader
    const elementWiseFragmentShader = `
      precision mediump float;
      uniform sampler2D u_matrixA;
      uniform sampler2D u_matrixB;
      uniform float u_operation; // 0=add, 1=subtract, 2=multiply
      uniform float u_scalar;
      uniform float u_useScalar;
      varying vec2 v_texCoord;
      
      void main() {
        float a = texture2D(u_matrixA, v_texCoord).r;
        float b = u_useScalar > 0.5 ? u_scalar : texture2D(u_matrixB, v_texCoord).r;
        
        float result;
        if (u_operation < 0.5) {
          result = a + b; // add
        } else if (u_operation < 1.5) {
          result = a - b; // subtract
        } else {
          result = a * b; // multiply
        }
        
        gl_FragColor = vec4(result, 0.0, 0.0, 1.0);
      }
    `;

    this.programs.matMul = this.createProgram(vertexShaderSource, matMulFragmentShader);
    this.programs.elementWise = this.createProgram(vertexShaderSource, elementWiseFragmentShader);

    // Create vertex buffer for quad
    const positions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    this.quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
  }

  createProgram(vertexSource, fragmentSource) {
    const gl = this.gl;
    
    const vertexShader = this.createShader(gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = this.createShader(gl.FRAGMENT_SHADER, fragmentSource);
    
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error('Program link error: ' + gl.getProgramInfoLog(program));
    }
    
    return program;
  }

  createShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error('Shader compile error: ' + gl.getShaderInfoLog(shader));
    }
    
    return shader;
  }

  createTexture(data, width, height) {
    const gl = this.gl;
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, width, height, 0, gl.LUMINANCE, gl.FLOAT, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    
    return texture;
  }

  matrixMultiply(a, b) {
    if (!this.isWebGLAvailable) return null;
    
    const gl = this.gl;
    const program = this.programs.matMul;
    
    // Create textures
    const texA = this.createTexture(a.data, a.cols, a.rows);
    const texB = this.createTexture(b.data, b.cols, b.rows);
    
    // Create framebuffer for output
    const fb = gl.createFramebuffer();
    const outputTex = this.createTexture(null, b.cols, a.rows);
    
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outputTex, 0);
    
    // Setup viewport and program
    gl.viewport(0, 0, b.cols, a.rows);
    gl.useProgram(program);
    
    // Set uniforms
    gl.uniform1f(gl.getUniformLocation(program, 'u_aRows'), a.rows);
    gl.uniform1f(gl.getUniformLocation(program, 'u_aCols'), a.cols);
    gl.uniform1f(gl.getUniformLocation(program, 'u_bCols'), b.cols);
    
    // Bind textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texA);
    gl.uniform1i(gl.getUniformLocation(program, 'u_matrixA'), 0);
    
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, texB);
    gl.uniform1i(gl.getUniformLocation(program, 'u_matrixB'), 1);
    
    // Setup vertex attributes
    const positionLoc = gl.getAttribLocation(program, 'a_position');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
    
    // Draw
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    
    // Read result
    const result = new Float32Array(a.rows * b.cols);
    gl.readPixels(0, 0, b.cols, a.rows, gl.LUMINANCE, gl.FLOAT, result);
    
    // Cleanup
    gl.deleteTexture(texA);
    gl.deleteTexture(texB);
    gl.deleteTexture(outputTex);
    gl.deleteFramebuffer(fb);
    
    return result;
  }

  elementWiseOperation(a, b, operation, scalar = 0, useScalar = false) {
    if (!this.isWebGLAvailable) return null;
    
    const gl = this.gl;
    const program = this.programs.elementWise;
    
    // Create textures
    const texA = this.createTexture(a.data, a.cols, a.rows);
    const texB = useScalar ? null : this.createTexture(b.data, b.cols, b.rows);
    
    // Create framebuffer for output
    const fb = gl.createFramebuffer();
    const outputTex = this.createTexture(null, a.cols, a.rows);
    
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outputTex, 0);
    
    // Setup viewport and program
    gl.viewport(0, 0, a.cols, a.rows);
    gl.useProgram(program);
    
    // Set uniforms
    gl.uniform1f(gl.getUniformLocation(program, 'u_operation'), operation);
    gl.uniform1f(gl.getUniformLocation(program, 'u_scalar'), scalar);
    gl.uniform1f(gl.getUniformLocation(program, 'u_useScalar'), useScalar ? 1.0 : 0.0);
    
    // Bind textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texA);
    gl.uniform1i(gl.getUniformLocation(program, 'u_matrixA'), 0);
    
    if (!useScalar) {
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, texB);
      gl.uniform1i(gl.getUniformLocation(program, 'u_matrixB'), 1);
    }
    
    // Setup vertex attributes
    const positionLoc = gl.getAttribLocation(program, 'a_position');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
    
    // Draw
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    
    // Read result
    const result = new Float32Array(a.rows * a.cols);
    gl.readPixels(0, 0, a.cols, a.rows, gl.LUMINANCE, gl.FLOAT, result);
    
    // Cleanup
    gl.deleteTexture(texA);
    if (texB) gl.deleteTexture(texB);
    gl.deleteTexture(outputTex);
    gl.deleteFramebuffer(fb);
    
    return result;
  }
}

// Global WebGL compute instance
const webglCompute = new WebGLCompute();

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

  // Create a matrix filled with ones
  static ones(rows, cols) {
    const m = new Matrix(rows, cols);
    m.data.fill(1);
    return m;
  }

  // Create a matrix with random values in [-scale, scale]
  static random(rows, cols, scale = 1) {
    const m = new Matrix(rows, cols);
    for (let i = 0; i < m.data.length; i++) {
      m.data[i] = (Math.random() * 2 - 1) * scale;
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

  // Create matrix from 2D array
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

  // Convert matrix data to a JS array
  toArray() {
    return Array.from(this.data);
  }

  // Convert to 2D array
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

  // Clone matrix
  clone() {
    return new Matrix(this.rows, this.cols, new Float32Array(this.data));
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

  // Add matrix or scalar (WebGL accelerated)
  add(n) {
    if (n instanceof Matrix) {
      if (n.rows !== this.rows || n.cols !== this.cols) {
        throw new Error('Dimension mismatch');
      }
      
      // Try WebGL first
      const webglResult = webglCompute.elementWiseOperation(this, n, 0);
      if (webglResult) {
        return new Matrix(this.rows, this.cols, webglResult);
      }
      
      // Fallback to CPU
      return this.map((v, i, j) => v + n.get(i, j));
    }
    
    // Scalar addition
    const webglResult = webglCompute.elementWiseOperation(this, null, 0, n, true);
    if (webglResult) {
      return new Matrix(this.rows, this.cols, webglResult);
    }
    
    return this.map(v => v + n);
  }

  // Subtract matrix or scalar (WebGL accelerated)
  subtract(n) {
    if (n instanceof Matrix) {
      if (n.rows !== this.rows || n.cols !== this.cols) {
        throw new Error('Dimension mismatch');
      }
      
      // Try WebGL first
      const webglResult = webglCompute.elementWiseOperation(this, n, 1);
      if (webglResult) {
        return new Matrix(this.rows, this.cols, webglResult);
      }
      
      // Fallback to CPU
      return this.map((v, i, j) => v - n.get(i, j));
    }
    
    // Scalar subtraction
    const webglResult = webglCompute.elementWiseOperation(this, null, 1, n, true);
    if (webglResult) {
      return new Matrix(this.rows, this.cols, webglResult);
    }
    
    return this.map(v => v - n);
  }

  // Element-wise multiply (Hadamard) or scalar (WebGL accelerated)
  multiply(n) {
    if (n instanceof Matrix) {
      if (n.rows !== this.rows || n.cols !== this.cols) {
        throw new Error('Dimension mismatch');
      }
      
      // Try WebGL first
      const webglResult = webglCompute.elementWiseOperation(this, n, 2);
      if (webglResult) {
        return new Matrix(this.rows, this.cols, webglResult);
      }
      
      // Fallback to CPU
      return this.map((v, i, j) => v * n.get(i, j));
    }
    
    // Scalar multiplication
    const webglResult = webglCompute.elementWiseOperation(this, null, 2, n, true);
    if (webglResult) {
      return new Matrix(this.rows, this.cols, webglResult);
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

  // Dot product: matrix multiplication (WebGL accelerated)
  static dot(a, b) {
    if (a.cols !== b.rows) {
      throw new Error(`Cannot multiply ${a.rows}x${a.cols} with ${b.rows}x${b.cols}`);
    }
    
    // Try WebGL first
    const webglResult = webglCompute.matrixMultiply(a, b);
    if (webglResult) {
      return new Matrix(a.rows, b.cols, webglResult);
    }
    
    // Fallback to CPU
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

  // Get performance info
  static getPerformanceInfo() {
    return {
      webglAvailable: webglCompute.isWebGLAvailable,
      webglContext: webglCompute.gl ? 'Available' : 'Not available'
    };
  }
}

export default Matrix;