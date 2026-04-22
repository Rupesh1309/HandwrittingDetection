// ============================================
// Canvas Drawing Module
// ============================================

const CanvasDrawing = {
  canvas: null,
  ctx: null,
  isDrawing: false,
  currentTool: 'pen',
  currentColor: '#000000',
  brushSize: 3,
  history: [],
  historyIndex: -1,
  points: [],
  hasDrawn: false,

  init() {
    this.canvas = document.getElementById('drawingCanvas');
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());
    this.setupDrawingEvents();
    this.setupToolbar();
    this.setupModeTabs();
    this.setupRecognizeButton();
    this.saveState();
  },

  resizeCanvas() {
    const container = document.getElementById('canvasContainer');
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const w = Math.max(Math.floor(rect.width) - 4, 300);
    const h = Math.max(Math.floor(rect.height) - 4, 400);
    const snapshot = this.canvas.toDataURL();
    this.canvas.width = w;
    this.canvas.height = h;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fillRect(0, 0, w, h);
    if (this.hasDrawn) {
      const img = new Image();
      img.onload = () => this.ctx.drawImage(img, 0, 0);
      img.src = snapshot;
    }
  },

  getPos(e) {
    const rect = this.canvas.getBoundingClientRect();
    const sx = this.canvas.width / rect.width;
    const sy = this.canvas.height / rect.height;
    if (e.touches && e.touches[0]) {
      return { x: (e.touches[0].clientX - rect.left) * sx, y: (e.touches[0].clientY - rect.top) * sy };
    }
    return { x: (e.clientX - rect.left) * sx, y: (e.clientY - rect.top) * sy };
  },

  setupDrawingEvents() {
    this.canvas.addEventListener('mousedown',  (e) => this.startDraw(e));
    this.canvas.addEventListener('mousemove',  (e) => this.draw(e));
    this.canvas.addEventListener('mouseup',    () => this.endDraw());
    this.canvas.addEventListener('mouseleave', () => this.endDraw());
    this.canvas.addEventListener('touchstart', (e) => { e.preventDefault(); this.startDraw(e); }, { passive: false });
    this.canvas.addEventListener('touchmove',  (e) => { e.preventDefault(); this.draw(e); },      { passive: false });
    this.canvas.addEventListener('touchend',   (e) => { e.preventDefault(); this.endDraw(); });
    this.canvas.addEventListener('touchcancel',() => this.endDraw());
  },

  startDraw(e) {
    this.isDrawing = true;
    const pos = this.getPos(e);
    this.points = [pos];
    this.ctx.beginPath();
    this.ctx.moveTo(pos.x, pos.y);
    // Hide hint on first draw
    const hint = document.getElementById('canvasHint');
    if (hint) hint.style.display = 'none';
    this.hasDrawn = true;
  },

  draw(e) {
    if (!this.isDrawing) return;
    const pos = this.getPos(e);
    this.points.push(pos);
    this.ctx.lineWidth = this.currentTool === 'eraser' ? this.brushSize * 4 : this.brushSize;
    this.ctx.strokeStyle = this.currentTool === 'eraser' ? '#ffffff' : this.currentColor;
    this.ctx.globalCompositeOperation = 'source-over';
    if (this.points.length >= 3) {
      const last = this.points.length - 1;
      const cp = { x: (this.points[last - 1].x + this.points[last].x) / 2, y: (this.points[last - 1].y + this.points[last].y) / 2 };
      this.ctx.quadraticCurveTo(this.points[last - 1].x, this.points[last - 1].y, cp.x, cp.y);
    } else {
      this.ctx.lineTo(pos.x, pos.y);
    }
    this.ctx.stroke();
    this.ctx.beginPath();
    this.ctx.moveTo(pos.x, pos.y);
  },

  endDraw() {
    if (!this.isDrawing) return;
    this.isDrawing = false;
    this.points = [];
    this.saveState();
  },

  saveState() {
    if (this.historyIndex < this.history.length - 1) {
      this.history = this.history.slice(0, this.historyIndex + 1);
    }
    this.history.push(this.canvas.toDataURL());
    this.historyIndex = this.history.length - 1;
    if (this.history.length > 40) { this.history.shift(); this.historyIndex--; }
  },

  undo() {
    if (this.historyIndex > 0) { this.historyIndex--; this.restoreState(); }
  },

  redo() {
    if (this.historyIndex < this.history.length - 1) { this.historyIndex++; this.restoreState(); }
  },

  restoreState() {
    const img = new Image();
    img.onload = () => { this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height); this.ctx.drawImage(img, 0, 0); };
    img.src = this.history[this.historyIndex];
  },

  clearCanvas() {
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    const hint = document.getElementById('canvasHint');
    if (hint) hint.style.display = 'flex';
    this.hasDrawn = false;
    this.saveState();
  },

  getImageData() {
    // Export as JPEG with high quality for better recognition
    return this.canvas.toDataURL('image/jpeg', 0.95);
  },

  isCanvasBlank() {
    // Use the hasDrawn flag for fast, reliable check
    return !this.hasDrawn;
  },

  setTool(tool) {
    this.currentTool = tool;
    document.querySelectorAll('.tool-btn[data-tool]').forEach(b => b.classList.remove('active'));
    document.getElementById(tool === 'pen' ? 'penTool' : 'eraserTool')?.classList.add('active');
    this.canvas.style.cursor = tool === 'eraser' ? 'cell' : 'crosshair';
  },

  setupToolbar() {
    document.getElementById('penTool')?.addEventListener('click', () => this.setTool('pen'));
    document.getElementById('eraserTool')?.addEventListener('click', () => this.setTool('eraser'));

    document.querySelectorAll('.color-preset').forEach(preset => {
      preset.addEventListener('click', () => {
        this.currentColor = preset.dataset.color;
        this.setTool('pen');
        document.querySelectorAll('.color-preset').forEach(p => p.classList.remove('active'));
        preset.classList.add('active');
        const cpBtn = document.getElementById('colorPickerBtn');
        if (cpBtn) cpBtn.style.background = this.currentColor;
        const cp = document.getElementById('colorPicker');
        if (cp) cp.value = this.currentColor;
      });
    });

    const colorPicker = document.getElementById('colorPicker');
    const colorPickerBtn = document.getElementById('colorPickerBtn');
    colorPickerBtn?.addEventListener('click', () => colorPicker?.click());
    colorPicker?.addEventListener('input', (e) => {
      this.currentColor = e.target.value;
      colorPickerBtn.style.background = e.target.value;
      document.querySelectorAll('.color-preset').forEach(p => p.classList.remove('active'));
      this.setTool('pen');
    });

    const brushSizeInput = document.getElementById('brushSize');
    const sizeLabel = document.getElementById('sizeLabel');
    brushSizeInput?.addEventListener('input', (e) => {
      this.brushSize = parseInt(e.target.value);
      if (sizeLabel) sizeLabel.textContent = `${this.brushSize}px`;
    });

    document.getElementById('undoBtn')?.addEventListener('click', () => this.undo());
    document.getElementById('redoBtn')?.addEventListener('click', () => this.redo());
    document.getElementById('clearBtn')?.addEventListener('click', () => {
      if (confirm('Clear the canvas? This cannot be undone.')) this.clearCanvas();
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.ctrlKey && e.key === 'z') { e.preventDefault(); this.undo(); }
      if (e.ctrlKey && e.key === 'y') { e.preventDefault(); this.redo(); }
      if (e.key === 'p' || e.key === 'P') this.setTool('pen');
      if (e.key === 'e' || e.key === 'E') this.setTool('eraser');
    });
  },

  setupModeTabs() {
    const drawTab    = document.getElementById('drawTab');
    const uploadTab  = document.getElementById('uploadTab');
    const canvasPanel = document.getElementById('canvasPanel');
    const uploadPanel = document.getElementById('uploadPanel');

    drawTab?.addEventListener('click', () => {
      drawTab.classList.add('active');
      uploadTab.classList.remove('active');
      canvasPanel.style.display = 'flex';
      uploadPanel.style.display = 'none';
    });

    uploadTab?.addEventListener('click', () => {
      uploadTab.classList.add('active');
      drawTab.classList.remove('active');
      uploadPanel.style.display = 'flex';
      canvasPanel.style.display = 'none';
    });
  },

  setupRecognizeButton() {
    document.getElementById('recognizeBtn')?.addEventListener('click', async () => {
      const isDrawMode = document.getElementById('drawTab')?.classList.contains('active');

      let imageData = null;
      if (isDrawMode) {
        if (this.isCanvasBlank()) {
          API.showToast('Please draw something on the canvas first!', 'error');
          return;
        }
        imageData = this.getImageData();
      } else {
        imageData = typeof ImageUpload !== 'undefined' ? ImageUpload.getImageData() : null;
        if (!imageData) {
          API.showToast('Please upload an image first!', 'error');
          return;
        }
      }

      try {
        await API.recognize(imageData);
      } catch (_) { /* errors handled inside API.recognize */ }
    });
  },
};

document.addEventListener('DOMContentLoaded', () => CanvasDrawing.init());
