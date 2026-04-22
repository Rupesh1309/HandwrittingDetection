// ============================================
// API Communication Module
// ============================================

const API = {
  baseUrl: '/api',

  // Show toast notification
  showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icons = { success: '✅', error: '❌', info: 'ℹ️' };
    toast.innerHTML = `<span>${icons[type] || icons.info}</span><span style="flex:1">${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(120%)';
      toast.style.transition = 'all 0.4s ease-out';
      setTimeout(() => toast.remove(), 400);
    }, 4000);
  },

  // Send image for recognition
  async recognize(imageBase64, customPrompt = null) {
    const btn = document.getElementById('recognizeBtn');
    const btnText = document.getElementById('recognizeBtnText');

    // Loading state
    btn.disabled = true;
    btnText.innerHTML = '<div class="spinner"></div>&nbsp; Analyzing handwriting…';

    try {
      const response = await fetch(`${this.baseUrl}/recognize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: imageBase64,
          prompt: customPrompt,
        }),
      });

      const data = await response.json();

      if (!response.ok || data.error) {
        // Show disguised user-friendly error — no mention of API
        throw new Error(data.error || 'The recognition model encountered an issue. Please try again.');
      }

      this.displayResults(data);
      this.saveToHistory(data);
      this.showToast('Recognition complete!', 'success');
      return data;

    } catch (error) {
      console.error('[InkMind] Error:', error.message);
      // Show the server's disguised message, or a generic fallback
      const msg = error.message && error.message.length < 200
        ? error.message
        : 'The recognition model could not process this image. Please try again.';
      this.showToast(msg, 'error');
      this.showErrorState(msg);
      throw error;
    } finally {
      btn.disabled = false;
      btnText.innerHTML = '🔍 Recognize Handwriting';
    }
  },

  // Show error state in results panel
  showErrorState(message) {
    const resultEmpty = document.getElementById('resultEmpty');
    const resultContent = document.getElementById('resultContent');
    if (resultEmpty) {
      resultEmpty.style.display = 'flex';
      resultEmpty.innerHTML = `
        <div class="empty-icon">⚠️</div>
        <p style="color:#f87171;">${message}</p>
        <p style="font-size:0.8rem;margin-top:0.5rem;">Try a clearer image or draw more clearly on the canvas.</p>
      `;
    }
    if (resultContent) resultContent.style.display = 'none';
  },

  // Display recognition results
  displayResults(data) {
    const resultEmpty = document.getElementById('resultEmpty');
    const resultContent = document.getElementById('resultContent');
    const recognizedText = document.getElementById('recognizedText');
    const confidenceContainer = document.getElementById('confidenceContainer');
    const analysisContainer = document.getElementById('analysisContainer');
    const analysisText = document.getElementById('analysisText');

    // Reset empty state
    if (resultEmpty) {
      resultEmpty.style.display = 'none';
      resultEmpty.innerHTML = `
        <div class="empty-icon">📝</div>
        <p>Draw or upload an image, then click<br><strong>"Recognize Handwriting"</strong> to see results</p>
      `;
    }

    resultContent.style.display = 'flex';
    resultContent.style.flexDirection = 'column';
    resultContent.style.gap = '1rem';

    // Recognized text — preserve line breaks for multi-line content
    const rawText = data.recognizedText || 'No text could be detected in this image.';
    // Safely escape HTML, then convert newlines to <br> for proper multi-line display
    const escaped = rawText.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    recognizedText.innerHTML = escaped.replace(/\n/g, '<br>');
    recognizedText.style.whiteSpace = 'pre-wrap';
    recognizedText.style.lineHeight = '1.8';
    recognizedText.style.fontFamily = "'Segoe UI', monospace";
    recognizedText.style.fontSize = '1.05rem';

    // Confidence badge
    const conf = (data.confidence || 'medium').toLowerCase();
    const confMap = {
      high:   { cls: 'high',   label: '🟢 High Accuracy' },
      medium: { cls: 'medium', label: '🟡 Medium Accuracy' },
      low:    { cls: 'low',    label: '🔴 Low Accuracy' },
    };
    const cm = confMap[conf] || confMap.medium;
    confidenceContainer.innerHTML = `<span class="confidence-badge ${cm.cls}">${cm.label}</span>`;

    // Analysis
    if (data.analysis) {
      analysisContainer.style.display = 'block';
      analysisText.textContent = data.analysis;
    } else {
      analysisContainer.style.display = 'none';
    }
  },

  // Save to localStorage history
  saveToHistory(data) {
    const history = JSON.parse(localStorage.getItem('inkmind_history') || '[]');
    history.unshift({
      text: (data.recognizedText || '').substring(0, 120),
      confidence: data.confidence || 'medium',
      timestamp: new Date().toISOString(),
    });
    if (history.length > 20) history.length = 20;
    localStorage.setItem('inkmind_history', JSON.stringify(history));
    this.renderHistory();
  },

  // Render history list
  renderHistory() {
    const historyList = document.getElementById('historyList');
    const historySection = document.getElementById('historySection');
    if (!historyList) return;

    const history = JSON.parse(localStorage.getItem('inkmind_history') || '[]');
    if (history.length === 0) {
      if (historySection) historySection.style.display = 'none';
      return;
    }

    if (historySection) historySection.style.display = 'block';
    historyList.innerHTML = history.map((item, idx) => `
      <div class="history-item" data-idx="${idx}">
        <div class="history-text">${item.text || '(No text detected)'}</div>
        <div class="history-time">${new Date(item.timestamp).toLocaleString()}</div>
      </div>
    `).join('');

    // Click to restore
    historyList.querySelectorAll('.history-item').forEach(el => {
      el.addEventListener('click', () => {
        const idx = parseInt(el.dataset.idx);
        const item = history[idx];
        if (!item) return;
        const recognizedText = document.getElementById('recognizedText');
        const resultEmpty = document.getElementById('resultEmpty');
        const resultContent = document.getElementById('resultContent');
        if (recognizedText) recognizedText.textContent = item.text;
        if (resultEmpty) resultEmpty.style.display = 'none';
        if (resultContent) {
          resultContent.style.display = 'flex';
          resultContent.style.flexDirection = 'column';
          resultContent.style.gap = '1rem';
        }
        const conf = (item.confidence || 'medium').toLowerCase();
        const confMap = {
          high:   { cls: 'high',   label: '🟢 High Accuracy' },
          medium: { cls: 'medium', label: '🟡 Medium Accuracy' },
          low:    { cls: 'low',    label: '🔴 Low Accuracy' },
        };
        const cm = confMap[conf] || confMap.medium;
        const cc = document.getElementById('confidenceContainer');
        if (cc) cc.innerHTML = `<span class="confidence-badge ${cm.cls}">${cm.label}</span>`;
      });
    });
  },

  // Initialize
  init() {
    // Copy button
    document.getElementById('copyBtn')?.addEventListener('click', () => {
      const text = document.getElementById('recognizedText')?.textContent;
      if (text && text !== 'No text could be detected in this image.') {
        navigator.clipboard.writeText(text).then(() => {
          this.showToast('Copied to clipboard!', 'success');
        }).catch(() => {
          this.showToast('Copy failed — please select and copy manually.', 'error');
        });
      }
    });

    // Clear history
    document.getElementById('clearHistory')?.addEventListener('click', () => {
      localStorage.removeItem('inkmind_history');
      this.renderHistory();
      this.showToast('History cleared', 'info');
    });

    // History toggle
    document.getElementById('historyToggle')?.addEventListener('click', () => {
      const section = document.getElementById('historySection');
      if (section) {
        const isHidden = section.style.display === 'none' || !section.style.display;
        section.style.display = isHidden ? 'block' : 'none';
        this.renderHistory();
      }
    });

    this.renderHistory();
  },
};

document.addEventListener('DOMContentLoaded', () => API.init());
