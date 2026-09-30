/**
 * Fingernail Anemia Screening - Live Research Prototype
 * Real-time dynamic frontend orchestrator.
 */

(function () {
  'use strict';

  function getApiBase() {
    const saved = localStorage.getItem('ANEMIA_API_BASE');
    if (saved) return saved.replace(/\/+$/, '');

    if (window.location.hostname.endsWith('github.io')) {
      return 'https://anemia-ai.onrender.com';
    }

    return window.location.origin;
  }

  const API_BASE = getApiBase();
  const REQUEST_TIMEOUT_MS = 25000;

  // DOM Elements - Navigation & Views
  const cameraTabBtn = document.getElementById('cameraTabBtn');
  const uploadTabBtn = document.getElementById('uploadTabBtn');
  const cameraViewContainer = document.getElementById('cameraViewContainer');
  const uploadViewContainer = document.getElementById('uploadViewContainer');
  const analyzingOverlay = document.getElementById('analyzingOverlay');
  const deviceBadge = document.getElementById('deviceBadge');

  // Camera Elements
  const cameraVideo = document.getElementById('cameraVideo');
  const captureCanvas = document.getElementById('captureCanvas');
  const captureBtn = document.getElementById('captureBtn');

  // File Upload Elements
  const imageFileInput = document.getElementById('imageFileInput');
  const uploadDropZone = document.getElementById('uploadDropZone');
  const uploadPlaceholder = document.getElementById('uploadPlaceholder');
  const uploadPreviewWrapper = document.getElementById('uploadPreviewWrapper');
  const uploadPreviewImg = document.getElementById('uploadPreviewImg');
  const analyzeUploadBtn = document.getElementById('analyzeUploadBtn');
  const reselectUploadBtn = document.getElementById('reselectUploadBtn');

  // Result Section Elements
  const statusPill = document.getElementById('statusPill');
  const emptyResultState = document.getElementById('emptyResultState');
  const activeResultState = document.getElementById('activeResultState');
  const resultBanner = document.getElementById('resultBanner');
  const bannerStateLabel = document.getElementById('bannerStateLabel');
  const bannerSubtext = document.getElementById('bannerSubtext');

  // Primary Metrics
  const metricConfidence = document.getElementById('metricConfidence');
  const metricLatency = document.getElementById('metricLatency');
  const metricDevice = document.getElementById('metricDevice');
  const metricThreshold = document.getElementById('metricThreshold');

  // Probability Bar
  const probBarNumber = document.getElementById('probBarNumber');
  const probBarFill = document.getElementById('probBarFill');

  // Inconclusive Box
  const inconclusiveNotice = document.getElementById('inconclusiveNotice');
  const inconclusiveReasonText = document.getElementById('inconclusiveReasonText');

  // Dual Image Preview
  const displayOriginalImg = document.getElementById('displayOriginalImg');
  const displayRoiImg = document.getElementById('displayRoiImg');
  const noRoiPlaceholder = document.getElementById('noRoiPlaceholder');

  // Transparent Panel Intermediate Values
  const valEffnetLogit = document.getElementById('valEffnetLogit');
  const valEffnetProb = document.getElementById('valEffnetProb');
  const valJetxProb = document.getElementById('valJetxProb');
  const valFusionProb = document.getElementById('valFusionProb');
  const valCalibratedProb = document.getElementById('valCalibratedProb');
  const valThreshold = document.getElementById('valThreshold');
  const valFinalState = document.getElementById('valFinalState');
  const valLatencyBreakdown = document.getElementById('valLatencyBreakdown');

  // History Table
  const historyTableBody = document.getElementById('historyTableBody');
  const clearHistoryBtn = document.getElementById('clearHistoryBtn');

  // Application State
  let stream = null;
  let activeTab = 'camera';
  let selectedFileBlob = null;
  let currentOriginalDataUrl = null;
  const sessionHistory = [];

  // Initialize Application
  async function init() {
    setupEventListeners();
    await fetchSystemHealth();
    switchTab('camera');
  }

  // Event Listeners Setup
  function setupEventListeners() {
    cameraTabBtn.addEventListener('click', () => switchTab('camera'));
    uploadTabBtn.addEventListener('click', () => switchTab('upload'));

    // Camera Actions
    captureBtn.addEventListener('click', onCaptureFrame);

    // Upload Dropzone & File Pick Actions
    uploadDropZone.addEventListener('click', () => imageFileInput.click());
    uploadDropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadDropZone.classList.add('dragover');
    });
    uploadDropZone.addEventListener('dragleave', () => {
      uploadDropZone.classList.remove('dragover');
    });
    uploadDropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadDropZone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileSelection(e.dataTransfer.files[0]);
      }
    });

    imageFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
      }
    });

    analyzeUploadBtn.addEventListener('click', onAnalyzeUploadedImage);
    reselectUploadBtn.addEventListener('click', () => imageFileInput.click());
    clearHistoryBtn.addEventListener('click', clearHistory);
  }

  // Tab Navigation
  function switchTab(tab) {
    activeTab = tab;
    if (tab === 'camera') {
      cameraTabBtn.classList.add('active');
      cameraTabBtn.setAttribute('aria-selected', 'true');
      uploadTabBtn.classList.remove('active');
      uploadTabBtn.setAttribute('aria-selected', 'false');

      cameraViewContainer.classList.add('active');
      uploadViewContainer.classList.remove('active');

      startCamera();
    } else {
      uploadTabBtn.classList.add('active');
      uploadTabBtn.setAttribute('aria-selected', 'true');
      cameraTabBtn.classList.remove('active');
      cameraTabBtn.setAttribute('aria-selected', 'false');

      uploadViewContainer.classList.add('active');
      cameraViewContainer.classList.remove('active');

      stopCamera();
    }
  }

  // Camera Management
  async function startCamera() {
    stopCamera();
    try {
      const constraints = {
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      };
      stream = await navigator.mediaDevices.getUserMedia(constraints);
      cameraVideo.srcObject = stream;
      await cameraVideo.play();
    } catch (err) {
      console.warn('Camera access unavailable or denied:', err);
    }
  }

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
    }
  }

  // Capture Frame
  async function onCaptureFrame() {
    if (!cameraVideo.srcObject && cameraVideo.readyState < 2) {
      alert('Camera is not active or video is not ready.');
      return;
    }

    const vw = cameraVideo.videoWidth || 640;
    const vh = cameraVideo.videoHeight || 480;
    captureCanvas.width = vw;
    captureCanvas.height = vh;
    const ctx = captureCanvas.getContext('2d');
    ctx.drawImage(cameraVideo, 0, 0, vw, vh);

    currentOriginalDataUrl = captureCanvas.toDataURL('image/jpeg', 0.95);

    captureCanvas.toBlob(async (blob) => {
      if (blob) {
        await executeInference(blob, 'camera_frame.jpg');
      }
    }, 'image/jpeg', 0.95);
  }

  // File Upload Handlers
  function handleFileSelection(file) {
    if (!file || !file.type.startsWith('image/')) {
      alert('Please select a valid image file (JPEG, PNG, or WEBP).');
      return;
    }

    selectedFileBlob = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      currentOriginalDataUrl = e.target.result;
      uploadPreviewImg.src = currentOriginalDataUrl;
      uploadPlaceholder.classList.add('hidden');
      uploadPreviewWrapper.classList.remove('hidden');

      analyzeUploadBtn.disabled = false;
      analyzeUploadBtn.classList.remove('disabled');
      reselectUploadBtn.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
  }

  async function onAnalyzeUploadedImage() {
    if (!selectedFileBlob) return;
    await executeInference(selectedFileBlob, selectedFileBlob.name || 'uploaded_nail.jpg');
  }

  // API Call & Pipeline Execution
  async function executeInference(imageBlob, filename) {
    console.log('[CLIENT] REQUEST START:', { filename, sizeBytes: imageBlob.size, type: imageBlob.type, timestamp: new Date().toISOString() });
    
    // UI Scanning State
    analyzingOverlay.classList.remove('hidden');
    statusPill.textContent = 'ANALYZING';
    statusPill.style.color = '#06b6d4';

    const formData = new FormData();
    formData.append('file', imageBlob, filename);

    console.log('[CLIENT] IMAGE SENT to POST /predict');

    const tStart = performance.now();
    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData,
      });

      const latencyClientMs = (performance.now() - tStart).toFixed(1);
      console.log('[CLIENT] SERVER RESPONSE received in', latencyClientMs, 'ms with status', response.status);

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned error ${response.status}`);
      }

      const resultData = await response.json();
      console.log('[CLIENT] MODEL OUTPUT:', resultData);
      console.log('[CLIENT] FINAL RESULT:', {
        state: resultData.state,
        probability: resultData.probability,
        threshold: resultData.threshold,
        latency_ms: resultData.latency_ms,
      });

      renderAssessmentResult(resultData, latencyClientMs);

    } catch (err) {
      console.error('[CLIENT] INFERENCE ERROR:', err);
      renderInconclusiveResult(err.message || 'Network or server communication error.');
    } finally {
      analyzingOverlay.classList.add('hidden');
    }
  }

  // Render Result on Dashboard
  function renderAssessmentResult(data, clientLatencyMs) {
    emptyResultState.classList.add('hidden');
    activeResultState.classList.remove('hidden');

    const state = data.state || 'INCONCLUSIVE';
    const isAnemia = (state === 'ANEMIA');
    const isNonAnemia = (state === 'NO_ANEMIA');
    const isInconclusive = (!isAnemia && !isNonAnemia);

    // Update Banner
    resultBanner.className = 'result-banner';
    inconclusiveNotice.classList.add('hidden');

    if (isAnemia) {
      resultBanner.classList.add('banner-anemia');
      bannerStateLabel.textContent = 'ANEMIA RISK';
      bannerSubtext.textContent = data.description || 'Model-estimated ensemble probability indicates potential anemia.';
      statusPill.textContent = 'ANEMIA RISK';
      statusPill.style.color = '#ef4444';
    } else if (isNonAnemia) {
      resultBanner.classList.add('banner-non-anemia');
      bannerStateLabel.textContent = 'NON-ANEMIA';
      bannerSubtext.textContent = data.description || 'Model-estimated ensemble probability indicates no anemia detected.';
      statusPill.textContent = 'NON-ANEMIA';
      statusPill.style.color = '#10b981';
    } else {
      resultBanner.classList.add('banner-inconclusive');
      bannerStateLabel.textContent = 'INCONCLUSIVE';
      bannerSubtext.textContent = data.description || 'Image could not be reliably assessed for anemia risk.';
      statusPill.textContent = 'INCONCLUSIVE';
      statusPill.style.color = '#f59e0b';

      inconclusiveNotice.classList.remove('hidden');
      inconclusiveReasonText.textContent = data.description || 'Image quality or physiological validation check failed.';
    }

    // Confidence / Probability
    const prob = data.probability;
    const probPctStr = (typeof prob === 'number') ? (prob * 100).toFixed(2) + '%' : 'N/A';
    metricConfidence.textContent = probPctStr;
    probBarNumber.textContent = probPctStr;

    if (typeof prob === 'number') {
      const fillPct = Math.min(100, Math.max(0, prob * 100));
      probBarFill.style.width = `${fillPct}%`;
    } else {
      probBarFill.style.width = '0%';
    }

    // Latency
    const totalMs = data.latency_ms ? data.latency_ms.total : clientLatencyMs;
    metricLatency.textContent = `${totalMs} ms`;

    // Device
    const deviceStr = data.device ? data.device.toUpperCase() : 'CUDA';
    metricDevice.textContent = deviceStr;

    // Threshold
    const tau = data.threshold !== undefined ? data.threshold.toFixed(4) : '0.9000';
    metricThreshold.innerHTML = `&tau; = ${tau}`;

    // Dual Image Preview
    if (currentOriginalDataUrl) {
      displayOriginalImg.src = currentOriginalDataUrl;
    }
    if (data.roi_image_base64) {
      displayRoiImg.src = data.roi_image_base64;
      displayRoiImg.classList.remove('hidden');
      noRoiPlaceholder.classList.add('hidden');
    } else {
      displayRoiImg.classList.add('hidden');
      noRoiPlaceholder.classList.remove('hidden');
    }

    // Intermediate Panel Values
    valEffnetLogit.textContent = (data.efficientnet_raw_logit !== undefined) ? data.efficientnet_raw_logit.toFixed(4) : '--';
    valEffnetProb.textContent = (data.efficientnet_probability !== undefined) ? data.efficientnet_probability.toFixed(4) : '--';
    valJetxProb.textContent = (data.jetx_gt_probability !== undefined) ? data.jetx_gt_probability.toFixed(4) : '--';
    valFusionProb.textContent = (data.raw_fusion_probability !== undefined) ? data.raw_fusion_probability.toFixed(4) : '--';
    valCalibratedProb.textContent = (typeof prob === 'number') ? prob.toFixed(4) : '--';
    valThreshold.textContent = tau;
    valFinalState.textContent = state;

    if (data.latency_ms) {
      valLatencyBreakdown.textContent = `EffNet: ${data.latency_ms.efficientnet}ms | JetX: ${data.latency_ms.jetx_gt}ms | Fusion: ${data.latency_ms.fusion}ms | Total: ${data.latency_ms.total}ms`;
    } else {
      valLatencyBreakdown.textContent = `${clientLatencyMs} ms`;
    }

    // Record to Session History
    recordSessionHistory({
      time: new Date().toLocaleTimeString(),
      state: isAnemia ? 'ANEMIA' : (isNonAnemia ? 'NON-ANEMIA' : 'INCONCLUSIVE'),
      probStr: probPctStr,
      effnetLogit: (data.efficientnet_raw_logit !== undefined) ? data.efficientnet_raw_logit.toFixed(3) : '--',
      jetxProb: (data.jetx_gt_probability !== undefined) ? data.jetx_gt_probability.toFixed(3) : '--',
      latency: `${totalMs}ms`,
      device: deviceStr,
    });
  }

  function renderInconclusiveResult(errorMessage) {
    renderAssessmentResult({
      success: true,
      state: 'INCONCLUSIVE',
      probability: null,
      description: errorMessage,
      threshold: 0.9000,
    }, '0.0');
  }

  // Session History Management
  function recordSessionHistory(entry) {
    sessionHistory.unshift(entry);
    if (sessionHistory.length > 20) sessionHistory.pop();
    renderHistoryTable();
  }

  function renderHistoryTable() {
    if (sessionHistory.length === 0) {
      historyTableBody.innerHTML = '<tr class="empty-history-row"><td colspan="7">No screening requests recorded in this session.</td></tr>';
      return;
    }

    let rowsHtml = '';
    sessionHistory.forEach(item => {
      let tagClass = 'inconclusive';
      if (item.state === 'ANEMIA') tagClass = 'anemia';
      else if (item.state === 'NON-ANEMIA') tagClass = 'non-anemia';

      rowsHtml += `
        <tr>
          <td>${item.time}</td>
          <td><span class="history-tag ${tagClass}">${item.state}</span></td>
          <td>${item.probStr}</td>
          <td>${item.effnetLogit}</td>
          <td>${item.jetxProb}</td>
          <td>${item.latency}</td>
          <td>${item.device}</td>
        </tr>
      `;
    });
    historyTableBody.innerHTML = rowsHtml;
  }

  function clearHistory() {
    sessionHistory.length = 0;
    renderHistoryTable();
  }

  // System Health & Device Verification
  async function fetchSystemHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        const devName = data.gpu && data.gpu !== 'None' ? `${data.gpu} (${data.device.toUpperCase()})` : data.device.toUpperCase();
        deviceBadge.textContent = `⚡ DEVICE: ${devName}`;
      }
    } catch (e) {
      console.warn('Could not query /health:', e);
      deviceBadge.textContent = '⚡ DEVICE: LOCALHOST (READY)';
    }
  }

  // Kickoff on DOM Ready
  document.addEventListener('DOMContentLoaded', init);
})();
