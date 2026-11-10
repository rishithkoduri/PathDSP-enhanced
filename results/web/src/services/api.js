import { PRESET_CELL_LINES, PRESET_COMPOUNDS, MODEL_METRICS, generateInferenceResult } from '../data/mockData';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

/**
 * Fetch GPU hardware status from FastAPI endpoint or fallback to simulated healthy state
 */
export async function getGpuStatus() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);
    const res = await fetch(`${API_BASE}/gpu-status`, { signal: controller.signal });
    clearTimeout(timeoutId);
    if (!res.ok) throw new Error('Status non-200');
    return await res.json();
  } catch (err) {
    return {
      status: 'online',
      device: 'NVIDIA A100-SXM4 (MIG 20GB)',
      utilizationPct: 12,
      memoryUsedMb: 3420,
      memoryTotalMb: 20480,
      temperatureC: 42,
      driverVersion: '535.129.03',
      cudaVersion: '12.2',
      backendConnected: false,
    };
  }
}

/**
 * Fetch preloaded cell lines, compounds, and model metadata
 */
export async function getPresets() {
  try {
    const res = await fetch(`${API_BASE}/presets`);
    if (res.ok) return await res.json();
  } catch (e) {
    // fallback
  }
  return {
    cellLines: PRESET_CELL_LINES,
    compounds: PRESET_COMPOUNDS,
    modelMetrics: MODEL_METRICS,
  };
}

/**
 * Execute real-time GNN forward pass inference
 */
export async function executeInference(payload) {
  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (res.ok) {
      const data = await res.json();
      return { ...data, backendConnected: true };
    }
  } catch (err) {
    // fallback
  }

  // Local evaluation emulation
  await new Promise((r) => setTimeout(r, 600));
  return {
    ...generateInferenceResult(
      payload.cellId,
      payload.compoundId,
      payload.smiles
    ),
    backendConnected: false,
  };
}

/**
 * Export tabular results to CSV file
 */
export function exportToCsv(filename, rows, headers) {
  const headerLine = headers.join(',');
  const lines = rows.map((r) =>
    headers
      .map((h) => {
        const val = r[h] !== undefined ? r[h] : '';
        return typeof val === 'string' && val.includes(',') ? `"${val}"` : val;
      })
      .join(',')
  );
  const csvContent = 'data:text/csv;charset=utf-8,' + [headerLine, ...lines].join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
