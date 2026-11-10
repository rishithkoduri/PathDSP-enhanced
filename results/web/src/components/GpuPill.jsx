import { useState } from 'react';
import { Server } from 'lucide-react';

const statusConfig = {
  online: { dot: 'bg-emerald-500', label: 'Online', pulse: true },
  computing: { dot: 'bg-amber-400', label: 'Computing...', pulse: true },
  offline: { dot: 'bg-slate-400', label: 'Offline', pulse: false },
};

export default function GpuPill({ gpuState }) {
  const [show, setShow] = useState(false);

  const {
    status = 'offline',
    device = 'Unknown',
    utilizationPct = 0,
    memoryUsedMb = 0,
    memoryTotalMb = 0,
    backendConnected = false,
  } = gpuState || {};

  const cfg = statusConfig[status] || statusConfig.offline;

  return (
    <div
      className="relative flex items-center gap-1.5 cursor-pointer select-none"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
      onClick={() => setShow((v) => !v)}
    >
      <span
        className={`w-2 h-2 rounded-full ${cfg.dot} ${cfg.pulse ? 'animate-pulse-dot' : ''}`}
      />
      <span className="text-[11px] font-mono text-slate-500 dark:text-slate-400">
        {cfg.label}
      </span>

      {show && (
        <div className="absolute top-full right-0 mt-2 w-72 rounded-2xl p-5 bg-white dark:bg-slate-900 shadow-lg border border-slate-200/60 dark:border-slate-800/60 z-50">
          <div className="flex items-center gap-2 mb-4">
            <Server className="w-4 h-4 text-emerald-500" />
            <span className="text-xs font-semibold text-slate-900 dark:text-slate-100">
              GPU Status
            </span>
          </div>

          <div className="space-y-3 text-[12px] font-mono">
            <Row label="Device" value={device} />
            <Row label="Utilization" value={`${utilizationPct}%`} />
            <Row
              label="VRAM"
              value={`${(memoryUsedMb / 1024).toFixed(1)} / ${(memoryTotalMb / 1024).toFixed(1)} GB`}
            />
            <Row label="Precision" value="BF16 / TF32" />
            <Row
              label="Backend"
              value={
                <span
                  className={
                    backendConnected
                      ? 'text-emerald-500'
                      : 'text-slate-400'
                  }
                >
                  {backendConnected ? 'Connected' : 'Disconnected'}
                </span>
              }
            />
          </div>
        </div>
      )}
    </div>
  );
}

function Row({ label, value }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-slate-400 dark:text-slate-500">{label}</span>
      <span className="text-slate-700 dark:text-slate-300">{value}</span>
    </div>
  );
}
