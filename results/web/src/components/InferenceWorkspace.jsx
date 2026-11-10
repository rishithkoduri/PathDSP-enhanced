import React, { useState } from 'react';
import InputPanel from './InputPanel';
import ResultsPanel from './ResultsPanel';
import { PRESET_CELL_LINES, PRESET_COMPOUNDS, generateInferenceResult } from '../data/mockData';
import { executeInference } from '../services/api';

export default function InferenceWorkspace({ gpuState, setComputing }) {
  const [selectedCellId, setSelectedCellId] = useState(PRESET_CELL_LINES[0].id);
  const [selectedCompoundId, setSelectedCompoundId] = useState(PRESET_COMPOUNDS[0].id);
  const [customSmiles, setCustomSmiles] = useState(PRESET_COMPOUNDS[0].smiles);
  const [rawExpression, setRawExpression] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(() => {
    return generateInferenceResult(PRESET_CELL_LINES[0].id, PRESET_COMPOUNDS[0].id);
  });

  const handleRunInference = async () => {
    setIsLoading(true);
    setComputing(true);
    try {
      const res = await executeInference({
        cellId: selectedCellId,
        compoundId: selectedCompoundId,
        smiles: customSmiles,
        rawExpression,
      });
      setResult(res);
    } catch {
      setResult(generateInferenceResult(selectedCellId, selectedCompoundId, customSmiles));
    } finally {
      setIsLoading(false);
      setComputing(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto py-10 px-6 space-y-8 text-left">
      {/* Header */}
      <div>
        <span className="text-[11px] font-mono text-emerald-600 dark:text-emerald-400 tracking-widest uppercase">
          Inference Engine &bull; A100 Tensor Core
        </span>
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-slate-900 dark:text-slate-50 mt-1">
          Drug Sensitivity Workspace
        </h1>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1.5 max-w-2xl font-sans">
          Select a verified cancer cell line and small molecule from the oncology library, or supply custom transcriptomic expression arrays and SMILES structures.
        </p>
      </div>

      {/* Two-column Segregated Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        
        {/* Left Column: Clearly Segregated Input Panels (5 Columns) */}
        <div className="lg:col-span-5 lg:sticky lg:top-20">
          <InputPanel
            selectedCellId={selectedCellId}
            setSelectedCellId={setSelectedCellId}
            selectedCompoundId={selectedCompoundId}
            setSelectedCompoundId={setSelectedCompoundId}
            customSmiles={customSmiles}
            setCustomSmiles={setCustomSmiles}
            rawExpression={rawExpression}
            setRawExpression={setRawExpression}
            onRunInference={handleRunInference}
            isLoading={isLoading}
          />
        </div>

        {/* Right Column: Results Dashboard (7 Columns) */}
        <div className="lg:col-span-7">
          <ResultsPanel
            result={result}
            isLoading={isLoading}
          />
        </div>

      </div>
    </div>
  );
}
