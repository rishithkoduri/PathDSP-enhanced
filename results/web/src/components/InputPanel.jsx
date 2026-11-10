import React, { useState, useMemo, useRef, useEffect } from 'react';
import { Play, Upload, Search, Check, RefreshCw, ChevronDown, Dna, Pill, FileText, Sparkles, Filter } from 'lucide-react';
import { PRESET_CELL_LINES, PRESET_COMPOUNDS } from '../data/mockData';

export default function InputPanel({
  selectedCellId,
  setSelectedCellId,
  selectedCompoundId,
  setSelectedCompoundId,
  customSmiles,
  setCustomSmiles,
  rawExpression,
  setRawExpression,
  onRunInference,
  isLoading
}) {
  // Modes
  const [cellMode, setCellMode] = useState('preset'); // 'preset' | 'upload'
  const [compoundMode, setCompoundMode] = useState('library'); // 'library' | 'custom'

  // Dropdown open states
  const [cellDropdownOpen, setCellDropdownOpen] = useState(false);
  const [drugDropdownOpen, setDrugDropdownOpen] = useState(false);

  // Search filters
  const [cellSearch, setCellSearch] = useState('');
  const [drugSearch, setDrugSearch] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');

  // File uploads
  const [uploadedCellFile, setUploadedCellFile] = useState(null);
  const [uploadedFpFile, setUploadedFpFile] = useState(null);

  const cellRef = useRef(null);
  const drugRef = useRef(null);

  // Close dropdowns on outside click
  useEffect(() => {
    function handleClickOutside(e) {
      if (cellRef.current && !cellRef.current.contains(e.target)) {
        setCellDropdownOpen(false);
      }
      if (drugRef.current && !drugRef.current.contains(e.target)) {
        setDrugDropdownOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Filtered Cell Lines
  const filteredCellLines = useMemo(() => {
    if (!cellSearch.trim()) return PRESET_CELL_LINES;
    const term = cellSearch.toLowerCase();
    return PRESET_CELL_LINES.filter(c =>
      c.name.toLowerCase().includes(term) ||
      c.fullName.toLowerCase().includes(term) ||
      c.tissue.toLowerCase().includes(term) ||
      c.driverMutations.some(m => m.toLowerCase().includes(term))
    );
  }, [cellSearch]);

  // Drug Categories
  const drugCategories = useMemo(() => {
    const cats = new Set(PRESET_COMPOUNDS.map(c => c.category));
    return ['All', ...Array.from(cats)];
  }, []);

  // Filtered Drugs
  const filteredDrugs = useMemo(() => {
    return PRESET_COMPOUNDS.filter(d => {
      const matchesCategory = selectedCategory === 'All' || d.category === selectedCategory;
      if (!drugSearch.trim()) return matchesCategory;
      const term = drugSearch.toLowerCase();
      const matchesSearch =
        d.name.toLowerCase().includes(term) ||
        d.targetClass.toLowerCase().includes(term) ||
        d.mechanism.toLowerCase().includes(term) ||
        d.formula.toLowerCase().includes(term);
      return matchesCategory && matchesSearch;
    });
  }, [drugSearch, selectedCategory]);

  const currentCell = PRESET_CELL_LINES.find(c => c.id === selectedCellId) || PRESET_CELL_LINES[0];
  const currentCompound = PRESET_COMPOUNDS.find(c => c.id === selectedCompoundId) || PRESET_COMPOUNDS[0];

  const handleSelectCell = (cell) => {
    setSelectedCellId(cell.id);
    setCellDropdownOpen(false);
  };

  const handleSelectCompound = (compound) => {
    setSelectedCompoundId(compound.id);
    setCustomSmiles(compound.smiles);
    setDrugDropdownOpen(false);
  };

  const handleCsvUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedCellFile(file.name);
      const reader = new FileReader();
      reader.onload = (event) => setRawExpression(event.target.result);
      reader.readAsText(file);
    }
  };

  const handleFpUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedFpFile(file.name);
    }
  };

  return (
    <div className="w-full space-y-5 text-left">

      {/* ─────────────────────────────────────────────────────────────
          FIELD 1: CELL LINE PROFILE INPUT (CLEARLY SEGREGATED CARD)
      ────────────────────────────────────────────────────────────── */}
      <div className="rounded-2xl bg-white dark:bg-slate-900/60 p-6 border border-slate-200/60 dark:border-slate-800/60 shadow-xs space-y-4">
        
        {/* Header with Title & Mode Toggle */}
        <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800/80">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 flex items-center justify-center">
              <Dna className="w-3.5 h-3.5" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                Cell Line Profile
              </h3>
              <p className="text-[11px] text-slate-400 font-sans">
                Cancer cell transcriptomic multi-omics baseline
              </p>
            </div>
          </div>

          {/* Pill Toggle for Cell Line Input Mode */}
          <div className="inline-flex rounded-full bg-slate-100 dark:bg-slate-800 p-0.5 text-[11px] font-mono">
            <button
              type="button"
              onClick={() => setCellMode('preset')}
              className={`px-3 py-1 rounded-full transition-colors ${
                cellMode === 'preset'
                  ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 font-medium shadow-xs'
                  : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-300'
              }`}
            >
              GDSC / CCLE Cohort
            </button>
            <button
              type="button"
              onClick={() => setCellMode('upload')}
              className={`px-3 py-1 rounded-full transition-colors ${
                cellMode === 'upload'
                  ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 font-medium shadow-xs'
                  : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-300'
              }`}
            >
              Upload CSV
            </button>
          </div>
        </div>

        {/* Preset Mode: Custom Clear Dropdown Bar */}
        {cellMode === 'preset' ? (
          <div className="space-y-3" ref={cellRef}>
            <label className="text-[11px] font-mono text-slate-400 uppercase tracking-wider block">
              Select Cell Line
            </label>

            {/* Custom Dropdown Trigger Bar */}
            <div className="relative">
              <button
                type="button"
                onClick={() => setCellDropdownOpen(v => !v)}
                className="w-full flex items-center justify-between px-4 py-3 text-xs font-mono rounded-xl bg-slate-50 dark:bg-slate-950 border border-slate-200/80 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700 focus:outline-none focus:ring-2 focus:ring-emerald-500/30 transition-all text-left"
              >
                <div className="flex items-center gap-2.5 truncate">
                  <span className="font-bold text-slate-900 dark:text-slate-100 text-sm">
                    {currentCell.name}
                  </span>
                  <span className="text-[11px] text-slate-400 font-sans truncate">
                    &bull; {currentCell.tissue} ({currentCell.lineage})
                  </span>
                </div>
                <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${cellDropdownOpen ? 'rotate-180' : ''}`} />
              </button>

              {/* Dropdown Menu Popover */}
              {cellDropdownOpen && (
                <div className="absolute top-full left-0 right-0 mt-2 z-50 rounded-xl bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800 shadow-xl overflow-hidden animate-fade-in">
                  
                  {/* Search inside dropdown */}
                  <div className="p-3 border-b border-slate-100 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-950/50">
                    <div className="relative">
                      <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                      <input
                        type="text"
                        value={cellSearch}
                        onChange={(e) => setCellSearch(e.target.value)}
                        placeholder="Search by name, tissue, or mutation (e.g. BRAF)..."
                        className="w-full pl-8 pr-3 py-1.5 text-xs font-sans rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-slate-900 dark:text-slate-100 placeholder:text-slate-400 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                        autoFocus
                      />
                    </div>
                  </div>

                  {/* Cell Line List */}
                  <div className="max-h-56 overflow-y-auto divide-y divide-slate-100/60 dark:divide-slate-800/40 p-1">
                    {filteredCellLines.map((c) => (
                      <button
                        key={c.id}
                        type="button"
                        onClick={() => handleSelectCell(c)}
                        className={`w-full flex items-center justify-between p-2.5 rounded-lg text-left text-xs transition-colors ${
                          c.id === selectedCellId
                            ? 'bg-emerald-50 dark:bg-emerald-950/40 text-emerald-900 dark:text-emerald-200'
                            : 'hover:bg-slate-50 dark:hover:bg-slate-800/50 text-slate-700 dark:text-slate-300'
                        }`}
                      >
                        <div className="space-y-0.5">
                          <div className="flex items-center gap-2">
                            <span className="font-mono font-bold text-slate-900 dark:text-slate-100">
                              {c.name}
                            </span>
                            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-slate-500">
                              {c.tissue}
                            </span>
                          </div>
                          <p className="text-[11px] text-slate-400 font-sans truncate">
                            {c.fullName}
                          </p>
                        </div>
                        {c.id === selectedCellId && (
                          <Check className="w-4 h-4 text-emerald-500" />
                        )}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Selected Cell Annotation Card */}
            {currentCell && (
              <div className="p-3.5 rounded-xl bg-slate-50/80 dark:bg-slate-950/50 border border-slate-100 dark:border-slate-800/70 text-xs space-y-1.5">
                <div className="flex justify-between items-center text-[11px]">
                  <span className="text-slate-400 font-mono">Tissue Lineage:</span>
                  <span className="font-medium text-slate-800 dark:text-slate-200">{currentCell.fullName}</span>
                </div>
                <div className="flex justify-between items-center text-[11px]">
                  <span className="text-slate-400 font-mono">Driver Mutations:</span>
                  <span className="font-mono text-emerald-600 dark:text-emerald-400 font-semibold">
                    {currentCell.driverMutations.join(', ')}
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 dark:text-slate-400 pt-1 border-t border-slate-200/60 dark:border-slate-800/60 font-sans">
                  {currentCell.expressionSummary}
                </p>
              </div>
            )}
          </div>
        ) : (
          /* Upload Mode */
          <div className="space-y-3">
            <p className="text-xs text-slate-400 font-sans">
              Provide normalized expression vector (gene symbols and expression scores):
            </p>
            <div className="border border-dashed border-slate-300 dark:border-slate-700 rounded-xl p-5 text-center">
              <input
                type="file"
                id="csv-cell-upload"
                accept=".csv,.tsv,.txt"
                onChange={handleCsvUpload}
                className="hidden"
              />
              <label
                htmlFor="csv-cell-upload"
                className="cursor-pointer flex flex-col items-center justify-center space-y-1 text-xs font-mono"
              >
                <Upload className="w-5 h-5 text-slate-400 mb-1" />
                <span className="text-slate-700 dark:text-slate-200 font-medium">
                  {uploadedCellFile ? uploadedCellFile : 'Click to select expression .CSV file'}
                </span>
                <span className="text-[10px] text-slate-400 font-sans">
                  Standard format: gene_symbol, expression_value
                </span>
              </label>
            </div>
            {uploadedCellFile && (
              <div className="flex items-center gap-1.5 text-xs font-mono text-emerald-600 dark:text-emerald-400">
                <Check className="w-3.5 h-3.5" />
                <span>Transcriptomic array loaded ({rawExpression.length} bytes)</span>
              </div>
            )}
          </div>
        )}

      </div>


      {/* ─────────────────────────────────────────────────────────────
          FIELD 2: SMALL-MOLECULE DRUG INPUT (CLEARLY SEGREGATED CARD)
      ────────────────────────────────────────────────────────────── */}
      <div className="rounded-2xl bg-white dark:bg-slate-900/60 p-6 border border-slate-200/60 dark:border-slate-800/60 shadow-xs space-y-4">
        
        {/* Header with Title & Mode Toggle */}
        <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800/80">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 flex items-center justify-center">
              <Pill className="w-3.5 h-3.5" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                Target Drug Compound
              </h3>
              <p className="text-[11px] text-slate-400 font-sans">
                Oncology library ({PRESET_COMPOUNDS.length} validated compounds) &amp; SMILES
              </p>
            </div>
          </div>

          {/* Pill Toggle for Compound Input Mode */}
          <div className="inline-flex rounded-full bg-slate-100 dark:bg-slate-800 p-0.5 text-[11px] font-mono">
            <button
              type="button"
              onClick={() => setCompoundMode('library')}
              className={`px-3 py-1 rounded-full transition-colors ${
                compoundMode === 'library'
                  ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 font-medium shadow-xs'
                  : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-300'
              }`}
            >
              Drug Library
            </button>
            <button
              type="button"
              onClick={() => setCompoundMode('custom')}
              className={`px-3 py-1 rounded-full transition-colors ${
                compoundMode === 'custom'
                  ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 font-medium shadow-xs'
                  : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-300'
              }`}
            >
              Custom Morgan FP
            </button>
          </div>
        </div>

        {compoundMode === 'library' ? (
          <div className="space-y-3" ref={drugRef}>
            
            <div className="flex items-center justify-between">
              <label className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">
                Select Drug Candidate
              </label>
              <span className="text-[10px] font-mono text-indigo-500">
                {PRESET_COMPOUNDS.length} Compounds Available
              </span>
            </div>

            {/* Custom Clear Dropdown Bar for Drugs */}
            <div className="relative">
              <button
                type="button"
                onClick={() => setDrugDropdownOpen(v => !v)}
                className="w-full flex items-center justify-between px-4 py-3 text-xs font-mono rounded-xl bg-slate-50 dark:bg-slate-950 border border-slate-200/80 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500/30 transition-all text-left"
              >
                <div className="flex items-center gap-2.5 truncate">
                  <span className="font-bold text-slate-900 dark:text-slate-100 text-sm">
                    {currentCompound.name}
                  </span>
                  <span className="text-[11px] font-sans px-2 py-0.5 rounded-full bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 border border-indigo-200/60 dark:border-indigo-800/40">
                    {currentCompound.category}
                  </span>
                </div>
                <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${drugDropdownOpen ? 'rotate-180' : ''}`} />
              </button>

              {/* Dropdown Menu Popover */}
              {drugDropdownOpen && (
                <div className="absolute top-full left-0 right-0 mt-2 z-50 rounded-xl bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800 shadow-xl overflow-hidden animate-fade-in">
                  
                  {/* Search and Category Filter Toolbar inside dropdown */}
                  <div className="p-3 border-b border-slate-100 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-950/50 space-y-2">
                    <div className="relative">
                      <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                      <input
                        type="text"
                        value={drugSearch}
                        onChange={(e) => setDrugSearch(e.target.value)}
                        placeholder="Search by drug name, target, or mechanism..."
                        className="w-full pl-8 pr-3 py-1.5 text-xs font-sans rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-slate-900 dark:text-slate-100 placeholder:text-slate-400 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                        autoFocus
                      />
                    </div>

                    {/* Category Filter Pills */}
                    <div className="flex items-center gap-1 overflow-x-auto pb-1 scrollbar-none text-[10px] font-mono">
                      {drugCategories.map((cat) => (
                        <button
                          key={cat}
                          type="button"
                          onClick={() => setSelectedCategory(cat)}
                          className={`px-2 py-0.5 rounded-full whitespace-nowrap transition-colors ${
                            selectedCategory === cat
                              ? 'bg-indigo-600 text-white font-medium'
                              : 'bg-slate-100 dark:bg-slate-800 text-slate-500 hover:text-slate-900 dark:hover:text-slate-200'
                          }`}
                        >
                          {cat}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Drug List */}
                  <div className="max-h-64 overflow-y-auto divide-y divide-slate-100/60 dark:divide-slate-800/40 p-1">
                    {filteredDrugs.map((d) => (
                      <button
                        key={d.id}
                        type="button"
                        onClick={() => handleSelectCompound(d)}
                        className={`w-full flex items-center justify-between p-2.5 rounded-lg text-left text-xs transition-colors ${
                          d.id === selectedCompoundId
                            ? 'bg-indigo-50 dark:bg-indigo-950/40 text-indigo-900 dark:text-indigo-200'
                            : 'hover:bg-slate-50 dark:hover:bg-slate-800/50 text-slate-700 dark:text-slate-300'
                        }`}
                      >
                        <div className="space-y-0.5 truncate pr-2">
                          <div className="flex items-center gap-2">
                            <span className="font-mono font-bold text-slate-900 dark:text-slate-100">
                              {d.name}
                            </span>
                            <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-100 dark:bg-slate-800 text-slate-500">
                              {d.formula}
                            </span>
                          </div>
                          <p className="text-[11px] text-slate-400 font-sans truncate">
                            {d.targetClass} &bull; {d.category}
                          </p>
                        </div>
                        {d.id === selectedCompoundId && (
                          <Check className="w-4 h-4 text-indigo-500 shrink-0" />
                        )}
                      </button>
                    ))}
                    {filteredDrugs.length === 0 && (
                      <div className="p-4 text-center text-xs text-slate-400">
                        No compounds match the filter
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Selected Compound Mechanism Card */}
            {currentCompound && (
              <div className="p-3.5 rounded-xl bg-slate-50/80 dark:bg-slate-950/50 border border-slate-100 dark:border-slate-800/70 text-xs space-y-1.5">
                <div className="flex justify-between items-center text-[11px]">
                  <span className="text-slate-400 font-mono">Target Class:</span>
                  <span className="font-medium text-slate-800 dark:text-slate-200">{currentCompound.targetClass}</span>
                </div>
                <div className="flex justify-between items-center text-[11px]">
                  <span className="text-slate-400 font-mono">Formula / Weight:</span>
                  <span className="font-mono text-slate-700 dark:text-slate-300">
                    {currentCompound.formula} &bull; {currentCompound.mw} g/mol
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 dark:text-slate-400 pt-1 border-t border-slate-200/60 dark:border-slate-800/60 font-sans">
                  {currentCompound.mechanism}
                </p>
              </div>
            )}

            {/* Canonical SMILES Input Field */}
            <div className="space-y-1">
              <label className="text-[11px] font-mono text-slate-400 flex items-center justify-between">
                <span>Canonical SMILES String</span>
                <span className="text-[10px] text-slate-400">Auto-filled or editable</span>
              </label>
              <input
                type="text"
                value={customSmiles}
                onChange={(e) => setCustomSmiles(e.target.value)}
                placeholder="Enter or modify canonical SMILES..."
                className="w-full px-3 py-2 text-xs font-mono rounded-xl bg-slate-50 dark:bg-slate-950 border border-slate-200/80 dark:border-slate-800 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </div>

          </div>
        ) : (
          /* Custom Morgan FP Mode */
          <div className="space-y-3">
            <p className="text-xs text-slate-400 font-sans">
              Upload pre-computed 256-bit binary Morgan fingerprint (radius 3) in CSV or NPY format:
            </p>
            <div className="border border-dashed border-slate-300 dark:border-slate-700 rounded-xl p-5 text-center">
              <input
                type="file"
                id="fp-file-upload"
                accept=".csv,.npy,.json,.txt"
                onChange={handleFpUpload}
                className="hidden"
              />
              <label
                htmlFor="fp-file-upload"
                className="cursor-pointer flex flex-col items-center justify-center space-y-1 text-xs font-mono"
              >
                <FileText className="w-5 h-5 text-slate-400 mb-1" />
                <span className="text-slate-700 dark:text-slate-200 font-medium">
                  {uploadedFpFile ? uploadedFpFile : 'Click to select 256-bit Fingerprint array'}
                </span>
                <span className="text-[10px] text-slate-400 font-sans">
                  Array dimension: [256 bits]
                </span>
              </label>
            </div>
            {uploadedFpFile && (
              <div className="flex items-center gap-1.5 text-xs font-mono text-emerald-600 dark:text-emerald-400">
                <Check className="w-3.5 h-3.5" />
                <span>Morgan Fingerprint loaded into memory</span>
              </div>
            )}
          </div>
        )}

      </div>


      {/* ─────────────────────────────────────────────────────────────
          EXECUTION BAR: PROMINENT RUN INFERENCE BUTTON
      ────────────────────────────────────────────────────────────── */}
      <div className="rounded-2xl bg-white dark:bg-slate-900/60 p-5 border border-slate-200/60 dark:border-slate-800/60 shadow-xs space-y-2">
        <button
          type="button"
          onClick={onRunInference}
          disabled={isLoading}
          className={`w-full py-3 px-6 rounded-full font-mono text-xs sm:text-sm font-semibold tracking-wide flex items-center justify-center space-x-2 transition-all shadow-sm ${
            isLoading
              ? 'bg-slate-200 dark:bg-slate-800 text-slate-400 cursor-not-allowed'
              : 'bg-emerald-600 hover:bg-emerald-500 active:scale-[0.99] text-white shadow-emerald-600/20'
          }`}
        >
          {isLoading ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin text-amber-300" />
              <span>Computing Forward Pass on A100...</span>
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>Run GNN Inference</span>
            </>
          )}
        </button>

        <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 px-1 pt-1">
          <span>Checkpoint: <code>best_cv_fold8.pt</code></span>
          <span>A100 Tensor Core BF16</span>
        </div>
      </div>

    </div>
  );
}
