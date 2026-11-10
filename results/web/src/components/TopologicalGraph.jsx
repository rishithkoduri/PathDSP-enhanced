import React, { useRef, useEffect, useState } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, Download } from 'lucide-react';

export default function TopologicalGraph({ genes = [], edges = [], compoundName, cellName }) {
  const canvasRef = useRef(null);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const nodesRef = useRef([]);

  useEffect(() => {
    const topGenes = genes.slice(0, 15);
    const canvas = canvasRef.current;
    if (!canvas) return;

    const width = canvas.width || 600;
    const height = canvas.height || 360;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) * 0.36;

    const positioned = topGenes.map((g, idx) => {
      if (idx === 0) return { ...g, x: centerX, y: centerY - 10, r: 15 };
      if (idx === 1) return { ...g, x: centerX - 50, y: centerY + 30, r: 13 };
      if (idx === 2) return { ...g, x: centerX + 50, y: centerY + 30, r: 13 };
      const angle = ((idx - 3) / Math.max(1, topGenes.length - 3)) * 2 * Math.PI;
      return {
        ...g,
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius,
        r: Math.max(9, Math.min(13, 9 + (g.saliency || 0) * 50)),
      };
    });

    nodesRef.current = positioned;
    draw();
  }, [genes, edges, zoom, offset, hoveredNode, selectedNode]);

  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.save();
    ctx.translate(offset.x, offset.y);
    ctx.scale(zoom, zoom);

    const nodes = nodesRef.current;
    const nodeMap = new Map(nodes.map((n) => [n.gene, n]));

    // Draw Edges
    edges.forEach((edge) => {
      const src = nodeMap.get(edge.source);
      const tgt = nodeMap.get(edge.target);
      if (src && tgt) {
        ctx.beginPath();
        ctx.moveTo(src.x, src.y);
        ctx.lineTo(tgt.x, tgt.y);

        const isHighlighted =
          (hoveredNode && (hoveredNode.gene === src.gene || hoveredNode.gene === tgt.gene)) ||
          (selectedNode && (selectedNode.gene === src.gene || selectedNode.gene === tgt.gene));

        const baseWidth = Math.max(1, (edge.attention || 0) * 40);
        ctx.lineWidth = isHighlighted ? baseWidth * 1.8 : baseWidth;
        ctx.strokeStyle = isHighlighted
          ? 'rgba(16, 185, 129, 0.9)'
          : 'rgba(148, 163, 184, 0.35)';
        ctx.stroke();
      }
    });

    // Draw Nodes
    nodes.forEach((node) => {
      const isHovered = hoveredNode && hoveredNode.gene === node.gene;
      const isSelected = selectedNode && selectedNode.gene === node.gene;

      if (isHovered || isSelected || node.rank <= 3) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.r + 5, 0, Math.PI * 2);
        ctx.fillStyle = isHovered || isSelected
          ? 'rgba(16, 185, 129, 0.25)'
          : 'rgba(99, 102, 241, 0.15)';
        ctx.fill();
      }

      ctx.beginPath();
      ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);

      if (node.rank === 1) ctx.fillStyle = '#10b981';
      else if (node.rank <= 3) ctx.fillStyle = '#6366f1';
      else ctx.fillStyle = '#64748b';
      ctx.fill();

      ctx.lineWidth = isHovered || isSelected ? 2.5 : 1;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();

      ctx.font = node.rank <= 3 ? 'bold 10px monospace' : '9px monospace';
      ctx.fillStyle = '#ffffff';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(node.gene, node.x, node.y);
    });

    ctx.restore();
  };

  const handleMouseDown = (e) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      setOffset({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left - offset.x) / zoom;
    const mouseY = (e.clientY - rect.top - offset.y) / zoom;

    const hit = nodesRef.current.find((n) => {
      const dx = n.x - mouseX;
      const dy = n.y - mouseY;
      return Math.sqrt(dx * dx + dy * dy) <= n.r + 4;
    });

    setHoveredNode(hit || null);
  };

  const handleMouseUp = () => setIsDragging(false);

  const downloadHighResPng = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = 1200;
    exportCanvas.height = 800;
    const expCtx = exportCanvas.getContext('2d');

    expCtx.fillStyle = '#0f172a';
    expCtx.fillRect(0, 0, 1200, 800);

    expCtx.fillStyle = '#ffffff';
    expCtx.font = 'bold 22px monospace';
    expCtx.fillText(`GNN-PathDSP Topological Saliency: ${compoundName} × ${cellName}`, 40, 50);

    expCtx.save();
    expCtx.translate(100, 100);
    expCtx.scale(1.7, 1.7);

    const nodes = nodesRef.current;
    const nodeMap = new Map(nodes.map((n) => [n.gene, n]));

    edges.forEach((edge) => {
      const src = nodeMap.get(edge.source);
      const tgt = nodeMap.get(edge.target);
      if (src && tgt) {
        expCtx.beginPath();
        expCtx.moveTo(src.x, src.y);
        expCtx.lineTo(tgt.x, tgt.y);
        expCtx.lineWidth = Math.max(1, (edge.attention || 0) * 35);
        expCtx.strokeStyle = 'rgba(148, 163, 184, 0.4)';
        expCtx.stroke();
      }
    });

    nodes.forEach((node) => {
      expCtx.beginPath();
      expCtx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
      expCtx.fillStyle = node.rank === 1 ? '#10b981' : node.rank <= 3 ? '#6366f1' : '#64748b';
      expCtx.fill();
      expCtx.strokeStyle = '#ffffff';
      expCtx.lineWidth = 1;
      expCtx.stroke();

      expCtx.font = 'bold 9px monospace';
      expCtx.fillStyle = '#ffffff';
      expCtx.textAlign = 'center';
      expCtx.textBaseline = 'middle';
      expCtx.fillText(node.gene, node.x, node.y);
    });

    expCtx.restore();

    const link = document.createElement('a');
    link.download = `gnn_pathdsp_${compoundName}_${cellName}_graph.png`;
    link.href = exportCanvas.toDataURL('image/png');
    link.click();
  };

  return (
    <div className="w-full rounded-2xl bg-white dark:bg-slate-900/60 p-6 text-left border border-slate-200/60 dark:border-slate-800/60 shadow-xs flex flex-col">
      {/* Header Controls */}
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-100 dark:border-slate-800">
        <div>
          <h4 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            Topological PPI Network
          </h4>
          <p className="text-[11px] text-slate-400 font-sans mt-0.5">
            Top 15 driver genes with GATv2 attention edges
          </p>
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={() => setZoom((z) => Math.min(2.2, z + 0.15))}
            className="p-1.5 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400 hover:text-slate-600 transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setZoom((z) => Math.max(0.7, z - 0.15))}
            className="p-1.5 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400 hover:text-slate-600 transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => {
              setZoom(1);
              setOffset({ x: 0, y: 0 });
            }}
            className="p-1.5 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400 hover:text-slate-600 transition-colors"
            title="Reset"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
          <div className="h-3 w-px bg-slate-200 dark:border-slate-800 mx-1" />
          <button
            onClick={downloadHighResPng}
            className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-100 dark:bg-slate-800 text-[11px] font-mono text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white transition-colors"
            title="Download PNG"
          >
            <Download className="w-3 h-3" />
            <span>PNG</span>
          </button>
        </div>
      </div>

      {/* Canvas */}
      <div
        className="relative w-full h-[320px] rounded-xl bg-slate-950 overflow-hidden cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
      >
        <canvas ref={canvasRef} width={600} height={320} className="w-full h-full block" />

        {hoveredNode && (
          <div className="absolute bottom-3 left-3 p-3 rounded-xl bg-slate-900/90 border border-slate-800 text-xs font-mono shadow-md backdrop-blur-xs pointer-events-none text-left">
            <span className="font-bold text-emerald-400">
              {hoveredNode.gene} (#{hoveredNode.rank})
            </span>
            <div className="text-[11px] text-slate-400 mt-0.5">
              Saliency: {(hoveredNode.saliency || 0).toFixed(4)}
            </div>
            <p className="text-[10px] text-slate-300 font-sans mt-1 max-w-[200px]">
              {hoveredNode.role}
            </p>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="mt-3 flex items-center justify-between text-[10px] font-mono text-slate-400">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" /> Top 1
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-indigo-500 inline-block" /> Top 2-3
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-slate-500 inline-block" /> Subordinate
          </span>
        </div>
        <span>Edge width: GATv2 Attention Weight α</span>
      </div>
    </div>
  );
}
