import React, { useState } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, Layers, Image as ImageIcon, AlertTriangle } from 'lucide-react';

interface ImageViewerProps {
  /** URL of the original uploaded image (always shown by default) */
  originalSrc: string;
  /** URL of the aligned/warped image — may be absent */
  alignedSrc?: string;
  /** True only when alignment produced a real perspective warp (not a letterbox fallback) */
  isWarped?: boolean;
  selectedField?: string;
  // Legacy single-src prop — used when only one URL is provided
  imageSrc?: string;
}

export const ImageViewer: React.FC<ImageViewerProps> = ({
  originalSrc,
  alignedSrc,
  isWarped = false,
  imageSrc,
  selectedField,
}) => {
  const [zoom, setZoom] = useState(1.0);
  const [showAligned, setShowAligned] = useState(false);

  // Legacy fallback: if only imageSrc is provided (old usage), treat it as original
  const effectiveOriginal = originalSrc || imageSrc || '';
  const effectiveAligned = alignedSrc || '';

  const currentSrc = showAligned && isWarped && effectiveAligned
    ? effectiveAligned
    : effectiveOriginal;

  const handleZoomIn = () => setZoom((prev) => Math.min(prev + 0.25, 3.0));
  const handleZoomOut = () => setZoom((prev) => Math.max(prev - 0.25, 0.5));
  const handleReset = () => setZoom(1.0);

  const alignedTooltip = !effectiveAligned
    ? 'No aligned image produced for this page'
    : !isWarped
    ? 'Alignment fell back to letterboxing — no valid perspective warp was found. Original is displayed.'
    : 'Toggle between original and perspective-corrected aligned image';

  return (
    <div className="bg-slate-900 rounded-lg p-4 text-white flex flex-col h-full">
      {/* Header */}
      <div className="flex justify-between items-center mb-3">
        <div className="flex items-center space-x-2">
          <h3 className="text-sm font-semibold">
            {showAligned && isWarped ? 'Aligned Form (Perspective-Corrected)' : 'Original Form Image'}
          </h3>
          {!isWarped && effectiveAligned && (
            <span
              className="flex items-center text-[10px] bg-amber-700/50 text-amber-200 px-1.5 py-0.5 rounded"
              title={alignedTooltip}
            >
              <AlertTriangle className="w-3 h-3 mr-1" />
              No valid warp
            </span>
          )}
        </div>

        <div className="flex items-center space-x-2">
          {/* Original / Aligned toggle */}
          <div className="flex rounded overflow-hidden border border-slate-700 text-xs">
            <button
              onClick={() => setShowAligned(false)}
              className={`px-2 py-1 flex items-center space-x-1 ${
                !showAligned ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
              title="Show original uploaded image"
            >
              <ImageIcon className="w-3 h-3" />
              <span>Original</span>
            </button>
            <button
              onClick={() => isWarped && setShowAligned(true)}
              disabled={!isWarped || !effectiveAligned}
              className={`px-2 py-1 flex items-center space-x-1 ${
                showAligned && isWarped
                  ? 'bg-emerald-600 text-white'
                  : isWarped && effectiveAligned
                  ? 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                  : 'bg-slate-800 text-slate-600 cursor-not-allowed'
              }`}
              title={alignedTooltip}
            >
              <Layers className="w-3 h-3" />
              <span>Aligned</span>
            </button>
          </div>

          {/* Zoom controls */}
          <button
            onClick={handleZoomOut}
            className="p-1.5 bg-slate-800 hover:bg-slate-700 rounded text-slate-200"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <span className="text-xs self-center font-mono">{Math.round(zoom * 100)}%</span>
          <button
            onClick={handleZoomIn}
            className="p-1.5 bg-slate-800 hover:bg-slate-700 rounded text-slate-200"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={handleReset}
            className="p-1.5 bg-slate-800 hover:bg-slate-700 rounded text-slate-200"
            title="Reset Zoom"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Image area */}
      <div className="flex-1 overflow-auto bg-black rounded flex items-center justify-center relative min-h-[400px]">
        {currentSrc ? (
          <div
            style={{ transform: `scale(${zoom})`, transformOrigin: 'top center' }}
            className="transition-transform"
          >
            <img
              src={currentSrc}
              alt={showAligned && isWarped ? 'Aligned Form' : 'Original Form'}
              className="max-w-full h-auto object-contain rounded"
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = 'none';
              }}
            />
          </div>
        ) : (
          <p className="text-slate-500 text-sm">Select a page to view the form image</p>
        )}
      </div>
    </div>
  );
};
