import React, { useState, useRef } from 'react';
import { api } from '../services/api';
import { Upload, FileCheck, AlertCircle, Loader2, FileText, X } from 'lucide-react';

export const UploadPage: React.FC = () => {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const addFiles = (incomingFiles: File[]) => {
    setFiles(prev => {
      const existingKeys = new Set(prev.map(f => `${f.name}-${f.size}-${f.lastModified}`));
      const uniqueFiles = incomingFiles.filter(
        f => !existingKeys.has(`${f.name}-${f.size}-${f.lastModified}`)
      );
      return [...prev, ...uniqueFiles];
    });
    setError(null);
    setResult(null);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      addFiles(Array.from(e.target.files));
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFiles = Array.from(e.dataTransfer.files);
      const validFiles = droppedFiles.filter(f =>
        /\.(png|jpe?g|pdf)$/i.test(f.name)
      );

      if (validFiles.length === 0) {
        setError('Please drop supported files (.png, .jpg, .jpeg, .pdf).');
        return;
      }

      addFiles(validFiles);
    }
  };

  const handleRemoveFile = (indexToRemove: number) => {
    setFiles(prev => prev.filter((_, idx) => idx !== indexToRemove));
  };

  const handleClearAll = () => {
    setFiles([]);
    setError(null);
    setResult(null);
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (files.length === 0) return;

    setUploading(true);
    setError(null);
    setResult(null);

    const total = files.length;
    let current = 1;
    setUploadProgress(`Uploading ${current}/${total} files...`);

    const progressInterval = setInterval(() => {
      if (current < total) {
        current += 1;
        setUploadProgress(`Uploading ${current}/${total} files...`);
      }
    }, Math.max(300, Math.min(1000, Math.floor(3000 / total))));

    try {
      const res = await api.uploadBatch(files);
      setResult(res);
      setFiles([]);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Document upload and processing failed.');
    } finally {
      clearInterval(progressInterval);
      setUploading(false);
      setUploadProgress(null);
    }
  };

  const processedCount = result
    ? result.batch_count ?? (Array.isArray(result.documents) ? result.documents.length : 1)
    : 0;

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Upload Feedback Forms</h2>
        <p className="text-sm text-slate-500">
          Upload multiple scanned images (PNG/JPEG) or multi-page PDF documents for automated extraction & verification.
        </p>
      </div>

      <form onSubmit={handleUpload} className="bg-white p-6 rounded-lg shadow border border-slate-200 space-y-4">
        {/* Drag & Drop Area */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
            isDragging
              ? 'border-blue-500 bg-blue-50/60'
              : 'border-slate-300 hover:border-blue-500 bg-white'
          }`}
        >
          <Upload className={`w-10 h-10 mx-auto mb-3 transition-colors ${isDragging ? 'text-blue-600' : 'text-slate-400'}`} />
          <p className="text-sm font-medium text-slate-700">
            {isDragging ? 'Drop multiple files here...' : 'Drag & Drop scanned forms here, or click to browse'}
          </p>
          <p className="text-xs text-slate-400 mt-1">Supports PNG, JPG, JPEG, and multi-page PDF files</p>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".png,.jpg,.jpeg,.pdf"
            onChange={handleFileChange}
            onClick={e => e.stopPropagation()}
            className="mt-4 text-xs mx-auto text-slate-500 cursor-pointer"
          />
        </div>

        {/* Selected Files List with Counter Badge */}
        {files.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Selected Files
                </span>
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-blue-100 text-blue-800">
                  {files.length} {files.length === 1 ? 'file' : 'files'}
                </span>
              </div>
              {!uploading && (
                <button
                  type="button"
                  onClick={handleClearAll}
                  className="text-xs text-slate-400 hover:text-rose-600 transition-colors"
                >
                  Clear all
                </button>
              )}
            </div>

            <div className="max-h-56 overflow-y-auto space-y-1.5 pr-1 border border-slate-100 rounded-md p-1">
              {files.map((f, idx) => (
                <div
                  key={`${f.name}-${idx}`}
                  className="flex items-center justify-between p-2.5 bg-slate-50 hover:bg-slate-100/80 rounded border border-slate-200 text-sm transition-colors"
                >
                  <div className="flex items-center space-x-2.5 min-w-0 mr-2">
                    <FileText className="w-4 h-4 text-blue-600 flex-shrink-0" />
                    <span className="font-medium text-slate-800 truncate text-xs sm:text-sm" title={f.name}>
                      {f.name}
                    </span>
                  </div>
                  <div className="flex items-center space-x-3 flex-shrink-0">
                    <span className="text-xs text-slate-500 font-mono">
                      {formatFileSize(f.size)}
                    </span>
                    <button
                      type="button"
                      disabled={uploading}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleRemoveFile(idx);
                      }}
                      className="text-slate-400 hover:text-rose-600 p-1 rounded hover:bg-rose-50 disabled:opacity-30 disabled:pointer-events-none transition-colors"
                      title="Remove file"
                      aria-label={`Remove ${f.name}`}
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Upload Progress Indicator */}
        {uploading && (
          <div className="p-3 bg-blue-50 border border-blue-200 rounded text-blue-800 text-sm flex items-center justify-between animate-pulse">
            <div className="flex items-center">
              <Loader2 className="w-4 h-4 mr-2 animate-spin text-blue-600" />
              <span className="font-medium">{uploadProgress || `Uploading 1/${files.length} files...`}</span>
            </div>
            <span className="text-xs text-blue-600 font-mono">Dual-Engine OCR & CNN</span>
          </div>
        )}

        {/* Error message */}
        {error && (
          <div className="p-3 bg-rose-50 border border-rose-200 rounded text-rose-800 text-sm flex items-center">
            <AlertCircle className="w-4 h-4 mr-2 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Batch Upload Result */}
        {result && (
          <div className="p-4 bg-emerald-50 border border-emerald-200 rounded text-emerald-900 text-sm space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center font-bold">
                <FileCheck className="w-5 h-5 mr-2 text-emerald-600" />
                Batch Upload & Processing Completed!
              </div>
              <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-200 text-emerald-800">
                {processedCount} {processedCount === 1 ? 'file' : 'files'} processed
              </span>
            </div>

            {result.documents && Array.isArray(result.documents) && result.documents.length > 0 ? (
              <div className="mt-2 space-y-1.5 max-h-48 overflow-y-auto text-xs text-emerald-800 pr-1 divide-y divide-emerald-200/60">
                {result.documents.map((doc: any, i: number) => (
                  <div key={doc.document_id || i} className="pt-1.5 flex items-center justify-between">
                    <span className="font-medium truncate max-w-xs">{doc.filename}</span>
                    <span className="text-emerald-700 font-mono">
                      Doc #{doc.document_id} | Status: <span className="font-bold">{doc.status}</span> | Pages: {doc.total_pages}
                    </span>
                  </div>
                ))}
              </div>
            ) : result.document_id ? (
              <p className="text-xs text-emerald-800">
                Document ID: <span className="font-mono">{result.document_id}</span> | Status: <span className="font-bold">{result.status}</span> | Total Pages: {result.total_pages}
              </p>
            ) : null}
          </div>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          disabled={files.length === 0 || uploading}
          className="w-full py-2.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium rounded-md text-sm transition-colors flex justify-center items-center"
        >
          {uploading ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              {uploadProgress || `Uploading 1/${files.length} files...`}
            </>
          ) : (
            `Start Batch Processing (${files.length} ${files.length === 1 ? 'file' : 'files'})`
          )}
        </button>
      </form>
    </div>
  );
};
