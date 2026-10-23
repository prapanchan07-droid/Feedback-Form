import React from 'react';
import { PageResult } from '../types';
import { CheckCircle2, AlertTriangle, HelpCircle, Edit3 } from 'lucide-react';

interface ComparisonTableProps {
  pageResult: PageResult;
  onVerifyField: (fieldName: string, currentValue: string) => void;
}

/**
 * Renders a comparison table between Vision VLM (Qwen) extraction and the
 * fallback CNN recognizer. The component now gracefully handles both the
 * `/pages/{id}/results` payload (which includes a `verification_decisions`
 * array with explicit field keys) and the older `/pages/{id}/comparison`
 * payload (which provides a `comparisons` array with shorter keys).
 */
export const ComparisonTable: React.FC<ComparisonTableProps> = ({ pageResult, onVerifyField }) => {
  const fields = pageResult.fields || {};
  // Support both payload shapes
  const decisions: any[] = (pageResult.verification_decisions || pageResult.comparisons) || [];

  const getDecisionBadge = (decisionStr?: string) => {
    switch (decisionStr) {
      case 'VERIFIED':
      case 'AGREEMENT':
        return <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-emerald-100 text-emerald-800"><CheckCircle2 className="w-3 h-3 mr-1"/> Verified</span>;
      case 'NEEDS_REVIEW':
      case 'MANUAL_REVIEW':
      case 'CONFLICT':
        return <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-amber-100 text-amber-800"><AlertTriangle className="w-3 h-3 mr-1"/> Needs Review</span>;
      case 'EXTRACTED':
      case 'CNN_UNAVAILABLE':
        return <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800">Extracted</span>;
      case 'HUMAN_VERIFIED':
        return <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-800">Human Verified</span>;
      default:
        return <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-slate-100 text-slate-800">Extracted</span>;
    }
  };

  const getConfidenceColor = (conf: number) => {
    if (conf >= 0.9) return 'text-emerald-600';
    if (conf >= 0.7) return 'text-amber-600';
    return 'text-rose-600';
  };

  return (
    <div className="bg-white shadow rounded-lg overflow-hidden border border-slate-200">
      <div className="px-4 py-3 bg-slate-50 border-b border-slate-200 flex justify-between items-center">
        <h3 className="text-sm font-bold text-slate-800">Qwen Vision Extraction Results</h3>
        <span className="text-xs text-slate-500 font-mono">Page ID: {pageResult.page_id}</span>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-slate-200 text-left text-sm">
          <thead className="bg-slate-100 text-slate-600 font-semibold uppercase text-xs">
            <tr>
              <th className="px-4 py-2.5">Field Name</th>
              <th className="px-4 py-2.5">Vision VLM</th>
              <th className="px-4 py-2.5">Pass 2 Verify</th>
              <th className="px-4 py-2.5">Final Value</th>
              <th className="px-4 py-2.5">Confidence</th>
              <th className="px-4 py-2.5">Status</th>
              <th className="px-4 py-2.5 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200 bg-white">
            {Object.keys(fields).map((fieldName) => {
              const fData = fields[fieldName];
              const dec = decisions.find((d) => d.field_name === fieldName);

              // Normalise payload shape – older endpoint used short keys
              const geminiVal = dec?.gemini ?? dec?.gemini_value ?? (fData.raw_value || '—');
              const cnnVal = dec?.cnn ?? dec?.cnn_value ?? 'N/A';
              const finalVal = dec?.final ?? dec?.final_value ?? fData.normalized_value ?? '—';
              const conf = typeof dec?.confidence === 'number' ? dec.confidence : fData.confidence ?? 0;
              const decisionStr = dec?.decision ?? dec?.status ?? fData.status;

              return (
                <tr key={fieldName} className="hover:bg-slate-50">
                  <td className="px-4 py-3 font-medium text-slate-900 capitalize">
                    {fieldName.replace(/_/g, ' ')}
                  </td>
                  <td className="px-4 py-3 text-slate-700 font-mono text-xs">{geminiVal || '—'}</td>
                  <td className="px-4 py-3 text-slate-700 font-mono text-xs">{cnnVal || '—'}</td>
                  <td className="px-4 py-3 font-semibold text-slate-900 font-mono text-xs">{finalVal || '—'}</td>
                  <td className={`px-4 py-3 font-mono text-xs font-semibold ${getConfidenceColor(conf)}`}>
                    {(conf * 100).toFixed(0)}%
                  </td>
                  <td className="px-4 py-3">{getDecisionBadge(decisionStr)}</td>
                  <td className="px-4 py-3 text-right">
                    <button
                      onClick={() => onVerifyField(fieldName, finalVal || '')}
                      className="inline-flex items-center text-xs font-medium text-blue-600 hover:text-blue-800"
                    >
                      <Edit3 className="w-3.5 h-3.5 mr-1" />
                      Verify
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
