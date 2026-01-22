import React from 'react';
import { EcgRecord } from '../types';

interface HistoryViewProps {
  records: EcgRecord[];
  onSelect: (record: EcgRecord) => void;
  onClose: () => void;
}

const HistoryView: React.FC<HistoryViewProps> = ({ records, onSelect, onClose }) => {
  return (
    <div className="fixed inset-0 bg-gray-900/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-4xl max-h-[80vh] flex flex-col overflow-hidden">
        <div className="p-6 border-b border-gray-100 flex justify-between items-center bg-gray-50">
          <div>
            <h2 className="text-xl font-bold text-gray-900">Patient History</h2>
            <p className="text-sm text-gray-500">Local Database Records ({records.length})</p>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-gray-200 rounded-full transition-colors">
            <svg className="w-6 h-6 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        <div className="overflow-y-auto flex-grow p-0">
          {records.length === 0 ? (
            <div className="p-12 text-center text-gray-400">
              No records found in local database.
            </div>
          ) : (
            <table className="w-full text-left border-collapse">
              <thead className="bg-gray-50 sticky top-0 z-10">
                <tr>
                  <th className="p-4 text-xs font-bold text-gray-500 uppercase tracking-wider">Date</th>
                  <th className="p-4 text-xs font-bold text-gray-500 uppercase tracking-wider">Diagnosis</th>
                  <th className="p-4 text-xs font-bold text-gray-500 uppercase tracking-wider">Rate</th>
                  <th className="p-4 text-xs font-bold text-gray-500 uppercase tracking-wider">Urgency</th>
                  <th className="p-4 text-xs font-bold text-gray-500 uppercase tracking-wider">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {records.map((rec) => (
                  <tr key={rec.id} className="hover:bg-blue-50 transition-colors group">
                    <td className="p-4 text-sm text-gray-600">
                      {new Date(rec.timestamp).toLocaleDateString()} <span className="text-gray-400 text-xs">{new Date(rec.timestamp).toLocaleTimeString()}</span>
                    </td>
                    <td className="p-4 font-medium text-gray-900">{rec.diagnosis}</td>
                    <td className="p-4 text-sm text-gray-600">{rec.heartRate}</td>
                    <td className="p-4">
                      <span className={`px-2 py-1 rounded text-[10px] font-bold uppercase ${
                        rec.urgency === 'Emergency' ? 'bg-red-100 text-red-700' :
                        rec.urgency === 'Urgent' ? 'bg-orange-100 text-orange-700' :
                        'bg-blue-100 text-blue-700'
                      }`}>
                        {rec.urgency}
                      </span>
                    </td>
                    <td className="p-4">
                      <button 
                        onClick={() => onSelect(rec)}
                        className="text-blue-600 hover:text-blue-800 text-sm font-medium hover:underline"
                      >
                        View Report
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
};

export default HistoryView;
