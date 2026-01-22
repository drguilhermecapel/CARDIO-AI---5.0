import React, { useState } from 'react';

const Disclaimer: React.FC = () => {
  const [isVisible, setIsVisible] = useState(true);

  if (!isVisible) return null;

  return (
    <div className="bg-blue-50 border-b border-blue-100 p-4">
      <div className="max-w-7xl mx-auto flex gap-3 items-start sm:items-center justify-between">
        <div className="flex gap-3 items-start sm:items-center">
          <div className="flex-shrink-0 text-blue-600">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <p className="text-sm text-blue-800">
            <strong>Investigational Device:</strong> Cardio.AI is an assistive tool for educational and supportive purposes only. 
            It is not a diagnostic device. Always verify findings with a qualified cardiologist.
          </p>
        </div>
        <button 
          onClick={() => setIsVisible(false)}
          className="text-blue-600 hover:text-blue-800 font-medium text-sm"
        >
          Dismiss
        </button>
      </div>
    </div>
  );
};

export default Disclaimer;
