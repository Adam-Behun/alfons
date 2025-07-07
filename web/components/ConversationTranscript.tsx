// Conversation Transcript Component
// Shows real-time conversation between Insurance Agent and Provider Agent (Alfons)

import { useState, useEffect } from 'react';

interface ConversationLog {
  id: number;
  call_sid: string;
  user_input: string;
  bot_response: string;
  patient_id?: string;
  procedure_code?: string;
  insurance?: string;
  escalated: boolean;
  timestamp: string;
}

export default function ConversationTranscript() {
  const [logs, setLogs] = useState<ConversationLog[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const fetchLogs = async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/logs`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const data = await response.json();
      setLogs(data.slice(-20).reverse()); // Show last 20 interactions, newest first
      setIsLoading(false);
    } catch (error) {
      console.error('Error fetching logs:', error);
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
    const interval = setInterval(fetchLogs, 3000); // Update every 3 seconds
    return () => clearInterval(interval);
  }, []);

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b">
        <h3 className="text-lg font-semibold text-gray-900">Live Conversation</h3>
        <p className="text-sm text-gray-600">Real-time transcript of prior authorization call</p>
      </div>

      {/* Transcript Content */}
      <div className="flex-1 overflow-auto p-4">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-gray-500">Loading conversation...</div>
          </div>
        ) : logs.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-500">
              <div className="text-lg mb-2">No active conversation</div>
              <div className="text-sm">Start a prior authorization call to see the transcript</div>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {logs.map((log, index) => (
              <div key={log.id || index} className="space-y-3">
                {/* Insurance Agent Message */}
                {log.user_input && (
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
                      <svg className="w-4 h-4 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm font-medium text-blue-600">Insurance Agent</span>
                        <span className="text-xs text-gray-500">{formatTimestamp(log.timestamp)}</span>
                      </div>
                      <div className="bg-blue-50 rounded-lg p-3 text-sm text-gray-800">
                        {log.user_input}
                      </div>
                    </div>
                  </div>
                )}

                {/* Provider Agent (Alfons) Message */}
                {log.bot_response && (
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0">
                      <svg className="w-4 h-4 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M2 3a1 1 0 011-1h2.153a1 1 0 01.986.836l.74 4.435a1 1 0 01-.54 1.06l-1.548.773a11.037 11.037 0 006.105 6.105l.774-1.548a1 1 0 011.059-.54l4.435.74a1 1 0 01.836.986V17a1 1 0 01-1 1h-2C7.82 18 2 12.18 2 5V3z" />
                      </svg>
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm font-medium text-green-600">Provider Agent (Alfons)</span>
                        <span className="text-xs text-gray-500">{formatTimestamp(log.timestamp)}</span>
                      </div>
                      <div className="bg-green-50 rounded-lg p-3 text-sm text-gray-800">
                        {log.bot_response}
                      </div>
                      
                      {/* Show extracted data if available */}
                      {(log.patient_id || log.procedure_code || log.insurance) && (
                        <div className="mt-2 flex flex-wrap gap-2">
                          {log.patient_id && (
                            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-purple-100 text-purple-800">
                              Patient: {log.patient_id}
                            </span>
                          )}
                          {log.procedure_code && (
                            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-orange-100 text-orange-800">
                              CPT: {log.procedure_code}
                            </span>
                          )}
                          {log.insurance && (
                            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-cyan-100 text-cyan-800">
                              Insurance: {log.insurance}
                            </span>
                          )}
                          {log.escalated && (
                            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-red-100 text-red-800">
                              Escalated
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}