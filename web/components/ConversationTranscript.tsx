// Conversation Transcript Component
// Shows real-time conversation between Insurance Agent and Provider Agent (Alfons)

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { MessageSquare, User, Bot, Loader2, AlertTriangle } from 'lucide-react';
import { cn, formatTimestamp } from '@/lib/utils';

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

  if (isLoading) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MessageSquare className="w-5 h-5" />
            Live Conversation
          </CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-64">
          <div className="flex flex-col items-center space-y-4">
            <Loader2 className="w-8 h-8 animate-spin text-primary" />
            <p className="text-muted-foreground">Loading conversation...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          <MessageSquare className="w-5 h-5 text-primary" />
          Live Conversation
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          Real-time transcript of prior authorization call
        </p>
      </CardHeader>

      <CardContent className="flex-1 overflow-auto">
        {logs.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center space-y-4">
              <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mx-auto">
                <MessageSquare className="w-8 h-8 text-muted-foreground" />
              </div>
              <div>
                <h3 className="font-semibold text-lg mb-2">No Active Conversation</h3>
                <p className="text-sm text-muted-foreground">
                  Start a prior authorization call to see the transcript
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            {logs.map((log, index) => (
              <div key={log.id || index} className="space-y-4">
                {/* Insurance Agent Message */}
                {log.user_input && (
                  <div className="flex gap-3">
                    <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
                      <User className="w-4 h-4 text-blue-600" />
                    </div>
                    <div className="flex-1 space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-blue-600">Insurance Agent</span>
                        <span className="text-xs text-muted-foreground">
                          {formatTimestamp(log.timestamp)}
                        </span>
                      </div>
                      <Card className="bg-blue-50 border-blue-200">
                        <CardContent className="p-3">
                          <p className="text-sm text-slate-800">{log.user_input}</p>
                        </CardContent>
                      </Card>
                    </div>
                  </div>
                )}

                {/* Provider Agent (Alfons) Message */}
                {log.bot_response && (
                  <div className="flex gap-3">
                    <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0">
                      <Bot className="w-4 h-4 text-green-600" />
                    </div>
                    <div className="flex-1 space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-green-600">
                          Provider Agent (Alfons)
                        </span>
                        <span className="text-xs text-muted-foreground">
                          {formatTimestamp(log.timestamp)}
                        </span>
                      </div>
                      <Card className="bg-green-50 border-green-200">
                        <CardContent className="p-3">
                          <p className="text-sm text-slate-800">{log.bot_response}</p>
                        </CardContent>
                      </Card>
                      
                      {/* Show extracted data if available */}
                      {(log.patient_id || log.procedure_code || log.insurance || log.escalated) && (
                        <div className="flex flex-wrap gap-2 pt-2">
                          {log.patient_id && (
                            <Badge variant="secondary" className="text-xs bg-purple-100 text-purple-800">
                              Patient: {log.patient_id}
                            </Badge>
                          )}
                          {log.procedure_code && (
                            <Badge variant="secondary" className="text-xs bg-orange-100 text-orange-800">
                              CPT: {log.procedure_code}
                            </Badge>
                          )}
                          {log.insurance && (
                            <Badge variant="secondary" className="text-xs bg-cyan-100 text-cyan-800">
                              Insurance: {log.insurance}
                            </Badge>
                          )}
                          {log.escalated && (
                            <Badge variant="destructive" className="text-xs gap-1">
                              <AlertTriangle className="w-3 h-3" />
                              Escalated
                            </Badge>
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
      </CardContent>
    </Card>
  );
}