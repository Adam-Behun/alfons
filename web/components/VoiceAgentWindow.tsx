// Voice Agent Window Component
// Shows AI bubble and conversation transcript in one window

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Mic, Activity, MessageSquare, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

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

export default function VoiceAgentWindow() {
  const [isActive, setIsActive] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [logs, setLogs] = useState<ConversationLog[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const checkActiveCall = async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/logs`);
      if (!response.ok) return;
      
      const data = await response.json();
      setLogs(data.slice(-10).reverse());
      
      const thirtySecondsAgo = Date.now() - 30000;
      const recentActivity = data.find((log: ConversationLog) => 
        new Date(log.timestamp).getTime() > thirtySecondsAgo
      );
      
      const wasActive = isActive;
      const nowActive = !!recentActivity;
      
      setIsActive(nowActive);
      
      if (!wasActive && nowActive) {
        setIsSpeaking(true);
        setTimeout(() => setIsSpeaking(false), 2000);
      }
      
      setIsLoading(false);
    } catch (error) {
      console.error('Error checking call status:', error);
      setIsActive(false);
      setIsLoading(false);
    }
  };

  useEffect(() => {
    checkActiveCall();
    const interval = setInterval(checkActiveCall, 2000);
    return () => clearInterval(interval);
  }, [isActive]);

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-6 border-b bg-gradient-to-r from-emerald-600 to-green-700 text-white">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-white/20 rounded-lg flex items-center justify-center">
            <Mic className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold">Alfons Voice Agent</h2>
            <p className="text-emerald-100 text-sm">AI-powered prior authorization assistant</p>
          </div>
        </div>
      </div>

      {/* Voice Agent Status */}
      <div className="p-8 border-b bg-gradient-to-br from-slate-50 to-blue-50">
        <div className="flex flex-col items-center space-y-6">
          {/* Voice Bubble */}
          <div className="relative">
            <div className={cn(
              "w-20 h-20 rounded-full flex items-center justify-center transition-all duration-500 shadow-lg",
              isActive 
                ? "bg-gradient-to-br from-emerald-500 to-green-600 shadow-emerald-200" 
                : "bg-gradient-to-br from-slate-400 to-slate-500 shadow-slate-200",
              isSpeaking && "animate-gentle-pulse"
            )}>
              {/* Ripple Effects */}
              {isSpeaking && (
                <>
                  <div className="absolute inset-0 rounded-full border-2 border-emerald-400/30 animate-pulse-ring" />
                  <div className="absolute inset-0 rounded-full border-2 border-emerald-400/20 animate-pulse-ring" style={{ animationDelay: '0.3s' }} />
                </>
              )}
              
              <Mic className="w-8 h-8 text-white" />
            </div>
          </div>

          {/* Status */}
          <div className="text-center">
            <div className={cn(
              "text-lg font-semibold transition-colors",
              isActive ? "text-emerald-600" : "text-slate-600"
            )}>
              {isActive ? 'ACTIVE CALL' : 'STANDBY'}
            </div>
            <div className="text-sm text-muted-foreground mt-1">
              {isActive ? 'Communicating with insurance' : 'Waiting for authorization request'}
            </div>
          </div>
        </div>
      </div>

      {/* Conversation Section */}
      <div className="flex-1 flex flex-col min-h-0">
        <div className="p-4 border-b bg-white">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <MessageSquare className="w-5 h-5 text-muted-foreground" />
              <h3 className="font-semibold">Live Conversation</h3>
            </div>
            <Badge variant="secondary">{logs.length} exchanges</Badge>
          </div>
        </div>

        <div className="flex-1 overflow-auto p-4">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <div className="flex flex-col items-center space-y-4">
                <Loader2 className="w-6 h-6 animate-spin text-primary" />
                <p className="text-muted-foreground text-sm">Loading conversation...</p>
              </div>
            </div>
          ) : logs.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <Card className="max-w-sm">
                <CardContent className="pt-6 text-center">
                  <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                    <Activity className="w-6 h-6 text-blue-600" />
                  </div>
                  <h3 className="font-semibold mb-2">No Active Conversation</h3>
                  <p className="text-muted-foreground text-sm">
                    Start a prior authorization call to see the transcript
                  </p>
                </CardContent>
              </Card>
            </div>
          ) : (
            <div className="space-y-4">
              {logs.map((log, index) => (
                <div key={log.id || index} className="space-y-3">
                  {/* Insurance Agent Message */}
                  {log.user_input && (
                    <div className="flex gap-3">
                      <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
                        <div className="w-4 h-4 bg-blue-600 rounded-full" />
                      </div>
                      <div className="flex-1 max-w-[85%]">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-sm font-medium text-blue-600">Insurance Agent</span>
                          <span className="text-xs text-muted-foreground">{formatTimestamp(log.timestamp)}</span>
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
                    <div className="flex gap-3 justify-end">
                      <div className="flex-1 max-w-[85%] flex flex-col items-end">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-xs text-muted-foreground">{formatTimestamp(log.timestamp)}</span>
                          <span className="text-sm font-medium text-emerald-600">Provider Agent (Alfons)</span>
                        </div>
                        <Card className="bg-emerald-50 border-emerald-200">
                          <CardContent className="p-3">
                            <p className="text-sm text-slate-800">{log.bot_response}</p>
                          </CardContent>
                        </Card>
                        
                        {/* Extracted Data */}
                        {(log.patient_id || log.procedure_code || log.insurance || log.escalated) && (
                          <div className="flex flex-wrap gap-1 mt-2">
                            {log.patient_id && (
                              <Badge variant="secondary" className="text-xs bg-purple-100 text-purple-700">
                                Patient: {log.patient_id}
                              </Badge>
                            )}
                            {log.procedure_code && (
                              <Badge variant="secondary" className="text-xs bg-orange-100 text-orange-700">
                                CPT: {log.procedure_code}
                              </Badge>
                            )}
                            {log.insurance && (
                              <Badge variant="secondary" className="text-xs bg-cyan-100 text-cyan-700">
                                Insurance: {log.insurance}
                              </Badge>
                            )}
                            {log.escalated && (
                              <Badge variant="destructive" className="text-xs">
                                Escalated
                              </Badge>
                            )}
                          </div>
                        )}
                      </div>
                      <div className="w-8 h-8 bg-emerald-100 rounded-full flex items-center justify-center flex-shrink-0">
                        <div className="w-4 h-4 bg-emerald-600 rounded-full" />
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}