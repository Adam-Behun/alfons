// Voice Agent Window Component
// Shows AI bubble and conversation transcript in one window, indicating active call state
// Closes to standby when insurance agent hangs up

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Mic, Activity, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import ConversationTranscript from './ConversationTranscript';

export default function VoiceAgentWindow() {
  const [isActive, setIsActive] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [callSid, setCallSid] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const wsRef = useRef<WebSocket | null>(null);

  const checkActiveCall = async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/queue/status`);
      if (!response.ok) {
        throw new Error(`Failed to check queue status: ${response.statusText}`);
      }
      const data = await response.json();
      const nowActive = data.status?.active_tasks > 0;
      
      if (nowActive && !isActive && data.status?.call_sid) {
        setCallSid(data.status.call_sid); // Assume backend includes active callSid
        setIsActive(true);
        setIsSpeaking(true);
        setTimeout(() => setIsSpeaking(false), 2000);
      } else if (!nowActive && isActive) {
        setIsActive(false);
        setCallSid(null);
      }
      
      setIsLoading(false);
    } catch (error) {
      console.error('Error checking call status:', error);
      setIsActive(false);
      setCallSid(null);
      setIsLoading(false);
    }
  };

  // Setup WebSocket to detect call termination
  useEffect(() => {
    checkActiveCall();
    const interval = setInterval(checkActiveCall, 2000);

    if (callSid) {
      const apiUrl = new URL(process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000');
      const wsProtocol = apiUrl.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProtocol}//${apiUrl.host}/ws/transcript/${callSid}`;

      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log(`WebSocket connected for call status (callSid: ${callSid})`);
      };

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'stop') { // Assume backend sends 'stop' on call end
          setIsActive(false);
          setCallSid(null);
          wsRef.current?.close();
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsActive(false);
        setCallSid(null);
      };

      wsRef.current.onclose = () => {
        console.log('WebSocket closed');
      };
    }

    return () => {
      clearInterval(interval);
      wsRef.current?.close();
    };
  }, [callSid, isActive]);

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
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <div className="flex flex-col items-center space-y-4">
              <Loader2 className="w-6 h-6 animate-spin text-primary" />
              <p className="text-muted-foreground text-sm">Loading...</p>
            </div>
          </div>
        ) : (
          <ConversationTranscript callSid={callSid || ''} />
        )}
      </div>
    </div>
  );
}