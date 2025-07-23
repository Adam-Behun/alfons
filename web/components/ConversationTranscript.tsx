// Conversation Transcript Component
// Shows real-time streaming conversation transcript between Insurance Agent and Provider Agent (Alfons)
// Implements streaming via WebSocket for word-by-word updates, with full transcript and thoughts stored in MongoDB post-message

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { MessageSquare, User, Bot, Loader2, AlertTriangle } from 'lucide-react';
import { cn, formatTimestamp } from '@/lib/utils';

interface ConversationMessage {
  role: 'user' | 'assistant' | 'assistant_thought';
  content: string;
  timestamp: string;
  isComplete: boolean;
  extracted?: {
    patient_id?: string;
    procedure_code?: string;
    insurance?: string;
    escalated?: boolean;
  };
}

interface ConversationLog {
  id: string;
  call_sid: string;
  user_input: string;
  bot_thoughts: string;
  bot_response: string;
  patient_id?: string;
  procedure_code?: string;
  insurance?: string;
  escalated: boolean;
  timestamp: string;
}

interface ConversationTranscriptProps {
  callSid: string;
}

export default function ConversationTranscript({ callSid }: ConversationTranscriptProps) {
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentUserChunk, setCurrentUserChunk] = useState('');
  const [currentBotChunk, setCurrentBotChunk] = useState('');
  const [currentThoughtChunk, setCurrentThoughtChunk] = useState('');
  const wsRef = useRef<WebSocket | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement>(null);

  // Fetch initial history from /logs
  const fetchInitialLogs = async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/logs`);
      if (!response.ok) {
        throw new Error(`Failed to fetch logs: ${response.statusText}`);
      }
      const data: ConversationLog[] = await response.json();
      const initialMessages: ConversationMessage[] = data
        .filter(log => log.call_sid === callSid)
        .reduce<ConversationMessage[]>((acc, log) => {
          const messages: ConversationMessage[] = [];
          if (log.user_input) {
            messages.push({
              role: 'user',
              content: log.user_input,
              timestamp: log.timestamp,
              isComplete: true,
              extracted: undefined,
            });
          }
          if (log.bot_thoughts) {
            messages.push({
              role: 'assistant_thought',
              content: log.bot_thoughts,
              timestamp: log.timestamp,
              isComplete: true,
              extracted: undefined,
            });
          }
          if (log.bot_response) {
            messages.push({
              role: 'assistant',
              content: log.bot_response,
              timestamp: log.timestamp,
              isComplete: true,
              extracted: {
                patient_id: log.patient_id,
                procedure_code: log.procedure_code,
                insurance: log.insurance,
                escalated: log.escalated,
              },
            });
          }
          return [...acc, ...messages];
        }, []);
      setMessages(initialMessages);
      setError(null);
    } catch (error) {
      console.error('Error fetching initial logs:', error);
      setError('Failed to load conversation history. Please check the API connection.');
    } finally {
      setIsLoading(false);
    }
  };

  // Scroll to bottom on new messages
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, currentUserChunk, currentBotChunk, currentThoughtChunk]);

  // Setup WebSocket for real-time streaming
  useEffect(() => {
    fetchInitialLogs();

    const apiUrl = new URL(process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000');
    const wsProtocol = apiUrl.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${apiUrl.host}/ws/transcript/${callSid}`;

    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onopen = () => {
      console.log(`WebSocket connected for transcript streaming (callSid: ${callSid})`);
    };

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'chunk') {
        if (data.role === 'user') {
          setCurrentUserChunk(prev => prev + data.chunk);
        } else if (data.role === 'assistant_thought') {
          setCurrentThoughtChunk(prev => prev + data.chunk);
        } else if (data.role === 'assistant') {
          setCurrentBotChunk(prev => prev + data.chunk);
        }
      } else if (data.type === 'complete') {
        if (data.role === 'user' && currentUserChunk) {
          setMessages(prev => [...prev, {
            role: 'user',
            content: currentUserChunk,
            timestamp: data.timestamp || new Date().toISOString(),
            isComplete: true,
          }]);
          setCurrentUserChunk('');
        } else if (data.role === 'assistant' && (currentThoughtChunk || currentBotChunk)) {
          if (currentThoughtChunk) {
            setMessages(prev => [...prev, {
              role: 'assistant_thought',
              content: currentThoughtChunk,
              timestamp: data.timestamp || new Date().toISOString(),
              isComplete: true,
            }]);
            setCurrentThoughtChunk('');
          }
          if (currentBotChunk) {
            setMessages(prev => [...prev, {
              role: 'assistant',
              content: currentBotChunk,
              timestamp: data.timestamp || new Date().toISOString(),
              isComplete: true,
              extracted: data.extracted,
            }]);
            setCurrentBotChunk('');
          }
        }
        // Backend stores full message in MongoDB here (post-complete)
      }
    };

    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('WebSocket connection failed. Falling back to polling if implemented.');
    };

    wsRef.current.onclose = () => {
      console.log('WebSocket closed');
    };

    return () => {
      wsRef.current?.close();
    };
  }, [callSid]);

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

  if (error) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MessageSquare className="w-5 h-5" />
            Live Conversation
          </CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-64">
          <div className="text-center space-y-4">
            <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center mx-auto">
              <AlertTriangle className="w-6 h-6 text-red-600" />
            </div>
            <h3 className="font-semibold text-lg mb-2">Error Loading Conversation</h3>
            <p className="text-sm text-muted-foreground">{error}</p>
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
          Real-time streaming transcript of prior authorization call
        </p>
      </CardHeader>

      <CardContent className="flex-1 overflow-auto">
        {messages.length === 0 && !currentUserChunk && !currentBotChunk && !currentThoughtChunk ? (
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
            {messages.map((msg, index) => (
              <div key={index} className="space-y-4">
                <div className="flex gap-3">
                  <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
                    style={{ backgroundColor: msg.role === 'user' ? 'rgb(219 234 254)' : msg.role === 'assistant_thought' ? 'rgb(229 231 235)' : 'rgb(220 252 231)' }}>
                    {msg.role === 'user' ? <User className="w-4 h-4 text-blue-600" /> : <Bot className="w-4 h-4 text-gray-600" />}
                  </div>
                  <div className="flex-1 space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium" style={{ color: msg.role === 'user' ? 'rgb(37 99 235)' : msg.role === 'assistant_thought' ? 'rgb(75 85 99)' : 'rgb(22 163 74)' }}>
                        {msg.role === 'user' ? 'Insurance Agent' : msg.role === 'assistant_thought' ? 'Alfons Thinking' : 'Provider Agent (Alfons)'}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {formatTimestamp(msg.timestamp)}
                      </span>
                    </div>
                    <Card style={{ backgroundColor: msg.role === 'user' ? 'rgb(239 246 255)' : msg.role === 'assistant_thought' ? 'rgb(243 244 246)' : 'rgb(240 253 244)', borderColor: msg.role === 'user' ? 'rgb(191 219 254)' : msg.role === 'assistant_thought' ? 'rgb(209 213 219)' : 'rgb(187 247 208)' }}>
                      <CardContent className="p-3">
                        <p className={`text-sm text-slate-800 ${msg.role === 'assistant_thought' ? 'italic' : ''}`}>
                          {msg.content}
                        </p>
                      </CardContent>
                    </Card>
                    {msg.role === 'assistant' && msg.extracted && (
                      <div className="flex flex-wrap gap-2 pt-2">
                        {msg.extracted.patient_id && (
                          <Badge variant="secondary" className="text-xs bg-purple-100 text-purple-800">
                            Patient: {msg.extracted.patient_id}
                          </Badge>
                        )}
                        {msg.extracted.procedure_code && (
                          <Badge variant="secondary" className="text-xs bg-orange-100 text-orange-800">
                            CPT: {msg.extracted.procedure_code}
                          </Badge>
                        )}
                        {msg.extracted.insurance && (
                          <Badge variant="secondary" className="text-xs bg-cyan-100 text-cyan-800">
                            Insurance: {msg.extracted.insurance}
                          </Badge>
                        )}
                        {msg.extracted.escalated && (
                          <Badge variant="destructive" className="text-xs gap-1">
                            <AlertTriangle className="w-3 h-3" />
                            Escalated
                          </Badge>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
            {/* Streaming thought chunk */}
            {currentThoughtChunk && (
              <div className="flex gap-3">
                <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center flex-shrink-0">
                  <Bot className="w-4 h-4 text-gray-600" />
                </div>
                <div className="flex-1 space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-gray-600">Alfons Thinking</span>
                    <span className="text-xs text-muted-foreground">
                      {formatTimestamp(new Date().toISOString())}
                    </span>
                  </div>
                  <Card className="bg-gray-50 border-gray-200">
                    <CardContent className="p-3">
                      <p className="text-sm italic text-slate-600">{currentThoughtChunk}</p>
                    </CardContent>
                  </Card>
                </div>
              </div>
            )}
            {/* Streaming user chunk */}
            {currentUserChunk && (
              <div className="flex gap-3">
                <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
                  <User className="w-4 h-4 text-blue-600" />
                </div>
                <div className="flex-1 space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-blue-600">Insurance Agent</span>
                    <span className="text-xs text-muted-foreground">
                      {formatTimestamp(new Date().toISOString())}
                    </span>
                  </div>
                  <Card className="bg-blue-50 border-blue-200">
                    <CardContent className="p-3">
                      <p className="text-sm text-slate-800">{currentUserChunk}</p>
                    </CardContent>
                  </Card>
                </div>
              </div>
            )}
            {/* Streaming bot chunk */}
            {currentBotChunk && (
              <div className="flex gap-3">
                <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0">
                  <Bot className="w-4 h-4 text-green-600" />
                </div>
                <div className="flex-1 space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-green-600">Provider Agent (Alfons)</span>
                    <span className="text-xs text-muted-foreground">
                      {formatTimestamp(new Date().toISOString())}
                    </span>
                  </div>
                  <Card className="bg-green-50 border-green-200">
                    <CardContent className="p-3">
                      <p className="text-sm text-slate-800">{currentBotChunk}</p>
                    </CardContent>
                  </Card>
                </div>
              </div>
            )}
            <div ref={transcriptEndRef} />
          </div>
        )}
      </CardContent>
    </Card>
  );
}