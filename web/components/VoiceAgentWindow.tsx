// Voice Agent Window Component
// Shows AI bubble and conversation transcript in one window

import { useState, useEffect } from 'react';
import styles from './VoiceAgentWindow.module.css';

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
      setLogs(data.slice(-10).reverse()); // Show last 10 interactions, newest first
      
      // Check if there's activity in the last 30 seconds
      const thirtySecondsAgo = Date.now() - 30000;
      const recentActivity = data.find((log: ConversationLog) => 
        new Date(log.timestamp).getTime() > thirtySecondsAgo
      );
      
      const wasActive = isActive;
      const nowActive = !!recentActivity;
      
      setIsActive(nowActive);
      
      // Trigger speaking animation when new activity is detected
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
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <h2 className={styles.title}>Alfons Voice Agent</h2>
        <p className={styles.subtitle}>AI-powered prior authorization assistant</p>
      </div>

      {/* Voice Agent Section */}
      <div className={styles.agentSection}>
        <div className={styles.agentContainer}>
          {/* Speaking Bubble */}
          <div className={`${styles.bubble} ${isActive ? styles.active : ''} ${isSpeaking ? styles.speaking : ''}`}>
            {/* Voice Wave Animation */}
            {isSpeaking && (
              <>
                <div className={`${styles.ripple} ${styles.ripple1}`} />
                <div className={`${styles.ripple} ${styles.ripple2}`} />
              </>
            )}
            
            {/* Microphone Icon */}
            <svg 
              width="32" 
              height="32" 
              viewBox="0 0 24 24" 
              fill="none" 
              stroke="white" 
              strokeWidth="2"
            >
              <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
              <line x1="12" y1="19" x2="12" y2="23"/>
              <line x1="8" y1="23" x2="16" y2="23"/>
            </svg>
          </div>

          {/* Status Text */}
          <div className={styles.status}>
            <div className={`${styles.statusText} ${isActive ? styles.activeStatus : ''}`}>
              {isActive ? 'ACTIVE CALL' : 'STANDBY'}
            </div>
            <div className={styles.statusSubtext}>
              {isActive ? 'Communicating with insurance' : 'Waiting for authorization request'}
            </div>
          </div>
        </div>
      </div>

      {/* Conversation Section */}
      <div className={styles.conversationSection}>
        <div className={styles.conversationHeader}>
          <h3 className={styles.conversationTitle}>Live Conversation</h3>
          <div className={styles.conversationCount}>{logs.length} exchanges</div>
        </div>

        <div className={styles.conversationContent}>
          {isLoading ? (
            <div className={styles.loading}>Loading conversation...</div>
          ) : logs.length === 0 ? (
            <div className={styles.emptyState}>
              <div className={styles.emptyStateText}>No active conversation</div>
              <div className={styles.emptyStateSubtext}>Start a prior authorization call to see the transcript</div>
            </div>
          ) : (
            <div className={styles.messageList}>
              {logs.map((log, index) => (
                <div key={log.id || index} className={styles.messageGroup}>
                  {/* Insurance Agent Message */}
                  {log.user_input && (
                    <div className={`${styles.message} ${styles.insuranceMessage}`}>
                      <div className={styles.messageHeader}>
                        <span className={styles.speaker}>Insurance Agent</span>
                        <span className={styles.timestamp}>{formatTimestamp(log.timestamp)}</span>
                      </div>
                      <div className={styles.messageContent}>{log.user_input}</div>
                    </div>
                  )}

                  {/* Provider Agent (Alfons) Message */}
                  {log.bot_response && (
                    <div className={`${styles.message} ${styles.alfonsMessage}`}>
                      <div className={styles.messageHeader}>
                        <span className={styles.speaker}>Provider Agent (Alfons)</span>
                        <span className={styles.timestamp}>{formatTimestamp(log.timestamp)}</span>
                      </div>
                      <div className={styles.messageContent}>{log.bot_response}</div>
                      
                      {/* Show extracted data if available */}
                      {(log.patient_id || log.procedure_code || log.insurance) && (
                        <div className={styles.extractedData}>
                          {log.patient_id && (
                            <span className={`${styles.dataBadge} ${styles.patientBadge}`}>
                              Patient: {log.patient_id}
                            </span>
                          )}
                          {log.procedure_code && (
                            <span className={`${styles.dataBadge} ${styles.procedureBadge}`}>
                              CPT: {log.procedure_code}
                            </span>
                          )}
                          {log.insurance && (
                            <span className={`${styles.dataBadge} ${styles.insuranceBadge}`}>
                              Insurance: {log.insurance}
                            </span>
                          )}
                          {log.escalated && (
                            <span className={`${styles.dataBadge} ${styles.escalatedBadge}`}>
                              Escalated
                            </span>
                          )}
                        </div>
                      )}
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