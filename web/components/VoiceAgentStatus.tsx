import { useState, useEffect } from 'react';
import styles from './VoiceAgentStatus.module.css';

interface ConversationLog {
  id: number;
  call_sid: string;
  user_input: string;
  bot_response: string;
  patient_id?: string;
  procedure_code?: string;
  insurance?: string;
  approval_status?: string;
  auth_number?: string;
  escalated: boolean;
  timestamp: string;
}

export default function VoiceAgentStatus() {
  const [logs, setLogs] = useState<ConversationLog[]>([]);
  const [currentCall, setCurrentCall] = useState<ConversationLog | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [lastFetch, setLastFetch] = useState<Date | null>(null);

  const fetchLogs = async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/logs`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const data = await response.json();
      setLogs(data.slice(-10).reverse()); // Show last 10 interactions, newest first
      setLastFetch(new Date());
      
      // Find active call (within last 5 minutes)
      const fiveMinutesAgo = Date.now() - 300000;
      const activeCall = data.find((log: ConversationLog) => 
        new Date(log.timestamp).getTime() > fiveMinutesAgo
      );
      setCurrentCall(activeCall || null);
      
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

  const truncateText = (text: string, maxLength: number = 100) => {
    if (!text) return '';
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getStatusBadgeClass = (status?: string) => {
    switch (status?.toLowerCase()) {
      case 'approved':
        return styles.badgeApproved;
      case 'denied':
        return styles.badgeDenied;
      default:
        return styles.badgePending;
    }
  };

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerContent}>
          <h2 className={styles.title}>Alfons Voice Agent</h2>
          <p className={styles.subtitle}>Live conversation monitoring and transcription</p>
        </div>
        
        <div className={styles.headerControls}>
          <div className={`${styles.statusBadge} ${
            currentCall 
              ? `${styles.statusActive} ${styles.pulse}` 
              : styles.statusStandby
          }`}>
            {currentCall ? 'ACTIVE CALL' : 'STANDBY'}
          </div>
          
          {lastFetch && (
            <div className={styles.lastUpdate}>
              Updated: {lastFetch.toLocaleTimeString()}
            </div>
          )}
        </div>
      </div>

      {/* Current Active Call */}
      {currentCall && (
        <div className={styles.activeCall}>
          <div className={styles.activeCallHeader}>
            <h3 className={styles.activeCallTitle}>Live Conversation</h3>
            <span className={styles.activeCallTime}>
              {formatTimestamp(currentCall.timestamp)}
            </span>
          </div>
          
          <div className={styles.conversationPair}>
            <div className={`${styles.message} ${styles.messageInsurance}`}>
              <div className={styles.messageLabel}>Insurance Representative:</div>
              <div className={styles.messageText}>{currentCall.user_input || 'Listening...'}</div>
            </div>
            
            <div className={`${styles.message} ${styles.messageAlfons}`}>
              <div className={styles.messageLabel}>Alfons Response:</div>
              <div className={styles.messageText}>{currentCall.bot_response || 'Processing...'}</div>
            </div>
          </div>

          {/* Extracted Data Preview */}
          {(currentCall.patient_id || currentCall.procedure_code || currentCall.approval_status) && (
            <div className={styles.extractedData}>
              <div className={styles.extractedDataLabel}>Extracted Data:</div>
              <div className={styles.badgeContainer}>
                {currentCall.patient_id && (
                  <span className={`${styles.dataBadge} ${styles.badgePatient}`}>
                    Patient: {currentCall.patient_id}
                  </span>
                )}
                {currentCall.procedure_code && (
                  <span className={`${styles.dataBadge} ${styles.badgeProcedure}`}>
                    CPT: {currentCall.procedure_code}
                  </span>
                )}
                {currentCall.approval_status && (
                  <span className={`${styles.dataBadge} ${getStatusBadgeClass(currentCall.approval_status)}`}>
                    Status: {currentCall.approval_status}
                  </span>
                )}
                {currentCall.auth_number && (
                  <span className={`${styles.dataBadge} ${styles.badgeAuth}`}>
                    Auth: {currentCall.auth_number}
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Conversation History */}
      <div className={styles.historySection}>
        <div className={styles.historyHeader}>
          <h3 className={styles.historyTitle}>Conversation History</h3>
          <span className={styles.historyCount}>{logs.length} interactions</span>
        </div>

        {isLoading ? (
          <div className={styles.loadingContainer}>
            <div className={styles.spinner}></div>
            <p className={styles.loadingText}>Loading conversation history...</p>
          </div>
        ) : logs.length === 0 ? (
          <div className={styles.emptyState}>
            <p className={styles.emptyStateText}>No conversations yet</p>
            <p className={styles.emptyStateSubtext}>Start a call to see interactions here</p>
          </div>
        ) : (
          <div className={styles.logContainer}>
            {logs.map((log, index) => (
              <div 
                key={log.id || index} 
                className={`${styles.logEntry} ${
                  log.id === currentCall?.id ? styles.logEntryActive : ''
                }`}
              >
                <div className={styles.logHeader}>
                  <span className={styles.logTime}>
                    {formatTimestamp(log.timestamp)}
                  </span>
                  <span className={styles.logCallId}>
                    Call: {log.call_sid.slice(-8)}
                  </span>
                </div>
                
                <div className={styles.logMessages}>
                  <div className={styles.logMessage}>
                    <span className={styles.logSpeaker}>Rep:</span>
                    <span className={styles.logText}>{truncateText(log.user_input)}</span>
                  </div>
                  <div className={styles.logMessage}>
                    <span className={styles.logSpeaker}>Alfons:</span>
                    <span className={styles.logText}>{truncateText(log.bot_response)}</span>
                  </div>
                </div>

                {/* Show extracted data if available */}
                {(log.patient_id || log.procedure_code || log.approval_status) && (
                  <div className={styles.logExtractedData}>
                    <div className={styles.logBadgeContainer}>
                      {log.patient_id && (
                        <span className={`${styles.logBadge} ${styles.logBadgePatient}`}>
                          {log.patient_id}
                        </span>
                      )}
                      {log.procedure_code && (
                        <span className={`${styles.logBadge} ${styles.logBadgeProcedure}`}>
                          {log.procedure_code}
                        </span>
                      )}
                      {log.approval_status && (
                        <span className={`${styles.logBadge} ${getStatusBadgeClass(log.approval_status)}`}>
                          {log.approval_status}
                        </span>
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