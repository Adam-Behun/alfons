import { useState, useEffect } from 'react';
import { createClient } from '@supabase/supabase-js';
import styles from './EHRInterface.module.css';

interface AuthorizationData {
  patient_id?: string;
  procedure_code?: string;
  insurance?: string;
  approval_status?: string;
  auth_number?: string;
  timestamp?: string;
  call_sid?: string;
}

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

export default function EHRInterface() {
  const [authData, setAuthData] = useState<AuthorizationData>({});
  const [isLiveCall, setIsLiveCall] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [callStatus, setCallStatus] = useState<string>('waiting');

  useEffect(() => {
    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_KEY!
    );

    // Subscribe to real-time updates from conversations table
    const subscription = supabase
      .channel('auth_updates')
      .on(
        'postgres_changes',
        { 
          event: '*', 
          schema: 'public', 
          table: 'conversations' 
        },
        (payload) => {
          console.log('Received real-time update:', payload);
          
          const data = payload.new as ConversationLog || payload.old as ConversationLog;
          
          if (data) {
            setIsLiveCall(true);
            setCallStatus('active');
            setLastUpdate(new Date());
            
            // Update authorization data with new information
            setAuthData(prevData => ({
              ...prevData,
              patient_id: data.patient_id || prevData.patient_id,
              procedure_code: data.procedure_code || prevData.procedure_code,
              insurance: data.insurance || prevData.insurance,
              approval_status: data.approval_status || prevData.approval_status,
              auth_number: data.auth_number || prevData.auth_number,
              timestamp: data.timestamp,
              call_sid: data.call_sid
            }));

            // Auto-hide live indicator after 10 seconds
            setTimeout(() => {
              setIsLiveCall(false);
              setCallStatus('completed');
            }, 10000);
          }
        }
      )
      .subscribe();

    // Cleanup subscription on unmount
    return () => {
      supabase.removeChannel(subscription);
    };
  }, []);

  const getStatusColorClass = (status?: string) => {
    switch (status?.toLowerCase()) {
      case 'approved':
        return styles.statusApproved;
      case 'denied':
        return styles.statusDenied;
      case 'pending':
      case 'under_review':
        return styles.statusPending;
      default:
        return styles.statusDefault;
    }
  };

  const getFieldColorClass = (value?: string) => {
    return value ? styles.fieldPopulated : styles.fieldEmpty;
  };

  const clearData = () => {
    setAuthData({});
    setIsLiveCall(false);
    setCallStatus('waiting');
    setLastUpdate(null);
  };

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerContent}>
          <h2 className={styles.title}>Prior Authorization Request</h2>
          <p className={styles.subtitle}>EHR Integration Demo - Real-time Field Population</p>
        </div>
        
        <div className={styles.headerControls}>
          <div className={`${styles.statusBadge} ${
            isLiveCall 
              ? `${styles.statusLive} ${styles.pulse}` 
              : callStatus === 'completed'
              ? styles.statusCompleted
              : styles.statusWaiting
          }`}>
            {isLiveCall ? 'LIVE CALL - Data Updating' : 
             callStatus === 'completed' ? 'Call Completed' : 
             'Waiting for Alfons Call'}
          </div>
          
          {authData.call_sid && (
            <button
              onClick={clearData}
              className={styles.clearButton}
            >
              Clear Data
            </button>
          )}
        </div>
      </div>

      {/* Form Grid */}
      <div className={styles.formGrid}>
        {/* Left Column - Patient Information */}
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={`${styles.sectionIndicator} ${styles.indicatorBlue}`}></span>
            Patient Information
          </div>
          
          <div className={styles.fieldGroup}>
            <div className={styles.field}>
              <label className={styles.fieldLabel}>
                Patient ID
                {authData.patient_id && (
                  <span className={styles.autoPopulatedLabel}>
                    Auto-populated by Alfons
                  </span>
                )}
              </label>
              <input
                type="text"
                value={authData.patient_id || ''}
                className={`${styles.input} ${getFieldColorClass(authData.patient_id)}`}
                placeholder="Waiting for Alfons to extract..."
                readOnly
              />
            </div>

            <div className={styles.field}>
              <label className={styles.fieldLabel}>
                Insurance Provider
                {authData.insurance && (
                  <span className={styles.autoPopulatedLabel}>
                    Auto-populated by Alfons
                  </span>
                )}
              </label>
              <input
                type="text"
                value={authData.insurance || ''}
                className={`${styles.input} ${getFieldColorClass(authData.insurance)}`}
                placeholder="Waiting for Alfons to extract..."
                readOnly
              />
            </div>

            <div className={styles.field}>
              <label className={styles.fieldLabel}>Patient Name</label>
              <input
                type="text"
                value=""
                className={`${styles.input} ${styles.fieldStatic}`}
                placeholder="Demo: Static field (not extracted)"
                readOnly
              />
            </div>

            <div className={styles.field}>
              <label className={styles.fieldLabel}>Date of Birth</label>
              <input
                type="text"
                value=""
                className={`${styles.input} ${styles.fieldStatic}`}
                placeholder="Demo: Static field (not extracted)"
                readOnly
              />
            </div>
          </div>
        </div>

        {/* Right Column - Procedure Information */}
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={`${styles.sectionIndicator} ${styles.indicatorPurple}`}></span>
            Procedure Details
          </div>
          
          <div className={styles.fieldGroup}>
            <div className={styles.field}>
              <label className={styles.fieldLabel}>
                CPT/Procedure Code
                {authData.procedure_code && (
                  <span className={styles.autoPopulatedLabel}>
                    Auto-populated by Alfons
                  </span>
                )}
              </label>
              <input
                type="text"
                value={authData.procedure_code || ''}
                className={`${styles.input} ${getFieldColorClass(authData.procedure_code)}`}
                placeholder="Waiting for Alfons to extract..."
                readOnly
              />
            </div>

            <div className={styles.field}>
              <label className={styles.fieldLabel}>
                Authorization Status
                {authData.approval_status && (
                  <span className={styles.autoPopulatedLabel}>
                    Auto-populated by Alfons
                  </span>
                )}
              </label>
              <input
                type="text"
                value={authData.approval_status ? authData.approval_status.toUpperCase() : ''}
                className={`${styles.input} ${styles.statusInput} ${
                  authData.approval_status ? getStatusColorClass(authData.approval_status) : styles.fieldEmpty
                }`}
                placeholder="Waiting for Alfons to extract..."
                readOnly
              />
            </div>

            <div className={styles.field}>
              <label className={styles.fieldLabel}>
                Authorization Number
                {authData.auth_number && (
                  <span className={styles.autoPopulatedLabel}>
                    Auto-populated by Alfons
                  </span>
                )}
              </label>
              <input
                type="text"
                value={authData.auth_number || ''}
                className={`${styles.input} ${getFieldColorClass(authData.auth_number)}`}
                placeholder="Waiting for Alfons to extract..."
                readOnly
              />
            </div>

            <div className={styles.field}>
              <label className={styles.fieldLabel}>Procedure Description</label>
              <textarea
                value=""
                className={`${styles.textarea} ${styles.fieldStatic}`}
                placeholder="Demo: Static field (not extracted)"
                readOnly
              />
            </div>
          </div>
        </div>
      </div>

      {/* Status Footer */}
      <div className={styles.footer}>
        <div className={styles.footerContent}>
          <div className={styles.statusInfo}>
            {lastUpdate && (
              <div className={styles.lastUpdate}>
                <span className={styles.label}>Last updated:</span> {lastUpdate.toLocaleString()}
              </div>
            )}
            
            {authData.call_sid && (
              <div className={styles.callId}>
                <span className={styles.label}>Call ID:</span> {authData.call_sid.slice(-8)}
              </div>
            )}
          </div>

          <div className={styles.statusIndicators}>
            {Object.values(authData).filter(val => val && val !== authData.timestamp && val !== authData.call_sid).length > 0 && (
              <div className={styles.fieldsPopulated}>
                {Object.values(authData).filter(val => val && val !== authData.timestamp && val !== authData.call_sid).length} fields populated
              </div>
            )}
            
            <div className={`${styles.statusDot} ${
              isLiveCall ? `${styles.dotLive} ${styles.pulse}` : 
              callStatus === 'completed' ? styles.dotCompleted : styles.dotWaiting
            }`}></div>
          </div>
        </div>

        {/* Demo Instructions */}
        {!authData.call_sid && (
          <div className={styles.demoInstructions}>
            <h4 className={styles.instructionsTitle}>Demo Instructions</h4>
            <p className={styles.instructionsText}>
              Click "Start Prior Auth Call" above, then answer the phone and roleplay as an insurance representative. 
              Watch this EHR form populate in real-time as you provide authorization details to Alfons.
            </p>
            <p className={styles.instructionsExample}>
              <strong>Example response:</strong> "Patient 12345 is approved for CPT code 99213, authorization number AUTH-789123"
            </p>
          </div>
        )}
      </div>
    </div>
  );
}