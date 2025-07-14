/**
 * index.tsx
 *
 * This is the main frontend page for the Alfons Prior Authorization Bot.
 * It displays two symmetrical windows: EHR interface and voice agent status.
 * Now includes navigation to Call Analytics.
 */

import { useState, useEffect } from 'react';
import { createClient } from '@supabase/supabase-js';
import EHRInterface from '../components/EHRInterface';
import VoiceAgentWindow from '../components/VoiceAgentWindow';
import CallAnalytics from '../components/CallAnalytics';
import styles from '../styles/Home.module.css';

// Initialize Supabase client using environment variables
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_KEY!;
const supabase = createClient(supabaseUrl, supabaseKey);

type ViewMode = 'dashboard' | 'analytics';

export default function Home() {
  const [currentView, setCurrentView] = useState<ViewMode>('dashboard');

  if (currentView === 'analytics') {
    return (
      <div className={styles.container}>
        {/* Header with Navigation */}
        <div className={styles.header}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h1 className={styles.title}>Prior Authorization System</h1>
              <p className={styles.subtitle}>Demo AI Voice Agent</p>
            </div>
            <button
              onClick={() => setCurrentView('dashboard')}
              style={{
                background: 'rgba(255, 255, 255, 0.2)',
                color: 'white',
                border: 'none',
                padding: '8px 16px',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '0.9rem'
              }}
            >
              ← Back to Dashboard
            </button>
          </div>
        </div>

        {/* Analytics Content */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <CallAnalytics />
        </div>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      {/* Header with Navigation */}
      <div className={styles.header}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1 className={styles.title}>Prior Authorization System</h1>
            <p className={styles.subtitle}>Demo AI Voice Agent</p>
          </div>
          <button
            onClick={() => setCurrentView('analytics')}
            style={{
              background: 'rgba(255, 255, 255, 0.2)',
              color: 'white',
              border: 'none',
              padding: '8px 16px',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '0.9rem',
              transition: 'all 0.2s ease'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.3)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.2)';
            }}
          >
            📊 Call Analytics
          </button>
        </div>
      </div>

      {/* Main Content - Two Symmetrical Windows */}
      <div className={styles.mainContent}>
        {/* Left Window - EHR Interface */}
        <div className={styles.leftWindow}>
          <EHRInterface />
        </div>
        
        {/* Right Window - Voice Agent */}
        <div className={styles.rightWindow}>
          <VoiceAgentWindow />
        </div>
      </div>
    </div>
  );
}