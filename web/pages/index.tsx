/**
 * index.tsx
 *
 * This is the main frontend page for the Alfons Prior Authorization Bot.
 * It displays two symmetrical windows: EHR interface and voice agent status.
 */

import { useState, useEffect } from 'react';
import { createClient } from '@supabase/supabase-js';
import EHRInterface from '../components/EHRInterface';
import VoiceAgentWindow from '../components/VoiceAgentWindow';
import styles from '../styles/Home.module.css';

// Initialize Supabase client using environment variables
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_KEY!;
const supabase = createClient(supabaseUrl, supabaseKey);

export default function Home() {
  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <h1 className={styles.title}>Prior Authorization System</h1>
        <p className={styles.subtitle}>Demo AI Voice Agent</p>
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