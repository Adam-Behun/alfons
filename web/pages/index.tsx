/**
 * index.tsx
 *
 * This is the main frontend page for the Alfons Prior Authorization Bot.
 * It displays the EHR interface and voice agent status with real-time updates.
 */

import { useState, useEffect } from 'react';
import axios from 'axios';
import { createClient } from '@supabase/supabase-js';
import EHRInterface from '../components/EHRInterface';
import VoiceAgentStatus from '../components/VoiceAgentStatus';

// Initialize Supabase client using environment variables
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_KEY!;
const supabase = createClient(supabaseUrl, supabaseKey);

export default function Home() {
  const [phoneNumber, setPhoneNumber] = useState('');
  const [isCallInProgress, setIsCallInProgress] = useState(false);
  const [callStatus, setCallStatus] = useState('');

  /**
   * Sends a POST request to the backend to trigger a phone call.
   */
  const triggerCall = async () => {
    if (!phoneNumber.trim()) {
      alert('Please enter a phone number');
      return;
    }

    const formData = new FormData();
    formData.append('phone_number', phoneNumber);

    try {
      setIsCallInProgress(true);
      setCallStatus('Initiating call...');
      
      const response = await axios.post(
        `${process.env.NEXT_PUBLIC_API_URL}/trigger-call`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      
      setCallStatus(`Call initiated successfully! Call SID: ${response.data.call_sid || 'Unknown'}`);
      
      // Clear status after 5 seconds
      setTimeout(() => {
        setCallStatus('');
        setIsCallInProgress(false);
      }, 5000);
      
    } catch (error) {
      console.error('Error triggering call:', error);
      setCallStatus('Failed to trigger call. Please check your backend connection.');
      setIsCallInProgress(false);
      
      // Clear error after 5 seconds
      setTimeout(() => setCallStatus(''), 5000);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Alfons Prior Authorization System
              </h1>
              <p className="text-gray-600 mt-1">
                Real-time EHR integration with AI voice agent
              </p>
            </div>
            
            {/* Call Trigger Controls */}
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={phoneNumber}
                  onChange={(e) => setPhoneNumber(e.target.value)}
                  placeholder="Enter phone number (+1234567890)"
                  className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  disabled={isCallInProgress}
                />
                <button
                  onClick={triggerCall}
                  disabled={isCallInProgress}
                  className={`px-4 py-2 rounded-md font-medium transition-colors ${
                    isCallInProgress
                      ? 'bg-gray-400 text-white cursor-not-allowed'
                      : 'bg-blue-600 text-white hover:bg-blue-700'
                  }`}
                >
                  {isCallInProgress ? 'Calling...' : 'Start Prior Auth Call'}
                </button>
              </div>
            </div>
          </div>
          
          {/* Call Status */}
          {callStatus && (
            <div className={`mt-4 p-3 rounded-md ${
              callStatus.includes('Failed') || callStatus.includes('error')
                ? 'bg-red-50 text-red-700 border border-red-200'
                : 'bg-green-50 text-green-700 border border-green-200'
            }`}>
              {callStatus}
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - EHR Interface */}
          <div>
            <EHRInterface />
          </div>
          
          {/* Right Column - Voice Agent Status */}
          <div>
            <VoiceAgentStatus />
          </div>
        </div>
      </div>
    </div>
  );
}