/**
 * index.tsx
 *
 * This is the main frontend page for the Alfons Prior Authorization Bot.
 * It allows users to trigger outbound calls and view conversation logs.
 * Integrates with Supabase for real-time updates and a FastAPI backend for call logic.
 */

import { useState, useEffect } from 'react';
import axios from 'axios';
import { createClient } from '@supabase/supabase-js'

// Initialize Supabase client using environment variables
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_KEY!
const supabase = createClient(supabaseUrl, supabaseKey)

export default function Home() {
  // State for the phone number input and conversation logs
  const [phoneNumber, setPhoneNumber] = useState('');
  const [logs, setLogs] = useState<any[]>([]);

  /**
   * Fetches conversation logs from the backend API and updates state.
   */
  const fetchLogs = async () => {
    const response = await axios.get(`${process.env.NEXT_PUBLIC_API_URL}/logs`);
    setLogs(response.data);
  };

  /**
   * useEffect runs once on mount:
   * - Fetches initial logs.
   * - Subscribes to Supabase real-time updates for the 'conversations' table.
   * - Cleans up the subscription on unmount.
   */
  useEffect(() => {
    fetchLogs();
    // Subscribe to all changes in the 'conversations' table
    const subscription = supabase
      .channel('conversations')
      .on(
        'postgres_changes',
        { event: '*', schema: 'public', table: 'conversations' },
        (payload) => {
          fetchLogs(); // Refresh logs on any change
        }
      )
      .subscribe();

    // Cleanup: remove the subscription when the component unmounts
    return () => {
      supabase.removeChannel(subscription);
    };
  }, []);

  /**
   * Sends a POST request to the backend to trigger a phone call.
   * The phone number is sent as form data.
   */
  const triggerCall = async () => {
    const formData = new FormData();
    formData.append('phone_number', phoneNumber);

    try {
      await axios.post(
        `${process.env.NEXT_PUBLIC_API_URL}/trigger-call`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      // Show a success message or clear the input
    } catch (error) {
      // Log the error for debugging
      console.error('Error triggering call:', error);
      alert('Failed to trigger call. Please check your backend connection.');
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Alfons: Prior Authorization Bot</h1>
      <div className="mb-4">
        
        {/* Input for the phone number to call */}
        <input
          type="text"
          value={phoneNumber}
          onChange={(e) => setPhoneNumber(e.target.value)}
          placeholder="Enter phone number (e.g., +1234567890)"
          className="p-2 border rounded"
        />

        {/* Button to trigger the call */}
        <button
          onClick={triggerCall}
          className="ml-2 bg-blue-500 text-white p-2 rounded"
        >
          Trigger Call
        </button>
      </div>
      <h2 className="text-xl font-bold">Conversation Logs</h2>

      {/* Table displaying conversation logs */}
      <table className="w-full border">
        <thead>
          <tr>
            <th className="border p-2">Call SID</th>
            <th className="border p-2">User Input</th>
            <th className="border p-2">Bot Response</th>
            <th className="border p-2">Patient ID</th>
            <th className="border p-2">Procedure Code</th>
            <th className="border p-2">Insurance</th>
            <th className="border p-2">Approval Status</th>
            <th className="border p-2">Auth Number</th>
            <th className="border p-2">Escalated</th>
            <th className="border p-2">Timestamp</th>
          </tr>
        </thead>
        <tbody>

          {/* Render each log entry as a table row */}
          {logs.map((log: any) => (
            <tr key={log.id}>
              <td className="border p-2">{log.call_sid}</td>
              <td className="border p-2">{log.user_input}</td>
              <td className="border p-2">{log.bot_response}</td>
              <td className="border p-2">{log.patient_id}</td>
              <td className="border p-2">{log.procedure_code}</td>
              <td className="border p-2">{log.insurance}</td>
              <td className="border p-2">
                <span className={`px-2 py-1 rounded text-sm ${
                  log.approval_status === 'approved' ? 'bg-green-100 text-green-800' :
                  log.approval_status === 'denied' ? 'bg-red-100 text-red-800' :
                  log.approval_status === 'pending' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-gray-100 text-gray-800'
                }`}>
                  {log.approval_status || 'N/A'}
                </span>
              </td>
              <td className="border p-2">{log.auth_number || 'N/A'}</td>
              <td className="border p-2">{log.escalated ? 'Yes' : 'No'}</td>
              <td className="border p-2">{log.timestamp}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}