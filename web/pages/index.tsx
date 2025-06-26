import { useState, useEffect } from 'react';
import axios from 'axios';
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_KEY!
const supabase = createClient(supabaseUrl, supabaseKey)

export default function Home() {
  const [phoneNumber, setPhoneNumber] = useState('');
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    fetchLogs();
    const subscription = supabase
      .channel('conversations')
      .on('postgres_changes', { event: '*', schema: 'public', table: 'conversations' }, (payload) => {
        fetchLogs();
      })
      .subscribe();

    return () => {
      supabase.removeChannel(subscription);
    };
  }, []);

  const fetchLogs = async () => {
    const response = await axios.get(`${process.env.NEXT_PUBLIC_API_URL}/logs`);
    setLogs(response.data);
  };

  const triggerCall = async () => {
    await axios.post(`${process.env.NEXT_PUBLIC_API_URL}/trigger-call`, { phone_number: phoneNumber });
    alert('Call triggered!');
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Alfons: Prior Authorization Bot</h1>
      <div className="mb-4">
        <input
          type="text"
          value={phoneNumber}
          onChange={(e) => setPhoneNumber(e.target.value)}
          placeholder="Enter phone number (e.g., +1234567890)"
          className="p-2 border rounded"
        />
        <button
          onClick={triggerCall}
          className="ml-2 bg-blue-500 text-white p-2 rounded"
        >
          Trigger Call
        </button>
      </div>
      <h2 className="text-xl font-bold">Conversation Logs</h2>
      <table className="w-full border">
        <thead>
          <tr>
            <th className="border p-2">Call SID</th>
            <th className="border p-2">User Input</th>
            <th className="border p-2">Bot Response</th>
            <th className="border p-2">Patient ID</th>
            <th className="border p-2">Procedure Code</th>
            <th className="border p-2">Insurance</th>
            <th className="border p-2">Escalated</th>
            <th className="border p-2">Timestamp</th>
          </tr>
        </thead>
        <tbody>
          {logs.map((log: any) => (
            <tr key={log.id}>
              <td className="border p-2">{log.call_sid}</td>
              <td className="border p-2">{log.user_input}</td>
              <td className="border p-2">{log.bot_response}</td>
              <td className="border p-2">{log.patient_id}</td>
              <td className="border p-2">{log.procedure_code}</td>
              <td className="border p-2">{log.insurance}</td>
              <td className="border p-2">{log.escalated ? 'Yes' : 'No'}</td>
              <td className="border p-2">{log.timestamp}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}