// Simple EHR Interface for Alfons Prior Authorization Bot
// Shows patients needing prior auth with call functionality

import { useState, useEffect } from 'react';
import styles from './EHRInterface.module.css';
import Patient from './Patient';

interface PatientRecord {
  id: number;
  patient_name: string;
  date_of_birth: string;
  insurance_company_name: string;
  insurance_member_id: string;
  insurance_phone_number: string;
  cpt_code: string;
  appointment_time: string;
  prior_auth_status: string;
}

export default function EHRInterface() {
  const [patients, setPatients] = useState<PatientRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [callingPatient, setCallingPatient] = useState<number | null>(null);
  const [selectedPatientId, setSelectedPatientId] = useState<number | null>(null);

  // Demo phone number for all calls
  const DEMO_PHONE = "516-566-7132";

  const fetchPatients = async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/patients`);
      if (!response.ok) {
        throw new Error(`Failed to fetch patients: ${response.statusText}`);
      }
      const data = await response.json();
      setPatients(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching patients:', err);
      setError('Failed to load patients from the database. Please check the API connection.');
      setPatients([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPatients();
  }, []);

  const triggerCall = async (patient: PatientRecord) => {
    setCallingPatient(patient.id);
    
    const formData = new FormData();
    formData.append('phone_number', DEMO_PHONE);
    
    try {
      await fetch(`${process.env.NEXT_PUBLIC_API_URL}/trigger-call`, {
        method: 'POST',
        body: formData,
      });
      alert(`Alfons is calling for ${patient.patient_name}'s prior authorization`);
    } catch (err) {
      alert('Failed to trigger call');
    } finally {
      setCallingPatient(null);
    }
  };

  const handlePatientClick = (patientId: number) => {
    setSelectedPatientId(patientId);
  };

  const handleBackToList = () => {
    setSelectedPatientId(null);
  };

  // Show patient details if a patient is selected
  if (selectedPatientId) {
    return <Patient patientId={selectedPatientId} onBack={handleBackToList} />;
  }

  if (loading) {
    return (
      <div className={styles.container}>
        <div className={styles.loading}>Loading patients...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={styles.container}>
        <div className={styles.error}>{error}</div>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <h2 className={styles.title}>EHR - Prior Authorization Queue</h2>
        <p className={styles.subtitle}>Patients requiring prior authorization</p>
      </div>
      
      {/* Patient List */}
      <div className={styles.content}>
        <div className={styles.patientList}>
          {patients.length === 0 ? (
            <div className={styles.noPatients}>No patients requiring prior authorization at this time.</div>
          ) : (
            patients.map((patient) => (
              <div 
                key={patient.id} 
                className={styles.patientCard}
                onClick={() => handlePatientClick(patient.id)}
                style={{ cursor: 'pointer' }}>
                <div className={styles.patientInfo}>
                  <div className={styles.patientHeader}>
                    <h3 className={styles.patientName}>
                      {patient.patient_name}
                    </h3>
                    <span className={`${styles.statusBadge} ${styles[patient.prior_auth_status.toLowerCase()]}`}>
                      {patient.prior_auth_status}
                    </span>
                  </div>
                  
                  <div className={styles.patientDetails}>
                    <div className={styles.detailRow}>
                      <span className={styles.label}>DOB:</span>
                      <span className={styles.value}>{patient.date_of_birth}</span>
                    </div>
                    <div className={styles.detailRow}>
                      <span className={styles.label}>Insurance:</span>
                      <span className={styles.value}>{patient.insurance_company_name}</span>
                    </div>
                    <div className={styles.detailRow}>
                      <span className={styles.label}>Member ID:</span>
                      <span className={styles.value}>{patient.insurance_member_id}</span>
                    </div>
                    <div className={styles.detailRow}>
                      <span className={styles.label}>CPT Code:</span>
                      <span className={styles.value}>{patient.cpt_code}</span>
                    </div>
                    <div className={styles.detailRow}>
                      <span className={styles.label}>Appointment:</span>
                      <span className={styles.value}>{patient.appointment_time}</span>
                    </div>
                  </div>
                </div>
                
                {patient.prior_auth_status === 'Pending' && (
                  <div className={styles.actionSection}>
                    <button
                      onClick={(e) => {
                        e.stopPropagation(); // Prevent card click
                        triggerCall(patient);
                      }}
                      disabled={callingPatient === patient.id}
                      className={`${styles.callButton} ${callingPatient === patient.id ? styles.calling : ''}`}
                    >
                      {callingPatient === patient.id ? 'Calling...' : 'Call Insurance'}
                    </button>
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}