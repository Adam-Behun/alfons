// Simple EHR Interface for Alfons Prior Authorization Bot
// Shows patients needing prior auth with call functionality

import { useState, useEffect } from 'react';
import styles from './EHRInterface.module.css';

interface Patient {
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
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState(true);
  const [callingPatient, setCallingPatient] = useState<number | null>(null);

  // Demo phone number for all calls
  const DEMO_PHONE = "516-566-7132";

  const fetchPatients = async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/patients`);
      const data = await response.json();
      setPatients(data);
    } catch (err) {
      console.error('Error fetching patients:', err);
      // Mock data for demo
      setPatients([
        {
          id: 1,
          patient_name: "John Smith",
          date_of_birth: "1980-05-15",
          insurance_company_name: "Blue Cross Blue Shield",
          insurance_member_id: "ABC123456",
          insurance_phone_number: "1-800-555-0123",
          cpt_code: "99213",
          appointment_time: "2024-01-15 10:00",
          prior_auth_status: "Pending"
        },
        {
          id: 2,
          patient_name: "Mary Johnson",
          date_of_birth: "1975-08-22",
          insurance_company_name: "Aetna",
          insurance_member_id: "DEF789012",
          insurance_phone_number: "1-800-555-0124",
          cpt_code: "99214",
          appointment_time: "2024-01-15 14:30",
          prior_auth_status: "Pending"
        },
        {
          id: 3,
          patient_name: "Robert Davis",
          date_of_birth: "1990-12-03",
          insurance_company_name: "UnitedHealth",
          insurance_member_id: "GHI345678",
          insurance_phone_number: "1-800-555-0125",
          cpt_code: "99215",
          appointment_time: "2024-01-16 09:15",
          prior_auth_status: "Approved"
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPatients();
  }, []);

  const triggerCall = async (patient: Patient) => {
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

  if (loading) {
    return (
      <div className={styles.container}>
        <div className={styles.loading}>Loading patients...</div>
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
          {patients.map((patient) => (
            <div key={patient.id} className={styles.patientCard}>
              <div className={styles.patientInfo}>
                <div className={styles.patientHeader}>
                  <h3 className={styles.patientName}>{patient.patient_name}</h3>
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
                    onClick={() => triggerCall(patient)}
                    disabled={callingPatient === patient.id}
                    className={`${styles.callButton} ${callingPatient === patient.id ? styles.calling : ''}`}
                  >
                    {callingPatient === patient.id ? 'Calling...' : 'Call Insurance'}
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}