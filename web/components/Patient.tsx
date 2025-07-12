// Patient Component - Shows detailed patient information
// Displays all patient data fetched from the database

import { useState, useEffect } from 'react';

interface PatientDetails {
  id: number;
  patient_name: string;
  date_of_birth: string;
  sex?: string;
  patient_phone_number?: string;
  address?: string;
  city?: string;
  state?: string;
  zip_code?: string;
  insurance_company_name: string;
  insurance_member_id: string;
  insurance_phone_number: string;
  plan_type?: string;
  cpt_code: string;
  icd10_code?: string;
  appointment_time: string;
  provider_name?: string;
  provider_npi?: string;
  provider_phone_number?: string;
  provider_specialty?: string;
  facility_name?: string;
  facility_npi?: string;
  place_of_service_code?: string;
  prior_auth_status: string;
  created_at?: string;
  updated_at?: string;
}

interface PatientProps {
  patientId: number;
  onBack: () => void;
}

export default function Patient({ patientId, onBack }: PatientProps) {
  const [patient, setPatient] = useState<PatientDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [calling, setCalling] = useState(false);

  const fetchPatientDetails = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/patients/${patientId}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch patient details: ${response.statusText}`);
      }
      
      const data = await response.json();
      setPatient(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching patient details:', err);
      setError('Failed to load patient details from the database. Please check the API connection.');
    } finally {
      setLoading(false);
    }
  };

  const triggerCall = async () => {
    setCalling(true);
    
    const formData = new FormData();
    formData.append('phone_number', '516-566-7132');
    
    try {
      await fetch(`${process.env.NEXT_PUBLIC_API_URL}/trigger-call`, {
        method: 'POST',
        body: formData,
      });
      alert(`Alfons is calling for ${patient?.patient_name}'s prior authorization`);
    } catch (err) {
      alert('Failed to trigger call');
    } finally {
      setCalling(false);
    }
  };

  useEffect(() => {
    fetchPatientDetails();
  }, [patientId]);

  if (loading) {
    return (
      <div style={{ padding: '20px', textAlign: 'center' }}>
        <div>Loading patient details...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: '20px' }}>
        <button 
          onClick={onBack}
          style={{
            background: '#6b7280',
            color: 'white',
            border: 'none',
            padding: '8px 16px',
            borderRadius: '6px',
            cursor: 'pointer',
            marginBottom: '20px'
          }}
        >
          ← Back to Patient List
        </button>
        <div style={{ color: '#dc2626', textAlign: 'center' }}>
          {error}
        </div>
      </div>
    );
  }

  if (!patient) {
    return (
      <div style={{ padding: '20px', textAlign: 'center' }}>
        <div>Patient not found</div>
      </div>
    );
  }

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <div style={{
        padding: '24px',
        borderBottom: '1px solid #e5e7eb',
        background: 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)',
        color: 'white'
      }}>
        <button 
          onClick={onBack}
          style={{
            background: 'rgba(255, 255, 255, 0.2)',
            color: 'white',
            border: 'none',
            padding: '8px 16px',
            borderRadius: '6px',
            cursor: 'pointer',
            marginBottom: '12px'
          }}
        >
          ← Back to Patient List
        </button>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', margin: '0 0 4px 0' }}>
          Patient Details
        </h2>
        <p style={{ fontSize: '0.9rem', opacity: 0.9, margin: 0 }}>
          {patient.patient_name}
        </p>
      </div>

      {/* Call Insurance Button */}
      {patient.prior_auth_status === 'Pending' && (
        <div style={{ padding: '20px', borderBottom: '1px solid #e5e7eb', background: '#f9fafb' }}>
          <button
            onClick={triggerCall}
            disabled={calling}
            style={{
              background: calling ? '#9ca3af' : 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
              color: 'white',
              border: 'none',
              padding: '12px 24px',
              borderRadius: '8px',
              fontSize: '1rem',
              fontWeight: '500',
              cursor: calling ? 'not-allowed' : 'pointer',
              boxShadow: calling ? 'none' : '0 2px 4px rgba(16, 185, 129, 0.2)',
              transition: 'all 0.2s ease'
            }}
          >
            {calling ? 'Calling...' : 'Call Insurance for Prior Authorization'}
          </button>
        </div>
      )}

      {/* Patient Details Content */}
      <div style={{ flex: 1, overflow: 'auto', padding: '20px' }}>
        <div style={{ 
          background: 'white', 
          border: '1px solid #e5e7eb', 
          borderRadius: '12px', 
          padding: '24px',
          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)'
        }}>
          {/* Patient Name and Status */}
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center', 
            marginBottom: '24px',
            paddingBottom: '16px',
            borderBottom: '1px solid #f3f4f6'
          }}>
            <h3 style={{ fontSize: '1.25rem', fontWeight: '600', margin: 0 }}>
              {patient.patient_name}
            </h3>
            <span style={{
              padding: '4px 12px',
              borderRadius: '20px',
              fontSize: '0.8rem',
              fontWeight: '500',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
              background: patient.prior_auth_status === 'Pending' ? '#fef3c7' : 
                         patient.prior_auth_status === 'Approved' ? '#d1fae5' : '#fee2e2',
              color: patient.prior_auth_status === 'Pending' ? '#92400e' : 
                     patient.prior_auth_status === 'Approved' ? '#065f46' : '#991b1b'
            }}>
              {patient.prior_auth_status}
            </span>
          </div>

          {/* Patient Information Grid */}
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', 
            gap: '20px' 
          }}>
            {/* Basic Information */}
            <div>
              <h4 style={{ fontSize: '1rem', fontWeight: '600', marginBottom: '12px', color: '#1f2937' }}>
                Basic Information
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <span style={{ fontWeight: '500', color: '#6b7280', minWidth: '120px' }}>Date of Birth:</span>
                  <span style={{ color: '#1f2937' }}>{patient.date_of_birth}</span>
                </div>
                {patient.sex && (
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <span style={{ fontWeight: '500', color: '#6b7280', minWidth: '120px' }}>Sex:</span>
                    <span style={{ color: '#1f2937' }}>{patient.sex}</span>
                  </div>
                )}
                {patient.patient_phone_number && (
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <span style={{ fontWeight: '500', color: '#6b7280', minWidth: '120px' }}>Phone:</span>
                    <span style={{ color: '#1f2937' }}>{patient.patient_phone_number}</span>
                  </div>
                )}
                {patient.address && (
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <span style={{ fontWeight: '500', color: '#6b7280', minWidth: '120px' }}>Address:</span>
                    <span style={{ color: '#1f2937' }}>
                      {patient.address}
                      {patient.city && `, ${patient.city}`}
                      {patient.state && `, ${patient.state}`}
                      {patient.zip_code && ` ${patient.zip_code}`}
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Insurance Information */}
            <div>
              <h4 style={{ fontSize: '1rem', fontWeight: '600', marginBottom: '12px', color: '#1f2937' }}>
                Insurance Information
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <span style={{ fontWeight: '500', color: '#6b7280', minWidth: '120px' }}>Company:</span>
                  <span style={{ color: '#1f2937' }}>{patient.insurance_company_name}</span>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <span style={{ fontWeight: '500', color: '#6b7280', minWidth: '120px' }}>Member ID:</span>
                  <span style={{ color: '#1f2937' }}>{patient.insurance_member_id}</span>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <span style={{ fontWeight: '500', color: '#6b7280', minWidth: '120px' }}>Phone:</span>
                  <span style={{ color: '#1f2937' }}>{patient.insurance_phone_number}</span>
                </div>
                {patient.plan_type && (
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <span style={{ fontWeight: '500', color: '#6b7280', minWidth: '120px' }}>Plan Type:</span>
                    <span style={{ color: '#1f2937' }}>{patient.plan_type}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Medical Information */}
            <div>
              <h4 style={{ fontSize: '1rem', fontWeight: '600', marginBottom: '12px', color: '#1f2937' }}>
                Medical Information
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <span style={{ fontWeight: '500', color: '#6b7280', minWidth: '120px' }}>CPT Code:</span>
                  <span style={{ color: '#1f2937' }}>{patient.cpt_code}</span>
                </div>
                {patient.icd10_code && (
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <span style={{ fontWeight: '500', color: '#6b7280', minWidth: '120px' }}>ICD-10:</span>
                    <span style={{ color: '#1f2937' }}>{patient.icd10_code}</span>
                  </div>
                )}
                <div style={{ display: 'flex', gap: '8px' }}>
                  <span style={{ fontWeight: '500', color: '#6b7280', minWidth: '120px' }}>Appointment:</span>
                  <span style={{ color: '#1f2937' }}>{patient.appointment_time}</span>
                </div>
                {patient.place_of_service_code && (
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <span style={{ fontWeight: '500', color: '#6b7280', minWidth: '120px' }}>Place of Service:</span>
                    <span style={{ color: '#1f2937' }}>{patient.place_of_service_code}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Provider Information */}
            <div>
              <h4 style={{ fontSize: '1rem', fontWeight: '600', marginBottom: '12px', color: '#1f2937' }}>
                Provider Information
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {patient.provider_name && (
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <span style={{ fontWeight: '500', color: '#6b7280', minWidth: '120px' }}>Provider:</span>
                    <span style={{ color: '#1f2937' }}>{patient.provider_name}</span>
                  </div>
                )}
                {patient.provider_npi && (
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <span style={{ fontWeight: '500', color: '#6b7280', minWidth: '120px' }}>NPI:</span>
                    <span style={{ color: '#1f2937' }}>{patient.provider_npi}</span>
                  </div>
                )}
                {patient.provider_phone_number && (
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <span style={{ fontWeight: '500', color: '#6b7280', minWidth: '120px' }}>Phone:</span>
                    <span style={{ color: '#1f2937' }}>{patient.provider_phone_number}</span>
                  </div>
                )}
                {patient.provider_specialty && (
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <span style={{ fontWeight: '500', color: '#6b7280', minWidth: '120px' }}>Specialty:</span>
                    <span style={{ color: '#1f2937' }}>{patient.provider_specialty}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Facility Information */}
            {(patient.facility_name || patient.facility_npi) && (
              <div>
                <h4 style={{ fontSize: '1rem', fontWeight: '600', marginBottom: '12px', color: '#1f2937' }}>
                  Facility Information
                </h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {patient.facility_name && (
                    <div style={{ display: 'flex', gap: '8px' }}>
                      <span style={{ fontWeight: '500', color: '#6b7280', minWidth: '120px' }}>Facility:</span>
                      <span style={{ color: '#1f2937' }}>{patient.facility_name}</span>
                    </div>
                  )}
                  {patient.facility_npi && (
                    <div style={{ display: 'flex', gap: '8px' }}>
                      <span style={{ fontWeight: '500', color: '#6b7280', minWidth: '120px' }}>Facility NPI:</span>
                      <span style={{ color: '#1f2937' }}>{patient.facility_npi}</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}