// EHR Interface for Alfons Prior Authorization Bot
// Shows patients needing prior auth with call functionality

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { User, Calendar, Shield, FileText, Phone, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
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

  const getStatusVariant = (status: string) => {
    switch (status.toLowerCase()) {
      case 'pending':
        return 'pending';
      case 'approved':
        return 'approved';
      case 'denied':
        return 'denied';
      default:
        return 'secondary';
    }
  };

  if (selectedPatientId) {
    return <Patient patientId={selectedPatientId} onBack={handleBackToList} />;
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="flex flex-col items-center space-y-4">
          <Loader2 className="w-8 h-8 animate-spin text-primary" />
          <p className="text-muted-foreground">Loading patients...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full p-6">
        <Card className="max-w-md">
          <CardContent className="pt-6 text-center">
            <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <FileText className="w-6 h-6 text-red-600" />
            </div>
            <h3 className="font-semibold text-lg mb-2">Connection Error</h3>
            <p className="text-muted-foreground text-sm">{error}</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-6 border-b bg-gradient-to-r from-blue-600 to-blue-700 text-white">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-white/20 rounded-lg flex items-center justify-center">
            <User className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold">Prior Authorization Queue</h2>
            <p className="text-blue-100 text-sm">Patients requiring authorization</p>
          </div>
        </div>
      </div>
      
      {/* Patient List */}
      <div className="flex-1 overflow-auto p-6">
        {patients.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <Card className="max-w-md">
              <CardContent className="pt-6 text-center">
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <Shield className="w-6 h-6 text-green-600" />
                </div>
                <h3 className="font-semibold text-lg mb-2">All Clear!</h3>
                <p className="text-muted-foreground text-sm">
                  No patients requiring prior authorization at this time.
                </p>
              </CardContent>
            </Card>
          </div>
        ) : (
          <div className="space-y-4">
            {patients.map((patient) => (
              <Card 
                key={patient.id} 
                className={cn(
                  "cursor-pointer transition-all duration-200 hover:shadow-md hover:-translate-y-1",
                  "border-l-4",
                  patient.prior_auth_status.toLowerCase() === 'pending' && "border-l-amber-400",
                  patient.prior_auth_status.toLowerCase() === 'approved' && "border-l-green-400",
                  patient.prior_auth_status.toLowerCase() === 'denied' && "border-l-red-400"
                )}
                onClick={() => handlePatientClick(patient.id)}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg flex items-center space-x-3">
                      <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                        <User className="w-4 h-4 text-blue-600" />
                      </div>
                      <span>{patient.patient_name}</span>
                    </CardTitle>
                    <Badge variant={getStatusVariant(patient.prior_auth_status)}>
                      {patient.prior_auth_status}
                    </Badge>
                  </div>
                </CardHeader>
                
                <CardContent className="pt-0">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                    <div className="flex items-center space-x-2 text-sm">
                      <Calendar className="w-4 h-4 text-muted-foreground" />
                      <span className="font-medium">DOB:</span>
                      <span>{patient.date_of_birth}</span>
                    </div>
                    <div className="flex items-center space-x-2 text-sm">
                      <Shield className="w-4 h-4 text-muted-foreground" />
                      <span className="font-medium">Insurance:</span>
                      <span>{patient.insurance_company_name}</span>
                    </div>
                    <div className="flex items-center space-x-2 text-sm">
                      <FileText className="w-4 h-4 text-muted-foreground" />
                      <span className="font-medium">Member ID:</span>
                      <span>{patient.insurance_member_id}</span>
                    </div>
                    <div className="flex items-center space-x-2 text-sm">
                      <FileText className="w-4 h-4 text-muted-foreground" />
                      <span className="font-medium">CPT Code:</span>
                      <span>{patient.cpt_code}</span>
                    </div>
                    <div className="flex items-center space-x-2 text-sm md:col-span-2">
                      <Calendar className="w-4 h-4 text-muted-foreground" />
                      <span className="font-medium">Appointment:</span>
                      <span>{patient.appointment_time}</span>
                    </div>
                  </div>
                  
                  {patient.prior_auth_status === 'Pending' && (
                    <div className="flex justify-end">
                      <Button
                        onClick={(e) => {
                          e.stopPropagation();
                          triggerCall(patient);
                        }}
                        disabled={callingPatient === patient.id}
                        className="gap-2"
                      >
                        {callingPatient === patient.id ? (
                          <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            Calling...
                          </>
                        ) : (
                          <>
                            <Phone className="w-4 h-4" />
                            Call Insurance
                          </>
                        )}
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}