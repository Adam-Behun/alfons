// Patient Component - Shows detailed patient information
// Displays all patient data fetched from the database

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ArrowLeft, User, Calendar, Shield, FileText, Phone, Building, Loader2 } from 'lucide-react';
import { cn, getStatusColor } from '@/lib/utils';

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
      <div className="flex items-center justify-center h-full">
        <div className="flex flex-col items-center space-y-4">
          <Loader2 className="w-8 h-8 animate-spin text-primary" />
          <p className="text-muted-foreground">Loading patient details...</p>
        </div>
      </div>
    );
  }

  if (error || !patient) {
    return (
      <div className="flex flex-col h-full">
        <div className="p-6 border-b">
          <Button variant="ghost" onClick={onBack} className="gap-2 mb-4">
            <ArrowLeft className="w-4 h-4" />
            Back to Patient List
          </Button>
        </div>
        <div className="flex items-center justify-center flex-1">
          <Card className="max-w-md">
            <CardContent className="pt-6 text-center">
              <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <User className="w-6 h-6 text-red-600" />
              </div>
              <h3 className="font-semibold text-lg mb-2">Error Loading Patient</h3>
              <p className="text-muted-foreground text-sm">{error || 'Patient not found'}</p>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  const getStatusVariant = (status: string) => {
    switch (status.toLowerCase()) {
      case 'pending': return 'pending';
      case 'approved': return 'approved';
      case 'denied': return 'denied';
      default: return 'secondary';
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-6 border-b bg-gradient-to-r from-blue-600 to-blue-700 text-white">
        <Button 
          variant="ghost" 
          onClick={onBack} 
          className="gap-2 mb-4 text-white hover:bg-white/20"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Patient List
        </Button>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-white/20 rounded-lg flex items-center justify-center">
              <User className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-xl font-bold">{patient.patient_name}</h2>
              <p className="text-blue-100 text-sm">Patient Details</p>
            </div>
          </div>
          <Badge variant={getStatusVariant(patient.prior_auth_status)} className="text-sm">
            {patient.prior_auth_status}
          </Badge>
        </div>
      </div>

      {/* Call Insurance Button */}
      {patient.prior_auth_status === 'Pending' && (
        <div className="p-6 border-b bg-gradient-to-r from-emerald-50 to-green-50">
          <Button
            onClick={triggerCall}
            disabled={calling}
            size="lg"
            className="gap-2"
          >
            {calling ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Calling...
              </>
            ) : (
              <>
                <Phone className="w-4 h-4" />
                Call Insurance for Prior Authorization
              </>
            )}
          </Button>
        </div>
      )}

      {/* Patient Details Content */}
      <div className="flex-1 overflow-auto p-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 max-w-6xl mx-auto">
          {/* Basic Information */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <User className="w-5 h-5 text-primary" />
                Basic Information
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium text-muted-foreground">Date of Birth</label>
                  <p className="font-medium">{patient.date_of_birth}</p>
                </div>
                {patient.sex && (
                  <div>
                    <label className="text-sm font-medium text-muted-foreground">Sex</label>
                    <p className="font-medium">{patient.sex}</p>
                  </div>
                )}
              </div>
              {patient.patient_phone_number && (
                <div>
                  <label className="text-sm font-medium text-muted-foreground">Phone Number</label>
                  <p className="font-medium">{patient.patient_phone_number}</p>
                </div>
              )}
              {patient.address && (
                <div>
                  <label className="text-sm font-medium text-muted-foreground">Address</label>
                  <p className="font-medium">
                    {patient.address}
                    {patient.city && `, ${patient.city}`}
                    {patient.state && `, ${patient.state}`}
                    {patient.zip_code && ` ${patient.zip_code}`}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Insurance Information */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="w-5 h-5 text-primary" />
                Insurance Information
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium text-muted-foreground">Insurance Company</label>
                <p className="font-medium">{patient.insurance_company_name}</p>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium text-muted-foreground">Member ID</label>
                  <p className="font-medium">{patient.insurance_member_id}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-muted-foreground">Phone</label>
                  <p className="font-medium">{patient.insurance_phone_number}</p>
                </div>
              </div>
              {patient.plan_type && (
                <div>
                  <label className="text-sm font-medium text-muted-foreground">Plan Type</label>
                  <p className="font-medium">{patient.plan_type}</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Medical Information */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-primary" />
                Medical Information
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium text-muted-foreground">CPT Code</label>
                  <p className="font-medium">{patient.cpt_code}</p>
                </div>
                {patient.icd10_code && (
                  <div>
                    <label className="text-sm font-medium text-muted-foreground">ICD-10 Code</label>
                    <p className="font-medium">{patient.icd10_code}</p>
                  </div>
                )}
              </div>
              <div>
                <label className="text-sm font-medium text-muted-foreground">Appointment Time</label>
                <p className="font-medium">{patient.appointment_time}</p>
              </div>
              {patient.place_of_service_code && (
                <div>
                  <label className="text-sm font-medium text-muted-foreground">Place of Service</label>
                  <p className="font-medium">{patient.place_of_service_code}</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Provider Information */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Building className="w-5 h-5 text-primary" />
                Provider Information
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {patient.provider_name && (
                <div>
                  <label className="text-sm font-medium text-muted-foreground">Provider Name</label>
                  <p className="font-medium">{patient.provider_name}</p>
                </div>
              )}
              <div className="grid grid-cols-2 gap-4">
                {patient.provider_npi && (
                  <div>
                    <label className="text-sm font-medium text-muted-foreground">Provider NPI</label>
                    <p className="font-medium">{patient.provider_npi}</p>
                  </div>
                )}
                {patient.provider_phone_number && (
                  <div>
                    <label className="text-sm font-medium text-muted-foreground">Provider Phone</label>
                    <p className="font-medium">{patient.provider_phone_number}</p>
                  </div>
                )}
              </div>
              {patient.provider_specialty && (
                <div>
                  <label className="text-sm font-medium text-muted-foreground">Specialty</label>
                  <p className="font-medium">{patient.provider_specialty}</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Facility Information */}
          {(patient.facility_name || patient.facility_npi) && (
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Building className="w-5 h-5 text-primary" />
                  Facility Information
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  {patient.facility_name && (
                    <div>
                      <label className="text-sm font-medium text-muted-foreground">Facility Name</label>
                      <p className="font-medium">{patient.facility_name}</p>
                    </div>
                  )}
                  {patient.facility_npi && (
                    <div>
                      <label className="text-sm font-medium text-muted-foreground">Facility NPI</label>
                      <p className="font-medium">{patient.facility_npi}</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}