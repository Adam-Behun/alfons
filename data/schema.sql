DROP TABLE IF EXISTS conversations;
DROP TABLE IF EXISTS patients;

-- Create patients table
CREATE TABLE patients (
    id SERIAL PRIMARY KEY,
    patient_name VARCHAR(255) NOT NULL,
    date_of_birth DATE NOT NULL,
    sex VARCHAR(10),
    patient_phone_number VARCHAR(20),
    address TEXT,
    city VARCHAR(100),
    state VARCHAR(50),
    zip_code VARCHAR(20),
    insurance_company_name VARCHAR(255) NOT NULL,
    insurance_member_id VARCHAR(100) NOT NULL,
    insurance_phone_number VARCHAR(20) NOT NULL,
    plan_type VARCHAR(100),
    cpt_code VARCHAR(10) NOT NULL,
    icd10_code VARCHAR(10),
    appointment_time TIMESTAMP NOT NULL,
    provider_name VARCHAR(255),
    provider_npi VARCHAR(20),
    provider_phone_number VARCHAR(20),
    provider_specialty VARCHAR(100),
    facility_name VARCHAR(255),
    facility_npi VARCHAR(20),
    place_of_service_code VARCHAR(5),
    prior_auth_status VARCHAR(50) DEFAULT 'Pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create conversations table with foreign key
CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    call_sid VARCHAR(255) NOT NULL,
    user_input TEXT,
    bot_response TEXT,
    patient_id INTEGER REFERENCES patients(id),
    procedure_code VARCHAR(50),
    icd10_code VARCHAR(10),
    insurance VARCHAR(100),
    approval_status VARCHAR(50),
    auth_number VARCHAR(100),
    denial_reason TEXT,
    appeal_status VARCHAR(50),
    clinical_notes TEXT,
    requested_start_date DATE,
    requested_end_date DATE,
    escalated BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);