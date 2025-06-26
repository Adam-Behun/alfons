CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    call_sid VARCHAR(255) NOT NULL,
    user_input TEXT,
    bot_response TEXT,
    patient_id VARCHAR(50),
    procedure_code VARCHAR(50),
    insurance VARCHAR(100),
    escalated BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);