-- Database initialization script for Pain Diagnosis Application
-- This script runs automatically when MySQL container starts for the first time

USE pain_diagnosis;

-- Create patients table
CREATE TABLE IF NOT EXISTS patients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    gender ENUM('male', 'female', 'other') DEFAULT NULL,
    date_of_birth DATE DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_name (name),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create clinical_data table
CREATE TABLE IF NOT EXISTS clinical_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT NOT NULL,
    age INT DEFAULT NULL,
    pain_intensity INT DEFAULT NULL CHECK (pain_intensity BETWEEN 0 AND 10),
    duration_days INT DEFAULT NULL,
    frequency_per_week INT DEFAULT NULL,
    sleep_hours DECIMAL(3,1) DEFAULT NULL,
    stress_level INT DEFAULT NULL CHECK (stress_level BETWEEN 0 AND 10),
    pain_location VARCHAR(100) DEFAULT NULL,
    pain_type VARCHAR(100) DEFAULT NULL,
    trigger_factor VARCHAR(255) DEFAULT NULL,
    relief_factor VARCHAR(255) DEFAULT NULL,
    medication_use VARCHAR(255) DEFAULT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
    INDEX idx_patient_id (patient_id),
    INDEX idx_recorded_at (recorded_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create scales table
CREATE TABLE IF NOT EXISTS scales (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT NOT NULL,
    scale_name VARCHAR(100) NOT NULL,
    scale_score DECIMAL(5,2) DEFAULT NULL,
    scale_data JSON DEFAULT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
    INDEX idx_patient_id (patient_id),
    INDEX idx_scale_name (scale_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create diagnoses table with JSON fields for ML results
CREATE TABLE IF NOT EXISTS diagnoses (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT NOT NULL,
    clinical_data_id INT DEFAULT NULL,
    predicted_class VARCHAR(50) NOT NULL,
    probabilities_json JSON NOT NULL,
    shap_values_json JSON NOT NULL,
    confidence_score DECIMAL(5,4) DEFAULT NULL,
    model_version VARCHAR(50) DEFAULT NULL,
    diagnosed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
    FOREIGN KEY (clinical_data_id) REFERENCES clinical_data(id) ON DELETE SET NULL,
    INDEX idx_patient_id (patient_id),
    INDEX idx_predicted_class (predicted_class),
    INDEX idx_diagnosed_at (diagnosed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create flags table for clinical flag detection results
CREATE TABLE IF NOT EXISTS flags (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT NOT NULL,
    diagnosis_id INT DEFAULT NULL,
    flag_type ENUM('red', 'yellow', 'blue', 'black') NOT NULL,
    flag_id VARCHAR(20) NOT NULL,
    flag_name VARCHAR(255) NOT NULL,
    confidence ENUM('High', 'Medium', 'Low') NOT NULL,
    keyword_matched VARCHAR(255) DEFAULT NULL,
    context_snippet TEXT DEFAULT NULL,
    threshold_met BOOLEAN DEFAULT FALSE,
    priority_rank INT DEFAULT NULL,
    status ENUM('active', 'excluded') DEFAULT 'active',
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
    FOREIGN KEY (diagnosis_id) REFERENCES diagnoses(id) ON DELETE SET NULL,
    INDEX idx_patient_id (patient_id),
    INDEX idx_flag_type (flag_type),
    INDEX idx_confidence (confidence),
    INDEX idx_status (status),
    INDEX idx_detected_at (detected_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create documents table for uploaded medical records
CREATE TABLE IF NOT EXISTS documents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_type ENUM('pdf', 'docx') NOT NULL,
    file_size BIGINT DEFAULT NULL,
    extracted_text LONGTEXT DEFAULT NULL,
    patient_age_extracted INT DEFAULT NULL,
    patient_temperature DECIMAL(4,2) DEFAULT NULL,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
    INDEX idx_patient_id (patient_id),
    INDEX idx_file_type (file_type),
    INDEX idx_uploaded_at (uploaded_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Insert sample data for testing (optional, comment out in production)
-- INSERT INTO patients (name, gender) VALUES 
--     ('Иванов Иван', 'male'),
--     ('Петрова Анна', 'female');

-- Grant privileges to application user
CREATE USER IF NOT EXISTS 'pain_user'@'%' IDENTIFIED BY 'pain_password_123';
GRANT SELECT, INSERT, UPDATE, DELETE ON pain_diagnosis.* TO 'pain_user'@'%';
FLUSH PRIVILEGES;
