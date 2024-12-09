drop table if exists framingham_data;
CREATE TABLE framingham_data (
    patient_id SERIAL PRIMARY KEY,
    male INT,
    age INT,
    education INT,
    currentSmoker INT,
    cigsPerDay INT,
    BPMeds INT,
    prevalentStroke INT,
    prevalentHyp INT,
    diabetes INT,
    totChol INT,
    sysBP FLOAT,
    diaBP FLOAT,
    BMI FLOAT,
    heartRate INT,
    glucose FLOAT,
    TenYearCHD INT
);