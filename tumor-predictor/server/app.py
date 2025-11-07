import os
import io
import random
from typing import List, Optional, Tuple
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Database imports
from database import get_db, create_tables
from models.database_models import Patient, PatientFollowup, Prediction, RiskFactor, TreatmentOutcome, UserSession, User, PatientAssignment, UserRoleEnum

# Placeholder imports for ML; wire real model later
import numpy as np
try:
    import tensorflow as tf
    from models.model_utils import train_and_save, load_model, predict_trajectory
    from models.rl_agent import TumorRLAgent, PatientState as RLPatientState, TreatmentAction
except Exception:
    tf = None  # fallback if TF is unavailable
    train_and_save = None
    load_model = None
    predict_trajectory = None
    TumorRLAgent = None
    RLPatientState = None
    TreatmentAction = None

app = FastAPI(title="Oral Tumor Evolution Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory to serve uploaded images
app.mount("/data", StaticFiles(directory="data"), name="data")

# Initialize database tables on startup
@app.on_event("startup")
async def startup_event():
    create_tables()


class PatientState(BaseModel):
    patient_id: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    stage: Optional[str] = None
    location: Optional[str] = None
    lifestyle: Optional[dict] = None
    history: Optional[List[dict]] = None
    treatment: Optional[str] = "chemo"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    try:
        if not file.filename or not file.filename.endswith((".csv", ".CSV")):
            raise HTTPException(status_code=400, detail="Only CSV supported")
        
        # Save file first (fast) - read and write
        os.makedirs("data", exist_ok=True)
        save_path = os.path.join("data", file.filename)
        
        # Read file content
        content = await file.read()
        
        # Write to disk
        with open(save_path, "wb") as f:
            f.write(content)
        
        # Quick row count
        row_count = content.count(b'\n')
        if row_count == 0 and len(content) > 0:
            row_count = 1
        
        return {"rows": max(1, row_count), "path": save_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")


@app.post("/image-ingest")
async def image_ingest(image: UploadFile = File(...)):
    """Accept an image upload from the doctor and persist it for later processing.

    For the demo, we store the image under data/uploads/images and return basic metadata.
    """
    try:
        if not image.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Basic validation
        allowed_ext = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
        lower_name = image.filename.lower()
        if not any(lower_name.endswith(ext) for ext in allowed_ext):
            raise HTTPException(status_code=400, detail="Only image files are supported (png, jpg, jpeg, bmp, gif)")

        # Ensure directory exists
        base_dir = os.path.join("data", "uploads", "images")
        os.makedirs(base_dir, exist_ok=True)

        # Read and write file
        content = await image.read()
        save_path = os.path.join(base_dir, image.filename)
        
        with open(save_path, "wb") as f:
            f.write(content)

        return {
            "message": "Image uploaded successfully",
            "filename": image.filename,
            "path": save_path,
            "size_bytes": len(content),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/image-ingest-batch")
async def image_ingest_batch(images: List[UploadFile] = File(...)):
    """Accept multiple image uploads and persist them for later processing."""
    try:
        if not images:
            raise HTTPException(status_code=400, detail="No images provided")

        allowed_ext = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
        base_dir = os.path.join("data", "uploads", "images")
        os.makedirs(base_dir, exist_ok=True)

        saved = []
        for img in images:
            if not img.filename:
                continue
                
            lower_name = img.filename.lower()
            if not any(lower_name.endswith(ext) for ext in allowed_ext):
                continue  # Skip invalid files instead of failing
            
            # Read and write file
            content = await img.read()
            save_path = os.path.join(base_dir, img.filename)
            
            with open(save_path, "wb") as f:
                f.write(content)
            
            saved.append({
                "filename": img.filename,
                "path": save_path,
                "size_bytes": len(content),
            })

        if not saved:
            raise HTTPException(status_code=400, detail="No valid images were uploaded")

        return {
            "message": "Images uploaded successfully",
            "count": len(saved),
            "files": saved,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")


class AnalyzeRequest(BaseModel):
    csv_path: str
    image_files: Optional[List[str]] = None
    treatment: Optional[str] = None


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Analyze an uploaded CSV (and optional images) and return dashboard-ready data."""
    if not os.path.exists(req.csv_path):
        raise HTTPException(status_code=400, detail="CSV path not found")

    try:
        df = pd.read_csv(req.csv_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    # Normalize expected column names
    colmap = {
        "Follow_Up_Month": "month_index",
        "Tumor_Size_cm": "tumor_size_cm",
        "Stage_TNM": "stage",
        "Treatment_Type": "treatment_type",
        "Response_to_Treatment": "response",
    }
    for k, v in colmap.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    if "tumor_size_cm" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must include tumor_size_cm (or Tumor_Size_cm)")

    # Order by month if available
    if "month_index" in df.columns:
        df = df.sort_values(by=["month_index"]).reset_index(drop=True)
        months = [f"Month {int(m)}" for m in df["month_index"].fillna(0).astype(int).tolist()]
    else:
        months = [f"Month {i}" for i in range(len(df))]

    sizes_series = df["tumor_size_cm"].astype(float)
    sizes_series = sizes_series.fillna(method="ffill").fillna(method="bfill")
    sizes = sizes_series.tolist()
    # Optional stage from CSV
    stage_value = None
    if "stage" in df.columns:
        try:
            # Use the most common or first non-null stage
            stage_value = df["stage"].dropna().astype(str).iloc[0] if df["stage"].dropna().size > 0 else None
        except Exception:
            stage_value = None

    # Build evolution from historical CSV data
    treatment = req.treatment or "chemo"
    evolution = []
    for i, (m, sz) in enumerate(zip(months, sizes)):
        survival = 100 - i * 2 + (5 if treatment == "combined" else 0)
        evolution.append({
            "month": m,
            "tumorSize": round(max(0.1, float(sz)), 2),
            "survivalProb": round(min(100.0, max(40.0, float(survival))), 1),
        })

    # Extend evolution with ML predictions if model available (or train on the fly)
    model = load_model("artifacts") if load_model else None
    if model is None and train_and_save is not None:
        try:
            # Train using all CSVs currently under data/
            train_and_save("data", "artifacts", lookback=3, epochs=10, batch_size=16)
            model = load_model("artifacts") if load_model else None
        except Exception:
            model = None
    if model is not None and predict_trajectory is not None and len(sizes) > 0:
        horizon = 12
        start_size = float(sizes[-1])
        preds = predict_trajectory(model, start_size=start_size, months=horizon, lookback=3)
        # Skip the first element if it's essentially the start point duplicated
        pred_sizes = preds[1:] if len(preds) > 1 else preds
        # Determine numeric month for continuation
        next_index_base = 0
        try:
            if "month_index" in df.columns:
                next_index_base = int(df["month_index"].dropna().astype(int).max())
            else:
                next_index_base = len(sizes) - 1
        except Exception:
            next_index_base = len(sizes) - 1
        for j, sz in enumerate(pred_sizes, start=1):
            idx = next_index_base + j
            survival = 100 - idx * 2 + (5 if treatment == "combined" else 0)
            evolution.append({
                "month": f"Month {idx}",
                "tumorSize": round(max(0.1, float(sz)), 2),
                "survivalProb": round(min(100.0, max(40.0, float(survival))), 1),
            })

    # Enhanced risk factors based on the CSV and number of images
    baseline = max(0.1, sizes[0])
    last = max(0.1, sizes[-1])
    growth_ratio = last / baseline
    months_count = max(1, len(sizes) - 1)
    growth_rate_pm = (last - baseline) / months_count

    # Response distribution (if present)
    response_mix = None
    response_impact = 0
    if "response" in df.columns:
        counts = df["response"].dropna().astype(str).str.lower().value_counts().to_dict()
        total_r = sum(counts.values()) or 1
        good_like = (counts.get("excellent", 0) + counts.get("good", 0)) / total_r
        response_impact = int((1 - good_like) * 100)
        response_mix = {k: int(v) for k, v in counts.items()}

    # Stage severity (if present)
    stage_severity = 0
    stage_str = None
    if stage_value:
        stage_str = str(stage_value)
        # crude parse: look for T[1-4]
        import re
        m = re.search(r"T(\d)", stage_str.upper())
        if m:
            t_num = int(m.group(1))
            stage_severity = min(100, 25 * max(0, t_num - 1))

    risk_factors = [
        {"factor": "Tumor Size Trend", "impact": min(100, int(growth_ratio * 50)), "description": "Relative growth from first to last"},
        {"factor": "Growth Rate (/mo)", "impact": min(100, int(abs(growth_rate_pm) * 30)), "description": "Absolute monthly growth magnitude"},
    ]
    if stage_str is not None:
        risk_factors.append({"factor": "Stage Severity", "impact": stage_severity, "description": f"Stage parsed from CSV: {stage_str}"})
    if response_mix is not None:
        risk_factors.append({"factor": "Treatment Response Mix", "impact": response_impact, "description": f"Distribution: {response_mix}"})

    # Detailed levels
    def to_level(score: int) -> str:
        return "High" if score >= 70 else ("Medium" if score >= 40 else "Low")

    risk_details = [
        {"factor": rf["factor"], "score": rf["impact"], "level": to_level(int(rf["impact"])), "description": rf.get("description", "")}
        for rf in risk_factors
    ]

    overall_score = int(np.mean([int(rf["impact"]) for rf in risk_factors]) if risk_factors else 0)
    overall_risk = to_level(overall_score)

    return {
        "evolution": evolution,
        "riskFactors": risk_factors,
        "treatmentImpact": 92 if treatment == "combined" else (78 if treatment == "chemo" else 74),
        "confidence": round(min(0.98, 0.65 + 0.02 * len(evolution) + (0.05 if model is not None else 0)), 2),
        "stage": stage_value,
        "riskDetails": risk_details,
        "overallRisk": overall_risk,
    }


@app.post("/predict")
def predict(state: PatientState):
    # If a trained TF model exists, use it; otherwise use demo logic
    treatment = state.treatment or "chemo"
    start_size = 2.3
    model = load_model("artifacts") if load_model else None
    if model is not None:
        sizes = predict_trajectory(model, start_size=start_size, months=12, lookback=3)
        evolution = []
        for m in range(1, 13):  # Months 1-12
            idx = m - 1  # Index into sizes array (0-based)
            if idx < len(sizes):
                size = sizes[idx]
            else:
                size = start_size * np.exp(-0.15 * idx)
            survival = 100 - (m - 1) * 2 + (5 if treatment == "combined" else 0)
            evolution.append({
                "month": f"Month {m}",
                "tumorSize": round(max(0.1, float(size)), 2),
                "survivalProb": round(min(100.0, max(60.0, float(survival))), 1),
            })
    else:
        base_growth = -0.15 if treatment == "chemo" else (-0.12 if treatment == "radiation" else -0.18)
        evolution = []
        for m in range(1, 13):  # Months 1-12
            size = start_size * np.exp(base_growth * (m - 1))
            survival = 100 - (m - 1) * 2 + (5 if treatment == "combined" else 0)
            evolution.append({
                "month": f"Month {m}",
                "tumorSize": round(max(0.1, float(size)), 2),
                "survivalProb": round(min(100.0, max(60.0, float(survival))), 1),
            })
    return {
        "evolution": evolution,
        "riskFactors": [
            {"factor": "Age", "impact": 65, "description": "Moderate risk factor"},
            {"factor": "Tumor Stage", "impact": 75, "description": "Significant concern"},
            {"factor": "Location", "impact": 70, "description": "Accessible for treatment"},
            {"factor": "Lifestyle", "impact": 55, "description": "Improving factors"},
            {"factor": "Response Rate", "impact": 82, "description": "Good prognosis"},
        ],
        "treatmentImpact": 92 if treatment == "combined" else (78 if treatment == "chemo" else 74),
        "confidence": 0.87,
    }


class ExplainRequest(BaseModel):
    question: Optional[str] = None
    prediction: dict


@app.post("/explain")
def explain(req: ExplainRequest):
    # Placeholder: return deterministic explanation; integrate Gemini later
    q = (req.question or "").lower()
    if "treatment" in q:
        text = "Combined therapy shows the highest predicted impact due to complementary mechanisms."
    elif "risk" in q:
        text = "Stage and tumor location contribute most to risk, with age as a moderate factor."
    elif "survival" in q:
        text = "Projected survival remains above 85% across the next year under continued therapy."
    else:
        text = "Predictions indicate a gradual reduction in tumor size with associated survival stability."
    return {"answer": text}


# ================= RL Agent Instance ================= #
rl_agent = TumorRLAgent() if TumorRLAgent else None


@app.post("/train")
def train_model():
    data_dir = "data"
    os.makedirs("artifacts", exist_ok=True)
    if train_and_save is None:
        raise HTTPException(status_code=500, detail="TensorFlow not available")
    try:
        stats = train_and_save(data_dir, "artifacts", lookback=3, epochs=20, batch_size=16)
        return stats
    except Exception as e:
        # Return a readable error instead of silent 500s
        raise HTTPException(status_code=500, detail=f"TRAIN_ERROR: {type(e).__name__}: {e}")


@app.post("/export")
def export_to_sheets_like(payload: dict):
    """Return CSV text for easy import into Google Sheets.
    If no prediction is provided, generate one using current model and optional treatment.
    """
    prediction = payload.get("prediction") if isinstance(payload, dict) else None
    evol = prediction.get("evolution", []) if isinstance(prediction, dict) else []
    if not evol:
        # Fallback: generate from server using model and provided treatment
        try:
            treatment = payload.get("treatment", "chemo") if isinstance(payload, dict) else "chemo"
            pred = predict(PatientState(treatment=treatment))
            evol = pred.get("evolution", [])
        except Exception:
            evol = []
    buf = io.StringIO()
    buf.write("Month,Tumor Size (cm),Survival %\n")
    for row in evol:
        buf.write(f"{row.get('month')},{row.get('tumorSize')},{row.get('survivalProb')}\n")
    return {"csv": buf.getvalue()}


# ================= RL Endpoints ================= #

@app.post("/rl/train")
def train_rl_agent():
    """Train the RL agent with simulated episodes"""
    try:
        if not rl_agent:
            raise HTTPException(status_code=500, detail="RL agent not available")
        
        # Simulate training episodes
        episodes = 100
        total_rewards = []
        
        for episode in range(episodes):
            # Create random patient state
            initial_state = RLPatientState(
                tumor_size=random.uniform(1.0, 5.0),
                age=random.randint(30, 80),
                stage=random.choice(['T1', 'T2', 'T3', 'T4']),
                treatment_history=[],
                months_elapsed=0,
                qol_score=random.uniform(0.3, 0.8),
                toxicity_level=random.uniform(0.0, 0.3),
                resistance_risk=random.uniform(0.0, 0.2)
            )
            
            # Run episode
            rewards, actions = rl_agent.simulate_treatment_episode(initial_state, max_months=12)
            total_rewards.append(sum(rewards))
            
            # Decay exploration
            rl_agent.decay_epsilon()
        
        avg_reward = np.mean(total_rewards)
        return {
            "episodes": episodes,
            "avg_reward": float(avg_reward),
            "epsilon": rl_agent.epsilon,
            "q_table_size": len(rl_agent.q_table)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RL_TRAIN_ERROR: {type(e).__name__}: {e}")


@app.post("/rl/optimize")
def optimize_treatment(patient_data: dict):
    """Get optimal treatment plan for a specific patient"""
    if not rl_agent:
        raise HTTPException(status_code=500, detail="RL agent not available")
    
    # Convert patient data to RLPatientState
    initial_state = RLPatientState(
        tumor_size=float(patient_data.get("tumor_size", 2.3)),
        age=int(patient_data.get("age", 58)),
        stage=patient_data.get("stage", "T2"),
        treatment_history=patient_data.get("treatment_history", []),
        months_elapsed=int(patient_data.get("months_elapsed", 0)),
        qol_score=float(patient_data.get("qol_score", 0.5)),
        toxicity_level=float(patient_data.get("toxicity_level", 0.0)),
        resistance_risk=float(patient_data.get("resistance_risk", 0.0))
    )
    
    # Get optimal treatment plan
    optimal_plan = rl_agent.get_optimal_treatment_plan(initial_state, horizon_months=12)
    
    # Convert to API response format
    plan_data = []
    for i, action in enumerate(optimal_plan):
        plan_data.append({
            "month": i + 1,
            "treatment": action.treatment_type,
            "intensity": action.intensity,
            "duration_months": action.duration_months,
            "expected_reward": rl_agent.q_table.get(
                rl_agent._state_to_key(initial_state), {}
            ).get(rl_agent._action_to_key(action), 0.0)
        })
    
    return {
        "optimal_plan": plan_data,
        "total_months": len(optimal_plan),
        "agent_confidence": 1.0 - rl_agent.epsilon
    }


@app.get("/rl/status")
def get_rl_status():
    """Get RL agent status and statistics"""
    if not rl_agent:
        return {"status": "not_available"}
    
    return {
        "status": "active",
        "epsilon": rl_agent.epsilon,
        "q_table_size": len(rl_agent.q_table),
        "learning_rate": rl_agent.learning_rate,
        "discount_factor": rl_agent.discount_factor
    }


# ================= Database Endpoints ================= #

@app.get("/patients")
def get_patients(db: Session = Depends(get_db)):
    """Get all patients from database"""
    patients = db.query(Patient).all()
    return [
        {
            "patient_id": p.patient_id,
            "age": p.age,
            "gender": p.gender,
            "stage_tnm": p.stage_tnm,
            "initial_tumor_size_cm": p.initial_tumor_size_cm,
            "smoking_status": p.smoking_status,
            "hpv_status": p.hpv_status,
            "comorbidities": p.comorbidities,
            "created_at": p.created_at
        }
        for p in patients
    ]


@app.get("/patients/{patient_id}")
def get_patient(patient_id: str, db: Session = Depends(get_db)):
    """Get specific patient with follow-up data"""
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    followups = db.query(PatientFollowup).filter(
        PatientFollowup.patient_id == patient_id
    ).order_by(PatientFollowup.follow_up_month).all()
    
    predictions = db.query(Prediction).filter(
        Prediction.patient_id == patient_id
    ).order_by(Prediction.prediction_date.desc()).all()
    
    return {
        "patient": {
            "patient_id": patient.patient_id,
            "age": patient.age,
            "gender": patient.gender,
            "stage_tnm": patient.stage_tnm,
            "initial_tumor_size_cm": patient.initial_tumor_size_cm,
            "smoking_status": patient.smoking_status,
            "alcohol_use": patient.alcohol_use,
            "oral_hygiene": patient.oral_hygiene,
            "hpv_status": patient.hpv_status,
            "comorbidities": patient.comorbidities,
            "created_at": patient.created_at
        },
        "followups": [
            {
                "month": f.follow_up_month,
                "tumor_size_cm": f.tumor_size_cm,
                "recurrence": f.recurrence,
                "treatment_type": f.treatment_type,
                "response_to_treatment": f.response_to_treatment,
                "follow_up_date": f.follow_up_date,
                "notes": f.notes
            }
            for f in followups
        ],
        "predictions": [
            {
                "id": p.id,
                "treatment_type": p.treatment_type,
                "predicted_evolution": p.predicted_evolution,
                "risk_factors": p.risk_factors,
                "treatment_impact": p.treatment_impact,
                "confidence": p.confidence,
                "model_version": p.model_version,
                "prediction_date": p.prediction_date
            }
            for p in predictions
        ]
    }


@app.post("/patients")
def create_patient(patient_data: dict, db: Session = Depends(get_db)):
    """Create a new patient"""
    try:
        patient = Patient(
            patient_id=patient_data["patient_id"],
            age=patient_data["age"],
            gender=patient_data["gender"],
            stage_tnm=patient_data["stage_tnm"],
            initial_tumor_size_cm=patient_data["initial_tumor_size_cm"],
            smoking_status=patient_data["smoking_status"],
            alcohol_use=patient_data["alcohol_use"],
            oral_hygiene=patient_data["oral_hygiene"],
            hpv_status=patient_data["hpv_status"],
            comorbidities=patient_data.get("comorbidities", "")
        )
        db.add(patient)
        db.commit()
        return {"message": "Patient created successfully", "patient_id": patient.patient_id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error creating patient: {str(e)}")


@app.get("/analytics/summary")
def get_analytics_summary(db: Session = Depends(get_db)):
    """Get analytics summary from database"""
    total_patients = db.query(Patient).count()
    total_predictions = db.query(Prediction).count()
    total_followups = db.query(PatientFollowup).count()
    
    # Get treatment effectiveness
    treatment_stats = db.query(
        PatientFollowup.treatment_type,
        PatientFollowup.response_to_treatment
    ).all()
    
    treatment_effectiveness = {}
    for treatment, response in treatment_stats:
        if treatment not in treatment_effectiveness:
            treatment_effectiveness[treatment] = {}
        if response not in treatment_effectiveness[treatment]:
            treatment_effectiveness[treatment][response] = 0
        treatment_effectiveness[treatment][response] += 1
    
    return {
        "total_patients": total_patients,
        "total_predictions": total_predictions,
        "total_followups": total_followups,
        "treatment_effectiveness": treatment_effectiveness
    }


@app.get("/analytics/patient-trends")
def get_patient_trends(db: Session = Depends(get_db)):
    """Get patient trend analysis"""
    # Get monthly tumor size trends for each patient
    patients = db.query(Patient).all()
    trends = []
    
    for patient in patients:
        followups = db.query(PatientFollowup).filter(
            PatientFollowup.patient_id == patient.patient_id
        ).order_by(PatientFollowup.follow_up_month).all()
        
        if followups:
            trend_data = {
                "patient_id": patient.patient_id,
                "stage": patient.stage_tnm,
                "tumor_evolution": [
                    {
                        "month": f.follow_up_month,
                        "tumor_size_cm": f.tumor_size_cm,
                        "recurrence": f.recurrence
                    }
                    for f in followups
                ]
            }
            trends.append(trend_data)
    
    return {"patient_trends": trends}


# User Management Endpoints
@app.post("/users")
def create_user(user_data: dict, db: Session = Depends(get_db)):
    """Create a new user"""
    try:
        user = User(
            user_id=user_data["user_id"],
            username=user_data["username"],
            email=user_data["email"],
            full_name=user_data["full_name"],
            role=user_data["role"],
            specialization=user_data.get("specialization"),
            license_number=user_data.get("license_number"),
            hospital_affiliation=user_data.get("hospital_affiliation")
        )
        db.add(user)
        db.commit()
        return {"message": "User created successfully", "user_id": user.user_id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error creating user: {str(e)}")

@app.get("/users/{user_id}")
def get_user(user_id: str, db: Session = Depends(get_db)):
    """Get user profile and assigned patients"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get assigned patients
    assignments = db.query(PatientAssignment).filter(
        PatientAssignment.user_id == user.id,
        PatientAssignment.is_active == True
    ).all()
    
    assigned_patients = []
    for assignment in assignments:
        patient = db.query(Patient).filter(Patient.patient_id == assignment.patient_id).first()
        if patient:
            assigned_patients.append({
                "patient_id": patient.patient_id,
                "age": patient.age,
                "gender": patient.gender,
                "stage_tnm": patient.stage_tnm,
                "initial_tumor_size_cm": patient.initial_tumor_size_cm,
                "assignment_type": assignment.assignment_type,
                "assigned_at": assignment.assigned_at
            })
    
    return {
        "user": {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "specialization": user.specialization,
            "license_number": user.license_number,
            "hospital_affiliation": user.hospital_affiliation,
            "is_active": user.is_active,
            "created_at": user.created_at
        },
        "assigned_patients": assigned_patients
    }

@app.get("/users/{user_id}/dashboard")
def get_user_dashboard(user_id: str, db: Session = Depends(get_db)):
    """Get user-specific dashboard data"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get assigned patients
    assignments = db.query(PatientAssignment).filter(
        PatientAssignment.user_id == user.id,
        PatientAssignment.is_active == True
    ).all()
    
    assigned_patient_ids = [a.patient_id for a in assignments]
    
    # Get patient statistics
    total_patients = len(assigned_patient_ids)
    recent_predictions = db.query(Prediction).filter(
        Prediction.patient_id.in_(assigned_patient_ids)
    ).order_by(Prediction.prediction_date.desc()).limit(5).all()
    
    # Get treatment effectiveness for assigned patients
    treatment_stats = {}
    for patient_id in assigned_patient_ids:
        followups = db.query(PatientFollowup).filter(
            PatientFollowup.patient_id == patient_id
        ).all()
        
        for followup in followups:
            treatment = followup.treatment_type
            if treatment not in treatment_stats:
                treatment_stats[treatment] = {"total": 0, "excellent": 0, "good": 0, "fair": 0, "poor": 0}
            
            treatment_stats[treatment]["total"] += 1
            response = followup.response_to_treatment
            if response in treatment_stats[treatment]:
                treatment_stats[treatment][response.lower()] += 1
    
    # Calculate effectiveness percentages
    for treatment in treatment_stats:
        total = treatment_stats[treatment]["total"]
        if total > 0:
            excellent = treatment_stats[treatment]["excellent"]
            good = treatment_stats[treatment]["good"]
            treatment_stats[treatment]["effectiveness"] = round(((excellent + good) / total) * 100, 1)
    
    # Get recent activity
    recent_sessions = db.query(UserSession).filter(
        UserSession.user_id == user.id
    ).order_by(UserSession.created_at.desc()).limit(10).all()
    
    return {
        "user_profile": {
            "full_name": user.full_name,
            "role": user.role,
            "specialization": user.specialization,
            "hospital_affiliation": user.hospital_affiliation
        },
        "statistics": {
            "total_patients": total_patients,
            "recent_predictions": len(recent_predictions),
            "treatment_effectiveness": treatment_stats
        },
        "recent_predictions": [
            {
                "patient_id": p.patient_id,
                "treatment_type": p.treatment_type,
                "confidence": p.confidence,
                "treatment_impact": p.treatment_impact,
                "prediction_date": p.prediction_date
            }
            for p in recent_predictions
        ],
        "recent_activity": [
            {
                "action_type": s.action_type,
                "patient_id": s.patient_id,
                "created_at": s.created_at
            }
            for s in recent_sessions
        ]
    }

@app.get("/users/{user_id}/patients/{patient_id}/report")
def get_patient_report(user_id: str, patient_id: str, db: Session = Depends(get_db)):
    """Get comprehensive patient report for a specific user"""
    # Verify user has access to this patient
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    assignment = db.query(PatientAssignment).filter(
        PatientAssignment.user_id == user.id,
        PatientAssignment.patient_id == patient_id,
        PatientAssignment.is_active == True
    ).first()
    
    if not assignment and user.role != UserRoleEnum.ADMIN:
        raise HTTPException(status_code=403, detail="Access denied to this patient")
    
    # Get patient data
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Get comprehensive patient data
    followups = db.query(PatientFollowup).filter(
        PatientFollowup.patient_id == patient_id
    ).order_by(PatientFollowup.follow_up_month).all()
    
    predictions = db.query(Prediction).filter(
        Prediction.patient_id == patient_id
    ).order_by(Prediction.prediction_date.desc()).all()
    
    outcomes = db.query(TreatmentOutcome).filter(
        TreatmentOutcome.patient_id == patient_id
    ).order_by(TreatmentOutcome.outcome_date.desc()).all()
    
    # Calculate risk factors
    risk_factors = []
    if patient.age > 65:
        risk_factors.append({"factor": "Age", "level": "High", "description": f"Patient age {patient.age} increases risk"})
    if patient.smoking_status in ["Current Smoker", "Former Smoker"]:
        risk_factors.append({"factor": "Smoking History", "level": "High", "description": f"Smoking status: {patient.smoking_status}"})
    if patient.hpv_status == "Positive":
        risk_factors.append({"factor": "HPV Status", "level": "Medium", "description": "HPV positive - may affect treatment response"})
    if patient.oral_hygiene in ["Poor", "Very Poor"]:
        risk_factors.append({"factor": "Oral Hygiene", "level": "Medium", "description": f"Oral hygiene: {patient.oral_hygiene}"})
    
    return {
        "patient": {
            "patient_id": patient.patient_id,
            "age": patient.age,
            "gender": patient.gender,
            "stage_tnm": patient.stage_tnm,
            "initial_tumor_size_cm": patient.initial_tumor_size_cm,
            "smoking_status": patient.smoking_status,
            "alcohol_use": patient.alcohol_use,
            "oral_hygiene": patient.oral_hygiene,
            "hpv_status": patient.hpv_status,
            "comorbidities": patient.comorbidities,
            "created_at": patient.created_at
        },
        "followups": [
            {
                "month": f.follow_up_month,
                "tumor_size_cm": f.tumor_size_cm,
                "recurrence": f.recurrence,
                "treatment_type": f.treatment_type,
                "response_to_treatment": f.response_to_treatment,
                "follow_up_date": f.follow_up_date,
                "notes": f.notes
            }
            for f in followups
        ],
        "predictions": [
            {
                "id": p.id,
                "treatment_type": p.treatment_type,
                "predicted_evolution": p.predicted_evolution,
                "risk_factors": p.risk_factors,
                "treatment_impact": p.treatment_impact,
                "confidence": p.confidence,
                "model_version": p.model_version,
                "prediction_date": p.prediction_date
            }
            for p in predictions
        ],
        "outcomes": [
            {
                "actual_tumor_size_cm": o.actual_tumor_size_cm,
                "actual_survival_probability": o.actual_survival_probability,
                "actual_response": o.actual_response,
                "outcome_date": o.outcome_date,
                "notes": o.notes
            }
            for o in outcomes
        ],
        "risk_assessment": {
            "risk_factors": risk_factors,
            "overall_risk": "High" if len([rf for rf in risk_factors if rf["level"] == "High"]) > 1 else "Medium" if risk_factors else "Low"
        },
        "treatment_recommendations": {
            "primary_treatment": "Surgery + Chemotherapy + Radiation" if patient.stage_tnm.startswith("T3") or patient.stage_tnm.startswith("T4") else "Surgery + Radiation",
            "alternative_treatments": ["Chemotherapy + Radiation", "Surgery only"] if patient.stage_tnm.startswith("T1") or patient.stage_tnm.startswith("T2") else ["Palliative care"],
            "follow_up_schedule": "Every 3 months for first year, then every 6 months"
        }
    }

@app.post("/users/{user_id}/patients/{patient_id}/assign")
def assign_patient_to_user(user_id: str, patient_id: str, assignment_data: dict, db: Session = Depends(get_db)):
    """Assign a patient to a user"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Check if assignment already exists
    existing = db.query(PatientAssignment).filter(
        PatientAssignment.user_id == user.id,
        PatientAssignment.patient_id == patient_id,
        PatientAssignment.is_active == True
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Patient already assigned to this user")
    
    assignment = PatientAssignment(
        user_id=user.id,
        patient_id=patient_id,
        assignment_type=assignment_data.get("assignment_type", "primary")
    )
    
    db.add(assignment)
    db.commit()
    
    return {"message": "Patient assigned successfully"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


