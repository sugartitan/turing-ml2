from fastapi import FastAPI
import pickle
import imblearn
import pandas as pd


def categorize_age(age: float) -> str:
    '''
    Categorizes age value into one of these categories:
    - child
    - adult
    - elder
    '''
    if age < 18:
        return 'child'
    elif age < 65:
        return 'adult'
    
    return 'elder'


def categorize_bmi(bmi: float) -> str:
    '''
    Categorizes BMI value into one of these categories:
    - underweight
    - normal
    - overweight
    - obese
    - extremely obese
    '''
    if bmi < 18.5:
        return 'underweight'
    elif bmi < 25:
        return 'normal'
    elif bmi < 30:
        return 'overweight'
    elif bmi < 35:
        return 'obese'
    
    return 'extremely obese'


def categorize_glucose(avg_glucose_level: float) -> str:
    '''
    Categorizes glucose value into one of these categories:
    - low
    - normal
    - prediabetes
    - diabetes
    '''
    if avg_glucose_level < 70:
        return 'low'
    elif avg_glucose_level < 100:
        return 'normal'
    elif avg_glucose_level < 125:
        return 'prediabetes'
    
    return 'diabetes'

loaded_model = pickle.load(open('./app/model.pkl', 'rb'))

app = FastAPI()

@app.post("/predict")
def predict(gender: str, age: int, hypertension: int, heart_disease: int, ever_married: int, work_type: str, Residence_type: str, avg_glucose_level: float, bmi: float, smoking_status: str) -> str:
    '''
    Values for:
    gender - 'Male', 'Female' or 'Other'
    hypertension - 1 (Yes) or 0 (No)
    heart_disease - 1 (Yes) or 0 (No)
    ever_married - 1 (Yes) or 0 (No)
    work_type - 'Private', 'Self-employed', 'Govt_job', 'children' or 'Never_worked'
    Residence_type - 'Urban' or 'Rural'
    smoking_status - 'formerly smoked', 'never smoked', 'smokes' or 'Unknown'
    '''

    Residence_type = 1 if 'Urban' else 0

    data = [
        gender,
        age,
        hypertension,
        heart_disease,
        ever_married,
        work_type, 
        Residence_type,
        avg_glucose_level,
        bmi,
        smoking_status,
        categorize_age(age),
        categorize_bmi(bmi),
        categorize_glucose(avg_glucose_level)
    ]

    cols = [
        'gender',
        'age',
        'hypertension',
        'heart_disease',
        'ever_married',
        'work_type',
        'Residence_type',
        'avg_glucose_level',
        'bmi',
        'smoking_status',
        'age_category',
        'bmi_category',
        'avg_glucose_level_category'
    ]

    df = pd.DataFrame(columns=cols)
    df.loc[0] = data

    return 'stroke' if loaded_model.predict(df)[0] else 'no stroke'


# http://127.0.0.1/predict?gender=Male&age=21&hypertension=0&heart_disease=0&ever_married=0&work_type=Private&Residence_type=Urban&avg_glucose_level=80&bmi=25&smoking_status=smokes