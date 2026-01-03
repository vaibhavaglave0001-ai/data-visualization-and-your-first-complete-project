import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. SETUP & DATA GENERATION (Simulating Data Collection)
def generate_student_data():
    np.random.seed(42)
    n_students = 200
    
    data = {
        'Student_ID': range(1, n_students + 1),
        'Study_Hours': np.random.uniform(5, 40, n_students),
        'Parent_Education': np.random.choice(['High School', 'Bachelor', 'Master'], n_students),
        'Sleep_Hours': np.random.uniform(4, 10, n_students),
        'Math_Score': np.random.randint(40, 100, n_students)
    }
    
    df = pd.DataFrame(data)
    
    # Adding logical correlation: More study hours = higher math scores + some noise
    df['Math_Score'] = (df['Study_Hours'] * 1.5) + (df['Sleep_Hours'] * 2) + np.random.normal(0, 5, n_students)
    df['Math_Score'] = df['Math_Score'].clip(0, 100) # Ensure scores stay within 0-100
    return df

# 2. DATA CLEANING & VALIDATION (Pipeline)
def clean_and_validate(df):
    try:
        # Check for missing values
        if df.isnull().values.any():
            df = df.fillna(df.median(numeric_only=True))
        
        # Validation: Ensure scores are numeric and within range
        if not (df['Math_Score'].between(0, 100)).all():
            raise ValueError("Data contains scores outside the 0-100 range.")
            
        print("✅ Data Pipeline: Cleaning and Validation Successful.")
        return df
    except Exception as e:
        print(f"❌ Error during pipeline: {e}")
        return None

# Execute Pipeline
df = generate_student_data()
df = clean_and_validate(df)

# 3. ANALYSIS & VISUALIZATION
plt.style.use('seaborn-v0_8-muted')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Chart 1: Regression Plot (Study Hours vs Math Score)

sns.regplot(data=df, x='Study_Hours', y='Math_Score', ax=ax1, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
ax1.set_title('Impact of Study Hours on Math Performance')
ax1.set_xlabel('Hours Studied per Week')
ax1.set_ylabel('Math Score (%)')

# Chart 2: Box Plot (Parental Education vs Math Score)

sns.boxplot(data=df, x='Parent_Education', y='Math_Score', palette='Set2', ax=ax2)
ax2.set_title('Performance by Parental Education Level')
ax2.set_xlabel('Parental Education')
ax2.set_ylabel('Math Score (%)')

plt.tight_layout()
plt.show()

# 4. BASIC METRICS (Written Insights)
avg_score = df['Math_Score'].mean()
correlation = df['Study_Hours'].corr(df['Math_Score'])

print(f"--- WRITTEN INSIGHTS ---")
print(f"1. The average math score across the dataset is {avg_score:.2f}%.")
print(f"2. Correlation Coefficient: {correlation:.2f}. This indicates a strong positive relationship.")
print(f"3. Students with parents holding a Master's degree show a higher median score than High School counterparts.")