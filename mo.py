# ✅ الكود النهائي لتطبيق Streamlit لتحليل وتوقع Churn 
# مع الحفاظ على الموديل الأصلي وإضافة visualizations و y-range ديناميكي

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import io


def get_y_range(df_grouped):
    min_val = df_grouped['churn'].min()
    max_val = df_grouped['churn'].max()
    return [min_val - 0.5, max_val + 0.5]

st.set_page_config(page_title="📈 Telecom Churn Dashboard", layout="wide")

st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: #f1f1f1;
        }
        h1, h2, h3 {
            color: #00c0f2;
        }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("📊 Telecom Churn")
    st.markdown("### Dashboard Navigation")
    st.info("Use the sections below to explore churn insights.")
    st.markdown("- Data Preview\n- Churn Summary\n- Visual Insights\n- Model Results\n- Live Prediction")

st.title("📊 Telecom Churn Analysis & Prediction")

# Load data
df = pd.read_csv("telecom_churn.csv")

# Data Preview
st.header("📂 1. Data Preview")
st.dataframe(df.head(), use_container_width=True)
buffer = io.StringIO()
df.info(buf=buffer)
st.text(buffer.getvalue())
st.write("### Missing Values")
st.write(df.isnull().sum())
st.write("### Numerical Stats")
st.dataframe(df.describe())
st.write("### Categorical Stats")
st.dataframe(df.describe(include=['object', 'category']))

# Clean invalid rows
# 🔍 Invalid Values Detection
invalid_rows = (df['calls_made'] < 0) | (df['sms_sent'] < 0) | (df['data_used'] < 0)
total_invalid = invalid_rows.sum()
percentage_invalid = (total_invalid / len(df)) * 100

st.write("### 🔍 Invalid Values Check")
st.write(f"Number of rows with negative values (calls, SMS, data): **{total_invalid}**")
st.write(f"Percentage of invalid rows: **{percentage_invalid:.2f}%**")

# 🧼 Remove invalid rows
df = df[(df['calls_made'] >= 0) & (df['sms_sent'] >= 0) & (df['data_used'] >= 0)].copy()

# Feature Engineering
df['date_of_registration'] = pd.to_datetime(df['date_of_registration'])
df['reg_year'] = df['date_of_registration'].dt.year
df['reg_month'] = df['date_of_registration'].dt.month
df['age_group'] = pd.cut(df['age'], bins=[18, 30, 45, 60, 75], labels=['18-30', '31-45', '46-60', '61-75'])
df['activity_score'] = df['calls_made'] + df['sms_sent'] + df['data_used']
df['activity_level'] = pd.qcut(df['activity_score'], 4, labels=['Low', 'Medium', 'High', 'Very High'])


def usage_type(row):
    if row['calls_made'] > 0 and row['data_used'] == 0:
        return 'Calls Only'
    elif row['calls_made'] == 0 and row['data_used'] > 0:
        return 'Data Only'
    elif row['calls_made'] > 0 and row['data_used'] > 0:
        return 'Mixed'
    else:
        return 'Inactive'


df['usage_type'] = df.apply(usage_type, axis=1)
df['salary_group'] = pd.qcut(df['estimated_salary'], 4, labels=['Low', 'Mid', 'High', 'Very High'])

# Churn Summary
st.header("📈 2. Churn Summary")
churn_rate = df['churn'].mean() * 100
st.metric("Churn Rate", f"{churn_rate:.2f}%")
labels = ['Not Churned', 'Churned']
values = df['churn'].value_counts().values
fig = px.pie(names=labels, values=values, title="📊 Customer Distribution by Churn Status", color_discrete_sequence=px.colors.qualitative.Pastel)
st.plotly_chart(fig, use_container_width=True)
fig2 = go.Figure([go.Bar(x=labels, y=values, marker_color=['green', 'red'])])
fig2.update_layout(title="📊 Number of Churned vs Non-Churned Customers", xaxis_title="Churn Status", yaxis_title="Number of Customers")
st.plotly_chart(fig2, use_container_width=True)

# Visual Insights
st.header("📊 3. Visual Insights")

with st.expander("🧍‍♂️ Demographics"):
    age_churn = df.groupby('age_group')['churn'].mean().reset_index()
    age_churn['churn'] *= 100
    fig = px.bar(age_churn, x='age_group', y='churn', color='churn', text_auto='.2f', title="Churn by Age Group")
    fig.update_layout(yaxis=dict(range=get_y_range(age_churn)))
    st.plotly_chart(fig)

    gender_churn = df.groupby('gender')['churn'].mean().reset_index()
    gender_churn['churn'] *= 100
    st.plotly_chart(px.pie(gender_churn, names='gender', values='churn', title="Churn by Gender", color_discrete_sequence=px.colors.qualitative.Set2))

    dep_churn = df.groupby('num_dependents')['churn'].mean().reset_index()
    dep_churn['churn'] *= 100
    fig = px.line(dep_churn, x='num_dependents', y='churn', markers=True, title="Churn by Number of Dependents")
    fig.update_layout(yaxis=dict(range=get_y_range(dep_churn)))
    st.plotly_chart(fig)

    salary_churn = df.groupby('salary_group')['churn'].mean().reset_index()
    salary_churn['churn'] *= 100
    fig = px.bar(salary_churn, x='salary_group', y='churn', color='churn', text_auto='.2f', title="Churn by Salary Group")
    fig.update_layout(yaxis=dict(range=get_y_range(salary_churn)))
    st.plotly_chart(fig)

with st.expander("📱 Usage Behavior"):
    activity_churn = df.groupby('activity_level')['churn'].mean().reset_index()
    activity_churn['churn'] *= 100
    fig = go.Figure([go.Bar(x=activity_churn['activity_level'], y=activity_churn['churn'], text=activity_churn['churn'].round(2), textposition='outside')])
    fig.update_layout(title="Churn by Activity Level", yaxis=dict(range=get_y_range(activity_churn)))
    st.plotly_chart(fig)

    usage_churn = df.groupby('usage_type')['churn'].mean().reset_index()
    usage_churn['churn'] *= 100
    fig = px.bar(usage_churn, x='usage_type', y='churn', color='churn', text_auto='.2f', title="Churn by Usage Type", color_continuous_scale='Plasma')
    fig.update_layout(yaxis=dict(range=get_y_range(usage_churn)))
    st.plotly_chart(fig)

with st.expander("🗺️ Geographical Insights"):
    partner_churn = df.groupby('telecom_partner')['churn'].mean().reset_index()
    partner_churn['churn'] *= 100
    fig = px.bar(
        partner_churn,
        x='telecom_partner',
        y='churn',
        text_auto='.2f',
        title="Churn by Telecom Partner",
        color='telecom_partner',  # ✅ لتلوين الأعمدة حسب اسم الشركة
        color_discrete_sequence=px.colors.qualitative.Set2  # ✅ مجموعة ألوان لطيفة
    )
    fig.update_layout(yaxis=dict(range=get_y_range(partner_churn)))
    st.plotly_chart(fig)

    state_churn = df.groupby('state')['churn'].mean().reset_index()
    state_churn['churn'] *= 100
    top_states = state_churn.sort_values(by='churn', ascending=False).head(10)
    fig = px.bar(
        top_states,
        x='state',
        y='churn',
        text_auto='.2f',
        title="Top 10 States with Highest Churn",
        color='state',  # ✅ لتلوين الأعمدة حسب اسم الولاية
        color_discrete_sequence=px.colors.qualitative.Set3  # ✅ ألوان مختلفة عن الشكل الأول
    )
    fig.update_layout(yaxis=dict(range=get_y_range(top_states)))
    st.plotly_chart(fig)


    sp_churn = df.groupby(['state', 'telecom_partner'])['churn'].mean().reset_index()
    sp_churn['churn'] *= 100
    filtered = sp_churn[sp_churn['state'].isin(top_states['state'])]
    fig = px.bar(filtered, x='state', y='churn', color='telecom_partner', barmode='group', text_auto='.2f', title="Churn by Partner in Top 10 States")
    fig.update_layout(yaxis=dict(range=get_y_range(filtered)))
    st.plotly_chart(fig)

with st.expander("📆 Time-Based Trends"):
    year_churn = df.groupby('reg_year')['churn'].mean().reset_index()
    year_churn['churn'] *= 100
    fig = px.bar(year_churn, x='reg_year', y='churn', text_auto='.2f', title="Churn by Registration Year", color='churn')
    fig.update_layout(yaxis=dict(range=get_y_range(year_churn)))
    st.plotly_chart(fig)

    month_churn = df.groupby('reg_month')['churn'].mean().reset_index()
    month_churn['churn'] *= 100
    fig = px.line(month_churn, x='reg_month', y='churn', markers=True, title="Churn by Registration Month")
    fig.update_layout(yaxis=dict(range=get_y_range(month_churn)))
    st.plotly_chart(fig)



# Model
st.header("🤖 4. Churn Prediction Model")
features = ['age', 'num_dependents', 'estimated_salary', 'calls_made', 'sms_sent', 'data_used', 'gender', 'telecom_partner']
df_model = df[features + ['churn']].copy()
le_gender = LabelEncoder()
le_partner = LabelEncoder()
df_model['gender'] = le_gender.fit_transform(df_model['gender'])
df_model['telecom_partner'] = le_partner.fit_transform(df_model['telecom_partner'])
X = df_model.drop('churn', axis=1)
y = df_model['churn']
ros = RandomOverSampler()
X_bal, y_bal = ros.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=2529)
rfc = RandomForestClassifier(random_state=2529)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
st.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Live Prediction
# Live Prediction based on the input
st.header("🔮 5. Live Churn Prediction")
with st.form("prediction_form"):
    st.subheader("📋 Customer Information")
    age = st.slider("Age", 18, 100, 30)
    num_dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3, 4])
    estimated_salary = st.number_input("Estimated Salary", min_value=0, max_value=500000, value=60000)
    calls_made = st.slider("Calls Made per Month", 0, 200, 50)
    sms_sent = st.slider("SMS Sent per Month", 0, 200, 20)
    data_used = st.slider("Data Used (GB)", 0.0, 100.0, 5.0)
    gender_display = st.selectbox("Gender", ["Male", "Female"])
    gender = "M" if gender_display == "Male" else "F"
    telecom_partner_display = st.selectbox("Telecom Partner", df['telecom_partner'].unique())
    submitted = st.form_submit_button("Predict")

if submitted:
    # Data Preprocessing
    gender_encoded = le_gender.transform([gender])[0]
    partner_encoded = le_partner.transform([telecom_partner_display])[0]
    user_input = pd.DataFrame([[age, num_dependents, estimated_salary, calls_made, sms_sent, data_used, gender_encoded, partner_encoded]], columns=features)
    
    # Prediction
    prediction = rfc.predict(user_input)[0]
    result = "❌ Will Churn" if prediction == 1 else "✅ Will Stay"
    st.success(f"Prediction Result: {result}")
    
    # Recommending Plans Based on Age, Usage, and Dependents
    if age >= 18 and age <= 25:
        if data_used >= 8 :
            st.info("🎁 Recommended Plan: 10 GB+ data with Unlimited calls! Perfect for high usage!")
        else:
            st.info("📱 Recommended Plan: 6 GB data with 200 calls. Ideal for moderate usage!")
    elif age > 25 and age <= 45:
        if num_dependents > 1:
            if data_used >= 5 and calls_made >= 150:
                st.info("🎉 Family Plan: 10GB+ data and Unlimited calls for high usage!")
            else:
                st.info("🎉 Family Plan: 5GB data with 100 calls. Ideal for family users!")
        else:
            if data_used >= 8 and calls_made >= 150:
                st.info("📱 Recommended Plan: 8GB+ data and Unlimited calls!")
            else:
                st.info("📱 Recommended Plan: 5GB data with 100 calls. Perfect for moderate users!")
    elif age >= 45 and age <= 60:
        if calls_made >= 150:
            st.info("📞 Recommended Plan: Unlimited calls with 5GB data. Tailored for frequent callers!")
        else:
            st.info("📱 Recommended Plan: 3GB data with 100 calls. Suitable for moderate users!")
    elif age >= 61:
        st.info("📞 Recommended Plan: Calls Only Plan with 2GB data. Best for senior citizens who prefer voice calls.")

    # Check Salary Group and Recommend Premium or Standard Plan
    if estimated_salary > 50000:
        st.info("💼 Premium Plan: Get exclusive high-end plans with additional benefits!")
    else:
        st.info("💵 Standard Plan: Affordable plans with good value for money.")

    # Check Telecom Partner for Specific Offers
    if telecom_partner_display == "Airtel":
        st.info("🎁 Special Offer: Exclusive plans for Airtel users! Get free compensatory plans!")

#streamlit run mo.py