import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="What's My Salary?", page_icon="💶", layout="wide")

# ── PersonalFav-style theme ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', system-ui, sans-serif;
}

.stApp {
    background-color: #080c10;
}

[data-testid="stSidebar"] {
    background-color: #060910;
    border-right: 1px solid #141920;
}

[data-testid="stSidebar"] .stMarkdown p {
    color: #4a5568;
}

.sidebar-title {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #c8a96e;
    margin-bottom: 2px;
}

.sidebar-sub {
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #2a3040;
    margin-bottom: 24px;
}

.hero-title {
    font-size: 42px;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #e8e8e8;
    margin: 24px 0 4px 0;
}

.hero-sub {
    font-size: 13px;
    color: #3a4a5a;
    letter-spacing: 0.05em;
    margin-bottom: 32px;
    font-family: 'DM Mono', monospace;
}

.metric-card {
    background: #0c1018;
    border: 1px solid #141920;
    border-radius: 4px;
    padding: 22px 24px;
    position: relative;
    height: 100%;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 2px; height: 100%;
    background: #c8a96e;
    border-radius: 4px 0 0 4px;
    opacity: 0.4;
}

.metric-label {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #2e3a4a;
    margin-bottom: 12px;
    font-family: 'DM Mono', monospace;
}

.metric-value {
    font-size: 22px;
    font-weight: 700;
    color: #d4c5a9;
    letter-spacing: -0.02em;
    font-family: 'DM Mono', monospace;
}

.section-label {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #2a3040;
    margin: 40px 0 6px 0;
    font-family: 'DM Mono', monospace;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #c8a96e;
    letter-spacing: -0.01em;
    margin-bottom: 4px;
}

.section-caption {
    font-size: 11px;
    color: #2e3a4a;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.03em;
    margin-bottom: 16px;
}

.divider {
    border: none;
    border-top: 1px solid #0f141a;
    margin: 40px 0;
}

/* Predict button */
.stButton button {
    background-color: #c8a96e;
    color: #080c10;
    border: none;
    border-radius: 4px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 12px;
    width: 100%;
}

.stButton button:hover {
    background-color: #d4b87a;
}

/* Selectbox & slider labels */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    color: #2e3a4a !important;
    font-size: 9px !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    font-family: 'DM Mono', monospace !important;
}

/* Slider accent */
div[data-testid="stSlider"] [data-testid="stSliderThumb"] {
    background-color: #c8a96e;
}

/* Success box */
div[data-testid="stAlert"] {
    background-color: #0c1018;
    border: 1px solid #c8a96e;
    border-radius: 4px;
    color: #c8a96e;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
}

/* Multiselect tags */
span[data-baseweb="tag"] {
    background-color: rgba(200,169,110,0.12) !important;
    border: 1px solid rgba(200,169,110,0.25) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Shared chart config ───────────────────────────────────────────────────────
GOLD_PALETTE = ['#c8a96e', '#4a7c59', '#3a5a7c', '#7c5a3a', '#5a4a7c', '#7c3a5a']

CHART_LAYOUT = dict(
    paper_bgcolor='#080c10',
    plot_bgcolor='#0c1018',
    font=dict(family='DM Mono, monospace', color='#2a3a4a', size=11),
    title_font=dict(color='#c8a96e', family='Space Grotesk', size=16),
    xaxis=dict(
        gridcolor='#0f141a', zeroline=False, showline=False,
        tickfont=dict(color='#2a3a4a')
    ),
    yaxis=dict(
        gridcolor='#0f141a', zeroline=False, showline=False,
        tickfont=dict(color='#2a3a4a')
    ),
    legend=dict(
        bgcolor='rgba(0,0,0,0)',
        font=dict(color='#3a4a5a', size=11)
    ),
    margin=dict(l=0, r=0, t=48, b=0),
    hovermode='x unified',
    hoverlabel=dict(
        bgcolor='#0c1018',
        font=dict(color='#d4c5a9', family='DM Mono')
    ),
)

# ── Helper functions ──────────────────────────────────────────────────────────
def metric_card(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def section(tag, title, caption):
    st.markdown(f'<div class="section-label">{tag}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">What\'s My Salary?</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Global salary intelligence · 250,000 data points · ML-powered prediction</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('the_dataset.csv')

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Salary Intel</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Dashboard Filters</div>', unsafe_allow_html=True)

    selected_locations = st.multiselect(
        "Locations", options=df['location'].unique(), default=df['location'].unique()
    )
    selected_job_title = st.multiselect(
        "Job Titles", options=df['job_title'].unique(), default=df['job_title'].unique()
    )
    selected_industry = st.multiselect(
        "Industries", options=df['industry'].unique(), default=df['industry'].unique()
    )
    selected_remote_work = st.multiselect(
        "Remote Work", options=df['remote_work'].unique(), default=df['remote_work'].unique()
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#1a2030;font-size:10px;letter-spacing:0.15em;text-transform:uppercase;font-family:DM Mono,monospace;">Random Forest · 250K records</p>',
        unsafe_allow_html=True
    )

filtered_df = df[
    (df['location'].isin(selected_locations)) &
    (df['job_title'].isin(selected_job_title)) &
    (df['industry'].isin(selected_industry)) &
    (df['remote_work'].isin(selected_remote_work))
]

# ── KPI cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Average Salary", f"${filtered_df['salary'].mean():,.0f}")
with c2:
    metric_card("Median Salary", f"${filtered_df['salary'].median():,.0f}")
with c3:
    metric_card("Total Listings", f"{len(filtered_df):,}")
with c4:
    metric_card("Top Paying Role", filtered_df.groupby('job_title')['salary'].mean().idxmax())

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Chart 1 ───────────────────────────────────────────────────────────────────
section("01 / Distribution", "Salary by Job Title", "Box spread per role — wider boxes indicate higher pay variability. Split by remote work arrangement.")
fig = px.box(
    filtered_df, x='job_title', y='salary', color='remote_work',
    color_discrete_sequence=GOLD_PALETTE
)
fig.update_layout(xaxis_title="Job Title", yaxis_title="Salary", legend_title="Remote", **CHART_LAYOUT)
st.plotly_chart(fig, use_container_width=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Chart 2 ───────────────────────────────────────────────────────────────────
section("02 / Trajectory", "Experience vs Salary", "Each point is a listing. The upward trend shows how years of experience compounds earning potential.")
Sal_ex = px.scatter(
    filtered_df, x='experience_years', y='salary', color='remote_work',
    color_discrete_sequence=GOLD_PALETTE, opacity=0.5
)
Sal_ex.update_layout(xaxis_title="Years of Experience", yaxis_title="Salary", legend_title="Remote", **CHART_LAYOUT)
st.plotly_chart(Sal_ex, use_container_width=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Chart 3 ───────────────────────────────────────────────────────────────────
section("03 / Geography", "Salary by Location", "Average compensation by region. Darker bars represent higher-paying markets.")
salary_by_location = filtered_df.groupby('location')['salary'].mean().reset_index()
Sal_loc = px.bar(
    salary_by_location, x='location', y='salary',
    color='salary', color_continuous_scale=['#1a2030', '#c8a96e']
)
Sal_loc.update_layout(xaxis_title="Location", yaxis_title="Avg Salary", coloraxis_showscale=False, **CHART_LAYOUT)
st.plotly_chart(Sal_loc, use_container_width=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Chart 4 ───────────────────────────────────────────────────────────────────
section("04 / Credentials", "Salary by Education Level", "Does a higher degree translate to higher pay? Average earnings broken down by qualification.")
education_salary = filtered_df.groupby('education_level')['salary'].mean().reset_index()
Sal_edu = px.bar(
    education_salary, x='education_level', y='salary',
    color='salary', color_continuous_scale=['#1a2030', '#c8a96e']
)
Sal_edu.update_layout(xaxis_title="Education Level", yaxis_title="Avg Salary", coloraxis_showscale=False, **CHART_LAYOUT)
st.plotly_chart(Sal_edu, use_container_width=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Chart 5 ───────────────────────────────────────────────────────────────────
section("05 / Capability", "Skills Count vs Salary", "More skills, more money? Average salary mapped against skills breadth.")
skill_salary = filtered_df.groupby('skills_count')['salary'].mean().reset_index()
Sal_skill = px.bar(
    skill_salary, x='skills_count', y='salary',
    color='salary', color_continuous_scale=['#1a2030', '#c8a96e']
)
Sal_skill.update_layout(xaxis_title="Skills Count", yaxis_title="Avg Salary", coloraxis_showscale=False, **CHART_LAYOUT)
st.plotly_chart(Sal_skill, use_container_width=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Chart 6 ───────────────────────────────────────────────────────────────────
section("06 / Scale", "Salary by Company Size", "Startup vs corporate — how organisation scale affects compensation.")
company_salary = filtered_df.groupby('company_size')['salary'].mean().reset_index()
Sal_comp = px.bar(
    company_salary, x='company_size', y='salary',
    color='salary', color_continuous_scale=['#1a2030', '#c8a96e']
)
Sal_comp.update_layout(xaxis_title="Company Size", yaxis_title="Avg Salary", coloraxis_showscale=False, **CHART_LAYOUT)
st.plotly_chart(Sal_comp, use_container_width=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── ML Model ──────────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(_df):
    df = _df.copy()
    encoders = {}
    categorical_cols = ['job_title', 'education_level', 'industry', 'location', 'remote_work', 'company_size']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    x = df.drop('salary', axis=1)
    y = df['salary']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x, y)
    return model, encoders

model, encoders = train_model(df)

# ── Predictor ─────────────────────────────────────────────────────────────────
section("07 / Predictor", "What's Your Salary?", "Fill in your profile — the model estimates your market value based on 250,000 real listings.")

col1, col2 = st.columns(2)

with col1:
    job_title_input = st.selectbox("Job Title", options=encoders['job_title'].classes_)
    education_input = st.selectbox("Education Level", options=encoders['education_level'].classes_)
    industry_input  = st.selectbox("Industry", options=encoders['industry'].classes_)
    location_input  = st.selectbox("Location", options=encoders['location'].classes_)
    remote_input    = st.selectbox("Remote Work", options=encoders['remote_work'].classes_)
    company_input   = st.selectbox("Company Size", options=encoders['company_size'].classes_)

with col2:
    experience_input = st.slider("Years of Experience", min_value=0, max_value=30, value=5)
    skills_input     = st.slider("Skills Count", min_value=1, max_value=19, value=5)
    cert_input       = st.slider("Certifications", min_value=0, max_value=10, value=0)

if st.button("Run Prediction"):
    input_data = pd.DataFrame({
        'job_title':        [encoders['job_title'].transform([job_title_input])[0]],
        'education_level':  [encoders['education_level'].transform([education_input])[0]],
        'industry':         [encoders['industry'].transform([industry_input])[0]],
        'location':         [encoders['location'].transform([location_input])[0]],
        'remote_work':      [encoders['remote_work'].transform([remote_input])[0]],
        'company_size':     [encoders['company_size'].transform([company_input])[0]],
        'experience_years': experience_input,
        'skills_count':     skills_input,
        'certifications':   cert_input
    })
    input_data = input_data[['job_title', 'experience_years', 'education_level',
                              'skills_count', 'industry', 'company_size',
                              'location', 'remote_work', 'certifications']]
    predicted_salary = model.predict(input_data)[0]
    st.success(f"Estimated Market Salary — ${predicted_salary:,.0f}")