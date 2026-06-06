# ============================================================
# RNA Marker Finder — Python Streamlit Version
# Data: GSE45827 (real breast cancer RNA-seq from NCBI GEO)
# Model: scikit-learn (Random Forest / Logistic Regression / SVM)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import gzip
import io
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                              accuracy_score, f1_score, matthews_corrcoef,
                              classification_report)
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="RNA Marker Finder",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #3498db;
    }
    .metric-val { font-size: 2.2rem; font-weight: 700; color: #2980b9; }
    .metric-lab { font-size: 0.85rem; color: #777; margin-top: 4px; }
    .stButton>button {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-size: 1rem; font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(46,204,113,0.4); }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #2c3e50;
        border-bottom: 2px solid #3498db; padding-bottom: 6px; margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 1. DATA LOADING
# ============================================================

@st.cache_data
def load_geo_data(filepath):
    """
    Parse GSE45827 GEO series matrix file.
    Returns DataFrame with label + gene expression columns.
    """
    try:
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        else:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = lines = f.readlines()

        # Find sample characteristics (cancer vs normal)
        sample_titles = []
        labels = []
        data_lines = []
        in_table = False

        for line in lines:
            line = line.strip()
            if line.startswith('!Sample_title'):
                parts = line.split('\t')
                sample_titles = [p.strip('"') for p in parts[1:]]
            if line.startswith('!Sample_characteristics_ch1') and 'tissue' in line.lower():
                parts = line.split('\t')
                for p in parts[1:]:
                    p = p.strip('"').lower()
                    if 'tumor' in p or 'cancer' in p or 'carcinoma' in p:
                        labels.append('Cancer')
                    else:
                        labels.append('Normal')
            if line.startswith('!series_matrix_table_begin'):
                in_table = True
                continue
            if line.startswith('!series_matrix_table_end'):
                in_table = False
                continue
            if in_table:
                data_lines.append(line)

        if not data_lines:
            return None, "Could not parse GEO file format"

        # Parse expression table
        header = data_lines[0].split('\t')
        sample_ids = [h.strip('"') for h in header[1:]]
        n_samples = len(sample_ids)

        # If labels not found from characteristics, assign by pattern
        if len(labels) != n_samples:
            # GSE45827: samples labeled by cell line type
            labels = []
            for title in sample_titles[:n_samples] if sample_titles else sample_ids:
                t = title.lower()
                if any(x in t for x in ['normal', 'mcf10', 'hmec', 'healthy']):
                    labels.append('Normal')
                else:
                    labels.append('Cancer')

        # Pad labels if needed
        while len(labels) < n_samples:
            labels.append('Cancer')
        labels = labels[:n_samples]

        # Parse expression values
        gene_names = []
        expr_data = []
        for line in data_lines[1:500]:  # limit to 500 genes for speed
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            gene = parts[0].strip('"')
            try:
                vals = [float(v.strip('"')) for v in parts[1:n_samples+1]]
                if len(vals) == n_samples:
                    gene_names.append(gene)
                    expr_data.append(vals)
            except:
                continue

        if not expr_data:
            return None, "No expression data found"

        # Build DataFrame
        expr_matrix = np.array(expr_data).T  # samples x genes
        df = pd.DataFrame(expr_matrix, columns=gene_names)
        df.insert(0, 'label', labels)
        df['label'] = pd.Categorical(df['label'], categories=['Normal', 'Cancer'])

        return df, None

    except Exception as e:
        return None, str(e)


def simulate_tcga_brca(n_cancer=100, n_normal=100, n_genes=500, seed=42):
    """
    Simulate TCGA-BRCA-like RNA-seq expression data.
    Based on published TCGA-BRCA expression statistics.
    """
    np.random.seed(seed)

    # Real BRCA-associated gene symbols
    brca_genes = [
        "BRCA1","BRCA2","TP53","PIK3CA","CDH1","PTEN","AKT1","MYC",
        "CCND1","ERBB2","ESR1","PGR","MKI67","KRT5","KRT14","KRT17",
        "EGFR","VEGFA","MMP9","MMP2","CDK4","CDK6","RB1","CDKN2A",
        "MDM2","BCL2","BAX","CASP3","CASP9","PCNA","TOP2A","AURKA",
        "AURKB","PLK1","BUB1","CENPA","MCM2","MCM6","TYMS","DHFR",
        "FOLR1","TACSTD2","EPCAM","CD44","CD24","ALDH1A1","SOX2",
        "NANOG","TWIST1","SNAI1","SNAI2","VIM","FN1","CDH2","ZEB1",
        "ZEB2","TGFB2","SMAD2","SMAD3","SMAD4","WNT5A","CTNNB1",
        "NOTCH1","NOTCH2","JAG1","HEY1","HES1","STAT3","JAK2","IL6",
        "CXCR4","CXCL12","FOXP3","CD8A","CD274","PDCD1","IDO1",
        "VEGFC","IGF1R","MAPK1","MAPK3","BRAF","KRAS","NRAS","HRAS",
        "MTOR","HIF1A","VHL","LDHA","HK2","IDH1","IDH2","EZH2",
        "DNMT1","DNMT3A","TET2","HDAC1","HDAC2","SIRT1","PARP1",
        "RAD51","CHEK2","ATM","ATR","BRIP1","PALB2","FANCA","FANCD2",
    ]
    # Extend with generic gene names
    extra = [f"GENE_{i:04d}" for i in range(1, n_genes - len(brca_genes) + 10)]
    all_genes = (brca_genes + extra)[:n_genes]

    n_samples = n_cancer + n_normal
    labels = ['Cancer'] * n_cancer + ['Normal'] * n_normal

    # Baseline: log2(TPM+1) ~ N(4, 1.5)
    expr = np.random.normal(4, 1.5, (n_samples, n_genes))
    expr = np.clip(expr, 0, None)

    # True marker genes: 13 up, 8 down in Cancer
    up_idx   = [0,2,4,6,8,10,12,14,16,18,20,22,24]
    down_idx = [1,3,5,7,9,11,13,15]

    cancer_rows = np.where(np.array(labels) == 'Cancer')[0]
    normal_rows = np.where(np.array(labels) == 'Normal')[0]

    for i in up_idx:
        fc = np.random.uniform(1.5, 2.2)
        expr[cancer_rows, i] += fc + np.random.normal(0, 0.8, len(cancer_rows))
        expr[normal_rows, i] += np.random.normal(0, 0.7, len(normal_rows))

    for i in down_idx:
        fc = np.random.uniform(1.2, 1.8)
        expr[cancer_rows, i] -= fc
        expr[cancer_rows, i] += np.random.normal(0, 0.8, len(cancer_rows))
        expr[normal_rows, i] += np.random.normal(0, 0.7, len(normal_rows))

    expr = np.clip(expr, 0, None)

    df = pd.DataFrame(expr, columns=all_genes)
    df.insert(0, 'label', labels)
    df['label'] = pd.Categorical(df['label'], categories=['Normal', 'Cancer'])
    return df


# ============================================================
# 2. PREPROCESSING & TRAINING
# ============================================================

def preprocess(df):
    y = (df['label'] == 'Cancer').astype(int).values
    X = df.drop('label', axis=1).values.astype(float)
    gene_names = df.drop('label', axis=1).columns.tolist()

    # Log2 transform if raw counts
    if X.max() > 100:
        X = np.log2(X + 1)

    # Z-score normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    return X, y, gene_names


def train_model(X, y, model_type='rf', test_ratio=0.2, cv_folds=5, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y
    )

    if model_type == 'rf':
        base = RandomForestClassifier(n_estimators=300, max_features='sqrt',
                                       random_state=seed, n_jobs=-1)
        model = base
    elif model_type == 'lr':
        model = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    else:  # svm
        base = LinearSVC(max_iter=2000, random_state=seed, C=1.0)
        model = CalibratedClassifierCV(base, cv=3)

    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring='roc_auc', n_jobs=-1)

    # Train on full training set
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Feature importance
    if model_type == 'rf':
        importances = model.feature_importances_
    elif model_type == 'lr':
        importances = np.abs(model.coef_[0])
    else:
        importances = np.abs(model.calibrated_classifiers_[0].estimator.coef_[0])

    imp_norm = importances / (importances.max() + 1e-10) * 100

    return {
        'model': model,
        'fpr': fpr, 'tpr': tpr, 'auc': roc_auc,
        'acc': acc, 'f1': f1, 'mcc': mcc,
        'cm': cm, 'cv_scores': cv_scores,
        'importances': imp_norm,
        'y_test': y_test, 'y_pred': y_pred, 'y_prob': y_prob,
    }


# ============================================================
# 3. PLOTS
# ============================================================

def plot_roc(fpr, tpr, roc_auc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
        line=dict(dash='dash', color='gray', width=1.5), name='Random', showlegend=True))
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
        line=dict(color='#3498db', width=2.5),
        fill='tozeroy', fillcolor='rgba(52,152,219,0.1)',
        name=f'AUC = {roc_auc:.3f}'))
    fig.update_layout(
        xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
        xaxis=dict(tickformat='.0%'), yaxis=dict(tickformat='.0%'),
        legend=dict(x=0.6, y=0.1), height=380,
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(t=20, b=50, l=60, r=20)
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    return fig


def plot_cm(cm):
    labels = ['Normal', 'Cancer']
    text = [[f'TN<br>{cm[0,0]}', f'FP<br>{cm[0,1]}'],
            [f'FN<br>{cm[1,0]}', f'TP<br>{cm[1,1]}']]
    fig = go.Figure(go.Heatmap(
        z=cm, x=['Pred Normal', 'Pred Cancer'],
        y=['True Normal', 'True Cancer'],
        text=text, texttemplate='%{text}',
        colorscale=[[0,'#1a252f'],[0.5,'#1a6b9a'],[1,'#3498db']],
        showscale=False, textfont=dict(size=16, color='white')
    ))
    fig.update_layout(height=380, plot_bgcolor='white', paper_bgcolor='white',
                      margin=dict(t=20, b=60, l=80, r=20))
    return fig


def plot_feature_importance(gene_names, importances, top_n=20):
    idx = np.argsort(importances)[::-1][:top_n]
    genes = [gene_names[i] for i in idx]
    imps = [importances[i] for i in idx]

    colors = px.colors.sequential.Blues[2:]
    color_list = [colors[int(i / top_n * (len(colors)-1))] for i in range(top_n)]

    fig = go.Figure(go.Bar(
        x=imps[::-1], y=genes[::-1],
        orientation='h',
        marker_color=color_list[::-1],
    ))
    fig.update_layout(
        xaxis_title='Feature Importance (scaled 0–100)',
        height=max(400, top_n * 22),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(t=20, b=50, l=100, r=20)
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    return fig


def plot_cv_scores(cv_scores):
    fig = go.Figure(go.Bar(
        x=list(range(1, len(cv_scores)+1)), y=cv_scores,
        marker_color='#2ecc71', text=[f'{s:.3f}' for s in cv_scores],
        textposition='outside'
    ))
    fig.update_layout(
        xaxis_title='Fold', yaxis_title='AUC',
        yaxis=dict(range=[0, 1.1]),
        height=260, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(t=20, b=40, l=60, r=20)
    )
    return fig


def plot_gene_boxplot(df, gene):
    cancer = df[df['label']=='Cancer'][gene].values
    normal = df[df['label']=='Normal'][gene].values
    fig = go.Figure()
    fig.add_trace(go.Box(y=normal, name='Normal', marker_color='#2ecc71',
                         boxpoints='all', jitter=0.3, pointpos=-1.8))
    fig.add_trace(go.Box(y=cancer, name='Cancer', marker_color='#e74c3c',
                         boxpoints='all', jitter=0.3, pointpos=-1.8))
    fig.update_layout(
        yaxis_title='log2(expression)',
        height=300, plot_bgcolor='white', paper_bgcolor='white',
        showlegend=False, margin=dict(t=20, b=40, l=60, r=20)
    )
    return fig


# ============================================================
# 4. STREAMLIT APP
# ============================================================

# Session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'gene_names' not in st.session_state:
    st.session_state.gene_names = None

# ── Header ────────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(135deg, #2c3e50, #3498db);
     padding: 2rem; border-radius: 12px; margin-bottom: 2rem;'>
    <h1 style='color:white; margin:0; font-size:2rem;'>🧬 RNA Marker Finder</h1>
    <p style='color:#bde0f5; margin:0.5rem 0 0;'>
        Cancer Biomarker Discovery from RNA-seq Expression Data
        &nbsp;·&nbsp; Python + scikit-learn
        &nbsp;·&nbsp; Real GEO / TCGA-BRCA Data
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Data Source")

    data_source = st.radio("Choose data:", [
        "🔬 Real GEO Data (GSE45827)",
        "🧪 Simulated TCGA-BRCA",
        "📤 Upload CSV"
    ])

    if data_source == "🔬 Real GEO Data (GSE45827)":
        geo_path = os.path.expanduser("~/Desktop/rna-marker-python/GSE45827.txt.gz")
        if st.button("Load GEO Data"):
            with st.spinner("Parsing GSE45827 (real breast cancer data)..."):
                df, err = load_geo_data(geo_path)
                if err:
                    st.error(f"Error: {err}")
                else:
                    st.session_state.df = df
                    st.session_state.results = None
                    st.success(f"✅ Loaded! {df.shape[0]} samples × {df.shape[1]-1} genes")

    elif data_source == "🧪 Simulated TCGA-BRCA":
        n_cancer = st.slider("Cancer samples", 50, 200, 100)
        n_normal = st.slider("Normal samples", 50, 200, 100)
        n_genes  = st.slider("Genes", 100, 500, 500)
        if st.button("Generate Data"):
            with st.spinner("Simulating TCGA-BRCA data..."):
                df = simulate_tcga_brca(n_cancer, n_normal, n_genes)
                st.session_state.df = df
                st.session_state.results = None
                st.success(f"✅ Generated! {df.shape[0]} samples × {df.shape[1]-1} genes")

    else:
        uploaded = st.file_uploader("Upload CSV", type=['csv','tsv'])
        if uploaded:
            sep = '\t' if uploaded.name.endswith('.tsv') else ','
            df = pd.read_csv(uploaded, sep=sep)
            if 'label' not in df.columns:
                st.error("CSV must have a 'label' column (Cancer/Normal)")
            else:
                df['label'] = pd.Categorical(df['label'].str.strip(),
                                              categories=['Normal','Cancer'])
                st.session_state.df = df
                st.session_state.results = None
                st.success(f"✅ {df.shape[0]} samples × {df.shape[1]-1} genes")

    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")

    model_type = st.selectbox("Algorithm", [
        ("Random Forest", "rf"),
        ("Logistic Regression", "lr"),
        ("SVM (Linear)", "svm")
    ], format_func=lambda x: x[0])

    test_ratio = st.slider("Test set ratio", 0.1, 0.4, 0.2, 0.05,
                            format="%.0f%%",
                            help="Fraction of data used for testing")
    cv_folds   = st.select_slider("CV folds", [3,5,10], 5)
    top_n      = st.slider("Top N markers", 5, 50, 20)

    st.markdown("---")
    run = st.button("🚀 Run Analysis", disabled=st.session_state.df is None)

# ── Main area ──────────────────────────────────────────────

if st.session_state.df is None:
    # Welcome screen
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Step 1 — Data**\n\nLoad real GEO breast cancer data (GSE45827), simulated TCGA-BRCA data, or upload your own CSV.")
    with c2:
        st.info("**Step 2 — Train**\n\nChoose Random Forest, Logistic Regression, or SVM. Click Run Analysis.")
    with c3:
        st.info("**Step 3 — Results**\n\nView ROC/AUC, Confusion Matrix, and Top Marker Genes with interactive charts.")

    st.markdown("""
    ---
    #### 📋 CSV Format
    ```
    label,BRCA1,TP53,EGFR,...
    Cancer,5.2,3.1,7.8,...
    Normal,2.1,4.8,3.2,...
    ```
    First column must be named **label** with values **Cancer** or **Normal**.
    """)

else:
    df = st.session_state.df

    # Data summary
    n_cancer = (df['label'] == 'Cancer').sum()
    n_normal = (df['label'] == 'Normal').sum()
    n_genes  = df.shape[1] - 1

    st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Total Samples", df.shape[0])
    with c2: st.metric("Cancer", n_cancer)
    with c3: st.metric("Normal", n_normal)
    with c4: st.metric("Genes", n_genes)

    with st.expander("Preview data (first 5 rows, first 8 genes)"):
        show_cols = ['label'] + df.columns[1:9].tolist()
        st.dataframe(df[show_cols].head(), use_container_width=True)

    # Run training
    if run:
        with st.spinner("Preprocessing & training model..."):
            X, y, gene_names = preprocess(df)
            results = train_model(X, y,
                                   model_type=model_type[1],
                                   test_ratio=test_ratio,
                                   cv_folds=cv_folds)
            results['gene_names'] = gene_names
            st.session_state.results = results
            st.session_state.gene_names = gene_names

    # Show results
    if st.session_state.results is not None:
        res = st.session_state.results
        gene_names = res['gene_names']

        st.markdown("---")
        st.markdown('<div class="section-header">📈 Model Performance</div>', unsafe_allow_html=True)

        # Metrics
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("AUC – ROC", f"{res['auc']:.3f}")
        with c2: st.metric("Accuracy",  f"{res['acc']*100:.1f}%")
        with c3: st.metric("F1 Score",  f"{res['f1']:.3f}")
        with c4: st.metric("MCC",       f"{res['mcc']:.3f}")

        # ROC + CM
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**ROC Curve**")
            st.plotly_chart(plot_roc(res['fpr'], res['tpr'], res['auc']),
                            use_container_width=True)
        with c2:
            st.markdown("**Confusion Matrix**")
            st.plotly_chart(plot_cm(res['cm']), use_container_width=True)

        # CV scores
        st.markdown("**Cross-validation AUC (training set)**")
        cv_scores = res['cv_scores']
        st.plotly_chart(plot_cv_scores(cv_scores), use_container_width=True)
        st.caption(f"Mean CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Feature importance
        st.markdown("---")
        st.markdown('<div class="section-header">🧬 Top Marker Genes</div>', unsafe_allow_html=True)

        st.plotly_chart(plot_feature_importance(gene_names, res['importances'], top_n),
                        use_container_width=True)

        # Marker table
        idx = np.argsort(res['importances'])[::-1][:top_n]
        marker_df = pd.DataFrame({
            'Rank': range(1, top_n+1),
            'Gene': [gene_names[i] for i in idx],
            'Importance': [f"{res['importances'][i]:.2f}" for i in idx],
        })
        st.dataframe(marker_df, use_container_width=True, hide_index=True)

        # Gene explorer
        st.markdown("---")
        st.markdown('<div class="section-header">🔍 Gene Explorer</div>', unsafe_allow_html=True)
        top_genes = [gene_names[i] for i in idx]
        selected_gene = st.selectbox("Select a marker gene to explore:", top_genes)
        if selected_gene in df.columns:
            st.plotly_chart(plot_gene_boxplot(df, selected_gene),
                            use_container_width=True)

        # Download
        st.markdown("---")
        csv = marker_df.to_csv(index=False)
        st.download_button("⬇️ Download markers.csv", csv,
                           file_name="rna_markers.csv", mime="text/csv")
