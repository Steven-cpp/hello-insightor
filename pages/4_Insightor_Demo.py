"""
Data Source
    - debug score
    - benchmark score
    - original score
should cover all the scoring components
    - weights

* Benchmark Score: `scorer_comp.xlsx` derived from `scoreAnalysis.ipynb`
* Original Score: Company Score in prd (both P1 and P3) `score_comp.xlsx` derived from `scoreAnalysis.ipynb`
* Update Frequency: Every month we update the data.

Widgets
    - nav-bar
        - sel: ScoreP1, ScoreP3
        - slider: all the weights, defaut val from `weights`
        - button: SaveAsBM, AutoFT
    - multi-sel: columns to show
    - df: scoring components
        - button: tick -> good company; cross -> bad company
        - color: goodComp -> Green, badComp -> Red; scoring color -> significance
    - graph: score distribution
"""

import glob
import yaml
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px

# Function to load configurations from a YAML file
def load_config(config_path):
    with open(config_path, "r", encoding='ascii') as file:
        cfg = yaml.safe_load(file)
    return cfg

config = load_config("./config.yaml")
WEIGHT_MAP = config["WEIGHT_MAP"]
STRATEGY_MAP = config["STRATEGY_MAP"]

FN_BM_PATTERN = config["FN_BM_PATTERN"]
DATA_DIR = config["DATA_DIR"]
FN_SCORER_WEIGHTS = config["FN_SCORER_WEIGHTS"]
bm_list = sorted(map(lambda x: x.split('/')[-1], glob.glob(DATA_DIR + FN_BM_PATTERN)))

# Load data
@st.cache_data
def load_data():
    df_weights = pd.read_csv(DATA_DIR + FN_SCORER_WEIGHTS)
    df_weights.fillna(0, inplace=True)
    df = pd.read_csv(DATA_DIR + 'scorer_debug_merged_prd_v3.csv', index_col='Id').iloc[:, 1:]
    # df = pd.read_csv(DATA_DIR + bm_list[-1], index_col='Id').iloc[:, 1:]
    return df, df_weights

# Sidebar configuration
def configure_sidebar(df_weights):
    st.sidebar.title("Navigation Bar")
    score_type = st.sidebar.selectbox("Select Score", options=['ScoreP1', 'ScoreP3'])

    weightMap = df_weights.loc[STRATEGY_MAP[score_type]].drop("StrategyName").astype(float).items()
    for weight, value in weightMap:
        if weight in WEIGHT_MAP:
            if value >= 0:
                df_weights.loc[STRATEGY_MAP[score_type], weight] = st.sidebar.slider(weight, min_value=0.0, max_value=1.0, value=value, step=0.01)
            else:
                df_weights.loc[STRATEGY_MAP[score_type], weight] = st.sidebar.slider(weight, min_value=-1.0, max_value=0.0, value=value, step=0.01)
    return score_type

def calculate_scores(df, df_weights, score_type):
    """Calculate ScoreDebug and BM difference

    Args:
        df (_type_): _description_
        df_weights (_type_): _description_
        score_type (_type_): _description_
    """
    SCORING_COL = WEIGHT_MAP.values()
    scoring = df[SCORING_COL].fillna(0).to_numpy()
    weights = df_weights.loc[STRATEGY_MAP[score_type], WEIGHT_MAP.keys()].to_numpy().astype(float)
    scoreDebug = np.dot(scoring, weights)
    scoreDebug = (scoreDebug / df['Quality_Factor']) + \
            (df_weights.loc[STRATEGY_MAP[score_type], 'QualityFactor'] * df['Quality_Factor_Normalized'])
    df['ScoreDebug'] = scoreDebug
    return df

@st.cache_data
def calculate_bm(df, bm, prefix_delta):
    SCORING_COL = WEIGHT_MAP.values()
    DELTA_COL = list(SCORING_COL)
    DELTA_COL.append('Score')
    
    df_delta = None
    
    if bm is not None:
        df_bm = pd.read_csv(DATA_DIR + bm, index_col='Id')
        scoring, df_bm_aligned = df[DELTA_COL].align(df_bm, join='left', axis=0, fill_value=0)
        df_delta = scoring - df_bm_aligned[DELTA_COL].fillna(0).to_numpy()
        df_delta.columns = [f'{prefix_delta}{col}' for col in df_delta.columns]
    return df_delta
    

def plot_score_dist(df, score_type):
    data = df.melt(id_vars=['CompanyName'],
               value_vars=['Score', 'ScoreDebug'],
               var_name='ScoreType',
               value_name=score_type)
    # Plotting the score distribution with Altair
    chart = alt.Chart(data).mark_bar(opacity=0.3).encode(
        alt.X(f'{score_type}:Q', bin=alt.Bin(maxbins=40)),
        y=alt.Y('count()', stack=None),
        color='ScoreType:N'
    ).properties(
        title='Score Distribution'
    )
    st.altair_chart(chart, use_container_width=True)

def plot_score_comp(df, cols):
    df_fin = df[cols].astype(float)
    data_binned = pd.DataFrame()

    # Bin the data into 5 bins for each score
    bins = 5
    labels = ['< 0.2', '[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '>= 0.8']
    for column in df_fin.columns:
        if (pd.notna(df_fin[column]).sum() < bins):
            continue
        data_binned[column + "_bin"] = pd.cut(df_fin[column].to_numpy(), bins, labels=[labels[i] for i in range(bins)])

    # Transform the data to a long format suitable for Altair
    df_long = pd.melt(data_binned, value_vars=[column for column in data_binned.keys()], var_name='ScoreType', value_name='Bin')

    # Create a stacked bar chart
    barChart = alt.Chart(df_long).mark_bar().encode(
        x='ScoreType:N',
        y=alt.Y('count()', stack='zero'),
        color='Bin:N'
    ).properties(
        height=500,
        title='Key Financials Composition'
    )

    st.altair_chart(barChart, use_container_width=True)


def plot_corr_mat(df):
    SCORING_COL = WEIGHT_MAP.values()
    CORR_COL = list(SCORING_COL) + ['Score', 'ScoreDebug']
    df_corr = df[CORR_COL]
    df_corr.columns = [i.split('_')[0] for i in df_corr.columns]
    corr = df_corr.corr().loc[:, ['Score', 'ScoreDebug', 'Quality', 'ManagerQuality']]
    corr = corr.round(decimals=2)
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Key Components Correlation Matrix", height=800)
    st.plotly_chart(fig, theme="streamlit")

# Main page configuration
def configure_main_page(df, df_weights, score_type, prefix_delta='Δ'):
    st.title("Score Debugging App")
    bm = st.selectbox( 'Select benchmark you would like to compare', bm_list)
    df = calculate_scores(df, df_weights, score_type)
    df_delta = calculate_bm(df, bm, prefix_delta)
    col_default = ['CompanyName', 'PrimaryIndustry', 'Score', 'ScoreDebug', 'BusinessDescription']
    options = st.multiselect('Select columns to show', df.columns, default=col_default)
    
    if df_delta is not None:
        col_delta = [f'{prefix_delta}{i}' for i in options if f'{prefix_delta}{i}' in df_delta.columns]
        df_res = pd.merge(df[options], df_delta[col_delta], how='left', left_index=True, right_index=True)
        st.dataframe(df_res)
    else:
        st.dataframe(df[options])
        
    plot_score_dist(df, score_type)
    
    KEY_FIN_COLS = ['GrossMarginInd_normalized', 'EBITDAMarginInd_normalized', 'EBITMarginInd_normalized', 'LeverageInd_normalized', 'RevenueInd_normalized', 'RevenueGrowthInd_normalized']
    plot_score_comp(df, KEY_FIN_COLS)
    
    plot_corr_mat(df)
    

# Main function to run the app
def main():
    df, df_weights = load_data()
    score_select = configure_sidebar(df_weights)
    configure_main_page(df, df_weights, score_select)

if __name__ == "__main__":
    main()
