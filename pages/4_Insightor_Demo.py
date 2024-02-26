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
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

WEIGHT_MAP = {
    'QualityFactor': 'Quality_Factor_Normalized',
    'EmployeeNumber': 'EmployeeNumber_normalized',
    'Revenue1YearTotal': 'Revenue1YearTotal_normalized',
    'StrategySizeDistance': 'TargetAgeDistance_normalized',
    'StrategyAgeDistance': 'TargetSizeDistance_normalized',
    'Revenue2YearGrowthRate': 'Revenue2YearGrowthRate_normalized',
    'Revenue3YearGrowthRate': 'Revenue3YearGrowthRate_normalized',
    'PitchbookSizeMultiple': 'PitchbookSizeMultiple_normalized',
    'PitchbookGrowthRate': 'PitchbookGrowthRate_normalized',
    'ManagerDensity': 'ManagerDensity_normalized',
    'ManagerQuality': 'ManagerQuality_normalized',
    'ManagerNumber': 'ManagerNumber_normalized',
    'EmployeeEstimatedGrowthRate': 'EmployeeEstimatedGrowthRate_normalized',
    'KeyDevelopmentNumber': 'KeyDevelopmentNumber_normalized',
    'WeightedKeyDevelopmentNumber': 'WeightedKeyDevelopmentNumber_normalized',
    'ManagerDataAvailable': 'ManagerDataAvailable_normalized',
    'IndustryNFit': 'IndustryNFit_normalized',
    'DealStageNFit': 'DealStageNFit_normalized',
    'FundraiseInactivity': 'FundraiseInactivity_normalized',
    'FounderScore': 'FounderScore_normalized',
    'RaisedBeyondAvg': 'RaisedBeyondAvg_normalized',
    'ManagerQualityEst': 'ManagersQualityEst_normalized',
    'ManagerQualityTrendEst': 'ManagersQualityTrendEst_normalized',
    'GrossMarginInd': 'GrossMarginInd_normalized',
    'EBITDAMarginInd': 'EBITDAMarginInd_normalized',
    'EBITMarginInd': 'EBITMarginInd_normalized',
    'LeverageInd': 'LeverageInd_normalized',
    'RevenueInd': 'RevenueInd_normalized',
    'RevenueGrowthInd': 'RevenueGrowthInd_normalized'
}
WEIGHT_COL  = WEIGHT_MAP.keys()
SCORING_COL  = WEIGHT_MAP.values()

df_weights = pd.read_csv('./data/scorer_weights.csv')
df = pd.read_csv('./data/scorer_debug_merged_v2.csv')
df = df.iloc[:, 2:]

# Sidebar - Navigation Bar
st.sidebar.title("Navigation Bar")
score_select = st.sidebar.selectbox("Select Score", options=['ScoreP1', 'ScoreP3'])

STRATEGYNAME_COL = "StrategyName"
STRATEGY_MAP = {"ScoreP1": 0, "ScoreP3": 1}

# Display sliders for weights, default values from `weights` list
weightMap = df_weights.loc[STRATEGY_MAP[score_select]].drop(STRATEGYNAME_COL).astype(float).items()
# weightMap = sorted(weightMap, key=lambda item: abs(item[1]), reverse=True)
scorer_col = []
for weight, value in weightMap:
    scorer_col.append(WEIGHT_MAP[weight])
    if value >= 0:
        df_weights.loc[STRATEGY_MAP[score_select], weight] = st.sidebar.slider(weight, min_value=0.0, max_value=3.0, value=value, step=0.01)
    else:
        df_weights.loc[STRATEGY_MAP[score_select], weight] = st.sidebar.slider(weight, min_value=-1.0, max_value=0.0, value=value, step=0.01)

scoring = df[SCORING_COL].fillna(0).to_numpy()
weights = df_weights.loc[STRATEGY_MAP[score_select], WEIGHT_COL].to_numpy().astype(float)
scoreDebug = np.dot(scoring, weights)/10

df['ScoreDebug'] = scoreDebug

col_default = ['CompanyName', 'PrimaryIndustry', 'Score', 'ScoreDebug', 'BusinessDescription']

# Buttons in sidebar
if st.sidebar.button('SaveAsBM'):
    st.sidebar.write('SaveAsBM clicked')
if st.sidebar.button('AutoFT'):
    st.sidebar.write('AutoFT clicked')

# Main Page
st.title("Score Debugging App")

dir = './data/'
fname = 'scorer_debug_merged*'
# List all files in the current directory starting with 'merged'
bm_list = map(lambda x: x.split('/')[-1], glob.glob(dir + fname))

bm = st.selectbox(
    'Which benchmark you would like to compare',
    bm_list, index=None)

# Multi-select for columns to show
options = st.multiselect('Select columns to show', df.columns, default=col_default)



# Display dataframe with selected columns
st.dataframe(df[options])

data = df.melt(id_vars=['CompanyName'],
               value_vars=['Score', 'ScoreDebug'],
               var_name='ScoreType',
               value_name=score_select)

# Plotting the score distribution with Altair
chart = alt.Chart(data).mark_bar(opacity=0.3).encode(
    alt.X(f'{score_select}:Q', bin=alt.Bin(maxbins=40)),
    y=alt.Y('count()', stack=None),
    color='ScoreType:N'
).properties(
    title='Score Distribution'
)
st.altair_chart(chart, use_container_width=True)


NEW_COLS = ['GrossMarginInd_normalized', 'EBITDAMarginInd_normalized', 'EBITMarginInd_normalized', 'LeverageInd_normalized', 'RevenueInd_normalized', 'RevenueGrowthInd_normalized']
data_comp = df[NEW_COLS].astype(float)
data_binned = pd.DataFrame()

# Bin the data into 5 bins for each score
bins = 5
labels = ['< 0.2', '[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '>= 0.8']
for column in data_comp.columns:
    if (pd.notna(data_comp[column]).sum() < bins):
        continue
    data_binned[column + "_bin"] = pd.cut(data_comp[column].to_numpy(), bins, labels=[labels[i] for i in range(bins)])

# Transform the data to a long format suitable for Altair
df_long = pd.melt(data_binned, value_vars=[column for column in data_binned.keys()], var_name='ScoreType', value_name='Bin')

# # Count the occurrences in each bin for each score type
# df_long['Count'] = 1
# df_grouped = df_long.groupby(['ScoreType', 'Bin']).count().reset_index()

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

## creating AgGrid dynamic table and setting configurations
# gb = GridOptionsBuilder.from_dataframe(df_filtered)
# gb.configure_selection(selection_mode="single", use_checkbox=True)
# gb.configure_column(field='Company Name', width=270)
# gb.configure_column(field='Sector', width=260)
# gb.configure_column(field='Industry', width=350)
# gb.configure_column(field='Prospect Status', width=270)
# gb.configure_column(field='Product', width=240)

# gridOptions = gb.build()

# response = AgGrid(
#     df_filtered,
#     gridOptions=gridOptions,
#     enable_enterprise_modules=False,
#     height=600,
#     update_mode=GridUpdateMode.SELECTION_CHANGED,
#     data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
#     fit_columns_on_grid_load=False,
#     theme='alpine',
#     allow_unsafe_jscode=True
# )
