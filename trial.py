import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# --- Page Config ---
st.set_page_config(layout="wide")
st.sidebar.title("🔍 FlipOse-RE-Analytics")

# --- File Paths ---
df_path = "target_df.csv"
area_stats_path = "df_area_plot_stats.xlsx"
cat_plot_path = "original_df_description_year.xlsx"
summary = "data_summary.xlsx"
sample = "sample_df.csv"

# --- Load Data with Error Handling ---

def load_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.sidebar.error(f"File not found: {file_path}")
        st.stop()

def load_excel(file_path):
    try:
        return pd.read_excel(file_path)
    except FileNotFoundError:
        st.sidebar.error(f"File not found: {file_path}")
        st.stop()

# --- Load Main Dataset ---
df = load_csv(df_path)
st.sidebar.success("All data loaded, 🔍 Explore the Dash Board")

# --- Load Area Stats ---
df_area_plot_stats = load_excel(area_stats_path)

# --- Sidebar Navigation ---
sidebar_option = st.sidebar.radio("Choose View", [
    "Data Summary",
    "Pareto Analysis",
    "Univariate Analysis",
    "Bivariate Analysis",
    "Geo graphical Analysis",
    "Price Prediction Model"
])

# --- View 1: Data Summary ---
if sidebar_option == "Data Summary":
    st.subheader("📄 Transactions Data")
    tab1, tab2 = st.tabs(["Data Preview", "Summary"])
    with tab1:
        sample_df = pd.read_csv(sample)
        st.markdown("--> Repeated columns i.e Arabic and Id columns are dropped from Data")
        sample_df  = sample_df.drop(sample_df.columns[0], axis=1)
        st.dataframe(sample_df)


    with tab2:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Number Of Columns", value = 46)
        with col2:
            st.metric(label="Total Records", value = "1,424,588")
        with col3:
            st.metric(label="Start Date(Instance_date)", value="1966-01-18")
        with col4:
            st.metric(label="End Date(Instance_date)", value="2025-04-03")
        
        summary_df = pd.read_excel(summary)
        # Format all numeric columns with commas
        for col in summary_df.select_dtypes(include='number').columns:
            summary_df[col] = summary_df[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else x)

        summary_df.index = range(1, len(summary_df) + 1)
        summary_df.rename(columns={'No_of_units': 'Num_of_Unique_values'}, inplace=True)
        summary_df = summary_df.drop(columns = ["S.no", "Level"])
        st.dataframe(summary_df)

# --- View 2: Pareto Analysis ---
elif sidebar_option == "Pareto Analysis":
    st.subheader("📍Pareto Analysis")
    try:
            pereto_file = "pereto_analysis_file.xlsx"
            html_pereto_df = "pareto_analysis_plot.html"
            pereto_analyis = pd.ExcelFile(pereto_file)
            pereto_sheet_names = pereto_analyis.sheet_names
    except  FileNotFoundError:
             st.error(f"File not found: {pereto_file}")
             st.stop()
    all_sheets_df = pd.read_excel(pereto_analyis, sheet_name=pereto_sheet_names)
    
     # Extract specific sheets
    pareto_summary = all_sheets_df["Pereto_Analysis_by_area_name"]
    ABC_summary = all_sheets_df["ABC_Area_name"]

    tab1, tab2 = st.tabs(["ABC Summary", "Pareto Analysis Table"])
    with tab1:
        st.markdown("### 📊 Pareto Analysis Plot")
    
        if os.path.exists(html_pereto_df):
            with open(html_pereto_df, "r", encoding="utf-8") as f:
                dt_html = f.read()
            components.html(dt_html, height=500, width=3500, scrolling=False)
        else:
            st.error("Pareto analysis HTML file not found.")
            
            # ABC Summary Table
        st.markdown("### 📋 ABC Summary Table")
        #if 'nRecords' in ABC_summary.columns:
        ABC_summary['nRecords'] = ABC_summary['nRecords'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else x)
        ABC_summary.index = range(1, len(ABC_summary) + 1)
        st.dataframe(ABC_summary, use_container_width=True)
        else:
                st.warning("ABC Summary data is empty or not loaded.")
        with tab2:
            st.markdown("### Pareto Analysis by Area_name_en")
            pareto_summary['nRecords'] =  pareto_summary['nRecords'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else x)
            pareto_summary.index = range(1, len(pareto_summary) + 1)
            st.dataframe(pareto_summary, use_container_width=True)





   

# --- View 3: Plots on Categorical Columns ---
elif sidebar_option == "Plots on Categorical Columns":
    st.subheader("📊 Box Plot and Mean Line Plot by Categorical Columns")
    try:
        xls = pd.ExcelFile(cat_plot_path)
        sheet_names = xls.sheet_names
    except FileNotFoundError:
        st.error(f"File not found: {cat_plot_path}")
        st.stop()

    sheet = st.selectbox("Select Sheet to Visualize", sheet_names)
    df_plot = pd.read_excel(xls, sheet_name=sheet)
    st.write(f"### Sheet: {sheet}")
    st.dataframe(df_plot)

    def plot_boxplot(df):
        if 'instance_year' not in df.columns:
            return None

        group_col = df.columns[2] if len(df.columns) > 2 else None
        required_cols = {'count', 'min', 'mean', '25%', '50%', '75%', 'max'}
        if not required_cols.issubset(df.columns):
            return None

        # Aggregate statistics
        if group_col and group_col in df.columns:
            grouped = df.groupby(group_col).agg({
                'count': 'sum',
                'min': 'min',
                'mean': 'mean',
                '25%': 'mean',
                '50%': 'mean',
                '75%': 'mean',
                'max': 'max'
            }).reset_index()
        else:
            grouped = pd.DataFrame([{
                'count': df['count'].sum(),
                'min': df['min'].min(),
                'mean': df['mean'].mean(),
                '25%': df['25%'].mean(),
                '50%': df['50%'].mean(),
                '75%': df['75%'].mean(),
                'max': df['max'].max(),
                group_col: 'Overall'
            }])

        fig = go.Figure()
        colors = px.colors.qualitative.Set2  # You can change palette here

        for idx, (_, row) in enumerate(grouped.iterrows()):
            q1 = row['25%']
            q3 = row['75%']
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            name = row[group_col] if group_col else 'Overall'

            fig.add_trace(go.Box(
                name=name,
                y=[row['min'], q1, row['50%'], q3, row['max']],
                boxpoints='outliers',
                marker=dict(color=colors[idx % len(colors)]),
                line=dict(color=colors[idx % len(colors)]),
                q1=[q1],
                median=[row['50%']],
                q3=[q3],
                lowerfence=[lower_fence],
                upperfence=[upper_fence],
                orientation='v'  # vertical (default)
                ))

        fig.update_layout(
            title=f"Aggregated Box Plot by {group_col if group_col else 'Overall'}",
            yaxis_title="Meter Sale Price",
            xaxis_title=group_col if group_col else '',
            boxmode='group'
        )
        return fig


    # 📈 Mean Line Plot Function
    def plot_mean_line(df):
        if 'instance_year' not in df.columns or 'mean' not in df.columns:
            return None

        legend_col = df.columns[2] if len(df.columns) > 2 else None
        fig = go.Figure()

        if legend_col and legend_col in df.columns:
            for name, group_df in df.groupby(legend_col):
                fig.add_trace(go.Scatter(
                    x=group_df['instance_year'],
                    y=group_df['mean'],
                    mode='lines+markers',
                    name=str(name)
                ))
        else:
            fig.add_trace(go.Scatter(
                x=df['instance_year'],
                y=df['mean'],
                mode='lines+markers',
                name='Mean'
            ))

        fig.update_layout(
            title=f'Mean (Meter Sale Price) Over Years by {legend_col if legend_col else "N/A"}',
            xaxis_title='Instance Year',
            yaxis_title='Mean (Meter Sale Price)',
            hovermode='x unified'
        )
        return fig

    # 🎯 Layout with Plots
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Aggregated Box Plot")
        box_fig = plot_boxplot(df_plot)
        if box_fig:
            st.plotly_chart(box_fig, use_container_width=True)
        else:
            st.info("Box plot not available due to missing columns or data.")

    with col2:
        st.subheader("📈 Mean Line Plot")
        line_fig = plot_mean_line(df_plot)
        if line_fig:
            st.plotly_chart(line_fig, use_container_width=True)
        else:
            st.info("Mean line plot not available due to missing columns or data.")
