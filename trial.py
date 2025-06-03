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
    tab1, tab2 = st.tabs(["Preview", "Summary"])
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

    tab1, tab2, tab3= st.tabs(["Table", "Chart","ABC summary"])
    with tab1:
        
        st.markdown("### Pareto Analysis by Area_name_en")
        pareto_summary.rename(columns={'Cum%_areas': 'Cum%_Areas'}, inplace=True)
        pareto_summary['nRecords'] =  pareto_summary['nRecords'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else x)
        pareto_summary['Cumulative_%'] = pareto_summary['Cumulative_%'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else x)
        pareto_summary['Percentage(%)'] = pareto_summary['Percentage(%)'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else x)
        pareto_summary['Cum%_Areas'] = pareto_summary['Cum%_Areas'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else x)
        pareto_summary.index = range(1, len(pareto_summary) + 1)
        st.dataframe(pareto_summary, use_container_width=True)
        

    with tab2:
        if os.path.exists(html_pereto_df):
            with open(html_pereto_df, "r", encoding="utf-8") as f:
                dt_html = f.read()
            components.html(dt_html,width=10000,height=10000,scrolling=True)
        else:
            st.error("Pareto analysis HTML file not found.")



    with tab3:
        col1,col2 = st.columns(2)
        with col2:
            df = ABC_summary
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Add bar charts for %Area and %Records
            fig.add_trace(
                go.Bar(name='%Area', x=df['Group_name'], y=df['%Area'], marker_color='skyblue',
                hovertemplate='<b>%{x}</b><br>%Area: %{y:.2f}%<extra></extra>'),
                secondary_y=False,)
            fig.add_trace(
                go.Bar(name='%Records', x=df['Group_name'], y=df['%Records '], marker_color='lightcoral',
                   hovertemplate='<b>%{x}</b><br>%Records: %{y:.2f}%<extra></extra>'),
                   secondary_y=False,)

            # Add line charts for cumulative percentages
            fig.add_trace(
                go.Scatter(name='Cum%_records', x=df['Group_name'], y=df['Cum%_records'], mode='lines+markers', marker_color='green',
                   hovertemplate='<b>%{x}</b><br>Cum% Records: %{y:.2f}%<extra></extra>'),
                   secondary_y=True,)

            fig.add_trace(
                go.Scatter(name='Cum%_areas', x=df['Group_name'], y=df['Cum%_areas'], mode='lines+markers', marker_color='darkorange',
                   hovertemplate='<b>%{x}</b><br>Cum% Areas: %{y:.2f}%<extra></extra>'),
                   secondary_y=True,)

            # Set x-axis title
            fig.update_xaxes(title_text='Group_name')

            # Set y-axes titles
            fig.update_yaxes(title_text='Counts (%Area, %Records)', secondary_y=False)
            fig.update_yaxes(title_text='Cumulative Percentage', secondary_y=True)

            # Add title and legend
            fig.update_layout(
                title_text='ABC Analysis Summary',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode='x unified')

            # Show the plot in Streamlit
            st.plotly_chart(fig)

        with col1:
            # ABC Summary Table
            st.markdown("### 📋 ABC Summary Table")
            ABC_summary['nRecords'] = ABC_summary['nRecords'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else x)
            ABC_summary.index = range(1, len(ABC_summary) + 1)
            st.dataframe(ABC_summary, use_container_width=True)
        
# --- View 3: Univariate Analysis  ---
if sidebar_option == "Univariate Analysis":

    # Load Excel Sheets
    try:
        cat_plot_path = "original_df_description_tables.xlsx"
        xls = pd.ExcelFile(cat_plot_path)
        sheet_names = xls.sheet_names
    except FileNotFoundError:
        st.error(f"File not found: {cat_plot_path}")
        st.stop()

    main_tabs = st.tabs([ "Dimension","Metrix"])

    with main_tabs[0]:
        # Select sheet before tabs
        selected_sheet = st.selectbox("Select categorical column", sheet_names)
        df = pd.read_excel(xls, sheet_name=selected_sheet)
        col1 = df.columns[0]  # Category column
        st.markdown("### 📊 Bar Plot (nRecords)")
        if "nRecords" in df.columns:
            fig_bar = px.bar(df, x=col1, y="nRecords", title=f"nRecords by {col1}", color=col1)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("'nRecords' column not found.")

    with main_tabs[1]:
        # Select sheet before tabs
        selected_sheet = st.selectbox("Select categorical column", sheet_names)
        df = pd.read_excel(xls, sheet_name=selected_sheet)
        col1 = df.columns[0]  # Category column

        tab1, tab2 = st.tabs(["📋 Summary Table", "📈 Plots"])

        with tab1:
            st.dataframe(df, use_container_width=True)

        with tab2:
            colA, colB = st.columns(2)

            # Box plot per category using summary stats columns
            def plot_boxplot_per_category(df, cat_col):
                required_cols = {'min', '25%', '50%', '75%', 'max'}
                if not required_cols.issubset(df.columns):
                    return None

                fig = go.Figure()

                for _, row in df.iterrows():
                    category = row[cat_col]
                    q1 = row['25%']
                    median = row['50%']
                    q3 = row['75%']
                    min_val = row['min']
                    max_val = row['max']
                    iqr = q3 - q1
                    lower_fence = max(min_val, q1 - 1.5 * iqr)
                    upper_fence = min(max_val, q3 + 1.5 * iqr)

                    fig.add_trace(go.Box(
                        name=str(category),
                        q1=[q1],
                        median=[median],
                        q3=[q3],
                        lowerfence=[lower_fence],
                        upperfence=[upper_fence],
                        boxpoints=False
                    ))

                fig.update_layout(
                    title=f"Box Plot by {cat_col}",
                    yaxis_title="Meter Sale Price",
                    boxmode='group',
                    xaxis_title=cat_col
                )
                return fig

            with colA:
                st.markdown("### 📦 Box Plot by Category")
                fig_box = plot_boxplot_per_category(df, col1)
                if fig_box:
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.warning("Required columns ('min', '25%', '50%', '75%', 'max') not found.")

            with colB:
                st.markdown("### 📊 Bar Plot (nRecords)")
                if "nRecords" in df.columns:
                    fig_bar = px.bar(df, x=col1, y="nRecords", title=f"nRecords by {col1}", color=col1)
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.warning("'nRecords' column not found.")
                    
# --- View 3: Bivariate Analysis  ---
if sidebar_option == "Bivariate Analysis":
    st.markdown("### 📋nRecords and Avg_Meter_Sale_Price(Dirham) by")
    cat_cols = ["transaction_group", "property_type", "property_sub_type", "property_usage", 
                "landmark", "metro_station", "mall", "room_type","registration_type","procedure_name"]
    cat = st.selectbox("Select a categorical column:", cat_cols)
    plot_map = {
        "transaction_group":  "meter_sale_price&trans_group_en_plot.html",
        "property_type":  "meter_sale_price&property_type_en_plot.html",
        "property_sub_type": "meter_sale_price&property_sub_type_en_plot.html",
        "property_usage": "meter_sale_price&property_usage_en_plot.html",
        "metro_station": "meter_sale_price&nearest_metro_en_plot.html",
        "landmark": "meter_sale_price&nearest_landmark_en_plot.html",
        "mall": "meter_sale_price&nearest_mall_en_plot.html",
        "room_type": "meter_sale_price&rooms_en_plot.html",
        "registration_type" : "meter_sale_price&reg_type_en_plot.html",
        "procedure_name" : "meter_sale_price&procedure_name_en_plot.html"
        }
            
            

        
