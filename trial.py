
import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# --- Page Config ---
st.set_page_config(layout="wide")
st.title("🔍 Dubai Real Estate Dashboard")




# --- File Paths ---
df_path = "target_df.csv"
area_stats_path = "df_area_plot_stats.xlsx"
cat_plot_path = "original_df_description_year.xlsx"
summary = "data_summary.xlsx"
sample = "sample_df.csv"
html_chart_path = "pareto_chart_by_area.html" 

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
st.sidebar.success("Main data loaded.")

# --- Remove Outliers ---
def remove_outliers(df, col):
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    return df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]

df_clean = df.copy()
for col in ['meter_sale_price', 'procedure_area']:
    if col in df_clean.columns:
        df_clean = remove_outliers(df_clean, col)

# --- Load Area Stats ---
df_area_plot_stats = load_excel(area_stats_path)
st.sidebar.success("Area stats loaded.")

# --- Sidebar Navigation ---
sidebar_option = st.sidebar.radio("Choose View", [
    "Data Preview",
    "Map Visualization",
    "Plots on Categorical Columns",
])

# --- View 1: Data Preview ---
if sidebar_option == "Data Preview":
    tab1, tab2, tab3 = st.tabs(["Preview", "Summary", "Box Plots"])
    
    try:
        with open(html_chart_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            components.html(html_content, height=600, scrolling=True)
    except FileNotFoundError:
        st.error(f"HTML chart file not found: {html_chart_path}")
        
    with tab1:
        sample_df = pd.read_csv(sample)
        st.subheader("📄 Original DF Preview")
        st.dataframe(sample_df)
        
    html_chart_path = "pareto_chart_by_area.html"  # replace with your actual file name

    try:
        with open(html_chart_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            components.html(html_content, height=600, scrolling=True)
    except FileNotFoundError:
        st.error(f"HTML chart file not found: {html_chart_path}")

    with tab2:
         st.subheader("📊 Overview Metrics")

        # CSS for button-style metrics
        st.markdown("""
            <style>
            .metric-button {
                display: inline-block;
                background-color: #4CAF50;
                color: white;
                padding: 15px 25px;
                font-size: 18px;
                border-radius: 10px;
                margin: 10px;
                text-align: center;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
            }
            .metric-label {
                font-weight: bold;
                font-size: 14px;
                margin-bottom: 5px;
                display: block;
            }
            </style>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
                <div class="metric-button">
                    <span class="metric-label">Total Records</span>
                    1,424,588
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
                <div class="metric-button" style="background-color:#2196F3;">
                    <span class="metric-label">Total Columns</span>
                    46
                </div>
            """, unsafe_allow_html=True)


        st.subheader("📋 Data Summary for Original DF")
        summary_df = pd.read_excel(summary)
        st.dataframe(summary_df)

        st.subheader("📋 Data Summary for Original DF")
        summary_df = pd.read_excel(summary)
        st.dataframe(summary_df)



    with tab3:
        st.subheader("📦 Box Plot Comparison: Original vs Cleaned Data")
        for col in ['procedure_area', 'meter_sale_price']:
            if col in df.columns and col in df_clean.columns:
                st.markdown(f"### 🔍 `{col}`")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Original**")
                    fig1 = go.Figure(go.Box(y=df[col], name='Original', boxmean='sd', marker_color='royalblue'))
                    fig1.update_layout(yaxis_title=col)
                    st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    st.markdown("**Cleaned**")
                    fig2 = go.Figure(go.Box(y=df_clean[col], name='Cleaned', boxmean='sd', marker_color='seagreen'))
                    fig2.update_layout(yaxis_title=col)
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning(f"Column `{col}` not found in both datasets.")

    if 'instance_year' in df.columns and 'meter_sale_price' in df.columns:
        st.markdown("### 📊 Avg Meter Sale Price & Distribution by Instance Year (Original Data)")

        # Group data
        agg_data = df.groupby('instance_year')['meter_sale_price'].agg(['mean', 'count']).reset_index()

        # Create subplot with secondary y-axis
        fig_combo = make_subplots(specs=[[{"secondary_y": True}]])

        # Add line plot for average price
        fig_combo.add_trace(
            go.Scatter(
            x=agg_data['instance_year'],
            y=agg_data['mean'],
            name="Avg Meter Sale Price",
            mode="lines+markers",
            line=dict(color='darkorange')
            ),
            secondary_y=False,
        )

        # Add bar plot for count (distribution)
        fig_combo.add_trace(
            go.Bar(
            x=agg_data['instance_year'],
            y=agg_data['count'],
            name="Count",
            marker_color='lightskyblue',
            opacity=0.6
            ),
            secondary_y=True,
        )

        # Set axis titles
        fig_combo.update_layout(
            xaxis_title="Instance Year",
            title="Avg Meter Sale Price and Count per Year",
            legend=dict(x=0.5, y=1.1, orientation='h', xanchor='center'),
        )

        fig_combo.update_yaxes(title_text="Avg Meter Sale Price", secondary_y=False)
        fig_combo.update_yaxes(title_text="Count", secondary_y=True)

        #    Display plot
        st.plotly_chart(fig_combo, use_container_width=True)



# --- View 2: Map Visualization ---
elif sidebar_option == "Map Visualization":
    st.subheader("📍 Dubai Area-wise Avg. Meter Sale Price and Transaction Count")
    required_cols = {'area_lat', 'area_lon', 'Transaction Count', 'Average Meter Sale Price', 'area_name_en'}

    if required_cols.issubset(df_area_plot_stats.columns):
        fig = px.scatter_mapbox(
            df_area_plot_stats,
            lat='area_lat',
            lon='area_lon',
            size='Transaction Count',
            color='Average Meter Sale Price',
            hover_name='area_name_en',
            hover_data={'Transaction Count': True, 'Average Meter Sale Price': ':.2f'},
            color_continuous_scale='Viridis',
            size_max=30,
            zoom=9
        )
        fig.update_layout(mapbox_style='open-street-map', margin={"r": 0, "t": 40, "l": 0, "b": 0})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Required columns not found in area stats file.")

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
