import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


# --- Page Config ---

st.markdown("""
    <style>
    [data-testid ="collapsedControl"]{
    position: fixed;
    top: 500px;
    left: 50px;
    z-index: 200;
    }
    .block-container {
    padding-top: 2.5rem;
    padding-bottom: 1rem;
    }
        
    </style>
""", unsafe_allow_html=True)
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
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
sidebar_option = st.sidebar.radio("Choose View", [
    "Data Summary",
    "Pareto Analysis",
    "Univariate Analysis",
    "Bivariate Analysis",
    "Geo Graphical Analysis",
    "Price Prediction Model"
])

# --- View 1: Data Summary ---
if sidebar_option == "Data Summary":
    st.subheader("📄 Transactions Data")
    tab1, tab2, tab3 = st.tabs(["Preview", "Summary","Notes"])
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

    with tab3:
        notes = "notes.xlsx"
        notes_df = pd.read_excel(notes)
        if 'nRecords' in notes_df.columns:
            notes_df['nRecords'] = notes_df['nRecords'].apply(lambda x: f"{x:,}" if pd.notnull(x) else x)
            st.dataframe(notes_df)
        

# --- View 2: Pareto Analysis ---
elif sidebar_option == "Pareto Analysis":
    st.markdown("### Pareto Analysis by Area_name_en")

    try:
        pereto_file = "pereto_analysis_file.xlsx"
        pereto_analyis = pd.ExcelFile(pereto_file)
        pereto_sheet_names = pereto_analyis.sheet_names
    except FileNotFoundError:
        st.error(f"File not found: {pereto_file}")
        st.stop()

    # Read all sheets
    all_sheets_df = pd.read_excel(pereto_analyis, sheet_name=pereto_sheet_names)

    # Extract specific sheets
    pareto_summary = all_sheets_df["Pereto_Analysis_by_area_name"]
    ABC_summary = all_sheets_df["ABC_Area_name"]

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Table", "Chart", "ABC summary"])

    with tab1:
        #st.markdown("### Pareto Analysis by Area_name_en")
        pareto_summary.rename(columns={'Cum%_areas': 'Cum%_Areas'}, inplace=True)
        pareto_summary.rename(columns={'Percentage(%)': '%_nRecords'}, inplace=True) 
        pareto_summary.rename(columns={'Cumulative_%': 'Cum%_Records'}, inplace=True)
        pareto_summary['nRecords'] = pareto_summary['nRecords'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else x)
        pareto_summary['Cum%_Records'] = pareto_summary['Cum%_Records'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else x)
        pareto_summary['%_nRecords'] = pareto_summary['%_nRecords'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else x)
        pareto_summary['Cum%_Areas'] = pareto_summary['Cum%_Areas'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else x)
        pareto_summary.index = range(1, len(pareto_summary) + 1)
        st.dataframe(pareto_summary, use_container_width=True)

    with tab2:
        # Load Excel data
        excel_file_path = "pereto_analysis_only.xlsx"
        df2 = pd.read_excel(excel_file_path)

        # Remove any row where area_name_en is 'Total' (case-insensitive)
        df2 = df2[~df2['area_name_en'].str.strip().str.lower().eq('total')]

        # Sort and calculate cumulative values
        df2_sorted = df2.sort_values(by='nRecords', ascending=False).reset_index(drop=True)
        df2_sorted['Cumulative_nRecords'] = df2_sorted['nRecords'].cumsum()
        df2_sorted['Cumulative_%'] = (df2_sorted['Cumulative_nRecords'] / df2_sorted['nRecords'].sum()) * 100

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add bar for nRecords
        fig.add_trace(
            go.Bar(
                name='nRecords',
                x=df2_sorted['area_name_en'],
                y=df2_sorted['nRecords'],
                marker_color='blue',
                hovertemplate='<b>%{x}</b><br>nRecords: %{y}<extra></extra>'
            ),
            secondary_y=False,
        )

        # Add line for Cumulative %
        fig.add_trace(
            go.Scatter(
                name='Cumulative_%',
                x=df2_sorted['area_name_en'],
                y=df2_sorted['Cumulative_%'],
                mode='lines',
                marker_color='red',
                hovertemplate='<b>%{x}</b><br>Cumulative %: %{y:.2f}%<extra></extra>'
            ),
            secondary_y=True,
        )

        # Axis settings
        fig.update_xaxes(title_text='area_name_en')

        # Set fixed linear y-axis for better scaling
        fig.update_yaxes(
            title_text='nRecords',
            tickvals=np.arange(0, 100001, 20000),
            range=[0, 100000],
            secondary_y=False
        )
        fig.update_yaxes(title_text='Cumulative %', secondary_y=True)

        # Add breakdown lines at specified areas
        wadi_safa_index = df2_sorted[df2_sorted['area_name_en'] == 'Wadi Al Safa 5'].index
        al_hebiah_index = df2_sorted[df2_sorted['area_name_en'] == 'Al Hebiah Third'].index

        if not wadi_safa_index.empty:
            fig.add_vline(
                x=wadi_safa_index[0],
                line_dash="dash",
                line_color="green",
                #annotation_text="Wadi Al Safa 5 (40%)",
                #annotation_position="top"
            )

        if not al_hebiah_index.empty:
            fig.add_vline(
                x=al_hebiah_index[0],
                line_dash="dash",
                line_color="purple",
                #annotation_text="Al Hebiah Third (70%)",
                #annotation_position="top"
            )

        # Layout settings
        fig.update_layout(
            title_text='Pareto Analysis by Area',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified',
            height=600,
            barmode='group'
        )

        # Display chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)


    with tab3:
        st.markdown("ABC Table")
        ABC_summary.rename(columns={'Cum%_records': 'Cum%_Records'}, inplace=True)
        ABC_summary.rename(columns={'Cum%_areas': 'Cum%_Areas'}, inplace=True)
        ABC_summary.rename(columns={'Group_name': 'Group'}, inplace=True)

        ABC_summary['nRecords'] = ABC_summary['nRecords'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else x)
        ABC_summary['%Area'] = ABC_summary['%Area'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else x)
        ABC_summary['%Records '] = ABC_summary['%Records '].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else x)
        ABC_summary['Cum%_Records'] = ABC_summary['Cum%_Records'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else x)
        ABC_summary['Cum%_Areas'] = ABC_summary['Cum%_Areas'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else x)

        ABC_summary.index = range(1, len(ABC_summary) + 1)

        # Swap the columns
        cols = list(ABC_summary.columns)
        i, j = cols.index('Cum%_Records'), cols.index('Cum%_Areas')
        cols[i], cols[j] = cols[j], cols[i]
        ABC_summary = ABC_summary[cols]

        st.dataframe(ABC_summary, use_container_width=True)

        
        df = ABC_summary
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(name='%Area', x=df['Group'], y=df['%Area'], marker_color='skyblue',
                   hovertemplate='<b>%{x}</b><br>%Area: %{y:.2f}%<extra></extra>'),
            secondary_y=False)
        fig.add_trace(
            go.Bar(name='%Records', x=df['Group'], y=df['%Records '], marker_color='lightcoral',
                   hovertemplate='<b>%{x}</b><br>%Records: %{y:.2f}%<extra></extra>'),
            secondary_y=False)
        fig.add_trace(
            go.Scatter(name='Cum%_records', x=df['Group'], y=df['Cum%_Records'], mode='lines+markers',
                       marker_color='green',
                       hovertemplate='<b>%{x}</b><br>Cum% Records: %{y:.2f}%<extra></extra>'),
            secondary_y=True)
        fig.add_trace(
            go.Scatter(name='Cum%_areas', x=df['Group'], y=df['Cum%_Areas'], mode='lines+markers',
                       marker_color='darkorange',
                       hovertemplate='<b>%{x}</b><br>Cum% Areas: %{y:.2f}%<extra></extra>'),
            secondary_y=True)
        fig.update_xaxes(title_text='Group')
        fig.update_yaxes(title_text='Counts (%Area, %Records)', secondary_y=False)
        fig.update_yaxes(title_text='Cumulative Percentage', secondary_y=True)
        fig.update_layout(
            title_text='ABC chart',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified'
        )
        st.plotly_chart(fig)
        
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

    main_tabs = st.tabs([ "Dimensions","Metrics"])

    with main_tabs[0]:
        # Select sheet before tabs
        selected_sheet = st.selectbox("Distribution of nRecords by", sheet_names)
        df = pd.read_excel(xls, sheet_name=selected_sheet)
        col1 = df.columns[0]  # Category column
        #st.markdown("### 📊 Bar Plot (nRecords)")
        if "nRecords" in df.columns:
            fig_bar = px.bar(df, x=col1, y="nRecords", title=f"nRecords by {col1}", color=col1)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("'nRecords' column not found.")
    with main_tabs[1]:
        # Dropdown for selecting the category column
        cat_cols = ["meter_sale_price", "procedure_area"]
        cat = st.selectbox("Select the metrics column:", cat_cols)

        # Create sub-tabs under the selected category
        sub_tabs = st.tabs(["Table", "Histogram", "Boxplot"])

        # 1️⃣ TABLE TAB
        with sub_tabs[0]:
            # Define mapping of category to list of files
            table_files = {
                "meter_sale_price": [
                    "meter_sale_price_table_final.xlsx",
                    "bin_df_manual.xlsx"
                ],
                "procedure_area": [
                    "procedure_area_table_final.xlsx",
                    "bin_df_Procedure_area_manual_xyz.xlsx"
                ]
            }

            # Get the selected category (ensure 'cat' is assigned earlier in your code)
            selected_tables = table_files.get(cat)

            if selected_tables:
                for table_file in selected_tables:
                    try:
                        df = pd.read_excel(table_file)

                        # Apply comma formatting ONLY to bin files
                        if "bin_df" in table_file and "nRecords" in df.columns:
                            df['nRecords'] = df['nRecords'].apply(lambda x: f"{x:,}")

                        # Display the table
                        #st.markdown(f"#### Displaying: `{table_file}`")
                        st.dataframe(df, use_container_width=True)

                    except FileNotFoundError:
                        st.error(f"File not found: {table_file}")
                    except Exception as e:
                        st.error(f"Error reading `{table_file}`: {e}")


        # 2️⃣ HISTOGRAM TAB
        with sub_tabs[1]:
            # Mapping for bar chart Excel files (inside tab for clarity)
            plot_bar = {
                "meter_sale_price": "bin_df_manual.xlsx",
                "procedure_area": "bin_df_Procedure_area_manual_xyz.xlsx"
            }

            selected_bar = plot_bar.get(cat)
            if selected_bar:
                try:
                    df_bar = pd.read_excel(selected_bar)
                    #st.markdown(f"### Barchart for `{cat}`")
                    fig = px.bar(
                        df_bar,
                        x= "bin_range",
                        y="nRecords",
                        #labels={"meter_sale_price": "meter_sale_price", "nRecords": "Number of Records"},
                        title=f"Distribution of {cat.replace('_', ' ').title()}",
                        text_auto=True)
                    # Add black border and control bar width
                    fig.update_traces(marker_line_color='black', marker_line_width=1)

                    # Optional: customize layout
                    fig.update_layout(
                        #xaxis_title="meter_sale_price",
                        #yaxis_title="Number of Records",
                        bargap=0,  # Adjust space between bars
                        height=500
                        )
                    st.plotly_chart(fig, use_container_width=True)
                except FileNotFoundError:
                    st.error(f"File not found: {selected_bar}")
                except Exception as e:
                    st.error(f"Error creating bar chart: {e}")

        with sub_tabs[2]:
            # Mapping for boxplot HTML files
            plot_box = {
                "meter_sale_price": "meter_sale_price_with_boxplot.html",
                "procedure_area": "procedure_area_with_boxplot.html"
            }

            # Mapping for corresponding PNG images
            plot_images = {
                "meter_sale_price": "boxplot_meter_sale_price_raw.png",
                "procedure_area": "boxplot_procedure_area_raw.png"
            }

            selected_file = plot_box.get(cat)
            selected_image = plot_images.get(cat)

            # Adjusting columns: 2 for image (col1), 3 for boxplot (col2)
            col1, col2 = st.columns([2, 3])

            with col1:
                if selected_image:
                    try:
                        # Display image with automatic scaling to container width
                        st.image(selected_image, use_container_width=True)
                    except FileNotFoundError:
                        st.error(f"Image not found: {selected_image}")
                    except Exception as e:
                        st.error(f"Error loading image: {e}")

            with col2:
                if selected_file:
                    try:
                        with open(selected_file, "r") as file:
                            html_content = file.read()
                            components.html(html_content, height=500, width=800, scrolling=True)
                    except FileNotFoundError:
                        st.error(f"File not found: {selected_file}")
                    except Exception as e:
                        st.error(f"Error loading boxplot HTML: {e}")

        
                    
# --- View 3: Bivariate Analysis  ---
if sidebar_option == "Bivariate Analysis":
    
    # Step 1: Dropdown selector at the top
    cat_cols = [
        "trans_group_en", "property_type_en", "property_sub_type_en", "property_usage_en", 
        "nearest_metro_en","nearest_landmark_en","nearest_mall_en", "room_en", "reg_type_en", 
        "procedure_name_en","instance_year"
    ]
    cat = st.selectbox("nRecords and Avg_Meter_Sale_Price (Dirham) by:", cat_cols)
    main_tabs = st.tabs([ "Table","charts"])
    with main_tabs[1]:
        # Step 3: Read the Excel for box plot data
        try:
            cat_plot_path = "original_df_description_tables.xlsx"
            xls = pd.ExcelFile(cat_plot_path)
            sheet_names = xls.sheet_names
            selected_sheet = sheet_names[cat_cols.index(cat)]  # Optional: auto match sheet to cat
            df = pd.read_excel(xls, sheet_name=selected_sheet)
        except FileNotFoundError:
            st.error(f"Excel file not found: {cat_plot_path}")
            st.stop()
        except Exception as e:
            st.error(f"Error loading Excel sheet: {e}")
            st.stop()

        # Step 4: Display two columns
        col1, col2 = st.columns(2)

    with col1:
        # Function to create overlay plot for a single sheet
        def plot_avg_price_and_count_overlay(df1, df2, category_col, labels=("Raw data", "Model_Data")):
            """
            Returns a Plotly figure showing average meter sale price and record count overlay for two DataFrames.
            """
            target_col = "Avg_meter_sale_price"

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # First DataFrame
            fig.add_trace(go.Bar(
                x=df1[category_col],
                y=df1['nRecords'],
                name=f'nRecords ({labels[0]})',
                opacity=0.6
            ), secondary_y=False)

            fig.add_trace(go.Scatter(
                x=df1[category_col],
                y=df1[target_col],
                mode='lines+markers',
                name=f'Avg Price ({labels[0]})'
            ), secondary_y=True)

            # Second DataFrame
            fig.add_trace(go.Bar(
                x=df2[category_col],
                y=df2['nRecords'],
                name=f'nRecords ({labels[1]})',
                opacity=0.6
            ), secondary_y=False)

            fig.add_trace(go.Scatter(
                x=df2[category_col],
                y=df2[target_col],
                mode='lines+markers',
                name=f'Avg Price ({labels[1]})'
            ), secondary_y=True)

            # Layout
            fig.update_layout(
                title=(f'nRecords & Avg Price'),
                xaxis_title=category_col,
                yaxis=dict(title='nRecords'),
                yaxis2=dict(title='Average Meter Sale Price', overlaying='y', side='right'),
                legend=dict(x=0.99, y=1.2),
                hovermode='x unified',
                barmode='group'
            )

            return fig
            
        file1 = "description_raw.xlsx"
        file2 = "description_units20.xlsx"

        if file1 and file2:
                raw_excel = pd.read_excel(file1, sheet_name=None)
                model_excel = pd.read_excel(file2, sheet_name=None)

                #common_sheets = sorted(set(raw_excel.keys()) & set(model_excel.keys()))

                if selected_sheet:
                        # Load data from the selected sheet
                        df1 = raw_excel[selected_sheet]
                        df2 = model_excel[selected_sheet]

                        if len(df1.columns) > 0:
                            category_col = df1.columns[0]
                            fig = plot_avg_price_and_count_overlay(df1, df2, category_col)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"⚠️ Sheet '{selected_sheet}' has no columns to plot.")
        else:
                st.info("Upload both Excel files to continue.")


    with col2:
        def plot_boxplot_per_category(df, cat_col):
            required_cols = {'min', '25%', '50%', '75%', 'max'}
            if not required_cols.issubset(df.columns):
                st.warning("DataFrame missing required quantile columns.")
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
                    name=str(category),  # Label on x-axis
                    q1=[q1],
                    median=[median],
                    q3=[q3],
                    lowerfence=[lower_fence],
                    upperfence=[upper_fence],
                    boxpoints=False,
                ))

            fig.update_layout(
                title=("Distribution of meter_sale_price"),
                yaxis_title="Meter Sale Price",
                boxmode='group',
                xaxis_title=cat_col,
                xaxis=dict(tickangle=45, automargin=True),  # Label rotation
            )
            return fig

        box_plot = plot_boxplot_per_category(df, df.columns[0])
        if box_plot:
            st.plotly_chart(box_plot, use_container_width=True)
            
    with main_tabs[0]:
        note = "notes.xlsx"
        note_df = pd.read_excel(note)
        st.markdown("Data Explaination")
        st.dataframe(note_df)
        try:
            # Load the raw description data from the corresponding sheet
            description_data = pd.read_excel("table_stats_bivariate.xlsx", sheet_name=cat)
            description_data['Avg_meter_sale_price'] = description_data['Avg_meter_sale_price'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else x)
            description_data['q1'] = description_data['q1'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else x)
            description_data['Median'] = description_data['Median'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else x)
            description_data['q3'] = description_data['q3'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else x)
            #st.subheader(f"Raw Description Table - {cat}")
            st.dataframe(description_data, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load data for table view: {e}")


        
        
        
        
# --- View 5: Price Prediction Model ---
# Define file paths
EXCEL_PATH = "Over_all_output_final.xlsx"
model_perfomance =  "Model_performance.xlsx"
html_lr = "predicted_vs_actual_linear.html"
html_dt = "predicted_vs_actual_decision_tree.html"
html_xgb = "predicted_vs_actual_XGB_regressor.html"
html_comparision = "model_perfor_comparision.html"

# Load Excel file with caching
@st.cache_data
def load_excel(path):
    xls = pd.ExcelFile(path)
    sheets = xls.sheet_names
    data = {sheet: xls.parse(sheet) for sheet in sheets}
    return data


# === Sidebar Selection ===
if sidebar_option == "Price Prediction Model":

    # === Top-Level Tabs ===
    if st.sidebar.button("Show Data Preparation Details"):
        st.markdown("""
            - Data used for model is based on the following:
                - Outliers removed using `meter_sale_price` and `procedure_area` columns.
                - From outliers-removed data, we have considered data from the year **2020**.
                    - For the model, we have used data with property type **"Units"**.
            - We had a large number of independent variables in the dataset.
            - To identify the most relevant predictors, we applied a **stepwise regression model**.
            - This method helped us select the best combination of input variables for modeling.
            - Using these selected variables, we built the final model and obtained the results.
            """)
    main_tabs = st.tabs(["📈 Model Performance Tables","📉 Prediction Model Visuals"])
    
    # === Tab 1: Prediction Model Visuals ===
    with main_tabs[1]:
        if os.path.exists(EXCEL_PATH):
            xl = pd.ExcelFile(EXCEL_PATH)
            if 'nObservations' in df.columns:
                df['nObservations'] = df['nObservations'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else x)
                if 'MAPE' in df.columns:
                    df['MAPE'] = df['MAPE'].apply(lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else x)
                    df.index = range(1, len(df) + 1)
                    st.subheader(f"📊 {first_sheet_name}")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("The Excel file has less than 2 sheets.")
            
        st.subheader("🔍 Overall Comparison Report")
        if os.path.exists(html_comparision):
            with open(html_comparision, "r", encoding="utf-8") as f:
                components.html(f.read(), height=300, scrolling=True)
        else:
            st.warning(f"Comparison HTML not found at: {html_comparision}")

        st.subheader("📊 Logistic Regression")
        st.markdown("###Equation : Predicted_price = 0.40134 * Actual_price + 8966.97")
        if os.path.exists(html_lr):
            with open(html_lr, "r", encoding="utf-8") as f:
                components.html(f.read(), height=400, scrolling=True)
        else:
            st.warning(f"Logistic Regression HTML not found at: {html_lr}")

        st.subheader("🌳 Decision Tree")
        st.markdown("###Equation : Predicted_price = 0.465166 * Actual_price + 7993.22")
        if os.path.exists(html_dt):
            with open(html_dt, "r", encoding="utf-8") as f:
                components.html(f.read(), height=400, scrolling=True)
        else:
            st.warning(f"Decision Tree HTML not found at: {html_dt}")

        st.subheader("🚀 XGBoost")
        st.markdown("###Equation : Predicted_price = 0.463650 * Actual_price + 8055.86")
        if os.path.exists(html_xgb):
            with open(html_xgb, "r", encoding="utf-8") as f:
                components.html(f.read(), height=400, scrolling=True)
        else:
            st.warning(f"XGBoost HTML not found at: {html_xgb}")

    # === Tab 3: Area & Sector Sheets ===
    with main_tabs[0]:
            
        Over_all, sector_tab,area_tab = st.tabs(["Over All","Sector wise","Area wise"])
        with Over_all:
            abc = "Over_all_output_final.xlsx"

            # Read the only sheet directly
            df = pd.read_excel(abc)

            # Format 'MAPE' as percentage string
            if 'MAPE' in df.columns:
                df['MAPE'] = df['MAPE'].apply(lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else x)

            # Format 'nObservations' with commas
            if 'nObservations' in df.columns:
                df['nObservations'] = df['nObservations'].apply(lambda x: f"{x:,}" if pd.notnull(x) else x)

            # Display the final dataframe
            st.dataframe(df, use_container_width=True)

            
           
        with sector_tab:
            pqr = "sector_name_Output.xlsx"

            # Read all sheets
            sector_sheets = pd.read_excel(pqr, sheet_name=None)

            if sector_sheets:
                # Process each sheet
                for sheet_name in sector_sheets:
                    df = sector_sheets[sheet_name]
            
                    # Format 'MAPE' as percentage string
                    if 'MAPE' in df.columns:
                        df['MAPE'] = df['MAPE'].apply(lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else x)

                    # Format 'nObservations' with commas
                    if 'nObservations' in df.columns:
                        df['nObservations'] = df['nObservations'].apply(lambda x: f"{x:,}" if pd.notnull(x) else x)

                    sector_sheets[sheet_name] = df  # Update in dictionary

                # Display each sheet in a tab
                sector_tabs = st.tabs(list(sector_sheets.keys()))
                for tab, (sheet_name, df) in zip(sector_tabs, sector_sheets.items()):
                    with tab:
                        st.dataframe(df, use_container_width=True)
        
        with area_tab:
            xyz = "area_wise_file_final.xlsx"

            # Read all sheets
            area_sheets = pd.read_excel(xyz, sheet_name=None)

            if area_sheets:
                for sheet_name in area_sheets:
                    df = area_sheets[sheet_name]
            
                    # Convert MAPE to percentage
                    if 'MAPE' in df.columns:
                        df['MAPE'] = df['MAPE'].apply(lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else x)

                    # Format nObservations with commas
                    if 'nObservations' in df.columns:
                        df['nObservations'] = df['nObservations'].apply(lambda x: f"{x:,}" if pd.notnull(x) else x)

                    area_sheets[sheet_name] = df  # Update back in dict

                # Create tabs and display each sheet
                area_tabs = st.tabs(list(area_sheets.keys()))
                for tab, (sheet_name, df) in zip(area_tabs, area_sheets.items()):
                    with tab:
                        st.dataframe(df, use_container_width=True)

                    
                

# --- View 6: Geo Graphical Analysis ---
if sidebar_option == "Geo Graphical Analysis":
    st.subheader("Dubai Area-wise Bubble Map")
    
    df_excel = pd.read_excel("new_tdf.xlsx")
    units_excel = pd.read_excel("units_20.xlsx")
    outlier_excel = pd.read_excel("outliers.xlsx")  # Replace with your actual outlier dataset
    tab1, = st.tabs(["Average Meter Sale Price"])

    # Create the single tab
    with tab1:

        # Add filtered data (e.g., >= 2020)
        figs = px.scatter_mapbox(
            units_excel,
            lat='area_lat',
            lon='area_lon',
            size='Transaction Count',
            color='Average Meter Sale Price',
            hover_name='area_name_en',
            hover_data={
                'Transaction Count': True,
                'Average Meter Sale Price': ':.2f',
                'area_lat': False,
                'area_lon': False
            },
            color_continuous_scale='Hot',
            size_max=30,
            zoom=9,
            title="Dubai Area-wise Average Meter Sale Price(Dirham) and Transaction Count"
        )

        for trace in figs.data:
            trace.name = "Model_Data"
            trace.legendgroup = "Model_Data"
            trace.showlegend = True
            
            

        # Add filtered data (e.g., >= 2020)
        fig2 = px.scatter_mapbox(
            df_excel,
            lat='area_lat',
            lon='area_lon',
            size='Transaction Count',
            color='Average Meter Sale Price',
            hover_name='area_name_en',
            hover_data={
                'Transaction Count': True,
                'Average Meter Sale Price': ':.2f',
                'area_lat': False,
                'area_lon': False
            },
            color_continuous_scale='Hot',
            size_max=30,
            opacity=0.8,
            zoom=9,
        )

        for trace in fig2.data:
            trace.name = "Raw data"
            trace.legendgroup = "Raw data"
            trace.showlegend = True
            figs.add_trace(trace)
            

        # Add outlier data
        fig3 = px.scatter_mapbox(
            outlier_excel,
            lat='area_lat',
            lon='area_lon',
            size='Transaction Count',
            color='Average Meter Sale Price',
            hover_name='area_name_en',
            hover_data={
                'Transaction Count': True,
                'Average Meter Sale Price': ':.2f',
                'area_lat': False,
                'area_lon': False
            },
            color_continuous_scale='Hot',
            size_max=30,
            opacity=0.7,
            zoom=9
        )

        for trace in fig3.data:
            trace.name = "Non_Model_Data"
            trace.legendgroup = "Non_Model_Data"
            trace.showlegend = True
            figs.add_trace(trace)

        figs.update_layout(
            mapbox_style='open-street-map',
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            )
        )

        st.plotly_chart(figs, use_container_width=True)
        st.markdown(
            """
            <div style="text-align: left; font-size: 15px; margin-top: 10px;">
                <b>Size of bubble</b> = Number of Transactions &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; 
                <b>Colour of bubble</b> = Average Meter Sale Price
            </div>
            """,
            unsafe_allow_html=True
        )
    


    

    

 
            
            

        
