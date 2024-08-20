import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data
def load_data():
    file_path = 'case_information.csv'  
    df = pd.read_csv(file_path)
    return df


def main():
    st.title('COVID-19 Regional Patient Tracking')
    
    
    df = load_data()
    
    
    if st.checkbox('Show raw data'):
        st.write(df.head())
    
    
    st.sidebar.header('Regions')
    selected_region = st.sidebar.selectbox('Select Region', df['region'].unique())
    filtered_df = df[df['region'] == selected_region]
    
    
    case_counts_by_muni = filtered_df['muni_city'].value_counts()
    muni_with_cases = case_counts_by_muni[case_counts_by_muni > 0].index
    filtered_df = filtered_df[filtered_df['muni_city'].isin(muni_with_cases)]
    
    case_counts_by_province = filtered_df['province'].value_counts()
    provinces_with_cases = case_counts_by_province[case_counts_by_province > 0].index
    filtered_df = filtered_df[filtered_df['province'].isin(provinces_with_cases)]

    
    categorical_columns = ['age_group', 'sex', 'status', 'province', 'muni_city', 'health_status', 'home_quarantined', 'pregnant', 'region']
    for col in categorical_columns:
        if col in df.columns:
            st.subheader(f'{col} Distribution')
            
            
            fig, ax = plt.subplots(figsize=(5, 3))  
            
            if col in ['province', 'muni_city']:
                
                sns.countplot(data=filtered_df, y=col, ax=ax, palette='viridis')
                plt.xlabel('Count', fontsize=8)
                plt.ylabel(col, fontsize=8)
                plt.yticks(fontsize=8)
            else:
                
                sns.countplot(data=filtered_df, x=col, ax=ax, palette='viridis')
                plt.xticks(rotation=90, fontsize=7)  # Rotate x-tick labels vertically
                plt.xlabel(col, fontsize=8)
                plt.ylabel('Count', fontsize=8)
                plt.yticks(fontsize=8)
            
            plt.title(f'Distribution of {col}', fontsize=10)
            plt.tight_layout()  
            st.pyplot(fig)

    
    numerical_columns = ['age']
    for col in numerical_columns:
        if col in df.columns:
            st.subheader(f'{col} Distribution')
            fig, ax = plt.subplots(figsize=(5, 3))  # Even smaller figure size
            sns.histplot(filtered_df[col].dropna(), bins=15, kde=True, ax=ax, color='blue')
            plt.xlabel(col, fontsize=8)
            plt.ylabel('Frequency', fontsize=8)
            plt.title(f'Distribution of {col}', fontsize=10)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            plt.tight_layout()  
            st.pyplot(fig)

    
    date_columns = ['date_announced', 'date_recovered', 'date_of_death', 'date_announced_as_removed', 'date_of_onset_of_symptoms']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')  # Convert to datetime, handle errors
            st.subheader(f'{col} Time Series')
            fig, ax = plt.subplots(figsize=(5, 3))  # Even smaller figure size
            date_counts = df[col].dropna().dt.date.value_counts().sort_index()
            date_counts.plot(ax=ax, marker='o', linestyle='-', color='green')
            plt.xlabel('Date', fontsize=8)
            plt.ylabel('Count', fontsize=8)
            plt.title(f'Count of {col} Over Time', fontsize=10)
            plt.xticks(rotation=45, fontsize=7)
            plt.yticks(fontsize=7)
            plt.tight_layout()  
            st.pyplot(fig)

if __name__ == "__main__":
    main()
