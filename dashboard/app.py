import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import duckdb
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="Ocean Data Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .data-quality-good {
        color: #2e7d32;
        font-weight: bold;
    }
    
    .data-quality-warning {
        color: #f57f17;
        font-weight: bold;
    }
    
    .data-quality-error {
        color: #c62828;
        font-weight: bold;
    }
    
    .sidebar-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data_from_db():
    """Load data from DuckDB database with caching."""
    db_path = Path("data/ocean_data.duckdb")
    
    if not db_path.exists():
        st.error("❌ Database not found! Please run the pipeline first: `python run_pipeline.py`")
        st.stop()
    
    try:
        conn = duckdb.connect(str(db_path))
        
        # Get all data
        df = conn.execute("SELECT * FROM sea_surface ORDER BY time").df()
        
        # Get database metadata
        row_count = conn.execute("SELECT COUNT(*) FROM sea_surface").fetchone()[0]
        table_info = conn.execute("DESCRIBE sea_surface").fetchall()
        
        conn.close()
        
        # Convert time column to datetime if it's not already
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        
        return df, row_count, table_info
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

def create_map_visualization(df):
    """Create an interactive map showing data collection points."""
    if df.empty or 'latitude' not in df.columns or 'longitude' not in df.columns:
        st.warning("⚠️ No geographic data available for mapping")
        return None
    
    # Calculate map center
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    # Create folium map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,
        tiles='OpenStreetMap'
    )
    
    # Add data points
    for idx, row in df.iterrows():
        # Create popup content
        popup_content = f"""
        <div style="width: 200px;">
            <h4>🌊 Ocean Data Point</h4>
            <b>📍 Location:</b> {row['latitude']:.2f}°N, {row['longitude']:.2f}°W<br>
            <b>🌡️ Temperature:</b> {row['temperature']:.2f}°C<br>
            <b>🧂 Salinity:</b> {row['salinity']:.2f} PSU<br>
            <b>📏 Depth:</b> {row['depth']:.1f}m<br>
            <b>📅 Time:</b> {row['time'].strftime('%Y-%m-%d') if pd.notna(row['time']) else 'N/A'}
        </div>
        """
        
        # Color code by temperature
        if pd.notna(row['temperature']):
            temp = row['temperature']
            if temp < 15:
                color = 'blue'
            elif temp < 25:
                color = 'green'
            else:
                color = 'red'
        else:
            color = 'gray'
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            popup=folium.Popup(popup_content, max_width=250),
            color='black',
            weight=1,
            fillColor=color,
            fillOpacity=0.7,
            tooltip=f"Temp: {row['temperature']:.1f}°C | Salinity: {row['salinity']:.1f} PSU"
        ).add_to(m)
    
    # Don't add legend to map - will be displayed separately
    return m

def create_time_series_plots(df):
    """Create time series plots for temperature and salinity."""
    if df.empty or 'time' not in df.columns:
        st.warning("⚠️ No time series data available")
        return None
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('🌡️ Sea Surface Temperature Over Time', '🧂 Sea Surface Salinity Over Time'),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Temperature plot
    if 'temperature' in df.columns and df['temperature'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['temperature'],
                mode='lines+markers',
                name='Temperature',
                line=dict(color='#ff6b6b', width=3),
                marker=dict(size=8, color='#ff6b6b'),
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Temperature: %{y:.2f}°C<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Salinity plot
    if 'salinity' in df.columns and df['salinity'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['salinity'],
                mode='lines+markers',
                name='Salinity',
                line=dict(color='#4ecdc4', width=3),
                marker=dict(size=8, color='#4ecdc4'),
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Salinity: %{y:.2f} PSU<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="📈 Ocean Data Time Series Analysis",
        title_x=0.5,
        title_font_size=20,
        hovermode='x unified'
    )
    
    # Update x-axes
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    # Update y-axes
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Salinity (PSU)", row=2, col=1)
    
    return fig

def create_data_quality_metrics(df, row_count, table_info):
    """Create data quality metrics and statistics."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="📊 Total Records",
            value=f"{row_count:,}",
            help="Total number of data points in the database"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        completeness = (df.notna().sum().sum() / (len(df) * len(df.columns)) * 100) if not df.empty else 0
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="✅ Data Completeness",
            value=f"{completeness:.1f}%",
            help="Percentage of non-null values across all fields"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        if 'time' in df.columns and not df.empty:
            time_span = (df['time'].max() - df['time'].min()).days if len(df) > 1 else 0
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric(
                label="📅 Time Span",
                value=f"{time_span} days",
                help="Time range covered by the dataset"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric(label="📅 Time Span", value="N/A")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        db_path = Path("data/ocean_data.duckdb")
        db_size = db_path.stat().st_size / 1024 if db_path.exists() else 0
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="💾 Database Size",
            value=f"{db_size:.1f} KB",
            help="Size of the DuckDB database file"
        )
        st.markdown('</div>', unsafe_allow_html=True)

def create_statistical_summary(df):
    """Create statistical summary of the data."""
    if df.empty:
        st.warning("⚠️ No data available for statistical analysis")
        return
    
    st.subheader("📊 Statistical Summary")
    
    # Numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        # Create summary statistics
        summary_stats = df[numeric_cols].describe()
        
        # Display as a formatted table
        st.dataframe(
            summary_stats.round(3),
            use_container_width=True
        )
        
        # Create distribution plots for key variables
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'temperature' in df.columns:
                    fig_temp = px.histogram(
                        df, 
                        x='temperature',
                        title='🌡️ Temperature Distribution',
                        nbins=20,
                        color_discrete_sequence=['#ff6b6b']
                    )
                    fig_temp.update_layout(height=300)
                    st.plotly_chart(fig_temp, use_container_width=True)
            
            with col2:
                if 'salinity' in df.columns:
                    fig_sal = px.histogram(
                        df, 
                        x='salinity',
                        title='🧂 Salinity Distribution',
                        nbins=20,
                        color_discrete_sequence=['#4ecdc4']
                    )
                    fig_sal.update_layout(height=300)
                    st.plotly_chart(fig_sal, use_container_width=True)

def create_real_time_query_interface(df):
    """Create an interface for real-time database queries."""
    st.subheader("🔍 Real-time Data Explorer")
    
    with st.expander("📋 Custom Database Query", expanded=False):
        st.write("Execute custom SQL queries on the ocean data:")
        
        # Predefined queries
        predefined_queries = {
            "All Data": "SELECT * FROM sea_surface ORDER BY time",
            "Temperature > 20°C": "SELECT * FROM sea_surface WHERE temperature > 20 ORDER BY temperature DESC",
            "High Salinity": "SELECT * FROM sea_surface WHERE salinity > 35 ORDER BY salinity DESC",
            "Data Summary": "SELECT COUNT(*) as total_records, AVG(temperature) as avg_temp, AVG(salinity) as avg_salinity FROM sea_surface"
        }
        
        query_choice = st.selectbox("Choose a predefined query:", list(predefined_queries.keys()))
        
        if st.button("🚀 Execute Query"):
            try:
                db_path = Path("data/ocean_data.duckdb")
                conn = duckdb.connect(str(db_path))
                
                result = conn.execute(predefined_queries[query_choice]).df()
                conn.close()
                
                st.success(f"✅ Query executed successfully! Found {len(result)} results.")
                st.dataframe(result, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Query failed: {str(e)}")
    
    # Data filtering interface
    with st.expander("🎛️ Interactive Data Filters", expanded=True):
        if not df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'temperature' in df.columns and df['temperature'].notna().any():
                    temp_min = float(df['temperature'].min())
                    temp_max = float(df['temperature'].max())
                    
                    # Handle case where min == max (single data point)
                    if temp_min == temp_max:
                        st.info(f"🌡️ Temperature: {temp_min:.2f}°C (single value)")
                        temp_range = (temp_min, temp_max)
                    else:
                        temp_range = st.slider(
                            "🌡️ Temperature Range (°C)",
                            temp_min,
                            temp_max,
                            (temp_min, temp_max),
                            step=0.1
                        )
                else:
                    st.warning("⚠️ No temperature data available")
                    temp_range = None
                
            with col2:
                if 'salinity' in df.columns and df['salinity'].notna().any():
                    sal_min = float(df['salinity'].min())
                    sal_max = float(df['salinity'].max())
                    
                    # Handle case where min == max (single data point)
                    if sal_min == sal_max:
                        st.info(f"🧂 Salinity: {sal_min:.2f} PSU (single value)")
                        sal_range = (sal_min, sal_max)
                    else:
                        sal_range = st.slider(
                            "🧂 Salinity Range (PSU)",
                            sal_min,
                            sal_max,
                            (sal_min, sal_max),
                            step=0.1
                        )
                else:
                    st.warning("⚠️ No salinity data available")
                    sal_range = None
            
            # Apply filters
            filtered_df = df.copy()
            if 'temperature' in df.columns and temp_range is not None:
                filtered_df = filtered_df[
                    (filtered_df['temperature'] >= temp_range[0]) & 
                    (filtered_df['temperature'] <= temp_range[1])
                ]
            if 'salinity' in df.columns and sal_range is not None:
                filtered_df = filtered_df[
                    (filtered_df['salinity'] >= sal_range[0]) & 
                    (filtered_df['salinity'] <= sal_range[1])
                ]
            
            st.write(f"📊 Filtered Results: {len(filtered_df)} of {len(df)} records")
            if not filtered_df.empty:
                st.dataframe(filtered_df, use_container_width=True)

def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">🌊 Ocean Data Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("### 📊 Dashboard Info")
        st.markdown("""
        This dashboard visualizes oceanographic data collected from the **ERDDAP** database.
        
        **Features:**
        - 🗺️ Interactive geographic maps
        - 📈 Time series analysis
        - 📊 Data quality metrics
        - 🔍 Real-time queries
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Refresh button
        if st.button("🔄 Refresh Data", help="Reload data from database"):
            st.cache_data.clear()
            st.rerun()
        
        # Data source info
        st.markdown("---")
        st.markdown("### 📡 Data Source")
        st.markdown("""
        **Dataset:** SeaDataNet North Atlantic Climatology  
        **Source:** Ifremer ERDDAP Server  
        **Variables:** Temperature, Salinity, Location, Time  
        **Update:** Real-time from pipeline
        """)
    
    # Load data
    with st.spinner("🔄 Loading ocean data..."):
        df, row_count, table_info = load_data_from_db()
    
    if df.empty:
        st.warning("⚠️ No data found in database. Please run the pipeline to collect data.")
        st.code("python run_pipeline.py")
        return
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Geographic View", "📈 Time Series", "📊 Data Quality", "🔍 Data Explorer"])
    
    with tab1:
        st.header("🗺️ Geographic Distribution of Ocean Data")
        
        # Create and display map
        ocean_map = create_map_visualization(df)
        if ocean_map:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                map_data = st_folium(ocean_map, width=700, height=500)
            
            with col2:
                st.markdown("### 🌡️ Temperature Scale")
                st.markdown("""
                <div style="background: white; padding: 15px; border-radius: 8px; border: 2px solid #ddd;">
                    <div style="margin-bottom: 8px;">
                        <span style="color: blue; font-size: 16px;">●</span> 
                        <span style="font-size: 14px;">Cold (&lt;15°C)</span>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <span style="color: green; font-size: 16px;">●</span> 
                        <span style="font-size: 14px;">Moderate (15-25°C)</span>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <span style="color: red; font-size: 16px;">●</span> 
                        <span style="font-size: 14px;">Warm (&gt;25°C)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Display data summary below map
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📍 Location Summary")
            if 'latitude' in df.columns and 'longitude' in df.columns:
                st.write(f"**Latitude Range:** {df['latitude'].min():.2f}° to {df['latitude'].max():.2f}°")
                st.write(f"**Longitude Range:** {df['longitude'].min():.2f}° to {df['longitude'].max():.2f}°")
                st.write(f"**Data Points:** {len(df)} locations")
        
        with col2:
            st.subheader("🌡️ Temperature Summary")
            if 'temperature' in df.columns:
                st.write(f"**Average:** {df['temperature'].mean():.2f}°C")
                st.write(f"**Range:** {df['temperature'].min():.2f}°C to {df['temperature'].max():.2f}°C")
                st.write(f"**Standard Deviation:** {df['temperature'].std():.2f}°C")
    
    with tab2:
        st.header("📈 Time Series Analysis")
        
        # Create and display time series plots
        time_series_fig = create_time_series_plots(df)
        if time_series_fig:
            st.plotly_chart(time_series_fig, use_container_width=True)
        
        # Statistical summary
        create_statistical_summary(df)
    
    with tab3:
        st.header("📊 Data Quality & Metrics")
        
        # Data quality metrics
        create_data_quality_metrics(df, row_count, table_info)
        
        # Data completeness analysis
        st.subheader("🔍 Data Completeness Analysis")
        if not df.empty:
            # Calculate completeness by column
            completeness_by_col = (df.notna().sum() / len(df) * 100).round(1)
            
            fig_completeness = px.bar(
                x=completeness_by_col.index,
                y=completeness_by_col.values,
                title="📊 Data Completeness by Column",
                labels={'x': 'Column', 'y': 'Completeness (%)'},
                color=completeness_by_col.values,
                color_continuous_scale='RdYlGn'
            )
            fig_completeness.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_completeness, use_container_width=True)
        
        # Database schema info
        st.subheader("🗄️ Database Schema")
        schema_df = pd.DataFrame(table_info, columns=['Column', 'Type', 'Null', 'Key', 'Default', 'Extra'])
        st.dataframe(schema_df, use_container_width=True)
    
    with tab4:
        st.header("🔍 Interactive Data Explorer")
        create_real_time_query_interface(df)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**🌊 Ocean Data Pipeline Dashboard** | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Data points: {len(df)}"
    )

if __name__ == "__main__":
    main()