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
from datetime import datetime, timedelta, date
import json
import logging
import sys

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our new dynamic API components
try:
    from config import get_coverage_bounds, get_time_bounds, coordinate_converter
    from cache_manager import cache_manager
    from pipeline.extract import ERDDAPExtractor
    DYNAMIC_API_AVAILABLE = True
except ImportError as e:
    DYNAMIC_API_AVAILABLE = False
    print(f"Dynamic API components not available: {e}")

# Page configuration
st.set_page_config(
    page_title="Ocean Data Explorer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
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
    
    .selection-box {
        background: #f0f8ff;
        border: 2px solid #4a90e2;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .coordinate-display {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 14px;
    }
    
    .sidebar-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_coordinates' not in st.session_state:
    st.session_state.selected_coordinates = None
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'data_metadata' not in st.session_state:
    st.session_state.data_metadata = None
if 'last_query' not in st.session_state:
    st.session_state.last_query = None

@st.cache_data(ttl=300)
def load_data_from_db():
    """Load data from DuckDB database with caching (backwards compatibility)."""
    db_path = Path("data/ocean_data.duckdb")
    
    if not db_path.exists():
        return pd.DataFrame(), 0, []
    
    try:
        conn = duckdb.connect(str(db_path))
        df = conn.execute("SELECT * FROM sea_surface ORDER BY time").df()
        row_count = conn.execute("SELECT COUNT(*) FROM sea_surface").fetchone()[0]
        table_info = conn.execute("DESCRIBE sea_surface").fetchall()
        conn.close()
        
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        
        return df, row_count, table_info
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame(), 0, []

def create_interactive_map(coverage_bounds=None):
    """Create an interactive map for location selection."""
    
    # Get dataset coverage bounds
    if coverage_bounds is None and DYNAMIC_API_AVAILABLE:
        try:
            coverage_bounds = get_coverage_bounds()
        except:
            coverage_bounds = None
    
    # Default bounds if dynamic API not available
    if coverage_bounds is None:
        coverage_bounds = {
            'north': 70.0, 'south': 20.0,
            'east': 40.0, 'west': -80.0
        }
    
    # Calculate map center
    center_lat = (coverage_bounds['north'] + coverage_bounds['south']) / 2
    center_lon = (coverage_bounds['east'] + coverage_bounds['west']) / 2
    
    # Create folium map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=3,
        tiles='OpenStreetMap'
    )
    
    # Add dataset coverage rectangle (without popup to avoid click interference)
    folium.Rectangle(
        bounds=[
            [coverage_bounds['south'], coverage_bounds['west']],
            [coverage_bounds['north'], coverage_bounds['east']]
        ],
        color='blue',
        fill=True,
        fillColor='lightblue',
        fillOpacity=0.1,
        weight=1
    ).add_to(m)
    
    # Add current selection marker if exists
    if st.session_state.selected_coordinates:
        lat, lon = st.session_state.selected_coordinates
        folium.Marker(
            location=[lat, lon],
            popup=f"Selected: {lat:.3f}°N, {lon:.3f}°W",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
    
    return m

def process_map_click(map_data):
    """Process map click events and extract coordinates."""
    if not map_data or 'last_clicked' not in map_data:
        return None
    
    # Check if there was a click and it's not on a popup
    if map_data['last_clicked'] and not map_data.get('last_object_clicked_popup'):
        lat = map_data['last_clicked']['lat']
        lon = map_data['last_clicked']['lng']
        
        # Validate coordinates are within dataset bounds
        if DYNAMIC_API_AVAILABLE:
            try:
                is_valid, message = coordinate_converter.validate_coordinates(lat, lon)
                if not is_valid:
                    st.error(f"❌ {message}")
                    return None
            except:
                pass
        
        return (lat, lon)
    
    return None

def fetch_ocean_data(latitude, longitude, start_date, end_date):
    """Fetch ocean data for selected coordinates and date range."""
    
    if not DYNAMIC_API_AVAILABLE:
        st.error("❌ Dynamic API not available. Please ensure Phase 1 components are properly integrated.")
        return None, None
    
    try:
        # Create extractor instance
        extractor = ERDDAPExtractor()
        
        # Convert dates to strings
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Fetch data
        df, metadata = extractor.fetch_data_for_location(
            latitude=latitude,
            longitude=longitude,
            start_date=start_str,
            end_date=end_str
        )
        
        return df, metadata
        
    except Exception as e:
        st.error(f"❌ Failed to fetch data: {str(e)}")
        return None, None

def create_enhanced_time_series_plots(df, metadata=None):
    """Create enhanced time series plots with more detailed information."""
    if df.empty or 'time' not in df.columns:
        st.warning("⚠️ No time series data available")
        return None
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'🌡️ Sea Surface Temperature Over Time ({len(df)} data points)',
            f'🧂 Sea Surface Salinity Over Time ({len(df)} data points)'
        ),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Temperature plot
    if 'temperature' in df.columns and df['temperature'].notna().any():
        temp_data = df.dropna(subset=['temperature'])
        
        fig.add_trace(
            go.Scatter(
                x=temp_data['time'],
                y=temp_data['temperature'],
                mode='lines+markers',
                name='Temperature',
                line=dict(color='#ff6b6b', width=3),
                marker=dict(size=8, color='#ff6b6b'),
                hovertemplate='<b>Temperature</b><br>Date: %{x}<br>Value: %{y:.2f}°C<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Salinity plot
    if 'salinity' in df.columns and df['salinity'].notna().any():
        sal_data = df.dropna(subset=['salinity'])
        
        fig.add_trace(
            go.Scatter(
                x=sal_data['time'],
                y=sal_data['salinity'],
                mode='lines+markers',
                name='Salinity',
                line=dict(color='#4ecdc4', width=3),
                marker=dict(size=8, color='#4ecdc4'),
                hovertemplate='<b>Salinity</b><br>Date: %{x}<br>Value: %{y:.2f} PSU<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=700,
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

def create_data_summary_metrics(df, metadata=None):
    """Create enhanced data summary metrics."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="📊 Data Points",
            value=f"{len(df):,}",
            help="Number of time series data points retrieved"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if metadata and 'data_source' in metadata:
            data_source = metadata['data_source']
            source_icon = "💾" if data_source == "cache" else "🌐"
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric(
                label=f"{source_icon} Data Source",
                value=data_source.title(),
                help="Whether data came from cache or API"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric(label="📡 Data Source", value="Database")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        if not df.empty and 'time' in df.columns and len(df) > 1:
            time_span = (df['time'].max() - df['time'].min()).days
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
        if metadata and 'quality_score' in metadata:
            quality_score = metadata['quality_score']
            quality_color = "🟢" if quality_score > 0.8 else "🟡" if quality_score > 0.5 else "🔴"
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric(
                label=f"{quality_color} Quality Score",
                value=f"{quality_score:.1%}",
                help="Data quality based on completeness and value ranges"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            completeness = (df.notna().sum().sum() / (len(df) * len(df.columns)) * 100) if not df.empty else 0
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric(
                label="✅ Completeness",
                value=f"{completeness:.1f}%",
                help="Percentage of non-null values"
            )
            st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">Ocean Data Explorer - Global SST & Salinity</h1>', unsafe_allow_html=True)
    
    # Show dynamic API status
    if not DYNAMIC_API_AVAILABLE:
        st.warning("⚠️ Dynamic API features not available. Running in legacy mode.")
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("### 🎯 How to Use")
        if DYNAMIC_API_AVAILABLE:
            st.markdown("""
            **Step-by-step guide:**
            1. 🗺️ Click anywhere on the map to select location
            2. 📅 Choose your date range (1955-1960)
            3. 🌊 Click "Fetch Ocean Data"
            4. 📈 View results in analysis tabs
            """)
        else:
            st.markdown("""
            **Legacy Mode:**
            - 📊 View static pipeline data
            - 🔍 Basic data exploration
            - 📈 Time series visualization
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data source info
        st.markdown("---")
        st.markdown("### 📡 Data Source")
        st.markdown("""
        **Dataset:** SeaDataNet North Atlantic Climatology  
        **Coverage:** Limited region around 32.5°N, -70°W  
        **Period:** 1955-1960 (6 years)  
        **Resolution:** 0.25° grid, monthly  
        **Variables:** Temperature, Salinity
        """)
        
        # Refresh button
        if st.button("🔄 Refresh Dashboard"):
            st.cache_data.clear()
            st.rerun()
    
    # Main content tabs
    if DYNAMIC_API_AVAILABLE:
        tab1, tab2, tab3, tab4 = st.tabs([
            "🗺️ Location & Data", 
            "📈 Time Series Analysis", 
            "📊 Data Summary", 
            "🔍 Legacy Data Explorer"
        ])
        
        with tab1:
            # Create columns for map and controls
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader("🗺️ Select Ocean Location")
                # Create and display interactive map
                ocean_map = create_interactive_map()
                
                map_data = st_folium(
                    ocean_map, 
                    width=700, 
                    height=400,
                    returned_objects=['last_clicked', 'last_object_clicked_popup']
                )
                
                # Process map clicks
                clicked_coords = process_map_click(map_data)
                if clicked_coords:
                    st.session_state.selected_coordinates = clicked_coords
                    st.rerun()
            
            with col2:
                st.markdown("### 🎯 Selection Controls")
                
                # Display current selection
                if st.session_state.selected_coordinates:
                    lat, lon = st.session_state.selected_coordinates
                    st.markdown(f'<div class="coordinate-display">', unsafe_allow_html=True)
                    st.markdown(f"**Selected Location:**<br>")
                    st.markdown(f"Latitude: {lat:.3f}°N<br>")
                    st.markdown(f"Longitude: {lon:.3f}°W")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("👆 Click on the map to select a location")
                
                # Manual coordinate entry in expander
                with st.expander("📍 Manual Coordinates", expanded=False):
                    manual_lat = st.number_input(
                        "Latitude (°N)", 
                        min_value=20.0, 
                        max_value=80.0, 
                        value=40.0, 
                        step=0.1,
                        help="Latitude in decimal degrees (20°N - 80°N)"
                    )
                    manual_lon = st.number_input(
                        "Longitude (°W)", 
                        min_value=-80.0, 
                        max_value=40.0, 
                        value=-30.0, 
                        step=0.1,
                        help="Longitude in decimal degrees (80°W - 40°E)"
                    )
                    
                    if st.button("📍 Use Manual Coordinates", use_container_width=True):
                        st.session_state.selected_coordinates = (manual_lat, manual_lon)
                        st.rerun()
                
                # Date range selection
                st.markdown("### 📅 Select Date Range")
                
                # Get available date range
                min_date = date(1955, 1, 1)
                max_date = date(1960, 12, 31)
                
                start_date = st.date_input(
                    "Start Date",
                    value=date(1955, 1, 1),
                    min_value=min_date,
                    max_value=max_date,
                    help="Select start date (1955-01-01 to 1960-12-31)"
                )
                
                end_date = st.date_input(
                    "End Date",
                    value=date(1960, 12, 31),
                    min_value=min_date,
                    max_value=max_date,
                    help="Select end date (1955-01-01 to 1960-12-31)"
                )
                
                # Validate date range
                if start_date > end_date:
                    st.error("❌ Start date must be before end date")
                    date_range_valid = False
                else:
                    date_range_valid = True
                    # Check if date range is reasonable
                    days_diff = (end_date - start_date).days
                    if days_diff > 2190:  # ~6 years (full dataset)
                        st.warning(f"⚠️ Large date range ({days_diff} days)")
                
                # Query summary
                if st.session_state.selected_coordinates and date_range_valid:
                    lat, lon = st.session_state.selected_coordinates
                    st.markdown('<div class="selection-box">', unsafe_allow_html=True)
                    st.markdown(f"**📍 Location:** {lat:.3f}°N, {lon:.3f}°W")
                    st.markdown(f"**📅 Date Range:** {start_date} to {end_date}")
                    st.markdown(f"**📊 Duration:** {(end_date - start_date).days} days")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Fetch data button
                fetch_disabled = not (st.session_state.selected_coordinates and date_range_valid)
                
                if st.button("🌊 Fetch Ocean Data", type="primary", use_container_width=True, disabled=fetch_disabled):
                    if not st.session_state.selected_coordinates:
                        st.warning("⚠️ Please select a location on the map first")
                    elif not date_range_valid:
                        st.warning("⚠️ Please fix the date range")
                    else:
                        # Check if this is the same query as last time
                        lat, lon = st.session_state.selected_coordinates
                        current_query = (lat, lon, start_date, end_date)
                        if st.session_state.last_query == current_query and st.session_state.current_data is not None:
                            st.success("✅ Using cached data from previous query")
                        else:
                            # Fetch new data
                            with st.spinner("🌊 Fetching ocean data from ERDDAP..."):
                                df, metadata = fetch_ocean_data(lat, lon, start_date, end_date)
                            
                            if df is not None:
                                # Store in session state
                                st.session_state.current_data = df
                                st.session_state.data_metadata = metadata
                                st.session_state.last_query = current_query
                                
                                # Show success message
                                data_source = metadata.get('data_source', 'unknown') if metadata else 'unknown'
                                cache_icon = "💾" if data_source == "cache" else "🌐"
                                st.success(f"✅ Successfully fetched {len(df)} data points from {data_source} {cache_icon}")
                                
                                # Show data quality info
                                if metadata and 'quality_score' in metadata:
                                    quality_score = metadata['quality_score']
                                    if quality_score < 0.5:
                                        st.warning(f"⚠️ Data quality is low ({quality_score:.1%}). Some values may be missing or unreliable.")
                                
                                # Trigger rerun to update visualizations
                                st.rerun()
                
                # Clear selection button
                if st.button("🗑️ Clear Selection", use_container_width=True):
                    st.session_state.selected_coordinates = None
                    st.session_state.current_data = None
                    st.session_state.data_metadata = None
                    st.rerun()
            
            # Show data preview if available
            if st.session_state.current_data is not None and not st.session_state.current_data.empty:
                st.markdown("---")
                st.subheader("📊 Data Preview")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Data Points", f"{len(st.session_state.current_data):,}")
                with col2:
                    data_source = st.session_state.data_metadata.get('data_source', 'unknown') if st.session_state.data_metadata else 'unknown'
                    source_icon = "💾" if data_source == "cache" else "🌐"
                    st.metric(f"{source_icon} Source", data_source.title())
                with col3:
                    if st.session_state.data_metadata and 'quality_score' in st.session_state.data_metadata:
                        quality_score = st.session_state.data_metadata['quality_score']
                        quality_color = "🟢" if quality_score > 0.8 else "🟡" if quality_score > 0.5 else "🔴"
                        st.metric(f"{quality_color} Quality", f"{quality_score:.1%}")
                
                st.info("📈 Go to 'Time Series Analysis' tab to view detailed charts and data")
        
        with tab2:
            st.header("📈 Time Series Analysis")
            
            # Check if we have data
            if st.session_state.current_data is not None and not st.session_state.current_data.empty:
                # Create enhanced time series plots
                fig = create_enhanced_time_series_plots(
                    st.session_state.current_data, 
                    st.session_state.data_metadata
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show data table
                st.subheader("📋 Data Table")
                st.dataframe(st.session_state.current_data, use_container_width=True)
                
                # Data export
                if st.button("📥 Export Data as CSV"):
                    csv_data = st.session_state.current_data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"ocean_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("📍 Select a location and fetch data to see time series analysis")
        
        with tab3:
            st.header("📊 Data Summary & Quality")
            
            # Cache Status (moved from sidebar)
            if DYNAMIC_API_AVAILABLE:
                st.subheader("💾 Cache Status")
                try:
                    cache_stats = cache_manager.get_cache_stats()
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Active Queries", cache_stats['active_entries'])
                    with col2:
                        st.metric("Cache Size", f"{cache_stats['total_size_mb']:.1f} MB")
                    with col3:
                        st.metric("TTL Hours", cache_stats['ttl_hours'])
                    with col4:
                        cache_enabled = "✅ On" if cache_stats['cache_enabled'] else "❌ Off"
                        st.metric("Cache Status", cache_enabled)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🧹 Clear Cache"):
                            cache_manager.clear_cache()
                            st.success("Cache cleared!")
                            st.rerun()
                    with col2:
                        if st.button("🔄 Refresh Cache Stats"):
                            st.rerun()
                            
                except:
                    st.info("Cache system unavailable")
                
                st.markdown("---")
            
            # Current Data Summary (moved from landing page)
            if st.session_state.current_data is not None and not st.session_state.current_data.empty:
                st.subheader("📊 Current Data Summary")
                
                # Summary metrics
                create_data_summary_metrics(
                    st.session_state.current_data, 
                    st.session_state.data_metadata
                )
                
                # Statistical summary
                st.subheader("📈 Statistical Summary")
                numeric_cols = st.session_state.current_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    summary_stats = st.session_state.current_data[numeric_cols].describe()
                    st.dataframe(summary_stats.round(3), use_container_width=True)
                
                # Quality information
                if st.session_state.data_metadata and 'quality_issues' in st.session_state.data_metadata:
                    st.subheader("🔍 Data Quality Assessment")
                    issues = st.session_state.data_metadata['quality_issues']
                    if issues:
                        for issue in issues:
                            st.warning(f"⚠️ {issue}")
                    else:
                        st.success("✅ No data quality issues detected")
                
            else:
                st.info("📍 Select a location and fetch data to see summary statistics")
        
        with tab4:
            st.header("🔍 Legacy Data Explorer")
            st.info("This tab shows data from the original pipeline for comparison")
            
            # Load legacy data
            df_legacy, row_count, table_info = load_data_from_db()
            
            if not df_legacy.empty:
                st.subheader("📊 Legacy Data Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Records", row_count)
                with col2:
                    st.metric("Columns", len(df_legacy.columns))
                with col3:
                    if 'time' in df_legacy.columns and len(df_legacy) > 1:
                        time_span = (df_legacy['time'].max() - df_legacy['time'].min()).days
                        st.metric("Time Span", f"{time_span} days")
                
                st.dataframe(df_legacy, use_container_width=True)
            else:
                st.warning("⚠️ No legacy data found. Run the pipeline first: `python run_pipeline.py`")
    
    else:
        # Legacy mode - simplified interface
        st.header("🔍 Ocean Data Explorer (Legacy Mode)")
        
        # Load legacy data
        df_legacy, row_count, table_info = load_data_from_db()
        
        if not df_legacy.empty:
            # Create basic time series plots
            fig = create_enhanced_time_series_plots(df_legacy)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Show data summary
            create_data_summary_metrics(df_legacy)
            
            # Show data table
            st.subheader("📋 Data Table")
            st.dataframe(df_legacy, use_container_width=True)
        else:
            st.warning("⚠️ No data found. Run the pipeline first: `python run_pipeline.py`")
    
    # Footer
    st.markdown("---")
    
    if DYNAMIC_API_AVAILABLE and st.session_state.current_data is not None:
        data_info = f"Dynamic data: {len(st.session_state.current_data)} points"
        if st.session_state.data_metadata:
            source = st.session_state.data_metadata.get('data_source', 'unknown')
            data_info += f" from {source}"
    else:
        data_info = "Legacy mode active" if not DYNAMIC_API_AVAILABLE else "No dynamic data loaded"
    
    st.markdown(
        f"**Ocean Data Explorer** | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"{data_info}"
    )

if __name__ == "__main__":
    main()