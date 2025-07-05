"""
Cache Management System for Ocean Data Pipeline

This module handles caching of ERDDAP API responses to improve performance
and reduce API calls. Uses SQLite for persistence and includes cache
cleanup and validation.
"""

import sqlite3
import json
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

from config import CACHE_CONFIG

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages caching of ocean data queries and responses."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "query_cache.db"
        self.init_database()
    
    def init_database(self):
        """Initialize the cache database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT UNIQUE NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    variables TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    fetched_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    file_size_bytes INTEGER NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_hash ON query_cache(query_hash)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at ON query_cache(expires_at)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_coordinates ON query_cache(latitude, longitude)
            """)
    
    def _generate_query_hash(self, 
                           latitude: float, 
                           longitude: float, 
                           start_date: str, 
                           end_date: str, 
                           variables: List[str]) -> str:
        """Generate a unique hash for a query."""
        # Create a string representation of the query
        query_str = f"{latitude:.6f}_{longitude:.6f}_{start_date}_{end_date}_{'_'.join(sorted(variables))}"
        
        # Generate SHA256 hash
        return hashlib.sha256(query_str.encode()).hexdigest()[:16]  # First 16 characters
    
    def get_cached_data(self, 
                       latitude: float, 
                       longitude: float, 
                       start_date: str, 
                       end_date: str, 
                       variables: List[str]) -> Optional[pd.DataFrame]:
        """Retrieve cached data if available and not expired."""
        
        if not CACHE_CONFIG["enabled"]:
            return None
        
        query_hash = self._generate_query_hash(latitude, longitude, start_date, end_date, variables)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT data_json, expires_at, fetched_at, row_count 
                    FROM query_cache 
                    WHERE query_hash = ? AND expires_at > ?
                """, (query_hash, datetime.now()))
                
                result = cursor.fetchone()
                
                if result:
                    data_json, expires_at, fetched_at, row_count = result
                    
                    # Log cache hit
                    logger.info(f"📋 Cache hit for query {query_hash[:8]}... ({row_count} rows, fetched: {fetched_at})")
                    
                    # Parse JSON back to DataFrame
                    data_dict = json.loads(data_json)
                    df = pd.DataFrame(data_dict)
                    
                    # Convert time column back to datetime
                    if 'time' in df.columns:
                        df['time'] = pd.to_datetime(df['time'])
                    
                    return df
                
                else:
                    logger.debug(f"📋 Cache miss for query {query_hash[:8]}...")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Error retrieving cached data: {str(e)}")
            return None
    
    def cache_data(self, 
                  latitude: float, 
                  longitude: float, 
                  start_date: str, 
                  end_date: str, 
                  variables: List[str], 
                  data: pd.DataFrame) -> bool:
        """Cache query results."""
        
        if not CACHE_CONFIG["enabled"]:
            return False
        
        query_hash = self._generate_query_hash(latitude, longitude, start_date, end_date, variables)
        
        try:
            # Convert DataFrame to JSON
            data_json = data.to_json(orient='records', date_format='iso')
            
            # Calculate cache expiry
            expires_at = datetime.now() + timedelta(hours=CACHE_CONFIG["ttl_hours"])
            
            # Calculate data size
            data_size_bytes = len(data_json.encode('utf-8'))
            
            # Check cache size limits
            if data_size_bytes > CACHE_CONFIG["max_cache_size_mb"] * 1024 * 1024:
                logger.warning(f"⚠️ Data too large to cache: {data_size_bytes / (1024*1024):.1f} MB")
                return False
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO query_cache 
                    (query_hash, latitude, longitude, start_date, end_date, variables, 
                     data_json, row_count, fetched_at, expires_at, file_size_bytes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    query_hash,
                    latitude,
                    longitude,
                    start_date,
                    end_date,
                    json.dumps(variables),
                    data_json,
                    len(data),
                    datetime.now(),
                    expires_at,
                    data_size_bytes
                ))
            
            logger.info(f"💾 Cached query {query_hash[:8]}... ({len(data)} rows, {data_size_bytes/1024:.1f} KB)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error caching data: {str(e)}")
            return False
    
    def cleanup_expired_cache(self) -> int:
        """Remove expired cache entries."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM query_cache WHERE expires_at < ?
                """, (datetime.now(),))
                
                deleted_count = cursor.rowcount
                
                if deleted_count > 0:
                    logger.info(f"🧹 Cleaned up {deleted_count} expired cache entries")
                
                return deleted_count
                
        except Exception as e:
            logger.error(f"❌ Error cleaning up cache: {str(e)}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total entries
                total_entries = conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
                
                # Active entries (not expired)
                active_entries = conn.execute("""
                    SELECT COUNT(*) FROM query_cache WHERE expires_at > ?
                """, (datetime.now(),)).fetchone()[0]
                
                # Total size
                total_size = conn.execute("""
                    SELECT SUM(file_size_bytes) FROM query_cache WHERE expires_at > ?
                """, (datetime.now(),)).fetchone()[0] or 0
                
                # Oldest and newest entries
                oldest_entry = conn.execute("""
                    SELECT MIN(fetched_at) FROM query_cache WHERE expires_at > ?
                """, (datetime.now(),)).fetchone()[0]
                
                newest_entry = conn.execute("""
                    SELECT MAX(fetched_at) FROM query_cache WHERE expires_at > ?
                """, (datetime.now(),)).fetchone()[0]
                
                return {
                    "total_entries": total_entries,
                    "active_entries": active_entries,
                    "expired_entries": total_entries - active_entries,
                    "total_size_mb": total_size / (1024 * 1024),
                    "oldest_entry": oldest_entry,
                    "newest_entry": newest_entry,
                    "cache_enabled": CACHE_CONFIG["enabled"],
                    "ttl_hours": CACHE_CONFIG["ttl_hours"],
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting cache stats: {str(e)}")
            return {
                "total_entries": 0,
                "active_entries": 0,
                "expired_entries": 0,
                "total_size_mb": 0.0,
                "oldest_entry": None,
                "newest_entry": None,
                "cache_enabled": CACHE_CONFIG["enabled"],
                "ttl_hours": CACHE_CONFIG["ttl_hours"],
            }
    
    def get_cached_queries(self) -> List[Dict[str, Any]]:
        """Get list of cached queries for debugging/management."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT query_hash, latitude, longitude, start_date, end_date, 
                           variables, row_count, fetched_at, expires_at, file_size_bytes
                    FROM query_cache 
                    WHERE expires_at > ?
                    ORDER BY fetched_at DESC
                """, (datetime.now(),))
                
                queries = []
                for row in cursor.fetchall():
                    query_hash, lat, lon, start_date, end_date, variables, row_count, fetched_at, expires_at, file_size = row
                    
                    queries.append({
                        "hash": query_hash,
                        "latitude": lat,
                        "longitude": lon,
                        "start_date": start_date,
                        "end_date": end_date,
                        "variables": json.loads(variables),
                        "row_count": row_count,
                        "fetched_at": fetched_at,
                        "expires_at": expires_at,
                        "file_size_kb": file_size / 1024,
                    })
                
                return queries
                
        except Exception as e:
            logger.error(f"❌ Error getting cached queries: {str(e)}")
            return []
    
    def clear_cache(self) -> bool:
        """Clear all cache entries."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM query_cache")
                logger.info("🧹 Cache cleared completely")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error clearing cache: {str(e)}")
            return False
    
    def find_nearby_cached_queries(self, 
                                  latitude: float, 
                                  longitude: float, 
                                  radius_degrees: float = 0.5) -> List[Dict[str, Any]]:
        """Find cached queries near a given location."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT query_hash, latitude, longitude, start_date, end_date, 
                           variables, row_count, fetched_at, expires_at
                    FROM query_cache 
                    WHERE expires_at > ?
                    AND ABS(latitude - ?) < ?
                    AND ABS(longitude - ?) < ?
                    ORDER BY 
                        (ABS(latitude - ?) + ABS(longitude - ?)) ASC
                """, (datetime.now(), latitude, radius_degrees, longitude, radius_degrees, latitude, longitude))
                
                nearby_queries = []
                for row in cursor.fetchall():
                    query_hash, lat, lon, start_date, end_date, variables, row_count, fetched_at, expires_at = row
                    
                    # Calculate approximate distance
                    distance = ((lat - latitude) ** 2 + (lon - longitude) ** 2) ** 0.5
                    
                    nearby_queries.append({
                        "hash": query_hash,
                        "latitude": lat,
                        "longitude": lon,
                        "start_date": start_date,
                        "end_date": end_date,
                        "variables": json.loads(variables),
                        "row_count": row_count,
                        "fetched_at": fetched_at,
                        "expires_at": expires_at,
                        "distance_degrees": distance,
                    })
                
                return nearby_queries
                
        except Exception as e:
            logger.error(f"❌ Error finding nearby cached queries: {str(e)}")
            return []

# Global cache manager instance
cache_manager = CacheManager()