#!/usr/bin/env python3
"""
Proper Data Migration Script
This script migrates data from the original databases to the consolidated database
"""

import psycopg2
import sys
from datetime import datetime

def migrate_data():
    """Migrate data from original databases to consolidated database"""
    print("🔄 Starting proper data migration...")
    
    # Connect to original mlflow database
    source_conn = psycopg2.connect(
        host="localhost",
        port="5433",
        database="mlflow",
        user="mlflow",
        password="password"
    )
    source_conn.autocommit = True
    source_cur = source_conn.cursor()
    
    # Connect to consolidated database
    target_conn = psycopg2.connect(
        host="localhost",
        port="5433",
        database="ml_security_consolidated",
        user="mlflow",
        password="password"
    )
    target_conn.autocommit = True
    target_cur = target_conn.cursor()
    
    try:
        # Set search path to mlflow schema
        target_cur.execute("SET search_path TO mlflow, public;")
        
        # Get data from source
        print("📊 Getting data from original mlflow database...")
        
        # Migrate experiments
        print("   📋 Migrating experiments...")
        source_cur.execute("SELECT * FROM experiments;")
        experiments = source_cur.fetchall()
        source_cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'experiments' ORDER BY ordinal_position;")
        exp_columns = [row[0] for row in source_cur.fetchall()]
        
        if experiments:
            placeholders = ', '.join(['%s'] * len(exp_columns))
            insert_sql = f"INSERT INTO experiments ({', '.join(exp_columns)}) VALUES ({placeholders}) ON CONFLICT (experiment_id) DO NOTHING;"
            target_cur.executemany(insert_sql, experiments)
            print(f"   ✅ Migrated {len(experiments)} experiments")
        
        # Migrate runs
        print("   📋 Migrating runs...")
        source_cur.execute("SELECT * FROM runs;")
        runs = source_cur.fetchall()
        source_cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'runs' ORDER BY ordinal_position;")
        run_columns = [row[0] for row in source_cur.fetchall()]
        
        if runs:
            placeholders = ', '.join(['%s'] * len(run_columns))
            insert_sql = f"INSERT INTO runs ({', '.join(run_columns)}) VALUES ({placeholders}) ON CONFLICT (run_uuid) DO NOTHING;"
            target_cur.executemany(insert_sql, runs)
            print(f"   ✅ Migrated {len(runs)} runs")
        
        # Migrate model_versions
        print("   📋 Migrating model_versions...")
        source_cur.execute("SELECT * FROM model_versions;")
        model_versions = source_cur.fetchall()
        source_cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'model_versions' ORDER BY ordinal_position;")
        mv_columns = [row[0] for row in source_cur.fetchall()]
        
        if model_versions:
            placeholders = ', '.join(['%s'] * len(mv_columns))
            insert_sql = f"INSERT INTO model_versions ({', '.join(mv_columns)}) VALUES ({placeholders}) ON CONFLICT (name, version) DO NOTHING;"
            target_cur.executemany(insert_sql, model_versions)
            print(f"   ✅ Migrated {len(model_versions)} model versions")
        
        # Migrate metrics
        print("   📋 Migrating metrics...")
        source_cur.execute("SELECT * FROM metrics;")
        metrics = source_cur.fetchall()
        source_cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'metrics' ORDER BY ordinal_position;")
        met_columns = [row[0] for row in source_cur.fetchall()]
        
        if metrics:
            placeholders = ', '.join(['%s'] * len(met_columns))
            insert_sql = f"INSERT INTO metrics ({', '.join(met_columns)}) VALUES ({placeholders}) ON CONFLICT (key, timestamp, step, run_uuid) DO NOTHING;"
            target_cur.executemany(insert_sql, metrics)
            print(f"   ✅ Migrated {len(metrics)} metrics")
        
        # Migrate params
        print("   📋 Migrating params...")
        source_cur.execute("SELECT * FROM params;")
        params = source_cur.fetchall()
        source_cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'params' ORDER BY ordinal_position;")
        par_columns = [row[0] for row in source_cur.fetchall()]
        
        if params:
            placeholders = ', '.join(['%s'] * len(par_columns))
            insert_sql = f"INSERT INTO params ({', '.join(par_columns)}) VALUES ({placeholders}) ON CONFLICT (key, run_uuid) DO NOTHING;"
            target_cur.executemany(insert_sql, params)
            print(f"   ✅ Migrated {len(params)} params")
        
        # Migrate tags
        print("   📋 Migrating tags...")
        source_cur.execute("SELECT * FROM tags;")
        tags = source_cur.fetchall()
        source_cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'tags' ORDER BY ordinal_position;")
        tag_columns = [row[0] for row in source_cur.fetchall()]
        
        if tags:
            placeholders = ', '.join(['%s'] * len(tag_columns))
            insert_sql = f"INSERT INTO tags ({', '.join(tag_columns)}) VALUES ({placeholders}) ON CONFLICT (key, run_uuid) DO NOTHING;"
            target_cur.executemany(insert_sql, tags)
            print(f"   ✅ Migrated {len(tags)} tags")
        
        # Verify migration
        print("\n🔍 Verifying migration...")
        target_cur.execute("SELECT COUNT(*) FROM mlflow.experiments;")
        exp_count = target_cur.fetchone()[0]
        target_cur.execute("SELECT COUNT(*) FROM mlflow.runs;")
        run_count = target_cur.fetchone()[0]
        target_cur.execute("SELECT COUNT(*) FROM mlflow.model_versions;")
        mv_count = target_cur.fetchone()[0]
        
        print(f"✅ Consolidated database now has:")
        print(f"   - {exp_count} experiments")
        print(f"   - {run_count} runs")
        print(f"   - {mv_count} model versions")
        
        source_cur.close()
        source_conn.close()
        target_cur.close()
        target_conn.close()
        
        print("\n🎉 Data migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        source_cur.close()
        source_conn.close()
        target_cur.close()
        target_conn.close()
        return False

if __name__ == "__main__":
    success = migrate_data()
    sys.exit(0 if success else 1)
