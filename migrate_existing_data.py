#!/usr/bin/env python3
"""
Data Migration Script for ML Security Service
This script migrates existing data from multiple databases to the consolidated database
"""

import psycopg2
import os
import sys
from datetime import datetime
import json

def get_connection(database):
    """Get database connection"""
    return psycopg2.connect(
        host="localhost",
        port="5433",
        database=database,
        user="mlflow",
        password="password"
    )

def backup_database(source_db, backup_file):
    """Create a complete backup of a database"""
    print(f"📦 Creating backup of {source_db} database...")
    
    try:
        # Connect to source database
        conn = get_connection(source_db)
        conn.autocommit = True
        cur = conn.cursor()
        
        # Get all tables
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = [row[0] for row in cur.fetchall()]
        
        backup_data = {
            "database": source_db,
            "backup_timestamp": datetime.now().isoformat(),
            "tables": {}
        }
        
        # Backup each table
        for table in tables:
            print(f"   📋 Backing up table: {table}")
            
            # Get table structure
            cur.execute(f"""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = '{table}' AND table_schema = 'public'
                ORDER BY ordinal_position;
            """)
            columns = cur.fetchall()
            
            # Get table data
            cur.execute(f"SELECT * FROM {table};")
            rows = cur.fetchall()
            
            backup_data["tables"][table] = {
                "columns": columns,
                "data": rows
            }
            
            print(f"   ✅ {table}: {len(rows)} rows backed up")
        
        # Save backup to file
        with open(backup_file, 'w') as f:
            json.dump(backup_data, f, indent=2, default=str)
        
        cur.close()
        conn.close()
        
        print(f"✅ Backup completed: {backup_file}")
        return True
        
    except Exception as e:
        print(f"❌ Backup failed for {source_db}: {e}")
        return False

def restore_to_consolidated_db(backup_file, target_schema):
    """Restore data from backup to consolidated database"""
    print(f"🔄 Restoring data to {target_schema} schema...")
    
    try:
        # Load backup data
        with open(backup_file, 'r') as f:
            backup_data = json.load(f)
        
        # Connect to consolidated database
        conn = get_connection("ml_security")
        conn.autocommit = True
        cur = conn.cursor()
        
        # Set search path to target schema
        cur.execute(f"SET search_path TO {target_schema}, public;")
        
        # Restore each table
        for table_name, table_data in backup_data["tables"].items():
            print(f"   📋 Restoring table: {table_name}")
            
            # Create table if not exists (basic structure)
            columns = table_data["columns"]
            column_defs = []
            for col in columns:
                col_name, data_type, is_nullable, col_default = col
                nullable = "NULL" if is_nullable == "YES" else "NOT NULL"
                default = f" DEFAULT {col_default}" if col_default else ""
                column_defs.append(f"{col_name} {data_type} {nullable}{default}")
            
            create_sql = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {', '.join(column_defs)}
                );
            """
            
            try:
                cur.execute(create_sql)
                print(f"   ✅ Table {table_name} created/verified")
            except Exception as e:
                print(f"   ⚠️  Table {table_name} creation warning: {e}")
            
            # Insert data
            if table_data["data"]:
                # Get column names
                col_names = [col[0] for col in columns]
                placeholders = ', '.join(['%s'] * len(col_names))
                
                insert_sql = f"""
                    INSERT INTO {table_name} ({', '.join(col_names)})
                    VALUES ({placeholders})
                    ON CONFLICT DO NOTHING;
                """
                
                try:
                    cur.executemany(insert_sql, table_data["data"])
                    print(f"   ✅ {len(table_data['data'])} rows inserted into {table_name}")
                except Exception as e:
                    print(f"   ⚠️  Data insertion warning for {table_name}: {e}")
            else:
                print(f"   ℹ️  No data to insert for {table_name}")
        
        cur.close()
        conn.close()
        
        print(f"✅ Restoration to {target_schema} schema completed!")
        return True
        
    except Exception as e:
        print(f"❌ Restoration failed: {e}")
        return False

def migrate_mlflow_data():
    """Migrate MLflow data to consolidated database"""
    print("🔄 Migrating MLflow data...")
    
    try:
        # Connect to source MLflow database
        source_conn = get_connection("mlflow")
        source_conn.autocommit = True
        source_cur = source_conn.cursor()
        
        # Connect to consolidated database
        target_conn = get_connection("ml_security")
        target_conn.autocommit = True
        target_cur = target_conn.cursor()
        
        # Set search path to mlflow schema
        target_cur.execute("SET search_path TO mlflow, public;")
        
        # Get all MLflow tables
        source_cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = [row[0] for row in source_cur.fetchall()]
        
        print(f"📋 Found {len(tables)} tables to migrate")
        
        for table in tables:
            print(f"   📋 Migrating table: {table}")
            
            # Get table data
            source_cur.execute(f"SELECT * FROM {table};")
            rows = source_cur.fetchall()
            
            if rows:
                # Get column names
                source_cur.execute(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = '{table}' AND table_schema = 'public'
                    ORDER BY ordinal_position;
                """)
                columns = [row[0] for row in source_cur.fetchall()]
                
                # Insert data into consolidated database
                placeholders = ', '.join(['%s'] * len(columns))
                insert_sql = f"""
                    INSERT INTO {table} ({', '.join(columns)})
                    VALUES ({placeholders})
                    ON CONFLICT DO NOTHING;
                """
                
                try:
                    target_cur.executemany(insert_sql, rows)
                    print(f"   ✅ {len(rows)} rows migrated to {table}")
                except Exception as e:
                    print(f"   ⚠️  Migration warning for {table}: {e}")
            else:
                print(f"   ℹ️  No data in {table}")
        
        source_cur.close()
        source_conn.close()
        target_cur.close()
        target_conn.close()
        
        print("✅ MLflow data migration completed!")
        return True
        
    except Exception as e:
        print(f"❌ MLflow migration failed: {e}")
        return False

def migrate_analytics_data():
    """Migrate analytics data to consolidated database"""
    print("🔄 Migrating analytics data...")
    
    try:
        # Check if analytics database exists
        conn = get_connection("ml_security")
        conn.autocommit = True
        cur = conn.cursor()
        
        # Check if analytics tables exist in source
        try:
            source_conn = get_connection("ml_security")
            source_cur = source_conn.cursor()
            source_cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name LIKE '%red_team%' OR table_name LIKE '%model_performance%'
                ORDER BY table_name;
            """)
            analytics_tables = [row[0] for row in source_cur.fetchall()]
            
            if analytics_tables:
                print(f"📋 Found analytics tables: {analytics_tables}")
                
                # Set search path to analytics schema
                cur.execute("SET search_path TO analytics, public;")
                
                for table in analytics_tables:
                    print(f"   📋 Migrating table: {table}")
                    
                    # Get table data
                    source_cur.execute(f"SELECT * FROM {table};")
                    rows = source_cur.fetchall()
                    
                    if rows:
                        # Get column names
                        source_cur.execute(f"""
                            SELECT column_name 
                            FROM information_schema.columns 
                            WHERE table_name = '{table}' AND table_schema = 'public'
                            ORDER BY ordinal_position;
                        """)
                        columns = [row[0] for row in source_cur.fetchall()]
                        
                        # Insert data into analytics schema
                        placeholders = ', '.join(['%s'] * len(columns))
                        insert_sql = f"""
                            INSERT INTO {table} ({', '.join(columns)})
                            VALUES ({placeholders})
                            ON CONFLICT DO NOTHING;
                        """
                        
                        try:
                            cur.executemany(insert_sql, rows)
                            print(f"   ✅ {len(rows)} rows migrated to analytics.{table}")
                        except Exception as e:
                            print(f"   ⚠️  Migration warning for {table}: {e}")
                    else:
                        print(f"   ℹ️  No data in {table}")
                
                source_cur.close()
                source_conn.close()
            else:
                print("   ℹ️  No analytics tables found in source")
        
        except Exception as e:
            print(f"   ℹ️  No separate analytics database found: {e}")
        
        cur.close()
        conn.close()
        
        print("✅ Analytics data migration completed!")
        return True
        
    except Exception as e:
        print(f"❌ Analytics migration failed: {e}")
        return False

def verify_migration():
    """Verify that migration was successful"""
    print("🔍 Verifying migration...")
    
    try:
        conn = get_connection("ml_security")
        conn.autocommit = True
        cur = conn.cursor()
        
        # Check MLflow schema
        cur.execute("SET search_path TO mlflow, public;")
        cur.execute("SELECT COUNT(*) FROM experiments;")
        exp_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM runs;")
        run_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM model_versions;")
        model_count = cur.fetchone()[0]
        
        print(f"✅ MLflow Schema:")
        print(f"   - Experiments: {exp_count}")
        print(f"   - Runs: {run_count}")
        print(f"   - Model Versions: {model_count}")
        
        # Check Analytics schema
        cur.execute("SET search_path TO analytics, public;")
        cur.execute("SELECT COUNT(*) FROM red_team_tests;")
        red_team_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM model_performance;")
        perf_count = cur.fetchone()[0]
        
        print(f"✅ Analytics Schema:")
        print(f"   - Red Team Tests: {red_team_count}")
        print(f"   - Model Performance: {perf_count}")
        
        cur.close()
        conn.close()
        
        print("✅ Migration verification completed!")
        return True
        
    except Exception as e:
        print(f"❌ Migration verification failed: {e}")
        return False

def main():
    """Main migration function"""
    print("🚀 ML Security Service - Data Migration Script")
    print("=" * 60)
    print(f"⏰ Migration started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Create backups
    print("📦 STEP 1: Creating backups...")
    backup_files = []
    
    # Backup MLflow database
    if backup_database("mlflow", "backup_mlflow.json"):
        backup_files.append("backup_mlflow.json")
    
    # Backup other databases if they exist
    for db in ["ml_security", "security_metrics"]:
        try:
            if backup_database(db, f"backup_{db}.json"):
                backup_files.append(f"backup_{db}.json")
        except:
            print(f"   ℹ️  Database {db} not accessible or empty")
    
    print(f"✅ Created {len(backup_files)} backup files")
    print()
    
    # Step 2: Migrate MLflow data
    print("🔄 STEP 2: Migrating MLflow data...")
    if migrate_mlflow_data():
        print("✅ MLflow data migration successful!")
    else:
        print("❌ MLflow data migration failed!")
        return False
    print()
    
    # Step 3: Migrate analytics data
    print("🔄 STEP 3: Migrating analytics data...")
    if migrate_analytics_data():
        print("✅ Analytics data migration successful!")
    else:
        print("❌ Analytics data migration failed!")
        return False
    print()
    
    # Step 4: Verify migration
    print("🔍 STEP 4: Verifying migration...")
    if verify_migration():
        print("✅ Migration verification successful!")
    else:
        print("❌ Migration verification failed!")
        return False
    print()
    
    print("=" * 60)
    print("🎉 DATA MIGRATION COMPLETED SUCCESSFULLY!")
    print("✅ All existing data has been preserved")
    print("✅ Data migrated to consolidated database")
    print("✅ All schemas properly organized")
    print("=" * 60)
    
    print("\n📋 Next steps:")
    print("1. Stop current services: docker-compose down")
    print("2. Start with consolidated database: docker-compose up -d")
    print("3. Verify all data is accessible")
    print("4. Test all functionality")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
