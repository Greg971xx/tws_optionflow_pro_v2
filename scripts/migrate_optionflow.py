"""
Migrate options flow data - Robust version (no ATTACH)
Handles locked databases by copying data in batches
"""
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
import sys


def migrate_with_copy():
    """Copy data without ATTACH (more robust)"""

    old_db = Path(r"C:\Users\decle\PycharmProjects\flux_claude\db\optionflow.db")
    new_db = Path(r"C:\Users\decle\PycharmProjects\tws_optionflow_pro_v2\db\market_data.db")

    print("🔄 Options Flow Data Migration (Robust Mode)")
    print("=" * 70)

    if not old_db.exists():
        print(f"❌ Source not found: {old_db}")
        return False

    if not new_db.exists():
        print(f"❌ Destination not found: {new_db}")
        return False

    # Sizes
    old_size = old_db.stat().st_size / (1024 * 1024)
    new_size_before = new_db.stat().st_size / (1024 * 1024)

    print(f"\n📂 Source: {old_db}")
    print(f"   Size: {old_size:.2f} MB")
    print(f"\n📂 Destination: {new_db}")
    print(f"   Size: {new_size_before:.2f} MB")

    # Backup
    backup_name = f'market_data_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db'
    backup_path = new_db.parent / backup_name
    print(f"\n💾 Creating backup...")
    shutil.copy(new_db, backup_path)
    print(f"   ✓ {backup_name}")

    # Open both connections
    print("\n🔗 Opening databases...")

    try:
        # Open source in READ ONLY mode
        source_conn = sqlite3.connect(f"file:{old_db}?mode=ro", uri=True)
        source_cursor = source_conn.cursor()
    except Exception as e:
        print(f"❌ Cannot open source database: {e}")
        print("\n💡 Make sure:")
        print("   • No other application is using flux_claude/db/optionflow.db")
        print("   • Close DB Browser, old app instances, etc.")
        return False

    dest_conn = sqlite3.connect(new_db)
    dest_cursor = dest_conn.cursor()

    try:
        # Get tables from source
        source_cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        tables = [row[0] for row in source_cursor.fetchall()]

        print(f"\n📋 Found {len(tables)} table(s) in source:")
        for table in tables:
            source_cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = source_cursor.fetchone()[0]
            print(f"   • {table}: {count:,} rows")

        print(f"\n🚀 Starting migration...")
        print("=" * 70)

        for table in tables:
            if table == 'sqlite_sequence':
                continue

            print(f"\n📊 {table}")
            print("-" * 70)

            # Check if table exists in destination
            dest_cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            exists = dest_cursor.fetchone()

            if not exists:
                # Get CREATE TABLE statement
                source_cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'")
                create_sql = source_cursor.fetchone()[0]

                print(f"   Creating table...")
                dest_cursor.execute(create_sql)
            else:
                dest_cursor.execute(f"SELECT COUNT(*) FROM {table}")
                existing = dest_cursor.fetchone()[0]
                print(f"   Table exists ({existing:,} rows)")

            # Get row count
            source_cursor.execute(f"SELECT COUNT(*) FROM {table}")
            total_rows = source_cursor.fetchone()[0]

            if total_rows == 0:
                print(f"   ⊘ Empty table, skipping")
                continue

            print(f"   Copying {total_rows:,} rows...")

            # Get all data (for large tables, this could be batched)
            source_cursor.execute(f"SELECT * FROM {table}")
            rows = source_cursor.fetchall()

            # Get column names
            source_cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in source_cursor.fetchall()]
            placeholders = ','.join(['?' for _ in columns])

            # Insert data in batches
            batch_size = 10000
            inserted = 0

            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                dest_cursor.executemany(
                    f"INSERT OR REPLACE INTO {table} VALUES ({placeholders})",
                    batch
                )
                inserted += len(batch)

                if total_rows > 100000:  # Show progress for large tables
                    progress = (inserted / total_rows) * 100
                    print(f"   Progress: {inserted:,}/{total_rows:,} ({progress:.1f}%)", end='\r')

            if total_rows > 100000:
                print()  # New line after progress

            # Get indexes
            source_cursor.execute(f"""
                SELECT sql FROM sqlite_master 
                WHERE type='index' AND tbl_name='{table}' AND sql IS NOT NULL
            """)
            indexes = source_cursor.fetchall()

            if indexes:
                print(f"   Creating {len(indexes)} index(es)...")
                for idx_sql, in indexes:
                    try:
                        dest_cursor.execute(idx_sql)
                    except sqlite3.OperationalError as e:
                        if "already exists" not in str(e):
                            print(f"      ⚠️  {e}")

            # Final count
            dest_cursor.execute(f"SELECT COUNT(*) FROM {table}")
            final = dest_cursor.fetchone()[0]
            print(f"   ✅ Final: {final:,} rows")

        # Commit
        print(f"\n💾 Committing changes...")
        dest_conn.commit()

        # Optimize
        print(f"🔧 Optimizing database...")
        dest_cursor.execute("VACUUM")
        dest_cursor.execute("ANALYZE")

        # Final size
        new_size_after = new_db.stat().st_size / (1024 * 1024)

        print("\n" + "=" * 70)
        print("✅ MIGRATION COMPLETE!")
        print("=" * 70)

        print(
            f"\n💾 Size: {new_size_before:.2f} MB → {new_size_after:.2f} MB (+{new_size_after - new_size_before:.2f} MB)")
        print(f"📦 Backup: {backup_path}")

        return True

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()

        dest_conn.rollback()

        print("\n🔄 Restoring backup...")
        dest_conn.close()
        source_conn.close()
        shutil.copy(backup_path, new_db)
        print("   ✓ Restored")

        return False

    finally:
        source_conn.close()
        dest_conn.close()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("     OPTIONS FLOW DATABASE MIGRATION")
    print("     Robust Mode (no ATTACH, handles locks)")
    print("=" * 70)

    print("\n⚠️  BEFORE CONTINUING:")
    print("   1. Close ANY application using flux_claude/db/optionflow.db")
    print("   2. Close DB Browser for SQLite")
    print("   3. Close old app instances")
    print("   4. This will take several minutes for 5M+ rows")

    response = input("\n   Ready to continue? (yes/no): ").strip().lower()

    if response != 'yes':
        print("\n❌ Cancelled.")
        sys.exit(0)

    success = migrate_with_copy()

    if success:
        print("\n🎉 SUCCESS! Next steps:")
        print("   1. Create src/config.py")
        print("   2. Update all tabs to use src.config")
        print("   3. Restart app and test")
    else:
        print("\n⚠️  Migration failed. Database restored from backup.")