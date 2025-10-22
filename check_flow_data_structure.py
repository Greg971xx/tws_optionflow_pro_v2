"""
Check what data is actually loaded in flow_tab
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.options_flow_analyzer import load_comprehensive_data
from src.config import OPTIONFLOW_DB

# Load data like flow_tab does
df = load_comprehensive_data(
    db_path=OPTIONFLOW_DB,
    selected_expiry="ALL",
    sample_size=50000,
    min_volume_filter=0,
    confidence_threshold=0.6
)

print("=" * 70)
print("FLOW DATA STRUCTURE")
print("=" * 70)

if df.empty:
    print("\n❌ DataFrame is EMPTY!")
else:
    print(f"\n✅ Loaded {len(df)} rows")

    print("\n📋 Columns:")
    for col in df.columns:
        print(f"   • {col}")

    print("\n📊 Sample data (first 3 rows):")
    print(df.head(3))

    print("\n🔍 Key columns check:")
    required_cols = ['strike', 'right', 'qty', 'is_buy', 'is_sell', 'is_call', 'is_put']
    for col in required_cols:
        if col in df.columns:
            print(f"   ✓ {col}: {df[col].notna().sum()} non-null values")
        else:
            print(f"   ✗ {col}: MISSING")

    # Check for aggregation columns used in heatmap
    print("\n🎨 Heatmap columns check:")
    heatmap_cols = ['market_session', 'money_class', 'volume_category']
    for col in heatmap_cols:
        if col in df.columns:
            print(f"   ✓ {col}: {df[col].unique()[:5]}")
        else:
            print(f"   ✗ {col}: MISSING")