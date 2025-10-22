"""
UI Performance Benchmark
Measure rendering and interaction times
"""
import time
import sys
from pathlib import Path
from functools import wraps

sys.path.insert(0, str(Path(__file__).parent.parent))


def timer(func):
    """Decorator to measure function execution time"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"  ⏱️  {func.__name__}: {duration:.3f}s")
        return result

    return wrapper


def benchmark_ui():
    """Benchmark UI components"""
    from PyQt6.QtWidgets import QApplication

    print("=" * 80)
    print("UI RENDERING BENCHMARK")
    print("=" * 80)

    app = QApplication(sys.argv)

    # Benchmark each tab
    print("\n📋 Testing Dashboard Tab...")
    from src.ui.tabs.dashboard_tab import DashboardTab
    start = time.time()
    dashboard = DashboardTab()
    print(f"  ⏱️  Dashboard load: {time.time() - start:.3f}s")

    print("\n📊 Testing Flow Tab...")
    from src.ui.tabs.flow_tab import FlowTab
    start = time.time()
    flow = FlowTab()
    print(f"  ⏱️  Flow tab init: {time.time() - start:.3f}s")

    # Test refresh (this is usually slow)
    print("\n🔄 Testing Flow refresh...")
    start = time.time()
    flow.refresh_data()
    print(f"  ⏱️  Flow refresh: {time.time() - start:.3f}s")

    print("\n📈 Testing GEX Tab...")
    from src.ui.tabs.gex_tab import GEXTab
    start = time.time()
    gex = GEXTab()
    print(f"  ⏱️  GEX tab init: {time.time() - start:.3f}s")

    print("\n🔄 Testing GEX refresh...")
    start = time.time()
    gex.refresh_data()
    print(f"  ⏱️  GEX refresh: {time.time() - start:.3f}s")

    print("\n📊 Testing Volatility Tab...")
    from src.ui.tabs.volatility_tab import VolatilityTab
    start = time.time()
    vol = VolatilityTab()
    print(f"  ⏱️  Volatility tab init: {time.time() - start:.3f}s")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nIf any operation takes >2s, it needs optimization.")

    app.quit()


if __name__ == "__main__":
    benchmark_ui()