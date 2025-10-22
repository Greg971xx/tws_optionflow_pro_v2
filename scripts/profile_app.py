"""
Detailed Application Profiling
Uses cProfile to find bottlenecks
"""
import cProfile
import pstats
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def profile_flow_refresh():
    """Profile the flow tab refresh operation"""
    from PyQt6.QtWidgets import QApplication
    from src.ui.tabs.flow_tab import FlowTab

    app = QApplication(sys.argv)
    flow_tab = FlowTab()

    # Profile the refresh
    profiler = cProfile.Profile()
    profiler.enable()

    flow_tab.refresh_data()

    profiler.disable()

    # Print results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions

    print("=" * 80)
    print("FLOW TAB REFRESH - TOP 30 SLOWEST FUNCTIONS")
    print("=" * 80)
    print(s.getvalue())

    # Save to file
    output_file = Path("profile_flow_refresh.txt")
    with open(output_file, 'w') as f:
        ps = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
        ps.print_stats()

    print(f"\n💾 Full profile saved to: {output_file}")

    app.quit()


def profile_gex_analysis():
    """Profile GEX calculation"""
    from PyQt6.QtWidgets import QApplication
    from src.ui.tabs.gex_tab import GEXTab

    app = QApplication(sys.argv)
    gex_tab = GEXTab()

    profiler = cProfile.Profile()
    profiler.enable()

    gex_tab.refresh_data()

    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)

    print("=" * 80)
    print("GEX ANALYSIS - TOP 30 SLOWEST FUNCTIONS")
    print("=" * 80)
    print(s.getvalue())

    output_file = Path("profile_gex_analysis.txt")
    with open(output_file, 'w') as f:
        ps = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
        ps.print_stats()

    print(f"\n💾 Full profile saved to: {output_file}")

    app.quit()


if __name__ == "__main__":
    print("\n1. Profiling Flow Tab Refresh...")
    profile_flow_refresh()

    print("\n\n2. Profiling GEX Analysis...")
    profile_gex_analysis()