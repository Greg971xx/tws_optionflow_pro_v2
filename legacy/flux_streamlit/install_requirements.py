#!/usr/bin/env python3
# =============================================
# File: install_requirements.py
# Description: Automated setup script for SPX Options Flow Analysis tools
# Usage: Run directly in PyCharm or from terminal: python install_requirements.py
# =============================================

import subprocess
import sys
import os
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def print_step(step, description):
    """Print formatted step"""
    print(f"\n[Step {step}] {description}")
    print("-" * 40)


def run_command(command, description=""):
    """Run a command and handle errors"""
    try:
        print(f"Running: {command}")
        result = subprocess.run(command, shell=True, check=True,
                                capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False


def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required!")
        return False

    print("✅ Python version compatible")
    return True


def create_requirements_file():
    """Create requirements.txt file"""
    requirements = """
# Core Data Science Stack
pandas>=1.5.0
numpy>=1.21.0
sqlite3  # Built-in module

# Interactive Brokers API
ib-insync>=0.9.86

# Web Dashboard
streamlit>=1.28.0
plotly>=5.15.0

# Configuration Management
PyYAML>=6.0

# Enhanced Features (Optional but recommended)
scipy>=1.9.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Development & Testing (Optional)
pytest>=7.0.0
black>=22.0.0
flake8>=5.0.0

# Jupyter for analysis (Optional)
jupyter>=1.0.0
ipykernel>=6.0.0
""".strip()

    try:
        with open("requirements.txt", "w") as f:
            f.write(requirements)
        print("✅ requirements.txt created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create requirements.txt: {e}")
        return False


def create_config_template():
    """Create configuration template file"""
    config_template = """
# Configuration for SPX Options Flow Collector
# Copy this file to config.yaml and adjust settings

# Database settings
db_path: "db/multi_expiry_flow.db"

# Interactive Brokers connection
host: "127.0.0.1"
port: 7497  # TWS Paper: 7497, TWS Live: 7496, Gateway Paper: 4002, Gateway Live: 4001
client_id: 14
connection_timeout: 30.0
retry_attempts: 3

# Symbol and exchange
symbol: "SPX"
exchange: "CBOE"
option_exchange: "CBOE"

# Multi-expiry analysis settings
max_dte: 5  # Analyze 1DTE to 5DTE
n_strikes_per_expiry: 20  # Number of strikes per expiry
strike_spacing: 5  # Strike spacing (e.g., 5 points)
atm_range_pct: 0.05  # Focus on ±5% around spot

# Performance settings
batch_size: 150
batch_timeout: 3.0
log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR
health_check_interval: 45.0

# Market analysis settings
min_spread_pct: 0.001  # 0.1% minimum spread
aggressive_threshold: 0.8  # Confidence threshold for aggressive detection
""".strip()

    try:
        os.makedirs("config", exist_ok=True)
        with open("config/config_template.yaml", "w") as f:
            f.write(config_template)
        print("✅ Configuration template created at config/config_template.yaml")
        return True
    except Exception as e:
        print(f"❌ Failed to create config template: {e}")
        return False


def create_directory_structure():
    """Create necessary directory structure"""
    directories = [
        "db",
        "logs",
        "config",
        "data",
        "notebooks",
        "scripts"
    ]

    try:
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        return True
    except Exception as e:
        print(f"❌ Failed to create directories: {e}")
        return False


def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """
# Database files
*.db
*.db-wal
*.db-shm

# Log files
*.log
logs/

# Configuration files with sensitive data
config.yaml
*.env

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data files
data/*.csv
data/*.xlsx
*.pkl

# Jupyter Notebook
.ipynb_checkpoints

# OS
.DS_Store
Thumbs.db
""".strip()

    try:
        with open(".gitignore", "w") as f:
            f.write(gitignore_content)
        print("✅ .gitignore created")
        return True
    except Exception as e:
        print(f"❌ Failed to create .gitignore: {e}")
        return False


def install_packages():
    """Install required packages"""
    packages = [
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "ib-insync>=0.9.86",
        "streamlit>=1.28.0",
        "plotly>=5.15.0",
        "PyYAML>=6.0",
        "scipy>=1.9.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0"
    ]

    print("Installing core packages...")
    for package in packages:
        print(f"\nInstalling {package}...")
        if not run_command(f"pip install {package}"):
            print(f"⚠️  Failed to install {package}, but continuing...")

    return True


def create_startup_scripts():
    """Create convenient startup scripts"""

    # Collector startup script
    collector_script = '''#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run collector
if __name__ == "__main__":
    try:
        from collector_multi_expiry import run_multi_expiry_collector, load_config
        config = load_config("config/config.yaml")
        run_multi_expiry_collector(config)
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure collector_multi_expiry.py is in the same directory")
    except Exception as e:
        print(f"Error: {e}")
'''

    # Streamlit startup script
    streamlit_script = '''#!/usr/bin/env python3
import subprocess
import sys
import os

def start_streamlit():
    """Start Streamlit dashboard"""
    try:
        # Check if viewer exists
        viewer_file = "hybrid_options_flow_viewer.py"
        if not os.path.exists(viewer_file):
            print(f"❌ {viewer_file} not found!")
            print("Make sure the viewer file is in the current directory")
            return

        print("🚀 Starting Streamlit dashboard...")
        print("Dashboard will open in your browser automatically")
        print("Press Ctrl+C to stop")

        cmd = f"streamlit run {viewer_file} -- --db db/multi_expiry_flow.db"
        subprocess.run(cmd, shell=True)

    except KeyboardInterrupt:
        print("\\n✅ Streamlit dashboard stopped")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")

if __name__ == "__main__":
    start_streamlit()
'''

    try:
        # Create scripts directory
        os.makedirs("scripts", exist_ok=True)

        with open("scripts/start_collector.py", "w") as f:
            f.write(collector_script)

        with open("scripts/start_dashboard.py", "w") as f:
            f.write(streamlit_script)

        print("✅ Startup scripts created in scripts/ directory")
        return True
    except Exception as e:
        print(f"❌ Failed to create startup scripts: {e}")
        return False


def create_readme():
    """Create README with usage instructions"""
    readme_content = """# SPX Options Flow Analysis Tools

Advanced multi-expiry options flow analysis system for SPX (S&P 500 Index) options.

## 🚀 Quick Start

### 1. Setup
```bash
# Run the installation script
python install_requirements.py

# Copy and configure settings
cp config/config_template.yaml config/config.yaml
# Edit config/config.yaml with your IB settings
```

### 2. Start Interactive Brokers
- Open TWS (Trader Workstation) or IB Gateway
- Enable API connections (Configure → API → Settings)
- Set Socket Port: 7497 (Paper) or 7496 (Live)
- Add 127.0.0.1 to Trusted IPs

### 3. Run Data Collection
```bash
# Start the multi-expiry collector
python collector_multi_expiry.py --config config/config.yaml

# Or use the startup script
python scripts/start_collector.py
```

### 4. View Analytics Dashboard
```bash
# Start Streamlit dashboard
streamlit run hybrid_options_flow_viewer.py -- --db db/multi_expiry_flow.db

# Or use the startup script  
python scripts/start_dashboard.py
```

## 📊 Features

### Multi-Expiry Analysis (1DTE to 5DTE)
- Captures options flow across multiple expiration dates
- Focuses on near-term expiries for gamma exposure analysis
- Real-time market maker positioning estimates

### Advanced Flow Classification
- Enhanced Lee-Ready rule implementation
- Aggression scoring for trade intensity
- Quote-test based buy/sell classification
- Confidence weighting for ambiguous trades

### Market Microstructure Analytics
- Bid-ask spread analysis
- Price improvement tracking
- Market session segmentation (pre/regular/post)
- Greeks integration (Delta, Gamma, Theta, Vega)

### Dashboard Features
- Real-time flow visualization
- Gamma exposure heatmaps
- Strike-by-strike positioning analysis
- Historical flow patterns
- Market maker position estimates

## 📁 Project Structure
```
├── collector_multi_expiry.py          # Main data collector
├── hybrid_options_flow_viewer.py # Enhanced dashboard
├── install_requirements.py            # This setup script
├── requirements.txt                   # Python dependencies
├── config/
│   ├── config_template.yaml          # Configuration template
│   └── config.yaml                   # Your settings (create from template)
├── db/                               # SQLite databases
├── logs/                            # Application logs
├── scripts/                         # Utility scripts
└── data/                           # Analysis outputs
```

## ⚙️ Configuration

Key settings in `config/config.yaml`:

```yaml
# IB Connection
host: "127.0.0.1"
port: 7497  # Paper trading
client_id: 14

# Analysis Parameters
max_dte: 5  # Analyze 1-5 days to expiry
n_strikes_per_expiry: 20  # Strikes per expiry
atm_range_pct: 0.05  # ±5% around spot

# Performance
batch_size: 150
log_level: "INFO"
```

## 🔍 Analysis Focus

### Gamma Exposure Analysis
- Identifies high-gamma strikes where market makers are most exposed
- Tracks changes in positioning throughout the day
- Highlights potential support/resistance levels

### Aggressive Flow Detection  
- Distinguishes between passive and aggressive order flow
- Identifies when large players are forcing trades
- Measures market impact and liquidity consumption

### Market Maker Positioning
- Estimates net MM positions by strike and expiry
- Tracks cumulative flow imbalances
- Predicts likely hedging activity

## 📈 Use Cases

1. **Intraday Trading**: Identify high-probability support/resistance from gamma walls
2. **Risk Management**: Monitor aggregate positioning and flow imbalances  
3. **Market Structure**: Analyze liquidity patterns and market maker behavior
4. **Strategy Development**: Build strategies around predictable MM hedging flows

## 🛠️ Troubleshooting

### Common Issues
- **IB Connection Failed**: Check TWS/Gateway is running and API is enabled
- **No Data**: Verify market hours and symbol availability
- **Performance**: Reduce number of strikes or increase batch size

### Logs
Check `logs/` directory for detailed error messages and performance metrics.

## 📊 Database Schema

The system uses SQLite with optimized schemas for:
- **trades**: Individual option transactions with full classification
- **mm_positioning**: Aggregated market maker position estimates
- **health_metrics**: System performance monitoring

---

*This system is for educational and research purposes. Always verify data accuracy for trading decisions.*
"""

    try:
        with open("README.md", "w") as f:
            f.write(readme_content)
        print("✅ README.md created with comprehensive documentation")
        return True
    except Exception as e:
        print(f"❌ Failed to create README: {e}")
        return False


def main():
    """Main installation function"""
    print_header("SPX Options Flow Analysis - Setup Script")
    print("This script will install all requirements and set up the project structure.")

    # Check Python version
    print_step(1, "Checking Python Version")
    if not check_python_version():
        print("Please upgrade to Python 3.8 or higher and run again.")
        return False

    # Create directory structure
    print_step(2, "Creating Project Structure")
    if not create_directory_structure():
        print("Failed to create directories. Check permissions and try again.")
        return False

    # Create requirements file
    print_step(3, "Creating Requirements File")
    if not create_requirements_file():
        print("Failed to create requirements.txt")
        return False

    # Install packages
    print_step(4, "Installing Python Packages")
    print("This may take a few minutes...")
    if not install_packages():
        print("Some packages may have failed to install. Check the output above.")

    # Create configuration template
    print_step(5, "Creating Configuration Template")
    if not create_config_template():
        print("Failed to create configuration template")
        return False

    # Create startup scripts
    print_step(6, "Creating Startup Scripts")
    if not create_startup_scripts():
        print("Failed to create startup scripts")
        return False

    # Create .gitignore
    print_step(7, "Creating .gitignore")
    create_gitignore()

    # Create README
    print_step(8, "Creating Documentation")
    create_readme()

    # Final instructions
    print_header("Setup Complete! 🎉")
    print("""
Next steps:
1. Copy config/config_template.yaml to config/config.yaml
2. Edit config/config.yaml with your Interactive Brokers settings
3. Start TWS/Gateway and enable API connections
4. Run: python collector_multi_expiry.py --config config/config.yaml
5. In another terminal: python scripts/start_dashboard.py

For detailed instructions, see README.md
    """)

    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Installation completed successfully!")
        else:
            print("\n❌ Installation completed with errors. Check the output above.")
    except KeyboardInterrupt:
        print("\n⚠️  Installation interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check the error details above and try again.")