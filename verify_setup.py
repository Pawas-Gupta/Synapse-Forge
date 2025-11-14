"""
Verify RFP Agent System Setup
Run this to check all dependencies and configuration
"""
import sys
import os

print("=" * 80)
print("RFP Agent System - Setup Verification")
print("=" * 80)

# Step 1: Check Python Version
print("\nStep 1: Python Version")
print(f"   Version: {sys.version}")
if sys.version_info >= (3, 8):
    print("   [OK] Python version OK")
else:
    print("   [ERROR] Python 3.8+ required")
    sys.exit(1)

# Step 2: Check Required Packages
print("\nStep 2: Required Packages")
required_packages = {
    'streamlit': None,
    'flask': None,
    'flask_cors': None,
    'langchain': None,
    'langchain_core': None,
    'langchain_openai': None,
    'openai': None,
    'pandas': None,
    'bs4': 'beautifulsoup4',
    'requests': None,
    'dotenv': 'python-dotenv'
}

missing_packages = []
for package, pip_name in required_packages.items():
    try:
        module = __import__(package.split('.')[0])
        version = getattr(module, '__version__', 'unknown')
        print(f"   [OK] {package}: {version}")
    except ImportError:
        install_name = pip_name or package
        print(f"   [ERROR] {package}: NOT INSTALLED")
        missing_packages.append(install_name)

if missing_packages:
    print(f"\n   [WARNING] Missing packages: {', '.join(missing_packages)}")
    print(f"   Run: pip install {' '.join(missing_packages)}")
    sys.exit(1)

# Step 3: Check LangChain Imports
print("\nStep 3: LangChain Components")
try:
    from langchain_core.prompts import ChatPromptTemplate
    print("   [OK] ChatPromptTemplate")
except ImportError as e:
    print(f"   [ERROR] ChatPromptTemplate import failed: {e}")
    print("   Run: pip install langchain-core")
    sys.exit(1)

try:
    from langchain_openai import ChatOpenAI
    print("   [OK] ChatOpenAI (Groq)")
except ImportError as e:
    print(f"   [ERROR] ChatOpenAI import failed: {e}")
    print("   Run: pip install langchain-openai")
    sys.exit(1)

# Step 4: Check Environment Variables
print("\nStep 4: Environment Configuration")
try:
    from dotenv import load_dotenv
    load_dotenv()
    
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key:
        print(f"   [OK] GROQ_API_KEY: Set ({groq_key[:10]}...)")
    else:
        print("   [WARNING] GROQ_API_KEY: Not set (required for agents)")
        print("   Create .env file with: GROQ_API_KEY=your_key_here")
    
    flask_host = os.getenv('FLASK_HOST', '127.0.0.1')
    flask_port = os.getenv('FLASK_PORT', '5000')
    print(f"   [OK] Flask will run on: http://{flask_host}:{flask_port}")
    
except Exception as e:
    print(f"   [ERROR] Environment check failed: {e}")

# Step 5: Check Project Structure
print("\nStep 5: Project Structure")
required_files = [
    'backend/__init__.py',
    'backend/app.py',
    'backend/database.py',
    'backend/agents/__init__.py',
    'backend/agents/main_agent.py',
    'backend/agents/sales_agent.py',
    'backend/agents/technical_agent.py',
    'backend/agents/pricing_agent.py',
    'backend/tools/__init__.py',
    'backend/tools/rfp_tools.py',
    'backend/tools/sku_tools.py',
    'frontend/streamlit_app.py'
]

missing_files = []
for file_path in required_files:
    if os.path.exists(file_path):
        print(f"   [OK] {file_path}")
    else:
        print(f"   [ERROR] {file_path}: MISSING")
        missing_files.append(file_path)

if missing_files:
    print(f"\n   [WARNING] Missing {len(missing_files)} required files")

# Step 6: Test Database Initialization
print("\nStep 6: Database Test")
try:
    sys.path.insert(0, os.getcwd())
    from backend.database import get_db
    
    db = get_db()
    cursor = db.get_connection().cursor()
    cursor.execute("SELECT COUNT(*) FROM SKU_CATALOG")
    count = cursor.fetchone()[0]
    print(f"   [OK] Database initialized: {count} products loaded")
except Exception as e:
    print(f"   [ERROR] Database test failed: {e}")

# Summary
print("\n" + "=" * 80)
print("Verification Summary")
print("=" * 80)

if not missing_packages and not missing_files:
    print("\n[SUCCESS] All checks passed! System is ready to run.")
    print("\nNext steps:")
    print("   1. Start backend:  python backend/app.py")
    print("   2. Start frontend: streamlit run frontend/streamlit_app.py")
else:
    print("\n[WARNING] Some issues found. Please fix them before running the system.")
    if missing_packages:
        print(f"\n   Install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
    if missing_files:
        print(f"\n   Create missing files: {len(missing_files)} files need to be created")

print("=" * 80)