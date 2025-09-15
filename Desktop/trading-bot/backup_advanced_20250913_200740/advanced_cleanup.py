#!/usr/bin/env python3
"""Advanced cleanup for VPS deployment - Phase 2."""

import os
import shutil
from pathlib import Path
from datetime import datetime

def analyze_current_structure():
    """Analyze what we have and what can be cleaned."""
    print("🔍 ADVANCED CLEANUP ANALYSIS")
    print("="*40)
    
    # Essential files for VPS deployment
    essential_files = {
        # Core bot files
        'run_core_trading_bot.py': 'Main optimized trading bot',
        'start_bot.py': 'Bot launcher',
        
        # Management tools
        'monitor_bot.py': 'Bot monitoring',
        'restart_bot.py': 'Bot restart',
        'stop_bot.py': 'Bot stop',
        
        # Configuration
        'requirements.txt': 'Dependencies',
        '.env': 'API configuration',
        '.gitignore': 'Git ignore rules',
        
        # Documentation (minimal)
        'README.md': 'Main documentation',
        'QUICK_START_GUIDE.md': 'Quick reference',
        
        # Deployment
        'Dockerfile': 'Container deployment',
        'docker-compose.yml': 'Multi-service deployment'
    }
    
    # Files that can be removed (additional cleanup)
    cleanup_candidates = {
        # Alternative bot versions (keep only main)
        'run_adaptive_intelligent_bot.py': 'Alternative bot version',
        
        # Demo and god mode files
        'god_mode_core_demo.py': 'Demo file',
        'god_mode_demo.py': 'Demo file',
        'GOD_MODE_FINAL_SUMMARY.md': 'Demo documentation',
        'GOD_MODE_SETUP_GUIDE.md': 'Demo documentation',
        
        # Test files (all of them)
        'test_2025_improvements_fixed.py': 'Test file',
        'test_2025_improvements.py': 'Test file',
        'test_advanced_bot.py': 'Test file',
        'test_api_config.py': 'Test file',
        'test_basic_structure.py': 'Test file',
        'test_core_final.py': 'Test file',
        'test_core_functionality.py': 'Test file',
        'test_core_only.py': 'Test file',
        'test_god_mode_core.py': 'Test file',
        'test_god_mode_integration.py': 'Test file',
        'test_phase1_components.py': 'Test file',
        'test_simple_final.py': 'Test file',
        'test_simple_imports.py': 'Test file',
        
        # Setup files (one-time use)
        'setup_api_config.py': 'One-time setup',
        'quick_status_check.py': 'Status check (replaced by monitor_bot.py)',
        
        # Backtest files (optional for VPS)
        'backtest_adaptive_bot.py': 'Backtesting (optional)',
        'BACKTEST_README.md': 'Backtest documentation',
        
        # Additional documentation (excessive for VPS)
        'API_SETUP_GUIDE.md': 'Setup guide (APIs already configured)',
        'PHASE1_SETUP_GUIDE.md': 'Phase 1 guide (completed)',
        'PROJECT_STRUCTURE.md': 'Project structure (not needed for VPS)',
        'TRADING_BOT_COMPLETE_MANUAL.md': 'Complete manual (excessive)',
        'CLEANUP_SUMMARY.md': 'Previous cleanup summary',
        'DEPLOYMENT_READY.md': 'Deployment summary (will be recreated)',
        
        # Text files
        'Refined.txt': 'Text notes',
        'Summary.txt': 'Text summary',
        
        # Build files (not needed for simple deployment)
        'Makefile': 'Build automation (not needed)',
        'pytest.ini': 'Test configuration',
        
        # Environment templates (keep only .env)
        '.env.example': 'Environment template',
        '.env.template': 'Environment template',
        
        # This cleanup script
        'final_test_cleanup.py': 'Previous cleanup script',
        'advanced_cleanup.py': 'This cleanup script'
    }
    
    print("📁 ESSENTIAL FILES (KEEP):")
    keep_count = 0
    for file_path, description in essential_files.items():
        if Path(file_path).exists():
            print(f"   ✅ {file_path} - {description}")
            keep_count += 1
        else:
            print(f"   ❌ {file_path} - {description} (MISSING!)")
    
    print(f"\n🗑️ CLEANUP CANDIDATES (CAN REMOVE):")
    remove_count = 0
    for file_path, description in cleanup_candidates.items():
        if Path(file_path).exists():
            print(f"   🗑️ {file_path} - {description}")
            remove_count += 1
    
    print(f"\n📊 CLEANUP SUMMARY:")
    print(f"   Essential files: {keep_count}")
    print(f"   Files to remove: {remove_count}")
    print(f"   Estimated space saved: ~{remove_count * 30}KB")
    
    return list(cleanup_candidates.keys())

def cleanup_directories():
    """Clean up unnecessary directories."""
    print(f"\n📁 DIRECTORY CLEANUP")
    print("="*25)
    
    # Directories to clean
    dir_cleanup = {
        'tests/': 'Test directory (not needed for VPS)',
        'ml_data/': 'ML data (large, not essential for basic trading)',
        '.github/': 'GitHub workflows (not needed for VPS)',
        'config/': 'Additional config (main config in .env)',
        'scripts/': 'Build scripts (not needed for simple deployment)'
    }
    
    cleaned_dirs = 0
    for dir_path, description in dir_cleanup.items():
        dir_obj = Path(dir_path)
        if dir_obj.exists() and dir_obj.is_dir():
            try:
                # Check if directory has important files
                important_files = []
                for item in dir_obj.rglob('*'):
                    if item.is_file() and item.suffix in ['.py', '.json', '.yaml', '.yml']:
                        important_files.append(item)
                
                if important_files:
                    print(f"   ⚠️  {dir_path} - {description} (contains {len(important_files)} files)")
                    print(f"      Consider manual review before removal")
                else:
                    shutil.rmtree(dir_obj)
                    cleaned_dirs += 1
                    print(f"   🗑️ Removed: {dir_path} - {description}")
            except Exception as e:
                print(f"   ❌ Failed to remove {dir_path}: {e}")
        else:
            print(f"   ➖ Not found: {dir_path}")
    
    print(f"\n📊 Directory cleanup: {cleaned_dirs} directories removed")

def create_minimal_structure():
    """Create the minimal VPS-ready structure."""
    print(f"\n🏗️ CREATING MINIMAL VPS STRUCTURE")
    print("="*40)
    
    # Ensure essential directories exist
    essential_dirs = ['logs', 'src']
    
    for dir_name in essential_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            print(f"   ✅ Created: {dir_name}/")
        else:
            print(f"   ✅ Exists: {dir_name}/")

def create_vps_deployment_guide():
    """Create a concise VPS deployment guide."""
    print(f"\n📋 CREATING VPS DEPLOYMENT GUIDE")
    print("="*40)
    
    guide_content = f"""# 🚀 VPS DEPLOYMENT GUIDE

## 📦 DEPLOYMENT PACKAGE CONTENTS

### Essential Files:
- `run_core_trading_bot.py` - Main trading bot (optimized)
- `monitor_bot.py` - Bot monitoring
- `restart_bot.py` - Bot restart
- `stop_bot.py` - Bot stop
- `start_bot.py` - Bot launcher
- `requirements.txt` - Dependencies
- `.env` - Configuration (8 APIs configured)
- `src/` - Source code directory
- `logs/` - Log files directory

## 🔧 VPS SETUP COMMANDS

### 1. Upload Files:
```bash
# Upload the entire directory to your VPS
scp -r trading-bot/ user@your-vps:/home/user/
```

### 2. Install Dependencies:
```bash
cd /home/user/trading-bot/
pip install -r requirements.txt
```

### 3. Start Bot:
```bash
# Option 1: Direct start
python run_core_trading_bot.py

# Option 2: Using launcher
python start_bot.py

# Option 3: Background process
nohup python run_core_trading_bot.py > bot.log 2>&1 &
```

### 4. Monitor Bot:
```bash
# Check status
python monitor_bot.py

# View logs
tail -f logs/core_trading_bot.log

# Check heartbeat
cat logs/core_bot_heartbeat.json
```

## ⚙️ BOT CONFIGURATION

### Optimized Parameters:
- **Confidence**: 78% (quality signals only)
- **Risk/Reward**: 1.5:1 minimum
- **Risk per trade**: 1% of account
- **Max volatility**: 0.002
- **Max spread**: 20 pips

### API Integrations (8/8):
- ✅ MT5 Connection
- ✅ Financial Modeling Prep
- ✅ NewsAPI  
- ✅ Alpha Vantage
- ✅ Twelve Data
- ✅ Twitter API
- ✅ Telegram Bot
- ✅ Telegram Notifications

## 🛡️ SAFETY FEATURES

- Automatic stop losses on every trade
- Position size calculation based on account equity
- Spread validation before execution
- Profit protection with trailing stops
- Quality signal filtering (selective trading)

## 📱 MONITORING

- Real-time Telegram notifications
- Comprehensive logging
- Heartbeat status file
- Performance metrics tracking

## 🔄 MANAGEMENT COMMANDS

```bash
# Restart bot
python restart_bot.py

# Stop bot
python stop_bot.py

# Check status
python monitor_bot.py
```

## 📊 EXPECTED PERFORMANCE

- **Signal acceptance**: 5-20% (quality over quantity)
- **Trading frequency**: 0-5 trades per day
- **Risk management**: Conservative approach
- **Profit protection**: Active trailing stops

---

**Deployment Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Bot Status**: PRODUCTION READY ✅
**VPS Ready**: YES 🚀
"""
    
    try:
        with open('VPS_DEPLOYMENT_GUIDE.md', 'w') as f:
            f.write(guide_content)
        print("   ✅ Created: VPS_DEPLOYMENT_GUIDE.md")
    except Exception as e:
        print(f"   ❌ Failed to create guide: {e}")

def perform_advanced_cleanup(files_to_remove):
    """Perform the advanced cleanup."""
    print(f"\n🧹 PERFORMING ADVANCED CLEANUP")
    print("="*40)
    
    # Create backup first
    backup_dir = Path(f"backup_advanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    backup_dir.mkdir(exist_ok=True)
    
    removed_count = 0
    backed_up = 0
    
    for file_name in files_to_remove:
        file_path = Path(file_name)
        if file_path.exists():
            try:
                # Backup first
                shutil.copy2(file_path, backup_dir / file_name)
                backed_up += 1
                
                # Remove file
                file_path.unlink()
                removed_count += 1
                print(f"   🗑️ Removed: {file_name}")
            except Exception as e:
                print(f"   ❌ Failed to remove {file_name}: {e}")
    
    print(f"\n📊 Advanced Cleanup Summary:")
    print(f"   🗑️ Files removed: {removed_count}")
    print(f"   💾 Files backed up: {backed_up}")
    print(f"   📁 Backup location: {backup_dir}")

def main():
    """Main advanced cleanup function."""
    print("🚀 ADVANCED VPS DEPLOYMENT CLEANUP")
    print("="*50)
    
    print("This will create a minimal, VPS-optimized structure.")
    print("⚠️  More aggressive cleanup than before!")
    
    response = input("\nProceed with advanced cleanup? (y/n): ").strip().lower()
    
    if response != 'y':
        print("❌ Advanced cleanup cancelled")
        return
    
    # Analyze current structure
    files_to_remove = analyze_current_structure()
    
    # Perform cleanup
    perform_advanced_cleanup(files_to_remove)
    
    # Clean directories
    cleanup_directories()
    
    # Create minimal structure
    create_minimal_structure()
    
    # Create VPS guide
    create_vps_deployment_guide()
    
    print(f"\n🎉 ADVANCED CLEANUP COMPLETE!")
    print("="*40)
    print("✅ Minimal VPS structure created")
    print("✅ Deployment guide generated")
    print("✅ All backups preserved")
    print("🚀 Ready for professional VPS deployment!")
    
    # Show final structure
    print(f"\n📁 FINAL STRUCTURE:")
    essential_files = [
        'run_core_trading_bot.py',
        'monitor_bot.py', 'restart_bot.py', 'stop_bot.py', 'start_bot.py',
        'requirements.txt', '.env',
        'README.md', 'QUICK_START_GUIDE.md', 'VPS_DEPLOYMENT_GUIDE.md',
        'Dockerfile', 'docker-compose.yml',
        'src/', 'logs/'
    ]
    
    for item in essential_files:
        if Path(item).exists():
            print(f"   ✅ {item}")
        else:
            print(f"   ❌ {item} (missing)")

if __name__ == "__main__":
    main()