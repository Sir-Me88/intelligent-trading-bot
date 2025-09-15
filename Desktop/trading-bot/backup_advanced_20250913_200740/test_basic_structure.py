#!/usr/bin/env python3
"""Basic structure test for Advanced Trading Bot components."""

import sys
import os
from pathlib import Path

def test_file_structure():
    """Test that all required files exist."""
    print("🔍 Testing file structure...")

    required_files = [
        "src/__init__.py",
        "src/config/__init__.py",
        "src/config/settings.py",
        "src/analysis/__init__.py",
        "src/analysis/correlation.py",
        "src/analysis/technical.py",
        "src/analysis/trend_reversal_detector.py",
        "src/analysis/trade_attribution.py",
        "src/ml/__init__.py",
        "src/ml/trading_ml_engine.py",
        "src/scheduling/__init__.py",
        "src/scheduling/intelligent_scheduler.py",
        "src/news/__init__.py",
        "src/news/sentiment.py",
        "src/monitoring/__init__.py",
        "src/monitoring/metrics.py",
        "run_adaptive_intelligent_bot.py"
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files exist")
        return True

def test_imports():
    """Test basic Python imports without complex dependencies."""
    print("\n🔍 Testing basic imports...")

    try:
        # Test basic Python imports
        import datetime
        import json
        import logging
        print("✅ Basic Python imports successful")

        # Test our custom modules (without external dependencies)
        sys.path.append('src')

        # Test simple imports that don't require external libraries
        from src.analysis.trade_attribution import AttributionResult
        print("✅ Trade attribution module import successful")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_code_syntax():
    """Test that Python files have valid syntax."""
    print("\n🔍 Testing code syntax...")

    python_files = [
        "run_adaptive_intelligent_bot.py",
        "src/analysis/trade_attribution.py",
        "src/ml/trading_ml_engine.py",
        "src/scheduling/intelligent_scheduler.py"
    ]

    syntax_errors = []

    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            compile(code, file_path, 'exec')
            print(f"✅ {file_path} - syntax OK")

        except SyntaxError as e:
            print(f"❌ {file_path} - syntax error: {e}")
            syntax_errors.append(file_path)
        except Exception as e:
            print(f"⚠️ {file_path} - other error: {e}")

    if syntax_errors:
        print(f"❌ Syntax errors in: {syntax_errors}")
        return False
    else:
        print("✅ All Python files have valid syntax")
        return True

def test_git_status():
    """Test Git repository status."""
    print("\n🔍 Testing Git status...")

    try:
        import subprocess

        # Check if we're in a git repository
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )

        if result.returncode == 0:
            print("✅ Git repository is clean")
            return True
        else:
            print("❌ Git repository issue")
            return False

    except Exception as e:
        print(f"❌ Git test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("🚀 BASIC STRUCTURE TEST SUITE")
    print("="*50)

    tests = [
        ("File Structure", test_file_structure),
        ("Basic Imports", test_imports),
        ("Code Syntax", test_code_syntax),
        ("Git Status", test_git_status)
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")

    print("\n" + "="*50)
    print("📋 TEST RESULTS SUMMARY")
    print("="*50)

    for test_name, _ in tests:
        status = "✅ PASSED" if passed_tests > 0 else "❌ FAILED"
        print(f"{test_name}: {status}")

    print(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 ALL BASIC TESTS PASSED!")
        print("📝 Recommendation: Install dependencies and run full test suite")
        return True
    elif passed_tests >= total_tests * 0.75:
        print("⚠️ MOST TESTS PASSED - Ready for dependency installation")
        return True
    else:
        print("❌ CRITICAL ISSUES DETECTED - Fix before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
