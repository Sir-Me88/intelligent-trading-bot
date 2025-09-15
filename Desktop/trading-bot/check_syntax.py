#!/usr/bin/env python3
"""Syntax validation script for the adaptive intelligent bot."""

import sys
import traceback

def check_syntax():
    """Check syntax of the main bot file."""
    try:
        # Try to compile the file to check for syntax errors
        with open('run_adaptive_intelligent_bot.py', 'r', encoding='utf-8') as f:
            code = f.read()

        compile(code, 'run_adaptive_intelligent_bot.py', 'exec')
        print("✅ SYNTAX OK: No syntax errors found in run_adaptive_intelligent_bot.py")
        return True

    except SyntaxError as e:
        print(f"❌ SYNTAX ERROR in run_adaptive_intelligent_bot.py:")
        print(f"   Line {e.lineno}: {e.msg}")
        if e.text:
            print(f"   Code: {e.text.strip()}")
        print(f"   File: {e.filename}")
        return False

    except FileNotFoundError:
        print("❌ FILE NOT FOUND: run_adaptive_intelligent_bot.py not found")
        return False

    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_syntax()
    if not success:
        sys.exit(1)
    print("\n🎉 Ready to run the bot!")
    print("   Command: python run_adaptive_intelligent_bot.py")
