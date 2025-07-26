#!/usr/bin/env python3
"""
Test encoding fix for Windows emoji display issue
"""

import sys
import io

# Apply encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Test various emojis that appear in the codebase
print("Testing emoji display with UTF-8 encoding:")
print("📊 Chart emoji")
print("🚀 Rocket emoji") 
print("✅ Checkmark emoji")
print("❌ Cross emoji")
print("⚠️ Warning emoji")
print("🎯 Target emoji")
print("📈 Upward trend")
print("📉 Downward trend")
print("💡 Light bulb")
print("🔧 Wrench")

# Test that regular text still works
print("\nRegular text: Testing UTF-8 encoding fix")
print("Numbers: 123.456")
print("Special chars: @#$%^&*()")

print("\nIf you can see all emojis above without errors, the encoding fix is working!")