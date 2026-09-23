#!/bin/bash
set -euo pipefail

# ==============================================================================
# build_app.sh - Build standalone BiometricGuard.app bundle for macOS
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$PROJECT_DIR/venv/bin/python"
SCRIPT_PATH="$PROJECT_DIR/app_guard.py"
APP_NAME="BiometricGuard.app"
LOCAL_APP_PATH="$PROJECT_DIR/$APP_NAME"
APPLICATIONS_PATH="/Applications/$APP_NAME"

echo "============================================================"
echo "         Building macOS Application: $APP_NAME"
echo "============================================================"
echo "Project Directory : $PROJECT_DIR"
echo "Python Executable : $PYTHON_BIN"
echo "Script Path       : $SCRIPT_PATH"
echo "============================================================"

# 1. Validation
if [ ! -f "$PYTHON_BIN" ]; then
    echo "[ERROR] Virtualenv python not found at $PYTHON_BIN" >&2
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "[ERROR] Target script not found at $SCRIPT_PATH" >&2
    exit 1
fi

# Clean up previous build in project directory
rm -rf "$LOCAL_APP_PATH"

# 2. Compile native AppleScript applet with osacompile
echo "[1/4] Compiling applet using osacompile..."
# The AppleScript executes the python guard script detached in background
APPLE_SCRIPT="do shell script \"'$PYTHON_BIN' '$SCRIPT_PATH' > /dev/null 2>&1 &\""
osacompile -o "$LOCAL_APP_PATH" -e "$APPLE_SCRIPT"

# 3. Inject Info.plist properties
echo "[2/4] Updating Info.plist configuration..."
PLIST="$LOCAL_APP_PATH/Contents/Info.plist"

# Add/Set NSCameraUsageDescription for native Camera access permission
/usr/libexec/PlistBuddy -c "Set :NSCameraUsageDescription 'Biometric facial recognition to protect applications'" "$PLIST" 2>/dev/null || \
/usr/libexec/PlistBuddy -c "Add :NSCameraUsageDescription string 'Biometric facial recognition to protect applications'" "$PLIST"

# Add/Set LSUIElement = true to run headlessly as an agent without a Dock icon
/usr/libexec/PlistBuddy -c "Set :LSUIElement bool true" "$PLIST" 2>/dev/null || \
/usr/libexec/PlistBuddy -c "Add :LSUIElement bool true" "$PLIST"

# Set bundle identifier and display name
/usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier 'com.user.biometricguard'" "$PLIST" 2>/dev/null || \
/usr/libexec/PlistBuddy -c "Add :CFBundleIdentifier string 'com.user.biometricguard'" "$PLIST"

# 4. Ad-hoc codesign the bundle so macOS recognizes valid signature
echo "[3/4] Signing application bundle..."
codesign --force --deep --sign - "$LOCAL_APP_PATH"

# 5. Optionally install to /Applications if writable
echo "[4/4] Deploying application..."
if [ -w "/Applications" ]; then
    rm -rf "$APPLICATIONS_PATH"
    cp -R "$LOCAL_APP_PATH" "/Applications/"
    codesign --force --deep --sign - "$APPLICATIONS_PATH" 2>/dev/null || true
    echo "[SUCCESS] Installed to $APPLICATIONS_PATH"
else
    echo "[INFO] /Applications is not writable without sudo; keeping app at $LOCAL_APP_PATH"
fi

echo ""
echo "============================================================"
echo "               BUILD COMPLETED SUCCESSFULLY!"
echo "============================================================"
echo "App Bundle created at:"
echo "  $LOCAL_APP_PATH"
if [ -d "$APPLICATIONS_PATH" ]; then
    echo "  $APPLICATIONS_PATH"
fi
echo "============================================================"
