# macOS Biometric Application Launcher

A biometric application launcher for macOS that combines 128D facial recognition verification with real-time liveness detection (Eye Aspect Ratio blink detection using MediaPipe 468 Face Mesh landmarks).

Upon confirming an identity match and a natural eye blink, the launcher executes:
```bash
open -a "<Target Application>"
```

---

## Project Structure

```
├── .gitignore               # Ignores venv, bytecode, DS_Store, and data/authorized/*
├── data/
│   └── authorized/          # Stores authorized user reference photos (e.g. me.jpg)
│       └── .gitkeep
├── utils/
│   ├── __init__.py
│   └── liveness.py          # MediaPipe 468 Face Mesh EAR & BlinkDetector
├── launcher.py              # Main OpenCV camera loop & macOS launcher CLI
├── tests/
│   ├── test_liveness.py     # Unit tests for EAR calculation & state machine
│   └── test_launcher.py     # Unit tests for CLI args, subprocess & auth flow
└── requirements.txt         # Project dependencies
```

---

## Quick Start

### 1. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 2. Enroll Your Face
To snap your reference photo using your Mac webcam:
```bash
python launcher.py --register
```
*(Positions your face in the box and press SPACE to save to `data/authorized/me.jpg`)*

Alternatively, you can manually place any clear image of your face at:
```
data/authorized/me.jpg
```

### 3. Run the Launcher
Launch Google Chrome (default):
```bash
python launcher.py
```

Launch a different macOS application:
```bash
python launcher.py --app "Visual Studio Code"
python launcher.py --app "Terminal"
python launcher.py --app "Slack"
```

---

## CLI Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--app` | `-a` | `'Google Chrome'` | macOS application name to launch |
| `--authorized-image` | `-i` | `'data/authorized/me.jpg'` | Path to authorized face image |
| `--register` | `-r` | `False` | Web camera photo enrollment mode |
| `--tolerance` | `-t` | `0.55` | Face distance tolerance (lower is stricter) |
| `--ear-threshold` | `-e` | `0.21` | EAR threshold for closed eyes |
| `--camera-id` | `-c` | `0` | Camera device index |
| `--timeout` | | `60.0` | Timeout in seconds (0 = infinite) |
| `--headless` | | `False` | Run in console mode without GUI window |
| `--dry-run` | | `False` | Verify without opening the app |

---

## Running Automated Tests

```bash
python -m unittest discover tests
```
