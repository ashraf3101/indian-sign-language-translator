# Raw ISL Dataset Directory

This directory stores original webcam capture images or downloaded ISL video frames.

## Directory Structure
Organize your raw images by gesture class folder:

```text
data/raw/
  ├── A/
  │    ├── A_0001.jpg
  │    └── A_0002.jpg
  ├── B/
  │    ├── B_0001.jpg
  ├── HELLO/
  │    ├── HELLO_0001.jpg
  └── ...
```

## How to Collect Data
You can capture samples easily using the included webcam tool:

```bash
# Capture 100 samples for gesture 'A'
python src/collect_data.py --label A --samples 100

# Capture 100 samples for two-handed gesture 'HELLO'
python src/collect_data.py --label HELLO --samples 100
```
