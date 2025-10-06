# Facet

A command-line tool for organizing local photos by detected faces using facial recognition and clustering algorithms. Process your photo collections privately on your local machine without uploading to cloud services.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Future Work](#future-work)
- [License](#license)
- [Contributing](#contributing)

## Overview

**Facet** is a Python-based tool that automatically scans directories of photos, detects faces, and organizes photos into folders grouped by unique individuals. All processing happens locally, ensuring your photos remain private and secure.

### Motivation

Digital photo collections often grow uncontrollably, making it difficult to find pictures of specific individuals. While cloud services offer face recognition, privacy concerns arise from uploading personal photos to remote servers. Facet addresses this need by providing local face-based photo organization.

### Project Goal

Develop a command-line tool that:
- Automatically scans user-specified directories for photos
- Detects faces within those photos
- Groups photos containing the same individuals using facial recognition and clustering
- Organizes copies of photos into separate folders (Person_1, Person_2, etc.)
- Runs entirely on the user's local machine

## Features

- **Local Processing**: All face detection and clustering happens on your machine
- **Recursive Scanning**: Automatically scans subdirectories for images
- **Multiple Image Formats**: Supports JPG, JPEG, PNG, GIF, BMP, TIFF
- **Advanced Clustering**: Uses DBSCAN algorithm to group similar faces
- **Configurable Parameters**: Adjust clustering sensitivity and minimum samples
- **Progress Logging**: Real-time feedback on processing status
- **Metadata Preservation**: Copies files while maintaining original timestamps

## Technology Stack

- **Programming Language**: Python 3
- **Face Recognition**: face_recognition library (using dlib backend)
- **Clustering**: scikit-learn library (DBSCAN algorithm)
- **Numerical Computation**: numpy
- **Image Handling**: Pillow, opencv-python
- **Command-Line Interface**: argparse
- **File Operations**: os, shutil (Python standard library)

## Prerequisites

### Required Software

- **Python 3.x**: Verify with `python --version` or `python3 --version`
- **pip**: Python package installer (usually included with Python)
- **Build Tools**: Required for dlib compilation

#### Platform-Specific Build Tools

**Linux (Debian/Ubuntu):**
```bash
sudo apt update && sudo apt install build-essential cmake
```

**macOS:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install CMake via Homebrew
brew install cmake
```

**Windows:**
- Install C++ build tools for Visual Studio
- Select "Desktop development with C++" workload during VS Installer
- Ensure CMake is installed and added to PATH

## Installation

### 1. Clone or Download Project Files

Create a directory named `Facet` and place all project files in it:

```bash
mkdir Facet
cd Facet
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate environment
# Linux/macOS:
source venv/bin/activate

# Windows CMD:
venv\Scripts\activate.bat

# Windows PowerShell:
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: This step may take time, especially compiling dlib. Ensure prerequisites are installed first.

## Usage

### Basic Usage

1. **Activate virtual environment** (if not already active):
   ```bash
   source venv/bin/activate  # Linux/macOS
   ```

2. **Navigate to project directory**:
   ```bash
   cd Facet
   ```

3. **Run the face sorting script**:
   ```bash
   python sort_faces.py --input_folder "/path/to/your/photos" --output_folder "/path/to/sorted_output"
   ```

### Advanced Usage with Parameters

```bash
python sort_faces.py -i "/photos" -o "/sorted" --eps 0.5 --min_samples 3
```

### Command-Line Arguments

- `--input_folder` / `-i`: Path to directory containing photos to scan (required)
- `--output_folder` / `-o`: Path where organized folders will be created (required)
- `--eps`: DBSCAN epsilon parameter - similarity tolerance (default: 0.55, lower = stricter)
- `--min_samples`: Minimum faces needed to form a distinct cluster (default: 2)

### Example Commands

```bash
# Basic usage
python sort_faces.py -i "./photos" -o "./organized_photos"

# With custom clustering parameters
python sort_faces.py -i "/Users/john/Pictures" -o "/Users/john/SortedPhotos" --eps 0.4 --min_samples 3

# Strict clustering (faces must be very similar)
python sort_faces.py -i "./family_photos" -o "./sorted_family" --eps 0.3 --min_samples 5
```

## Project Structure

```
Facet/
├── .gitignore              # Git ignore rules
├── LICENSE                 # MIT License
├── README.md              # This documentation
├── sort_faces.py          # Main Python script
├── requirements.txt       # Python dependencies
├── venv/                  # Virtual environment (local)
└── output_folders/        # Generated during runtime
    ├── Person_1/          # Organized photos by person
    ├── Person_2/
    └── ...
```

## How It Works

### System Architecture

Facet processes images through a pipeline involving:

1. **Initialization**: Parse command-line arguments
2. **File Discovery**: Recursively scan input directory for supported image files
3. **Face Processing**: For each image:
   - Load image into memory
   - Detect face locations using face_recognition library
   - Generate 128-dimension face embeddings (fingerprints)
   - Store embeddings with original image paths
4. **Clustering**: Use DBSCAN algorithm to group similar face embeddings
5. **File Organization**: Create folders (Person_1, Person_2, etc.) and copy photos

### Workflow Diagram

```
User Input → Python Script → Scan Files → For Each Image:
Load Image → Detect Faces → Generate Embeddings → Collect All Embeddings →
Cluster Embeddings → Map Images to Clusters → Create Output Folders →
Copy Image Files → Organized Output Folders
```

### Face Detection Process

- Uses HOG (Histogram of Oriented Gradients) model by default for speed
- Alternative CNN model available for higher accuracy: `face_locations(image, model="cnn")`
- Generates unique 128-dimensional numerical embeddings for each detected face
- DBSCAN clusters similar embeddings without requiring predefined number of people

## Configuration

### Supported Image Formats

```python
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
```

### Default DBSCAN Parameters

- **eps (0.55)**: Maximum distance between samples for clustering
  - Lower values = stricter grouping (faces must be more similar)
  - Recommended range: 0.3-0.7
- **min_samples (2)**: Minimum faces needed to form a distinct person cluster
  - Higher values = only group frequently appearing faces
  - Recommended range: 2-5

### Tuning Guidelines

- **Too many separate folders for same person**: Decrease `eps` value
- **Different people grouped together**: Increase `eps` value
- **Want to exclude rarely appearing faces**: Increase `min_samples`

## Requirements File

Create `requirements.txt` with:

```
face_recognition>=1.3.0
scikit-learn>=1.0.0
numpy>=1.19.0
opencv-python>=4.5.0
Pillow>=8.0.0
```

## Future Work

Potential enhancements for future versions:

- **GUI Interface**: Graphical interface for easier operation and result viewing
- **Named Clusters**: Allow users to assign names to Person_X folders
- **Cluster Merging**: Merge folders representing the same person
- **Known Face Recognition**: Recognize specific individuals from reference photos
- **Metadata Tagging**: Write face information to image EXIF data
- **Performance Optimization**: GPU acceleration and batch processing
- **Incremental Updates**: Process only newly added photos
- **Alternative Algorithms**: Experiment with other clustering methods
- **Configuration Files**: YAML/JSON configuration instead of command-line only

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Troubleshooting

### Common Issues

**dlib installation fails:**
- Ensure build tools (CMake, C++ compiler) are properly installed
- Try installing dlib separately: `pip install dlib`
- On Windows, ensure Visual Studio Build Tools are installed

**No faces detected:**
- Verify image quality and face visibility
- Try different clustering parameters
- Check supported image formats

**Memory issues with large collections:**
- Process photos in smaller batches
- Ensure sufficient RAM available
- Consider using CNN model only for smaller datasets

**Clustering produces too many/few groups:**
- Adjust `eps` parameter (similarity tolerance)
- Modify `min_samples` (minimum cluster size)
- Review image quality and lighting conditions

## Support

For issues, questions, or contributions, please refer to the project documentation or create an issue in the repository.

---

**Facet Version 1.0** - Local Photo Organization by Face Detection
