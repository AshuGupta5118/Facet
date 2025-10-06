# Facet

A command-line tool to automatically scan a local directory of photos, detect faces, and organize copies of those photos into folders based on the people identified. All processing is done locally, keeping your photos private.

## Key Features

-   **Local & Private:** Scans photos on your local machine without uploading anything to the cloud.
-   **Automatic Grouping:** Uses unsupervised machine learning (DBSCAN) to group photos of the same person without needing to know who they are beforehand.
-   **Simple Organization:** Creates `Person_1`, `Person_2`, etc., folders in an output directory, containing copies of photos for each identified individual.
-   **Recursive Scan:** Scans through all subfolders of your specified photo directory.
-   **Tunable Clustering:** Allows adjusting sensitivity parameters to fine-tune how people are grouped.

## Technology Stack

-   **Programming Language:** Python 3
-   **Face Recognition:** `face_recognition` (dlib backend)
-   **Clustering:** `scikit-learn` (DBSCAN algorithm)
-   **Numerical Computation:** `numpy`
-   **Image Handling:** `Pillow`

## Installation

### 1. Prerequisites

The `dlib` library, a dependency of `face_recognition`, needs to be compiled from source. You **must** have a C++ compiler and CMake installed first.

-   **macOS:** Install Xcode Command Line Tools:
    ```sh
    xcode-select --install
    brew install cmake
    ```
-   **Linux (Debian/Ubuntu):**
    ```sh
    sudo apt-get update
    sudo apt-get install build-essential cmake
    ```
-   **Windows:** Install [Visual Studio with C++ tools](https://visualstudio.microsoft.com/visuals-cpp-build-tools/). During installation, select the "Desktop development with C++" workload. Also, install [CMake](https://cmake.org/download/) and add it to your system's PATH.

### 2. Project Setup

```sh
# Clone this repository (or create the files manually)
git clone https://github.com/your-username/Facet.git
cd Facet

# Create and activate a Python virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the required Python packages
pip install -r requirements.txt