import os
import shutil
import argparse
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
import logging
import face_recognition

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Supported image file extensions
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

# DBSCAN parameters (tune these based on your results)
# eps: Max distance between samples for one to be considered as in the neighborhood of the other.
#      Lower values mean faces need to be more similar to be grouped. Start around 0.5-0.6.
DBSCAN_EPS = 0.55
# min_samples: Number of samples (faces) in a neighborhood for a point to be considered as a core point.
#              Essentially, how many times does a face need to appear to be considered a distinct person?
DBSCAN_MIN_SAMPLES = 2 # Increase if you only want to group people appearing more often


# --- Helper Functions ---

def find_image_files(folder_path):
    """Recursively finds all image files in the given folder."""
    image_files = []
    logging.info(f"Scanning for images in: {folder_path}")
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image_files.append(os.path.join(root, filename))
    logging.info(f"Found {len(image_files)} image files.")
    return image_files

def extract_face_data(image_paths):
    """Extracts face encodings and their corresponding paths from images."""
    face_data = []  # List of dictionaries: {'path': path, 'encoding': encoding_vector}
    total_images = len(image_paths)
    logging.info(f"Starting face detection and encoding for {total_images} images...")

    for i, image_path in enumerate(image_paths):
        logging.info(f"Processing image {i + 1}/{total_images}: {os.path.basename(image_path)}")
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            # Find face locations (using HOG model is faster, CNN is more accurate)
            face_locations = face_recognition.face_locations(image)

            if not face_locations:
                logging.debug(f"No faces found in {os.path.basename(image_path)}")
                continue

            # Get face encodings
            # Providing known locations speeds up encoding
            face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
            logging.debug(f"Found {len(face_encodings)} face(s) in {os.path.basename(image_path)}")
            
            # Store each encoding with its path
            for encoding in face_encodings:
                face_data.append({'path': image_path, 'encoding': encoding})

        except Exception as e:
            logging.warning(f"Could not process image {image_path}: {e}")

    logging.info(f"Finished encoding. Found {len(face_data)} total face instances.")
    return face_data

def cluster_faces(face_data):
    """Clusters face encodings using DBSCAN."""
    if not face_data:
        logging.warning("No face data to cluster.")
        return None, {}  # Return None for labels, empty dict for data

    logging.info("Starting face clustering...")
    encodings = np.array([data['encoding'] for data in face_data])

    # Create and fit the DBSCAN model
    # n_jobs=-1 uses all available CPU cores
    clusterer = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric='euclidean', n_jobs=-1)
    clusterer.fit(encodings)

    labels = clusterer.labels_  # Get cluster labels for each face encoding
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # -1 label is for noise/outliers
    logging.info(f"Clustering complete. Found {num_clusters} distinct clusters (people) excluding noise.")
    
    return labels, face_data

def organize_photos_by_cluster(labels, face_data, output_dir):
    """Copies photos into folders based on cluster labels."""
    if labels is None:
        logging.error("Cannot organize photos, clustering failed or produced no results.")
        return

    logging.info(f"Organizing photos into: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Group image paths by cluster label
    images_by_cluster = defaultdict(set)  # Use set to store unique image paths per cluster
    for label, data in zip(labels, face_data):
        if label != -1:  # Ignore noise points
            images_by_cluster[label].add(data['path'])

    if not images_by_cluster:
        logging.warning("No valid clusters found to organize photos.")
        return

    # Copy files for each cluster
    cluster_count = 0
    for label, image_paths in images_by_cluster.items():
        cluster_count += 1
        person_folder_name = f"Person_{cluster_count}"  # Assign sequential names
        person_output_path = os.path.join(output_dir, person_folder_name)
        os.makedirs(person_output_path, exist_ok=True)
        
        logging.info(f"Copying {len(image_paths)} unique images for {person_folder_name} (Cluster Label {label})...")

        for image_path in image_paths:
            try:
                dest_path = os.path.join(person_output_path, os.path.basename(image_path))
                # Avoid copying if somehow the exact same file is listed twice for the cluster
                if not os.path.exists(dest_path):
                    # copy2 preserves metadata like creation/modification time
                    shutil.copy2(image_path, dest_path)
                else:
                    logging.debug(f"Skipping already copied file: {dest_path}")
            except Exception as e:
                logging.error(f"Failed to copy {os.path.basename(image_path)} to {person_folder_name}: {e}")

    logging.info("Finished organizing photos.")


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort photos locally by detected faces using clustering.")
    parser.add_argument("-i", "--input_folder", required=True, help="Path to the folder containing photos to scan.")
    parser.add_argument("-o", "--output_folder", required=True, help="Path to the folder where sorted photos (Person_1, Person_2, ...) will be copied.")
    
    # Optional arguments for tuning
    parser.add_argument("--eps", type=float, default=DBSCAN_EPS, help=f"DBSCAN epsilon (max distance). Default: {DBSCAN_EPS}")
    parser.add_argument("--min_samples", type=int, default=DBSCAN_MIN_SAMPLES, help=f"DBSCAN min samples per cluster. Default: {DBSCAN_MIN_SAMPLES}")

    args = parser.parse_args()

    # Validate input path
    if not os.path.isdir(args.input_folder):
        logging.error(f"Input folder not found or is not a directory: {args.input_folder}")
        exit(1)

    # Use provided tuning parameters if given
    DBSCAN_EPS = args.eps
    DBSCAN_MIN_SAMPLES = args.min_samples
    logging.info(f"Using DBSCAN settings: eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}")

    # 1. Find images
    image_paths = find_image_files(args.input_folder)
    if not image_paths:
        logging.info("No image files found in the specified folder.")
        exit(0)

    # 2. Extract face data (encodings)
    face_data = extract_face_data(image_paths)
    if not face_data:
        logging.info("No faces detected in any of the images.")
        exit(0)

    # 3. Cluster faces
    cluster_labels, clustered_face_data = cluster_faces(face_data)

    # 4. Organize photos based on clusters
    organize_photos_by_cluster(cluster_labels, clustered_face_data, args.output_folder)

    logging.info("Face sorting process finished.")