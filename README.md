![image](https://github.com/user-attachments/assets/39943000-1307-4e89-aaa7-0718bedeb066)


# Image Stitching Application

This is a simple image stitching application built using OpenCV and PyQt5. The application allows users to select two images, perform stitching to create a panorama, and visualize the result along with keypoint matches.

## Features

- Image selection through a graphical interface
- SIFT feature detection and matching
- Homography estimation for image alignment
- Smooth blending of images to create a seamless panorama
- Visualization of the stitched image and keypoint matches

## Requirements

- Python 3.x
- OpenCV 4.x
- PyQt5

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/image-stitching-app.git
    cd image-stitching-app
    ```

2. Install the required packages:
    ```sh
    pip install opencv-python-headless pyqt5
    ```

## Usage

1. Navigate to the project directory:
    ```sh
    cd image-stitching-app
    ```

2. Run the application:
    ```sh
    python app.py
    ```

3. Use the graphical interface to select the paths of the two images you want to stitch. Click the "Stitch Images" button to start the stitching process.

## How It Works

1. **Feature Detection**: SIFT (Scale-Invariant Feature Transform) is used to detect and compute keypoints and descriptors for both images.
2. **Feature Matching**: BFMatcher (Brute Force Matcher) with KNN (k-nearest neighbors) is used to find matches between descriptors.
3. **Homography Calculation**: A homography matrix is computed using RANSAC to find the best alignment between the two sets of keypoints.
4. **Image Warping and Blending**: The second image is warped according to the homography matrix and blended with the first image using a smoothing window to create a seamless panorama.

## File Structure

- `app.py`: Main application script containing the GUI and image stitching logic.

## Example

![image](https://github.com/user-attachments/assets/8f79877e-3d48-4a67-abb0-b57bdfcb3e0a)

![image](https://github.com/user-attachments/assets/38f8b7d2-dd2a-4b2a-b560-59af0d2a24f5)



## Acknowledgements

This project utilizes the OpenCV library for computer vision tasks and PyQt5 for creating the graphical user interface.

## License

This project is licensed under the MIT License.
