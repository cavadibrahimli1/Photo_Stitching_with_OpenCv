import cv2
import numpy as np
import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QMessageBox, QHBoxLayout, QFrame, QProgressBar
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

class ImageStitching:
    def __init__(self):
        self.ratio = 0.75
        self.min_match = 10
        self.sift = cv2.SIFT_create()
        self.smoothing_window_size = 800

    def registration(self, img1, img2):
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches = []
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        img_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite('matching.jpg', img_matches)
        if len(good_points) > self.min_match:
            image1_kp = np.float32([kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32([kp2[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
            return H
        return None

    def create_mask(self, img1, img2, version):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2
        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - offset

        mask = np.zeros((height_panorama, width_panorama))
        if version == 'left_image':
            start = max(0, barrier - offset)
            end = barrier + offset
            if end - start > 0:
                mask[:, start:end] = np.tile(np.linspace(1, 0, end - start).T, (height_panorama, 1))
            mask[:, :start] = 1
        else:
            start = max(0, barrier - offset)
            end = min(width_panorama, barrier + offset)
            if end - start > 0:
                mask[:, start:end] = np.tile(np.linspace(0, 1, end - start).T, (height_panorama, 1))
            mask[:, end:] = 1
        return cv2.merge([mask, mask, mask])

    def blending(self, img1, img2):
        H = self.registration(img1, img2)
        if H is None:
            return None
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3), dtype=np.float32)
        mask1 = self.create_mask(img1, img2, version='left_image')
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1.astype(np.float32) / 255.0
        panorama1 *= mask1

        mask2 = self.create_mask(img1, img2, version='right_image')
        panorama2 = cv2.warpPerspective(img2.astype(np.float32) / 255.0, H, (width_panorama, height_panorama)) * mask2

        result = panorama1 + panorama2
        result = np.clip(result * 255, 0, 255).astype(np.uint8)

        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]
        return final_result

class ImageStitchingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image_stitching = ImageStitching()

    def initUI(self):
        self.setWindowTitle('Image Stitching App')
        self.setGeometry(100, 100, 800, 800)

        main_layout = QVBoxLayout()

        title = QLabel('Image Stitching Application')
        title.setFont(QFont('Arial', 16))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)

        input_layout = QVBoxLayout()

        img1_layout = QHBoxLayout()
        self.label1 = QLabel('Image 1 Path:')
        img1_layout.addWidget(self.label1)
        self.image1Path = QLineEdit(self)
        img1_layout.addWidget(self.image1Path)
        input_layout.addLayout(img1_layout)

        img2_layout = QHBoxLayout()
        self.label2 = QLabel('Image 2 Path:')
        img2_layout.addWidget(self.label2)
        self.image2Path = QLineEdit(self)
        img2_layout.addWidget(self.image2Path)
        input_layout.addLayout(img2_layout)

        main_layout.addLayout(input_layout)

        self.stitchBtn = QPushButton('Stitch Images')
        self.stitchBtn.clicked.connect(self.stitchImages)
        main_layout.addWidget(self.stitchBtn)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)

        self.progressBar = QProgressBar(self)
        self.progressBar.setValue(0)
        main_layout.addWidget(self.progressBar)

        self.resultLabel = QLabel('Result')
        self.resultLabel.setAlignment(Qt.AlignCenter)
        self.resultLabel.setFixedHeight(300)
        main_layout.addWidget(self.resultLabel)

        self.matchingLabel = QLabel('Matches')
        self.matchingLabel.setAlignment(Qt.AlignCenter)
        self.matchingLabel.setFixedHeight(300)
        main_layout.addWidget(self.matchingLabel)

        self.setLayout(main_layout)

    def stitchImages(self):
        img1_path = self.image1Path.text()
        img2_path = self.image2Path.text()

        if img1_path and img2_path:
            img1_path = os.path.normpath(img1_path)
            img2_path = os.path.normpath(img2_path)

            # Print debug statements to verify file paths
            print(f"Loading Image 1 from: {img1_path}")
            print(f"Loading Image 2 from: {img2_path}")

            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            # Check if the images are loaded correctly
            if img1 is None:
                QMessageBox.information(self, 'Error', f'Could not load Image 1. Please check the file path: {img1_path}')
                return

            if img2 is None:
                QMessageBox.information(self, 'Error', f'Could not load Image 2. Please check the file path: {img2_path}')
                return

            self.progressBar.setValue(10)
            final = self.image_stitching.blending(img1, img2)
            self.progressBar.setValue(90)

            if final is not None:
                resultFileName = 'panorama.jpg'
                cv2.imwrite(resultFileName, final)
                pixmap = QPixmap(resultFileName)
                self.resultLabel.setPixmap(pixmap.scaled(self.resultLabel.size(), Qt.KeepAspectRatio))
                self.resultLabel.setAlignment(Qt.AlignCenter)

                matchPixmap = QPixmap('matching.jpg')
                self.matchingLabel.setPixmap(matchPixmap.scaled(self.matchingLabel.size(), Qt.KeepAspectRatio))
                self.matchingLabel.setAlignment(Qt.AlignCenter)
                self.progressBar.setValue(100)
            else:
                QMessageBox.information(self, 'Error', 'Image stitching failed due to insufficient matches.')
                self.progressBar.setValue(0)
        else:
            QMessageBox.information(self, 'Error', 'Please enter both image paths before stitching.')

def main():
    app = QApplication(sys.argv)
    ex = ImageStitchingApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
