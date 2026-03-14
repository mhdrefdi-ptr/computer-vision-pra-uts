from __future__ import annotations

import cv2
import numpy as np

from modules.preprocessing import convert_to_grayscale


def detect_orb_features(image: np.ndarray, nfeatures: int = 500):
    gray = convert_to_grayscale(image)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    keypoint_vis = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return keypoints, descriptors, keypoint_vis


def match_features(image1: np.ndarray, image2: np.ndarray, top_k: int = 50):
    kp1, des1, _ = detect_orb_features(image1)
    kp2, des2, _ = detect_orb_features(image2)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return [], None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)[:top_k]
    vis = cv2.drawMatches(image1, kp1, image2, kp2, matches, None, flags=2)
    return matches, vis

