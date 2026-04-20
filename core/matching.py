from __future__ import annotations

import cv2
import numpy as np

from core.image_ops import to_gray


def orb_detect(image: np.ndarray, nfeatures: int = 700):
    gray = to_gray(image)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    vis = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return keypoints, descriptors, vis


def bf_match(image_a: np.ndarray, image_b: np.ndarray, top_k: int = 80):
    kp_a, des_a, vis_a = orb_detect(image_a)
    kp_b, des_b, vis_b = orb_detect(image_b)

    if des_a is None or des_b is None:
        return {
            "keypoints_a": kp_a,
            "keypoints_b": kp_b,
            "descriptors_a": des_a,
            "descriptors_b": des_b,
            "vis_a": vis_a,
            "vis_b": vis_b,
            "matches": [],
            "match_vis": None,
        }

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des_a, des_b), key=lambda m: m.distance)[:top_k]
    match_vis = cv2.drawMatches(image_a, kp_a, image_b, kp_b, matches, None, flags=2)

    return {
        "keypoints_a": kp_a,
        "keypoints_b": kp_b,
        "descriptors_a": des_a,
        "descriptors_b": des_b,
        "vis_a": vis_a,
        "vis_b": vis_b,
        "matches": matches,
        "match_vis": match_vis,
    }
