import numpy as np
import cv2

# landmark 5-point: [left_eye, right_eye, nose, left_mouth, right_mouth]
src = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def warp_and_crop_face(img, landmark, image_size=(112, 112)):
    dst = src.copy()
    dst[:, 0] = dst[:, 0] * (image_size[0] / 112)
    dst[:, 1] = dst[:, 1] * (image_size[1] / 112)

    tform = cv2.estimateAffinePartial2D(landmark, dst, method=cv2.LMEDS)[0]
    if tform is None:
        return None

    aligned_face = cv2.warpAffine(img, tform, image_size, borderValue=0.0)
    return aligned_face