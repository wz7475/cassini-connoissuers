import numpy as np
import cv2

def close_mask_clusters(img: np.ndarray, dilation_kernel_size: int = 3, closing_kernel_size: int = 5):
  img = img.astype('uint8')
  dilation_kernel = np.ones((3, 3), dtype=np.uint8)
  closing_kernel = np.ones((5, 5), dtype=np.uint8)
  dilated_mask = cv2.dilate(img, dilation_kernel, iterations=3)
  closed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, closing_kernel)
  eroded_mask = cv2.erode(closed_mask, dilation_kernel, iterations=3)
  return eroded_mask
