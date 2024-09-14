import numpy as np
import cv2

def close_mask_clusters(
  img: np.ndarray,
  dilation_kernel_size: int = 3,
  opening_kernel_size: int = 5,
  closing_kernel_size: int = 5,
  iterations: int = 1
):
  img = img.astype('uint8')
  dilation_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), dtype=np.uint8)
  opening_kernel = np.ones((opening_kernel_size, opening_kernel_size), dtype=np.uint8)
  closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), dtype=np.uint8)
  dilated_mask = cv2.dilate(img, dilation_kernel, iterations=iterations)
  closed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, closing_kernel)
  opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, opening_kernel)
  eroded_mask = cv2.erode(opened_mask, dilation_kernel, iterations=iterations)
  return eroded_mask
