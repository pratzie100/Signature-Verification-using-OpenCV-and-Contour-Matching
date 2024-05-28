
import cv2

# Load the reference signature image
reference_image = cv2.imread('original_signature.jpg', 0)

# Load the test signature image
test_image = cv2.imread("Test_signature1.jpg", 0)

# Check if the images are loaded correctly
if reference_image is None:
    print("Error loading original_signature.jpg")
if test_image is None:
    print("Error loading Test_signature2.jpg")

# Apply binary thresholding to the images
ret, reference_image = cv2.threshold(reference_image, 127, 255, cv2.THRESH_BINARY)
ret, test_image = cv2.threshold(test_image, 127, 255, cv2.THRESH_BINARY)

# Find the contours in the reference image
reference_contours, _ = cv2.findContours(reference_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the contours in the test image
test_contours, _ = cv2.findContours(test_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get the largest contour in the reference image
reference_contour = max(reference_contours, key=cv2.contourArea)

# Get the largest contour in the test image
test_contour = max(test_contours, key=cv2.contourArea)

# Calculate the similarity between the two contours
score = cv2.matchShapes(reference_contour, test_contour, 1, 0.0)

# Set the threshold for the similarity score
threshold = 0.1

# Print the result
if score < threshold:
    print('The test signature matches the reference signature.')
else:
    print('The test signature does not match the reference signature.')
