from helpers import *
import cv2 as cv
import numpy as np
from utils import visualization_utils as vis_util
from PIL import Image
from heapq import heappush, heappop
from datetime import datetime

sess, detection_graph, category_index = load_model()
tensors = get_tensors(detection_graph)

# if exception is raised, just create the output folder named below
OUTPUT_FOLDER = 'frame_output/'
video = cv.VideoCapture(0)
cache = []
cache_size = 50
thresh = 0.85

while True:

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        tensors['out'],
        feed_dict={tensors['in']: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=thresh)

    # All the results have been drawn on the frame, so it's time to display it.
    cv.imshow('ID CARD DETECTOR', frame)

    heappush(cache, (-np.max(scores), len(cache), frame))
    if len(cache) == cache_size:
        score, _, best_frame = heappop(cache)
        if -score > thresh:
            best_frame = convert_rgb(best_frame)
            Image.fromarray(best_frame).save(f'{OUTPUT_FOLDER}{datetime.now()}.png', quality=95)
        cache = []

    # Press 'q' to quit
    if cv.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv.destroyAllWindows()