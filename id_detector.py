from helpers import *
import cv2 as cv
import numpy as np
import dlib
from utils import visualization_utils as vis_util
from PIL import Image

ID_THRESH = 0.9
FACE_UPSAMPLE_TIMES = 3  # more upsampling means bigger image, more easier to find faces, as well as other objects
CONFIDENCE_RATE = 0.005  # larger values lead to smaller confidence score

dlib_model = 'model/saved_model/mmod_human_face_detector.dat'
output_path = 'out.png'


def crop_id(image, sess, tensors, category_index, debug=False):
    image_expanded = np.expand_dims(image, axis=0)
    (boxes, scores, classes, num) = sess.run(
        tensors['out'],
        feed_dict={tensors['in']: image_expanded})
    image, array_coord = vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=0,
        min_score_thresh=ID_THRESH
    )

    ymin, xmin, ymax, xmax = array_coord

    shape = np.shape(image)
    im_width, im_height = shape[1], shape[0]
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    image = convert_rgb(image)
    image = Image.fromarray(image)
    cropped_img = image.crop((left, top, right, bottom))  # might raise exception due to box out of scope
    if debug:
        cropped_img.save(output_path, quality=95)  # for debug
        print('saved ', output_path)
    return cropped_img, (left, right, top, bottom)


def find_face(image, model, debug=False):
    image = np.asarray(image)
    dets = model(image, FACE_UPSAMPLE_TIMES)
    print("Number of faces detected: {}".format(len(dets)))
    if len(dets) == 0:
        raise ValueError("No faces is found. Try increasing the UPSAMPLE parameter and run again.")
    dets = sorted(dets, key=lambda x: x.confidence, reverse=True)
    box = dets[0].rect
    if debug:
        win = dlib.image_window()
        rects = dlib.rectangles()
        rects.extend([d.rect for d in dets])

        win.clear_overlay()
        win.set_image(image)
        win.add_overlay(rects)
        dlib.hit_enter_to_continue()
    return dets[0].confidence, (box.left(), box.right(), box.top(), box.bottom())


def _calc_mid(left, right, top, bot):
    return round(left+right)//2, round(top+bot)//2


def _dist(point, target):
    x,y = point
    m,n = target
    return np.sqrt((x-m)**2 + (y-n)**2)


def _cal_confidence(dist):
    # linear function. score decreases as the distance increases
    return -CONFIDENCE_RATE * dist + 1


def is_id_card(crop_box, face_box, debug=False):
    xleft, xright, xtop, xbottom = crop_box
    yleft, yright, ytop, ybottom = face_box
    yleft += xleft; yright += xleft
    ytop += xtop; ybottom += xtop
    if debug:
        print('crop box: ', xleft, xright, xtop, xbottom)
        print('face box: ', yleft, yright, ytop, ybottom)

    crop_mid = _calc_mid(xleft, xright, xtop, xbottom)
    face_mid = _calc_mid(yleft, yright, ytop, ybottom)
    target = (xright+crop_mid[0])//2, crop_mid[1]

    return _cal_confidence(_dist(face_mid, target))


def detect(img, crop_args, face_reg_args, debug=False):
    sess, detection_graph, category_index = crop_args
    cnn_face_detector = face_reg_args

    tensors = get_tensors(detection_graph)
    try:
        # crop
        cropped_img, crop_box = crop_id(img, sess, tensors, category_index, debug)
        # find face
        face_confidence, face_box = find_face(cropped_img, cnn_face_detector, debug)
    except:
        import traceback
        traceback.print_exc()
        return 0, 0, 0
    # return
    confidence = is_id_card(crop_box, face_box, debug)
    return 1, confidence, face_confidence


if __name__ == "__main__":
    img = cv.imread('test_images/1.png')

    args = load_model()
    cnn_face_detector = dlib.cnn_face_detection_model_v1(dlib_model)

    ret = detect(img, args, cnn_face_detector, debug=True)

    print(ret)



