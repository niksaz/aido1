import numpy as np
import cv2
from features.straight import constants

magic = 10 - 1


class LineDetectorHSV:
    # parameters from catkin_ws/src/00-infrastructure/duckietown/config/baseline/line_detector/line_detector_node/default.yaml
    dilation_kernel_size = 3
    canny_thresholds = [80, 200]
    hough_threshold = 2
    hough_min_line_length = 3
    hough_max_line_gap = 1

    hsv_white1 = np.array([0, 0, 150])
    hsv_white2 = np.array([180, 100, 255])
    hsv_yellow1 = np.array([25, 140, 100])
    hsv_yellow2 = np.array([45, 255, 255])
    hsv_red1 = np.array([0, 140, 100])
    hsv_red2 = np.array([15, 255, 255])
    hsv_red3 = np.array([165, 140, 100])
    hsv_red4 = np.array([180, 255, 255])


def notBlack(img):
    arr = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if 0 != img[i, j]:
                arr.append([i, j])
    return arr


def kernel_filter(img):
    kernelOpen = np.ones((constants.kernOpenSq, constants.kernOpenSq))
    kernelClose = np.ones((constants.kernClSq, constants.kernClSq))
    maskOpen_dotted = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernelOpen)
    maskClose_dotted = cv2.morphologyEx(maskOpen_dotted, cv2.MORPH_CLOSE, kernelClose)
    return maskClose_dotted


def find_edge(gray):
    edges = cv2.Canny(gray, LineDetectorHSV.canny_thresholds[0], LineDetectorHSV.canny_thresholds[1], apertureSize=3)
    return edges


def get_max_connected(gray):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(gray, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    if len(sizes) < 2:
        return np.uint(np.zeros(output.shape))
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    return np.uint8(img2)


def extremums(img):
    with_extremum = img
    cnts = cv2.findContours(with_extremum.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]

    c = max(cnts, key=cv2.contourArea)
    top = tuple(c[c[:, :, 1].argmin()][0])
    bottom = tuple(c[c[:, :, 1].argmax()][0])
    return top, bottom


def vector_from_points(point_a, point_b):
    return point_a[0] - point_b[0], point_a[1] - point_b[1]


def normal_vector(vector):
    return -vector[1], vector[0]


def point_plus_vector(point, vector):
    return point[0] + vector[0], point[1] + vector[1]


def with_coefficient(coefficient, point):
    return point[0] * coefficient, point[1] * coefficient


def int_point(point):
    return int(point[0]), int(point[1])


def needed_dots(top, bottom):
    step = ((top[0] - bottom[0]) / magic, (top[1] - bottom[1]) / magic)
    need_dots = [(bottom[0] + step[0] * i, bottom[1] + step[1] * i) for i in range(magic)]
    #need_dots.append(top)
    return need_dots


def precalc_normal_points(img):
    top, bottom = extremums(img)
    dots_between = needed_dots(top, bottom)
    normal = with_coefficient(1, normal_vector(vector_from_points(top, bottom)))
    normal_reverse = with_coefficient(-1, normal)
    return dots_between, normal, normal_reverse


def normal_points_to_this(dot, normal, normal_reverse):
    second = point_plus_vector(dot, normal)
    third = point_plus_vector(dot, normal_reverse)
    return second, third


def get_centroids(src):
    ret, thresh = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # You need to choose 4 or 8 for connectivity type
    connectivity = 8
    # Perform the operation
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]
    return centroids[1:]


tmpls = []

def line_approx(frame, pref, lower, upper):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    img = kernel_filter(mask)
    max_con = get_max_connected(img)

    dots_between, normal, normal_reverse = precalc_normal_points(max_con)

    empty_image = np.zeros(max_con.shape, dtype='uint8')

    for dot in dots_between:
        second, third = normal_points_to_this(dot, normal, normal_reverse)
        cv2.line(empty_image, int_point(third), int_point(second), (255, 255, 255), 3)

    intersections = cv2.bitwise_and(empty_image, max_con)

    for dot in get_centroids(intersections):
        global tmpls
        tmpls.append((dot[0], dot[1]))
        cv2.circle(frame, int_point(dot), 5, (255, 0, 0), -1)

    return frame



lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])
lower_white = np.array([0, 0, 150])
upper_white = np.array([180, 60, 255])


def go():
    # for cnt in range(1):
    #     screen_counter = cnt + 1
    #     screens_name_len = 4
    #     str_counter = str(screen_counter)
    #     nulls_to_add = screens_name_len - len(str_counter)
    #     for i in range(nulls_to_add):
    #         str_counter = '0' + str_counter

    frame = cv2.imread("./samples/sample.png")
    output_white = line_approx(np.copy(frame), 'white', lower_white, upper_white)
    output_dotted = line_approx(np.copy(frame), 'dotted', lower_yellow, upper_yellow)
    cv2.imwrite('./samples/white.png', output_white)
    cv2.imwrite('./samples/dotted.png', output_dotted)


if __name__ == '__main__':
    go()
