from matplotlib import pyplot as plt
from scipy import ndimage
import cv2
import os
import numpy as np
import glob


# np.set_printoptions(threshold='nan')
def show_img(winname, img):
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, 600, 600)
    cv2.imshow(winname, img)
    cv2.waitKey(0)


# plt.imshow(img)
# plt.title(winname)
# plt.show()

class Segment:
    def __init__(self, segments=5):
        # define number of segments, with default 5
        self.segments = segments

    def kmeans(self, image):
        image = cv2.GaussianBlur(image, (7, 7), 0)
        vectorized = image.reshape(-1, 3)
        vectorized = np.float32(vectorized)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(vectorized, self.segments, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        res = center[label.flatten()]
        segmented_image = res.reshape((image.shape))
        # print('show')
        # cv2.imshow("label", label)
        # cv2.waitKey(0)
        # print('label', label.shape)
        # print('center', center)
        # print('ret', ret)
        # plt.imshow(label)
        # plt.title('label')
        # plt.show()
        # print('unique', np.unique(label))
        return label.reshape((image.shape[0], image.shape[1])), segmented_image.astype(np.uint8)

    def extractComponent(self, image, label_image, label):
        component = np.zeros(image.shape, np.uint8)
        component[label_image == label] = image[label_image == label]
        return component


def dewarp(img, approx):
    hei, wid, _ = img.shape
    pts = approx.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (255, 0, 255), 3)
    show_img('before', img)
    # dst = np.array([
    #     [0, 0],
    #     [wid - 1, 0],
    #     [wid - 1, hei - 1],
    #     [0, hei - 1]], dtype="float32")
    approx = approx.astype(np.float32)
    dst = np.array([
        [wid - 1, 0],
        [wid - 1, hei - 1],
        [0, hei - 1],
        [0, 0]], dtype="float32")
    M = cv2.getPerspectiveTransform(approx, dst)
    warp = cv2.warpPerspective(img, M, (wid, hei))
    # show_img('dewarped', warp)
    return warp


def find_paper(org_img, ls_img):
    hei, wid = ls_img[0].shape
    S= hei*wid
    print('hei, wid', hei, wid)
    found_paper = 0
    paper = np.array([
        [wid - 1, 0],
        [wid - 1, hei - 1],
        [0, hei - 1],
        [0, 0]])
    ls_approx = []
    ls_area = np.empty([0])
    areas = np.empty([0])
    # ls_cnts = []
    ls_rects = []
    kernel = np.ones((25, 25))
    print('leen', len(ls_img))
    best_cand = []
    for img in ls_img:
        # img = 255-img
        thresh = cv2.erode(img,  kernel=kernel, iterations=3)
        thresh = 255 - thresh
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # print('hi,', hierarchy.shape)
        if hierarchy is not None and len(hierarchy.shape)>2:
            hierarchy = np.squeeze(hierarchy)
        cl_im = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        for i,cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            rect = cv2.minAreaRect(cnt)
                   # x,y,w,h = rect[0][0],rect[0][1], rect[1][0],rect[1][1]
            # print('xywh', int(x),int(y),int(w),int(h), wid-5, hei -5)
            area = cv2.contourArea(cnt)
            print('area', area)
            # if hierarchy[2]<0: continue
            # if area/S<0.2: continue
            # print(hierarchy[i])
            # k= cv2.isContourConvex(cnt)
            # if not k:continue
            if w> wid-5 and h> hei-5:
                if len(hierarchy.shape)!=1 and hierarchy[i][-2]!=-1: continue
            if abs((x+w)/2-wid/2)> wid/4 or abs((y+h)/2-hei/2)> hei/4: continue
            # if w * h / (hei * wid) < 0.1: continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx)>10: continue
            print('len approx', len(approx))
            if len(approx)==4 and area> S*0.4 and w> wid*0.7 and h> hei*0.7:
                found_paper+=1
                best_cand = approx
            ls_approx.append(approx)
            ls_rects.append([x, y, w, h])
            cv2.polylines(cl_im, [approx], True, (255, 255, 0), 3)
            areas = np.append(areas, area)
            ls_area = np.append(ls_area, w * h)
        show_img('cnt', cl_im)
    # print('lsrect', ls_rects)
    print('found paper', found_paper)
    if found_paper>=1:
        select = best_cand
        select = np.squeeze(select)
        select = mapping(hei, wid, select)
        # if not check_non_dewarp(org_img,select):
        paper = select
        print('found polygon')
    elif len(ls_area)>0:
        print('found candidate areas')
        # max_area_id = np.argmax(ls_area)
        max_area_id = np.argmax(areas)
        # max_area_id = np.argmax(ls_circle)
        max_area = np.max(ls_area)
        select = ls_approx[max_area_id]
        # print('select', select, max_area)
        x, y, w, h = ls_rects[max_area_id]
        if w * h / (hei * wid) > 0.4:
            select = np.squeeze(select)
            select = mapping(hei, wid, select)
            # if not check_non_dewarp(org_img, select):
            paper = select
            print('found')
    return paper

def check_non_dewarp(img, approx):
    hei, wid,_ = img.shape
    tl, tr, br, bl = approx
    ops = True
    if np.all(image[:,0:max(tl[0],bl[0])]==255) or np.all(image[:,wid-min(tr[0],br[0]):wid]==255)\
            or np.all(image[0:max(tl[1],tr[1]),:]==255) or np.all(image[hei-min(bl[1],br[1]):hei,:]==255):
        ops = False
    return ops

def mapping(hei, wid, coors):
    print('hei, wid', hei, wid)
    tl, tr, br, bl = hei + wid, hei + wid, hei + wid, hei + wid
    a, b, c, d = [], [], [], []
    for coor in coors:
        dist_1 = np.linalg.norm(coor - [wid - 1, 0])
        if dist_1 < tr:
            tr = dist_1
            a = coor
        dist_2 = np.linalg.norm(coor - [wid - 1, hei - 1])
        if dist_2 < br:
            br = dist_2
            b = coor
        dist_3 = np.linalg.norm(coor - [0, hei - 1])
        if dist_3 < bl:
            bl = dist_3
            c = coor
        dist_4 = np.linalg.norm(coor - [0, 0])
        if dist_4 < tl:
            tl = dist_4
            d = coor
    mapped_coor = np.vstack([a, b, c, d])
    return mapped_coor


def norm_img(image):
    k = 35
    seg = Segment(k)
    label, result = seg.kmeans(image)
    # cv2.imshow("segmented", result)
    # cv2.waitKey(0)
    ls_img = []

    hei, wid = image.shape[0], image.shape[1]
    for i in range(k):
        result = seg.extractComponent(image, label, i)
        # cv2.imwrite('%s%i.png' % (outdir, i), result)
        # gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        ret3, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # thresh = gray
        thresh = 255-thresh
        # if np.average(thresh[hei//2-10:hei//2+10,wid//2-10:wid//2+10])==255: continue
        # show_img('thre', thresh)
        ls_img.append(thresh)
    # ret, thresh = cv2.threshold(gray, 120, 255, 0)
    # # show_img('den', denoised_img)
    # cv2.imwrite('%sthresh%i.png' % (outdir, i), thresh)
    #
        # cv2.namedWindow('extracted', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('extracted', 600, 600)
        # # cv2.imwrite('%sgray%i.png' % (outdir, i), gray)
        # cv2.imshow("extracted",result)
        # cv2.waitKey(0)
    # show_img('gray', gray )
    paper = find_paper(image,ls_img)
    print('paper', paper, paper.shape)
    warp = dewarp(image, paper)
    return warp


if __name__ == "__main__":
    import argparse
    import sys

    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    # ap.add_argument("-n", "--segments", required = False, type = int,
    #     help = "# of clusters")
    # args = vars(ap.parse_args())
    # dir = '/media/warrior/DATA/corporate_workspace/dewarp/inputs/Archive/Statement/'
    dir = '/media/warrior/DATA/corporate_workspace/dewarp/inputs/Archive/Payslip/'
    outdir = '%s/dw/' % dir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    files = glob.glob('%s/*.jpg'%dir)+glob.glob('%s/*.png'%dir)
    files = ['%s/013.png' % dir]
    # files = ['%s/001.jpg' % dir]
    for filename in files:
        # filename = '%s%s' % (dir, file)
        file = filename.split('/')[-1]
        print('file', file)
        image = cv2.imread(filename)
        warp = norm_img(image)
        cv2.imwrite('%s/dewarped%s' % (outdir, file), warp)
