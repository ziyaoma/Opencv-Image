import cv2.cv2 as cv2
import numpy as np

class Sketcher:
    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        cv2.namedWindow(self.windowname, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.windowname+": mask", cv2.WINDOW_NORMAL)
        self.show()
        cv2.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv2.imshow(self.windowname, self.dests[0])
        cv2.imshow(self.windowname + ": mask", self.dests[1])

    # onMouse function for Mouse Handling
    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv2.EVENT_LBUTTONUP:
            self.prev_pt = None

        if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv2.line(dst, self.prev_pt, pt, color, 5)
            self.dirty = True
            self.prev_pt = pt
            self.show()


img = cv2.imread("images//4//2.jpg")
img_mask = img.copy()


# gray = cv2.cvtColor(img_mask,cv2.COLOR_BGR2GRAY)
# #inpaintMask = np.zeros(img.shape[:2], np.uint8)
# _,inpaintMask = cv2.threshold(gray,50,255,cv2.THRESH_BINARY_INV)
# cv2.imwrite("images\\4\\mask2.jpg",inpaintMask)
# res = cv2.inpaint(src=img_mask, inpaintMask=inpaintMask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
# #res = cv.inpaint(src=img_mask, inpaintMask=inpaintMask, inpaintRadius=3, flags=cv.INPAINT_NS)
# cv2.imwrite("images\\4\\res2.jpg",res)


inpaintMask = np.zeros(img.shape[:2], np.uint8)
# Create sketch using OpenCV Utility Class: Sketcher
sketch = Sketcher('image', [img_mask, inpaintMask], lambda: ((255, 255, 255), 255))
#cv2.namedWindow('Inpaint Output using NS Technique', cv2.WINDOW_NORMAL)
#cv2.namedWindow('Inpaint Output using FMM', cv2.WINDOW_NORMAL)

while True:
    ch = cv2.waitKey()
    if ch == 27:
        break
    if ch == ord('t'):
        res = cv2.inpaint(src=img_mask, inpaintMask=inpaintMask, inpaintRadius=30, flags=cv2.INPAINT_TELEA)

        cv2.imshow('Inpaint Output using FMM', res)
        cv2.imwrite("images\\4\\FMM-eye.png", res)
    if ch == ord('n'):
        # Use Algorithm proposed by Bertalmio, Marcelo, Andrea L. Bertozzi, and Guillermo Sapiro: Navier-Stokes, Fluid Dynamics, 		    and Image and Video Inpainting (2001)

        res = cv2.inpaint(src=img_mask, inpaintMask=inpaintMask, inpaintRadius=30, flags=cv2.INPAINT_NS)

        # cv.namedWindow('Inpaint Output using NS Technique', cv.WINDOW_NORMAL)
        cv2.imshow('Inpaint Output using NS Technique', res)
        cv2.imwrite("images\\4\\NS-eye.png", res)
    if ch == ord('r'):
        img_mask[:] = img
        inpaintMask[:] = 0
        sketch.show()


