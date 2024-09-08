import cv2
import numpy as np
import glob

# ฟังก์ชัน calibrate กล้อง
def calibrate_camera(images_folder, grid_size=(7, 7)):
    # ข้อมูลของจุดในโลก 3D (เช่นจุด chessboard)
    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)

    # จัดเก็บจุดในโลก 3D และจุดในภาพ 2D จากการ calibrate
    objpoints = []
    imgpoints = []

    # โหลดภาพจากโฟลเดอร์ที่มีแพทเทิร์นสำหรับ calibrate (เช่น chessboard)
    images = glob.glob(images_folder + '/*.jpg')

    gray = None  # เริ่มต้นค่าเป็น None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Failed to load image {fname}")
            continue  # ถ้าโหลดภาพไม่สำเร็จ ข้ามไป

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # หาจุด corners จาก chessboard
        ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # แสดงจุด corners
            cv2.drawChessboardCorners(img, grid_size, corners, ret)
            cv2.imshow('Chessboard', img)
            cv2.waitKey(500)

    if gray is not None:
        # ทำการ calibrate กล้อง
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        cv2.destroyAllWindows()
        return mtx, dist
    else:
        raise ValueError("No valid images found for calibration. Make sure the chessboard pattern is visible.")
