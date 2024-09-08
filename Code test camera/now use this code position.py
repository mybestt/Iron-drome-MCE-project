import cv2
import numpy as np

# เปิดกล้องสองตัว
cap1 = cv2.VideoCapture(0)  # กล้องตัวที่ 1
cap2 = cv2.VideoCapture(1)  # กล้องตัวที่ 2

# กำหนดขอบเขตสี HSV ของลูกปิงปอง (สีเหลือง)
lower_yellow = np.array([20, 100, 100])  # ค่าสี HSV ต่ำสุดสำหรับสีเหลือง
upper_yellow = np.array([30, 255, 255])  # ค่าสี HSV สูงสุดสำหรับสีเหลือง

# ขนาดของกริด (pixels)
grid_size = 100

def draw_grid(frame, grid_size):
    height, width = frame.shape[:2]
    # วาดเส้นกริดแนวนอน
    for y in range(0, height, grid_size):
        cv2.line(frame, (0, y), (width, y), (255, 255, 255), 1)
    # วาดเส้นกริดแนวตั้ง
    for x in range(0, width, grid_size):
        cv2.line(frame, (x, 0), (x, height), (255, 255, 255), 1)
    return frame

while True:
    # อ่านภาพจากกล้องตัวที่ 1
    ret1, frame1 = cap1.read()
    # อ่านภาพจากกล้องตัวที่ 2
    ret2, frame2 = cap2.read()
    
    # แปลงภาพจาก BGR เป็น HSV สำหรับกล้องตัวที่ 1
    hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    # แปลงภาพจาก BGR เป็น HSV สำหรับกล้องตัวที่ 2
    hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    
    # สร้าง Mask เพื่อกรองเฉพาะสีเหลืองสำหรับกล้องตัวที่ 1
    mask1 = cv2.inRange(hsv1, lower_yellow, upper_yellow)
    # สร้าง Mask เพื่อกรองเฉพาะสีเหลืองสำหรับกล้องตัวที่ 2
    mask2 = cv2.inRange(hsv2, lower_yellow, upper_yellow)
    
    # ใช้การ Blur เพื่อลด Noise สำหรับกล้องตัวที่ 1
    mask1 = cv2.GaussianBlur(mask1, (5, 5), 0)
    # ใช้การ Blur เพื่อลด Noise สำหรับกล้องตัวที่ 2
    mask2 = cv2.GaussianBlur(mask2, (5, 5), 0)
    
    # หาคอนทัวร์ (Contours) จาก mask ที่ได้จากกล้องตัวที่ 1
    contours1, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # หาคอนทัวร์ (Contours) จาก mask ที่ได้จากกล้องตัวที่ 2
    contours2, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # วนลูปในคอนทัวร์ที่เจอสำหรับกล้องตัวที่ 1
    for contour in contours1:
        if cv2.contourArea(contour) > 300:
            # หาจุดศูนย์กลางและรัศมีของวงกลมที่ครอบคอนทัวร์
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            if radius > 10 and radius < 50:
                # วาดวงกลมรอบๆ ลูกปิงปอง
                cv2.circle(frame1, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                # แสดงข้อความว่าตรวจพบลูกปิงปอง
                cv2.putText(frame1, "Ping Pong Detected (Cam 1)", (int(x - radius), int(y - radius)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # แสดงพิกัด (x, y) ในหน่วยพิกเซลบนเฟรม
                cv2.putText(frame1, f"Position: ({int(x)}, {int(y)})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # วนลูปในคอนทัวร์ที่เจอสำหรับกล้องตัวที่ 2
    for contour in contours2:
        if cv2.contourArea(contour) > 300:
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            if radius > 10 and radius < 50:
                # วาดวงกลมรอบๆ ลูกปิงปอง
                cv2.circle(frame2, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                # แสดงข้อความว่าตรวจพบลูกปิงปอง
                cv2.putText(frame2, "Ping Pong Detected (Cam 2)", (int(x - radius), int(y - radius)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # แสดงพิกัด (x, y) ในหน่วยพิกเซลบนเฟรม
                cv2.putText(frame2, f"Position: ({int(x)}, {int(y)})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # ปรับขนาดของภาพ
    frame1_resized = cv2.resize(frame1, (1920, 1080))
    frame2_resized = cv2.resize(frame2, (640, 480))

    # วาดกริดลงในภาพ
    frame1_grid = draw_grid(frame1_resized, grid_size)
    frame2_grid = draw_grid(frame2_resized, grid_size)

    # แสดงภาพผลลัพธ์จากกล้องตัวที่ 1
    cv2.imshow('Ping Pong Ball Detection - Camera 1', frame1_grid)
    # แสดงภาพผลลัพธ์จากกล้องตัวที่ 2
    cv2.imshow('Ping Pong Ball Detection - Camera 2', frame2_grid)
    
    # หากกดปุ่ม 'q' จะออกจากลูป
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องทั้งสองตัวและปิดหน้าต่างแสดงผล
cap1.release()
cap2.release()
cv2.destroyAllWindows()
