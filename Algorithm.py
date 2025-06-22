import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import time
from scipy.ndimage import gaussian_filter1d
import openpyxl
from threading import Thread, Barrier
import serial # Required for arduino communication
from queue import Queue, Full
import keyboard # use for test save data
import psutil, os
from sklearn.linear_model import RANSACRegressor, LinearRegression

# Set process priority for better real-time performance
p = psutil.Process(os.getpid())
p.nice(psutil.HIGH_PRIORITY_CLASS)  # Windows-specific

# Barrier for synchronizing camera starts
start_barrier = Barrier(2)  # 2 cameras

class CameraThread(Thread):
    def __init__(self, cam_id, queue, barrier):
        super().__init__()
        self.cam_id = cam_id
        self.queue = queue
        self.cap = cv.VideoCapture(cam_id)
        # Convert constants to float32 for consistency and potential speedup
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv.CAP_PROP_EXPOSURE, -7.0) # Use float for exposure if possible
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
        self.running = True

    def run(self):
        start_barrier.wait() # Wait for both cameras to be ready
        while self.running:
            # It's better to read one frame and then retrieve, rather than grab multiple
            ret, frame = self.cap.read()
            if ret:
                timestamp = time.time()
                try:
                    # Clear the queue before putting a new frame to ensure the latest frame
                    if self.queue.full():
                        self.queue.get_nowait()
                    self.queue.put_nowait((timestamp, frame))
                except Full:
                    # This should ideally not happen with the queue.get_nowait() above,
                    # but good to have for robustness.
                    pass

    def stop(self):
        self.running = False
        self.cap.release()

class BallDetectionThread(Thread):
    def __init__(self, frame, lower_color, upper_color):
        Thread.__init__(self)
        self.frame = frame
        self.lower_color = lower_color.astype(np.uint8) # Ensure color arrays are uint8
        self.upper_color = upper_color.astype(np.uint8)
        self.result = None

    def run(self):
        # Apply low-pass filter (median blur) directly on BGR before HSV conversion
        filtered_frame = cv.medianBlur(self.frame, 5)
        # Run blob detection
        self.result = find_ball_BLOB(filtered_frame, self.lower_color, self.upper_color)
        
# Function for center of mass BLOB detection
def find_ball_BLOB(image, lower_color, upper_color):
    """Detects the largest colored blob in an image and returns its centroid."""
    
    # Convert image to HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Apply threshold to detect color
    mask = cv.inRange(hsv, lower_color, upper_color)
    
    # Morphological operations for noise reduction and shape closing
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7)) # Ellipse kernel can be better for circular objects
    closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # If contours are found, process them
    if contours:
        # Sort contours by area (largest first)
        largest_contour = max(contours, key=cv.contourArea)
        area = cv.contourArea(largest_contour)

        # Ensure the detected area is large enough to be a ball
        if area > 30: # This threshold might need tuning
            # Calculate centroid
            M = cv.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return cx, cy
                
    return None, None  # Return None if no ball is found

# Undistort a single point using camera matrix and distortion coefficients
def undistort_single_point(u, v, camera_matrix, dist_coeffs):
    distorted_point = np.array([[[u, v]]], dtype=np.float32)
    undistorted_point = cv.undistortPoints(distorted_point, camera_matrix.astype(np.float32), 
                                            dist_coeffs.astype(np.float32), None, camera_matrix.astype(np.float32))
    u_corrected, v_corrected = undistorted_point[0, 0]
    return u_corrected, v_corrected

# Triangulate 3D point from 2D image coordinates
def triangulate(R, T, mtx1, mtx2, point1, point2):
      #RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    P1 = mtx1 @ RT1 #projection matrix for C1
 
    #RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis = -1)
    P2 = mtx2 @ RT2 #projection matrix for C2
    #DLT
    A = [point1[1]*P1[2,:] - P1[1,:],
        P1[0,:] - point1[0]*P1[2,:],
        point2[1]*P2[2,:] - P2[1,:],
        P2[0,:] - point2[0]*P2[2,:]]
    A = np.array(A).reshape((4,4))
    #print('A: ')
    #print(A)
 
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices = False)
 
#     print('Triangulated point: ')
#     print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]

# Fit parabolic curve for trajectory prediction
def fit_parabolic_curve(time_data, position_data):
    time_data = np.array(time_data, dtype=np.float32)
    position_data = np.array(position_data, dtype=np.float32)
    
    # For X and Y, assuming linear motion (or nearly linear for short trajectories)
    # This aligns with your original code's use of B_matrix
    B_matrix_xy = np.vstack([time_data, np.ones(len(time_data), dtype=np.float32)]).T
    
    # For Z, assuming parabolic motion due to gravity
    A_matrix_z = np.vstack([time_data**2, time_data, np.ones(len(time_data), dtype=np.float32)]).T
    
    # Using np.linalg.lstsq for least squares fitting
    coefficients_x, _, _, _ = np.linalg.lstsq(B_matrix_xy, position_data[:, 0], rcond=None)
    coefficients_y, _, _, _ = np.linalg.lstsq(B_matrix_xy, position_data[:, 1], rcond=None)
    coefficients_z, _, _, _ = np.linalg.lstsq(A_matrix_z, position_data[:, 2], rcond=None)
    
    return coefficients_x, coefficients_y, coefficients_z

# def fit_curve_yaxis(time_data, datay, sigma=1.2):
#     time_data = np.array(time_data, dtype=np.float32)
#     datay = np.array(datay, dtype=np.float32)
#     
#     # Smooth y with Gaussian Filter
#     y_smooth = gaussian_filter1d(datay, sigma=sigma).astype(np.float32)
# 
#     # Fit y = a·t + b
#     B_matrix = np.vstack([time_data, np.ones(len(time_data), dtype=np.float32)]).T
#     coefficients_y, _, _, _ = np.linalg.lstsq(B_matrix, y_smooth, rcond=None)
# 
#     return coefficients_y

def fit_curve_yaxis(time, datay):
    time = np.array(time).reshape(-1, 1)
    datay = np.array(datay)

    model = RANSACRegressor(estimator=LinearRegression(), residual_threshold=0.05)
    model.fit(time, datay)

    a = model.estimator_.coef_[0]
    b = model.estimator_.intercept_

    return np.array([a, b])

def fit_parabola_with_r2_plot(coefficients, dis_data, time_data, name, save_path=None):
    """
    Fits a parabolic or linear curve, plots it, and optionally saves the plot.

    Args:
        coefficients (np.array): Coefficients from the curve fit.
        dis_data (np.array): Measured displacement data.
        time_data (np.array): Corresponding time data.
        name (str): Name of the axis (e.g., 'X', 'Y', 'Z') for labeling.
        save_path (str, optional): Directory to save the plot. If None, plot is shown.
                                   If provided, plots are saved as 'fit_name.png'.
    """
    t_data = np.array(time_data, dtype=np.float32)
    dis_data = np.array(dis_data, dtype=np.float32)
    
    # Calculate fitted values
    if len(coefficients) == 3:
        a, b, c = coefficients
    else: # Assuming linear fit for 2 coefficients (as in Y-axis)
        b, c = coefficients
        a = 0.0
    
    t_fit = np.linspace(np.min(time_data), np.max(time_data), 100, dtype=np.float32)
    dis_fit = a * t_fit**2 + b * t_fit + c

    # Calculate R^2
    dis_pred = a * t_data**2 + b * t_data + c
    ss_res = np.sum((dis_data - dis_pred)**2)
    ss_tot = np.sum((dis_data - np.mean(dis_data))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0 # Handle division by zero
    
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.scatter(time_data, dis_data, color='blue', label='Measured Data')
    # Adjusted label to correctly show coefficients for both linear and parabolic
    if a != 0.0:
        plt.plot(t_fit, dis_fit, color='red', label=f'Fit: $y = {a:.2f}x^2 + {b:.2f}x + {c:.2f}$\n$R^2 = {r_squared:.4f}$')
    else:
        plt.plot(t_fit, dis_fit, color='red', label=f'Fit: $y = {b:.2f}x + {c:.2f}$\n$R^2 = {r_squared:.4f}$')
    
    plt.title(f"Parabolic Curve Fit for {name}-axis")
    plt.xlabel("Time (s)")
    plt.ylabel(f"Displacement {name} (m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        # Construct filename
        filename = os.path.join(save_path, f"fit_{name}_axis_plot.png")
        plt.savefig(filename)
        print(f"Plot saved to: {filename}")
    else:
        plt.show()

    plt.close() # Close the plot to free up memory

    return r_squared

# Predict position after a given time using the fitted equations
def predict(Eqx, Eqy, Eqz, sh_time):
    # Ensure all coefficients and time are float32
    Eqx = Eqx.astype(np.float32)
    Eqy = Eqy.astype(np.float32)
    Eqz = Eqz.astype(np.float32)
    sh_time = np.float32(sh_time)
    
    px = Eqx[0] * sh_time + Eqx[1]
    py = Eqy[0] * sh_time + Eqy[1]
    pz = Eqz[0] * sh_time**2 + Eqz[1] * sh_time + Eqz[2]
    return (px, py, pz)

# Save data for analysis
def save_excel(time_data, measured_posit, kalman_posit, k, AX_coeffs, AY_coeffs, AZ_coeffs):
    file_path = "Data.xlsx"
    try:
        wb = openpyxl.load_workbook(file_path)
    except FileNotFoundError:
        wb = openpyxl.Workbook()
        
    new_sheet_name = f"Data_noise{k}"
    if new_sheet_name in wb.sheetnames:
        ws = wb[new_sheet_name]
    else:
        ws = wb.create_sheet(new_sheet_name)
        
    ws.append(["Time", "Measured X", "Kalman X", "Velosity X (coeff)",
               "Measured Y", "Kalman Y", "Velosity Y (coeff)",
               "Measured Z", "Kalman Z", "Velosity Z (coeff)"])
    
    # Ensure coefficients are numpy arrays for consistent indexing
    AX_coeffs = np.array(AX_coeffs)
    AY_coeffs = np.array(AY_coeffs)
    AZ_coeffs = np.array(AZ_coeffs)
    time_data = np.array(time_data)
    
    for step in range(len(time_data)):
        # Use .item() to extract scalar values from numpy arrays for Excel
        ws.append([
            time_data[step].item(),
            measured_posit[step][0].item(), kalman_posit[step][0].item(), AX_coeffs[1].item() if len(AX_coeffs) > 1 else 0.0,
            measured_posit[step][1].item(), kalman_posit[step][1].item(), AY_coeffs[1].item() if len(AY_coeffs) > 1 else 0.0,
            measured_posit[step][2].item(), kalman_posit[step][2].item(), AZ_coeffs[1].item() if len(AZ_coeffs) > 1 else 0.0
        ])

    wb.save(file_path)
    print(f"Data appended to sheet '{new_sheet_name}' in '{file_path}'!")

# Function for changing the origin camera frame to the Defender frame (physical)
def send_position(position,time_pre):
    # Ensure position is a NumPy array of float32
    position = np.array(position, dtype=np.float32)
    
    # Scale and round for Arduino (assuming integers are expected)
    x_send = round(position[0] * 100)
    y_send = round(position[1] * 100)
    z_send = round(position[2] * 100)
    time_pre = round((time_pre -0.06)* 1000 - 469 )
    print(f'time_pre : {time_pre}')
    if time_pre > 300:
        time_pre = 50
    # Arduino communication logic
    if z_send > 0:
        
        arduino.write(f"({x_send},{y_send},{z_send},{time_pre})\n".encode())
        time.sleep(1.5)
        arduino.write(f"G00\n".encode())
    else:
        # Default Z if it's below ground
        arduino.write(f"({x_send},{y_send},{20},{time_pre})\n".encode())
        time.sleep(1.5)
        arduino.write(f"G00\n".encode())

def fine_time_to_ground(coefficients_z):
    # Ensure coefficients are float32
    coefficients_z = coefficients_z.astype(np.float32)
    
    a, b, c = coefficients_z
    
    # Solve for roots of a*t^2 + b*t + c = 0
    z_taget = 0.4
    # Use np.roots for robust root finding
    roots = np.roots(np.array([a, b, c - z_taget], dtype=np.float32))
    
    # Filter only the positive real root (time can't be negative)
    real_roots = [t.real for t in roots if t.imag == 0 and t.real >= 0]

    if real_roots:
        t_ground = np.min(real_roots)
        print(f"The ball hits the ground at t = {t_ground:.4f} seconds.")
    else:
        print("The ball does not hit the ground in this equation or roots are complex/negative.")
        t_ground = 0.6 # Default value if no valid root is found
    return t_ground

# --- Main Program ---

# Connect Arduino
arduino = None
try:
    arduino = serial.Serial('COM6', 115200 , timeout=1)
    time.sleep(2)  # Wait for Arduino to initialize
    print("Successfully connected to Arduino.")
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    print("Continuing without Arduino connection.")

# Load calibration data
try:
    data = np.load("calibration_data.npz", allow_pickle=True)
    mtx1, mtx2 = data["mtx"].astype(np.float32)
    dist1, dist2 = data["dist"].astype(np.float32)
    # Assuming R and T are also in your calibration_data.npz
    # If not, you might need to adjust or perform stereo calibration to get them.
    # For now, using your hardcoded values as fallback.
    R_calib =  np.eye(3, dtype=np.float32) 
    T_calib =  np.array([[-0.49],[0],[0]], dtype=np.float32)
    print("Calibration data loaded.")
except FileNotFoundError:
    print("calibration_data.npz not found. pls move the calibration_data.npz in same folder.")
    # Fallback if calibration data is not found
except Exception as e:
    print(f"Error loading calibration data: {e}. .")
# Initral floder
shoot_counter = 0

counter = 0
FPS = 1/30.0 # Use float for calculations

# Calibration mapping constants (ensure these are float32)
kx = 1.269036
ky = 1.571709234

# Physical setting and approximate the Y axis by different focal length
# It's better to get R and T from a stereo calibration (like in data.npz)
# If your R1 and T1 are meant for the 'triangulate' function as the
# transformation from camera1 to camera2, these should come from stereoCalibrate.
# For now, using your hardcoded values but recommending proper stereo calibration.
R_cam_transform = np.array([[-1,0,0],[0,0,1],[0,1,0]], dtype=np.float32) # Transformation matrix
T_cam_offset = np.array([1.555, -1.62, 1.345], dtype=np.float32) # Translation vector

# For triangulation, the R and T should be the relative rotation and translation
# between cam1 and cam2. If you calibrated them, use those values.
# If R1 and T1 in your `triangulate` function refer to an identity for cam1
# and the actual R and T for cam2, then your current setup is trying to pass
# a relative R and T directly. Let's assume R_calib and T_calib from stereo calibration.
R_triangulation = R_calib
T_triangulation = T_calib

# Data storage (ensure lists hold NumPy arrays of float32)
positions_3d = [] # Original 3D positions
measured_3d = [] # Corrected 3D positions
measured_3d1 = [] # This variable appears unused in the revised main loop.
velo_x = []
velo_y = []
velo_z = []
ball1_coords = []
ball2_coords = []
data_y_smoothed = []
time_y_smoothed = []
time_data = []
time_cam1 = []
time_cam2 = []
# Start multi-threaded camera capture
queue_left = Queue(maxsize=1)
queue_right = Queue(maxsize=1)

thread_left = CameraThread(2, queue_left, start_barrier)
thread_right = CameraThread(1, queue_right, start_barrier)

thread_left.start()
thread_right.start()

time.sleep(2) # Give cameras time to warm up

# Set thresholding to detect ball (ensure these are uint8 for cv.inRange)
lower_orange = np.array([30, 132, 42], dtype=np.uint8) # HSV values
upper_orange = np.array([65, 255, 255], dtype=np.uint8) # HSV values

# Initial call to triangulate to pre-load functions for speed
triangulate(R_triangulation, T_triangulation, mtx1, mtx2, (180.0, 96.0), (1.0, 182.0))
keyboard.is_pressed('s')

# Main loop flag
running_program = True
while running_program:
    # Mode selection
    print('\n--- Choose Mode ---')
    print('1: Servo control mode')
    print('2: Camera detection mode')
    print('3: Test static mode')
    print('4: Exit program')
    
    try:
        key_input = input("Enter choice: ")
    except EOFError: # Handle Ctrl+D or unexpected EOF
        print("EOF received, exiting.")
        running_program = False
        break
    
    if key_input == '1':
        print('\n--- Servo Control Mode ---')
        print('Enter "q" to return to main menu')
        print('Enter "c" to check Arduino response')
        while True:
            servo_cmd = input("Send command to servo: ").strip()
            if servo_cmd == 'q':
                break
            elif servo_cmd == 'c':
                itera = 0
                while arduino and arduino.in_waiting > 0 and itera < 10:
                    time.sleep(0.05) # Shorter sleep for faster checks
                    try:
                        line = arduino.readline().decode('utf-8', errors='ignore').strip()
                        print(f"Received from Arduino: {line}")
                    except Exception as e:
                        print(f"Error reading from Arduino: {e}")
                    itera += 1
            elif arduino: # Only send command if Arduino is connected
                arduino.write(servo_cmd.encode() + b'\n') # Add newline for serial consistency
            else:
                print("Arduino not connected.")
    
    elif key_input == '2' or key_input == '3':
        if key_input == '2':
            print('\n--- Camera Detection Mode ---')
        else :
            print('\n--- Test static Mode ---')
        print('Press "s" to save data')
        print('Press "q" to stop detection and return to main menu')
        
        # Reset state for a new detection run
        measured_3d.clear()
        data_y_smoothed.clear()
        time_y_smoothed.clear()
        time_data.clear()
        time_cam1.clear()
        time_cam2.clear()
        ball2_coords.clear()
        ball1_coords.clear()
        y_old = 0.0 # Initialize y_old for diffy calculation
        z_old = 0.0
        old_ax_coeffs = np.array([999.0, 999.0], dtype=np.float32)
        old_az_coeffs = np.array([999.0, 999.0, 999.0], dtype=np.float32)
        ball1_old = (180.0, 96.0)
        ball2_old = (180.0, 96.0)
        detection_active = True
        start_detection_time = 0.0
        
        while detection_active:
            #frame_process_start_time = time.time()
            frame_process_start_time = time.perf_counter()
            try:
                t1, frame1 = queue_left.get(timeout=1.0) # Add timeout for robustness
                t2, frame2 = queue_right.get(timeout=1.0)
            except Exception as e:
                print(f"Error getting frames: {e}. Retrying...")
                continue # Skip to next iteration if frames are not available
            time_cam1.append(t1)
            time_cam2.append(t2)
            # Calculate time difference between frames
            diff_timestamp = abs(t1 - t2)
            if diff_timestamp < 0.007: # If difference is significant, print it
                print(f"Timestamp difference: {diff_timestamp:.4f} s (potential desynchronization)")
            
            # Start ball detection threads
            det_thread1 = BallDetectionThread(frame1, lower_orange, upper_orange)
            det_thread2 = BallDetectionThread(frame2, lower_orange, upper_orange)
            det_thread1.start()
            det_thread2.start()
            det_thread1.join() # Wait for detection to complete
            det_thread2.join()

            ball1_pos = det_thread1.result
            ball2_pos = det_thread2.result
            if ball1_old != ball1_pos and ball2_old != ball2_pos:
                new = True
                #print('OK:')
            elif ball1_old == ball1_pos:
                ball2_old = ball2_pos
                new = False
            elif ball2_old == ball2_pos:
                ball1_old = ball1_pos
                new = False
            else:
                ball1_old = ball1_pos
                ball2_old = ball2_pos
                new = False
                
            
            if ball1_pos[0] is not None and ball2_pos[0] is not None and new == True: # Check if both balls are detected
                # Undistort points
                undis_ball1 = undistort_single_point(ball1_pos[0], ball1_pos[1], mtx1, dist1)
                undis_ball2 = undistort_single_point(ball2_pos[0], ball2_pos[1], mtx2, dist2)
                ball1_coords.append(undis_ball1) # Store (x, y) np.float32 arrays
                ball2_coords.append(undis_ball2)
                #print(f'ball1 : {undis_ball1}, ball2 : {undis_ball2}')
                
                # Assume t1 and t2 are already float64 from time.time()
                # diff_timestamp is naturally float64
                
#                 if diff_timestamp < 0.015:
#                     if t2 > t1: # Camera 2's frame is newer, interpolate Camera 1's position
#                         if len(ball1_coords) > 1:
#                             # Time difference between the last two captured frames of CAM1
#                             # Ensure time_cam1 stores float64 timestamps
#                             frame_dt_cam1 = time_cam1[-1] - time_cam1[-2]
#                             
#                             # Velocity of ball from CAM1's perspective (x,y in undistorted pixel space)
#                             # Calculations will promote to float64
#                             v_x = 0.5*(ball1_coords[-1][0] - ball1_coords[-2][0]) / frame_dt_cam1
#                             v_y = 0.5*(ball1_coords[-1][1] - ball1_coords[-2][1]) / frame_dt_cam1
#                             #v_x = 0.8*v_x
#                             # Time gap to interpolate: from cam1's last capture to cam2's current capture
#                             interpolation_time_gap = t2 - t1
#                             
#                             # Interpolated position at cam2's timestamp (will be float64)
#                             x_n = ball1_coords[-1][0] + v_x * interpolation_time_gap
#                             y_n = ball1_coords[-1][1] + v_y * interpolation_time_gap
#                             
#                             # Update undis_ball1 with the interpolated position
#                             # Explicitly cast back to float32 if triangulation expects float32
#                             undis_ball1 = np.array([x_n, y_n], dtype=np.float32)
#                             
#                     elif t1 > t2: # Camera 1's frame is newer, interpolate Camera 2's position
#                         if len(ball2_coords) > 1:
#                             # Time difference between the last two captured frames of CAM2
#                             frame_dt_cam2 = time_cam2[-1] - time_cam2[-2]
#                             
#                             # Velocity of ball from CAM2's perspective
#                             v_x = 0.5*(ball2_coords[-1][0] - ball2_coords[-2][0]) / frame_dt_cam2
#                             v_y = 0.5*(ball2_coords[-1][1] - ball2_coords[-2][1]) / frame_dt_cam2
#                             #v_x = 0.8*v_x
#                             # Time gap to interpolate: from cam2's last capture to cam1's current capture
#                             interpolation_time_gap = t1 - t2
#                             
#                             # Interpolated position at cam1's timestamp
#                             x_n = ball2_coords[-1][0] + v_x * interpolation_time_gap
#                             y_n = ball2_coords[-1][1] + v_y * interpolation_time_gap
#                             
#                             # Update undis_ball2 with the interpolated position
#                             undis_ball2 = np.array([x_n, y_n], dtype=np.float32)
#                 
                # After this block, undis_ball1 and undis_ball2 (potentially interpolated) 
                # will be used for triangulation. Ensure they are float32 if your
                # triangulate function expects float32.
                

                    
                    
                
                # Triangulate 3D position
                ball_3d_original = triangulate(R_triangulation, T_triangulation, mtx1, mtx2, undis_ball1, undis_ball2)*[-1,-1,1]
                
                # Apply transformation to Defender frame (physical)
                # Note: The `*[-1,-1,1]` part is applied after transformation,
                # consider if it's a part of the R_cam_transform or just a coordinate system flip.
                # Assuming it's a coordinate system flip after the R.dot(ball_3d) + T
                current_N_posi = (R_cam_transform.dot(ball_3d_original) + T_cam_offset)
                
                # Your manual calibration mapping
                x, y, z = current_N_posi[0], current_N_posi[1], current_N_posi[2]
                
                x1 = x * kx
                y1 = (y * ky) - (-0.169 * (x1)**2 + 0.7593 * (x1)) + 0.8
                y2 = y1 + (0.2 * y1**2 + 0.128 * y1 + 0.0319)
                z1 = z - (-0.031 * y2) - (-0.0209 * x1**2 + 0.0813 * x1)
                mz = -0.10768 * (z1 - 0.9213) + 0.0707
                z2 = z1 - mz * y2
                z3 = z2 + (-0.0408 * z2**2 + 0.393 * z2 - 0.3722) + 0.29
                mx = -0.1118 * (x1 - 1.5283) + 0.0673
                x2 = x1 - mx * y2
                z4 = z3 + (-0.049 * z3 + 0.11874)
                
                N_posi_calibrated = np.array([x2 + 0.07, y2, z4 + 0.09], dtype=np.float32)
                
                # Calculate `diffy` after the first measurement
                if len(measured_3d) > 0:
                    diffy = abs(N_posi_calibrated[1] - y_old)
                    diffz = abs(N_posi_calibrated[2] - z_old)
                else:
                    diffy = 0.0
                    diffz = 0.0
                y_old = N_posi_calibrated[1]
                z_old = N_posi_calibrated[2]

                #print(f"Current 3D Position: {N_posi_calibrated}")
                if len(measured_3d) >= 8 and key_input == '2':       
                    # Prediction trigger
                    if (error_x <= 0.1 and error_z <= 0.15 and len(data_y_smoothed) >= 4) or (time.time()- start_detection_time >= 0.35):
                        print("SHOOTING SEQUENCE INITIATED!")
                        shoot_counter += 1
                        plots_save_dir = os.path.join("trajectory_plots", f"shoot_{shoot_counter}")
                        # Determine time to hit the ground
                        # Ensure az_coeffs is a 3-element array (a, b, c) for fine_time_to_ground
                        # If fit_parabolic_curve returns 2 elements for Z, adjust this.
                        # Based on your fit_parabolic_curve for Z, it returns 3 coefficients.
                        
                        # Before calling fine_time_to_ground, ensure az_coeffs is (a, b, c) for Z-axis parabola
                        # If your `fit_parabolic_curve` for Z already returns 3 coefficients, use it directly.
                        # If it returns (b,c) for linear or (a,b) for quadratic, adjust.
                        # Assuming `fit_parabolic_curve` returns (a, b, c) for Z as designed.
                        
                        time_to_ground = fine_time_to_ground(az_coeffs)
                        
                        predicted_position = predict(ax_coeffs, ay_coeffs, az_coeffs, sh_time=time_to_ground)
                        if predicted_position[1] > 0 :
                            predicted_position[1] - 0.05
                        else:
                            predicted_position[1] + 0.05
                        predicted_position = [predicted_position[0],predicted_position[1],predicted_position[2]]
                            
                        print(f"Predicted target position: {predicted_position}")
                        time_predict = time_to_ground - time_data[-1]
                        if arduino:
                            send_position(predicted_position,time_predict)
                        else:
                            print("Arduino not connected, cannot send position.")

                        # Plot and save data
                        print("\n--- Plotting and Saving Data ---")
                        fit_parabola_with_r2_plot(ax_coeffs, np.array(measured_3d)[:, 0], time_data, 'X',save_path=plots_save_dir)
                        fit_parabola_with_r2_plot(ay_coeffs, data_y_smoothed, time_y_smoothed, 'Y',save_path=plots_save_dir)
                        fit_parabola_with_r2_plot(az_coeffs, np.array(measured_3d)[:, 2], time_data, 'Z',save_path=plots_save_dir)
                        
                        counter += 1
                        save_excel(time_data, measured_3d, measured_3d, counter, ax_coeffs, ay_coeffs, az_coeffs) # Kalman_posit is placeholder
                        
                        # Clear data for next detection run
                        measured_3d.clear()
                        data_y_smoothed.clear()
                        time_y_smoothed.clear()
                        time_data.clear()
                        time_cam1.clear()
                        time_cam2.clear()
                        ball2_coords.clear()
                        ball1_coords.clear()
                        
                        detection_active = False # Stop detection after a shot

                # Check if position is within acceptable bounds and timestamp difference is small
                elif (1.4 <= N_posi_calibrated[0] <= 3.11 and 
                    -0.65 <= N_posi_calibrated[1] <= 0.65 and 
                    diff_timestamp < 0.017 ):
                    
                    
                    measured_3d.append(N_posi_calibrated) # Store calibrated 3D position
                    
                    if len(measured_3d) == 1:
                        if arduino and key_input == '2': # Send G30 only if Arduino is connected
                            arduino.write(f"G30\n".encode()) # Signal Arduino to prepare
                            print("Sent G30 to Arduino.")
                        start_detection_time = time.time()
                    
                    current_elapsed_time = time.time() - start_detection_time
                    time_data.append(current_elapsed_time)

                    #if diffy <= 0.25: # Only add to y data if change is not too drastic
                    data_y_smoothed.append(N_posi_calibrated[1])
                    time_y_smoothed.append(current_elapsed_time)
                        
                    # Trajectory fitting and prediction
                    print(f"Current 3D Position: {N_posi_calibrated}")
                    if len(measured_3d) >= 3:
                        ax_coeffs, ay_coeffs, az_coeffs = fit_parabolic_curve(time_data, measured_3d)
                        
                        # Use the smoothed Y coefficients if enough data
                        if len(data_y_smoothed) >= 2:
                            ay_coeffs = fit_curve_yaxis(time_y_smoothed, data_y_smoothed)
                            
                        # Calculate error in coefficients for prediction trigger
                        error_x = np.linalg.norm(ax_coeffs - old_ax_coeffs) if len(old_ax_coeffs) == len(ax_coeffs) else float('inf')
                        error_z = np.linalg.norm(az_coeffs - old_az_coeffs) if len(old_az_coeffs) == len(az_coeffs) else float('inf')
                        
                        old_ax_coeffs, old_az_coeffs = ax_coeffs, az_coeffs
                        
                        
                
                
                            
                else:
                    # If ball not within bounds or time diff is too large, reset sequence
                    if len(measured_3d) > 0 and time.time() - start_detection_time  > 3 and key_input == '2': # Only clear if some data was collected
                        print("Ball out of bounds or desynchronized. Resetting trajectory data.")
                        measured_3d.clear()
                        data_y_smoothed.clear()
                        time_y_smoothed.clear()
                        time_data.clear()
                        time_cam1.clear()
                        time_cam2.clear()
                        ball2_coords.clear()
                        ball1_coords.clear()
                        if arduino:
                            arduino.write(f"G00\n".encode())
                            
                        y_old = 0.0 # Reset y_old
                        z_old =0.0

            else: # If one or both balls not detected
                if len(measured_3d) > 0 and time.time() - start_detection_time  > 3 and key_input == '2': # Clear partial data if detection is lost
                    print("Ball not detected in one or both cameras. Resetting trajectory data.")
                    measured_3d.clear()
                    data_y_smoothed.clear()
                    time_y_smoothed.clear()
                    time_data.clear()
                    time_cam1.clear()
                    time_cam2.clear()
                    ball2_coords.clear()
                    ball1_coords.clear()
                    y_old = 0.0 # Reset y_old
                    z_old = 0.0
                    if arduino:
                        arduino.write(f"G00\n".encode())

            # Calculate and apply delay to maintain target FPS
            frame_process_end_time = time.perf_counter()
            processing_duration = frame_process_end_time - frame_process_start_time
            delay_time = max(0.0, FPS - processing_duration - diff_timestamp*0.0155)
            #print(delay_time)
            time.sleep((delay_time**2)*0.95)

            # Check for keyboard commands
            if keyboard.is_pressed('s'):
                if len(measured_3d) > 3: # Only save if meaningful data exists
                    print("\n--- Manual Save Triggered ---")
                    # Re-calculate coefficients if not already done for saving
                    ax_coeffs, ay_coeffs, az_coeffs = fit_parabolic_curve(time_data, measured_3d)
                    if len(data_y_smoothed) >= 2:
                        ay_coeffs = fit_curve_yaxis(time_y_smoothed, data_y_smoothed) 
                    counter += 1
                    save_excel(time_data, measured_3d, measured_3d, counter, ax_coeffs, ay_coeffs, az_coeffs)
                else:
                    print("Not enough data to save. Need at least 3 points.")
                
                # Clear data after manual save or a "shot" for a fresh start
                measured_3d.clear()
                data_y_smoothed.clear()
                time_y_smoothed.clear()
                time_data.clear()
                time_cam1.clear()
                time_cam2.clear()
                ball2_coords.clear()
                ball1_coords.clear()
                y_old = 0.0 # Reset y_old
                z_old = 0.0
                time.sleep(0.5) # Debounce for 's' key
                
            elif keyboard.is_pressed('q'):
                print("Stopping camera detection mode.")
                detection_active = False
                
            elif keyboard.is_pressed('u') and arduino and len(measured_3d) > 0:
                print("Sending current position to Arduino (manual trigger).")
                arduino.write(f"G30\n".encode()) # Signal Arduino to prepare
                time.sleep(0.15)
                send_position(N_posi_calibrated,0.3)
                time.sleep(0.1) # Debounce for 'u' key
        
    elif key_input == '4':
        running_program = False
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

# --- Cleanup ---
print("\nExiting program. Cleaning up resources...")
thread_left.stop()
thread_right.stop()
thread_left.join()
thread_right.join()

cv.destroyAllWindows()
if arduino:
    arduino.close()
    print("Arduino connection closed.")
print("Program terminated.")