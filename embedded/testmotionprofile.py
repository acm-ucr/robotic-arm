# USED TO FIND THE POSITION OF THE MOTORS MANUALLY AND EXECUTE MOTION PROFILES

# https://files.seeedstudio.com/products/Feetech/101090142_Feetech_ST-3215-C046_Datasheet.pdf
# motor range = 0 - 4095, neutral pos - 2048
# physical dimensions:
# arm max length : 16 inches
# arm min length : 10 inches
# upper arm: 5.592 inches
# forearm: 124.09mm - 4.885 inches
# min and max motor positions:
# (min, max)
# Servo 1: (800, 3450)
# Servo 2: (850, 3200)
# Servo 3: (3100, 940)
# Servo 4: (2940, 880)
# Servo 5: (1050, 760) <- max rolls over 4095
# Servo 6: (2030, 3500) 
# waypoints:
# 



import time
import serial
import math
import os
import threading
import json
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()
serial_lock = threading.Lock()


# If you have a specific external file you want to use, uncomment the line below:
# from interpolate import generate_profile 

# ====== CONFIG ======
# change to your port name and check device manager if unsure
PORT = os.getenv("SERIAL_PORT")
BAUDRATE = 1000000
SERVO_IDS = [1, 2, 3, 4, 5, 6]
TARGET_MAX_POSITIONS = {1: 3450, 2: 3200, 3: 940, 4: 880, 5: 4095, 6: 3500}
TARGET_MIN_POSITIONS = {1: 800, 2: 850, 3: 3100, 4: 2940, 5: 0, 6: 2030}
MOVE_TIME = 200  # Default fallback for raw writes
# Expected camera ranges   
CAM_X_MIN, CAM_X_MAX = 0.0, 1.0
CAM_Y_MIN, CAM_Y_MAX = 0.0, 1.0
CAM_Z_MIN, CAM_Z_MAX = 0.0, 1.0
OP_MIN, OP_MAX = 0, 300
# =====================

# --- MQTT SETTINGS ---
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT   = 1883
MQTT_TOPIC  = "robotic_arm/command"

try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
except serial.SerialException:
    print(f"Warning: Could not open port {PORT}. Running without serial hardware.")
    ser = None

def checksum(data):
    return (~sum(data) & 0xFF)

def write_position(servo_id, position, move_time=None):
    if ser is None:
        return
    position = max(0, min(4095, int(position)))
    pos_low   = position & 0xFF
    pos_high  = (position >> 8) & 0xFF
    mt = MOVE_TIME if move_time is None else int(move_time)
    time_low  = mt & 0xFF
    time_high = (mt >> 8) & 0xFF
    packet = [0xFF, 0xFF, servo_id, 7, 0x03, 0x2A,
              pos_low, pos_high, time_low, time_high]
    packet.append(checksum(packet[2:]))
    
    # LOCK THE PORT WHILE WRITING
    with serial_lock:
        ser.write(bytearray(packet))

def read_position(servo_id):
    if ser is None:
        return None
    # LOCK THE PORT WHILE READING
    with serial_lock:
        ser.reset_input_buffer()
        start_address = 0x38
        read_length = 2
        packet = [0xFF, 0xFF, servo_id, 4, 0x02, start_address, read_length]
        packet.append(checksum(packet[2:]))
        
        ser.write(bytearray(packet))
        time.sleep(0.05) # Brief pause for hardware to respond
        
        if ser.in_waiting >= 7:
            resp = ser.read(ser.in_waiting)
            if len(resp) >= 7:
                pos = resp[5] + (resp[6] << 8)
                return pos
    return None

def read_all_positions():
    positions = {}
    for servo_id in SERVO_IDS:
        pos = read_position(servo_id)
        if pos is None:
            print(f"Servo {servo_id}: No response")
        else:
            print(f"Servo {servo_id}: {pos}")
        positions[servo_id] = pos
        time.sleep(0.02)  # small delay so servos don't collide on serial
    return positions

def default_pos():
    move_claw = {5: TARGET_MIN_POSITIONS[5], 6:TARGET_MIN_POSITIONS[6]}
    move_arm = {1: TARGET_MIN_POSITIONS[1], 2: TARGET_MIN_POSITIONS[2], 3: TARGET_MIN_POSITIONS[3], 4: 2000}
    move_wrist = {4: TARGET_MIN_POSITIONS[4]}
    move_multiple(move_claw, 2)
    move_multiple(move_arm, 2)
    move_multiple(move_wrist, 2)

# ==========================================
# ====== MOTION PROFILING / INTERPOLATION ======
# ==========================================

def generate_motion_profile(start_pos, target_pos, steps):
    """Generates a list of waypoint positions forming a motion profile."""
    positions = []
    for i in range(steps + 1):
        t = i / steps if steps > 0 else 1
        
        """Returns a smooth ease-in-ease-out factor between 0 and 1 using a cosine wave."""
        factor = (1 - math.cos(t * math.pi)) / 2
            
        current_target = int(start_pos + (target_pos - start_pos) * factor)
        positions.append(current_target)
    return positions

def move_single(servo_id, target_pos, duration_sec=1.0, steps=100):
    """
    Executes a smooth, interpolated movement to a target position.
    """
    start_pos = read_position(servo_id)
    
    if start_pos is None:
        print(f"Error: Could not read starting position for Servo {servo_id}. Aborting move.")
        return

    # Generate the trajectory waypoints
    waypoints = generate_motion_profile(start_pos, target_pos, steps)
    
    # Calculate the time to wait between sending each waypoint
    step_delay = duration_sec / steps
    
    # Tell the motor to execute each tiny step in exactly 'step_delay' milliseconds 
    # to maintain fluid motion without stalling
    step_move_time_ms = max(1, int(step_delay * 1000))

    print(f"Profiled Move -> Servo: {servo_id} | Path: {start_pos} to {target_pos} | Duration: {duration_sec}s")

    for pos in waypoints:
        write_position(servo_id, pos, move_time=step_move_time_ms)
        time.sleep(step_delay)

def execute_profiled_move_background(servo_id, target_pos, duration_sec=1.0, steps=100):
    """
    Spawns a background thread to execute a profiled move without blocking the main script.
    Allows multiple servos to be commanded to move simultaneously.
    """
    thread = threading.Thread(
        target=execute_profiled_move, 
        args=(servo_id, target_pos, duration_sec, steps),
        daemon=True # Ensures the thread dies if the main program closes
    )
    thread.start()
    return thread

# def execute_synchronized_group_move(targets_dict, duration_sec=1.0):
#     """
#     Moves multiple servos in perfect synchronization without threading jitter.
#     Pass in a dictionary: {servo_id: target_position}
#     Example: execute_synchronized_group_move({1: 2048, 2: 1024, 3: 4000}, duration_sec=2.0)
#     """

#     steps = math.ceil(duration_sec * 100)
#     starting_positions = {}
#     waypoints_dict = {}

#     # 1. Grab all starting positions
#     for servo_id, target_pos in targets_dict.items():
#         start_pos = read_position(servo_id)
#         if start_pos is None:
#             print(f"Skipping Servo {servo_id} - could not read start pos.")
#             continue
        
#         starting_positions[servo_id] = start_pos
#         # Generate the math profile for this specific servo
#         waypoints_dict[servo_id] = generate_motion_profile(start_pos, target_pos, steps)

#     if not waypoints_dict:
#         return

#     # 2. Execute the synchronized steps
#     step_delay = duration_sec / steps
#     step_move_time_ms = max(1, int(step_delay * 1000))

#     for step_index in range(steps + 1):
#         for servo_id, waypoints in waypoints_dict.items():
#             write_position(servo_id, waypoints[step_index], move_time=step_move_time_ms)
        
#         # Wait once per global step, rather than once per motor
#         time.sleep(step_delay)

def move_multiple(targets_dict, speed_units_per_sec=1500.0):
    """
    Moves multiple servos in perfect synchronization.
    Duration and steps are dynamically calculated based on the maximum change 
    in position to mimic a natural human hand speed.
    
    Pass in a dictionary: {servo_id: target_position}
    Example: execute_synchronized_group_move({1: 2048, 2: 1024, 3: 4000})
    """
    
    STEPS_PER_SECOND = 100.0  # Defines the resolution of the motion profile (100 Hz)
    starting_positions = {}
    max_delta = 0

    # 1. Grab all starting positions and find the longest required travel distance
    for servo_id, target_pos in targets_dict.items():
        start_pos = read_position(servo_id)
        if start_pos is None:
            print(f"Skipping Servo {servo_id} - could not read start pos.")
            continue
        
        starting_positions[servo_id] = start_pos
        
        # Calculate the absolute distance this specific motor needs to travel
        delta = abs(target_pos - start_pos)
        if delta > max_delta:
            max_delta = delta

    # Exit if no valid starting positions were found
    if not starting_positions:
        return

    # If no motors need to move, we can safely exit
    if max_delta == 0:
        return

    # 2. Calculate dynamic duration and steps based on the longest move
    duration_sec = max_delta / speed_units_per_sec
    
    # Enforce a minimum floor for very tiny movements to prevent jitter or 0 steps
    duration_sec = max(0.05, duration_sec) 
    
    # Calculate steps based on the required duration
    steps = math.floor(duration_sec * STEPS_PER_SECOND)
    steps = max(1, steps) # Ensure there is always at least 1 step

    # 3. Generate the math profile for every servo based on the shared `steps`
    waypoints_dict = {}
    for servo_id, start_pos in starting_positions.items():
        target_pos = targets_dict[servo_id]
        waypoints_dict[servo_id] = generate_motion_profile(start_pos, target_pos, steps)

    if not waypoints_dict:
        return

    # 4. Execute the synchronized steps
    step_delay = duration_sec / steps
    step_move_time_ms = max(1, int(step_delay * 1000))

    for step_index in range(steps + 1):
        for servo_id, waypoints in waypoints_dict.items():
            write_position(servo_id, waypoints[step_index], move_time=step_move_time_ms)
        
        # Wait once per global step, rather than once per motor
        time.sleep(step_delay)

# calculates the position the given motor needs to move by "move_angle" degrees 
# from the neutral angle of 180 degrees (pos 2048)
def position_of_angle(servo_id, move_angle):
    pos_offset = move_angle * 11.3
    return 2048 + pos_offset

# UNTESTED
# calculates the angle the arm needs to point to,
# taken from the x-z axis
# moves motor 1
# y is unused
def get_base_angle(dict, x, y, z) :
    if (x == 0) :
        baseAngle = 0
    elif (z == 0) :
        baseAngle = math.degrees(math.atan(2*x))
    else : 
        baseAngle = math.degrees(math.atan((2*x)/z)) # gets angle relative to the z axis
    dict[1] = position_of_angle(1, baseAngle)

# UNTESTED
# calculates length of the point from origin
# and extends arm to that length with motor 2 and 3
# (0, .., 0) : 2:850, 3:3100
# (1, .., 1) : 2:3200, 3:940
# ranges: motor 2: 2350, motor 3: 2160
def get_arm_length(dict, x, y, z) :
    if x==0 and y==0 and z==0 :
        lineLength = 0
    else :
        lineLength = math.sqrt(x*x + z*z)
    if (lineLength > 1) :
        lineLength = 1
    twoOffset = 2350 * lineLength
    threeOffset = 2160 * lineLength
    dict[2] = 850 + twoOffset
    dict[3] = 3100 - threeOffset

# calculates pitch angle (complementary angle to phi in spherical coords)
# and adds offset of that angle to motor 2's position
def get_pitch_angle(dict, x, y, z) :
    adjacent = math.sqrt(x*x + z*z)
    opposite = y
    if (adjacent == 0) :
        adjacent = 1
    pitchAngle = abs(math.degrees(math.atan(opposite/adjacent)))
    posOffset = pitchAngle * 15
    dict[2] = dict[2] - posOffset

def get_wrist_amt(dict, wrist) :
    wristPos = 3795 * wrist # 3795 is range of motion
    wristPos = 1060 + wristPos
    if (wristPos >= 4095) :
        wristPos = 4095
    dict[5] = wristPos

# ensures motors do not go outside of operating range
def motor_clamping(dict):
    for id in dict.keys():
        lo = min(TARGET_MIN_POSITIONS[id], TARGET_MAX_POSITIONS[id])
        hi = max(TARGET_MIN_POSITIONS[id], TARGET_MAX_POSITIONS[id])
        dict[id] = max(lo, min(hi, dict[id]))

# translates the x,y,z coords given from software in the perspective of the camera
# X = [-0.5, 0.5], -0.5 = left of camera, 0.5 = right of camera
# Y = [0, 1], 0 = bottom of camera, 1 = top of camera
# Z = [0, 1], 0 = claw as close to base as possible, 1 = arm fully extended
def move_arm(x, y, z, grip, wrist): 
    # dict = {1:2048, 2:850, 3:3100, 4:2940, 5:1050, 6:2030}
    print("x:", x, ", y:", y, ", z:", z, ", grip:", grip, " wrist:", wrist)
    dict = {1:2048, 2:0, 3:0, 4:0, 5:0, 6:0}
    get_base_angle(dict, x, y, z)
    print(dict)
    get_arm_length(dict, x, y, z)
    print(dict)
    get_pitch_angle(dict, x, y, z)
    print(dict)
    get_wrist_amt(dict, wrist)
    print(dict)
    motor_clamping(dict)
    print(dict)

    move_multiple(dict)

test = {1: 2048, 2: 850, 3:3100, 4: 880}
move_multiple(test)
time.sleep(2)

move_arm(0, 0, 0, 0, 0)
time.sleep(2)
# move_arm(0, 0, 1, 0, 0)
# time.sleep(2)
move_arm(0, 1, 1, 0, 0)
time.sleep(1)
move_arm(0, 1, 1, 0, 1)

# move_arm(0.5, 0, 0, 0, 0)
# time.sleep(2)
# move_arm(-0.5, 0, 0, 0, 0)
# time.sleep(2)
# move_arm(0.5, 1, 0, 0, 0)
# time.sleep(2)
# move_arm(-0.5, 1, 0, 0, 0)
# time.sleep(2)

# move_arm(-0.5, 0, 1, 0, 0)
# time.sleep(2)
# move_arm(0.5, 0, 1, 0, 0)
# time.sleep(2)
# move_arm(0.5, 1, 1, 0, 0)
# time.sleep(2)
# move_arm(-0.5, 1, 1, 0, 0)




