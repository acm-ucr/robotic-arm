# USED TO FIND THE POSITION OF THE MOTORS MANUALLY AND EXECUTE MOTION PROFILES

import time
import serial
import math
import os
import threading
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
TARGET_MAX_POSITIONS = [3000, 3000, 1000, 1000, 3800, 3300]
TARGET_MIN_POSITIONS = [1000, 1000, 3000, 3000, 1500, 1900]
MOVE_TIME = 200  # Default fallback for raw writes
# Expected camera ranges
CAM_X_MIN, CAM_X_MAX = 0.0, 1.0
CAM_Y_MIN, CAM_Y_MAX = 0.0, 1.0
CAM_Z_MIN, CAM_Z_MAX = 0.0, 1.0
OP_MIN, OP_MAX = 0, 300
# =====================

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
    move_claw = {5: TARGET_MIN_POSITIONS[4], 6:TARGET_MIN_POSITIONS[5]}
    move_arm = {1: TARGET_MIN_POSITIONS[0], 2: TARGET_MIN_POSITIONS[1], 3: TARGET_MIN_POSITIONS[2], 4: 2000}
    move_wrist = {4: TARGET_MIN_POSITIONS[3]}
    execute_synchronized_group_move(move_claw, 1)
    execute_synchronized_group_move(move_arm, 1)
    execute_synchronized_group_move(move_wrist, 1)

# ==========================================
# ====== MOTION PROFILING / INTERPOLATION ======
# ==========================================

def ease_in_out(t):
    """Returns a smooth ease-in-ease-out factor between 0 and 1 using a cosine wave."""
    return (1 - math.cos(t * math.pi)) / 2

def generate_motion_profile(start_pos, target_pos, steps, profile_type="ease"):
    """Generates a list of waypoint positions forming a motion profile."""
    positions = []
    for i in range(steps + 1):
        t = i / steps if steps > 0 else 1
        
        if profile_type == "ease":
            factor = ease_in_out(t)
        else: # linear fallback
            factor = t
            
        current_target = int(start_pos + (target_pos - start_pos) * factor)
        positions.append(current_target)
    return positions

def execute_profiled_move(servo_id, target_pos, duration_sec=1.0, steps=20, profile_type="ease"):
    """
    Executes a smooth, interpolated movement to a target position.
    """
    start_pos = read_position(servo_id)
    
    if start_pos is None:
        print(f"Error: Could not read starting position for Servo {servo_id}. Aborting move.")
        return

    # Generate the trajectory waypoints
    waypoints = generate_motion_profile(start_pos, target_pos, steps, profile_type)
    
    # Calculate the time to wait between sending each waypoint
    step_delay = duration_sec / steps
    
    # Tell the motor to execute each tiny step in exactly 'step_delay' milliseconds 
    # to maintain fluid motion without stalling
    step_move_time_ms = max(1, int(step_delay * 1000))

    print(f"Profiled Move -> Servo: {servo_id} | Path: {start_pos} to {target_pos} | Duration: {duration_sec}s")

    for pos in waypoints:
        write_position(servo_id, pos, move_time=step_move_time_ms)
        time.sleep(step_delay)

def execute_profiled_move_background(servo_id, target_pos, duration_sec=1.0, steps=20, profile_type="ease"):
    """
    Spawns a background thread to execute a profiled move without blocking the main script.
    Allows multiple servos to be commanded to move simultaneously.
    """
    thread = threading.Thread(
        target=execute_profiled_move, 
        args=(servo_id, target_pos, duration_sec, steps, profile_type),
        daemon=True # Ensures the thread dies if the main program closes
    )
    thread.start()
    return thread

def execute_synchronized_group_move(targets_dict, duration_sec=1.0, profile_type="ease"):
    """
    Moves multiple servos in perfect synchronization without threading jitter.
    Pass in a dictionary: {servo_id: target_position}
    Example: execute_synchronized_group_move({1: 2048, 2: 1024, 3: 4000}, duration_sec=2.0)
    """
    steps = duration_sec * 100
    starting_positions = {}
    waypoints_dict = {}

    # 1. Grab all starting positions
    for servo_id, target_pos in targets_dict.items():
        start_pos = read_position(servo_id)
        if start_pos is None:
            print(f"Skipping Servo {servo_id} - could not read start pos.")
            continue
        
        starting_positions[servo_id] = start_pos
        # Generate the math profile for this specific servo
        waypoints_dict[servo_id] = generate_motion_profile(start_pos, target_pos, steps, profile_type)

    if not waypoints_dict:
        return

    # 2. Execute the synchronized steps
    step_delay = duration_sec / steps
    step_move_time_ms = max(1, int(step_delay * 1000))

    for step_index in range(steps + 1):
        for servo_id, waypoints in waypoints_dict.items():
            write_position(servo_id, waypoints[step_index], move_time=step_move_time_ms)
        
        # Wait once per global step, rather than once per motor
        time.sleep(step_delay)
#execute_profiled_move_background(4, 1000, 4, 100)
#execute_profiled_move_background(6, 3000, 4, 100)
#execute_profiled_move(4, 2000, 4, 100)
#execute_profiled_move(6, 1000, 4, 100)
# item1 = {4: 1000, 6: 3000}
# item2= {4: 2000, 6:1000}

# execute_synchronized_group_move(item1)
# default_pos()
test = {1: 2000, 2: 2000, 3: 2000, 4: 2000, 5: 1000, 6: 1000}
execute_synchronized_group_move(test, 2)
time.sleep(1)
default_pos()


