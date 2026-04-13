# USED TO FIND THE POSITION OF THE MOTORS MANUALLY AND EXECUTE MOTION PROFILES

import time
import serial
import math
import os
from dotenv import load_dotenv
load_dotenv()

# If you have a specific external file you want to use, uncomment the line below:
# from interpolate import generate_profile 

# ====== CONFIG ======
# change to your port name and check device manager if unsure
PORT = os.getenv("SERIAL_PORT")
BAUDRATE = 1000000
SERVO_IDS = [1, 2, 3, 4, 5, 6]
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
    # allow override of MOVE_TIME if provided
    mt = MOVE_TIME if move_time is None else int(move_time)
    time_low  = mt & 0xFF
    time_high = (mt >> 8) & 0xFF
    packet = [0xFF, 0xFF, servo_id, 7, 0x03, 0x2A,
              pos_low, pos_high, time_low, time_high]
    packet.append(checksum(packet[2:]))
    ser.write(bytearray(packet))

def read_position(servo_id):
    """Read actual current position from a servo"""
    if ser is None:
        return None
    ser.reset_input_buffer()
    start_address = 0x38
    read_length = 2
    packet = [
        0xFF, 0xFF,
        servo_id,
        4,
        0x02,
        start_address,
        read_length
    ]
    packet.append(checksum(packet[2:]))
    ser.write(bytearray(packet))
    time.sleep(0.05)
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

execute_profiled_move(4, 1000, 4, 100)
# Example Usage (Uncomment to test):
# if __name__ == "__main__":
#     print("Starting positions:", read_all_positions())
#     # Move Servo 1 to position 2048 smoothly over 2 seconds using 40 intermediate steps
#     execute_profiled_move(servo_id=1, target_pos=2048, duration_sec=2.0, steps=40, profile_type="ease")