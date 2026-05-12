# USED TO FIND THE POSITION OF THE MOTORS MANUALLY WITH DIRECT KEYBOARD CONTROL (LINUX WAYLAND SAFE)

import time
import threading
import serial
import keyboard  # Option B: Replaces pynput

# ====== CONFIG ======
PORT = "/dev/ttyACM0"  
BAUDRATE = 1000000

# Keyboard movement settings
STEP_SIZE = 50            # How many position units to move per tick
LOOP_DELAY = 0.05         # 50ms tick rate (20 updates per second)
MOVE_TIME_MS = int(LOOP_DELAY * 1000) # Sync motor move time to tick rate

# Map keys to (Servo_ID, Direction_Multiplier)
KEY_MAPPING = {
    'a': (1, 1),   # Hold E -> Servo 2 moves up
    'd': (1, -1),
    'w': (2, 1),   # Hold W -> Servo 1 moves up
    's': (2, -1),  # Hold S -> Servo 1 moves down
    'r': (3, 1),   # Hold W -> Servo 1 moves up
    'f': (3, -1),
    't': (4, 1),
    'g': (4,-1),
    'j': (5, -1),
    'k': (5, 1),
    'u': (6, 1),
    'i': (6,-1),
    
}

# Track theoretical current positions (Defaulting to middle: 2048)
current_positions = {1: 2048, 2: 2048, 3: 2048, 4: 2048, 5: 2048, 6: 2048}
# =====================

# Setup Serial and Thread Lock
serial_lock = threading.Lock() 
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
except serial.SerialException:
    print(f"Warning: Could not open port {PORT}. Running without serial hardware.")
    ser = None


def checksum(data):
    return (~sum(data) & 0xFF)

def write_position(servo_id, position, move_time=MOVE_TIME_MS):
    if ser is None:
        return
    position = max(0, min(4095, int(position)))
    pos_low   = position & 0xFF
    pos_high  = (position >> 8) & 0xFF
    mt = int(move_time)
    time_low  = mt & 0xFF
    time_high = (mt >> 8) & 0xFF
    packet = [0xFF, 0xFF, servo_id, 7, 0x03, 0x2A,
              pos_low, pos_high, time_low, time_high]
    packet.append(checksum(packet[2:]))
    
    # LOCK THE PORT WHILE WRITING
    with serial_lock:
        ser.write(bytearray(packet))

def read_position(servo_id):
    """Read actual current position from a servo"""
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
        time.sleep(0.05)
        if ser.in_waiting >= 7:
            resp = ser.read(ser.in_waiting)
            if len(resp) >= 7:
                pos = resp[5] + (resp[6] << 8)
                return pos
    return None

def sync_positions_from_hardware():
    """Updates our software tracker with real hardware positions."""
    print("Syncing initial motor positions...")
    for servo_id in KEY_MAPPING.values():
        sid = servo_id[0]
        actual_pos = read_position(sid)
        if actual_pos is not None:
            current_positions[sid] = actual_pos
            print(f"Servo {sid} synced to position {actual_pos}")
    print("Sync complete.\n")


def continuous_movement_loop():
    """
    Main loop that directly polls the keyboard hardware.
    """
    print("========================================")
    print("🎮 KEYBOARD CONTROL ACTIVE (SUDO REQUIRED)")
    print("Hold W/S to move Servo 1")
    print("Hold A/D to move Servo 2")
    print("Hold ESC to stop and exit")
    print("========================================")
    
    while True:
        # Check for exit command
        if keyboard.is_pressed('esc'):
            print("Esc pressed. Exiting keyboard control...")
            break

        moved_servos = set()
        
        # Check every key in our mapping
        for key, (servo_id, direction) in KEY_MAPPING.items():
            if keyboard.is_pressed(key):
                # Calculate new position
                new_pos = current_positions[servo_id] + (direction * STEP_SIZE)
                
                # Clamp between 0 and 4095 to prevent hardware damage
                new_pos = max(0, min(4095, new_pos)) 
                
                # Only update if it actually changed
                if new_pos != current_positions[servo_id]:
                    current_positions[servo_id] = new_pos
                    moved_servos.add(servo_id)

        # Write updates to hardware for any servo that moved this tick
        for servo_id in moved_servos:
            write_position(servo_id, current_positions[servo_id], move_time=MOVE_TIME_MS)

        # Pause briefly to prevent overloading the CPU and serial port
        time.sleep(LOOP_DELAY)


if __name__ == "__main__":
    if ser is not None:
        sync_positions_from_hardware()
    continuous_movement_loop()
