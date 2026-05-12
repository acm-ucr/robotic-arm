import math

# curX = 0;
# curY = 0;
# curZ = 0;
# # Physical lengths (Replace these with your actual measurements in mm)
# L1 = 50.0  # Base to shoulder
# L2 = 120.0 # Shoulder to elbow
# L3 = 100.0 # Elbow to wrist

# def calculate_ik(x, y, z):
#     """Calculates inverse kinematics and returns angles in radians."""
#     # 1. Calculate Base Angle
#     theta1 = math.atan2(y, x)
    
#     # 2. Calculate horizontal distance
#     r = math.sqrt(x**2 + y**2)
    
#     # 3. Calculate distance from shoulder to target
#     z_adjusted = z - L1
#     D = math.sqrt(r**2 + z_adjusted**2)
    
#     # Check if target is out of physical reach
#     if D > (L2 + L3):
#         print("Target out of reach!")
#         return None
        
#     # 4. Calculate Elbow Angle (Law of Cosines)
#     cos_theta3 = (D**2 - L2**2 - L3**2) / (2 * L2 * L3)
#     # Clamp value between -1 and 1 to prevent math domain errors from float inaccuracies
#     cos_theta3 = max(-1.0, min(1.0, cos_theta3)) 
#     theta3 = math.acos(cos_theta3)
    
#     # 5. Calculate Shoulder Angle
#     alpha = math.atan2(z_adjusted, r)
#     beta = math.atan2(L3 * math.sin(theta3), L2 + L3 * math.cos(theta3))
#     theta2 = alpha + beta
    
#     # Return degrees for easier human debugging, or keep in radians
#     return math.degrees(theta1), math.degrees(theta2), math.degrees(theta3)

# def angle_to_servo_pos(servo_id, angle_degrees, min_angle=0, max_angle=180):
#     """Maps a physical angle to the servo's raw position units."""
#     # Adjust indexing since SERVO_IDS is 1-indexed but Python lists are 0-indexed
#     idx = servo_id - 1 
#     min_pos = TARGET_MIN_POSITIONS[idx]
#     max_pos = TARGET_MAX_POSITIONS[idx]
    
#     # Standard linear mapping (y = mx + b)
#     pos = min_pos + (angle_degrees - min_angle) * (max_pos - min_pos) / (max_angle - min_angle)
#     return int(max(min(pos, max_pos), min_pos)) # Clamp to safety limits


