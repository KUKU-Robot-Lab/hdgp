```python
def get_interaction_data(keypoint_coordinates):
    """
    Scene reading:
    - The beige board on the right is the shoe rack. Blue keypoints 9, 11, 12, 15, 16, 17, 19, 20
      lie on its top surface. It is raised above the black table (its front edge thickness is visible).
    - One shoe already sits on the rack (keypoints 6, 7, 8, 13, 14, 18). Its toe points +x (up in the
      image) and its heel is at the bottom near keypoint 6. Keypoint 8 is on its +y (left) sole edge.
      It is stationary.
    - The loose shoe lies on the table to the left of the rack (keypoints 1, 2, 3, 4):
      1 = toe tip, 2 = heel (sole, next to the opening), 4 = +y side, 3 = -y side.
      Its toe is also roughly +x, but it is yawed about 13 deg toward -y.

    Task: move the loose shoe (keypoints 1-4) onto the rack beside the other shoe. The only free rack
    space is on the +y (left) side of the rack shoe, over the columns 19/15/11 and 20/16/12. The
    right side (17, 9) is at the rack edge.

    Placement, using all four loose-shoe keypoints as one rigid body:
    - Orientation: yaw the loose shoe (heel 2 -> toe 1) until it is parallel to the rack's fore-aft
      direction, the average of the columns 12->20, 11->19 and 9->17. The two shoes then point the
      same way, as a neat pair does.
    - Fore-aft: line its heel (2) up with the rack shoe's heel (6).
    - Lateral: put its centre line at the rack shoe's +y sole edge (8) + a small gap + the loose
      shoe's half width (|3-4|/2) + a sole margin (the sole is wider than the upper points 3/4).
      Cap it at the outermost rack column (20/16/12) so the shoe stays on the rack.
    - Height: raise it by the rack's height. The heel sole point 2 should sit at about the same
      height as the rack shoe's sole points 6/8. It is kept at least 2.5 cm above the rack surface
      to avoid penetration.
    - It must be lifted onto a raised rack and rotated, so this is a grasp (not a push).
    """
    import numpy as np

    kp = {k: np.asarray(v, dtype=float) for k, v in keypoint_coordinates.items()}
    keypoint_coordinates = dict(keypoint_coordinates)

    object_to_interact = 'loose shoe on the table (to the left of the rack)'
    keypoint_indices_to_interact = ['1', '2', '3', '4']
    grasp_mode = True

    def xy_unit(v):
        flat = np.array([v[0], v[1], 0.0])
        return flat / (np.linalg.norm(flat) + 1e-9)

    # Rack frame: forward (+x-ish, along the rack columns) and left (+y-ish).
    rack_fwd = xy_unit((kp['20'] - kp['12']) + (kp['19'] - kp['11']) + (kp['17'] - kp['9']))
    rack_left = np.array([-rack_fwd[1], rack_fwd[0], 0.0])

    # Yaw that aligns the loose shoe heading (heel 2 -> toe 1) with the rack forward direction.
    shoe_fwd = xy_unit(kp['1'] - kp['2'])
    yaw = np.arctan2(rack_fwd[1], rack_fwd[0]) - np.arctan2(shoe_fwd[1], shoe_fwd[0])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    # Lateral offset from the rack shoe heel (6), measured along rack_left.
    half_width = 0.5 * np.linalg.norm((kp['3'] - kp['4'])[:2])
    sole_margin = 0.015
    gap = 0.015
    lat_rack_shoe_edge = np.dot(kp['8'] - kp['6'], rack_left)
    lat_rack_outer_col = np.mean([np.dot(kp[k] - kp['6'], rack_left) for k in ('20', '16', '12')])
    lat_target = min(lat_rack_shoe_edge + gap + half_width + sole_margin, lat_rack_outer_col)

    # Heel of the loose shoe goes beside the rack shoe heel (same fore-aft position).
    heel_target_xy = kp['6'][:2] + lat_target * rack_left[:2]

    # Height: lift onto the rack so the heel sole point matches the rack shoe's sole points.
    rack_surface_z = np.mean([kp[k][2] for k in ('9', '11', '12', '15', '16', '17', '19', '20')])
    rack_shoe_sole_z = 0.5 * (kp['6'][2] + kp['8'][2])
    z_shift = max(rack_shoe_sole_z - kp['2'][2], rack_surface_z + 0.025 - kp['2'][2])

    def place(k):
        rel = kp[k] - kp['2']
        rel_xy = rot @ np.array([rel[0], rel[1], 0.0])
        return np.array([heel_target_xy[0] + rel_xy[0],
                         heel_target_xy[1] + rel_xy[1],
                         kp[k][2] + z_shift])

    ## final keypoint calculation for each keypoint in keypoint_indices_to_interact
    keypoint_coordinates['1'] = place('1')  # toe tip: in front of the heel, parallel to the rack shoe
    keypoint_coordinates['2'] = place('2')  # heel: beside keypoint 6, +y side, on the rack
    keypoint_coordinates['3'] = place('3')  # -y side: faces the rack shoe with a small gap
    keypoint_coordinates['4'] = place('4')  # +y side: toward the free left part of the rack

    return object_to_interact, keypoint_indices_to_interact, grasp_mode, keypoint_coordinates
```
