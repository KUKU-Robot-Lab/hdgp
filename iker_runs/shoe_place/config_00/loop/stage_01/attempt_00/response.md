The loose shoe on the table (red keypoints 1-4, toe at 1, heel at 2, sides 4 = +y and 3 = -y) is the one to move. The tan rack (blue keypoints 9, 11, 12, 15, 16, 17, 19, 20) and the shoe already on it (green keypoints 6 = heel sole, 8 = left sole side) stay put. The rack shoe sits in the right (-y) part of the rack with its toe pointing +x, so the free slot is the left (+y) strip between grid columns 12/16/20 and 11/15/19. The loose shoe reads as a left shoe (its forefoot curves toward -y), so putting it on the +y side of the rack shoe also matches the semantics. The loose shoe is yawed about 13 deg (toe leaning toward -y), so it is rotated about z to match the rack's forward direction, then translated as a rigid body. It has to be lifted onto a raised rack, so grasp mode is used.

```python
def get_interaction_data(keypoint_coordinates):
    """
    Move the loose (left) shoe, keypoints 1-4, onto the rack beside the shoe that is
    already there. Keypoints on the rack (9, 11, 12, 15, 16, 17, 19, 20) and on the
    rack shoe (6, 8) do not move.

    Alignment, computed from all matching keypoints and applied as one rigid motion
    to 1-4:
    - Yaw: turn the loose shoe's heel->toe axis (2->1) about its keypoint centroid so it
      matches the rack's forward direction (rack columns 12/11 -> 20/19). That makes it
      parallel to the rack shoe, which also points +x.
    - Forward (x): average two estimates. (a) The centroid sits on the middle rack row
      (15, 16), which is the rack shoe's mid-length. (b) The rotated heel (2) lines up
      in x with the rack shoe's heel keypoint 6.
    - Lateral (y): place the centroid in the middle of the free strip left (+y) of the
      rack shoe. Average two scale-free estimates. (a) 20 % of the way from column
      point 16 toward 15. (b) 1.13 grid spacings (16-15) to the +y side of the rack
      shoe's left sole keypoint 8. This leaves about 1 cm to the rack shoe and to the
      rack's left edge.
    - Height (z): raise the shoe so its heel keypoint 2 matches the rack shoe's heel
      keypoint 6. The shoe's lowest keypoint also stays at least 1 cm above the rack
      surface.
    The shoe has to be lifted onto a raised rack, so grasp_mode = True.
    """
    import numpy as np

    object_to_interact = "left shoe"
    keypoint_indices_to_interact = ["1", "2", "3", "4"]
    grasp_mode = True

    kp = {k: np.array(v, dtype=float) for k, v in keypoint_coordinates.items()}

    # Loose shoe: centroid and heading
    shoe_pts = np.stack([kp[i] for i in keypoint_indices_to_interact])
    centroid = shoe_pts.mean(axis=0)
    heading = kp["1"][:2] - kp["2"][:2]

    # Rack forward direction from the two visible grid columns
    rack_front = (kp["20"][:2] + kp["19"][:2]) / 2.0
    rack_back = (kp["12"][:2] + kp["11"][:2]) / 2.0
    rack_fwd = rack_front - rack_back

    theta = np.arctan2(rack_fwd[1], rack_fwd[0]) - np.arctan2(heading[1], heading[0])
    theta = (theta + np.pi) % (2.0 * np.pi) - np.pi
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])

    def rotated_offset(idx):
        return rot @ (kp[idx][:2] - centroid[:2])

    # Target centroid, x: middle rack row, and heel aligned with the rack shoe heel (6)
    x_row = (kp["15"][0] + kp["16"][0]) / 2.0
    x_heel = kp["6"][0] - rotated_offset("2")[0]
    target_x = (x_row + x_heel) / 2.0

    # Target centroid, y: middle of the free strip left (+y) of the rack shoe
    y_grid = kp["16"][1] + 0.2 * (kp["15"][1] - kp["16"][1])
    y_shoe = kp["8"][1] + 1.13 * (kp["16"][1] - kp["15"][1])
    target_y = (y_grid + y_shoe) / 2.0

    target_xy = np.array([target_x, target_y])

    # Height: heel keypoint level with the rack shoe heel, and at least 1 cm above the rack top
    rack_ids = ["9", "11", "12", "15", "16", "17", "19", "20"]
    rack_z = np.mean([kp[i][2] for i in rack_ids])
    dz_heel = kp["6"][2] - kp["2"][2]
    dz_min = (rack_z + 0.01) - shoe_pts[:, 2].min()
    dz = max(dz_heel, dz_min)

    def placed(idx):
        xy = target_xy + rotated_offset(idx)
        return np.array([xy[0], xy[1], kp[idx][2] + dz])

    ## final keypoint calculation for each keypoint in keypoint_indices_to_interact
    keypoint_coordinates["1"] = placed("1")  # toe
    keypoint_coordinates["2"] = placed("2")  # heel
    keypoint_coordinates["3"] = placed("3")  # right (-y) side, faces the rack shoe
    keypoint_coordinates["4"] = placed("4")  # left (+y) side, toward the rack's left edge

    return object_to_interact, keypoint_indices_to_interact, grasp_mode, keypoint_coordinates
```
