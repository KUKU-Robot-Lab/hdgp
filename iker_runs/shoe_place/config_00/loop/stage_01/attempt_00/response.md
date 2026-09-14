```python
import numpy as np


def get_interaction_data(keypoint_coordinates):
    """
    Scene: a wooden shoe rack (beige board, blue keypoints 9, 11, 12, 15, 16, 17, 19, 20 on its
    top surface) already holds one shoe (white keypoints 7, 13, 14, 18 and green 6, 8), placed
    toe-forward (+x) on the right-hand (-y) part of the board. A second shoe lies on the table
    left of the rack (red keypoints 1-4): 1 = toe tip, 2 = heel (back of the sole),
    4 = left (+y) side of the midfoot, 3 = right (-y) side of the midfoot. It is yawed about
    13 degrees off the +x axis.

    Semantics: the rack shoe's toe box leans to the image-left and its lateral bulge is on the
    image-right, so it reads as a RIGHT shoe. The loose shoe's toe leans the other way, so it is
    the LEFT shoe, and a left shoe goes on the left (+y) of the right shoe. The only free space
    on the rack is also on the +y side (the column of keypoints 20/16/12 and 19/15/11). Only the
    loose shoe (keypoints 1-4) moves. The rack and the shoe already on it stay where they are.

    The rack is raised above the table, so the shoe has to be picked up and set down on it,
    which means grasp_mode = True.

    Placement, using all four shoe keypoints as one rigid body:
      * Current shoe frame: forward is the average of the heel->toe direction (2->1) and the
        direction perpendicular to the side-to-side vector (3->4). Each keypoint is written in
        local (forward, left) coordinates around the centroid of the four points, and its
        height is kept relative to the lowest shoe keypoint.
      * Target frame: forward follows the rack's rows (mean of 20,19 minus mean of 12,11),
        which is parallel to the shoe already on the rack, so the new shoe also points toe +x.
      * Target centroid: between rack keypoints 16 and 15, 25% of the way from 16 toward 15.
        That puts the shoe centreline in the middle of the free strip, leaving room from both
        the rack's left edge and the other shoe. The centroid sits level with 16/15 (level with
        the middle of the other shoe), moved 1 cm toward -x so the toe stays on the board.
      * Height: rack surface z (mean of the blue keypoints) + 2.5 cm for the lowest shoe
        keypoint (sole/heel edge), with the shoe's own relative heights kept.
    """
    object_to_interact = 'left shoe (on the table, keypoints 1-4)'
    keypoint_indices_to_interact = ['1', '2', '3', '4']
    grasp_mode = True

    kp = {k: np.asarray(v, dtype=float) for k, v in keypoint_coordinates.items()}

    def unit(vec):
        n = np.linalg.norm(vec)
        if n < 1e-9:
            raise ValueError('degenerate keypoint direction')
        return vec / n

    # ---- current shoe frame (yaw in the xy-plane) from all four shoe keypoints ----
    fwd_from_axis = unit(kp['1'][:2] - kp['2'][:2])          # heel -> toe
    left_from_sides = unit(kp['4'][:2] - kp['3'][:2])        # right side -> left side
    fwd_from_sides = np.array([left_from_sides[1], -left_from_sides[0]])
    cur_fwd = unit(fwd_from_axis + fwd_from_sides)
    cur_left = np.array([-cur_fwd[1], cur_fwd[0]])

    shoe_ids = keypoint_indices_to_interact
    cur_centroid = np.mean([kp[i][:2] for i in shoe_ids], axis=0)
    z_min = min(kp[i][2] for i in shoe_ids)

    # ---- target frame on the rack, aligned with the rack rows / the shoe already there ----
    rack_fwd = unit(((kp['20'] + kp['19']) - (kp['12'] + kp['11']))[:2])
    rack_left = np.array([-rack_fwd[1], rack_fwd[0]])

    rack_z = np.mean([kp[i][2] for i in ['11', '12', '15', '16', '19', '20']])
    sole_clearance = 0.025

    target_centroid = (0.75 * kp['16'][:2] + 0.25 * kp['15'][:2]) + (-0.01) * rack_fwd

    def place(p):
        rel = p[:2] - cur_centroid
        u = np.dot(rel, cur_fwd)
        v = np.dot(rel, cur_left)
        xy = target_centroid + u * rack_fwd + v * rack_left
        z = rack_z + sole_clearance + (p[2] - z_min)
        return np.array([xy[0], xy[1], z])

    ## final keypoint calculation for each keypoint in keypoint_indices_to_interact
    # toe tip: front of the placement, in line with the other shoe's toe
    keypoint_coordinates['1'] = place(kp['1'])
    # heel: back of the placement, between rack keypoints 12 and 11
    keypoint_coordinates['2'] = place(kp['2'])
    # right side of midfoot: faces the right shoe already on the rack (toward keypoint 15 / 8)
    keypoint_coordinates['3'] = place(kp['3'])
    # left side of midfoot: faces the rack's free left edge (toward keypoint 16)
    keypoint_coordinates['4'] = place(kp['4'])

    return object_to_interact, keypoint_indices_to_interact, grasp_mode, keypoint_coordinates
```
