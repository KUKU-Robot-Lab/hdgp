"""RH56F1 hand synergy basis — rh56f1_grasp_pca5.pt(uncentered PCA) 리터럴화.

inspire grasp 시연 22개 → RH56F1 6관절 remap 후 PCA5 (compute_rh56f1_grasp_pca.py).
uncentered(q = coeffs·basis, anchor=0): PC1(97.9%)=엄지+4지 조율 감김(firm envelope).
수동 편집 금지 — .pt에서 재생성. 행=PC1..5, 열=[thumb_1,thumb_2,index,middle,ring,pinky].
"""

HAND_SYNERGY_BASIS = [
    [ 0.505589,  0.125215,  0.423345,  0.435809,  0.402949,  0.444052],
    [ 0.819645, -0.072335, -0.268828, -0.407132,  0.035564, -0.289240],
    [-0.199004, -0.206525, -0.453234, -0.313665,  0.654395,  0.430937],
    [ 0.148294,  0.130984, -0.263052, -0.108269, -0.614424,  0.708814],
    [ 0.103775, -0.511733, -0.497201,  0.681511, -0.090817, -0.086290],
]

# uncentered PCA: anchor 없음 (q* = coeffs @ basis)
HAND_SYNERGY_ANCHOR = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

HAND_SYNERGY_COEFF_MINS = [ 0.225818, -0.079133, -0.372891, -0.190409, -0.065222]
HAND_SYNERGY_COEFF_MAXS = [ 3.726285,  0.817982,  0.174442,  0.025599,  0.040431]
