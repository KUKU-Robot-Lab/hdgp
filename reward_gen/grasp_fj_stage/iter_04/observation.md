The round was judged over at the four-hour limit, at epoch 2788 of 3000. The playback uses a snapshot of the checkpoint at that point and shows two consecutive episodes of one environment (30 s). The run itself was left training while the recording was made and has since reached epoch 2989; the figures below are given for both points where they differ.

What the robot does in the recording:
- In both episodes the hand is at the cup's side within the first three seconds and stays there for the rest of the episode. It does not drift away and it does not turn away from its start orientation: the palm keeps facing the cup's side and the fingers keep pointing forward across it.
- The fingers stay extended for the whole recording. They reach across the near side of the cup and touch it with their tips and middle segments, but they never curl around it. At no point in either episode does a fingertip pass the far side of the cup, and the thumb hangs beside and below the cup instead of opposing the fingers around it.
- A gap between the palm and the cup's side is visible in every frame. The palm never rests against the cup.
- The cup stands upright at its spawn position in every frame of both episodes. It is never tipped, never pushed noticeably and never lifted off the table.
- Which of the two episodes starts beside the cup and which starts from the raised pose cannot be told apart in the recording; after the first three seconds both look the same.

Metrics (mean over the last 150 epochs, at epoch 2788):
- approach_done was set in 0.97 of far-start episodes and in 1.00 of near-start episodes. envelope_done was set in 0.0097 of far-start and 0.0033 of near-start episodes, and the episode funnel value for envelope is 0.0000.
- Episode funnel: reach 1.00, grasp 1.00 (at least three digits touched at some step), envelope 0.0000, lift 0.0017, success 0.0000.
- On an average step 3.2 to 3.3 of the five digits touch the cup (thumb 0.85, middle 0.85, ring 0.82, index 0.81, little finger 0.0003), while the palm touches on between 0.001 and 0.013 of steps depending on the window (0.0013 at epoch 2788, 0.013 at epoch 2989).
- The mean palm-to-cup gap is 0.046 m.
- The total reward is 0.958. Its largest parts are grasp_stay 0.330, grasp_fingers 0.210, grasp_palm_close 0.155, grasp_thumb 0.155, grasp_wrap 0.077 and grip_squeeze 0.026. envelope_quality is 0.0001 and grasp_palm_touch is 0.0002; every lift term and the success bonus are 0.
- cup_disturb_pen is -0.002 and cup_tilt_pen is -0.005; no episode ended with the cup tipped.
- There were no successes in the last 600 epochs. The contact values recorded at the moment of success stopped changing after about epoch 2300 and still carry the values from the successes before that: palm touching 0.80 and 2.11 fingers touching. Those were the only grasps in the whole round in which the palm touched the cup, and they stopped.
- The total reward rose through the entire round, from 0.73 at epoch 728 to 0.87 at 1757 and 0.90 to 0.96 at the end, while over the same period palm contact stayed inside a band between about 0.001 and 0.013 of steps with no sustained direction, and the envelope was never completed in a measurable share of episodes.
