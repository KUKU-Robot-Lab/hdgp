import isaaclab, openarm
print("IMPORT_OK isaaclab+openarm")
from openarm.tesollo.left.grasp_v2 import grasp_left_constants as C
print(f"LEFT NUM_OBS_BASE={C.NUM_OBS_BASE} CRITIC={C.NUM_CRITIC_OBS_BASE} STUDENT={C.NUM_STUDENT_OBS}")
import openarm.tesollo.left.grasp_v2.grasp_left_env_cfg as cfgmod
cfg = cfgmod.GraspLeftEnvCfg() if hasattr(cfgmod,'GraspLeftEnvCfg') else None
if cfg is not None:
    print(f"couple_four_fingers={getattr(cfg,'couple_four_fingers',None)} start_adr={getattr(cfg,'starting_adr_increments',None)} adr_max={getattr(cfg,'adr_num_increments',None)}")
    print(f"n_active_objects={len(cfgmod._ACTIVE_OBJECT_NAMES)} state_space={cfg.state_space}")
