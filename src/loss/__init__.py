from .loss import Loss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_pose_relative import LossPoseRelative, LossPoseRelativeCfgWrapper

LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossPoseRelativeCfgWrapper: LossPoseRelative,
}

LossCfgWrapper = LossDepthCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper | LossPoseRelativeCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
