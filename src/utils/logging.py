from lightning_fabric.utilities.logger import _flatten_dict


def cfg2hparams(cfg: dict):
    cfg_flat = _flatten_dict(cfg, delimiter=".")
    keys = list(cfg_flat.keys())
    for k in keys:
        if k.endswith("_target_"):
            v = cfg_flat.pop(k)
            v_new = v.split(".")[-1]
            k_new = k.removesuffix("._target_")
            cfg_flat[k_new] = v_new

    return cfg_flat
