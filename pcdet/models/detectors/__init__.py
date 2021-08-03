from .detector3d_template import Detector3DTemplate
from .pointpillar import PointPillar
from .p3d_rcnn import P3DRCNN

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'PointPillar': PointPillar,
    'P3DRCNN': P3DRCNN
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
