import yaml
import itertools
import numpy as np


class Anchor:
    def __init__(self, yml_config, smin=.2, smax=.9, ssd='SSD300'):
        self.config = self._parse_yml(yml_config, ssd)
        self.smin = smin
        self.smax = smax

    def _parse_yml(self, yml_config, ssd):
        # config = yaml.load(open(yml_config))
        # BaseLoader / SafeLoader / FullLoader / UnsafeLoader
        config = yaml.load(open(yml_config), Loader=yaml.FullLoader)
        return config[ssd]

    def generate_anchors(self):
        config = self.config
        default_boxes = []
        scales = config['scales']
        fm_sizes = config['fm_sizes']
        ratios = config['ratios']
        for m, fm_size in enumerate(fm_sizes):
            for i, j in itertools.product(range(fm_size), repeat=2):
                cx = (j + 0.5) / fm_size
                cy = (i + 0.5) / fm_size
                default_boxes.append([
                    cx,
                    cy,
                    scales[m],
                    scales[m]
                ])
                # aspect ratio为1时，作者还增加一种scale的default box：np.sqrt(scales[m] * scales[m + 1])
                default_boxes.append([
                    cx,
                    cy,
                    np.sqrt(scales[m] * scales[m + 1]),
                    np.sqrt(scales[m] * scales[m + 1])
                ])

                for ratio in ratios[m]:
                    r = np.sqrt(ratio)
                    default_boxes.append([
                        cx,
                        cy,
                        scales[m] * r,
                        scales[m] / r
                    ])

                    default_boxes.append([
                        cx,
                        cy,
                        scales[m] / r,
                        scales[m] * r
                    ])
        default_boxes = np.clip(default_boxes, 0.0, 1.0)
        # ( 8732, 4)
        return default_boxes

    def generate_scales(self):
        feats_num = len(self.config['fm_sizes'])
        scales = np.linspace(self.smin, self.smax, feats_num + 1)
        return scales
