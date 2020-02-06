from keras.layers import Layer
import tensorflow as tf
import keras.backend as K
import numpy as np


class PriorBox(Layer):
    def __init__(self, img_size, scales, ratios, offset=.5, variances=[0.1, 0.1, 0.2, 0.2], clip=True, normalizer=True,
                 **kwargs):
        super(PriorBox, self).__init__(**kwargs)
        # img_size (height, width)
        self.img_size = img_size
        self.ratios = ratios
        if isinstance(offset, (tuple, list)):
            # (height, width)
            self.offset = offset
        elif isinstance(offset, (int, float)):
            self.offset = (offset, offset)
        else:
            self.offset = (.5, .5)
        if len(scales) != 2:
            # 至少要有curr_scale next_scale
            raise ValueError('scales length should == 2')
        self.scales = scales
        # bounding regression中权重
        self.variances = variances
        self.clip = clip
        self.normalizer = normalizer

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        # ratios 里面不包含1  每个ratio其实对于两个ratio (ratio, 1/ratio)
        boxes_num = (len(self.ratios) + 1) * 2
        # 4个坐标 4个权重值 这里直接返回数值 不要反悔Tensorshape 否则ssd后续Concatenate会出现维度计算错误
        # 因为这里封装成了Dimension
        # 而不是数值 会出现计算错误 unhashable type: 'Dimension'
        # return tf.TensorShape((input_shape[0], input_shape[1], input_shape[2], boxes_num, 8))
        return input_shape[0], input_shape[1], input_shape[2], boxes_num, 8

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        fh, fw = input_shape[1:3]
        img_height, img_width = self.img_size
        size = min(img_height, img_width)
        # get box
        if 1 not in self.ratios:
            self.ratios.append(1)
        default_boxes = []
        for ratio in self.ratios:
            if ratio == 1:
                default_boxes.append([
                    self.scales[0],
                    self.scales[0]
                ])
                default_boxes.append([
                    np.sqrt(self.scales[0] * self.scales[1]),
                    np.sqrt(self.scales[0] * self.scales[1]),
                ])
            else:
                r = np.sqrt(ratio)
                default_boxes.append([
                    self.scales[0] * r,
                    self.scales[0] / r
                ])
                default_boxes.append([
                    self.scales[0] / r,
                    self.scales[0] * r
                ])
        boxes_num = len(default_boxes)
        default_boxes = np.array(default_boxes)
        # 将其转到原图
        default_boxes = default_boxes * size
        # get center coord
        offset_y = self.offset[0]
        offset_x = self.offset[1]
        step_y, step_x = img_height / fh, img_width / fw
        cy = np.linspace(offset_y * step_y, img_height - step_y / 2, fh)
        cx = np.linspace(offset_x * step_x, img_width - step_x / 2, fw)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)
        cy_grid = np.expand_dims(cy_grid, -1)

        prior_boxes = np.zeros((fh, fw, boxes_num, 4))

        prior_boxes[:, :, :, 0] = np.tile(cx_grid, (1, 1, boxes_num))  # Set cx
        prior_boxes[:, :, :, 1] = np.tile(cy_grid, (1, 1, boxes_num))  # Set cy
        prior_boxes[:, :, :, 2] = default_boxes[:, 0]  # Set w
        prior_boxes[:, :, :, 3] = default_boxes[:, 1]  # Set h

        # xmin ymin, xmin, ymin
        boxes_xy, boxes_wh = prior_boxes[..., 0:2], prior_boxes[..., 2:4]
        prior_boxes[:, :, :, 0:2] = boxes_xy - boxes_wh / 2
        prior_boxes[:, :, :, 2:4] = boxes_xy + boxes_wh / 2
        # 进行裁剪
        if self.clip:
            x_coords = prior_boxes[:, :, :, [0, 2]]
            x_coords[x_coords >= img_width] = img_width - 1
            x_coords[x_coords < 0] = 0
            prior_boxes[:, :, :, [0, 2]] = x_coords
            y_coords = prior_boxes[:, :, :, [1, 3]]
            y_coords[y_coords >= img_height] = img_height - 1
            y_coords[y_coords < 0] = 0
            prior_boxes[:, :, :, [1, 3]] = y_coords
        if self.normalizer:
            prior_boxes[:, :, :, [0, 2]] /= img_width
            prior_boxes[:, :, :, [1, 3]] /= img_height
            # 让其最大值小于等于1.
            prior_boxes = np.minimum(np.maximum(prior_boxes, 0.), 1.)
        variances = np.zeros_like(prior_boxes)
        variances += self.variances
        prior_boxes = np.concatenate([prior_boxes, variances], axis=-1)
        prior_boxes = K.expand_dims(K.constant(prior_boxes, dtype=K.dtype(inputs)), 0)
        prior_boxes = K.tile(prior_boxes, (K.shape(inputs)[0], 1, 1, 1, 1))
        return prior_boxes
