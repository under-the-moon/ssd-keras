from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Concatenate, Reshape, Activation, Input
from keras.regularizers import l2
from models.prior_box import PriorBox
import numpy as np


def ssd300(input_shape,
           classes_num,
           weights_path=None,
           ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
           smin=.2,
           smax=.9,
           feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
           l2_reg=0.0005,
           mode='train'):
    end_points = {}
    input_tensor = Input(shape=input_shape)
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='block1_conv1')(input_tensor)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='block1_conv2')(conv1_1)
    end_points['block1'] = conv1_2
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block1_pool')(conv1_2)

    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='block2_conv1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='block2_conv2')(conv2_1)
    end_points['block2'] = conv2_2
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block2_pool')(conv2_2)

    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='block3_conv1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='block3_conv2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='block3_conv3')(conv3_2)
    end_points['block3'] = conv3_3

    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block3_pool')(conv3_3)

    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='block4_conv1')(pool3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='block4_conv2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='block4_conv3')(conv4_2)
    end_points['block4'] = conv4_3
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block4_pool')(conv4_3)

    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='block5_conv1')(pool4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='block5_conv2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='block5_conv3')(conv5_2)
    end_points['block5'] = conv5_3
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='block5_pool')(conv5_3)

    conv6_1 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='block6_conv1')(pool5)
    end_points['block6'] = conv6_1
    conv7_1 = Conv2D(1024, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='block7_conv1')(conv6_1)
    end_points['block7'] = conv7_1
    conv8_1 = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='block8_conv1')(conv7_1)
    conv8_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv8_1)
    conv8_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='block8_conv2')(conv8_1)
    end_points['block8'] = conv8_2
    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='block9_conv1')(conv8_2)
    conv9_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='block9_zeropadding1')(conv9_1)
    conv9_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='block9_conv2')(conv9_1)
    end_points['block9'] = conv9_2
    conv10_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='block10_conv1')(conv9_2)
    conv10_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='block10_conv2')(conv10_1)
    end_points['block10'] = conv10_2
    conv11_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='block11_conv1')(conv10_2)
    conv11_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='block11_conv2')(conv11_1)
    end_points['block11'] = conv11_2

    predictions = get_predictions(input_shape, end_points, feat_layers, ratios, classes_num, l2_reg, mode, smin, smax)

    model = Model(input_tensor, predictions)
    if weights_path is not None and weights_path.strip() != '':
        model.load_weights(weights_path)
    return model


def get_predictions(input_shape, end_points, feat_layers, ratios, classes_num, l2_reg, mode, smin, smax):
    confs = []
    locs = []
    priorboxes = []
    scales = np.linspace(smin, smax, len(feat_layers) + 1)
    for idx, feat_layer in enumerate(feat_layers):
        # [4, 6, 6, 6, 4, 4]
        # cls
        conf = Conv2D(2 * (len(ratios[idx]) + 1) * classes_num, (3, 3), padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='%s_conf_prediction' % feat_layer)(end_points[feat_layer])
        loc = Conv2D(2 * (len(ratios[idx]) + 1) * 4, (3, 3), padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='%s_loc_prediction' % feat_layer)(end_points[feat_layer])
        priorbox = PriorBox(img_size=input_shape[0:2], ratios=ratios[idx], scales=scales[idx:idx + 2],
                                 variances=[0.1, 0.1, 0.2, 0.2], offset=.5)(end_points[feat_layer])

        # reshape
        conf = Reshape((-1, classes_num), name='%s_conf_reshape' % feat_layer)(conf)
        loc = Reshape((-1, 4), name='%s_loc_reshape' % feat_layer)(loc)
        priorbox = Reshape((-1, 8), name='%s_priorbox_reshape' % feat_layer)(priorbox)
        confs.append(conf)
        locs.append(loc)
        priorboxes.append(priorbox)
    # (batch_size, total_boxes, classesnum)
    # total_boxes（38*38*4 + 19*19*6 + 10*10*6 + 5*5*6 + 3*3*4 + 1*1*4）= 8732
    mbox_confs = Concatenate(axis=1, name='mbox_confs')(confs)
    mbox_locs = Concatenate(axis=1, name='mbox_locs')(locs)
    mbox_priorboxes = Concatenate(axis=1, name='mbox_priorboxes')(priorboxes)
    # (batch, n_boxes_total, n_classes)
    mbox_confs_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_confs)
    # (batch, n_boxes_total, n_classes + 4 + 8)
    # predictions = Concatenate(axis=2, name='predictions')([mbox_confs_softmax, mbox_locs, mbox_priorboxes])
    if mode == 'train':
        predictions = [mbox_confs_softmax, mbox_locs, mbox_priorboxes]
    elif mode == 'test':
        # TODO
        predictions = None
    else:
        raise ValueError('mode: {%s} is not supported, please check it' % mode)
    return predictions


model = ssd300((300, 300, 3), classes_num=21)
model.summary()
