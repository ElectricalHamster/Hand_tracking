from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Add
from tensorflow.keras.models import Model

def _osa_module(inputs, num_layers, layer_c_out):
    # 定义OSA模块，这里简化为几层卷积层
    x = inputs
    for _ in range(num_layers):
        x = Conv2D(layer_c_out, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    return x

def transition_layer(inputs, filters, is_pool=True):
    # 转换层可以包括池化
    x = Conv2D(filters, (1, 1), strides=(1 if is_pool else 2))(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    if is_pool:
        x = MaxPooling2D(pool_size=(2, 2))(x)
    return x

def create_model(input_shape):
    inputs = Input(shape=input_shape)

    # Steam Block
    x = Conv2D(64, (3, 3), strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Repeated OSA Modules and Transition Layers
    x = _osa_module(x, num_layers=5, layer_c_out=32)
    x = transition_layer(x, 64, is_pool=True)

    x = _osa_module(x, num_layers=5, layer_c_out=40)
    x = transition_layer(x, 128, is_pool=True)

    x = _osa_module(x, num_layers=5, layer_c_out=48)
    x = transition_layer(x, 192, is_pool=False)

    x = _osa_module(x, num_layers=5, layer_c_out=56)  # More layers can be added similarly

    # Create and compile model
    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# 创建模型实例
import visualkeras
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D
from collections import defaultdict

model = create_model((384, 384, 3))

# visualkeras.layered_view(model).show() # display using your system viewer
# visualkeras.layered_view(model, to_file='output.png') # write to disk
# visualkeras.layered_view(model, to_file='output.png').show() # write and show

color_map = defaultdict(dict)
color_map[Conv2D]['fill'] = 'orange'
color_map[ZeroPadding2D]['fill'] = 'gray'
color_map[Dropout]['fill'] = 'pink'
color_map[MaxPooling2D]['fill'] = 'red'
color_map[Dense]['fill'] = 'green'
color_map[Flatten]['fill'] = 'teal'

model.summary()
visualkeras.layered_view(model, color_map=color_map, to_file='output.png').show()