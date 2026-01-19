"""
分析 ae_mlp_cv 保存的模型与 run_ae_mlp_hyperopt_cv.py 中定义的模型是否有差异

比较内容:
1. 模型架构 (层结构、层名称、参数数量)
2. 层配置 (units, activation, dropout 等)
3. 输入输出形状
4. 编译配置 (optimizer, loss, loss_weights)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# ============================================================================
# 从 run_ae_mlp_hyperopt_cv.py 复制的模型构建函数
# ============================================================================

def build_ae_mlp_model_from_script(params: dict) -> Model:
    """原脚本中的 build_ae_mlp_model 函数 (完全复制)"""
    num_columns = params['num_columns']
    hidden_units = params['hidden_units']
    dropout_rates = params['dropout_rates']
    lr = params['lr']
    loss_weights = params['loss_weights']

    inp = layers.Input(shape=(num_columns,), name='input')

    # 输入标准化
    x0 = layers.BatchNormalization(name='input_bn')(inp)

    # Encoder
    encoder = layers.GaussianNoise(dropout_rates[0], name='noise')(x0)
    encoder = layers.Dense(hidden_units[0], name='encoder_dense')(encoder)
    encoder = layers.BatchNormalization(name='encoder_bn')(encoder)
    encoder = layers.Activation('swish', name='encoder_act')(encoder)

    # Decoder (重建原始输入)
    decoder = layers.Dropout(dropout_rates[1], name='decoder_dropout')(encoder)
    decoder = layers.Dense(num_columns, dtype='float32', name='decoder')(decoder)

    # 辅助预测分支 (基于 decoder 输出)
    x_ae = layers.Dense(hidden_units[1], name='ae_dense1')(decoder)
    x_ae = layers.BatchNormalization(name='ae_bn1')(x_ae)
    x_ae = layers.Activation('swish', name='ae_act1')(x_ae)
    x_ae = layers.Dropout(dropout_rates[2], name='ae_dropout1')(x_ae)
    out_ae = layers.Dense(1, dtype='float32', name='ae_action')(x_ae)

    # 主分支: 原始特征 + encoder 特征
    x = layers.Concatenate(name='concat')([x0, encoder])
    x = layers.BatchNormalization(name='main_bn0')(x)
    x = layers.Dropout(dropout_rates[3], name='main_dropout0')(x)

    # MLP 主体
    for i in range(2, len(hidden_units)):
        dropout_idx = min(i + 2, len(dropout_rates) - 1)
        x = layers.Dense(hidden_units[i], name=f'main_dense{i-1}')(x)
        x = layers.BatchNormalization(name=f'main_bn{i-1}')(x)
        x = layers.Activation('swish', name=f'main_act{i-1}')(x)
        x = layers.Dropout(dropout_rates[dropout_idx], name=f'main_dropout{i-1}')(x)

    # 主输出 (使用 float32 确保数值稳定性)
    out = layers.Dense(1, dtype='float32', name='action')(x)

    model = Model(inputs=inp, outputs=[decoder, out_ae, out], name='AE_MLP')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss={
            'decoder': 'mse',
            'ae_action': 'mse',
            'action': 'mse',
        },
        loss_weights=loss_weights,
    )

    return model


def get_layer_config(layer):
    """提取层的关键配置"""
    config = layer.get_config()
    result = {
        'class': layer.__class__.__name__,
        'name': layer.name,
    }

    # 提取关键参数
    if 'units' in config:
        result['units'] = config['units']
    if 'rate' in config:
        result['rate'] = config['rate']
    if 'stddev' in config:
        result['stddev'] = config['stddev']
    if 'activation' in config:
        result['activation'] = config['activation']
    if 'axis' in config:
        result['axis'] = config['axis']
    if 'dtype' in config and config['dtype'] != 'float32':
        result['dtype'] = config['dtype']

    return result


def compare_models(model1, model2, name1="Model 1", name2="Model 2"):
    """比较两个模型的详细差异"""
    print("\n" + "=" * 80)
    print(f"模型对比: {name1} vs {name2}")
    print("=" * 80)

    differences = []

    # 1. 基本信息
    print("\n[1] 基本信息")
    print("-" * 40)
    print(f"{'属性':<25} {name1:<25} {name2:<25}")
    print("-" * 40)

    info_items = [
        ("模型名称", model1.name, model2.name),
        ("总参数量", f"{model1.count_params():,}", f"{model2.count_params():,}"),
        ("可训练参数", f"{sum(p.numpy().size for p in model1.trainable_weights):,}",
                     f"{sum(p.numpy().size for p in model2.trainable_weights):,}"),
        ("层数量", len(model1.layers), len(model2.layers)),
        ("输入形状", str(model1.input_shape), str(model2.input_shape)),
        ("输出数量", len(model1.outputs) if isinstance(model1.outputs, list) else 1,
                    len(model2.outputs) if isinstance(model2.outputs, list) else 1),
    ]

    for name, v1, v2 in info_items:
        match = "✓" if str(v1) == str(v2) else "✗"
        print(f"{name:<25} {str(v1):<25} {str(v2):<25} {match}")
        if str(v1) != str(v2):
            differences.append(f"基本信息 - {name}: {v1} vs {v2}")

    # 2. 层结构对比
    print("\n[2] 层结构详细对比")
    print("-" * 100)
    print(f"{'序号':<5} {'层名称':<20} {name1:<35} {name2:<35} {'匹配':<5}")
    print("-" * 100)

    max_layers = max(len(model1.layers), len(model2.layers))

    for i in range(max_layers):
        if i < len(model1.layers) and i < len(model2.layers):
            l1 = model1.layers[i]
            l2 = model2.layers[i]

            c1 = get_layer_config(l1)
            c2 = get_layer_config(l2)

            # 简化显示
            s1 = f"{c1['class']}"
            if 'units' in c1:
                s1 += f"({c1['units']})"
            elif 'rate' in c1:
                s1 += f"(rate={c1['rate']:.4f})"
            elif 'stddev' in c1:
                s1 += f"(std={c1['stddev']:.4f})"

            s2 = f"{c2['class']}"
            if 'units' in c2:
                s2 += f"({c2['units']})"
            elif 'rate' in c2:
                s2 += f"(rate={c2['rate']:.4f})"
            elif 'stddev' in c2:
                s2 += f"(std={c2['stddev']:.4f})"

            # 比较 (忽略数值上的微小差异)
            match = "✓"
            if c1['class'] != c2['class']:
                match = "✗"
                differences.append(f"层{i} 类型不同: {c1['class']} vs {c2['class']}")
            elif c1['name'] != c2['name']:
                match = "~"  # 名称不同但类型相同
            if 'units' in c1 and 'units' in c2 and c1['units'] != c2['units']:
                match = "✗"
                differences.append(f"层{i} units不同: {c1['units']} vs {c2['units']}")

            print(f"{i:<5} {l1.name:<20} {s1:<35} {s2:<35} {match:<5}")
        elif i < len(model1.layers):
            l1 = model1.layers[i]
            print(f"{i:<5} {l1.name:<20} {l1.__class__.__name__:<35} {'(不存在)':<35} ✗")
            differences.append(f"层{i} 仅存在于 {name1}")
        else:
            l2 = model2.layers[i]
            print(f"{i:<5} {l2.name:<20} {'(不存在)':<35} {l2.__class__.__name__:<35} ✗")
            differences.append(f"层{i} 仅存在于 {name2}")

    # 3. 输出层对比
    print("\n[3] 输出层对比")
    print("-" * 60)

    outputs1 = model1.outputs if isinstance(model1.outputs, list) else [model1.outputs]
    outputs2 = model2.outputs if isinstance(model2.outputs, list) else [model2.outputs]

    for i, (o1, o2) in enumerate(zip(outputs1, outputs2)):
        # 获取产生该输出的层名称
        name1_out = o1.node.layer.name if hasattr(o1, 'node') else str(o1)
        name2_out = o2.node.layer.name if hasattr(o2, 'node') else str(o2)
        print(f"输出{i}: {name1} -> {name1_out}, {name2} -> {name2_out}")

    # 4. 编译配置对比 (如果有)
    print("\n[4] 编译配置")
    print("-" * 60)

    if model1.optimizer and model2.optimizer:
        opt1 = model1.optimizer.__class__.__name__
        opt2 = model2.optimizer.__class__.__name__
        lr1 = float(model1.optimizer.learning_rate)
        lr2 = float(model2.optimizer.learning_rate)
        print(f"Optimizer: {opt1} (lr={lr1:.6f}) vs {opt2} (lr={lr2:.6f})")

        if opt1 != opt2:
            differences.append(f"Optimizer不同: {opt1} vs {opt2}")
    else:
        print("模型未编译或无法获取optimizer信息")

    # 如果有 loss_weights
    if hasattr(model1, 'loss_weights') and model1.loss_weights:
        print(f"Loss weights ({name1}): {model1.loss_weights}")
    if hasattr(model2, 'loss_weights') and model2.loss_weights:
        print(f"Loss weights ({name2}): {model2.loss_weights}")

    # 5. 汇总
    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)

    if differences:
        print(f"\n发现 {len(differences)} 处差异:")
        for i, diff in enumerate(differences, 1):
            print(f"  {i}. {diff}")
    else:
        print("\n✓ 两个模型架构完全一致!")

    return differences


def print_model_summary(model, name="Model"):
    """打印模型详细信息"""
    print(f"\n{'='*80}")
    print(f"{name} 详细信息")
    print("="*80)

    print("\n层结构:")
    for i, layer in enumerate(model.layers):
        config = get_layer_config(layer)
        output_shape = layer.output_shape if hasattr(layer, 'output_shape') else "N/A"
        params = layer.count_params()
        print(f"  [{i:2d}] {layer.name:<25} {config['class']:<20} output={str(output_shape):<30} params={params:>8,}")

    print(f"\n输入形状: {model.input_shape}")
    print(f"输出形状: {model.output_shape}")
    print(f"总参数: {model.count_params():,}")


def main():
    # 模型路径
    saved_model_path = "/home/kownse/code/qlib-proj/my_models/ae_mlp_cv_alpha158-enhanced-v7_sp500_5d_best.keras"

    print("=" * 80)
    print("AE-MLP 模型对比分析")
    print("=" * 80)
    print(f"保存的模型: {saved_model_path}")
    print(f"参考脚本: run_ae_mlp_hyperopt_cv.py 中的 build_ae_mlp_model()")

    # 1. 加载保存的模型
    print("\n[*] 加载保存的模型...")
    saved_model = keras.models.load_model(saved_model_path)
    print(f"    ✓ 加载成功")

    # 打印保存模型的详细信息
    print_model_summary(saved_model, "保存的模型")

    # 2. 从保存模型推断参数，然后用脚本创建新模型
    print("\n[*] 从保存模型推断参数...")

    # 推断 num_columns (从输入层)
    num_columns = saved_model.input_shape[1]
    print(f"    num_columns: {num_columns}")

    # 推断 hidden_units (从各 Dense 层)
    # 根据 build_ae_mlp_model 的结构:
    # hidden_units[0] = encoder_dim -> encoder_dense
    # hidden_units[1] = decoder_hidden -> ae_dense1
    # hidden_units[2] = main_layer1 -> main_dense1
    # hidden_units[3] = main_layer2 -> main_dense2
    # hidden_units[4] = main_layer3 -> main_dense3

    layer_units = {}
    dropout_rates = []

    for layer in saved_model.layers:
        config = layer.get_config()
        name = layer.name

        # 收集所有 Dense 层的 units
        if name == 'encoder_dense':
            layer_units['encoder_dim'] = config['units']
            print(f"    encoder_dim: {config['units']}")
        elif name == 'ae_dense1':
            layer_units['decoder_hidden'] = config['units']
            print(f"    decoder_hidden: {config['units']}")
        elif name == 'main_dense1':
            layer_units['main_layer1'] = config['units']
            print(f"    main_layer1: {config['units']}")
        elif name == 'main_dense2':
            layer_units['main_layer2'] = config['units']
            print(f"    main_layer2: {config['units']}")
        elif name == 'main_dense3':
            layer_units['main_layer3'] = config['units']
            print(f"    main_layer3: {config['units']}")

        # dropout rates
        elif name == 'noise':
            if 'stddev' in config:
                dropout_rates.append(config['stddev'])
                print(f"    noise_std: {config['stddev']:.6f}")
        elif 'dropout' in name.lower():
            if 'rate' in config:
                dropout_rates.append(config['rate'])

    # 按正确顺序组装 hidden_units
    hidden_units = [
        layer_units.get('encoder_dim', 64),
        layer_units.get('decoder_hidden', 64),
        layer_units.get('main_layer1', 256),
        layer_units.get('main_layer2', 128),
        layer_units.get('main_layer3', 64),
    ]

    # 确保 dropout_rates 有足够的元素
    while len(dropout_rates) < 7:
        if dropout_rates:
            dropout_rates.append(dropout_rates[-1])
        else:
            dropout_rates.append(0.1)

    print(f"\n    推断的 hidden_units (正确顺序): {hidden_units}")
    print(f"    [encoder_dim, decoder_hidden, main_layer1, main_layer2, main_layer3]")
    print(f"    推断的 dropout_rates: {[f'{r:.4f}' for r in dropout_rates]}")

    # 从 optimizer 获取 learning rate
    lr = 1e-3  # 默认值
    if saved_model.optimizer:
        lr = float(saved_model.optimizer.learning_rate)
    print(f"    learning_rate: {lr:.6f}")

    # 获取 loss_weights
    loss_weights = {'decoder': 0.1, 'ae_action': 0.1, 'action': 1.0}  # 默认值
    if hasattr(saved_model, 'loss_weights') and saved_model.loss_weights:
        loss_weights = dict(saved_model.loss_weights)
    print(f"    loss_weights: {loss_weights}")

    # 3. 使用推断的参数构建新模型
    print("\n[*] 使用推断参数构建参考模型...")

    params = {
        'num_columns': num_columns,
        'hidden_units': hidden_units,
        'dropout_rates': dropout_rates,
        'lr': lr,
        'loss_weights': loss_weights,
    }

    reference_model = build_ae_mlp_model_from_script(params)
    print(f"    ✓ 构建成功")

    # 打印参考模型的详细信息
    print_model_summary(reference_model, "参考模型 (脚本构建)")

    # 4. 对比两个模型
    differences = compare_models(
        saved_model, reference_model,
        "保存的模型", "脚本构建的模型"
    )

    # 5. 额外检查: 对比层的输出名称
    print("\n[5] 输出层名称对比")
    print("-" * 60)

    saved_outputs = saved_model.output_names if hasattr(saved_model, 'output_names') else []
    ref_outputs = reference_model.output_names if hasattr(reference_model, 'output_names') else []

    print(f"保存模型输出: {saved_outputs}")
    print(f"参考模型输出: {ref_outputs}")

    if saved_outputs == ref_outputs:
        print("✓ 输出名称一致")
    else:
        print("✗ 输出名称不一致")

    # 6. 检查是否有额外的自定义层或配置
    print("\n[6] 检查特殊配置")
    print("-" * 60)

    # 检查是否使用了混合精度
    for layer in saved_model.layers:
        dtype_policy = layer.dtype_policy if hasattr(layer, 'dtype_policy') else None
        if dtype_policy and 'mixed' in str(dtype_policy):
            print(f"层 {layer.name} 使用混合精度: {dtype_policy}")

    # 最终结论
    print("\n" + "=" * 80)
    print("最终结论")
    print("=" * 80)

    if not differences:
        print("""
✓ 模型架构完全一致

保存的模型和 run_ae_mlp_hyperopt_cv.py 中 build_ae_mlp_model()
函数创建的模型具有相同的架构。

两者都遵循 AE-MLP 结构:
  - Input -> BatchNorm -> GaussianNoise
  - Encoder: Dense -> BatchNorm -> Swish
  - Decoder: Dropout -> Dense (重建输入)
  - AE分支: Dense -> BatchNorm -> Swish -> Dropout -> Dense(1)
  - Main分支: Concat(input, encoder) -> BatchNorm -> Dropout -> Dense×3 -> Dense(1)
  - 输出: [decoder, ae_action, action]
""")
    else:
        print(f"""
✗ 发现 {len(differences)} 处差异

差异列表:
""")
        for diff in differences:
            print(f"  - {diff}")

        print("""
可能原因:
  1. 保存模型时使用了不同的超参数
  2. 代码在保存后被修改
  3. 使用了不同版本的 TensorFlow/Keras
""")


if __name__ == "__main__":
    main()
