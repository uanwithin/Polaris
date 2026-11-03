"""差速模型的状态灵敏度传播，以及用中心差分独立检查解析导数。"""

import numpy as np

DT = 0.2
INITIAL = np.array([0.0, 0.0, 0.3])


# states(5, 3)初始状态，加上执行 4 步控制后的状态
# sensitivity(5, 3, 8)每个时刻的每个状态分量，对 8 个控制分量的导数
def rollout(controls):
    """controls.shape=(N, 2)，列为 v、omega；S.shape=(N+1, 3, 2N)。"""
    n = len(controls)
    states = np.zeros((n + 1, 3))
    # sensitivity adj. 敏感的，灵敏度
    sensitivity = np.zeros((n + 1, 3, 2 * n))
    states[0] = INITIAL
    for k, (v, omega) in enumerate(controls):
        x, y, yaw = states[k]
        c, s = np.cos(yaw), np.sin(yaw)
        # 计算下一时刻的状态
        states[k + 1] = [x + DT * v * c, y + DT * v * s, yaw + DT * omega]
        # A = df/dx，B = df/du；链式法则传播过去控制的影响。
        a = np.array([[1, 0, -DT * v * s], [0, 1, DT * v * c], [0, 0, 1]])
        b = np.array([[DT * c, 0], [DT * s, 0], [0, DT]])
        # a @ sensitivity[k]：将控制对当前状态的影响，继续传递到下一状态
        sensitivity[k + 1] = a @ sensitivity[k]
        # 当前控制直接影响下一状态，对应本步的两列。
        # 这里是 E_k
        sensitivity[k + 1, :, 2 * k : 2 * k + 2] += b
    return states, sensitivity


# 把状态的导数转换成总代价的梯度，返回的是总代价对每个控制的梯度！！！
# 这里是整个项目的核心，将优化对象完整的转换到每个控制量了，后面只需要去优化这些控制量即可
"""
value：当前这整段控制的总代价
gradient：
[
    ∂J/∂v0,
    ∂J/∂omega0,
    ∂J/∂v1,
    ∂J/∂omega1,
    ...
]
"""


def objective(flat_controls):
    # 这里 -1 表示让 NumPy 自动计算行数。数据的含义没有改变，只是换了组织方式
    controls = flat_controls.reshape(-1, 2)
    # states[k] 机器人在第 k 步的朝向，sensitivity[k] 第 k 步状态对整段控制序列的导数
    # 重点是描述了两件事情： 车辆会走到哪里，以及这些位置对控制变化有多敏感
    states, sensitivity = rollout(controls)
    # 简单位置跟踪，不引入角度 wrap 的不可微边界。
    target = np.array([1.0, 0.5])
    # 计算位置误差
    # 取第 1 到第 \(N\) 个状态，跳过初始状态
    # 只取 x、y，不取朝向
    # 这个教学示例中，每个预测位置都与同一个目标点比较。 它没有为每个时刻设置不同的参考路径点，也不只考察最终位置
    errors = states[1:, :2] - target
    # 计算总代价： 位置误差                 控制代价
    value = np.sum(errors**2) + 0.1 * np.sum(controls**2)
    # 如何从状态的导数得到总代价的梯度
    #
    gradient = 2 * np.einsum("ki,kij->j", errors, sensitivity[1:, :2, :])
    gradient += 0.2 * flat_controls
    return float(value), gradient


"""
einsum 的平替
gradient = np.zeros(flat_controls.size)

for k in range(len(controls)):
    # 一趟 x,一趟 y
    for i in range(2):
        for j in range(flat_controls.size):
            gradient[j] += (
                2
                * errors[k, i]
                * sensitivity[k + 1, i, j]
            )
"""


# 它通过实际扰动控制、重新计算函数值，数值估计导数，用来核对解析梯度是否正确。
def central_difference(function, controls, epsilon):
    """逐个扰动控制，输出形状为 function 输出形状 + (控制维数,)。"""
    columns = []
    for j in range(controls.size):
        delta = np.zeros_like(controls)
        delta[j] = epsilon
        # 这里不能理解成只计算成一个,实际上是向量计算
        columns.append(
            (function(controls + delta) - function(controls - delta)) / (2 * epsilon)
        )
    return np.stack(columns, axis=-1)


def main():
    # 包含转弯和非零朝向，避免只检查直行这一特殊情况。
    controls = np.array([[0.8, 0.2], [1.0, -0.3], [0.6, 0.4], [0.9, -0.1]])
    flat = controls.ravel()  # [v0, omega0, v1, omega1, ...]
    # sensitivity 就是状态 jacobian
    states, sensitivity = rollout(controls)
    # gradient 就是通过链式法则计算的代价梯度
    _, gradient = objective(flat)
    print("预测轨迹：\n", states)
    print("末端状态对控制的雅可比：\n", sensitivity[-1])
    for epsilon in (1e-3, 1e-5, 1e-7):
        """
        # 这个 lambda 相当于定义了一个函数
        # 返回的是状态
        def states_from_controls(u):
          controls = u.reshape(-1, 2)
          states, sensitivity = rollout(controls)
          return states
        """
        # 在 flat 这组控制附近，数值计算整段状态轨迹对控制的导数。
        numerical_s = central_difference(
            lambda u: rollout(u.reshape(-1, 2))[0], flat, epsilon
        )
        """
        # 下面这个 lambad 相当于定义了这样一个函数
        def cost_from_controls(u):
            value, gradient = objective(u)
            # 这次只取代价值
            return value
        """
        # 计算数值梯度
        numerical_g = central_difference(lambda u: objective(u)[0], flat, epsilon)
        print(
            f"epsilon={epsilon:g}：状态导数最大误差={np.max(np.abs(sensitivity - numerical_s)):.3e}，"
            f"代价梯度最大误差={np.max(np.abs(gradient - numerical_g)):.3e}"
        )
        np.testing.assert_allclose(sensitivity, numerical_s, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(gradient, numerical_g, atol=1e-6, rtol=1e-5)
    print("所有梯度检查通过。")


if __name__ == "__main__":
    main()
