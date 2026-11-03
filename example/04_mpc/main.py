"""一维 MPC：滚动预测、只执行首项、移位热启动，并在受扰后重新规划。"""

import numpy as np
from scipy.optimize import minimize

# 本例每个时刻只有一个状态（位置）和一个控制（速度）。
# N 是预测控制的步数，不是状态的维数。
#   是每一轮向前预测的步数，也叫“预测时域长度”，不是车辆整个运动过程的总步数。
# NumPy 中 shape=(N,) 表示一维数组；shape=(N, N) 表示二维矩阵。
# 本例 controls.shape=(8,)，包含初始位置的预测轨迹 shape=(9,)。
N = 8
DT = 0.2
GOAL = 6.0
MAX_SPEED = 2.0
CONTROL_WEIGHT = 0.2


def rollout(initial, controls):
    # np.cumsum：沿数组累加，返回每个位置的累计和，而不是一个总和。
    # 例如 controls=np.array([1.0, 2.0, 3.0])：
    #   np.cumsum(controls) -> [1.0, 3.0, 6.0]
    #   np.sum(controls)    -> 6.0
    # 一维模型 x[k+1] = x[k] + DT * controls[k] 可以展开为：
    #   x[k+1] = initial + DT * sum(controls[0:k+1])。
    # 因而一次 cumsum 就能算出全部未来位置，不必显式编写状态更新循环。
    # 这种写法依赖本例的简单积分模型，不能直接用于朝向变化的差速模型。
    #
    # DT * 数组、initial + 数组使用“标量广播”：
    # 对数组的每个元素分别乘 DT、加 initial，结果仍是 shape=(N,)。
    # 例如 initial=1.0、DT=0.2、controls=[1.0, 2.0, 3.0]：
    #   initial + DT * np.cumsum(controls) -> [1.2, 1.6, 2.2]
    #
    # np.concatenate 把同维数的数组沿已有轴拼接；这里默认沿 axis=0。
    # [initial] 会被视为长度为 1 的一维数组，拼到未来位置数组的前面。
    # 注意不能直接用标量 initial 代替 [initial]：标量没有可供拼接的轴。
    # 上例最终返回 [1.0, 1.2, 1.6, 2.2]，即“初始位置 + 未来位置”。
    # 本例 N=8，所以返回长度为 N+1=9 的一维数组。
    return np.concatenate(([initial], initial + DT * np.cumsum(controls)))


def objective(controls, initial):
    # [1:] 是切片：从索引 1 开始取到末尾，跳过固定的初始位置。
    # 如 [x0, x1, x2, x3][1:] -> [x1, x2, x3]。
    # 减去标量 GOAL 时广播到每个元素，得到每个未来位置相对目标的误差。
    # errors.shape=(N,)，errors[k] 对应 x[k+1]-GOAL，而不是 x[k]-GOAL。
    # 根据控制和初始值,计算出后续的状态
    errors = rollout(initial, controls)[1:] - GOAL

    # np.ones(N) 创建长度为 N、每个元素都是 1.0 的一维浮点数组。
    # 与 np.ones((N, N)) 不同，后者创建 N 行 N 列的二维矩阵。
    weights = np.ones(N)
    # 负索引 -1 表示最后一个元素；这里原地修改末端误差的权重。
    # N=8 时，weights 变成 [1, 1, 1, 1, 1, 1, 1, 5]。
    # 最后一个预测位置的误差比其它时刻受到更强的惩罚。
    weights[-1] = 5.0

    # errors**2：逐元素平方，仍是 shape=(N,)。
    # weights * errors**2：两个同形数组逐元素相乘，也仍是 shape=(N,)。
    # np.sum(...)：把所有元素加起来，得到一个标量位置代价。
    # 例如 errors=[2, -1]、weights=[1, 5]，则位置代价为 1*4+5*1=9。
    #
    # @ 的含义由数组维数决定：两个一维数组使用 @ 时计算内积。
    # controls @ controls = u0*u0 + u1*u1 + ...，也是一个标量。
    # 它在数学上等于 np.sum(controls**2)，用于惩罚控制量过大。
    # 注意 * 和 @ 不同：设 a=np.array([2, 3])、b=np.array([4, 5])，
    # a*b -> [8, 15]，而 a@b -> 23。
    cost = np.sum(weights * errors**2) + CONTROL_WEIGHT * (controls @ controls)

    # np.ones((N, N)) 创建一个 N×N 的全 1 矩阵。
    # np.tril(...) 保留下三角（包括主对角线），把上三角元素置零。
    # 用 N=3 举例：
    #   np.tril(np.ones((3, 3))) = [[1, 0, 0],
    #                              [1, 1, 0],
    #                              [1, 1, 1]]
    # 再乘标量 DT，就是让每个元素乘 DT。
    #
    # sensitivity 是状态对控制的 Jacobian，不是状态轨迹本身。
    # 本例没有存放初始状态的那一行，因此：
    #   sensitivity[k, j] = ∂x[k+1]/∂controls[j]。
    #   当 j <= k 时为 DT，否则为 0。
    # 例如 x2=initial+DT*(u0+u1)，所以对应第二行为 [DT, DT, 0]。
    # 未来的控制不能影响过去的位置，因此上三角为零。

    # 今天自己对于这个的计算，其实还是没有太理解到，这里只是一维的，自己还是以昨天两轮差速的计算过程来理解的。。。。所以出现了很多的问题！
    # 是因为 sensitivity 的每一列对应一个控制量，自己理解成一行对应于一个控制量了。。。
    sensitivity = DT * np.tril(np.ones((N, N)))

    # .T 对二维数组转置：交换行和列。
    # sensitivity.T[j, k] 就是原来的 sensitivity[k, j]。
    # 虽然这里转置前后形状都是 (N, N)，但元素的位置发生了改变。
    # 注意一维数组没有独立的行/列轴，controls.T 的形状仍是 (N,)。
    #
    # weights * errors 是逐元素乘积，形状为 (N,)。
    # 二维矩阵 @ 一维数组是矩阵-向量乘法：(N, N) @ (N,) -> (N,)。
    # 所以第 j 个结果为：
    #   sum_k sensitivity[k, j] * weights[k] * errors[k]。
    # 它沿预测时刻 k 求和，保留控制索引 j；乘 2 来自误差平方的求导。
    # 这就是链式法则：∂位置代价/∂u_j
    #   = sum_k 2*weights[k]*errors[k]*∂x[k+1]/∂u_j。
    #
    # 例如 N=3、DT=0.2、weights=[1, 1, 5]、errors=[-3, -2, -1]：
    #   weights*errors = [-3, -2, -5]
    #   sensitivity.T @ (weights*errors) = [-2.0, -1.4, -1.0]
    # 第一个控制会影响三个预测位置，所以其导数包含三个时刻的贡献。
    #
    # 最后一项 2*CONTROL_WEIGHT*controls 是控制平方惩罚的梯度。
    # 两个 shape=(N,) 的梯度逐元素相加，得到总代价对整段控制的梯度。
    gradient = 2 * sensitivity.T @ (weights * errors) + 2 * CONTROL_WEIGHT * controls
    # cost 通常是 NumPy 浮点标量；float(cost) 转成 Python 内置 float。
    # gradient 保持为 shape=(N,) 的 NumPy 数组，供优化器使用。
    return float(cost), gradient


# 这里自己对于 mpc 有个误解，其实 mpc 找的是本轮最优，而非全局最优
# 本轮最优：从当前位置出发，未来 N 步的总代价最小。
# 全程最优：从最初起点到最终终点，整段过程的总代价最小。
def main():
    state = 0.0
    # np.zeros(N) 创建长度为 N 的全 0 浮点数组，形状为 (N,)。
    # 这里是第一轮求解的“速度序列初始猜测”，不是车辆状态数组。
    guess = np.zeros(N)
    disturbed = False
    for cycle in range(80):
        # minimize 来自 SciPy。它把 guess 作为一维决策向量传给 objective。
        # args=(state,) 是 Python 单元素元组：额外传入当前状态 initial。
        # jac=True 表示 objective 同时返回 (代价值, 梯度数组)。
        # bounds 后的列表 * N 是 Python 列表重复，不是 NumPy 逐元素乘法。
        result = minimize(
            objective,
            guess,
            args=(state,),
            jac=True,
            method="SLSQP",
            bounds=[(-MAX_SPEED, MAX_SPEED)] * N,
            options={"ftol": 1e-9, "maxiter": 100},
        )
        # result.x 是求出的控制序列，NumPy 数组，形状为 (N,)
        # np.isfinite 逐元素检查是否为有限数，返回同形状的布尔数组
        # 例如 np.isfinite([1.0, np.nan, np.inf]) -> [True, False, False]
        # np.all 将这些布尔值归约为一个值：所有元素都是 True 才返回 True
        # 因此下面同时要求“求解成功”和“全部控制值有限”
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"第 {cycle} 轮求解失败：{result.message}")
        # np.abs 对每个控制值取绝对值，np.max 再取其中最大的值
        # 例如 [-1.5, 0.2, 2.1] -> [1.5, 0.2, 2.1] -> 2.1
        # 因为速度上下限对称，检查绝对值的最大值即可同时检查正负方向
        if np.max(np.abs(result.x)) > MAX_SPEED + 1e-7:
            raise RuntimeError("控制超过速度限制。")

        # result.x 就是控制量，这个案例中只有一个速度
        prediction = rollout(state, result.x)
        # 整数索引 [0] 取出首个元素，得到标量速度，而不是长度为 1 的数组
        command = result.x[0]  # 只执行第一项，后面是计划，不一次性执行
        state += DT * command
        if cycle == 6:
            state -= 0.8  # 模拟未包含在预测模型内的外部位置扰动
            disturbed = True
            print("施加 -0.8 m 位置扰动；下一轮使用新的观测位置重新求解。")
        # prediction[0] 是本轮起始位置，prediction[1] 才是预测的下一位置
        print(
            f"轮次 {cycle:02d}：执行速度={command:.4f}，实际位置={state:.4f}，"
            f"本轮原预测下一位置={prediction[1]:.4f}"
        )
        # 丢掉已执行的首项，末尾重复最后一项，作为下一轮初始猜测
        # result.x[1:] 取第 2 个到最后一个控制，形状为 (N-1,)
        # result.x[-1:] 只取最后一个控制，但切片仍保留一维形状 (1,)
        # 与 result.x[-1] 不同：后者是标量，不能直接与一维数组 concatenate
        # 两段数组拼接后重新得到 shape=(N,) 的新数组，不修改 result.x
        # 例如 [1, 2, 3, 4] -> 拼接 [2, 3, 4] 和 [4] -> [2, 3, 4, 4]
        guess = np.concatenate((result.x[1:], result.x[-1:]))
        # 这里 state-GOAL 和 command 都是标量，所以直接使用内置 abs
        if disturbed and abs(state - GOAL) < 0.01 and abs(command) < 0.05:
            print(f"到达目标并接近静止，位置误差={abs(state - GOAL):.6f} m。")
            return
    raise RuntimeError("在规定轮数内未收敛。")


if __name__ == "__main__":
    main()
