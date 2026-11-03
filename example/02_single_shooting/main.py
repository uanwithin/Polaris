"""Single shooting：只优化控制 U，通过 rollout 得到状态 X。

最小化 Σ u[k]²，满足 x[k+1] = x[k] + DT * u[k]、x[0] = 0、x[N] = 6。
建议先运行 ../01_rollout/main.py；概念说明见 ../README.md。
"""

import numpy as np
from scipy.optimize import minimize

N = 3  # 控制步数
DT = 1.0  # 每步时间，单位：秒
X_INITIAL = 0.0  # 初始位置
X_GOAL = 6.0  # 目标位置


# rollout 不寻找控制，只计算给定控制会产生什么轨迹
# 前向模拟
# 这一步正是 single shooting 的核心：把状态通过运动模型算出来，最终只把控制量交给优化器。
#                                 状态用控制量来表示
def rollout(U):
    """给定控制序列 U，计算轨迹 X 和状态雅可比 S。"""
    X = np.zeros(N + 1)  # N 次状态更新，加上初始状态，共 N + 1 个状态。

    # 本例状态是标量，所以每个 S_h 是 1×N 的行向量。
    # S[h, j] = ∂x_h / ∂U_j
    # 即第 j 个控制改变一点，第 h 个状态会改变多少。
    # 第一次阅读可先看 X 的更新；S 是为优化器额外计算的导数。
    S = np.zeros((N + 1, N))

    X[0] = X_INITIAL

    # range 左闭右开
    for h in range(N):
        # 运动模型：x_{h+1} = x_h + DT * u_h
        X[h + 1] = X[h] + DT * U[h]

        # 雅可比递推：S_{h+1} = A_h S_h + B_h E_h
        # 本例 A_h = 1，B_h = DT。
        S[h + 1] = S[h]
        # 本来是 s[h + 1] = DT * E_h 的，但是 E_h 只有一项，所以这里直接就 + DT 了
        S[h + 1, h] += DT

    return X, S


# 控制能量最小
# 其实也可以理解成，惩罚速度过大
def objective(U):
    """目标函数：J(U) = u_0² + u_1² + u_2²。"""
    return float(U @ U)


def objective_gradient(U):
    """目标函数梯度：∇J(U) = 2U。"""
    return 2.0 * U


def terminal_constraint(U):
    """终点等式约束：c(U) = x_N(U) - X_GOAL = 0。"""
    X, _ = rollout(U)
    return X[-1] - X_GOAL


def terminal_constraint_gradient(U):
    """终点约束梯度：∂c/∂U = ∂x_N/∂U = S_N。"""
    _, S = rollout(U)
    return S[-1]


def main():
    np.set_printoptions(precision=6, suppress=True)

    # 优化器的初始猜测，不是最终答案。
    U_initial = np.array([1.0, 2.0, 3.0])

    result = minimize(
        fun=objective,
        x0=U_initial,
        jac=objective_gradient,
        method="SLSQP",
        constraints={
            "type": "eq",  # 等式约束：要求 fun(U) = 0
            "fun": terminal_constraint,
            "jac": terminal_constraint_gradient,
        },
        options={"ftol": 1e-10, "maxiter": 100},
    )

    if not result.success:
        raise RuntimeError(f"优化失败：{result.message}")

    if abs(terminal_constraint(result.x)) > 1e-8:
        raise RuntimeError("终点约束未满足。")

    U_optimal = result.x
    X_optimal, S = rollout(U_optimal)

    print("初始控制：", U_initial)
    print("初始代价：", objective(U_initial))
    print("最优控制：", U_optimal)
    print("最优轨迹：", X_optimal)
    print("最优代价：", result.fun)
    print(f"终点误差：{terminal_constraint(U_optimal):.3e}")
    print("状态雅可比（每行对应一个时刻的 S_h）：")
    print(S)


if __name__ == "__main__":
    main()
