"""Rollout：从初始状态出发，按给定控制序列逐步预测状态。

运行：uv run python example/01_rollout/main.py
本例仅进行前向模拟；寻找更好的控制序列见 ../02_single_shooting/main.py。
"""

import numpy as np

DT = 1.0  # 每步持续时间，单位：秒
X_INITIAL = 0.0  # 初始位置，单位：米


def rollout(x_initial, controls, dt):
    """输入初始位置、速度序列和时间步长，返回包含初始位置的轨迹。

    controls[k] 是第 k 个时间区间内的速度，单位：米/秒。
    N 个控制对应 N 次状态更新，因此返回 N + 1 个位置。
    """
    states = np.empty(len(controls) + 1)
    states[0] = x_initial

    for k, control in enumerate(controls):
        # 下一位置 = 当前位置 + 本步持续时间 × 本步速度。
        states[k + 1] = states[k] + dt * control

    return states


def main():
    np.set_printoptions(precision=6, suppress=True)

    # 两组控制都是事先给定的，并非由此示例优化得到。
    control_sequences = (
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 2.0, 2.0]),
    )

    for controls in control_sequences:
        states = rollout(X_INITIAL, controls, DT)
        print(f"\n给定控制（m/s）：{controls}")
        for k, control in enumerate(controls):
            print(
                f"  t={k * DT:g} → {(k + 1) * DT:g} s："
                f"x[{k + 1}] = {states[k]:g} + {DT:g} × {control:g}"
                f" = {states[k + 1]:g} m"
            )
        print("预测轨迹（m）：", states)
        # 只评价给定控制，方便衔接下一个优化示例。
        print(f"控制平方和：{controls @ controls:g}")

    print("\n两组控制终点相同，但中间轨迹和控制平方和不同。")
    print("Rollout 回答：给定这些控制，会走出什么轨迹？")
    print("下一个示例回答：怎样自动找到满足终点约束且代价最小的控制？")


if __name__ == "__main__":
    main()
