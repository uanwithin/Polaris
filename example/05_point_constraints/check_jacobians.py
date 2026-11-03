"""在可微位置用中心差分独立核对解析雅可比；直接运行本文件即可。"""

import numpy as np

import main as demo


def central_difference(function, values, epsilon=1e-6):
    columns = []
    for j in range(values.size):
        delta = np.zeros_like(values)
        delta[j] = epsilon
        columns.append((function(values + delta) - function(values - delta)) / (2 * epsilon))
    return np.stack(columns, axis=-1)


def check(name, analytic, numerical):
    assert analytic.shape == numerical.shape, (analytic.shape, numerical.shape)
    np.testing.assert_allclose(analytic, numerical, atol=2e-7, rtol=2e-6)
    print(f"{name}：最大绝对误差 {np.max(np.abs(analytic - numerical)):.3e}")


def main():
    # 构造不同象限、边外、角外和内部的局部坐标，并转换成世界系中心。
    # 避开角点、内部最近面并列等没有唯一普通梯度的位置。
    local_points = np.array([
        [0.9, 0.5], [-0.7, 0.1], [0.1, 0.8],
        [0.1, 0.05], [0.3, -0.03], [-0.12, -0.2],
    ])
    centers = demo.POINTS[0] - local_points @ demo.ROTATION.T
    analytic = demo.point_rectangle_distance_jacobian(centers)
    numerical = np.stack([
        central_difference(lambda c: demo.point_rectangle_distance(c[None, :])[0], center)
        for center in centers
    ])
    check("距离对中心的雅可比（含内外部、旋转）", analytic, numerical)

    trajectory = np.linspace(demo.INITIAL, demo.GOAL, demo.N + 1)
    trajectory[:, 1] += np.sin(np.linspace(0, np.pi, demo.N + 1))
    controls = (np.diff(trajectory, axis=0) / demo.DT).ravel()
    for subdivisions in (1, 4, 20):
        analytic = demo.sampled_centers_jacobian(subdivisions)
        numerical = central_difference(lambda u: demo.sampled_centers(u, subdivisions), controls)
        check(f"采样中心灵敏度（每段 {subdivisions} 份）", analytic, numerical)
        np.testing.assert_array_equal(analytic[0], np.zeros((2, 2 * demo.N)))
        np.testing.assert_allclose(analytic[-1], demo.terminal_constraint_jacobian(controls))

    check(
        "终点约束雅可比",
        demo.terminal_constraint_jacobian(controls),
        central_difference(lambda u: demo.rollout(u)[-1] - demo.GOAL, controls),
    )
    # 多个控制序列核对完整链式法则，以及矩阵行顺序与约束 ravel() 的一致性。
    rng = np.random.default_rng(2026)
    for index in range(4):
        candidate = controls if index == 0 else controls + rng.normal(0, 0.15, controls.size)
        analytic = demo.clearance_constraint_jacobian(candidate)
        assert analytic.shape == (demo.clearance_constraint(candidate).size, candidate.size)
        check(
            f"距离约束雅可比（控制样本 {index}）",
            analytic,
            central_difference(demo.clearance_constraint, candidate),
        )
        np.testing.assert_array_equal(analytic[:len(demo.POINTS)], 0.0)

    # 不可微处只检查约定的分支结果有限、不除零，不拿中心差分充当唯一导数。
    corners_and_center = np.vstack((demo.HALF_SIZE, [0.0, 0.0]))
    centers = demo.POINTS[0] - corners_and_center @ demo.ROTATION.T
    selected = demo.point_rectangle_distance_jacobian(centers)
    assert np.all(np.isfinite(selected))
    np.testing.assert_allclose(np.linalg.norm(selected[:, 0], axis=-1), 1.0)
    print("所有雅可比检查通过。")


if __name__ == "__main__":
    main()
