"""用点到旋转矩形的距离约束优化轨迹；教学模型为固定朝向的二维平移。"""

# argparse：解析命令行参数，例如运行脚本时加上 --plot。
# from ... import ...：从某个模块中导入指定对象。
# import numpy as np：给 numpy 起一个较短的别名 np，后面用 np.array 等调用。
# minimize 是函数；传给它目标函数和约束，它负责搜索控制量。
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

# 本例只求解一次包含 N 步的完整轨迹，没有 04 中反复求解并执行首项的循环。
# 每一步控制为 [vx, vy]，状态为矩形中心 [x, y]，朝向 YAW 始终固定。
# 数组形状（shape）是理解下文的关键：
#   flat_controls: (2*N,) = (40,)      一维控制数组，供优化器使用
#   controls:      (N, 2) = (20, 2)    每行一个 [vx, vy]
#   states:        (N+1, 2) = (21, 2)  包含初始状态的所有中心位置
# (40,) 中的逗号表示只有一个维度；它不是 (40, 1) 的二维列向量。
N = 20
DT = 0.2
# np.array 将 Python 列表转换成 NumPy 数组，支持逐元素运算和广播。
# INITIAL、GOAL 的 shape 都是 (2,)；POINTS 是 (2, 2)，每行一个障碍点。
INITIAL = np.array([0.0, 0.0])
GOAL = np.array([4.0, 0.0])
POINTS = np.array([[2.0, -0.15], [2.0, 0.15]])
HALF_SIZE = np.array([0.4, 0.25])  # 半长、半宽
YAW = 0.2  # 本例朝向固定；vx、vy 可独立控制，不是差速车辆模型。
SAFETY = 0.2
MARGIN = 0.03
# 二维旋转矩阵 R 的形状为 (2, 2)，用于车体系和世界系之间的转换。
# np.sin / np.cos 的输入角度单位是弧度。
ROTATION = np.array([[np.cos(YAW), -np.sin(YAW)], [np.sin(YAW), np.cos(YAW)]])


def rollout(flat_controls):
    # reshape 改变数组形状，不改变元素顺序；元素总数必须相同。
    # 例如 [vx0, vy0, vx1, vy1].reshape(2, 2) 得到：
    # [[vx0, vy0],
    #  [vx1, vy1]]
    controls = flat_controls.reshape(N, 2)
    # axis=0 表示沿着“行的方向”累加，各列分别计算。
    # 例如 controls=[[1, 2], [3, 4]]：
    # np.cumsum(controls, axis=0) -> [[1, 2], [4, 6]]。
    # 这表示各时刻累计的 x、y 方向控制，而不是把两个方向混在一起。
    # INITIAL 的 (2,) 会广播到 (N, 2) 的每一行：
    # INITIAL + DT * 累积控制 -> N 个未来位置，形状 (N, 2)。
    # np.vstack 按行堆叠；先把 INITIAL 当作一行，再接上 N 个未来位置。
    # 双层括号中，内层 (INITIAL, ...) 是装着两个待拼接数组的元组。
    # 返回值形状为 (N+1, 2)，第 0 行就是 INITIAL。
    # 
    # 因为这里不涉及角度，所以直接累加 DT 即可
    return np.vstack((INITIAL, INITIAL + DT * np.cumsum(controls, axis=0)))


def point_rectangle_distance(centers):
    """返回每个位姿到每个障碍点的有符号距离，形状 (位姿数, 点数)。"""
    # 假设传入 K 个矩形中心、P 个障碍点，当前 P=2：
    # centers.shape=(K, 2)，POINTS.shape=(P, 2)。
    # 数组索引里“:”表示取该轴全部元素，“None”表示插入长度为 1 的新轴。
    # POINTS[None, :, :] -> (1, P, 2)
    # centers[:, None, :] -> (K, 1, 2)
    # 相减时广播成 (K, P, 2)，得到每个中心与每个障碍点的相对位置。
    # 例如中心 [[0,0], [1,0]]、点 [[2,0], [3,0]]，相减得到：
    # [[[2,0], [3,0]],       # 相对于第一个中心
    #  [[1,0], [2,0]]]。     # 相对于第二个中心
    # @ 是矩阵乘法。对三维数组，它对最后两个轴做矩阵乘法，保留前面的批次轴。
    # (K, P, 2) @ (2, 2) -> (K, P, 2)。
    # 行向量右乘 R，等价于列向量左乘 R.T：将世界坐标转换到车体坐标。
    
    # local_points 是一个三轴数组
    """
    local_points 是一个三轴数组， (K, P, 2)，第一个维度存储的是第几个矩形中心，第二个维度存储的是每个障碍物，第三个维度存储的是相对于中心的位姿
    """
    # local_points[k, p] 得到一个形状为 (2,) 的数组，意思是：当车辆中心位于 centers[k]、朝向为 YAW 时，第 p 个障碍点在车体坐标系中的坐标。
    # 这里不仅仅是“减去中心”，还包含了旋转：
    local_points = (POINTS[None, :, :] - centers[:, None, :]) @ ROTATION
    # np.abs 对每个坐标取绝对值：矩形关于自身 x、y 轴对称，距离与符号无关。
    # HALF_SIZE=(半长, 半宽)，shape=(2,)，会广播到所有 K*P 个局部坐标。
    # q[..., 0] 表示 x 方向超过矩形边界多少；q[..., 1] 同理。
    # 正数=该方向越界，负数=该方向仍在边界以内。
    # 例如局部点 [0.7, 0.1]、半尺寸 [0.4, 0.25]，q=[0.3, -0.15]。
    q = np.abs(local_points) - HALF_SIZE
    # np.maximum(q, 0.0) 是“逐元素”比较，把负值替换成 0，形状不变。
    # np.linalg.norm(..., axis=-1) 沿最后一维计算欧氏长度 sqrt(dx²+dy²)。
    # axis=-1 就是最后一个轴；这里把坐标轴的两个数合成一个距离。
    # 因而 (K, P, 2) -> (K, P)。例如 [0.3, 0] 的长度是 0.3。
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    # 注意 np.max 与 np.maximum 不同：
    # np.max(q, axis=-1) 对每个 [qx, qy] 取最大值，消去最后一个轴。
    # np.minimum(..., 0.0) 再逐元素与 0 比较，只保留负值。
    # 点在内部时，例如 q=[-0.3, -0.15]，最近边界距为 0.15，inside=-0.15。
    # 点在外部时，至少一个 q 分量为正，此时 inside=0。
    # 因此：外部 -> outside>0、inside=0；内部 -> outside=0、inside<0。
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    # 外部是点到矩形的欧氏距离；内部负值是到边界距离的负值。
    return outside + inside


# sampled_centers 的作用是：在相邻两个预测位置之间插入更多采样位置，用来检查车辆在运动途中是否靠障碍物太近。
def sampled_centers(flat_controls, subdivisions):
    # subdivisions 是每一段分成多少份，例如 4 表示每段取 4 个采样位置。
    states = rollout(flat_controls)
    # np.arange(start, stop) 包含 start、不包含 stop；所以这里用 +1。
    # subdivisions=4 时：arange(1, 5) -> [1,2,3,4]，除以 4 -> [.25,.5,.75,1]。
    # / 为逐元素除法，得到浮点数组；fractions.shape=(subdivisions,)。
    fractions = np.arange(1, subdivisions + 1) / subdivisions
    # Python 切片 a[start:stop:step] 不包含 stop，省略边界表示取到头/尾。
    # states[:-1] 取除最后一行外的所有行，即每段起点，shape=(N, 2)。
    # states[:-1, None, :] 在中间插入新轴，shape=(N, 1, 2)。
    # np.diff(states, axis=0) 做相邻行相减：states[1:] - states[:-1]，
    # 得到每段位移，shape=(N, 2)；再用 [:, None, :] 变为 (N, 1, 2)。
    # fractions[None, :, None] 则变为 (1, subdivisions, 1)。
    # 相乘后广播成 (N, subdivisions, 2)，每一段各自乘上所有插值比例。
    # 再加对应起点，得到“起点 + 比例 * 位移”，即线性插值。
    # 例：起点 [0,0]、终点 [1,2]、比例 .25 -> [0.25,0.5]。
    # 外面的括号允许表达式换行；这里没有逗号，所以不是元组。
    # samples.shape == (N, subdivisions, 2)
    samples = (
        states[:-1, None, :]
        + fractions[None, :, None] * np.diff(states, axis=0)[:, None, :]
    )
    # reshape(-1, 2) 中 -1 表示由 NumPy 自动推算该维度的长度。
    # (N, subdivisions, 2) -> (N*subdivisions, 2)，按“段、段内采样点”排列。
    # states[:1] 保留二维形状 (1, 2)，而 states[0] 会变成一维 (2,)。
    # fractions 不包含 0，所以额外加入整个轨迹的初始位置。
    # 各段末端已包含在 fractions=1 的采样里，无须重复添加下一段起点。
    # 返回 (1+N*subdivisions, 2)：4 份时为 (81, 2)，20 份时为 (401, 2)。
    # samples.reshape(-1, 2)：把分段的采样位置展开
    # samples.shape == (段数, 每段采样数, 2)  ===>   (总采样,2)
    # states[:1]：取出初始位置，并保留二维形状
    return np.vstack((states[:1], samples.reshape(-1, 2)))


# 把“车辆与障碍点必须保持足够距离”转换为优化器能够检查的不等式约束。
"""
候选控制序列
    ↓
生成预测轨迹并插入采样位置
    ↓
计算每个位置下，障碍点到车辆矩形的距离
    ↓
减去要求的安全距离
    ↓
返回每一项约束的余量
"""
def clearance_constraint(flat_controls):
    # SciPy 的 ineq 要求 >= 0；与 C++ NLopt 中的 <= 0 约定相反。
    # 函数嵌套调用先计算里面：先获得 81 个中心，再计算到 2 个点的距离。
    # distances.shape=(81, 2)，每个采样位置对每个障碍点都要满足安全距离。
    # ① 根据控制序列，得到要检查的车辆位置
    # ② 计算障碍点到车辆矩形的距离
    distances = point_rectangle_distance(sampled_centers(flat_controls, 4))
    # 两个标量会广播到全部距离；每项约束要求 distance - 0.2 - 0.03 >= 0。
    # ravel() 按默认行优先顺序展开成一维数组，形状 (162,)。
    # 例如 [[a,b],[c,d]].ravel() -> [a,b,c,d]。
    # 优化器会把返回数组中的每一个元素都当作一个不等式约束。
    # ③ 把距离转换成约束余量
    # 也就是优化时，每个距离都应该满足：  d >= 0.2 + 0.03 = 0.23
    # SciPy 中 "ineq" 约束的约定是：  g(U) >= 0
    # ④ .ravel() 将全部约束展开, 也就是每个位置对每个障碍点的余量
    # 把 (81, 2) 的数组展开为 (162,) 的一维数组：
    """
    位置0对障碍点0的余量
    位置0对障碍点1的余量
    位置1对障碍点0的余量
    位置1对障碍点1的余量
    ……
    """
    return (distances - SAFETY - MARGIN).ravel()


def point_rectangle_distance_jacobian(centers):
    """距离对世界系中心 [x, y] 的导数，形状 (中心数, 障碍点数, 2)。"""
    local_points = (POINTS[None, :, :] - centers[:, None, :]) @ ROTATION
    q = np.abs(local_points) - HALF_SIZE
    local_gradients = np.zeros_like(local_points)

    # 用显式循环展示分段求导；每个障碍点对应一个距离函数。
    for k in range(len(centers)):
        for p in range(len(POINTS)):
            outside = np.maximum(q[k, p], 0.0)
            length = np.linalg.norm(outside)
            if length > 0.0:
                # 外部：d = ||max(q, 0)||，对局部点坐标的梯度为
                # max(q,0) / ||max(q,0)|| * sign(local_point)。
                local_gradients[k, p] = outside / length * np.sign(local_points[k, p])
            else:
                # 内部或边界：d=max(qx,qy)，只有最近面的法向分量参与。
                # argmax 返回较大分量的索引（0=x，1=y），相等时选第一个。
                axis = np.argmax(q[k, p])
                local_gradients[k, p, axis] = (
                    1.0 if local_points[k, p, axis] >= 0.0 else -1.0
                )
                # 内部最近面并列、矩形角点等处没有唯一普通梯度。
                # 此时选择一个分支的极限梯度，不能用中心差分证明它唯一。

    # local = (point-center) @ R：中心移动与相对点移动方向相反，因此有负号。
    # local_gradients 中每个梯度按行存放，转换回世界系时右乘 R.T。
    return -local_gradients @ ROTATION.T


def sampled_centers_jacobian(subdivisions):
    """采样中心对全部控制的雅可比，形状 (1+N*subdivisions, 2, 2*N)。"""
    # 第 k 段的比例 alpha 处：
    # p(k,alpha) = INITIAL + DT*sum(u_j, j<k) + alpha*DT*u_k。
    # 所以前面的控制系数为 DT，当前控制系数为 alpha*DT，未来控制系数为 0。
    # 此模型为线性平移，因此这些导数与具体控制值无关。
    jacobian = np.zeros((1 + N * subdivisions, 2, 2 * N))
    fractions = np.arange(1, subdivisions + 1) / subdivisions
    for k in range(N):
        for j, fraction in enumerate(fractions):
            # 与 sampled_centers 的展开顺序相同：先按段，再按段内比例。
            row = 1 + k * subdivisions + j
            # 控制排列为 [vx0, vy0, vx1, vy1, ...]。
            # 步长 2 的切片分别选择以前的 vx 列和 vy 列。
            jacobian[row, 0, : 2 * k : 2] = DT
            jacobian[row, 1, 1 : 2 * k : 2] = DT
            # np.eye(2) 是 2x2 单位矩阵；当前 vx 只影响 x，vy 只影响 y。
            jacobian[row, :, 2 * k : 2 * k + 2] = fraction * DT * np.eye(2)
    # 第 0 个采样中心是固定 INITIAL，不能被控制改变，所以第 0 块全为 0。
    return jacobian


def clearance_constraint_jacobian(flat_controls):
    """每行一个距离约束，每列一个控制分量，当前形状为 (162, 40)。"""
    centers = sampled_centers(flat_controls, 4)
    distance_jacobian = point_rectangle_distance_jacobian(centers)  # (K, P, 2)
    center_jacobian = sampled_centers_jacobian(4)  # (K, 2, 2*N)
    # 链式法则：dg/dU = (dd/dcenter) @ (dcenter/dU)。
    # SAFETY、MARGIN 是常数，导数为零。
    # k=采样中心，p=障碍点，i=x/y，j=控制分量。
    # einsum 对重复而未出现在输出中的 i 求和：
    # jac[k,p,j] = sum(distance_jacobian[k,p,i] * center_jacobian[k,i,j], i=0,1)。
    jacobian = np.einsum("kpi,kij->kpj", distance_jacobian, center_jacobian)
    # 只合并“位置、障碍点”两个轴，保留控制列；行顺序必须与约束 ravel() 一致。
    return jacobian.reshape(-1, 2 * N)


def terminal_constraint_jacobian(flat_controls):
    """终点误差对控制的雅可比：第 0 行为 x 误差，第 1 行为 y 误差。"""
    # 终点 = INITIAL + DT*sum(u_k)，减去固定 GOAL 不改变导数。
    # 虽然结果不依赖控制值，仍接收 flat_controls，以满足优化器回调接口。
    jacobian = np.zeros((2, flat_controls.size))
    jacobian[0, 0::2] = DT
    jacobian[1, 1::2] = DT
    return jacobian


# 用来评价一组候选控制序列：既希望预测轨迹接近参考轨迹，又希望控制速度不要太大。
# 返回一个代价值
def objective(flat_controls):
    states = rollout(flat_controls)
    # linspace(start, stop, num) 在两端之间生成 num 个等间隔样本，默认包含两端。
    # 两端也可以是数组：linspace([0,0], [4,0], 3) -> [[0,0],[2,0],[4,0]]。
    # 这里 reference.shape=(N+1, 2)，是一条匀速直线参考轨迹。
    reference = np.linspace(INITIAL, GOAL, N + 1)
    # states[1:] 与 reference[1:] 都排除初始状态，形状为 (N, 2)。
    # 相减得到各时刻的 [x误差, y误差]；**2 表示逐元素平方。
    # np.sum 未指定 axis，会把所有行、列的元素加起来，得到标量。
    # 一维数组 flat_controls @ flat_controls 是点积，也就是全部控制量的平方和。
    # float(...) 把 NumPy 标量转换为 Python 内置浮点数。
    # 本函数只返回代价，没有返回解析梯度；下面的 minimize 将数值估计导数。
    # 它比较的是“同一步的位置误差”，而不只是预测位置到这条直线的最短距离。
    # 对于一维数组 @ 是点积，如果是 *，那么结果还是 N
    return float(
        np.sum((states[1:] - reference[1:]) ** 2)
        + 0.05 * (flat_controls @ flat_controls)
    )


def save_plot(states, initial_states, destination):
    # 函数内部也可以 import：只有请求 --plot、真正调用 save_plot 时才加载绘图库。
    import matplotlib

    # Agg 是不需要桌面窗口的绘图后端，适用于服务器；必须在导入 pyplot 前设置。
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    # 逗号赋值叫“解包”：subplots 返回两个对象，分别绑定到 fig 和 ax。
    # fig 是整张画布，ax 是其中的坐标区域。figsize 的单位是英寸。
    fig, ax = plt.subplots(figsize=(9, 4))
    # [:, 0] 取所有行的第 0 列，即所有 x；[:, 1] 取所有 y。
    # "--" 表示虚线；"o-" 表示带圆点的实线；label 是图例中的文字。
    ax.plot(initial_states[:, 0], initial_states[:, 1], "--", label="Initial guess")
    ax.plot(states[:, 0], states[:, 1], "o-", label="Optimized trajectory")
    # .T 是转置：POINTS 的每行是 [x,y]，转置后第 0 行是全部 x，第 1 行是全部 y。
    # 调用函数时的 * 表示“拆开位置参数”，与数组相乘的 * 含义不同。
    # ax.scatter(*POINTS.T, ...) 等价于
    # ax.scatter(POINTS[:, 0], POINTS[:, 1], ...)。c="red" 指定点的颜色。
    ax.scatter(*POINTS.T, c="red", label="Obstacle points")
    # 四行分别是局部坐标系中的四个角；乘 HALF_SIZE 时广播到每一行。
    # corners.shape=(4, 2)，得到 [-半长,-半宽] 等四个顶点。
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * HALF_SIZE
    # [::2] 表示从头到尾每隔 2 行取一行：索引 0、2、4……，避免矩形画得太密。
    # for 每次取出其中一行，因此 center.shape=(2,)。
    for center in states[::2]:
        # 与距离函数中的坐标转换方向相反：这里把局部顶点转回世界坐标。
        # 行向量右乘 ROTATION.T，再广播加上矩形中心，得到世界系四个顶点。
        # Polygon 创建多边形；fill=False 只画边框，alpha=.4 设置透明度。
        # add_patch 将多边形添加到坐标区域中。
        ax.add_patch(Polygon(corners @ ROTATION.T + center, fill=False, alpha=0.4))
    # 下方均为关键字参数调用（参数名=参数值），调用顺序不决定它们的含义。
    ax.set(xlabel="x (m)", ylabel="y (m)", title="Point-to-rectangle constraints")
    # x、y 两轴每米显示为相同长度，矩形与距离不会因坐标比例不同而变形。
    ax.axis("equal")
    # True / False 是 Python 布尔值，首字母大写；这里打开网格。
    ax.grid(True)
    # 根据前面各图形的 label 自动生成图例。
    ax.legend()
    # destination 是 pathlib.Path 对象；.parent 是它所在的目录。
    # parents=True 允许创建缺失的上级目录，exist_ok=True 允许目录已经存在。
    destination.parent.mkdir(parents=True, exist_ok=True)
    # dpi 是输出分辨率；bbox_inches="tight" 会裁掉多余空白。
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    # 保存完成后关闭画布、释放资源，不会删除已保存的图片。
    plt.close(fig)
    # f"...{表达式}..." 是 f-string，会把花括号内的表达式求值并嵌入文字。
    print(f"图像已保存：{destination}")


def main():
    # __doc__ 是本模块开头三引号字符串的内容，在这里作为命令行帮助说明。
    parser = argparse.ArgumentParser(description=__doc__)
    # action="store_true"：不写 --plot 时 args.plot=False；写了则为 True。
    # 可运行 python main.py --help 查看帮助，python main.py --plot 求解并保存图片。
    parser.add_argument(
        "--plot",
        action="store_true",
        help="将轨迹图保存到 output/examples/05_point_constraints.png",
    )
    # 读取实际命令行参数，返回 Namespace 对象；用 args.plot 访问解析出的选项。
    args = parser.parse_args()
    # 避障是非凸问题。用向上拱起的轨迹初始化，显式选择从上方绕行。
    # 先生成 (21, 2) 的直线路径，作为初始猜测的基础。
    initial_states = np.linspace(INITIAL, GOAL, N + 1)
    # [:, 1] 只选择 y 列；+= 表示在原数组的这些元素上累加。
    # linspace(0, pi, N+1) 产生从 0 到 pi 的角度，sin 从 0 上升到 1 再降回 0。
    # 因此中间位置向上拱起，而起终点不变（浮点计算可能有极小误差）。
    initial_states[:, 1] += np.sin(np.linspace(0, np.pi, N + 1))
    # 相邻位置相减得到 N 段位移，除以 DT 得到每段 [vx,vy]，再展开成 (40,)。
    # 这里由猜测的“轨迹”反推出“控制”，与 rollout 由控制积分出轨迹相对应。
    guess = (np.diff(initial_states, axis=0) / DT).ravel()
    # objective 不加括号表示传递“函数本身”，由优化器反复调用；
    # objective(guess) 则是立即调用函数，传入代价数值，含义不同。
    # guess 是控制序列初值；优化器改变的是全部 40 个控制分量。
    # 目标代价仍使用数值差分；下面单独给每个约束提供解析雅可比。
    # minimize 的 jac 参数控制目标梯度；约束字典的 "jac" 控制对应约束的雅可比。
    result = minimize(
        objective,
        guess,
        method="SLSQP",
        # [(-2.0, 2.0)] 是含一个元组的列表；列表 * 40 表示重复 40 次。
        # 每个元组给一个控制分量的 (下界, 上界)，分别限制 vx、vy。
        # 注意这不是限制 sqrt(vx**2+vy**2) <= 2，而是每个分量绝对值 <= 2。
        bounds=[(-2.0, 2.0)] * (2 * N),
        # [] 是列表，{} 是字典，字典用 "键": 值 保存信息。
        # 这里列表里有两个字典，分别描述终点等式约束和距离不等式约束。
        # lambda u: 表达式 是匿名函数，相当于：
        # def terminal_constraint(u):
        #     return rollout(u)[-1] - GOAL
        # [-1] 取最后一个状态，返回 [终点x误差, 终点y误差]，即两个等式。
        # "eq" 要求返回的每项等于 0；"ineq" 要求返回的每项大于等于 0。
        constraints=[
            {
                "type": "eq",
                "fun": lambda u: rollout(u)[-1] - GOAL,
                "jac": terminal_constraint_jacobian,  # 返回 (2, 40)
            },
            {
                "type": "ineq",
                "fun": clearance_constraint,
                "jac": clearance_constraint_jacobian,  # 返回 (162, 40)
            },
        ],
        # options 也是字典：ftol 是 SLSQP 的收敛容差参数，maxiter 是迭代次数上限。
        # 1e-9 是科学计数法，表示 10 的 -9 次方。
        options={"ftol": 1e-9, "maxiter": 300},
    )
    # result 是优化结果对象：.success 是成功标志，.x 是控制数组，.fun 是最终代价。
    # not 表示取反，or 表示逻辑“或者”，只要一个条件为真就进入 if。
    # np.isfinite 逐元素检查是不是有限数（不是 NaN、正无穷、负无穷）。
    # np.all 再把布尔数组归约成一个布尔值：必须所有元素都为 True。
    if not result.success or not np.all(np.isfinite(result.x)):
        # raise 主动抛出异常；没有捕获异常时，程序停止并打印错误信息。
        raise RuntimeError(f"求解失败：{result.message}")
    # 将最终控制量重新积分，得到优化后的轨迹；并不把 guess 当成最终结果。
    states = rollout(result.x)
    # 以更密的采样独立复核基本安全距离，不将求解成功当作可行性证明。
    # 每段改用 20 个采样点，距离数组形状为 (401, 2)。
    # np.min 没指定 axis，因此在所有位置、所有障碍点中取一个最小值。
    minimum = np.min(point_rectangle_distance(sampled_centers(result.x, 20)))
    # 括号允许 if 条件跨行，下面四个条件用 or 连接，任一失败就报错：
    # 1. 终点 x、y 误差绝对值中的最大值是否超过容差；
    # 2. 任意控制分量是否超限；
    # 3. 优化中使用的任意距离约束是否小于 -容差；
    # 4. 更密采样得到的最小距离是否低于基本安全距离。
    # 1e-6 的小容差用于接纳浮点求解误差，不要求计算结果精确为零。
    if (
        np.max(np.abs(states[-1] - GOAL)) > 1e-6
        or np.max(np.abs(result.x)) > 2.0 + 1e-6
        or np.min(clearance_constraint(result.x)) < -1e-6
        or minimum < SAFETY - 1e-6
    ):
        raise RuntimeError("求解结果未通过约束复核。")
    # :.6f 是 f-string 的格式说明：按小数点后 6 位输出浮点数。
    # {states[-1]} 未指定格式，按 NumPy 数组的默认格式显示终点坐标。
    print(f"求解成功，代价={result.fun:.6f}，终点={states[-1]}")
    print(f"密集采样最小距离={minimum:.6f} m，要求至少 {SAFETY} m。")
    print("本例只约束给定障碍点；离散采样不构成连续时间无碰撞证明。")
    # 只有命令行启用 --plot 才会走这个分支；否则只打印求解结果。
    if args.plot:
        # __file__ 是当前脚本路径；resolve() 将它解析为绝对路径。
        # parents[0] 是 05_point_constraints，parents[1] 是 example，
        # parents[2] 是项目根目录。Path 的 / 运算符表示拼接路径，不是数字除法。
        destination = (
            Path(__file__).resolve().parents[2]
            / "output/examples/05_point_constraints.png"
        )
        save_plot(states, initial_states, destination)


# 直接运行这个文件时 __name__ 为 "__main__"，因此调用 main()。
# 被其他文件 import 时 __name__ 为模块名，不会自动开始求解。
if __name__ == "__main__":
    main()
