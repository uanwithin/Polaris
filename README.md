# Polaris MPPC：无学习的数值优化基线

基于 `paper/NeuPAN.pdf` 第 III 节式（1）—（11），用 **C++20 + NLopt SLSQP**
求解直接点云、完整矩形车身的模型预测感知与控制（MPPC）。车辆采用差速运动学，
控制量为线速度 `v` 和角速度 `ω`。**不包含神经编码器、端到端训练或 PnP PAN 网络**。

模拟器用占据栅格产生激光点云，规划器只接收当前状态、上一次控制、当前点云和参考路径。
每个周期优化整段控制序列，只执行第一项，再观测、重新规划。

## 构建

需要 CMake ≥ 3.28、支持 C++20 的编译器（例如 GCC 13 / Clang 23）、Ninja 或 Make。
第三方依赖统一在 `third_party/` 中管理，以 `polaris::nlopt` 和 `polaris::rerun` 供主项目链接。
优先查找已安装的 NLopt ≥ 2.10 和 Rerun C++ SDK 0.40.0（也会查找 `/opt/sdk/rerun`）；
缺失时通过 CMake FetchContent 下载固定版本并校验 SHA256。
可用 `-DPOLARIS_FETCH_DEPENDENCIES=OFF` 禁止下载。Rerun 默认还会构建 Apache Arrow，首次构建需要网络并耗时较长。

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 6
```

Rerun Viewer 使用与 SDK 相同的版本。当前项目的 Python 环境已安装 0.40.0：

```bash
source .venv/bin/activate
rerun --version
```

其它环境可以单独安装 Viewer，无需安装本项目 Python 配置中的其它依赖：

```bash
uv tool install rerun-sdk==0.40.0
```

已有 SDK 时可以设置 `NLopt_DIR`、`rerun_sdk_DIR`；离线构建也可通过
`FETCHCONTENT_SOURCE_DIR_NLOPT`、`FETCHCONTENT_SOURCE_DIR_RERUN_SDK` 指向解压后的源码根目录。
已有 Arrow 时设置 `RERUN_DOWNLOAD_AND_BUILD_ARROW=OFF` 和 `Arrow_DIR`。

只构建算法及测试，不需要 Rerun/Arrow：

```bash
cmake -S . -B build-core -G Ninja -DCMAKE_BUILD_TYPE=Release -DPOLARIS_WITH_RERUN=OFF
cmake --build build-core -j 6
./build-core/polaris_demo --headless
```

## 运行与可视化

```bash
# 启动 Rerun Viewer；默认地图的障碍物直接挡住参考直线
./build/polaris_demo

# 按仿真时钟播放（求解耗时超过周期时不会跳过计算）
./build/polaris_demo --realtime

# 无图形界面运行，保存可回放文件和逐帧数据
./build/polaris_demo --headless --output output/avoidance.rrd --csv output/avoidance.csv
rerun output/avoidance.rrd

# 其它场景
./build/polaris_demo --scenario empty
./build/polaris_demo --scenario blocked --headless
./build/polaris_demo --help
```

Rerun 中选择 `sim_time` 时间轴，可播放、暂停和逐帧检查：

程序自动载入随仓库附带的 `resources/polaris.rbl`：左侧为地图主视图，右侧为控制量、
安全距离、耗时与图例。运行仿真不依赖 Python；仅重新生成布局时使用
`.venv/bin/python scripts/generate_blueprint.py`。

| 显示 | 含义 |
| --- | --- |
| 灰色栅格 | 仿真真实障碍物；不传入规划器 |
| 红色点 | 当前 360° 激光扫描的首个障碍物回波 |
| 蓝色线 | 参考路径 |
| 青色矩形、轮廓 | 实际车身、欧氏安全距离边界 |
| 黄色线、矩形 | 本轮优化的预测轨迹、未来车身 |
| 绿色线 | 已执行轨迹 |
| 标量曲线 | 线速度、角速度、距离、目标函数、求解耗时 |

`--output` 保存录制，不自动启动 Viewer。`--headless` 单独使用时不记录 Rerun 数据。
CSV 的状态是本轮执行前的状态，`speed/yaw_rate` 是本轮计划执行的控制。
程序返回码：`0` 到达并接近静止；`2` 阻塞、不可行或步数用尽；`3` 检测到碰撞；`1` 配置/运行错误。
封路场景预期返回 `2`，不会把停车当成到达目标。

## 与论文公式的对应

| 论文 | 实现 |
| --- | --- |
| （1）、（2）运动学 | 使用（1）的差速前向 Euler 模型；直接打靶消去状态变量，并解析传播状态对控制的 Jacobian。SLSQP 在迭代内线性化约束，不单独冻结整段运动学线性化模型。 |
| （3）控制可行集 | 限制 `v, ω` 的上下界及相邻控制的变化量，包含第一项相对上一执行控制的约束。 |
| （4）—（6）完整车身 | 以几何中心为状态位置的矩形，随状态平移和旋转。 |
| （7）、（8）点到车身距离 | 将点变换到车体系，用矩形投影解析求最近点；等价于该矩形上距离凸优化的闭式解，求欧氏距离而非平方距离。 |
| （9）、（11b）避碰 | 对每个有效激光点、每个预测时刻施加距离下界；额外检查区间内部。 |
| （10）、（11a）目标 | 位置/朝向跟踪的加权平方误差，加期望线速度误差；权重配置表示论文 `q², p²` 的系数。 |

距离在车身内部采用负值扩展，以便不可行初值也有退出障碍物的梯度；车身外的值就是准确欧氏距离，
对于正的安全距离，约束可行域与原距离定义一致。矩形棱角的不可微位置取一侧梯度。
不把点云变成障碍物包围盒、聚类、多边形或代价地图。

数值实现包含以下明确的工程选择：

- 参考路径按当前最近投影位置和期望速度向前采样；终点附近逐渐降低期望速度，以便停稳。
- SLSQP 使用解析目标梯度和约束 Jacobian，优化 `2 × horizon` 个控制变量。
- 上一轮序列移位作为热启动；必要时使用直行、左绕、右绕初值缓解非凸局部极小。最终轨迹仍由约束优化决定。
- 每个控制区间检查 1/4、1/2、3/4 和末端位姿。优化时额外加 `sampling_margin`；
  执行前以每区间 10 个位姿复核基本安全距离及控制约束。数值失败或迭代用尽的解也必须通过复核才能执行。
- 只有在点距离超出“最大预测行程 + 车身外接半径 + 安全余量”时才跳过其约束；不是任意截取最近的 N 个点。
- 找不到优化可行解时尝试受加速度限制的制动序列；制动也不可行时终止仿真并报告，绝不跳变为零控制伪装安全。
- 仿真用独立的矩形/栅格分离轴碰撞检测检查实际运动，每区间检查 20 个位姿。

这是第 III 节数学问题的数值基线，**不是后续章节神经求解器的复刻**。
SLSQP 是局部非线性求解器，不保证全局最优，也没有证明递归可行性。
稀疏点云、遮挡和离散时间检查不能保证连续空间/连续时间绝对无碰撞；`sampling_margin` 是工程余量，
不是消除盲区的证明。当前只处理静态障碍物，点云在一个预测时域内保持不动，定位和观测无噪声。
没有全局搜索，因此 U 形陷阱、长墙绕行或完全封路可能停在局部极小。
数值基线没有硬实时保证；终端和 Rerun 都记录实际求解耗时。

## 参数与代码位置

默认地图 `18 × 12 m`，分辨率 `0.1 m`；车辆 `1.0 × 0.6 m`；雷达 180 束、7 m 量程。
预测 18 步、步长 0.2 s，期望速度 1 m/s；安全距离 0.22 m，采样余量 0.04 m。

| 文件 | 修改内容 |
| --- | --- |
| `include/polaris/mppc.hpp` | 时域、权重、安全距离、速度/加速度限制、求解预算 |
| `include/polaris/grid_map.hpp` | 激光束数、量程 |
| `include/polaris/geometry.hpp` | 车辆尺寸、状态及控制类型 |
| `src/simulation.cpp` | 地图障碍物、起点、参考折线和目标点 |
| `src/mppc.cpp` | 目标、逐点约束、解析导数、数值求解与可行性复核 |
| `src/visualizer.cpp` | Rerun 记录和颜色 |

`GridMap::scan` 用栅格 DDA 求射线首个占据格的精确交点；没有回波的射线不生成假障碍物。
地图外视为占据。参考路径由折线顶点构造，不需要预先避开障碍物。

## 检查

```bash
ctest --test-dir build --output-on-failure
cmake --build build --target format-check
clang-tidy-23 -p build src/*.cpp tests/*.cpp
```

使用仓库原有 `.clang-format` 和 `.clang-tidy`，仅把后者遗留的头文件过滤路径改为 `include/polaris/`。
测试涵盖距离及梯度、旋转矩形、射线遮挡、薄墙碰撞、路径采样、运动学与控制约束，以及闭环到达。
可使用 `-DPOLARIS_SANITIZERS=ON` 开启 AddressSanitizer/UndefinedBehaviorSanitizer。

依赖参考：[Rerun C++ SDK](https://rerun.io/docs/getting-started/project-setup/cpp)、
[NLopt SLSQP](https://nlopt.readthedocs.io/en/stable/NLopt_Algorithms/#slsqp)。
