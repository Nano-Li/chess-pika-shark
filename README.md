# 象棋棋盘识别项目

基于Python + OpenCV的象棋棋盘识别系统，专门针对天天象棋等实际场景优化。

## 项目结构

```
AIchess/
├── chess_vision/              # 棋盘识别模块（核心）
│   ├── chess_recognizer.py    # 基础版本（已验证可用）
│   └── chess_recognizer_v2.py # 改进版本（推荐使用）
├── test_shark_gui/            # 鲨鱼象棋GUI测试工具
│   ├── test_window_detection.py
│   ├── test_control_tree.py
│   └── test_mouse_hover_spy.py
└── xiangqi-analysis/          # 参考项目（识别效果不佳，仅供参考）

```

## 快速开始

### 1. 安装依赖

```bash
pip install opencv-python numpy pillow
```

### 2. 运行识别测试

```bash
cd d:/Projects/AIchess/chess_vision
python chess_recognizer_v2.py
```

### 3. 选择测试方式

- **方式1**: 截取全屏（最简单）
- **方式2**: 截取指定区域（推荐，速度快）
- **方式3**: 从文件读取（用于测试固定图片）

## 功能特点

### chess_recognizer_v2.py（改进版）

✅ **调试模式** - 自动保存识别过程的中间图像
- `debug_00_original.png` - 原始图像
- `debug_01_gray.png` - 灰度图
- `debug_02_edges.png` - 边缘检测
- `debug_03_dilated.png` - 形态学处理
- `debug_04_detected_board.png` - 检测到的棋盘
- `debug_05_final_result.png` - 最终结果（标注棋子）

✅ **详细输出** - 显示每个步骤的进度和结果

✅ **统计信息** - 红方/黑方棋子数量

✅ **性能优化** - 识别速度 ~30-50ms

## 当前状态

### 已实现
- ✅ 棋盘位置检测
- ✅ 格子分割（10行×9列）
- ✅ 棋子存在检测
- ✅ 棋子颜色识别（红/黑）
- ✅ FEN码生成

### 待改进
- ⚠️ 棋子类型识别（目前都标记为"兵"）
- ⚠️ 网格线精确检测
- ⚠️ 透视变形校正

## 性能对比

| 方案 | 识别速度 | 准确率 | 适用场景 |
|------|---------|--------|---------|
| xiangqi-analysis (Web) | ~800ms | 低 | 标准棋盘图片 |
| chess_recognizer.py | ~35ms | 中 | 天天象棋 |
| chess_recognizer_v2.py | ~30-50ms | 中+ | 天天象棋（推荐）|

## 模板采集工具

### template_splitter.py - 单个棋盘分割

**功能：** 将完整棋盘图像分割为90个格子模板

**使用步骤：**

```bash
# 1. 手动截取完整棋盘（确保边缘对齐）
# 2. 运行分割工具
cd chess_vision
python template_splitter.py

# 3. 输入棋盘图像路径
# 4. 输入模板集名称（如 high_res, low_res）
# 5. 选择是否生成网格预览
```

**输出：**
```
templates/
├── high_res/              # 高分辨率模板集
│   ├── 1-1.png           # 第1行第1列
│   ├── 1-2.png           # 第1行第2列
│   ├── ...
│   ├── 10-9.png          # 第10行第9列
│   └── info.txt          # 模板信息
└── low_res/               # 低分辨率模板集
    └── ...
```

### batch_template_splitter.py - 批量处理

**功能：** 一次性处理多个不同分辨率的棋盘图像

**使用步骤：**

```bash
# 1. 将多个棋盘图像放在同一目录
#    - high_res.png (高分辨率)
#    - low_res.png (低分辨率)
#    - standard.png (标准)

# 2. 运行批量工具
python batch_template_splitter.py

# 3. 输入图像所在目录
# 4. 确认处理
```

**命名规则：**
- 格子命名：`行-列.png`（从1开始）
- 示例：`1-1.png` = 第1行第1列（左上角）
- 示例：`10-9.png` = 第10行第9列（右下角）

---

## 下一步计划

### 阶段1：模板采集（进行中）✅
1. ✅ 创建模板分割工具
2. ⏳ 截取不同分辨率的棋盘图像
3. ⏳ 生成90个格子模板

### 阶段2：模板匹配识别
1. **实现模板匹配算法**
   - 加载模板库
   - 多尺度匹配（可选）
   - 相似度计算

2. **优化识别流程**
   - 先匹配空格子（排除无棋子位置）
   - 再匹配棋子类型
   - 生成准确的FEN码

3. **性能测试**
   - 测试不同分辨率模板的准确率
   - 测试识别速度
   - 平衡速度和准确率

### 阶段3：集成RPA
- 与鲨鱼象棋自动化操作结合
- 实时棋盘监控
- 自动走棋

## 使用示例

```python
from chess_recognizer_v2 import ChessboardRecognizer
import cv2

# 创建识别器
recognizer = ChessboardRecognizer(debug=True)

# 读取图片
image = cv2.imread("chessboard.png")

# 识别
result = recognizer.recognize(image)

if result['success']:
    print(f"FEN码: {result['fen']}")
    print(f"检测到 {result['detected_pieces']} 个棋子")
    print(f"耗时: {result['elapsed_ms']}ms")
```

## 调试技巧

如果识别效果不好：

1. **查看调试图像** - 检查哪个步骤出问题
2. **调整参数** - 修改颜色阈值、对比度阈值
3. **截取精确区域** - 只截取棋盘部分，避免干扰

## 许可证

MIT License

