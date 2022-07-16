#滑块推理以求降低资源耗费，是另外一种可采用的推理方式。此部分代码与普通推理类似，故重复注释省略


# 定义影像裁块-拼接相关的类和函数

import numpy as np


class WindowGenerator:
    def __init__(self, h, w, ch, cw, si=1, sj=1):
        self.h = h
        self.w = w
        self.ch = ch
        self.cw = cw
        if self.h < self.ch or self.w < self.cw:
            raise NotImplementedError
        self.si = si
        self.sj = sj
        self._i, self._j = 0, 0

    def __next__(self):
        # 列优先移动（C-order）
        if self._i > self.h:
            raise StopIteration

        bottom = min(self._i + self.ch, self.h)
        right = min(self._j + self.cw, self.w)
        top = max(0, bottom - self.ch)
        left = max(0, right - self.cw)

        if self._j >= self.w - self.cw:
            if self._i >= self.h - self.ch:
                # 设置一个非法值，使得迭代可以early stop
                self._i = self.h + 1
            self._goto_next_row()
        else:
            self._j += self.sj
            if self._j > self.w:
                self._goto_next_row()

        return slice(top, bottom, 1), slice(left, right, 1)

    def __iter__(self):
        return self

    def _goto_next_row(self):
        self._i += self.si
        self._j = 0


def recons_prob_map(patches, ori_size, window_size, stride):
    """从裁块结果重建原始尺寸影像"""
    h, w = ori_size
    win_gen = WindowGenerator(h, w, window_size, window_size, stride, stride)
    prob_map = np.zeros((h, w), dtype=np.float)
    cnt = np.zeros((h, w), dtype=np.float)
    # XXX: 需要保证win_gen与patches具有相同长度。此处未做检查
    for (rows, cols), patch in zip(win_gen, patches):
        prob_map[rows, cols] += patch
        cnt[rows, cols] += 1
    prob_map /= cnt
    return prob_map



from operator import itemgetter
from subprocess import run

import cv2


# 设置滑窗大小与滑动步长
WINDOW_SIZE = 256
STRIDE = 128

A_PATH = "demo_data/A.png"
B_PATH = "demo_data/B.png"
GT_PATH = "demo_data/GT.png"


# 读入影像
im_a = cv2.imread(A_PATH)
im_b = cv2.imread(B_PATH)
im_gt = cv2.imread(GT_PATH, cv2.IMREAD_GRAYSCALE)
ori_size = im_a.shape[:2]

# 把滑窗中的内容全部取出，存在一个列表中
patch_pairs = []
for rows, cols in WindowGenerator(*ori_size, WINDOW_SIZE, WINDOW_SIZE, STRIDE, STRIDE):
    patch_pairs.append((im_a[rows, cols], im_b[rows, cols]))

# 导出输入尺寸为滑窗大小的模型，--fixed_input_shape中的batch size等于len(patch_pairs)
run(
    f"python PaddleRS/deploy/export/export_model.py \
        --model_dir=./dynamic_models/best_model \
        --save_dir=./static_models/{WINDOW_SIZE}x{WINDOW_SIZE} \
        --fixed_input_shape=[{len(patch_pairs)},3,{WINDOW_SIZE},{WINDOW_SIZE}]",
    shell=True,
    check=True
)

# 构建预测器并执行推理
# 需要注意的是，使用这种方式将一次性对所有patch进行推理，因此可能占用较大显存
# 根据实际情况，开发者可以适当减小推理阶段使用的batch size（即，采取串行或半串行半并行的策略），用时间换空间
predictor = Predictor(f"static_models/{WINDOW_SIZE}x{WINDOW_SIZE}", use_gpu=True)
res = predictor.predict(patch_pairs)

# 取出所有的概率图patch并重建
prob_patches = map(itemgetter((..., 1)), map(itemgetter('score_map'), res))
prob_map = recons_prob_map(prob_patches, ori_size, WINDOW_SIZE, STRIDE)

# 对概率图进行阈值分割，得到二值变化图
cm_slide = (prob_map>0.5).astype('int32')


 可视化推理结果

# 从左到右依次显示：第一时相影像、第二时相影像、整图推理结果、滑窗推理结果以及真值标签
plt.figure(constrained_layout=True)
plt.subplot(151)
plt.imshow(Image.open("demo_data/A.png"))
plt.gca().set_axis_off()
plt.title("Image1")
plt.subplot(152)
plt.imshow(Image.open("demo_data/B.png"))
plt.gca().set_axis_off()
plt.title("Image2")
plt.subplot(153)
plt.imshow((cm_1024x1024*255).astype('uint8'))
plt.gca().set_axis_off()
plt.title("Pred_whole")
plt.subplot(154)
plt.imshow((cm_slide*255).astype('uint8'))
plt.gca().set_axis_off()
plt.title("Pred_slide")
plt.subplot(155)
plt.imshow((np.asarray(Image.open("demo_data/GT.png"))*255).astype('uint8'))
plt.gca().set_axis_off()
plt.title("GT")

plt.show()