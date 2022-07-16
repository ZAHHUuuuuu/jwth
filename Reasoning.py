#导入库
import paddle
import numpy as np
from matplotlib import pylab
from paddlers.deploy import Predictor
from PIL import Image
from matplotlib import pyplot as plt


#模型转换#这部分将训练出的动态模型转化为静态模型，项目中已内置，故注释不用
#! python deploy/export/export_model.py --model_dir=./output/deeplabv3p/best_model/ --save_dir=./inference_model/

#构建predictor
predictor = Predictor(
                    "D:\pycharm\paddlesoftware\model\static model\Changing detection",#此处为对应模型对应路径，在交互后写个相应来确定加载哪个模型
                    use_gpu = True,
                    gpu_id = 0,#使用gpu的id，默认为零
                    cpu_thread_num = 1,#使用cpu进行预测时的线程数。默认1
                    use_mkl = False, #是否使用mkldn计算库，cpu情况下使用，默认false
                    mkl_thread_num = 4,#mkldn计算线程数，默认为4
                    use_trt = False,#是否使用TensorRT，默认False
                    use_glog = False,#是否启用glog日志，默认False
                    memory_optimize = True,#是否启用内存优化，默认为True
                    max_trt_batch_size = 1,#在使用TensorRT时配置的最大batch size，默认为1
                    trt_precision_mode = 'float32'#在使用TensorRT时采用的精度，可选值['float32', 'float16']。默认为'float32'
)

#用Predictor的predict()方法执行推理
res = predictor.predict(
    ("D:\pycharm\paddlesoftware\demo_data\A.png", "D:\pycharm\paddlesoftware\demo_data\B.png"),#此处由用户决定该输入一张图还是多张输入。根据所选项确认是哪一种要求。
        #对于场景分类、图像复原、目标检测和语义分割任务来说，该参数可为单一图像路径，或是解码后的、排列格式为（H, W, C）
        #且具有float32类型的BGR图像（表示为numpy的ndarray形式），或者是一组图像路径或np.ndarray对象构成的列表；对于变化检测任务来说，
        #该参数可以为图像路径二元组（分别表示前后两个时相影像路径），或是两幅图像组成的二元组，或者是上述两种二元组之一构成的列表。
    topk = 1,#场景分类模型预测时使用，表示预测前topk的结果。默认值为1。
    transforms = None,#数据预处理操作。默认值为None, 即使用`model.yml`中保存的数据预处理操作。
    warmup_iters = 0,# 预热轮数，用于评估模型推理以及前后处理速度。若大于1，会预先重复预测warmup_iters，而后才开始正式的预测及其速度评估。默认为0。
    repeats = 1#重复次数，用于评估模型推理以及前后处理速度。若大于1，会预测repeats次取时间平均值。默认值为1。
)

#第三步：解析predict()方法返回的结果。
cm_1024x1024 = res[0]['label_map']#此处是实现变化检测返回二值变化图
    #对于语义分割和变化检测任务而言，predict()方法返回的结果为一个字典或字典构成的列表。字典中的`label_map`键对应的值为类别标签图，对于二值变化检测
    #任务而言只有0（不变类）或者1（变化类）两种取值；`score_map`键对应的值为类别概率图，对于二值变化检测任务来说一般包含两个通道，第0个通道表示不发生
    #变化的概率，第1个通道表示发生变化的概率。如果返回的结果是由字典构成的列表，则列表中的第n项与输入的img_file中的第n项对应。





