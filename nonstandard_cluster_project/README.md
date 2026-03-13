# nonstandard_obstacle 细分聚类项目（DINOv2 + 聚类 + 簇命名回写）

本项目实现：
1) 扫描标注文件夹，解析每个标注文件，抽取 shapes 中 label/main 为 "nonstandard_obstacle" 的 polygon 实例；
2) 自动定位对应图片（优先用 annotation 内的 path，其次用 imageName，最后用文件名索引）；
3) 对每个实例做裁剪（bbox + 可选 pad 扩边），可选做 polygon mask 抑制背景；
4) 用 DINOv2 提取每个实例 crop 的特征向量（L2 normalize）；
5) PCA 降维（默认 256 维），然后聚类（默认 MiniBatchKMeans）；
6) 输出每个实例的 cluster_id，并为每个簇生成预览拼图（每簇抽样）；
7) 你根据预览图给簇命名（cluster_id -> 细分类标签），然后一键回写标注，生成“细分后标注文件夹”。

--------------------------------------------
一、环境准备

建议 Python 3.10+。

(1) 安装 PyTorch（按你机器 CUDA 版本选择对应命令）
    - 有 GPU：安装 CUDA 版 torch/torchvision
    - 无 GPU：安装 CPU 版 torch/torchvision（会慢很多）

(2) 安装其他依赖：
    pip install -r requirements.txt

注意：DINOv2 权重默认通过 torch.hub 首次自动下载，需要网络一次。
如无网络，请提前准备好 torch hub cache（见 torch 文档），或在有网机器先运行一次 extract_features.py。

--------------------------------------------
二、配置文件

编辑 config.yaml，确认以下路径正确：
- label_root: 标注文件夹
- rgb_root: 图片根目录
- out_root: 输出目录（会自动创建）
- target_label: nonstandard_obstacle

你可以调整：
- pad_ratio: bbox 扩边比例（建议 0.15~0.30，小目标更大）
- use_polygon_mask: true/false（true 通常更利于聚类）
- model_name: dinov2_vitb14 / dinov2_vitl14（vitl 更准更慢更吃显存）
- batch_size: GPU 视显存调整
- kmeans_k: 聚类簇数（建议“偏大一点再合并”：2w 实例可从 200~800试；50w 实例可 800~3000）

--------------------------------------------
三、运行方式（按步骤）

1) 构建实例索引（只抽 nonstandard_obstacle）：
   python run_pipeline.py --config config.yaml index

产物：
- out_root/index/instances.jsonl

2) 提取 DINOv2 特征：
   python run_pipeline.py --config config.yaml features

产物：
- out_root/features/features.npy（memmap）
- out_root/features/meta.csv

3) PCA + 聚类：
   python run_pipeline.py --config config.yaml cluster

产物：
- out_root/clusters/assignments.csv
- out_root/clusters/cluster_stats.csv

4) 生成每个簇的预览拼图（用于你快速命名）：
   python run_pipeline.py --config config.yaml preview

产物：
- out_root/clusters/previews/cluster_000123.jpg
- out_root/clusters/preview_index.html（简单索引页）

5) 你人工给簇命名：
   在 out_root/clusters/ 下创建 cluster_map.csv，内容示例：
     cluster_id,label
     12,cone
     31,hydrant
     105,water_barrier
     220,round_bollard
     7,other

建议标签集合：
- cone
- hydrant
- water_barrier
- round_bollard
- other

6) 回写标注（生成新标注目录，不覆盖原始标注）：
   python run_pipeline.py --config config.yaml apply --cluster-map out_root/clusters/cluster_map.csv

默认回写策略：
- 保留 main/label 为 "nonstandard_obstacle"
- 将细分类写入 shape["sub"] = 你的 label（更安全，兼容旧工具）
如你确定工具支持主类变化，可加参数：
   --overwrite-main
它会把 shape["main"] 和 shape["label"] 都改成细分类。

产物：
- out_root/label_refined/（与原 label_root 同结构，写入更新后的标注文件）

--------------------------------------------
四、常见问题/调参建议

1) 聚类混簇（一个簇里多类）
   - 提高 kmeans_k（先分碎再合并）
   - 提高 pad_ratio 或开启 use_polygon_mask（减少背景主导）
   - 尝试 model_name=dinov2_vitl14（更强但更慢）

2) 消防栓/远处小目标容易错
   - 增大 pad_ratio 到 0.25~0.35
   - 把 input_size 提到 336（更吃算力）

3) 图片路径找不到
   - 先确认 annotation 中 path 是相对 rgb_root 的路径
   - 本项目会建立 filename->fullpath 的索引兜底，但第一次 walk 会慢一点

--------------------------------------------
五、扩展：你想做“柱状水马 vs 方形水马”二次细分
建议先完成四大类 + other 的粗分，再在 water_barrier 子集上单独跑一次聚类/二分类，这样更稳。