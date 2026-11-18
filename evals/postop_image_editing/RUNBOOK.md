# 复现手册

从 Reveal 根目录执行。需要能合法访问的 HDA 数据与本地 FLUX 底座。复现联合 LoRA 需 CUDA，形变模型可在 CPU 上跑。原实验环境使用 Python 3.10、PyTorch 2.9.1+cu128、Diffusers 0.39.0、Transformers 4.57.6、PEFT 0.21.3、MediaPipe 0.10.31、OpenCV headless 4.11.0.86，并安装 Pillow、NumPy 和 Accelerate。

```bash
export POSTOP_WORKSPACE="$PWD/downloads/postop"
export POSTOP_FLUX_BASE="/path/to/FLUX.2-klein-base-4B"
mkdir -p "$POSTOP_WORKSPACE/checkpoints/eyelid_shape_v1"
curl -fL https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
  -o "$POSTOP_WORKSPACE/checkpoints/eyelid_shape_v1/face_landmarker.task"
```

将 HDA 的五个类别目录放在 `$POSTOP_WORKSPACE/HDA-PlasticSurgery/`。若资产缺失，记录缺少的路径后暂停相关步骤。小样本试跑可用独立工作目录，避免覆盖正式权重。

```bash
exp=evals/postop_image_editing
python "$exp/hda_joint_data.py"
python "$exp/hda_joint_shape.py" train
python "$exp/hda_joint_mean_shape.py"
python "$exp/hda_joint_lora.py" train --steps 110
python "$exp/hda_joint_lora.py" eval --split val --limit 999
python "$exp/hda_joint_lora.py" eval --split test --limit 999
python "$exp/hda_joint_lora.py" eval --frozen --split test --limit 999
python "$exp/eval_hda_joint_landmarks.py"
python "$exp/report_hda_joint.py"
python "$exp/review_hda_joint.py"
```

查看工作目录下 `validation_outputs/hda_joint_v1/README.md`。训练参数及权重在 `checkpoints/hda_joint_v1/`。单图推理示例：

```bash
python "$exp/hda_joint_shape.py" infer /path/to/before.jpg Nose "$POSTOP_WORKSPACE/preview.png"
```

项目参数可选 Eyelid、Eyebrow、Nose、FacialBones、Facelift。将脚本换为 `hda_joint_lora.py` 可调用共享 LoRA。
