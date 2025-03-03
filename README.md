# **AutoVideoDetectTrack: Multi-Model Video Object Detection and Segmentation**

## **Overview**
This is a multi-model-based **video object detection and segmentation** framework that integrates five powerful models:
- **Segment Anything Model 2 (SAM2)** for video and image prediction
- **Grounding DINO** for zero-shot object detection
- **YOLO-World** for high-performance real-time object detection
- **Qwen2.5-VL** for vision-language prompt generation
- **ClipSeg** for background filtering

This project provides **multi-modal object detection and segmentation** for various applications, including **autonomous driving, surveillance analysis, and video object tracking**.

---

## **Installation**
### **1. Download Pretrained Model Weights**
The project requires **five** pretrained models, which need to be downloaded separately.

#### **Segment Anything Model 2 (SAM2)**
```bash
cd checkpoints
bash download_ckpts.sh  # Downloads SAM2 weights
```
📌 Reference: [Grounded SAM2 Repo](https://github.com/IDEA-Research/Grounded-SAM-2?tab=readme-ov-file)

#### **Grounding DINO**
```bash
cd gdino_checkpoints
bash download_ckpts.sh  # Downloads Grounding DINO weights
```
📌 Reference: [Grounding DINO](https://github.com/IDEA-Research/Grounded-SAM-2?tab=readme-ov-file)

#### **YOLO-World**
```bash
cd yolo_world
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-worldv2.pt
```
📌 Reference: [YOLO-World Docs](https://docs.ultralytics.com/zh/models/yolo-world/#train-usage)

#### **Qwen2.5-VL (Vision-Language Model)**
**Option 1: Hugging Face (Online)**
```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
```
**Option 2: ModelScope (Offline, for Air-Gapped VM)**
```bash
mkdir -p qwen
cd qwen
wget https://modelscope.cn/api/v1/models/Qwen/Qwen2.5-VL-7B-Instruct/repo?Revision=master -O Qwen2.5-VL-7B-Instruct.zip
unzip Qwen2.5-VL-7B-Instruct.zip
```
📌 Reference: [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | [ModelScope](https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct/summary)

#### **ClipSeg (Background Segmentation)**
```bash
mkdir -p weights
wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O weights.zip
unzip -d weights -j weights.zip
```
📌 Reference: [ClipSeg Repo](https://github.com/timojl/clipseg/tree/master)

---

## **Model Initialization**
In the project, all models are initialized using:

```python
video_predictor, image_predictor, processor, grounding_model, yolo_model, device, qwen_processor, qwen_tokenizer, qwen_model, clipseg_model = initialize_models(
    sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
    model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
    grounding_model_id="IDEA-Research/grounding-dino-tiny",
    yolo_model_path="./yolo_world/yolov8x-worldv2.pt"
)
```
---

## **How to Run the Project**

To process a folder containing videos:
```bash
python New.py
```


---

## **Project Components**
### **1. SAM2 (Segment Anything Model 2)**
- Used for **video object segmentation**
- Generates **masks** for detected objects
- Handles **multi-frame video processing**

### **2. Grounding DINO**
- **Zero-shot object detection**
- Uses **natural language prompts** to detect objects
- Works alongside **YOLO-World** for improved accuracy

### **3. YOLO-World**
- **High-speed object detection**
- Used for **real-time tracking**
- Works with **Grounding DINO** for robust results

### **4. Qwen2.5-VL**
- **Generates textual descriptions of objects in an image**
- Classifies objects into **foreground** (vehicles) and **background**
- Helps create **prompts** for object detection models

### **5. ClipSeg**
- **Filters background objects** based on Qwen’s prompts
- Enhances **segmentation accuracy**
- Ensures only **foreground objects** are detected

---

## **Example Pipeline Flow**
1️⃣ **Extract frames from the video**  
2️⃣ **Use Qwen to classify objects as foreground/background**  
3️⃣ **Apply ClipSeg to remove background noise**  
4️⃣ **Detect objects using Grounding DINO + YOLO-World**  
5️⃣ **Generate final segmented masks with SAM2**  

---

## **Troubleshooting**
If you encounter issues:
1. **Make sure all models are downloaded correctly**
2. **Check if CUDA is available** (`torch.cuda.is_available()`)
3. **Ensure `requirements.txt` dependencies are installed**
4. **Verify API access for Qwen (Hugging Face / OpenAI alternative)**

---

## **Acknowledgements**
- [Segment Anything Model 2 (SAM2)](https://github.com/IDEA-Research/Grounded-SAM-2)
- [Grounding DINO](https://github.com/IDEA-Research/Grounded-SAM-2?tab=readme-ov-file)
- [YOLO-World](https://docs.ultralytics.com/zh/models/yolo-world/#train-usage)
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [ClipSeg](https://github.com/timojl/clipseg/tree/master)



这份 README **涵盖了所有的模型下载、安装、初始化和运行方法**，并且提供了**清晰的 pipeline 说明**，让用户可以快速理解并运行整个项目。🚀
