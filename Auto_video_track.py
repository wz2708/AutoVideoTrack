import re
import os
import cv2
import json
import copy
import torch
import requests
import numpy as np
from PIL import Image
from torchvision.ops import nms
from torchvision import transforms

from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoModelForZeroShotObjectDetection
from qwen_vl_utils import process_vision_info

from ultralytics import YOLOWorld
from clipseg.models.clipseg import CLIPDensePredT

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo

##############################################
# 1. 帧文件处理模块
##############################################
def get_sorted_frame_names(video_path, video_dir, frame_rate=20):
    """
    从视频中按指定帧率提取帧，对每帧进行预处理，然后保存到 video_dir 中。
    返回排序后的帧文件名列表（例如 ["0000.jpg", "0001.jpg", ...]）。
    """
    os.makedirs(video_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open the video:", video_path)
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps / frame_rate))
    count, saved = 0, 0
    frame_names = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 每隔 frame_interval 帧处理一次
        if count % frame_interval == 0:
            file_name = f"{saved:04d}.jpg"
            file_path = os.path.join(video_dir, file_name)
            if cv2.imwrite(file_path, frame):
                frame_names.append(file_name)
            else:
                print("保存帧失败:", file_path)
            saved += 1
        count += 1
    cap.release()
    frame_names.sort(key=lambda x: int(os.path.splitext(x)[0]))
    return frame_names


##############################################
# 2. 模型初始化模块
##############################################
def initialize_models(sam2_checkpoint, model_cfg, grounding_model_id, yolo_model_path):
    """
    初始化 SAM2 视频预测器、SAM2 图像预测器、Grounding DINO 模型、YOLO‑World 模型、
    QWen API（用于 prompt 生成）和 ClipSeg API（用于背景过滤）。

    返回：
      video_predictor, image_predictor, processor, grounding_model, yolo_model, device, qwen_client, clipseg_model
    """
    # 设置自动混合精度及 TF32（适用于支持的 GPU）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 初始化 YOLO‑World 模型
    yolo_model = YOLOWorld(yolo_model_path).to(device)
    yolo_model.overrides['conf'] = 0.15  # 置信度阈值
    yolo_model.overrides['iou'] = 0.45  # IoU阈值
    yolo_model.overrides['agnostic_nms'] = True  # 跨类别NMS
    yolo_model.eval()  # 设置为推理模式

    # 初始化 SAM2 视频预测器和图像预测器
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    # 初始化 Grounding DINO
    processor = AutoProcessor.from_pretrained(grounding_model_id, local_files_only=True)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id, local_files_only=True).to(device)
    grounding_model.eval()  # 设置为推理模式

    # 初始化 QWen 模型
    QWEN_MODEL_PATH = "/root/autodl-tmp/GroundSAM2/qwen/Qwen/Qwen2___5-VL-7B-Instruct"
    qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_PATH, local_files_only=True)
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH, local_files_only=True)
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_PATH,
        torch_dtype=torch.float16,
        max_memory={0: "30GB", "cpu": "60GB"},
        device_map=device,
        local_files_only=True
    )
    qwen_model.eval()  # 设置为推理模式

    # 初始化 ClipSeg 模型
    clipseg_model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    clipseg_model.eval()
    model_path = 'weights/rd64-uni.pth'
    clipseg_model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')), strict=False)
    clipseg_model.to(device)

    # 打印模型设备信息
    print("QWen model on:", next(qwen_model.parameters()).device)
    print("Grounding model on:", next(grounding_model.parameters()).device)
    print("YOLO‑World model on:", next(yolo_model.parameters()).device)
    print("ClipSeg model on:", next(clipseg_model.parameters()).device)

    return (
        video_predictor, image_predictor, processor, grounding_model, yolo_model, device,
        qwen_processor, qwen_tokenizer, qwen_model, clipseg_model
    )

##############################################
# 3. 单帧处理模块（检测 + 分割）
##############################################
def compute_iou(box1, box2):
    """
    计算两个边界框的 IoU。
    box 格式：[x1, y1, x2, y2]
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    inter_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter_area / max(float(area1 + area2 - inter_area), 1e-6)
    return iou

def center_distance(box1, box2):
    """
    计算两个边界框中心点之间的欧式距离。
    box1, box2 格式：[x1, y1, x2, y2]
    """
    center_x1, center_y1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    center_x2, center_y2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    return ((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2) ** 0.5

def nms(boxes, scores, iou_threshold=0.5):
    """
    非极大值抑制（NMS）。
    boxes: [N, 4], 格式为 [x1, y1, x2, y2]
    scores: [N], 置信度
    iou_threshold: IoU 阈值
    """
    if len(boxes) == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    # 按置信度降序排序
    sorted_indices = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_indices]
    scores = scores[sorted_indices]

    keep_indices = []
    while len(boxes) > 0:
        # 保留当前最高置信度的框
        keep_indices.append(sorted_indices[0].item())
        if len(boxes) == 1:
            break
        # 计算当前框与其他框的 IoU
        ious = torch.tensor([compute_iou(boxes[0].tolist(), box.tolist()) for box in boxes[1:]], device=boxes.device)
        # 保留 IoU 低于阈值的框
        keep = ious < iou_threshold
        boxes = boxes[1:][keep]
        scores = scores[1:][keep]
        sorted_indices = sorted_indices[1:][keep]

    return torch.tensor(keep_indices, dtype=torch.long, device=boxes.device)

# 提取 JSON 数据
def extract_json(response):
    """
    从模型响应中提取 JSON 数据。
    """
    try:
        # 使用正则表达式匹配 JSON 字符串
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            output = json.loads(json_str)
            foreground_objects = output.get("foreground_objects", [])
            background_objects = output.get("background_objects", [])
            return foreground_objects, background_objects
        else:
            raise ValueError("No JSON found in response")
    except Exception as e:
        print(f"解析错误: {str(e)}")
        print("原始响应:", response)
        return [], []  # 返回空列表

def extract_vehicle_boxes_and_scores(yolo_results, device, conf_threshold=0.1):
    """
    从 YOLO‑World 的检测结果中提取车辆检测框和置信度。
    假设 yolo_results 为一个列表，其中每个元素具备 .boxes 属性，
    且 .boxes.xyxy 为 [N, 4] 的 Tensor，.boxes.conf 为置信度。
    返回：
      - yolo_boxes: Tensor，[N, 4]，格式为 [x1, y1, x2, y2]
      - yolo_scores: Tensor，[N,]，检测置信度
    """
    boxes_list = []
    scores_list = []
    for result in yolo_results:
        boxes_obj = result.boxes  # Boxes 对象
        boxes_tensor = boxes_obj.xyxy  # Tensor, 格式 [x1, y1, x2, y2]
        if hasattr(boxes_obj, "conf") and boxes_obj.conf is not None:
            scores_tensor = boxes_obj.conf
        elif hasattr(boxes_obj, "probs") and boxes_obj.probs is not None:
            scores_tensor = boxes_obj.probs
        else:
            scores_tensor = torch.ones(boxes_tensor.shape[0], dtype=torch.float32, device=boxes_tensor.device)

        # 过滤低置信度检测
        keep = scores_tensor >= conf_threshold
        if keep.sum() > 0:
            boxes_list.append(boxes_tensor[keep].cpu().numpy())
            scores_list.append(scores_tensor[keep].cpu().numpy())

    if boxes_list:
        yolo_boxes = torch.tensor(np.concatenate(boxes_list, axis=0), dtype=torch.float32, device=device)
        yolo_scores = torch.tensor(np.concatenate(scores_list, axis=0), dtype=torch.float32, device=device)
        return yolo_boxes, yolo_scores
    else:
        return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)


def merge_boxes(dino_boxes, yolo_boxes, yolo_scores,
                iou_threshold=0.4,
                model_weights=(0.7, 0.3),
                size_aware=True,
                conf_threshold=0.25):
    """
    改进版多模型检测框融合策略

    参数说明：
    - dino_boxes: Grounding DINO检测框 [N,4] (xyxy格式)
    - yolo_boxes: YOLO-World检测框 [M,4]
    - yolo_scores: YOLO-World置信度 [M]
    - iou_threshold: 动态调整的NMS阈值
    - model_weights: (dino_weight, yolo_weight) 模型权重
    - size_aware: 是否启用尺寸感知加权
    - conf_threshold: 最终置信度过滤阈值
    """
    device = dino_boxes.device

    # 1. 模型权重分配（DINO精度优先，YOLO召回优先）
    dino_weight, yolo_weight = model_weights

    # 为DINO生成自适应置信度（基于框尺寸）
    if dino_boxes.shape[0] > 0:
        if size_aware:
            # 尺寸感知权重：小目标权重更高
            box_areas = (dino_boxes[:, 2] - dino_boxes[:, 0]) * (dino_boxes[:, 3] - dino_boxes[:, 1])
            min_area, max_area = box_areas.min(), box_areas.max()
            norm_areas = (box_areas - min_area) / (max_area - min_area + 1e-6)
            dino_scores = 0.5 + 0.5 * norm_areas  # 基础置信度0.5~1.0
            dino_scores *= dino_weight  # 应用模型权重
        else:
            dino_scores = torch.full((len(dino_boxes),), dino_weight, device=device)
    else:
        dino_scores = torch.empty((0,), device=device)

    # 对YOLO置信度应用权重
    yolo_scores = yolo_scores * yolo_weight

    # 2. 合并检测结果
    all_boxes = torch.cat([dino_boxes, yolo_boxes], dim=0)
    all_scores = torch.cat([dino_scores, yolo_scores], dim=0)

    # 3. 动态调整NMS阈值（检测目标越多阈值越严格）
    active_iou_thresh = max(0.3, iou_threshold - 0.05 * (len(all_boxes) // 5))

    # 4. 执行NMS
    keep_indices = nms(all_boxes, all_scores, active_iou_thresh)
    filtered_boxes = all_boxes[keep_indices]
    filtered_scores = all_scores[keep_indices]

    # 5. 自适应框融合（尺寸差异小时进行融合）
    final_boxes = []
    final_scores = []
    used = torch.zeros(len(filtered_boxes), dtype=torch.bool, device=device)

    for i in range(len(filtered_boxes)):
        if used[i]:
            continue

        current_box = filtered_boxes[i]
        current_score = filtered_scores[i]
        merge_group = [current_box]
        score_group = [current_score]
        used[i] = True

        # 寻找可合并的相邻框
        for j in range(i + 1, len(filtered_boxes)):
            if used[j]:
                continue

            # 双条件匹配策略
            j_box = filtered_boxes[j]
            iou = compute_iou(current_box, j_box)
            center_dist = center_distance(current_box, j_box)
            size_ratio = (j_box[2] - j_box[0]) / (current_box[2] - current_box[0] + 1e-6)

            if (iou > 0.3) or (center_dist < 30 and 0.5 < size_ratio < 2.0):
                merge_group.append(j_box)
                score_group.append(filtered_scores[j])
                used[j] = True

        # 加权融合（考虑置信度和面积）
        merged_box = torch.stack(merge_group)
        weights = torch.stack(score_group) * (merged_box[:, 2] - merged_box[:, 0]) * (
                    merged_box[:, 3] - merged_box[:, 1])
        weights /= weights.sum()
        final_box = (merged_box * weights[:, None]).sum(dim=0)

        # 分数融合（取最高置信度）
        final_score = max(score_group)

        final_boxes.append(final_box)
        final_scores.append(final_score)

    if len(final_boxes) == 0:
        return torch.empty((0, 4), device=device), torch.empty((0,), device=device)

    final_boxes = torch.stack(final_boxes)
    final_scores = torch.tensor(final_scores, device=device)

    # 6. 置信度过滤与尺寸过滤
    valid_mask = (final_scores >= conf_threshold) & \
                 ((final_boxes[:, 2] - final_boxes[:, 0]) * (final_boxes[:, 3] - final_boxes[:, 1]) > 100)  # 最小面积

    return final_boxes[valid_mask], final_scores[valid_mask]

def analyze_frame_with_qwen(processor, tokenizer, model, image):
    system_prompt = (
        "You are a traffic surveillance analyst. For the given image,"
        "your task is to identify all objects and classify them into two categories: "
        "foreground objects (vehicles like cars, trucks, buses, motorcycles, etc.) "
        "and background objects (all other elements such as buildings, trees, roads, sky, etc.). "
        "For each object, provide a concise description in the format 'adjective noun' (e.g., 'red car', 'blue truck', 'tall building'). "
        "Return the result as a JSON object with keys 'foreground_objects' and 'background_objects'."
    )
    user_prompt = (
        "Please analyze the image and output a JSON object with the keys 'foreground_objects' and 'background_objects'. "
        "Each value should be a list of 'adjective noun' descriptions. "
        "Ensure that vehicles are listed under 'foreground_objects' and all non-vehicle elements under 'background_objects'."
    )
    # 构造多模态输入
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt},
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt}
            ]
        }
    ]

    # 使用 processor 构造对话模板
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    # 构造模型输入
    inputs = processor(text=[text], images=[image_inputs], padding=True, return_tensors="pt").to(model.device)

    # 生成响应
    generated_ids = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)

    # 解码模型输出
    response = tokenizer.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    foreground_objects, background_objects = extract_json(response)

    return foreground_objects, background_objects


def filter_background_with_clipseg(model, image, prompt, device="cuda"):
    """
    改进版背景分割函数，修复OpenCV维度问题
    返回：
      - background_mask: 二值掩码 [H,W], 值0(前景)/1(背景)
      - filtered_image: PIL图像（可选）
      - useful_info: 调试信息字典
    """
    # 1. 输入预处理
    original_size = image.size
    image_np = np.array(image) if isinstance(image, Image.Image) else image

    # 2. 多尺度处理
    scales = [0.8, 1.0, 1.2]
    all_masks = []

    for scale in scales:
        scaled_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        scaled_img = image.resize(scaled_size) if scale != 1.0 else image

        # 2.1 图像转tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(scaled_img).unsqueeze(0).to(device)  # [1,3,H,W]

        # 2.2 多提示词处理
        with torch.no_grad():
            preds = []
        for p in prompt:
        # 生成单提示词掩码
            pred = model(img_tensor, [p])[0]  # [1,1,H,W]
        pred = torch.sigmoid(pred)

        # 提示词权重调整
        if p == "road":
            pred *= 1.5
        elif p in ["sky", "vegetation"]:
            pred *= 0.8
        preds.append(pred)

        # 多提示词加权平均
        combined = torch.stack(preds).mean(dim=0)  # [1,1,H,W]

        # 调整回原图尺寸
        combined = transforms.functional.resize(
            combined,
            original_size[::-1],
            interpolation=transforms.InterpolationMode.BILINEAR
        )
        all_masks.append(combined.squeeze())  # 移除批和通道维度 → [H,W]

        # 3. 多尺度融合（取最大值）
        final_mask = torch.stack(all_masks).max(dim=0)[0]  # [H,W]

        # 4. 动态阈值二值化
        adaptive_thresh = 0.4 + 0.2 * (1 - final_mask.mean().item())
        bg_mask = (final_mask > adaptive_thresh).float()  # [H,W], 0/1

        # 5. 形态学后处理（修复OpenCV错误的关键部分）
        # 5.1 转换到CPU和numpy
        bg_np = bg_mask.cpu().numpy()

        # 5.2 确保二维且类型正确
        assert bg_np.ndim == 2, f"掩码应为二维，实际维度：{bg_np.ndim}"
        bg_uint8 = np.where(bg_np > 0.5, 255, 0).astype(np.uint8)

        # 5.3 连通区域分析去除小噪声
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bg_uint8)
        min_area = image.width * image.height * 0.005  # 至少占0.5%面积
        cleaned_mask = np.zeros_like(bg_uint8)
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned_mask[labels == i] = 255

        # 5.4 光流平滑（视频场景）
        if hasattr(robust_clipseg_mask, "prev_mask"):
            # 使用光流传播前一帧结果
            flow = cv2.calcOpticalFlowFarneback(
                robust_clipseg_mask.prev_mask.astype(np.uint8),
                cleaned_mask,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            warped_mask = cv2.remap(
                robust_clipseg_mask.prev_mask,
                flow, None, cv2.INTER_LINEAR
            )
            cleaned_mask = cv2.addWeighted(cleaned_mask, 0.7, warped_mask, 0.3, 0)
        robust_clipseg_mask.prev_mask = cleaned_mask  # 缓存当前帧

        # 转换为Tensor
        final_mask = torch.tensor(cleaned_mask / 255.0, device=device)
        return final_mask


def adaptive_box_filter(boxes, scores, bg_mask,
                        min_foreground_ratio=0.25,
                        min_box_area=300):
    """
    自适应检测框过滤：
    1. 动态前景比例阈值
    2. 面积过滤小目标
    3. 置信度加权
    """
    valid_boxes = []
    valid_scores = []
    bg_np = bg_mask.cpu().numpy()

    # 根据背景复杂度调整阈值
    bg_complexity = bg_mask.mean().item()
    dynamic_threshold = max(0.2, min_foreground_ratio - bg_complexity * 0.1)

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box.tolist())
        h, w = bg_np.shape

        # 边界保护
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x1 >= x2 or y1 >= y2:
            continue

        # 计算有效区域
        box_area = (x2 - x1) * (y2 - y1)
        if box_area < min_box_area:
            continue  # 过滤极小框

        # 计算前景覆盖率
        foreground_pixels = np.sum(1 - bg_np[y1:y2, x1:x2])
        coverage = foreground_pixels / box_area

        # 动态阈值过滤（复杂背景放宽条件）
        if coverage >= dynamic_threshold or score > 0.3:  # 高置信度框保留
            valid_boxes.append(box)
            valid_scores.append(score)

    return torch.stack(valid_boxes) if valid_boxes else torch.empty((0, 4), device=boxes.device), \
           torch.tensor(valid_scores, device=scores.device) if valid_scores else torch.empty((0,), device=scores.device)

def filter_boxes_by_mask(boxes, scores, mask, min_coverage=0.6):
    """
    同步过滤boxes和对应的scores
    返回：
        filtered_boxes: [N,4] tensor
        filtered_scores: [N,] tensor
    """
    assert len(boxes) == len(scores), "Boxes and scores must have same length"

    mask_np = mask.cpu().numpy() if mask.is_cuda else mask.numpy()
    valid_boxes = []
    valid_scores = []

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box.tolist())
        # 边界保护
        h, w = mask_np.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if x1 >= x2 or y1 >= y2:
            continue

        box_area = (x2 - x1) * (y2 - y1)
        if box_area == 0:
            continue

        # 计算背景覆盖率
        bg_pixels = mask_np[y1:y2, x1:x2].sum()
        coverage = bg_pixels / box_area
        if coverage < min_coverage:
            valid_boxes.append(box)
            valid_scores.append(score)

    if valid_boxes:
        return (
            torch.stack(valid_boxes).to(boxes.device),
            torch.stack(valid_scores).to(scores.device))
    else:
        return (
            torch.empty((0, 4), device=boxes.device),
            torch.empty((0,), device=scores.device))


def process_frame(frame_path, processor, grounding_model, yolo_model, PROMPT_TYPE_FOR_VIDEO, image_predictor, device, global_mask_dict, qwen_processor, qwen_tokenizer, qwen_model, clipseg_model):
    """
    处理单帧：
      1. 用 QWen 生成 prompt
      2. 用 ClipSeg 过滤背景
      3. 用 Grounding DINO 和 YOLO-World 进行目标检测
      4. 用 SAM2 生成最终 mask
    """
    # 1. 加载图像
    image = Image.open(frame_path).convert("RGB")

    # 2. 使用 QWen 获取 Prompt
    foreground_objects, background_objects = analyze_frame_with_qwen(qwen_processor, qwen_tokenizer, qwen_model, image)
    print("Foreground Objects:", foreground_objects)
    print("Background Objects:", background_objects)
    # 如果 QWen 未生成有效 Prompt，使用默认值
    if not foreground_objects:
        foreground_objects = ["vehicle", "car", "truck", "bus"]  # 默认值
    if not background_objects:
        background_objects = ["road", "tree", "sky", "building"]  # 默认背景


    # # 3. 用 ClipSeg 过滤背景
    # background_mask = filter_background_with_clipseg(clipseg_model, image, background_objects)

    # 4. 目标检测
    # 4.1 Grounding DINO 检测
    inputs = processor(images=image, text=foreground_objects, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.22, # 阈值
        text_threshold=0.22,# 阈值
        target_sizes=[image.size[::-1]]
    )
    dino_boxes = results[0]["boxes"]
    OBJECTS = results[0]["labels"]

    # 4.2 YOLO-World 检测
    image_np = np.array(image)
    yolo_model.set_classes(foreground_objects)  # 直接使用 foreground_objects
    yolo_results = yolo_model.predict(image_np,
                                      conf=0.12,          # 置信度阈值
                                      iou=0.25,           # IoU阈值
                                      augment=True,       # 启用TTA
                                      verbose=False,
                                      agnostic_nms=True   # 跨类别NMS
                                      )
    yolo_boxes, yolo_scores = extract_vehicle_boxes_and_scores(yolo_results, device=device, conf_threshold=0.15)
    print("Grounding dino and YOLO-World",len(dino_boxes), len(yolo_boxes))

    # # 5. 结合前景 Mask 进行筛选
    # dino_boxes, dino_scores = adaptive_box_filter(
    #     dino_boxes,
    #     torch.ones(len(dino_boxes), device=device),  # DINO默认分数为1
    #     background_mask
    # )
    # yolo_boxes, yolo_scores = adaptive_box_filter(
    #     yolo_boxes,
    #     yolo_scores,
    #     background_mask
    # )
    # print("Grounding dino and YOLO-World after deleting background",len(dino_boxes), len(yolo_boxes))

    # 6. 合并检测结果
    if dino_boxes.shape[0] == 0 and yolo_boxes.shape[0] > 0:
        combined_boxes = yolo_boxes
    elif dino_boxes.shape[0] > 0 and yolo_boxes.shape[0] == 0:
        combined_boxes = dino_boxes
    elif dino_boxes.shape[0] > 0 and yolo_boxes.shape[0] > 0:
        combined_boxes, combined_scores = merge_boxes(dino_boxes, yolo_boxes, yolo_scores, iou_threshold=0.4,model_weights=(0.7, 0.3),size_aware=True,conf_threshold=0.25)
    else:
        combined_boxes = dino_boxes
    print("combined_boxes",len(combined_boxes))

    # 7. 使用 SAM2 生成最终 Mask
    base_name = os.path.splitext(os.path.basename(frame_path))[0]
    mask_dict = MaskDictionaryModel(promote_type=PROMPT_TYPE_FOR_VIDEO, mask_name=f"mask_{base_name}.npy")

    image_predictor.set_image(np.array(image))
    if combined_boxes.shape[0] != 0:
        masks, _, _ = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=combined_boxes,
            multimask_output=False,
        )

        if masks.ndim == 2:
            masks = masks[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)

        mask_dict.add_new_frame_annotation(
            mask_list=torch.tensor(masks).to(device),
            box_list=combined_boxes,
            label_list=OBJECTS
        )
    else:
        print(f"Frame {base_name}: No object detected, skipping frame.")
        mask_dict = global_mask_dict

    return mask_dict

##############################################
# 4. 视频跟踪模块（正向跟踪）
##############################################
def track_video_frames(frame_names, frames_dir, step, video_predictor, qwen_processor, qwen_tokenizer, qwen_model,  clipseg_model, inference_state, sam2_masks,
                       processor, grounding_model, PROMPT_TYPE_FOR_VIDEO, image_predictor, yolo_model, mask_data_dir,json_data_dir, device):
    """
    对采样帧进行正向跟踪，返回 video_segments（字典：帧索引 -> MaskDictionaryModel）和 frame_object_count。

    参数：
      frame_names: 帧文件名列表（例如 ["0000.jpg", "0001.jpg", ...]）
      frames_dir: 帧图片所在目录（完整路径）
      step: 采样间隔
      video_predictor: SAM2 视频预测器对象
      inference_state: 视频预测器状态（由 init_state 返回）
      sam2_masks: 初始 MaskDictionaryModel 对象
      processor, grounding_model, image_predictor, text, device: 检测和分割所需模型和参数
    """
    objects_count = 0
    frame_object_count = {}
    video_segments = {}
    for start_idx in range(0, len(frame_names), step):
        current_frame = os.path.join(frames_dir, frame_names[start_idx])
        mask_dict = process_frame(current_frame, processor, grounding_model, yolo_model,PROMPT_TYPE_FOR_VIDEO, image_predictor, device, sam2_masks, qwen_processor, qwen_tokenizer, qwen_model,  clipseg_model)

        objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8,objects_count=objects_count)
        frame_object_count[start_idx] = objects_count

        if len(mask_dict.labels) == 0:
            # 如果当前帧没有检测到目标，则保存空的结果（此处依赖于你 MaskDictionaryModel 的实现）
            mask_dict.save_empty_mask_and_json(mask_data_dir,json_data_dir, image_name_list=frame_names[start_idx:start_idx + step])
            print("Frame:{} No object detected in the frame, skip the frame".format(start_idx))
            continue
        else:
            video_predictor.reset_state(inference_state)

            for object_id, object_info in mask_dict.labels.items():
                frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(inference_state, start_idx, object_id, object_info.mask)

            # 利用视频预测器传播
            for out_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_idx):
                frame_masks = MaskDictionaryModel()
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0)
                    object_info = ObjectInfo(instance_id=out_obj_id,mask=out_mask[0],class_name=mask_dict.get_target_class_name(out_obj_id),logit=mask_dict.get_target_logit(out_obj_id))
                    object_info.update_box()
                    frame_masks.labels[out_obj_id] = object_info
                    base_name = os.path.splitext(os.path.basename(frame_names[out_idx]))[0]
                    frame_masks.mask_name = f"mask_{base_name}.npy"
                    frame_masks.mask_height = out_mask.shape[-2]
                    frame_masks.mask_width = out_mask.shape[-1]

                video_segments[out_idx] = frame_masks
                sam2_masks = copy.deepcopy(frame_masks)
    return video_segments, frame_object_count


##############################################
# 5. 反向跟踪模块
##############################################
def reverse_tracking(frame_names, frame_object_count, inference_state, video_predictor, mask_data_dir, json_data_dir, step):
    """
    对已处理视频结果进行反向跟踪，修正新出现目标在之前帧中的缺失。
    这里的逻辑根据原始代码实现，注意需要保证路径正确。
    """
    start_object_id = 0
    object_info_dict = {}
    for frame_idx, current_object_count in frame_object_count.items():
        print("reverse tracking frame", frame_idx, frame_names[frame_idx])
        if frame_idx != 0:
            video_predictor.reset_state(inference_state)
            image_base_name = os.path.splitext(frame_names[frame_idx])[0]
            json_data_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
            mask_data_path = os.path.join(mask_data_dir, f"mask_{image_base_name}.npy")
            try:
                json_data = MaskDictionaryModel().from_json(json_data_path)
                mask_array = np.load(mask_data_path)
            except Exception as e:
                print("Load data failure:", e)
                continue
            for object_id in range(start_object_id + 1, current_object_count + 1):
                print("reverse tracking object", object_id)
                object_info_dict[object_id] = json_data.labels[object_id]
                video_predictor.add_new_mask(inference_state, frame_idx, object_id, mask_array == object_id)
        start_object_id = current_object_count

        try:
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
                    inference_state, max_frame_num_to_track=step * 3, start_frame_idx=frame_idx, reverse=True):
                image_base_name = os.path.splitext(frame_names[out_frame_idx])[0]
                json_data_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
                mask_data_path = os.path.join(mask_data_dir, f"mask_{image_base_name}.npy")
                try:
                    json_data = MaskDictionaryModel().from_json(json_data_path)
                    mask_array = np.load(mask_data_path)
                except Exception as e:
                    print("Load data failure:", e)
                    continue
                # 合并反向追踪的 mask 与原始 mask
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0).cpu()
                    if out_mask.sum() == 0:
                        print("no mask for object", out_obj_id, "at frame", out_frame_idx)
                        continue
                    object_info = object_info_dict.get(out_obj_id)
                    if object_info is None:
                        print(f"object_info for object {out_obj_id} not found, skipping")
                        continue

                    object_info.mask = out_mask[0]
                    object_info.update_box()
                    json_data.labels[out_obj_id] = object_info
                    mask_array = np.where(mask_array != out_obj_id, mask_array, 0)
                    mask_array[object_info.mask] = out_obj_id

                np.save(mask_data_path, mask_array)
                json_data.to_json(json_data_path)
        except RuntimeError as e:
            if "No points are provided" in str(e):
                print(f"Skipping reverse tracking for frame {frame_idx} due to missing points.")
                continue
            else:
                raise e

##############################################
# 6. 结果保存模块
##############################################
def save_tracking_results(video_segments, mask_data_dir, json_data_dir):
    """
    将视频跟踪结果保存为 mask 的 numpy 文件和 JSON 文件。
    """
    for frame_idx, frame_masks_info in video_segments.items():
        mask = frame_masks_info.labels
        mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
        for obj_id, obj_info in mask.items():
            mask_img[obj_info.mask == True] = obj_id
        mask_img = mask_img.numpy().astype(np.uint16)
        np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)
        json_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
        frame_masks_info.to_json(json_path)



##############################################
# 示例主函数调用
##############################################
if __name__ == "__main__":
    #       1. 从 frames_dir 获取所有帧文件名（要求已分帧）。
    #       2. 对每隔 step 帧进行检测、分割和正向跟踪。
    #       3. 保存生成的 mask 数据和 JSON 文件到输出目录中。
    #       4. 绘制结果并生成视频。
    #       5. 执行反向跟踪修正。
    base_output_folder = "outputs/Nexar"
    videos_folder = "test"

    forward_step = 15 # call grandingdino every 15 frames
    reverse_step = 15 # call grandingdino every 15 frames
    frame_rate = 30 # divide one second into 30 frames
    PROMPT_TYPE_FOR_VIDEO = "mask"  # box, mask or point

    video_predictor, image_predictor, processor, grounding_model, yolo_model, device, qwen_processor, qwen_tokenizer, qwen_model, clipseg_model = initialize_models(
        sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
        model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        grounding_model_id="IDEA-Research/grounding-dino-tiny",
        yolo_model_path = "./yolo_world/yolov8x-worldv2.pt"
    )
    print("Finish initialization")


    video_files_all = [f for f in os.listdir(videos_folder) if f.lower().endswith((".mp4", ".avi", ".mov"))]
    video_files = [video_files_all[i] for i in [5,6,7,8]] # 1
    for video_file in video_files:
        video_path = os.path.join(videos_folder, video_file)
        video_id = os.path.splitext(video_file)[0]
        print("Processing video:", video_id)

        # 创建该视频的输出目录结构
        output_dir = os.path.join(base_output_folder, video_id)
        frames_dir = os.path.join(output_dir, "initial_frames")
        mask_data_dir = os.path.join(output_dir, "mask_data")
        json_data_dir = os.path.join(output_dir, "json_data")
        result_dir = os.path.join(output_dir, "result")
        for d in [output_dir, frames_dir, mask_data_dir, json_data_dir, result_dir]:
            CommonUtils.creat_dirs(d)

        # 分帧：将视频分帧保存到 frames_dir，返回仅文件名列表（例如 "0000.jpg"）
        frame_names = get_sorted_frame_names(video_path, frames_dir, frame_rate)
        print("Total frames:", len(frame_names))

        # 为该视频单独初始化视频跟踪状态（inference_state）和初始 mask（为空）
        inference_state = video_predictor.init_state(video_path=frames_dir)
        initial_mask_dict = MaskDictionaryModel()

        # 正向跟踪：对采样帧进行检测与跟踪
        video_segments, frame_object_count = track_video_frames(
            frame_names, frames_dir, forward_step, video_predictor, qwen_processor, qwen_tokenizer, qwen_model,
            clipseg_model, inference_state, initial_mask_dict,
            processor, grounding_model, PROMPT_TYPE_FOR_VIDEO, image_predictor, yolo_model,
            mask_data_dir, json_data_dir, device
        )

        # 保存正向跟踪结果：mask 和 JSON 文件
        save_tracking_results(video_segments, mask_data_dir, json_data_dir)

        CommonUtils.draw_masks_and_box_with_supervision(frames_dir, mask_data_dir, json_data_dir, result_dir)
        output_video_path_reverse = os.path.join(output_dir, "output.mp4")
        create_video_from_images(result_dir, output_video_path_reverse, frame_rate=15)

        # 反向跟踪：补充之前帧中未检测到的新目标
        reverse_tracking(frame_names, frame_object_count, inference_state,
                         video_predictor, mask_data_dir, json_data_dir, reverse_step)

        reverse_result_dir = result_dir + "_reverse"
        CommonUtils.draw_masks_and_box_with_supervision(frames_dir, mask_data_dir, json_data_dir, reverse_result_dir)
        output_video_path_reverse = os.path.join(output_dir, "output_reverse.mp4")
        create_video_from_images(reverse_result_dir, output_video_path_reverse, frame_rate=15)

        print("Completed video:", video_id, "results saved in:", output_dir)

