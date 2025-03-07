import os
import cv2
import torch
import numpy as np
import copy
from torchvision.ops import nms
from PIL import Image

from ultralytics import YOLOWorld

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo

##############################################
# 1. 帧文件处理模块
##############################################
def preprocess_frame(frame, clahe_clip=2.0, clahe_tile=(8,8), gamma=1.5, low_light_threshold=100, high_light_threshold = 180):
    """
    对输入的 BGR 格式帧进行预处理：
      - 将图像转换为 YCrCb 色彩空间
      - 如果图像整体亮度较低（低于阈值），则对亮度通道应用 CLAHE 和 gamma 校正（gamma > 1）。
      - 如果图像整体亮度较高（高于阈值），则对图像应用 gamma 校正（gamma < 1）以降低过曝。
      - 将处理后的图像转换回 BGR 格式并返回

    参数：
      frame: BGR 格式的原始图像（NumPy 数组）
      clahe_clip: CLAHE 的剪切限制（默认 2.0）
      clahe_tile: CLAHE 的网格大小（默认 (8,8)）
      gamma: 伽马校正值（默认 1.5，对于暗图像可使用大于 1 的值）
      brightness_threshold: 平均亮度阈值，低于此值则认为图像较暗
    返回：
      处理后的图像（BGR 格式）
    """
    # 将 BGR 图像转换为 LAB 颜色空间，以便提取亮度信息
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    avg_l = np.mean(l)

    if avg_l < low_light_threshold:
        # 低光环境下增强细节
        clahe = cv2.createCLAHE(clahe_clip, clahe_tile)
        cl = clahe.apply(l)
        # 伽马校正：gamma 值大于 1 (例如1.5) 可增强暗区细节
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
        cl = cv2.LUT(cl, table)
        lab = cv2.merge((cl, a, b))
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    elif avg_l > high_light_threshold:
        # 高光环境下抑制过曝：gamma 值小于 1 (例如0.8) 降低整体亮度
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
        processed = cv2.LUT(frame, table)
    else:
        # 其他情况保持原图
        processed = frame
    return processed


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
            # 在保存前对帧进行预处理（预处理内部会根据亮度判断是否处理）
            processed_frame = preprocess_frame(frame)
            file_name = f"{saved:04d}.jpg"
            file_path = os.path.join(video_dir, file_name)
            if cv2.imwrite(file_path, processed_frame):
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
    初始化 SAM2 视频预测器、SAM2 图像预测器、Grounding DINO 模型以及 YOLO‑World 模型，
    返回包含各模型对象及相关参数的元组。

    返回的元组依次为：
      video_predictor, image_predictor, processor, grounding_model, yolo_model, device
    """
    # 设置自动混合精度及 TF32（适用于支持的 GPU）
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化 YOLO‑World 模型
    yolo_model = YOLOWorld(yolo_model_path)
    # Define custom classes
    yolo_model.set_classes(["car", "vehicle", "truck", "bus"])
    # 如果 YOLOWorld 支持 .to(device)，则移动到相同设备（注意：根据库版本可能不需要这一步）
    try:
        yolo_model.to(device)
    except Exception:
        pass  # 如果 YOLOWorld API 没有 .to 方法，则默认模型已经在合适设备上

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    processor = AutoProcessor.from_pretrained(grounding_model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(device)

    print("Models initialized on device:", device)
    return video_predictor, image_predictor, processor, grounding_model, yolo_model, device


##############################################
# 3. 单帧处理模块（检测 + 分割）
##############################################
def compute_iou(box1, box2):
    """
    计算两个边界框的 IoU
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
    iou = inter_area / float(area1 + area2 - inter_area)
    return iou


def extract_vehicle_boxes_and_masks(yolo_results, conf_threshold=0.3):
    """
    从 YOLO‑World 的检测结果中提取车辆检测框、mask 和置信度。
    假设 yolo_results 为一个列表，其中每个元素具备 .boxes, .masks 和 .probs 属性。
    返回：
      - yolo_boxes: Tensor，[N, 4]，格式为 [x1, y1, x2, y2]
      - yolo_masks: Tensor，[N, H, W]，二值化后的 mask（如果模型没有 mask，则返回占位 mask）
      - yolo_scores: Tensor，[N,]，检测置信度
    """
    boxes_list = []
    masks_list = []
    scores_list = []
    for result in yolo_results:
        # 提取边界框和置信度
        boxes = result.boxes  # 假设为 Tensor，[N,4]
        scores = result.probs  # Tensor，[N,]

        # 判断 masks 是否存在
        if hasattr(result, "masks") and result.masks is not None:
            masks = result.masks  # 假设为 Tensor，[N, H, W] 或类似格式
            masks = (masks > 0.5).float()  # 二值化
        else:
            # 如果没有 mask，则构造一个占位 mask，形状为 [N, 1, 1]
            masks = torch.zeros((boxes.shape[0], 1, 1), dtype=torch.float32)
            print("no masks")

        # 转换为 NumPy 数组（后续统一转换为 Tensor）
        boxes_np = boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else boxes
        scores_np = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores
        masks_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks

        # 过滤低置信度检测
        keep = scores_np >= conf_threshold
        if keep.sum() > 0:
            boxes_list.append(boxes_np[keep])
            scores_list.append(scores_np[keep])
            masks_list.append(masks_np[keep])
    if boxes_list:
        yolo_boxes = torch.tensor(np.concatenate(boxes_list, axis=0), dtype=torch.float32)
        yolo_scores = torch.tensor(np.concatenate(scores_list, axis=0), dtype=torch.float32)
        yolo_masks = torch.tensor(np.concatenate(masks_list, axis=0), dtype=torch.float32)
        return yolo_boxes, yolo_masks, yolo_scores
    else:
        return (torch.empty((0, 4), dtype=torch.float32),
                torch.empty((0, 1, 1), dtype=torch.float32),
                torch.empty((0,), dtype=torch.float32))


def merge_boxes_and_masks(dino_boxes, dino_masks, yolo_boxes, yolo_masks, yolo_scores, iou_threshold=0.5,
                          mask_iou_threshold=0.7):
    """
    合并 Grounding DINO 和 YOLO‑World 的检测结果，不仅基于边界框 IoU，还结合 mask IoU 进行进一步筛选。
    参数：
      dino_boxes: Tensor，[N, 4]，格式 [x1, y1, x2, y2]
      dino_masks: Tensor，[N, H, W]，若 Grounding DINO 没有 mask，可设为 None
      yolo_boxes: Tensor，[M, 4]，格式 [x1, y1, x2, y2]
      yolo_masks: Tensor，[M, H, W]，二值 mask
      yolo_scores: Tensor，[M,]，置信度分数
      iou_threshold: 边界框合并的 IoU 阈值
      mask_iou_threshold: mask IoU 的阈值，用于判断重叠程度
    返回：
      combined_boxes: Tensor，[K, 4]，合并后的边界框
      combined_masks: Tensor，[K, H, W]，合并后的二值 mask
      combined_scores: Tensor，[K,]，合并后的分数（取较高的置信度）
    """
    device = dino_boxes.device if isinstance(dino_boxes, torch.Tensor) else torch.device("cuda")
    # 将 YOLO 的结果已为 Tensor 格式，确保 dino_boxes, dino_masks 均为 Tensor
    # 若 dino_masks 为空，则用一个全1 mask（后续由 SAM2 生成最终 mask）
    if dino_masks is None or dino_masks.shape[0] == 0:
        # 此处假设每个框 mask 尺寸与目标图像尺寸一致，此处简单使用全1
        dino_masks = torch.ones((dino_boxes.shape[0], 720, 1280), dtype=torch.float32, device=device)

    # 拼接两组边界框与 mask
    all_boxes = torch.cat([dino_boxes, yolo_boxes], dim=0)
    # 对于分数，DINO 设为1.0
    dino_scores = torch.ones((dino_boxes.shape[0],), dtype=torch.float32, device=device)
    all_scores = torch.cat([dino_scores, yolo_scores], dim=0)
    # 拼接 mask
    all_masks = torch.cat([dino_masks, yolo_masks], dim=0)

    # 先使用边界框进行 NMS 过滤
    keep_indices = nms(all_boxes, all_scores, iou_threshold)
    nms_boxes = all_boxes[keep_indices]
    nms_scores = all_scores[keep_indices]
    nms_masks = all_masks[keep_indices]

    # 对于相邻框，再利用 mask IoU 进行更精细的筛选：
    final_indices = []
    used = torch.zeros(len(nms_boxes), dtype=torch.bool, device=device)
    for i in range(len(nms_boxes)):
        if used[i]:
            continue
        current_idx = i
        current_box = nms_boxes[i]
        current_mask = nms_masks[i]
        current_score = nms_scores[i]
        merge_idxs = [i]
        for j in range(i + 1, len(nms_boxes)):
            if used[j]:
                continue
            candidate_box = nms_boxes[j]
            candidate_mask = nms_masks[j]
            # 计算 box IoU
            box_iou = compute_iou(current_box.tolist(), candidate_box.tolist())
            if box_iou > iou_threshold:
                # 计算 mask IoU
                intersection = torch.logical_and(current_mask > 0, candidate_mask > 0).float().sum()
                union = torch.logical_or(current_mask > 0, candidate_mask > 0).float().sum()
                mask_iou = (intersection / union).item() if union > 0 else 0.0
                if mask_iou > mask_iou_threshold:
                    merge_idxs.append(j)
                    used[j] = True
        # 这里简单地选择 merge_idxs 中最高分的索引作为最终结果
        best_idx = merge_idxs[np.argmax(nms_scores[merge_idxs]).item()]
        final_indices.append(best_idx)

    final_boxes = nms_boxes[final_indices]
    final_scores = nms_scores[final_indices]
    final_masks = nms_masks[final_indices]

    return final_boxes, final_masks, final_scores


def process_frame(frame_path, processor, grounding_model, yolo_model, PROMPT_TYPE_FOR_VIDEO, image_predictor, text, device, sam2_masks):
    """
    对单个帧进行目标检测和分割处理，流程如下：
      1. 用 Grounding DINO 检测并获得边界框 dino_boxes（及 mask，若有）。
      2. 用 YOLO‑World 对同一帧检测，获得边界框、mask 和置信度 yolo_boxes, yolo_masks, yolo_scores。
      3. 调用 merge_boxes_and_masks() 将两者检测结果进行融合，
         融合策略包括：先进行边界框 NMS 过滤，再结合 mask IoU 进行精细合并。
      4. 将融合后的检测框传入 SAM2 图像预测器，获得最终的分割掩码（SAM2 将基于融合的边界框生成 mask）。
      5. 将最终边界框、生成的 mask 和标签注册到 MaskDictionaryModel 中返回。
    """
    image = Image.open(frame_path).convert("RGB")

    # 1. Grounding DINO
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.25,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )
    dino_boxes = results[0]["boxes"]
    OBJECTS = results[0]["labels"]

    image_predictor.set_image(np.array(image))
    dino_masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=dino_boxes,
        multimask_output=False,
    )

    # 2. YOLO‑World
    image_np = np.array(image)
    yolo_results = yolo_model.predict(image_np)
    yolo_boxes, yolo_masks, yolo_scores = extract_vehicle_boxes_and_masks(yolo_results, conf_threshold=0.3)

    # 3. 合并检测结果
    if dino_boxes.shape[0] == 0 and yolo_boxes.shape[0] > 0:
        combined_boxes = yolo_boxes
        combined_masks = yolo_masks
        combined_scores = yolo_scores
    elif dino_boxes.shape[0] > 0 and yolo_boxes.shape[0] == 0:
        combined_boxes = dino_boxes
        combined_masks = dino_masks
        combined_scores = torch.ones((dino_boxes.shape[0],), dtype=torch.float32, device=device)
    elif dino_boxes.shape[0] > 0 and yolo_boxes.shape[0] > 0:
        combined_boxes, combined_masks, combined_scores = merge_boxes_and_masks(dino_boxes, dino_masks, yolo_boxes,
                                                                                yolo_masks, yolo_scores,
                                                                                iou_threshold=0.5)
    else:
        combined_boxes = dino_boxes
        combined_masks = dino_masks
        combined_scores = torch.ones((dino_boxes.shape[0],), dtype=torch.float32, device=device)

    base_name = os.path.splitext(os.path.basename(frame_path))[0]
    mask_dict = MaskDictionaryModel(promote_type=PROMPT_TYPE_FOR_VIDEO, mask_name=f"mask_{base_name}.npy")

    # 4. 使用 SAM2 生成最终的分割 mask

    if combined_boxes.shape[0] != 0:
        # prompt SAM 2 image predictor to get the mask for the object
        masks, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=combined_boxes,
            multimask_output=False,
        )
        # convert the mask shape to (n, H, W)
        if masks.ndim == 2:
            masks = masks[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)

        # Register each object's positive points to video predictor
        mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device),
                                           box_list=combined_boxes,
                                           label_list=OBJECTS)
    else:
        print("Frame:{} No object detected in the frame, skip merge the frame merge".format(base_name))
        mask_dict = sam2_masks

    return mask_dict



def compute_box_center(box):
    # box: [x1, y1, x2, y2]
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    return x_center, y_center

def center_distance(box1, box2):
    c1 = compute_box_center(box1)
    c2 = compute_box_center(box2)
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

def match_with_global_memory(current_mask_dict, global_mask_dict,
                               iou_thresh=0.5, center_thresh=20):
    """
    对当前帧检测结果（current_mask_dict.labels）中的每个目标，与全局记忆（global_mask_dict.labels）做匹配。
    若两者的边界框 IoU 大于 iou_thresh 或中心点距离小于 center_thresh，则认为匹配，
    并将当前目标的 instance_id 更新为全局记忆中的对应 ID。
    """
    # 对于当前帧中的每个目标
    for curr_id, curr_obj in current_mask_dict.labels.items():
        matched = False
        # 若全局记忆为空，直接跳过
        if not global_mask_dict.labels:
            continue
        # 对比全局每个目标
        for glob_id, glob_obj in global_mask_dict.labels.items():
            # 计算边界框 IoU（这里采用 MaskDictionaryModel 内部可能已有的方式或自己写）
            # 这里假设 ObjectInfo 中有 box 属性 [x1, y1, x2, y2]
            # 为简单起见，我们直接计算 IoU（也可以使用 MaskDictionaryModel.calculate_iou 对应 mask）
            iou_val = compute_iou(curr_obj.box, glob_obj.box)
            dist = center_distance(curr_obj.box, glob_obj.box)
            if iou_val >= iou_thresh or dist < center_thresh:
                # 匹配成功：更新当前目标ID为全局ID，并可考虑融合mask
                curr_obj.instance_id = glob_id
                # 可根据需要对 mask 做更新（例如：采用交集或者简单覆盖）
                matched = True
                break
        # 若没有匹配到任何全局目标，则保持当前目标的 ID（或由后续逻辑分配新ID）
        if not matched:
            # 当前目标保持原ID（后续 update_global_memory 会统一处理新ID）
            pass
    return current_mask_dict

def update_global_memory(global_mask_dict, current_mask_dict):
    """
    将当前帧经过匹配处理后的检测结果合并到全局记忆中。
    对于当前帧中与全局已有目标匹配的，更新其信息；
    对于没有匹配到的目标，分配新的 ID 并添加到全局记忆中。
    """
    # 获取当前全局中最大的 ID
    max_id = max(global_mask_dict.labels.keys(), default=0)
    for curr_id, curr_obj in current_mask_dict.labels.items():
        if curr_obj.instance_id in global_mask_dict.labels:
            # 更新已有目标（这里直接覆盖，也可根据实际策略进行融合）
            global_mask_dict.labels[curr_obj.instance_id] = curr_obj
        else:
            max_id += 1
            curr_obj.instance_id = max_id
            global_mask_dict.labels[max_id] = curr_obj
    return global_mask_dict

##############################################
# 4. 视频跟踪模块（正向跟踪）
##############################################
def track_video_frames(frame_names, frames_dir, step, video_predictor, inference_state, global_mask_dict,
                       processor, grounding_model, PROMPT_TYPE_FOR_VIDEO, image_predictor, yolo_model, mask_data_dir,json_data_dir,text, device):
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
        # 处理单帧：同时调用 DINO 和 YOLO‑World 检测，并融合结果，获得 mask_dict
        mask_dict = process_frame(current_frame, processor, grounding_model, yolo_model, PROMPT_TYPE_FOR_VIDEO, image_predictor, text, device, global_mask_dict)
        # 将当前帧的检测结果与全局记忆进行匹配，避免重复分配
        mask_dict = match_with_global_memory(mask_dict, global_mask_dict, iou_thresh=0.5, center_thresh=20)
        # 更新全局记忆
        global_mask_dict = update_global_memory(global_mask_dict, mask_dict)

        objects_count = mask_dict.update_masks(tracking_annotation_dict=global_mask_dict, iou_threshold=0.8,objects_count=objects_count)
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
                global_mask_dict = copy.deepcopy(frame_masks)

    return video_segments, frame_object_count, global_mask_dict


##############################################
# 5. 反向跟踪模块
##############################################
def compute_mask_iou(mask1, mask2):
    """
    计算两个二值 mask 的 IoU。
    mask1, mask2: NumPy 二值数组
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def compute_mask_center(mask):
    """
    计算二值 mask 的中心点坐标（x, y）。
    """
    indices = np.nonzero(mask)
    if len(indices[0]) == 0:
        return None
    y_mean = np.mean(indices[0])
    x_mean = np.mean(indices[1])
    return (x_mean, y_mean)

def compute_center_distance(center1, center2):
    """
    计算两个中心点之间的欧氏距离。
    """
    if center1 is None or center2 is None:
        return float('inf')
    return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

def match_existing_object(new_mask, global_memory, iou_thresh=0.5, center_thresh=20):
    """
    尝试在 global_memory（字典，key 为 object_id，value 为 ObjectInfo 对象）中匹配与 new_mask 对应的目标。
    采用 mask IoU 和中心点距离进行匹配。返回匹配到的 object_id（如果有），否则返回 None。
    """
    new_mask_np = new_mask.cpu().numpy().astype(np.uint8) > 0
    new_center = compute_mask_center(new_mask_np)
    best_match_id = None
    best_match_score = 0.0
    for obj_id, obj_info in global_memory.items():
        # 将已存储的 mask 转为 numpy 二值数组
        existing_mask = obj_info.mask.cpu().numpy().astype(np.uint8) > 0
        iou = compute_mask_iou(new_mask_np, existing_mask)
        existing_center = compute_mask_center(existing_mask)
        center_dist = compute_center_distance(new_center, existing_center)
        # 如果 IoU 高且中心点距离小，则认为匹配
        if iou >= iou_thresh or center_dist <= center_thresh:
            # 选择 IoU 较大者作为匹配（可根据需要进一步调整策略）
            if iou > best_match_score:
                best_match_score = iou
                best_match_id = obj_id
    return best_match_id

def reverse_tracking(frame_names, frame_object_count, inference_state, video_predictor, mask_data_dir, json_data_dir, step):
    """
    对已处理视频结果进行反向跟踪，修正新出现目标在之前帧中的缺失。
    这里的逻辑根据原始代码实现，注意需要保证路径正确。
    """
    global_memory = {}  # 用于保存全局目标信息，key 为 object_id，value 为 ObjectInfo
    start_object_id = 0
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
                candidate = json_data.labels.get(object_id)
                if candidate is None:
                    continue
                # 尝试匹配全局 memory
                match_id = match_existing_object(candidate.mask, global_memory, iou_thresh=0.5, center_thresh=20)
                if match_id is not None:
                    # 更新已有目标（更新 mask、边界框等，采用加权更新策略或直接替换，视情况而定）
                    global_memory[match_id] = candidate
                else:
                    # 新目标，分配新的 global id
                    global_memory[object_id] = candidate
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
                    # 使用全局 memory 匹配
                    match_id = match_existing_object(out_mask[0], global_memory, iou_thresh=0.5, center_thresh=20)
                    if match_id is not None:
                        object_info = global_memory[match_id]
                    else:
                        # 如果没有匹配上，则尝试从当前帧中取出新目标
                        object_info = json_data.labels.get(out_obj_id)
                        if object_info is None:
                            print(f"object_info for object {out_obj_id} not found, skipping")
                            continue
                        # 同时添加到全局 memory 中
                        global_memory[out_obj_id] = object_info

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

    text = "car."

    base_output_folder = "outputs/Nexar"
    videos_folder = "TestData/test"

    forward_step = 15 # call grandingdino every 15 frames
    reverse_step = 15 # call grandingdino every 15 frames
    frame_rate = 30 # divide one second into 30 frames
    PROMPT_TYPE_FOR_VIDEO = "mask"  # box, mask or point


    video_predictor, image_predictor, processor, grounding_model, yolo_model, device = initialize_models(
        sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
        model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        grounding_model_id="IDEA-Research/grounding-dino-tiny",
        yolo_model_path = "./yolo_world/yolov8x-worldv2.pt"
    )
    print("Finish initialization")


    video_files = [f for f in os.listdir(videos_folder) if f.lower().endswith((".mp4", ".avi", ".mov"))]
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
        video_segments, frame_object_count, global_mask_dict = track_video_frames(
            frame_names, frames_dir, forward_step,
            video_predictor, inference_state, initial_mask_dict,
            processor, grounding_model, PROMPT_TYPE_FOR_VIDEO, image_predictor, yolo_model,
            mask_data_dir,json_data_dir, text, device
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

        # 可选：删除该视频的分帧目录以节省空间
        # import shutil; shutil.rmtree(frames_dir)
        print("Completed video:", video_id, "results saved in:", output_dir)

