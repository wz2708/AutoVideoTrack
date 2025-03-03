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


def extract_vehicle_boxes_and_scores(yolo_results, conf_threshold=0.3):
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
        # 直接使用 xyxy 属性
        boxes_tensor = boxes_obj.xyxy  # Tensor, 格式 [x1, y1, x2, y2]
        # 置信度
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
        yolo_boxes = torch.tensor(np.concatenate(boxes_list, axis=0), dtype=torch.float32)
        yolo_scores = torch.tensor(np.concatenate(scores_list, axis=0), dtype=torch.float32)
        return yolo_boxes, yolo_scores
    else:
        return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)


def merge_boxes(dino_boxes, yolo_boxes, yolo_scores, iou_threshold=0.5):
    """
    合并 Grounding DINO 和 YOLO‑World 的边界框，只基于边界框 IoU 进行 NMS 过滤。
    参数：
      dino_boxes: Tensor，[N, 4]（格式 [x1, y1, x2, y2]），分数统一设为1.0
      yolo_boxes: Tensor，[M, 4]（格式 [x1, y1, x2, y2]）
      yolo_scores: Tensor，[M,]，检测置信度
      iou_threshold: IoU 阈值
    返回：
      combined_boxes: Tensor，[K, 4]
      combined_scores: Tensor，[K,]
    """
    device = dino_boxes.device if isinstance(dino_boxes, torch.Tensor) else torch.device("cuda")
    yolo_boxes = yolo_boxes.to(device)
    yolo_scores = yolo_scores.to(device)
    if dino_boxes.shape[0] > 0:
        dino_scores = torch.ones((dino_boxes.shape[0],), dtype=torch.float32, device=device)
    else:
        dino_scores = torch.empty((0,), dtype=torch.float32, device=device)
    all_boxes = torch.cat([dino_boxes, yolo_boxes], dim=0)
    all_scores = torch.cat([dino_scores, yolo_scores], dim=0)
    keep_indices = nms(all_boxes, all_scores, iou_threshold)
    combined_boxes = all_boxes[keep_indices]
    combined_scores = all_scores[keep_indices]
    return combined_boxes, combined_scores

def process_frame(frame_path, processor, grounding_model, yolo_model, PROMPT_TYPE_FOR_VIDEO, image_predictor, text, device, global_mask_dict):
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


    # 2. YOLO‑World
    image_np = np.array(image)
    yolo_results = yolo_model.predict(image_np)
    yolo_boxes, yolo_scores = extract_vehicle_boxes_and_scores(yolo_results, conf_threshold=0.4)

    # 3. 合并检测结果
    if dino_boxes.shape[0] == 0 and yolo_boxes.shape[0] > 0:
        combined_boxes = yolo_boxes
        combined_scores = yolo_scores
    elif dino_boxes.shape[0] > 0 and yolo_boxes.shape[0] == 0:
        combined_boxes = dino_boxes
        combined_scores = torch.ones((dino_boxes.shape[0],), dtype=torch.float32, device=device)
    elif dino_boxes.shape[0] > 0 and yolo_boxes.shape[0] > 0:
        combined_boxes, combined_scores = merge_boxes(dino_boxes, yolo_boxes, yolo_scores, iou_threshold=0.5)
    else:
        combined_boxes = dino_boxes
        combined_scores = torch.ones((dino_boxes.shape[0],), dtype=torch.float32, device=device)

    base_name = os.path.splitext(os.path.basename(frame_path))[0]
    mask_dict = MaskDictionaryModel(promote_type=PROMPT_TYPE_FOR_VIDEO, mask_name=f"mask_{base_name}.npy")

    # 4. 使用 SAM2 生成最终的分割 mask
    image_predictor.set_image(np.array(image))
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
        mask_dict = global_mask_dict

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
    对当前帧检测结果（current_mask_dict.labels）中的每个目标，
    与全局记忆（global_mask_dict.labels）做匹配。若两者的边界框 IoU 大于 iou_thresh
    或中心点距离小于 center_thresh，则认为匹配，并将当前目标的 instance_id 更新为全局记忆中的对应 ID。
    """
    # 对于当前帧中的每个目标
    for curr_id, curr_obj in current_mask_dict.labels.items():
        matched = False
        # 若全局记忆为空，直接跳过
        if not global_mask_dict.labels:
            continue
        # 提取当前目标的边界框坐标
        curr_box = [curr_obj.x1, curr_obj.y1, curr_obj.x2, curr_obj.y2]
        for glob_id, glob_obj in global_mask_dict.labels.items():
            glob_box = [glob_obj.x1, glob_obj.y1, glob_obj.x2, glob_obj.y2]
            iou_val = compute_iou(curr_box, glob_box)
            dist = center_distance(curr_box, glob_box)
            if iou_val >= iou_thresh or dist < center_thresh:
                # 匹配成功：更新当前目标ID为全局ID
                curr_obj.instance_id = glob_id
                matched = True
                break
        # 若没有匹配到任何全局目标，则保持当前目标的 ID（后续 update_global_memory 会处理新 ID）
        if not matched:
            pass
    return current_mask_dict

def update_global_memory(global_mask_dict, current_mask_dict):
    """
    将当前帧经过匹配处理后的检测结果合并到全局记忆中。
    对于当前帧中与全局已有目标匹配的，更新其信息；对于没有匹配到的目标，分配新的 ID 并添加到全局记忆中。
    """
    # 获取当前全局中最大的 ID
    max_id = max(global_mask_dict.labels.keys(), default=0)
    for curr_id, curr_obj in current_mask_dict.labels.items():
        if curr_obj.instance_id in global_mask_dict.labels:
            # 更新已有目标（这里直接覆盖，也可采用其他融合策略）
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

