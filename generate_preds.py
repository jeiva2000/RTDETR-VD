from ultralytics import RTDETR
import os
import cv2
import torchvision.ops as ops
import torchvision.transforms as T
import torchvision.transforms.functional as F
from scipy.optimize import linear_sum_assignment
import torch
import numpy as np
from infer_onnx_rtdetrv2 import *
from losses import xyxy_to_xcycwh, xyxy_to_xywh
import onnxruntime as ort
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

queries_out = {}

def decoder_hook(module, input, output):
   boxes = output[0]
   logits = output[1]
   if torch.is_tensor(boxes) and boxes.dim() == 4:
      boxes = boxes[-1]
   if torch.is_tensor(logits) and logits.dim() == 4:
      logits = logits[-1]
   queries_out['boxes'] = boxes.detach()   # [B, num_queries, 4]
   queries_out['logits'] = logits.detach()  # [B, num_queries, num_classes]

def queries_hook(module, input, output):
   queries_out['queries'] = output[0].detach()

def logits_to_probs(logits, class_names=None):
   while logits.dim() > 3 and logits.shape[1] == 1:
      logits = logits.squeeze(1)
   if class_names is not None:
      num_classes = len(class_names)
      if logits.dim() == 3 and logits.shape[1] in (num_classes, num_classes + 1):
         logits = logits.transpose(1, 2)
      if logits.dim() == 2 and logits.shape[0] in (num_classes, num_classes + 1):
         logits = logits.transpose(0, 1)
   if logits.shape[-1] == 1:
      return torch.sigmoid(logits)
   if class_names is not None and logits.shape[-1] == len(class_names) + 1:
      probs = torch.softmax(logits, dim=-1)
      return probs[..., :-1]
   return torch.softmax(logits, dim=-1)

def draw_boxes(image, boxes, color, label=None):
   if boxes is None:
      return
   if torch.is_tensor(boxes):
      if boxes.numel() == 0:
         return
      b_np = boxes.detach().cpu().numpy()
   else:
      if len(boxes) == 0:
         return
      b_np = np.array(boxes)
   for (x1, y1, x2, y2) in b_np.tolist():
      cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
      if label:
         cv2.putText(
            image,
            label,
            (int(x1), max(int(y1) - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
         )

dino = False
if dino:
   transform = T.Compose([
      T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
      T.CenterCrop(224),
      T.ToTensor(),
      T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
      ])
   dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to('cuda').eval()

mode = 'test'
file_name = 'test_damages.npy'
root_path = '/data/backup/serperzar/mot/damages/dataset/test' #'/data/backup/serperzar/mot/MOT20/mot_dataset/test' #'/data/backup/serperzar/mot/bdd100k/bdd100k/images/track/test' #'/data/backup/serperzar/mot/bdd100k/bdd100k/images/track/train' #'/data/backup/serperzar/mot/dancetrack/mot_dataset/val' #'/data/backup/serperzar/mot/BFT/val' #'/data/backup/serperzar/mot/bdd100k/bdd100k/images/track/train'
dataset_dict = {'dancetrack':['person'], 'bdd100k':['car','sign', 'light', 'person', 'truck', 'bus', 'bike', 'rider', 'motor', 'train'], 'bft':['bird'], 'mot17':['person','car'], 'damages':['damage']}
dataset = 'damages'

if dataset == 'damages' and mode == 'test':
   root_path = '/data/backup/serperzar/mot/damages/views'
   file_name = 'preds_fp.npy'

if dataset == 'damages':
   sess = ort.InferenceSession('rtdetr_damages.onnx',providers=["CUDAExecutionProvider"])
   print("ORT device:", ort.get_device())
   print("Available providers:", ort.get_available_providers())
   print("Session providers:", sess.get_providers())
   print("Provider options:", sess.get_provider_options())
else:
   #model = RTDETR('rtdetr-l.pt')
   #model = RTDETR('best.pt')
   model = RTDETR('rtdetr_damages.pt')
   #model = RTDETR('rtdetr_bft.pt')
   model.eval()

   decoder = model.model.model[28].decoder
   decoder_handle = decoder.register_forward_hook(decoder_hook)
   queries_handle = decoder.layers[-1].register_forward_hook(queries_hook) #cuidado

limit_images = 3000
folders = os.listdir(root_path)
folders = [folder for folder in folders if folder != 'gts' and os.path.isdir(root_path+'/'+folder)]
out_dict = {}
preds_fp = {}
test_viz_dir = 'test_viz'
test_score_thresh = 0.50
if mode in ['test']:
   os.makedirs(test_viz_dir, exist_ok=True)
for folder in folders:
   writer = None
   writer_size = None
   pred_writer = None
   if mode in ['test']:
      out_path = os.path.join(test_viz_dir, f'{folder}_results.mp4')
      pred_out_path = os.path.join(test_viz_dir, f'{folder}_preds.mp4')
   images = os.listdir(os.path.join(root_path,folder))
   images = [name for name in images if name.endswith(('.jpg', '.jpeg', '.png', '.PNG'))]
   images = sorted(images)
   if limit_images > 0:
      images = images[:limit_images]

   for image in images:
      img_path = os.path.join(root_path,folder,image)
      image_cv = cv2.imread(img_path)
      if mode in ['test'] and writer is None and image_cv is not None:
         height, width, _ = image_cv.shape
         writer_size = (width, height)
         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
         writer = cv2.VideoWriter(out_path, fourcc, 30, writer_size)
         pred_writer = cv2.VideoWriter(pred_out_path, fourcc, 30, writer_size)
      height, width, c = image_cv.shape
      out_dict[img_path] = {'boxes_det':[],'queries_det':[],'boxes_det_uns':[], 'targets':[], 'scores_det':[], 'results':[]}
      txt_path = img_path.replace('.jpg','.txt').replace('.PNG','.txt')
      if not os.path.exists(txt_path):
         alt_img_path = os.path.join(root_path, 'gts', folder, image)
         alt_txt_path = alt_img_path.replace('.jpg', '.txt').replace('.PNG', '.txt')
         if os.path.exists(alt_txt_path):
            txt_path = alt_txt_path
      target = {'boxes':[],'labels':[],'ids':[]}
      #txt
      if os.path.exists(txt_path):
         with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
               if len(line.strip().split(',')) < 2:
                  continue
               cat, obj_id, cx, cy, w, h, img_w, img_h = map(float, line.strip().split(','))
               # Convertir a coordenadas en píxeles
               cx *= width
               cy *= height
               w *= width
               h *= height

               x1 = cx - (w / 2)
               y1 = cy - (h / 2)
               x2 = cx + (w / 2)
               y2 = cy + (h / 2)

               target['boxes'].append([x1, y1, x2, y2])
               target['labels'].append(int(cat))
               target['ids'].append(int(obj_id))
            target['boxes'] = torch.tensor(target['boxes'], dtype=torch.int64)
            target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
            target['ids'] = torch.tensor(target['ids'], dtype=torch.int64)
      
      if dataset == 'damages':
         out_dict[img_path]['width'] = width
         out_dict[img_path]['height'] = height
         out_dict[img_path]['boxes_det_norm'] = []
         if not torch.is_tensor(target['boxes']):
            target['boxes'] = torch.tensor([], dtype=torch.int64)
            target['labels'] = torch.tensor([], dtype=torch.int64)
            target['ids'] = torch.tensor([], dtype=torch.int64)
         print('target boxes:',target['boxes'])
         if target['boxes'].shape[0] == 0:
            out_dict[img_path]['queries_det'].append([])
            out_dict[img_path]['boxes_det'].append([])
            out_dict[img_path]['boxes_det_uns'].append([])
            out_dict[img_path]['boxes_det_norm'].append([])
         else:
            #print('targets:',target['boxes'].shape)
            labels, boxes, scores, queries = infer_rtdetr(sess,img_path)
            #print('img_path:',img_path)
            #print('scores:',scores)
            labels, boxes, scores, queries = torch.tensor(labels).to('cuda'), torch.tensor(boxes).to('cuda'), torch.tensor(scores).to('cuda'), torch.tensor(queries).to('cuda')
            #print('boxes:',boxes.shape)
            """
            for box in target['boxes']:
               box = box.cpu().numpy()
               cv2.rectangle(image_cv, (box[0],box[1]), (box[2],box[3]), (255,0,0), 2)
            for box in boxes.squeeze(0).cpu().numpy():
               cv2.rectangle(image_cv, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,255,0), 2)
            cv2.imwrite('prueba.jpg',image_cv)
            """
            bbox_preds_uns = boxes.clone()
            # Convert xyxy -> cxcywh without changing the query dimension.
            cx = (bbox_preds_uns[:, :, 0] + bbox_preds_uns[:, :, 2]) * 0.5
            cy = (bbox_preds_uns[:, :, 1] + bbox_preds_uns[:, :, 3]) * 0.5
            w = bbox_preds_uns[:, :, 2] - bbox_preds_uns[:, :, 0]
            h = bbox_preds_uns[:, :, 3] - bbox_preds_uns[:, :, 1]
            bbox_preds_uns[:, :, 0] = cx
            bbox_preds_uns[:, :, 1] = cy
            bbox_preds_uns[:, :, 2] = w
            bbox_preds_uns[:, :, 3] = h
            bbox_preds_uns[:, :, [0,2]] = torch.clamp(bbox_preds_uns[:, :, [0,2]],min=0,max=width)
            bbox_preds_uns[:, :, [1,3]] = torch.clamp(bbox_preds_uns[:, :, [1,3]],min=0,max=height)
            bbox_preds_uns[:, :, [0,2]] /= width 
            bbox_preds_uns[:, :, [1,3]] /= height
            #print('boxes:',boxes)
            #print('target boxes:',target['boxes'])
            if mode in ['test']:
               gt_indexs = torch.empty(0, dtype=torch.long)
               indexs_2 = torch.arange(boxes.shape[1], dtype=torch.long)
               scores_flat = scores.reshape(-1)
               if scores_flat.numel() == boxes.shape[1]:
                  score_mask = (scores_flat >= test_score_thresh).detach().cpu()
                  indexs_2 = indexs_2[score_mask]
            else:
               ious_2 = ops.box_iou(target['boxes'], boxes.squeeze(0).cpu())
               #print('ious_2:',ious_2)
               cost = 1 - ious_2
               gt_indexs, indexs_2 = linear_sum_assignment(cost.cpu())
               gt_indexs = torch.tensor(gt_indexs, dtype=torch.long)
               indexs_2 = torch.tensor(indexs_2, dtype=torch.long)
               mask = (ious_2[gt_indexs, indexs_2] > 0.3).cpu()
               gt_indexs = gt_indexs[mask]
               indexs_2 = indexs_2[mask]
               valid = (indexs_2 >= 0) & (indexs_2 < boxes.shape[1])
               indexs_2 = indexs_2[valid]
            if indexs_2.numel() == 0:
               out_dict[img_path]['queries_det'].append(queries[:, :0].squeeze(0))
               out_dict[img_path]['boxes_det'].append(boxes[:, :0].squeeze(0))
               out_dict[img_path]['boxes_det_uns'].append(bbox_preds_uns[:, :0].squeeze(0))
               out_dict[img_path]['boxes_det_norm'].append(boxes[:, :0].squeeze(0))
               boxes_vis = []
            else:
               indexs_2_cuda = indexs_2.to(device=boxes.device)
               boxes_sel = boxes[:, indexs_2_cuda].squeeze(0)
               boxes_norm = boxes_sel.clone()
               boxes_norm[:, [0, 2]] = (boxes_norm[:, [0, 2]] / width).clamp(0.0, 1.0)
               boxes_norm[:, [1, 3]] = (boxes_norm[:, [1, 3]] / height).clamp(0.0, 1.0)
               out_dict[img_path]['queries_det'].append(queries[:, indexs_2_cuda].squeeze(0))
               out_dict[img_path]['boxes_det'].append(boxes_sel)
               out_dict[img_path]['boxes_det_uns'].append(bbox_preds_uns[:, indexs_2_cuda].squeeze(0))
               out_dict[img_path]['boxes_det_norm'].append(boxes_norm)
               boxes_vis = boxes_sel

            if mode in ['test']:
               if len(boxes_vis) == 0:
                  preds_fp[img_path] = np.zeros((0, 5), dtype=np.float32)
               else:
                  if torch.is_tensor(boxes_vis):
                     boxes_np = boxes_vis.detach().cpu().numpy()
                  else:
                     boxes_np = np.array(boxes_vis)
                  if indexs_2.numel() > 0:
                     scores_sel = scores[indexs_2_cuda].detach().cpu().numpy()
                  else:
                     scores_sel = np.zeros((0,), dtype=np.float32)
                  preds_fp[img_path] = np.concatenate(
                     [boxes_np, scores_sel.reshape(-1, 1)],
                     axis=1,
                  ).astype(np.float32)

            if mode in ['test'] and writer is not None and image_cv is not None:
               img_draw = image_cv.copy()
               draw_boxes(img_draw, target['boxes'], (0, 255, 0), 'gt')
               writer.write(img_draw)
            if mode in ['test'] and pred_writer is not None and image_cv is not None:
               img_pred = image_cv.copy()
               draw_boxes(img_pred, boxes_vis, (0, 0, 255), 'pred')
               pred_writer.write(img_pred)

            if len(target['boxes']) > 0:
               target['boxes'] = target['boxes'][gt_indexs]
               target['labels'] = target['labels'][gt_indexs]
               target['ids'] = target['ids'][gt_indexs]

         out_dict[img_path]['targets'].append(target)

         continue


      results = model(img_path,verbose=False,)
      if mode in ['test']:
         out_dict[img_path]['results'].append(results)
      bbox_preds = queries_out['boxes'][..., :4].clone()
      bbox_preds_uns = bbox_preds.clone()
      cx = bbox_preds[:, :, 0] * width
      cy = bbox_preds[:, :, 1] * height
      w  = bbox_preds[:, :, 2] * width
      h  = bbox_preds[:, :, 3] * height

      # Convertir a [x_min, y_min, x_max, y_max] en píxeles
      x_min = cx - 0.5 * w
      y_min = cy - 0.5 * h
      x_max = cx + 0.5 * w
      y_max = cy + 0.5 * h

      # Guardar de vuelta
      bbox_preds[:, :, 0] = x_min
      bbox_preds[:, :, 1] = y_min
      bbox_preds[:, :, 2] = x_max
      bbox_preds[:, :, 3] = y_max

      bbox_preds[:, :, [0,2]] = torch.clamp(bbox_preds[:, :, [0,2]],min=0,max=width)
      bbox_preds[:, :, [1,3]] = torch.clamp(bbox_preds[:, :, [1,3]],min=0,max=height)

      boxes_f = results[0].boxes.xyxy
      scores = results[0].boxes.conf
      labels = results[0].boxes.cls
      names = results[0].names
      boxes_vis = boxes_f
      scores_vis = scores
      #print('boxes_f shape:',boxes_f.shape)


      if mode in ['test']:
         nms_indexs = ops.nms(boxes=boxes_f,scores=scores,iou_threshold=0.2) #estaba en 0.4
         nms_indexs = [index for jk, index in zip(labels[nms_indexs], nms_indexs) if model.names[jk.detach().item()] in dataset_dict[dataset]]
         if len(nms_indexs) > 0:
            nms_indexs = torch.stack(nms_indexs)
         
            boxes_f = boxes_f[nms_indexs]
            scores = scores[nms_indexs]
            labels = labels[nms_indexs]

      if mode in ['test'] and writer is not None and image_cv is not None:
         img_draw = image_cv.copy()
         draw_boxes(img_draw, target['boxes'], (0, 255, 0), 'gt')
         writer.write(img_draw)
      if mode in ['test'] and pred_writer is not None and image_cv is not None:
         img_pred = image_cv.copy()
         draw_boxes(img_pred, boxes_f, (0, 0, 255), 'pred')
         pred_writer.write(img_pred)
         
      if len(target['boxes']) > 0:
         ious_2 = ops.box_iou(target['boxes'],boxes_f.cpu())
         cost = 1 - ious_2
         gt_indexs, indexs_2 = linear_sum_assignment(cost.cpu())
         gt_indexs = torch.tensor(gt_indexs)
         indexs_2 = torch.tensor(indexs_2)
         mask = (ious_2[gt_indexs,indexs_2]>0.3)
         mask = mask.cpu()
         gt_indexs = gt_indexs[mask]
         indexs_2 = indexs_2[mask]
      else:
         gt_indexs = []
         indexs_2 = []

      if len(target['boxes']) > 0:
         ious = ops.box_iou(boxes_f[indexs_2], bbox_preds.squeeze(0))
         cost = 1 - ious
         _, indexs = linear_sum_assignment(cost.cpu())
      else:
         gt_indexs = []
         indexs = []

      if mode in ['test']:
         ious_2 = ops.box_iou(boxes_f,bbox_preds.squeeze(0))
         cost = 1 - ious_2
         gt_indexs, indexs_2 = linear_sum_assignment(cost.cpu())
         gt_indexs = torch.tensor(gt_indexs)
         indexs_2 = torch.tensor(indexs_2)
         # print('indexs_2 stats:', indexs_2.dtype, int(indexs_2.min()) if indexs_2.numel() else None, int(indexs_2.max()) if indexs_2.numel() else None)
         mask = (ious_2[gt_indexs,indexs_2]>0.9)
         mask = mask.cpu()
         gt_indexs = gt_indexs[mask]
         indexs_2 = indexs_2[mask]
         if gt_indexs.numel() > 0:
            res_scores = scores[gt_indexs].detach().cpu().view(-1)
            det_scores = scores[gt_indexs].detach().cpu().view(-1)
            #print(f"[scores_check] results scores: {res_scores.tolist()}")
            #print(f"[scores_check] scores_det: {det_scores.tolist()}")
            if res_scores.numel() != det_scores.numel() or not torch.allclose(
               res_scores,
               det_scores,
               rtol=1e-4,
               atol=1e-6,
            ):
               max_diff = (res_scores - det_scores).abs().max().item() if det_scores.numel() else None
               raise ValueError(f"[scores_check] mismatch on {img_path} max_diff={max_diff}")
         out_dict[img_path]['queries_det'].append(queries_out['queries'][indexs_2])
         out_dict[img_path]['boxes_det'].append(bbox_preds[:,indexs_2].squeeze(0))
         out_dict[img_path]['boxes_det_uns'].append(bbox_preds_uns[:,indexs_2].squeeze(0))
         out_dict[img_path]['scores_det'].append(scores[gt_indexs].detach().view(-1))
      else:
         if dino:
            aux_boxes_f = boxes_f[indexs_2].int()
            image_cv_tensor = torch.from_numpy(image_cv)
            crops = [F.to_pil_image(image_cv_tensor[aux_box[1]:aux_box[3], aux_box[0]:aux_box[2]][:3]) for aux_box in aux_boxes_f]
            
            batch_crops = [transform(crop) for crop in crops]
            batch_tensor = torch.stack(batch_crops).to('cuda')
            with torch.no_grad():
               features = dinov2_vits14(batch_tensor)
            out_dict[img_path]['queries_det'].append(features.cpu())
            out_dict[img_path]['boxes_det'].append(aux_boxes_f)
         else:
            
            if len(target['boxes']) > 0: 
               out_dict[img_path]['queries_det'].append(queries_out['queries'][indexs])
               out_dict[img_path]['boxes_det'].append(bbox_preds[:,indexs].squeeze(0))
               out_dict[img_path]['boxes_det_uns'].append(bbox_preds_uns[:,indexs].squeeze(0))

         if len(target['boxes']) > 0:
            target['boxes'] = target['boxes'][gt_indexs]
            target['labels'] = target['labels'][gt_indexs]
            target['ids'] = target['ids'][gt_indexs]
         out_dict[img_path]['targets'].append(target)

   if mode in ['test'] and writer is not None:
      writer.release()
   if mode in ['test'] and pred_writer is not None:
      pred_writer.release()
      
if dataset == 'damages' and mode == 'test':
   np.save(file_name, preds_fp)
else:
   np.save(file_name, out_dict)

