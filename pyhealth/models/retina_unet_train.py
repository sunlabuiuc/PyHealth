"""
Training pipeline for Retina U-Net model.

Aligned with the original RetinaUNet train_forward function logic:
- Processes batches of images with segmentation masks
- Converts masks into bounding box and class annotations
- Performs anchor matching and loss computation
- Handles device placement (CPU/GPU) consistently
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import logging
from tqdm import tqdm
import json

from model import RetinaUNet
from data_loader import LIDCDataLoader


def setup_logger(log_dir: str, name: str = 'training') -> logging.Logger:
    """Setup logging."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    log_path = Path(log_dir) / f'{name}.log'
    handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    return logger


def get_device() -> torch.device:
    """Select device (CPU or GPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def extract_bboxes_from_mask(mask: np.ndarray, min_area: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract bounding boxes from a binary mask using connected components.
    
    This function handles multiple objects in the same mask by treating each 
    connected component as a separate object to detect.
    
    Args:
        mask: Binary segmentation mask (H, W) or (H, W, D)
        min_area: Minimum area (pixels) for a component to be considered an object
    
    Returns:
        boxes: Array of shape (num_objects, 2*dim) with coordinates [y1, x1, y2, x2] for 2D
               or [y1, x1, z1, y2, x2, z2] for 3D
        class_ids: Array of shape (num_objects,) with class labels (all 1 for now)
    """
    from scipy import ndimage
    
    if mask.max() == 0:
        return np.array([]), np.array([], dtype=np.int32)
    
    # Label connected components
    labeled, num_features = ndimage.label(mask > 0)
    
    boxes_list = []
    
    for component_id in range(1, num_features + 1):
        component = (labeled == component_id)
        
        # Filter by minimum area
        if component.sum() < min_area:
            continue
        
        if mask.ndim == 2:  # 2D case
            coords = np.argwhere(component)
            if len(coords) == 0:
                continue
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            boxes_list.append([y_min, x_min, y_max + 1, x_max + 1])  # +1 for inclusive range
        
        elif mask.ndim == 3:  # 3D case
            coords = np.argwhere(component)
            if len(coords) == 0:
                continue
            y_min, x_min, z_min = coords.min(axis=0)
            y_max, x_max, z_max = coords.max(axis=0)
            boxes_list.append([y_min, x_min, z_min, y_max + 1, x_max + 1, z_max + 1])
    
    if len(boxes_list) == 0:
        return np.array([]), np.array([], dtype=np.int32)
    
    boxes = np.array(boxes_list, dtype=np.float32)
    class_ids = np.ones(len(boxes_list), dtype=np.int32)  # All objects are class 1 (nodule)
    
    return boxes, class_ids


def transform_batch_to_retina_format(
    batch: Dict[str, Any],
    device: torch.device,
    dim: int = 2
) -> Dict[str, torch.Tensor]:
    """
    Transform batch from data loader format to RetinaUNet training format.
    
    Converts image + mask pairs into:
    - Images: (B, 1, H, W) or (B, 1, H, W, D)
    - Bounding boxes: list of arrays, one per batch element
    - Class IDs: list of arrays, one per batch element
    - Segmentation masks: (B, 1, H, W) or (B, 1, H, W, D)
    
    This handles multiple objects in the same image by extracting separate
    bounding boxes for each connected component in the mask.
    
    Args:
        batch: Dictionary from DataLoader with 'image' and 'mask' keys
        device: Device to place tensors on
        dim: Image dimension (2 or 3)
    
    Returns:
        Dictionary with keys:
            - 'images': (B, 1, H, W[, D]) float tensor
            - 'segmentation': (B, 1, H, W[, D]) float tensor
            - 'gt_boxes': list of (num_objects, 2*dim) arrays
            - 'gt_class_ids': list of (num_objects,) arrays
    """
    images = batch['image']  # (B, 1, H, W) or (B, 1, H, W, D)
    masks = batch['mask']    # (B, 1, H, W) or (B, 1, H, W, D)
    
    batch_size = images.shape[0]
    
    # Convert to torch tensors on device
    if isinstance(images, np.ndarray):
        images_tensor = torch.from_numpy(images).float().to(device)
    else:
        images_tensor = images.to(device)
    
    if isinstance(masks, np.ndarray):
        masks_tensor = torch.from_numpy(masks).float().to(device)
    else:
        masks_tensor = masks.to(device)
    
    # Extract bounding boxes from each mask in batch
    gt_boxes = []
    gt_class_ids = []
    
    for b in range(batch_size):
        # Get mask for this batch element (H, W) or (H, W, D)
        mask_numpy = masks[b, 0]  # Remove channel dimension
        
        # Extract bounding boxes from connected components
        boxes, class_ids = extract_bboxes_from_mask(mask_numpy, min_area=10)
        
        gt_boxes.append(boxes)
        gt_class_ids.append(class_ids)
    
    return {
        'images': images_tensor,
        'segmentation': masks_tensor,
        'gt_boxes': gt_boxes,
        'gt_class_ids': gt_class_ids
    }


def compute_anchor_matches(
    anchors: np.ndarray,
    gt_boxes: np.ndarray,
    gt_class_ids: np.ndarray,
    iou_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match ground truth boxes to anchors using IoU.
    
    Args:
        anchors: Array of shape (num_anchors, 2*dim)
        gt_boxes: Array of shape (num_gt_boxes, 2*dim)
        gt_class_ids: Array of shape (num_gt_boxes,)
        iou_threshold: IoU threshold for positive match
    
    Returns:
        anchor_class_match: (num_anchors,) with [-1=negative, 0=neutral, >0=class_id]
        anchor_target_deltas: (num_anchors, 2*dim) with regression targets
    """
    # Initialize with -1 (negative anchors)
    anchor_class_match = np.zeros(anchors.shape[0], dtype=np.int32)
    anchor_class_match[:] = -1
    
    # Initialize bbox deltas
    anchor_target_deltas = np.zeros(anchors.shape, dtype=np.float32)
    
    if len(gt_boxes) == 0:
        # No objects in this sample - all anchors are negative
        anchor_class_match[:] = -1
        return anchor_class_match, anchor_target_deltas
    
    # Compute IoU between each anchor and gt box
    # IoU = intersection / union
    def compute_iou_2d(anchor, gt_box):
        """Compute IoU for 2D boxes."""
        y1a, x1a, y2a, x2a = anchor
        y1g, x1g, y2g, x2g = gt_box
        
        # Intersection
        yi1 = max(y1a, y1g)
        xi1 = max(x1a, x1g)
        yi2 = min(y2a, y2g)
        xi2 = min(x2a, x2g)
        inter_area = max(0, yi2 - yi1) * max(0, xi2 - xi1)
        
        # Union
        anchor_area = (y2a - y1a) * (x2a - x1a)
        gt_area = (y2g - y1g) * (x2g - x1g)
        union_area = anchor_area + gt_area - inter_area
        
        return inter_area / max(union_area, 1e-5)
    
    # Match anchors to gt boxes
    for anchor_idx, anchor in enumerate(anchors):
        # Find best matching gt box
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = compute_iou_2d(anchor, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Assign class based on IoU threshold
        if best_iou >= iou_threshold:
            anchor_class_match[anchor_idx] = gt_class_ids[best_gt_idx]
            # Compute regression targets (simplified delta computation)
            anchor_target_deltas[anchor_idx] = (gt_boxes[best_gt_idx] - anchor) / (anchor + 1e-5)
        else:
            anchor_class_match[anchor_idx] = -1
    
    return anchor_class_match.astype(np.int32), anchor_target_deltas.astype(np.float32)


# def compute_focal_loss(
#     class_logits: torch.Tensor,
#     anchor_class_match: torch.Tensor,
#     alpha: float = 0.25,
#     gamma: float = 2.0
# ) -> torch.Tensor:
#     """
#     Compute focal loss for object detection.
    
#     Focal loss = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
#     Args:
#         class_logits: (num_anchors, num_classes) prediction logits
#         anchor_class_match: (num_anchors,) with class labels [-1=negative, 0=neutral, 1+=class_id]
#         alpha: Focal loss alpha parameter
#         gamma: Focal loss gamma parameter
    
#     Returns:
#         Scalar loss value
#     """
#     # Get softmax probabilities
#     probs = F.softmax(class_logits, dim=-1)
    
#     # Only use valid anchors (not neutral)
#     valid_mask = anchor_class_match >= 0
#     if valid_mask.sum() == 0:
#         return torch.tensor(0.0, device=class_logits.device)
    
#     valid_logits = class_logits[valid_mask]
#     valid_targets = anchor_class_match[valid_mask].long()
#     valid_probs = probs[valid_mask]
    
#     # Standard cross entropy loss
#     ce_loss = F.cross_entropy(valid_logits, valid_targets.squeeze(1).long(), reduction='none')
    
#     # Get probability of correct class
#     p_t = valid_probs.gather(1, valid_targets.unsqueeze(1)).squeeze(1)
    
#     # Focal loss modulation
#     focal_loss = alpha * ((1 - p_t) ** gamma) * ce_loss
    
#     return focal_loss.mean()


def compute_class_loss(class_pred_logits, anchor_matches, shem_poolsize=20):
    """
    Refactored to match the original Retina U-Net OHEM implementation.
    
    Args:
        class_pred_logits: (n_anchors, n_classes) Logits from the classifier sub-net.
        anchor_matches: (n_anchors) [-1=negative, 0=neutral, 1+=class_id]
        shem_poolsize: factor for top-k candidates in hard example mining.
    """
    # 1. Separate indices by type
    pos_indices = torch.nonzero(anchor_matches > 0).view(-1)
    neg_indices = torch.nonzero(anchor_matches == -1).view(-1)

    # 2. Calculate Positive Loss (Standard Cross Entropy)
    if pos_indices.numel() > 0:
        roi_logits_pos = class_pred_logits[pos_indices]
        targets_pos = anchor_matches[pos_indices].long()
        pos_loss = F.cross_entropy(roi_logits_pos, targets_pos)
    else:
        # Match original: FloatTensor([0]).cuda()
        pos_loss = torch.tensor(0.0, device=class_pred_logits.device)

    # 3. Calculate Negative Loss with OHEM
    if neg_indices.numel() > 0:
        roi_logits_neg = class_pred_logits[neg_indices]
        # Number of negatives should match positives (at least 1)
        negative_count = max(1, pos_indices.numel())
        
        # OHEM: Get probabilities to find "hard" negatives
        roi_probs_neg = F.softmax(roi_logits_neg, dim=1)
        
        # Hard sampling (assuming mutils.shem is available or similar logic)
        # For OHEM, we pick indices where background probability is lowest
        neg_ix = shem_sampling(roi_probs_neg, negative_count, shem_poolsize)
        
        # Target for negatives is always 0 (background class)
        neg_targets = torch.zeros(len(neg_ix), dtype=torch.long, device=class_pred_logits.device)
        neg_loss = F.cross_entropy(roi_logits_neg[neg_ix], neg_targets)
        
    else:
        neg_loss = torch.tensor(0.0, device=class_pred_logits.device)

    # 4. Final loss is average of the two
    loss = (pos_loss + neg_loss) / 2
    return loss

def shem_sampling(probs, count, poolsize):
    """
    Simulated mutils.shem: Picks 'count' hard negatives from a pool 
    of size count * poolsize.
    """
    # Background is usually class 0. "Hard" negatives have LOW background prob.
    bg_probs = probs[:, 0]
    num_candidates = min(count * poolsize, len(bg_probs))
    
    # Get candidates with lowest background confidence
    _, candidate_indices = torch.topk(bg_probs, num_candidates, largest=False)
    
    # Randomly sample 'count' from the candidates (as per some SHEM implementations)
    perm = torch.randperm(len(candidate_indices))[:count]
    return candidate_indices[perm]


def compute_bbox_loss(
    anchor_target_deltas: torch.Tensor,
    bbox_deltas: torch.Tensor,
    anchor_class_match: torch.Tensor
) -> torch.Tensor:
    """
    Compute bounding box regression loss.
    
    Args:
        anchor_target_deltas: (num_anchors, 2*dim) target deltas
        bbox_deltas: (num_anchors, 2*dim) predicted deltas
        anchor_class_match: (num_anchors,) class matches
    
    Returns:
        Scalar loss value
    """
    # Use Smooth L1 loss for positive anchors only
    pos_mask = anchor_class_match > 0
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=bbox_deltas.device)
    
    pos_deltas = bbox_deltas[pos_mask]
    pos_targets = anchor_target_deltas[pos_mask]
    
    # Smooth L1 loss
    bbox_loss = F.smooth_l1_loss(pos_deltas, pos_targets, reduction='mean')
    
    return bbox_loss


def compute_segmentation_loss(
    seg_logits: torch.Tensor,
    seg_masks: torch.Tensor,
    n_classes=2
) -> torch.Tensor:
    """
    Compute segmentation loss (combined Dice + Cross-entropy).
    
    Args:
        seg_logits: (B, 1, H, W[, D]) raw segmentation outputs
        seg_masks: (B, 1, H, W[, D]) ground truth masks
    
    Returns:
        Scalar loss value
    """
    # Compute cross-entropy loss
    target_masks = seg_masks.squeeze(1).long()
    ce_loss = F.cross_entropy(seg_logits, target_masks, reduction='mean')
    
    # Compute Dice loss
    # Convert logits to probabilities using Softmax
    probs = F.softmax(seg_logits, dim=1)
    
    # Convert target masks [B, 1, H, W] to One-Hot [B, C, H, W]
    # We squeeze the channel dim, one_hot it, then permute to put classes at dim 1
    target_ohe = F.one_hot(seg_masks.squeeze(1).long(), num_classes=n_classes)
    target_ohe = target_ohe.permute(0, 3, 1, 2).float() # [B, C, H, W]

    # Compute Dice per class (excluding background class 0 is often preferred)
    # But for standard implementation, we sum over all spatial dims
    dims = (0, 2, 3) 
    intersection = torch.sum(probs * target_ohe, dim=dims)
    cardinality = torch.sum(probs + target_ohe, dim=dims)
    
    dice_score = (2. * intersection + 1e-5) / (cardinality + 1e-5)
    dice_loss = 1 - dice_score.mean() # Average Dice loss across all classes
    
    # Combined loss
    return 0.5 * ce_loss + 0.5 * dice_loss


class Trainer:
    """Training orchestrator for Retina U-Net."""
    
    def __init__(
        self,
        model: RetinaUNet,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs',
        dim: int = 2
    ):
        """
        Initialize trainer.
        
        Args:
            model: Retina U-Net model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            device: Device to train on (auto-selected if None)
            lr: Learning rate
            weight_decay: Weight decay
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
            dim: Image dimension (2 or 3)
        """
        self.device = device if device is not None else get_device()
        self.model = model.to(self.device)
        self.dim = dim
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Logger
        self.logger = setup_logger(str(self.log_dir))
        self.writer = SummaryWriter(str(self.log_dir))
        
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Matches original train_forward logic:
        - Process each batch
        - Extract bounding boxes from masks
        - Match anchors to ground truth
        - Compute focal loss, bbox loss, and segmentation loss
        - Perform backward pass and optimization
        
        Returns:
            Dictionary with average loss components
        """
        self.model.train()
        metrics = {
            'total_loss': 0.0,
            'class_loss': 0.0,
            'bbox_loss': 0.0,
            'seg_loss': 0.0
        }
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            # Transform batch to RetinaUNet format
            batch_data = transform_batch_to_retina_format(batch, self.device, self.dim)
            
            images = batch_data['images']  # (B, 1, H, W)
            seg_masks = batch_data['segmentation']  # (B, 1, H, W)
            gt_boxes_list = batch_data['gt_boxes']  # list of arrays
            gt_class_ids_list = batch_data['gt_class_ids']  # list of arrays
            
            batch_size = images.shape[0]
            
            # Forward pass
            outputs = self.model(images)
            
            class_logits = outputs['class_logits']  # (B, num_anchors, num_classes)
            bbox_deltas = outputs['bbox_deltas']    # (B, num_anchors, 4)
            seg_logits = outputs['segmentation']    # (B, 1, H, W)
            
            # Initialize loss accumulators
            batch_class_loss = torch.tensor(0.0, device=self.device)
            batch_bbox_loss = torch.tensor(0.0, device=self.device)
            num_valid_batches = 0
            
            # Get anchors from model (numpy format)
            # For new RetinaUNet, need to extract anchors from cache
            if hasattr(self.model, 'anchor_cache') and len(self.model.anchor_cache) > 0:
                # Get cached anchors - they should be in cache after forward pass
                anchors_tensor = list(self.model.anchor_cache.values())[0]
                anchors_numpy = anchors_tensor.cpu().detach().numpy()
            else:
                # Fallback: use anchor generator directly
                features = self.model.fpn(images)
                anchors_tensor = self.model.anchor_generator(features)
                anchors_numpy = anchors_tensor.cpu().detach().numpy()
            
            # Process each batch element
            for b in range(batch_size):
                gt_boxes = gt_boxes_list[b]
                gt_class_ids = gt_class_ids_list[b]
                
                # Match anchors to ground truth
                anchor_class_match, anchor_target_deltas = compute_anchor_matches(
                    anchors_numpy,
                    gt_boxes,
                    gt_class_ids,
                    iou_threshold=0.5
                )
                
                # Convert to tensors
                anchor_class_match_t = torch.from_numpy(anchor_class_match).to(self.device)
                anchor_target_deltas_t = torch.from_numpy(anchor_target_deltas).to(self.device)
                
                # Compute losses for this batch element
                class_loss = compute_class_loss(class_logits[b], anchor_class_match_t)
                
                bbox_loss = compute_bbox_loss(
                    anchor_target_deltas_t,
                    bbox_deltas[b],
                    anchor_class_match_t
                )
                
                batch_class_loss = batch_class_loss + class_loss / batch_size
                batch_bbox_loss = batch_bbox_loss + bbox_loss / batch_size
                num_valid_batches += 1
            
            # Segmentation loss
            seg_loss = compute_segmentation_loss(seg_logits, seg_masks)
            
            # Total loss (with segmentation weight)
            seg_weight = 0.5
            total_loss = batch_class_loss + batch_bbox_loss + seg_weight * seg_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update metrics
            metrics['total_loss'] += total_loss.item()
            metrics['class_loss'] += batch_class_loss.item()
            metrics['bbox_loss'] += batch_bbox_loss.item()
            metrics['seg_loss'] += seg_loss.item()
            
            # Update progress bar
            pbar.set_postfix({k: f"{v/(batch_idx+1):.4f}" for k, v in metrics.items()})
            
            self.global_step += 1
        
        # Average metrics
        n_batches = len(self.train_loader)
        return {k: v / n_batches for k, v in metrics.items()}
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model.
        
        Uses eval() mode with gradients disabled to save memory and time.
        
        Returns:
            Dictionary with average loss components
        """
        self.model.eval()
        metrics = {
            'total_loss': 0.0,
            'class_loss': 0.0,
            'bbox_loss': 0.0,
            'seg_loss': 0.0
        }
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch_idx, batch in enumerate(pbar):
                # Transform batch
                batch_data = transform_batch_to_retina_format(batch, self.device, self.dim)
                
                images = batch_data['images']
                seg_masks = batch_data['segmentation']
                gt_boxes_list = batch_data['gt_boxes']
                gt_class_ids_list = batch_data['gt_class_ids']
                
                batch_size = images.shape[0]
                
                # Forward pass
                outputs = self.model(images)
                
                class_logits = outputs['class_logits']
                bbox_deltas = outputs['bbox_deltas']
                seg_logits = outputs['segmentation']
                
                # Initialize loss accumulators
                batch_class_loss = torch.tensor(0.0, device=self.device)
                batch_bbox_loss = torch.tensor(0.0, device=self.device)
                
                # Get anchors
                if hasattr(self.model, 'anchor_cache') and len(self.model.anchor_cache) > 0:
                    anchors_tensor = list(self.model.anchor_cache.values())[0]
                    anchors_numpy = anchors_tensor.cpu().detach().numpy()
                else:
                    features = self.model.fpn(images)
                    anchors_tensor = self.model.anchor_generator(features)
                    anchors_numpy = anchors_tensor.cpu().detach().numpy()
                
                # Process each batch element
                for b in range(batch_size):
                    gt_boxes = gt_boxes_list[b]
                    gt_class_ids = gt_class_ids_list[b]
                    
                    # Match anchors
                    anchor_class_match, anchor_target_deltas = compute_anchor_matches(
                        anchors_numpy,
                        gt_boxes,
                        gt_class_ids,
                        iou_threshold=0.5
                    )
                    
                    anchor_class_match_t = torch.from_numpy(anchor_class_match).to(self.device)
                    anchor_target_deltas_t = torch.from_numpy(anchor_target_deltas).to(self.device)
                    
                    # Compute losses
                    class_loss = compute_class_loss(class_logits[b], anchor_class_match_t)
                    bbox_loss = compute_bbox_loss(anchor_target_deltas_t, bbox_deltas[b], anchor_class_match_t)
                    
                    batch_class_loss = batch_class_loss + class_loss / batch_size
                    batch_bbox_loss = batch_bbox_loss + bbox_loss / batch_size
                
                # Segmentation loss
                seg_loss = compute_segmentation_loss(seg_logits, seg_masks)
                
                # Total loss
                seg_weight = 0.5
                total_loss = batch_class_loss + batch_bbox_loss + seg_weight * seg_loss
                
                # Update metrics
                metrics['total_loss'] += total_loss.item()
                metrics['class_loss'] += batch_class_loss.item()
                metrics['bbox_loss'] += batch_bbox_loss.item()
                metrics['seg_loss'] += seg_loss.item()
                
                pbar.set_postfix({k: f"{v/(batch_idx+1):.4f}" for k, v in metrics.items()})
        
        # Average metrics
        n_batches = len(self.val_loader)
        return {k: v / n_batches for k, v in metrics.items()}
    
    def train(self, num_epochs: int, save_interval: int = 1):
        """
        Train model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_interval: Save checkpoint every N epochs
        """
        self.logger.info("Starting training...")
        
        for epoch in range(num_epochs):
            self.logger.info(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            
            # Train
            train_metrics = self.train_epoch()
            self.logger.info(f"Train - {', '.join([f'{k}: {v:.4f}' for k, v in train_metrics.items()])}")
            
            # Log to tensorboard
            for key, val in train_metrics.items():
                self.writer.add_scalar(f'train/{key}', val, epoch)
            
            # Validate
            val_metrics = self.validate()
            self.logger.info(f"Val   - {', '.join([f'{k}: {v:.4f}' for k, v in val_metrics.items()])}")
            
            # Log to tensorboard
            for key, val in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', val, epoch)
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self._save_checkpoint(epoch, val_metrics['total_loss'])
        
        self.logger.info("Training complete!")
        self.writer.close()
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'global_step': self.global_step
        }
        
        # Save last checkpoint
        last_path = self.checkpoint_dir / 'last.pth'
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint with val_loss: {val_loss:.4f}")


def train_model(
    data_dir: str,
    checkpoint_dir: str = './checkpoints',
    log_dir: str = './logs',
    num_epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    num_workers: int = 0,
    dim: int = 2,
    target_size: Optional[Tuple[int, int]] = None
):
    """
    End-to-end training script.
    
    Args:
        data_dir: Path to LIDC data directory
        checkpoint_dir: Where to save checkpoints
        log_dir: Where to save logs
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        num_workers: Number of data loading workers
        dim: Dimension (2 or 3)
        target_size: Target image size
    """
    device = get_device()
    
    # Create dataloaders
    dataloaders = LIDCDataLoader.create_dataloaders(
        data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        target_size=target_size
    )
    
    # Create model
    model = RetinaUNet(
        in_channels=1,
        num_classes=2,
        dim=dim,
        fpn_base_channels=64,
        fpn_out_channels=192,
        rpn_hidden_channels=256
    )
    
    # Create trainer
    trainer = Trainer(
        model,
        dataloaders['train'],
        dataloaders['val'],
        device=device,
        lr=lr,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        dim=dim
    )
    
    # Train
    trainer.train(num_epochs=num_epochs, save_interval=1)
    
    return model, trainer


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Log directory')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--dim', type=int, default=2,
                        help='Dimension (2 or 3)')
    
    args = parser.parse_args()
    
    train_model(
        args.data_dir,
        args.checkpoint_dir,
        args.log_dir,
        args.num_epochs,
        args.batch_size,
        args.lr,
        dim=args.dim
    )

