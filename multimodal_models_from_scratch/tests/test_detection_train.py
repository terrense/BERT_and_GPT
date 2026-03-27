"""
Tests for detection training module.

Tests the DetectionTrainer class and data augmentation functions.
"""

import pytest
import torch
import torch.nn as nn

from multimodal_models_from_scratch.training.detection_train import (
    DetectionTrainer,
    DetectionTrainingConfig,
    DetectionAugmentation,
    random_horizontal_flip,
    random_resize,
    color_jitter,
    normalize_image,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from multimodal_models_from_scratch.detection.hungarian import HungarianMatcher
from multimodal_models_from_scratch.detection.losses import DETRLoss
from multimodal_models_from_scratch.detection.detr import DETR
from multimodal_models_from_scratch.config import DETRConfig


class TestDataAugmentation:
    """Tests for data augmentation functions."""
    
    def test_random_horizontal_flip_shape(self):
        """Test that horizontal flip preserves shape."""
        image = torch.rand(3, 224, 224)
        targets = {
            'labels': torch.tensor([1, 2]),
            'boxes': torch.tensor([[0.3, 0.4, 0.2, 0.3], [0.7, 0.6, 0.1, 0.2]])
        }
        
        flipped_image, flipped_targets = random_horizontal_flip(image, targets, prob=1.0)
        
        assert flipped_image.shape == image.shape
        assert flipped_targets['boxes'].shape == targets['boxes'].shape
        assert flipped_targets['labels'].shape == targets['labels'].shape
    
    def test_random_horizontal_flip_boxes(self):
        """Test that horizontal flip correctly transforms boxes."""
        image = torch.rand(3, 224, 224)
        targets = {
            'labels': torch.tensor([1]),
            'boxes': torch.tensor([[0.3, 0.4, 0.2, 0.3]])  # cx, cy, w, h
        }
        
        _, flipped_targets = random_horizontal_flip(image, targets, prob=1.0)
        
        # cx should be flipped: new_cx = 1 - cx = 1 - 0.3 = 0.7
        assert torch.allclose(flipped_targets['boxes'][0, 0], torch.tensor(0.7))
        # cy, w, h should remain the same
        assert torch.allclose(flipped_targets['boxes'][0, 1:], targets['boxes'][0, 1:])

    def test_random_resize_shape(self):
        """Test that random resize changes image size."""
        image = torch.rand(3, 224, 224)
        targets = {
            'labels': torch.tensor([1]),
            'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        }
        
        resized_image, resized_targets = random_resize(image, targets, scales=(0.5, 0.5))
        
        # Image should be resized
        assert resized_image.shape[1] == 112  # 224 * 0.5
        assert resized_image.shape[2] == 112
        # Normalized boxes should remain the same
        assert torch.allclose(resized_targets['boxes'], targets['boxes'])
    
    def test_color_jitter_shape(self):
        """Test that color jitter preserves shape."""
        image = torch.rand(3, 224, 224)
        
        jittered = color_jitter(image, brightness=0.4, contrast=0.4, saturation=0.4)
        
        assert jittered.shape == image.shape
        # Values should be clamped to [0, 1]
        assert jittered.min() >= 0
        assert jittered.max() <= 1
    
    def test_normalize_image(self):
        """Test ImageNet normalization."""
        image = torch.ones(3, 224, 224) * 0.5
        
        normalized = normalize_image(image)
        
        # Check that normalization is applied correctly
        expected_r = (0.5 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        expected_g = (0.5 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
        expected_b = (0.5 - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
        
        assert torch.allclose(normalized[0], torch.tensor(expected_r), atol=1e-5)
        assert torch.allclose(normalized[1], torch.tensor(expected_g), atol=1e-5)
        assert torch.allclose(normalized[2], torch.tensor(expected_b), atol=1e-5)


class TestDetectionAugmentation:
    """Tests for DetectionAugmentation class."""
    
    def test_augmentation_with_config(self):
        """Test augmentation with config."""
        config = DetectionTrainingConfig(
            use_augmentation=True,
            horizontal_flip_prob=0.5,
            resize_scales=(0.9, 1.1),
            color_jitter=False
        )
        augmentation = DetectionAugmentation(config)
        
        image = torch.rand(3, 224, 224)
        targets = {
            'labels': torch.tensor([1, 2]),
            'boxes': torch.tensor([[0.3, 0.4, 0.2, 0.3], [0.7, 0.6, 0.1, 0.2]])
        }
        
        aug_image, aug_targets = augmentation(image, targets)
        
        # Output should have 3 channels
        assert aug_image.shape[0] == 3
        # Labels should be preserved
        assert aug_targets['labels'].shape == targets['labels'].shape
    
    def test_augmentation_disabled(self):
        """Test augmentation when disabled."""
        config = DetectionTrainingConfig(use_augmentation=False)
        augmentation = DetectionAugmentation(config)
        
        image = torch.rand(3, 224, 224)
        targets = {
            'labels': torch.tensor([1]),
            'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        }
        
        aug_image, aug_targets = augmentation(image, targets)
        
        # Shape should be preserved (only normalization applied)
        assert aug_image.shape == image.shape
        # Boxes should be unchanged
        assert torch.allclose(aug_targets['boxes'], targets['boxes'])


class TestDetectionTrainingConfig:
    """Tests for DetectionTrainingConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DetectionTrainingConfig()
        
        assert config.loss_ce_weight == 1.0
        assert config.loss_bbox_weight == 5.0
        assert config.loss_giou_weight == 2.0
        assert config.gradient_accumulation_steps == 1
        assert config.use_augmentation == True
        assert config.horizontal_flip_prob == 0.5
        assert config.resize_scales == (0.8, 1.2)
        assert config.color_jitter == False
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = DetectionTrainingConfig(
            loss_ce_weight=2.0,
            loss_bbox_weight=10.0,
            loss_giou_weight=4.0,
            gradient_accumulation_steps=4,
            use_augmentation=False,
            learning_rate=1e-5
        )
        
        assert config.loss_ce_weight == 2.0
        assert config.loss_bbox_weight == 10.0
        assert config.loss_giou_weight == 4.0
        assert config.gradient_accumulation_steps == 4
        assert config.use_augmentation == False
        assert config.learning_rate == 1e-5


class TestDetectionTrainer:
    """Tests for DetectionTrainer class."""
    
    @pytest.fixture
    def small_detr_config(self):
        """Create a small DETR config for testing."""
        return DETRConfig(
            num_classes=10,
            num_queries=10,
            d_model=64,
            num_heads=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            d_ff=128,
            dropout_rate=0.0
        )
    
    @pytest.fixture
    def trainer(self, small_detr_config):
        """Create a trainer for testing."""
        model = DETR(small_detr_config, backbone_name='resnet18')
        matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
        loss_fn = DETRLoss(num_classes=10, matcher=matcher)
        config = DetectionTrainingConfig(
            learning_rate=1e-4,
            batch_size=2,
            use_augmentation=False,  # Disable for deterministic tests
            gradient_accumulation_steps=1
        )
        return DetectionTrainer(model, config, matcher, loss_fn)
    
    def test_trainer_initialization(self, trainer):
        """Test trainer initialization."""
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.global_step == 0
        assert trainer.accumulation_step == 0
    
    def test_train_step(self, trainer):
        """Test single training step."""
        batch = {
            'pixel_values': torch.rand(2, 3, 224, 224),
            'targets': [
                {
                    'labels': torch.tensor([1, 2]),
                    'boxes': torch.tensor([[0.3, 0.4, 0.2, 0.3], [0.7, 0.6, 0.1, 0.2]])
                },
                {
                    'labels': torch.tensor([3]),
                    'boxes': torch.tensor([[0.5, 0.5, 0.3, 0.3]])
                }
            ]
        }
        
        metrics = trainer.train_step(batch)
        
        assert 'loss' in metrics
        assert 'loss_ce' in metrics
        assert 'loss_bbox' in metrics
        assert 'loss_giou' in metrics
        assert 'did_update' in metrics
        assert metrics['did_update'] == True  # No gradient accumulation
        assert trainer.global_step == 1

    def test_gradient_accumulation(self, small_detr_config):
        """Test gradient accumulation."""
        model = DETR(small_detr_config, backbone_name='resnet18')
        matcher = HungarianMatcher()
        loss_fn = DETRLoss(num_classes=10, matcher=matcher)
        config = DetectionTrainingConfig(
            learning_rate=1e-4,
            gradient_accumulation_steps=2,
            use_augmentation=False
        )
        trainer = DetectionTrainer(model, config, matcher, loss_fn)
        
        batch = {
            'pixel_values': torch.rand(1, 3, 224, 224),
            'targets': [
                {
                    'labels': torch.tensor([1]),
                    'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]])
                }
            ]
        }
        
        # First step - should not update
        metrics1 = trainer.train_step(batch)
        assert metrics1['did_update'] == False
        assert trainer.global_step == 0
        assert trainer.accumulation_step == 1
        
        # Second step - should update
        metrics2 = trainer.train_step(batch)
        assert metrics2['did_update'] == True
        assert trainer.global_step == 1
        assert trainer.accumulation_step == 0
    
    def test_get_parameters(self, trainer):
        """Test parameter counting methods."""
        trainable = trainer.get_trainable_parameters()
        total = trainer.get_total_parameters()
        
        assert trainable > 0
        assert total > 0
        assert trainable <= total
    
    def test_freeze_backbone(self, trainer):
        """Test backbone freezing."""
        initial_trainable = trainer.get_trainable_parameters()
        
        trainer.freeze_backbone()
        frozen_trainable = trainer.get_trainable_parameters()
        
        # Should have fewer trainable parameters after freezing
        assert frozen_trainable < initial_trainable
        
        trainer.unfreeze_backbone()
        unfrozen_trainable = trainer.get_trainable_parameters()
        
        # Should be back to original
        assert unfrozen_trainable == initial_trainable
    
    def test_set_scheduler(self, trainer):
        """Test learning rate scheduler setup."""
        trainer.set_scheduler(total_steps=100, warmup_steps=10)
        
        assert trainer.scheduler is not None
        
        # Initial LR should be low (warmup)
        initial_lr = trainer.get_current_lr()
        
        # Step through warmup
        for _ in range(10):
            trainer.scheduler.step()
        
        # LR should be higher after warmup
        warmup_lr = trainer.get_current_lr()
        assert warmup_lr > initial_lr
    
    def test_loss_history(self, trainer):
        """Test loss history tracking."""
        batch = {
            'pixel_values': torch.rand(1, 3, 224, 224),
            'targets': [
                {
                    'labels': torch.tensor([1]),
                    'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]])
                }
            ]
        }
        
        trainer.train_step(batch)
        trainer.train_step(batch)
        
        history = trainer.get_loss_history()
        
        assert len(history) == 2
        assert 'loss' in history[0]
        assert 'loss_ce' in history[0]
        assert 'loss_bbox' in history[0]
        assert 'loss_giou' in history[0]


class TestDetectionTrainerEvaluation:
    """Tests for DetectionTrainer evaluation."""
    
    @pytest.fixture
    def trainer_for_eval(self):
        """Create a trainer for evaluation testing."""
        config = DETRConfig(
            num_classes=10,
            num_queries=10,
            d_model=64,
            num_heads=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            d_ff=128
        )
        model = DETR(config, backbone_name='resnet18')
        matcher = HungarianMatcher()
        loss_fn = DETRLoss(num_classes=10, matcher=matcher)
        train_config = DetectionTrainingConfig(use_augmentation=False)
        return DetectionTrainer(model, train_config, matcher, loss_fn)
    
    def test_evaluate(self, trainer_for_eval):
        """Test evaluation method."""
        # Create a simple dataloader-like list
        batches = [
            {
                'pixel_values': torch.rand(1, 3, 224, 224),
                'targets': [
                    {
                        'labels': torch.tensor([1]),
                        'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]])
                    }
                ]
            }
        ]
        
        # Simple iterator
        class SimpleDataLoader:
            def __init__(self, batches):
                self.batches = batches
            def __iter__(self):
                return iter(self.batches)
        
        dataloader = SimpleDataLoader(batches)
        metrics = trainer_for_eval.evaluate(dataloader)
        
        assert 'val_loss' in metrics
        assert 'val_loss_ce' in metrics
        assert 'val_loss_bbox' in metrics
        assert 'val_loss_giou' in metrics
        assert 'mAP' in metrics
        assert 'mAP_50' in metrics
        assert 'mAP_75' in metrics
