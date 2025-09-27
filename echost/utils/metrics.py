"""
https://idiotdeveloper.com/multiclass-segmentation-in-pytorch-using-u-net
"""

import torch

class DiceLoss(torch.nn.Module):
    def __init__(self, num_classes, smooth=1e-5, ignore_index=None, reduction="mean"):
        super(DiceLoss, self).__init__()

        self.num_classes    = num_classes
        self.smooth         = smooth
        self.ignore_index   = ignore_index
        self.reduction      = reduction # "mean", "sum" or "none"  

    def forward(self, logits, targets):
        """
        logits: Tensor of shape [B, C, H, W]
        targets: Tensor of shape [B, H, W]
        """
        probs           = torch.nn.functional.softmax(logits, dim=1)   # [B, C, H, W]
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()  # [B, H, W] -> [B, H, W, C] (one_hot) -> [B, C, H, W] (permute)

        if self.ignore_index is not None:
            mask        = (targets != self.ignore_index).unsqueeze(1) # [B, H, W] -> [B, 1, H, W]
            probs       = probs * mask # Broadcasting 

        
        intersection    = torch.sum(probs*targets_one_hot, dim=(2,3))   # [B, C]
        union           = torch.sum(probs + targets_one_hot, dim=(2,3)) # [B, C]
        dice            = (2 * intersection + self.smooth) / (union + self.smooth)  # [B, C]

        if self.reduction == 'mean': 
            return 1.0 - dice.mean() 
        elif self.reduction == 'sum': 
            return 1.0 - dice.sum() 
        else: 
            return 1.0 - dice # No reduction (returns [B, C])
        

class DiceCELoss(torch.nn.Module):
    def __init__(self, num_classes, alpha=0.5, ignore_index=None):
        super().__init__()
        self.alpha      = alpha
        self.dice       = DiceLoss(num_classes=num_classes,ignore_index=ignore_index)
        self.ce         = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, targets):
        return self.alpha * self.ce(logits, targets) + (1 - self.alpha) * self.dice(logits, targets)