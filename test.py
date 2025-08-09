import modelLoader
import cv2
import numpy as np
from PIL import Image

from attacks.adversarial_patch_attack import adversarial_patch_attack
from attacks.auto_attack import auto_attack
from attacks.auto_pgd_attack import auto_pgd_attack
from attacks.auto_cg_attack import auto_cg_attack
from attacks.boundary_attack import boundary_attack
from attacks.carlini_linf_attack import carlini_linf_attack
from attacks.composite_attack import composite_attack
from attacks.deepfool_attack import deepfool_attack
from attacks.elastic_net_attack import elastic_net_attack

resnet50_Model = modelLoader.loadModel("resnet50", ".models/resnet50.pth")
# densenet121_Model = modelLoader.loadModel("densenet121", ".models/densenet121.pth")
# densenet121_dct_Model = modelLoader.loadModel("densenet121_dct", ".models/densenet121_dct.pth")
# vit_b_16_Model = modelLoader.loadModel("vit_b_16", ".models/vit_b_16.pth")

imagePath = "D:/AADD-Dataset/lq/img_0003.png"


# Adversarial Patch Attack Example
# orig, noise, adv = adversarial_patch_attack(resnet50_Model, imagePath)


# Auto Attack Example
# orig, noise, adv = auto_attack(
#         model=resnet50_Model,
#         image=imagePath,
#         norm='inf',      # Use 'inf' for L-infinity norm, or '2' for L2
#         eps=8.0 / 255.0, # Epsilon perturbation limit (e.g., 8 pixel values)
#         verbose=True,
#         n_classes=2 # Must match the model's output classes
# )


# Auto PGD Attack Example
# orig, noise, adv = auto_pgd_attack(
#         model=resnet50_Model,
#         image=imagePath,
#         target_label=0, # Set to None for untargeted
#         norm='inf',
#         eps=8.0 / 255.0,
#         eps_step=2.0 / 255.0, # A slightly larger step size than the default
#         max_iter=40,
#         verbose=True,
#         n_classes=2
# )



# Conjugate Gradient Attack Example
# orig, noise, adv = auto_cg_attack(
#         model=resnet50_Model,
#         image=imagePath,
#         target_label=0, # Set to an integer (e.g., 1) for a targeted attack
#         norm='inf',
#         eps=8.0 / 255.0,
#         eps_step=2.0 / 255.0,
#         max_iter=50,
#         nb_random_init=3,
#         verbose=True,
#         n_classes=2
# )


# Boundary Attack Example
# orig, noise, adv = boundary_attack(
#         model=resnet50_Model,
#         image=imagePath,
#         target_label=0, # Must provide a target label
#         max_iter=50000,
#         verbose=True,
#         n_classes=2
# )


# Carlini L_inf Attack Example
# orig, noise, adv = carlini_linf_attack(
#         model=resnet50_Model,
#         image=imagePath,
#         target_label=0,    # Set to an integer for a targeted attack
#         confidence=0.5,    # Increase confidence for a more robust attack
#         learning_rate=0.001,
#         max_iter=1000,      # More iterations generally produce better results
#         verbose=True,
#         n_classes=2
# )



# Composite Attack Example
# orig, noise, adv = composite_attack(
#         model=resnet50_Model,
#         image=imagePath,

#         #Attacks list
#         # 0: Hue, 
#         # 1: Saturation, 
#         # 2: Rotation, 
#         # 3: Brightness, 
#         # 4: Contrast, 
#         # 5: PGD(L-infinity)
#         enabled_attack=(1, 2, 3, 4, 5),
#         attack_order='scheduled', # 'scheduled', 'random', or 'fixed'
#         max_iter=10,
#         max_inner_iter=15,
#         verbose=True,
#         n_classes=2
# )


# DeepFool Attack Example
# orig, noise, adv = deepfool_attack(
#         model=resnet50_Model,
#         image=imagePath,
#         max_iter=750,
#         epsilon=1e-5, # A small overshoot factor
#         verbose=True,
#         n_classes=2
# )




# orig, noise, adv = elastic_net_attack(
#         model=resnet50_Model,
#         image=imagePath,
#         target_label=0, # Set to None for untargeted
#         confidence=0.2,
#         max_iter=150,
#         beta=0.01,      # Try changing this to 0.1 or higher
#         verbose=True,
#         n_classes=2
# )

















cv2.imshow("original", cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
cv2.imshow("noise (amplified)", cv2.cvtColor((noise + 127).clip(0,255).astype("uint8"), cv2.COLOR_RGB2BGR))
cv2.imshow("adversarial", cv2.cvtColor(adv, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
