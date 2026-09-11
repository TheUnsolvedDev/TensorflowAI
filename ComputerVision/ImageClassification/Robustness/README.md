# 🛡️ Image-Classification Robustness

← [Image classification](../README.md) · [Repository](../../../README.md)

This folder contains standalone TensorFlow attack/interpretability scripts:
`adv_sal.py`, `black_box.py`, `fgsm.py`, and `illm.py`. They use model
gradients or score queries to construct adversarial perturbations or saliency
signals; they are not a shared benchmark suite.

| Script | Technique represented in source | Core operation |
| --- | --- | --- |
| `adv_sal.py` | Adversarial saliency map | Differentiate selected class scores with respect to input features |
| `fgsm.py` | Fast Gradient Sign Method | Add an epsilon-scaled sign of the input gradient |
| `illm.py` | Iterative least-likely method | Iteratively optimise an input toward a least-likely target |
| `black_box.py` | Black-box attack experiments | Query-driven perturbation/search code with TensorFlow variables |

For an input (x), loss (L), and perturbation budget \(\epsilon\), FGSM is
the familiar tensor update \(x' = x + \epsilon\,\mathrm{sign}(\nabla_x L)\).
The scripts use `tf.GradientTape` where gradients are available. Their
computational bottleneck is repeated forward/backward evaluation, so batches
and GPU placement matter more than Python control flow.

No retained accuracy/attack-success table is present; use the scripts as
implementation studies and record any new evaluation externally.
