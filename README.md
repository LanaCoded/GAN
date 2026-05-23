# MNIST Generative Adversarial Network (GAN) Portfolio

A complete, self-contained generative adversarial network (GAN) architecture implemented within a single, end-to-end Python environment. This project handles the entire deep learning lifecycle—from raw dataset streaming to adversarial optimization loops—to map continuous latent space noise distributions into realistic $28 \times 28$ handwritten digit matrices.

## 🪐 Mathematical Framework

The system operates as a zero-sum minimax optimization game between two neural networks using binary cross-entropy loss fields:

$$\min_{G} \max_{D} V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_{z}(z)}[\log(1 - D(G(z)))]$$

* **Generator ($G$):** Projects a 100-dimensional continuous noise vector $z \sim p_z(z)$ into synthetic $28 \times 28 \times 1$ image tensors using dense upsampling layers.
* **Discriminator ($D$):** Acts as a binary classifier scoring target image authenticity ($0.0 \le P \le 1.0$) to determine if a digit matrix is authentic or generated.

---

## 🛠️ Pipeline Architecture

The implementation is structured into clean, sequential execution boundaries within a single file workspace:

1. **Dataset Ingestion:** Streams the native MNIST tensor arrays directly into cache memory fields using `torchvision.datasets` with normalized transformation tensors.
2. **Model Graph Definitions:** Direct object-oriented PyTorch subclasses (`torch.nn.Module`) defining the dimensional layers for both the Generator and Discriminator network architectures.
3. **Adversarial Optimization Engine:** Configured with twin Adam optimizers ($\text{learning rate} = 2\times10^{-4}$, $\beta_1 = 0.5$) to synchronize gradient updates simultaneously.
4. **Inline Metrics Evaluation:** Generates and plots a visual grid of synthetic digits at the end of every epoch boundary using `matplotlib` to evaluate structural formation changes over time.

---

## 📊 Stabilization Design Choices

Standard GAN training is highly susceptible to structural optimization failures. This pipeline implements specific defenses to ensure training stability:
* **LeakyReLU Activations:** Configured with a negative slope coefficient of $0.2$ across both networks to prevent vanishing gradient problems.
* **One-Sided Label Smoothing:** Real image target arrays are smoothed down to $0.9$ to prevent the discriminator from becoming overconfident early in the training loop.
* **Hyperparameter Tuning:** Strictly adheres to Deep Convolutional GAN (DCGAN) structural defaults to eliminate mode collapse risks.
