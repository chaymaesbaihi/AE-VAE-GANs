# AE-VAE-GANs

## Objective
The main objective of this lab is to gain familiarity with the PyTorch library and build deep neural network architectures for Auto-encoders, Variational Auto-encoders (VAEs), and Generative Adversarial Networks (GANs) in the context of generative artificial intelligence.

## Part 1: Auto-encoder (AE) and Variational Auto-encoder (VAE)

### Dataset: [MNIST Dataset] => (https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

1. Auto-encoder (AE)
   - Establish an auto-encoder architecture.
   - Train the model on the MNIST dataset.
   - Specify the best hyper-parameters.

2. Variational Auto-encoder (VAE)
   - Establish a variational auto-encoder architecture.
   - Train the model on the MNIST dataset.
   - Specify the best hyper-parameters.

3. Evaluation
   - Evaluate both models by plotting metrics such as loss, KL divergence,

4. Latent Space Visualization
   - Plot the latent space of both AE and VAE models.

## Part 2: Generative Adversarial Networks (GANs)

### Dataset: [Abstract Art Gallery Dataset] => (https://www.kaggle.com/datasets/bryanb/abstract-art-gallery)

1. Model Definition and Training
   - Define a Generator, Discriminator, and Loss Function.
   - Initialize Generator and Discriminator.
   - Set up GPU settings.
   - Configure data loader.
   - Define optimizers and perform training.

2. Evaluation
   - Plot losses for both the Generator and Discriminator.

3. Data Generation
   - Generate new data using the trained GAN.
   - Compare the quality of generated data with the original dataset.


### Dependencies

- PyTorch
- Matplotlib
- NumPy
- torchvision

Install dependencies using:

```bash
pip install torch matplotlib numpy torchvision
