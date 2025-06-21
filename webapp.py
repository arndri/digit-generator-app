import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os

# ----- Constants matching training -----
n_classes = 10
latent_dim = 100
img_shape = (1, 28, 28)
MODEL_PATH = "cgan_generator.pth"

# ----- Generator Architecture -----
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        embedded_labels = self.label_embedding(labels)
        gen_input = torch.cat((embedded_labels, noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img

# ----- Load generator model -----
@st.cache_resource
def load_generator():
    if os.path.exists(MODEL_PATH):
        model = Generator()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        return None

generator = load_generator()

# ----- Streamlit UI -----
st.title("🧠 Handwritten Digit Generator (PyTorch)")

if generator is None:
    st.warning("⚠️ The generator model file `cgan_generator.pth` is missing. Please train and save it first.")
else:
    digit = st.selectbox("Select a digit (0-9):", list(range(n_classes)))
    generate = st.button("Generate Images")

    if generate:
        st.subheader(f"Generated Digit: {digit}")
        noise = torch.randn(5, latent_dim)
        labels = torch.tensor([digit] * 5, dtype=torch.long)

        with torch.no_grad():
            gen_imgs = generator(noise, labels).detach().cpu()

        gen_imgs = (gen_imgs + 1) / 2.0  # Rescale [-1, 1] → [0, 1]

        cols = st.columns(5)
        for i in range(5):
            img = gen_imgs[i].squeeze().numpy()
            img = Image.fromarray(np.uint8(img * 255), mode='L')
            cols[i].image(img, use_column_width=True)
