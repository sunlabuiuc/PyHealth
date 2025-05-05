import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2

from pyhealth.models.retinaunet import RetinaUNet

"""
ToyDataset is based on the testing dataset used for Retina U-Net model by the paper's authors.
It creates images with a randomly sized circle to imitate a labeled region on a medical image. Noise in introduced
to prevent overfitting and create a more realistic approach. A mask image of the first image is also created for
visual comparison during testing.
"""
class ToyDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=(320, 320), noise_factor=0.2, transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.noise_factor = noise_factor
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create a blank image (black background)
        image = np.zeros((320, 320, 3), dtype=np.uint8)

        # Randomly choose shape type: 0 = circle, 1 = donut
        shape_type = random.randint(0, 1)

        # Random center and radius
        center_x = random.randint(64, 256)
        center_y = random.randint(64, 256)
        radius = random.randint(20, 40)

        # Define custom colors
        color = (random.randint(0, 100), random.randint(100, 255), random.randint(200, 255))

        # Draw the shape on the image
        if shape_type == 0:
            # Draw filled circle
            cv2.circle(image, (center_x, center_y), radius, color, -1)
        else:
            # Draw ring
            cv2.circle(image, (center_x, center_y), radius, color, 3)
            cv2.circle(image, (center_x, center_y), radius - 10, (0, 0, 0), -1)

        # Generate noise with a blue hue, but with a slightly darker blue background
        noise = np.random.uniform(low=0, high=self.noise_factor, size=(320, 320, 3))

        # Create a background that is a little darker than the circle's color
        noise[..., 0] += np.random.uniform(0.0, 0.1, (320, 320))  # Slightly darker blue
        noise[..., 1] += np.random.uniform(0.0, 0.2, (320, 320))  # Slight green variation
        noise[..., 2] += np.random.uniform(0.1, 0.3, (320, 320))  # Slight red variation

        # Clip the values to ensure they stay within the valid image range (0-255)
        noisy_image = np.clip(image + noise * 255, 0, 255).astype(np.uint8)

        # Create the mask (ground truth)
        mask = np.zeros((320, 320), dtype=np.uint8)
        if shape_type == 0:
            # Circle mask
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        else:
            # Donut mask
            cv2.circle(mask, (center_x, center_y), radius, 255, 3)
            cv2.circle(mask, (center_x, center_y), radius - 10, 0, -1)


        # Normalize image and mask
        noisy_image = noisy_image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # Convert to tensors
        noisy_image = torch.from_numpy(noisy_image.transpose((2, 0, 1)))
        mask = torch.from_numpy(mask).unsqueeze(0)

        return noisy_image, mask

# Training loop
def train(model, dataloader, optimizer, device, num_epochs=5, dropout_p=0.5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (imgs, masks) in enumerate(dataloader):
            imgs = imgs.to(device)
            masks = masks.to(device)

            # Apply input dropout
            if dropout_p > 0:
                imgs = F.dropout(imgs, p=dropout_p, training=True)

            # Ensure proper mask dimensions
            if masks.ndimension() == 3:
                masks = masks.unsqueeze(1)

            # Resize masks and predictions
            masks_resized = F.interpolate(masks.float(), size=(320, 320), mode='nearest')
            optimizer.zero_grad()
            outputs = model(imgs)
            outputs_resized = F.interpolate(outputs, size=(320, 320), mode='nearest')

            # Compute and optimize loss
            loss = criterion(outputs_resized, masks_resized)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'>> Epoch {epoch+1}/{num_epochs}, Average Loss: {running_loss/len(dataloader):.4f}\n')


# Evaluating the Dice Score between the Ground Truth Mask and the Model's Prediction
def evaluate(model, loader, device):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device).float()

            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).float()

            # Calculate Dice score for each sample
            intersection = (preds * masks).sum(dim=(1,2,3))
            union = preds.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3))
            dice = (2. * intersection) / (union + 1e-8)
            dice_scores.extend(dice.cpu().numpy())

    # Compute average Dice score
    mean_dice = sum(dice_scores) / len(dice_scores)
    print(f"Mean Dice score: {mean_dice:.4f}")

# Create full dataset
full_dataset = ToyDataset(num_samples=1000)

# Split into train/val
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

# Create model
model = RetinaUNet()
# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)
# Set up optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()
# Run training, visualizing output and evaluation
train(model, train_loader, optimizer, device, num_epochs=5, dropout_p=0.3)
evaluate(model, val_loader, device)