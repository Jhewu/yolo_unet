import matplotlib.pyplot as plt
import numpy as np 
import cv2

class GaussianNoisePerChannel:
    def __init__(self, p=0.5, noise_std_range=(0.01, 0.05)):
        self.p = p
        self.noise_std_range = noise_std_range

    def __call__(self, labels):
        if np.random.rand() > self.p:
            return labels

        img = labels["img"]  # Shape: (H, W, C)
        
        # Apply different noise std per channel
        noise_std = np.random.uniform(*self.noise_std_range, size=img.shape[-1])
        
        # Add Gaussian noise per channel
        noise = np.random.normal(0, noise_std, img.shape).astype(img.dtype)
        img = np.clip(img + noise, 0, 1)

        labels["img"] = img
        return labels

class RandomResolution:
    def __init__(self, p=0.25, scale_range=(0.6, 0.9)):
        self.p = p
        self.scale_range = scale_range

    def __call__(self, labels):
        if np.random.rand() > self.p:
            return labels

        img = labels["img"]
        h, w = img.shape[:2]
        scale = np.random.uniform(*self.scale_range)
        new_h, new_w = int(h * scale), int(w * scale)

        # Downscale
        img_low = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Upscale back
        img = cv2.resize(img_low, (w, h), interpolation=cv2.INTER_LINEAR)

        labels["img"] = img
        return labels

class MildGaussianBlur:
    def __init__(self, p=0.3, kernel_size=3, sigma_range=(0.5, 1.5)):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range

    def __call__(self, labels):
        if np.random.rand() > self.p:
            return labels

        img = labels["img"]
        sigma = np.random.uniform(*self.sigma_range)
        
        # Apply Gaussian blur per channel
        for c in range(img.shape[2]):
            img[:, :, c] = cv2.GaussianBlur(img[:, :, c], (self.kernel_size, self.kernel_size), sigma)
        
        labels["img"] = img
        return labels

class RandomBiasField:
    def __init__(self, p=0.3, alpha_range=(0.1, 0.5)):
        self.p = p
        self.alpha_range = alpha_range

    def __call__(self, labels):
        if np.random.rand() > self.p:
            return labels

        img = labels["img"]
        h, w = img.shape[:2]
        
        # Random center position
        center_x = np.random.uniform(-0.5, 0.5)  # Keep it closer to center
        center_y = np.random.uniform(-0.5, 0.5)
        
        # Random strength
        alpha = np.random.uniform(*self.alpha_range)
        
        # Random elliptical shape
        scale_x = np.random.uniform(0.5, 2.0)
        scale_y = np.random.uniform(0.5, 2.0)
        
        # Create coordinate grids (normalized to -1, 1)
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)
        
        # Shift and scale coordinates
        X_shifted = (X - center_x) * scale_x
        Y_shifted = (Y - center_y) * scale_y
        
        # Create bias field
        bias = 1 + alpha * (X_shifted**2 + Y_shifted**2)
        
        # Randomly invert (bright center vs bright edges)
        if np.random.rand() > 0.5:
            bias = 2 - bias
            
        # Clip to reasonable range
        bias = np.clip(bias, 0.7, 1.3)
        
        # Apply to all channels (broadcasting)
        img = img * bias[..., None]  # [..., None] adds channel dimension
        
        # Clip final image
        img = np.clip(img, 0, 1)
        labels["img"] = img
        return labels

class RandomBiasField:
    def __init__(self, p=0.3, alpha_range=(0.1, 0.5), smoothness=0.3):
        self.p = p
        self.alpha_range = alpha_range
        self.smoothness = smoothness  # Controls how smooth the field is

    def __call__(self, labels):
        if np.random.rand() > self.p:
            return labels

        img = labels["img"]
        h, w = img.shape[:2]
        
        # Random center position (-1 to 1 range)
        center_x = np.random.uniform(-1, 1)
        center_y = np.random.uniform(-1, 1)
        
        # Random alpha strength
        alpha = np.random.uniform(*self.alpha_range)
        
        # Random elliptical shape (different scaling for x and y)
        scale_x = np.random.uniform(0.5, 2.0)
        scale_y = np.random.uniform(0.5, 2.0)
        
        # Create coordinate grids
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)
        
        # Shift coordinates to random center
        X_shifted = (X - center_x) * scale_x
        Y_shifted = (Y - center_y) * scale_y
        
        # Create smoother elliptical bias field with Gaussian smoothing
        # Use a more gradual transition
        distance_squared = X_shifted**2 + Y_shifted**2
        
        # Apply smoother bias field - reduce sharpness
        if self.smoothness > 0:
            # Apply Gaussian kernel to smooth the field
            bias = 1 + alpha * np.exp(-distance_squared / (2 * self.smoothness**2))
        else:
            # Original quadratic (but with better bounds)
            bias = 1 + alpha * distance_squared
        
        # Randomly invert the bias field (simulates different coil effects)
        if np.random.rand() > 0.5:
            bias = 2 - bias  # Invert: now stronger at center, weaker at edges
            
        # Clip to reasonable range
        bias = np.clip(bias, 0.5, 1.5)
        
        # Apply to all channels
        img = img * bias[..., None]
        img = np.clip(img, 0, 1)

        labels["img"] = img
        return labels

# Create a test image (4-channel)
def create_test_image(height=256, width=256):
    # Create 4 different patterns to simulate different MRI modalities
    img = np.zeros((height, width, 4), dtype=np.float32)
    
    # Channel 0: Brain-like structure
    y, x = np.ogrid[:height, :width]
    center_dist = np.sqrt((x - width//2)**2 + (y - height//2)**2)
    img[:, :, 0] = np.exp(-((center_dist - 80)**2) / (2 * 30**2))
    
    # Channel 1: Another structure
    img[:, :, 1] = np.exp(-((x - width//3)**2 + (y - height//3)**2) / (2 * 40**2))
    
    # Channel 2: Yet another structure
    img[:, :, 2] = np.exp(-((x - 2*width//3)**2 + (y - 2*height//3)**2) / (2 * 50**2))
    
    # Channel 3: Uniform background with some noise
    img[:, :, 3] = 0.2 + 0.1 * np.random.rand(height, width)
    
    return np.clip(img, 0, 1)


if __name__ == "__main__": 

    # Test labels dict (like what YOLO expects)
    test_labels = {
        "img": create_test_image(),
        "cls": np.array([0]),  # class labels
        "bboxes": np.array([[0.4, 0.4, 0.2, 0.2]]),  # normalized xywh format
        "keypoints": None
    }

    # Test your augmentation
    augmentation = RandomBiasField(p=1.0, alpha_range=(0.1, 0.3))
    # augmentation = GaussianNoisePerChannel(p=1)
    # augmentation = RandomResolution(p=1)
    # augmentation = MildGaussianBlur(p=1)

    # Apply augmentation
    augmented_labels = augmentation(test_labels.copy())

    # Visualize results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i in range(4):
        # Original
        axes[0, i].imshow(test_labels["img"][:, :, i], cmap='gray')
        axes[0, i].set_title(f'Original Channel {i}')
        axes[0, i].axis('off')
        
        # Augmented
        axes[1, i].imshow(augmented_labels["img"][:, :, i], cmap='gray')
        axes[1, i].set_title(f'Augmented Channel {i}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()