from imports import *
from models import *
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
from scipy.ndimage import gaussian_filter

def compute_NSS(pred_map, gt_map):
    gt_map = resize(gt_map, pred_map.shape)
    pred_map = (pred_map - np.mean(pred_map)) / np.std(pred_map)
    gt_map = (gt_map > 0.5).astype(int)
    NSS_score = np.mean(pred_map[gt_map == 1])
    return NSS_score

transform = transforms.ToTensor()
pix2pixA_folder_path = '/content/drive/MyDrive/Pix2PixA_test'
pix2pixB_folder_path = '/content/drive/MyDrive/Pix2PixB_t'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G = UNetGenerator().to(device)

model_path = 'generator_99.pth'
G.load_state_dict(torch.load(model_path, map_location=device))

total_nss_score = 0.0
num_images = 0

for filename in os.listdir(pix2pixA_folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path_A = os.path.join(pix2pixA_folder_path, filename)
        img_path_B = os.path.join(pix2pixB_folder_path, filename)

        img_A = Image.open(img_path_A).convert('RGB')
        img_B = Image.open(img_path_B).convert('L')  # Convert to grayscale if not

        input_image = transform(img_A).unsqueeze(0).to(device)

        with torch.no_grad():
            output_image = G(input_image)

        output_image = output_image.squeeze().permute(1, 2, 0).cpu().numpy()
        output_image = np.clip(output_image, 0, 1)

        pred_saliency_map = output_image[:, :, 0]  # Assuming saliency map is grayscale or red channel in RGB
        real_saliency_map = np.array(img_B)

        NSS_score = compute_NSS(pred_saliency_map, real_saliency_map)
        print(f'NSS score for {filename}: {NSS_score}')

        total_nss_score += NSS_score
        num_images += 1

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_A)
        axes[0].set_title('Original Image (A)')
        axes[0].axis('off')
        axes[1].imshow(output_image)
        axes[1].set_title('Generated Image')
        axes[1].axis('off')
        axes[2].imshow(img_B, cmap='gray')
        axes[2].set_title('Real Salient Map (B)')
        axes[2].axis('off')
        plt.suptitle(f'Comparison for {filename}')
        plt.show()

average_nss_score = total_nss_score / num_images
print(f'Average NSS score: {average_nss_score}')
