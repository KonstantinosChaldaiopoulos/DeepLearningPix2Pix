from imports import *
from models import *

transform = transforms.ToTensor()
pix2pixA_folder_path = '/content/drive/MyDrive/Pix2PixA_test'
pix2pixB_folder_path = '/content/drive/MyDrive/Pix2PixB_t'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G = UNetGenerator().to(device)

model_path = 'generator_19.pth'
G.load_state_dict(torch.load(model_path, map_location=device))

for filename in os.listdir(pix2pixA_folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path_A = os.path.join(pix2pixA_folder_path, filename)
        img_path_B = os.path.join(pix2pixB_folder_path, filename)

        img_A = Image.open(img_path_A).convert('RGB')
        img_B = Image.open(img_path_B).convert('RGB')

        input_image = transform(img_A).unsqueeze(0).to(device)

        with torch.no_grad():
            output_image = G(input_image)

        output_image = output_image.squeeze().permute(1, 2, 0).cpu().numpy()
        output_image = np.clip(output_image, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_A)
        axes[0].set_title('Original Image (A)')
        axes[0].axis('off')
        axes[1].imshow(output_image)
        axes[1].set_title('Generated Image')
        axes[1].axis('off')
        axes[2].imshow(img_B)
        axes[2].set_title('Real Salient Map (B)')
        axes[2].axis('off')
        plt.suptitle(f'Comparison for {filename}')
        plt.show()
