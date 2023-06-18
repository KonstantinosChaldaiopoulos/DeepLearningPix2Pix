from imports import *
from models import *
transform = transforms.ToTensor()
folder_path = '/content/drive/MyDrive/Pix2PixA_test'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G = UNetGenerator().to(device)


model_path = 'generator_1.pth'
G.load_state_dict(torch.load(model_path, map_location=device))


for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert('RGB')

        input_image = transform(img).unsqueeze(0).to(device)


        with torch.no_grad():
            output_image = G(input_image)


        output_image = output_image.squeeze().permute(1, 2, 0).cpu().numpy()


        print(f'Generated image for file: {filename}')
        plt.imshow(output_image)
        plt.title(f'Generated image for {filename}')
        plt.show()