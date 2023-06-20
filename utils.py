import os

def create_resized_copy(input_folder, output_folder, new_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        for directory in dirs:
            input_dir_path = os.path.join(root, directory)
            output_dir_path = input_dir_path.replace(input_folder, output_folder)
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_file_path = os.path.join(root, file)
                output_file_path = input_file_path.replace(input_folder, output_folder)

                img = Image.open(input_file_path)
                img_resized = img.resize(new_size, Image.ANTIALIAS)
                img_resized.save(output_file_path)


def get_image_files(folder):
    image_files = {}
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files[file] = os.path.join(root, file)
    return image_files

def extract_number(filename):
    num = re.findall(r'\d+', filename)
    return int(num[0]) if num else None


def resize(a,b):
    input_folder_a = '/content/drive/MyDrive/Pix2PixA'
    input_folder_b = '/content/drive/MyDrive/Pix2PixB'
    output_folder_a = '/content/drive/MyDrive/Pix2PixA_train'
    output_folder_b = '/content/drive/MyDrive/Pix2PixB_train'
    new_size = (a, b)

    create_resized_copy(input_folder_a, output_folder_a, new_size)
    create_resized_copy(input_folder_b, output_folder_b, new_size)

    print(f"Resized images saved in {output_folder_a} and {output_folder_b}")



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('InstanceNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)

class ImageBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.images = []

    def push_and_pop(self, image):
        if len(self.images) < self.buffer_size:
            self.images.append(image)
            return image
        else:
            if torch.rand(1).item() > 0.5:
                i = torch.randint(low=0, high=self.buffer_size - 1, size=(1,)).item()
                tmp = self.images[i]
                self.images[i] = image
                return tmp
            else:
                return image
