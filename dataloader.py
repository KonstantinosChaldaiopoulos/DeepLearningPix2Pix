def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

class PairedDataset(Dataset):
    def __init__(self, domain_a_path, domain_b_path, transform=None):
        domain_a_folders = ["Action", "Affective", "Art", "BlackWhite", "Cartoon", "Fractal",
                            "Indoor", "Inverted", "Jumbled", "LineDrawing",
                            "LowResolution", "Noisy", "Object", "OutdoorManMade", "OutdoorNatural",
                            "Pattern", "Random", "Satelite", "Sketch", "Social"]

        domain_b_folders = ["Output1","Output2","Output3", "Output4", "Output5", "Output6", "Output7", "Output8",
                            "Output9", "Output10", "Output11", "Output12", "Output13", "Output14",
                            "Output15", "Output16", "Output17", "Output18", "Output19", "Output20"]

        self.pairs = []

        for folder_a, folder_b in zip(domain_a_folders, domain_b_folders):
            folder_a_path = os.path.join(domain_a_path, folder_a)
            folder_b_path = os.path.join(domain_b_path, folder_b)

            images_a = sorted(get_image_files(folder_a_path).values(), key=natural_sort_key)
            images_b = sorted(get_image_files(folder_b_path).values(), key=natural_sort_key)

            for img_a, img_b in zip(images_a, images_b):
                self.pairs.append((img_a, img_b))

        self.transform = transform

    def __getitem__(self, index):
        img_a_path, img_b_path = self.pairs[index]

        img_a = Image.open(img_a_path).convert("RGB")
        img_b = Image.open(img_b_path).convert("RGB")

        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)

        return img_a, img_b

    def __len__(self):
        return len(self.pairs)