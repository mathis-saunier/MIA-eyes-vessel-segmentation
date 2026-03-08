from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageOps
import math

class VesselDataset(Dataset):
    def __init__(
        self,
        image_paths,
        label_paths,
        transform=None,
        image_transform=None,
        label_transform=None,
        use_patches=False,
        patch_size=512,
        return_metadata=False,
        pad_if_needed=True,
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
        self.image_transform = image_transform if image_transform is not None else transform
        self.label_transform = label_transform if label_transform is not None else transform
        self.use_patches = use_patches
        self.patch_size = patch_size
        self.return_metadata = return_metadata
        self.pad_if_needed = pad_if_needed

        self.samples = self._build_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_index = sample["image_index"]

        image_path = self.image_paths[image_index]
        image = self.load_img(image_path)

        label_path = self.label_paths[image_index]
        label = self.load_label(label_path)

        metadata = {
            "image_path": image_path,
            "label_path": label_path,
            "desease": self.find_desease_from_name(image_path),
        }

        if self.use_patches:
            original_width, original_height = image.size
            padded_width, padded_height = sample["padded_size"]

            if self.pad_if_needed and (padded_width != original_width or padded_height != original_height):
                image = ImageOps.expand(image, border=(0, 0, padded_width - original_width, padded_height - original_height), fill=0)
                label = ImageOps.expand(label, border=(0, 0, padded_width - original_width, padded_height - original_height), fill=0)

            left, top = sample["top_left"]
            right = left + self.patch_size
            bottom = top + self.patch_size
            image = image.crop((left, top, right, bottom))
            label = label.crop((left, top, right, bottom))

            metadata.update({
                "patch_index": sample["patch_index"],
                "num_patches": sample["num_patches"],
                "top_left": sample["top_left"],
                "original_size": sample["original_size"],
                "padded_size": sample["padded_size"],
                "patch_size": self.patch_size,
            })

        if self.image_transform:
            image = self.image_transform(image)

        if self.label_transform:
            label = self.label_transform(label)

        if self.return_metadata:
            return image, label, metadata

        return image, label, metadata["desease"]

    def _build_samples(self):
        if not self.use_patches:
            return [{"image_index": image_index} for image_index in range(len(self.image_paths))]

        samples = []
        for image_index, image_path in enumerate(self.image_paths):
            with Image.open(image_path) as image:
                width, height = image.size

            if self.pad_if_needed:
                padded_width = int(math.ceil(width / self.patch_size) * self.patch_size)
                padded_height = int(math.ceil(height / self.patch_size) * self.patch_size)
            else:
                padded_width = width
                padded_height = height

            x_positions = list(range(0, padded_width, self.patch_size))
            y_positions = list(range(0, padded_height, self.patch_size))
            num_patches = len(x_positions) * len(y_positions)

            patch_index = 0
            for top in y_positions:
                for left in x_positions:
                    samples.append({
                        "image_index": image_index,
                        "patch_index": patch_index,
                        "num_patches": num_patches,
                        "top_left": (left, top),
                        "original_size": (width, height),
                        "padded_size": (padded_width, padded_height),
                    })
                    patch_index += 1

        return samples

    def load_img(self, file_path):
        return Image.open(file_path).convert('RGB')

    def load_label(self, file_path):
        # In grayscale (1 canal)
        return Image.open(file_path).convert('L')

    def find_desease_from_name(self, file_name):
        # On enlève l'extension en .png et on regarde le dernier caractère (N : normal, A : age-related macular degeneration, G : glaucoma, D : diabetic retinopathy)
        letter = file_name.replace(".png", "")[-1]
        if letter not in ["N", "A", "G", "D"]:
            raise ValueError(f"La lettre {letter} n'est pas reconue")
        else:
            return letter