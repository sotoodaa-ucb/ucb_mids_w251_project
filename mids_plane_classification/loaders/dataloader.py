import glob
import os
import shutil
import zipfile
import pandas as pd
import numpy as np
import torch
import wget
from albumentations import Compose as ACompose
from mids_plane_classification.utils.progress import progress_bar
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms.transforms import Compose


class PlaneImageFolder(ImageFolder):
    def __init__(self, root):
        super(PlaneImageFolder, self).__init__(root)


class PlaneDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]

        # Annoyingly, albumentations and Pytorch Compose
        # transforms have different argument requirements.
        if self.transform:
            if isinstance(self.transform, Compose):
                x = self.transform(image)
            elif isinstance(self.transform, ACompose):
                x = self.transform(image=np.array(image))['image']
        else:
            x = image
        y = label
        return x, y

    def __len__(self):
        return len(self.dataset)


class PlaneDataModule:
    """Data module used to preprocess and store relevant data objects.

    Args:
        train_batch_size (int): Train DataLoader batch size.
        val_batch_size (int): Validation DataLoader batch size.
        train_ratio (float, optional): Ratio to apply for train validation split.
            Should be a fractional value [0, 1]. Defaults to 0.8.
        data_dir (str, optional): Where the data is stored or where the prepare_data
            method will store the data on download. Defaults to './data'.
        num_workers (int, optional): How many threads to use during training.
            Defaults to 8.
        train_transform (Compose, optional): Transforms to apply as a Compose.
            Supports both Albumentations and Pytorch composes. Defaults to None.
        val_transform (Compose, optional): Transforms to apply as a Compose.
            Supports both Albumentations and Pytorch composes. Defaults to None.
        seed (int, optional): Seed value for reproducibility and debugging. Defaults to None.
    """
    def __init__(
        self,
        train_batch_size: int,
        val_batch_size: int,
        train_ratio: float = 0.8,
        data_dir: str = './data',
        num_workers: int = 8,
        train_transform: Compose = None,
        val_transform: Compose = None,
        seed: int = None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_ratio = train_ratio
        self.seed = seed

        # Set seed if provided for reproducibility.
        if seed: torch.manual_seed(seed)

        # Apply transforms.
        self.transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ]) if not train_transform else train_transform
        self.transform_val = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ]) if not val_transform else val_transform
        self.num_workers = num_workers

    def prepare_data(self, cleanup: bool = True) -> None:
        """Perform various preprocessing steps here.

        Checks if the dataset exists locally. If not, then the download
        and extraction is performed for you.  This method only needs
        to be called once for a specified location.

        1. Checks if the labels are downloaded, if not download them.
        2. Checks if the images zip is downloaded, if not download it.
        3. Checks if the images have been unzipped, if not then unzip them.
        4. Remove various duplicates in the dataset.
           The dataset is relatively unclean and has duplicate images stored
           in different folders. Since ImageFolder is used, all folders in
           the data_dir are treated as separate classes. This will perform
           removal of duplicate folders until there are only 11 classes remaining,
           which are labeled according to the annotations csv.

        Args:
            cleanup (bool) - Boolean flag to indicate whether or not to
            perform the cleanup step to delete duplicates or excess folders.
        """
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        labels_path = os.path.join(self.data_dir, 'MTARSI.csv')
        images_zip_path = os.path.join(self.data_dir, 'airplane-dataset-asoc.zip')
        images_path = self.data_dir
        if not os.path.exists(labels_path):
            wget.download(
                'https://zenodo.org/record/3464319/files/MTARSI.csv?download=1',
                out=labels_path,
                bar=progress_bar
            )

        if not os.path.exists(images_zip_path):
            wget.download(
                'https://zenodo.org/record/3464319/files/airplane-dataset-asoc.zip?download=1',
                out=images_zip_path,
                bar=progress_bar
            )

        labels = pd.read_csv(labels_path)['Band'].unique()
        self.num_classes = len(labels)

        # Parse the zip archive, and extract images into their respective subdirs.
        with zipfile.ZipFile(images_zip_path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                # Ignore folders.
                if file[-1] == '/':
                    continue

                # Read byte contents.
                data = zip_ref.read(file)

                # Only extract files that are 3 levels deep in the tree.
                fields = file.split('/')
                if len(fields) == 3:
                    _, folder_name, file_name = fields
                    # There is a typo in the image folder, where KC-135 is incorrectly named as C-135.
                    folder_name = 'KC-135' if folder_name == 'C-135' else folder_name

                    # Only extract the folders contained in the annotations csv.
                    if folder_name in labels:
                        folder_path = os.path.join(self.data_dir, folder_name)
                        output_path = os.path.join(folder_path, file_name)

                        # Create the class subdir if it does not exist.
                        if not os.path.exists(folder_path):
                            os.mkdir(folder_path)

                        new_file = open(output_path, 'wb')
                        new_file.write(data)
                        new_file.close()

        # Perform backup cleanup if specified with flag.
        if cleanup:
            for file_or_dir in glob.glob(f'{images_path}/*'):
                # Ignore files.
                if file_or_dir[-1] != '/':
                    continue
                name = os.path.basename(file_or_dir)
                # Remove all the extraneous folders that are not part of the annotations.
                if name not in labels:
                    shutil.rmtree(file_or_dir)

    def setup(self) -> None:
        """Performs train and validation split and applies transforms.

        Split the dataset, builds the dataset as an ImageFolder, and applies
        the transforms initialized in the constructor.  Should be called after
        prepare_data().
        """
        self.dataset = PlaneImageFolder(self.data_dir)

        train_size = self.train_ratio
        self.train_count = int(train_size * len(self.dataset))
        self.val_count = len(self.dataset) - self.train_count

        # Randomly split.
        train, val = random_split(self.dataset, [self.train_count, self.val_count])

        # Assign train/val datasets for use in dataloaders
        self.train_dataset = PlaneDataset(train, transform=self.transform_train)
        self.val_dataset = PlaneDataset(val, transform=self.transform_val)

    def train_dataloader(self) -> DataLoader:
        self._validate_setup_called()

        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        self._validate_setup_called()

        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers
        )

    def _validate_setup_called(self) -> None:
        if not hasattr(self, 'train_dataset'):
            raise RuntimeError('Dataloader called before setup!')
