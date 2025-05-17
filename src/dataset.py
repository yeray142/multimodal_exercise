import torch
import torchaudio
import os

import pandas as pd

from torch.utils.data import Dataset
from PIL import Image


class MultimodalDataset(Dataset):
	"""Dataset for multimodal age classification with image, audio and text."""

	def __init__(self, data_dir, annotations_file, image_transform=None, audio_transform=None, split="train"):
		"""
		Args:
			data_dir (string): Directory with all the data files
			annotations_file (string): CSV file with annotations (image_path, audio_path, transcription, age_label)
			image_transform (callable, optional): Optional transform to be applied on images
			audio_transform (callable, optional): Optional transform to be applied on audio
		"""
		self.data_dir = data_dir
		self.annotations = pd.read_csv(annotations_file)
		self.image_transform = image_transform
		self.audio_transform = audio_transform

		self.split = split

	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, idx):
		# Get paths from annotations
		video_name = '.'.join(self.annotations.iloc[idx, 0].split('.')[:2])
		age_group = self.annotations.iloc[idx, 2]

		# Get the image, which should be in data_dir/<split>/<age_group>/<video_name>.jpg
		# and the audio, which should be in data_dir/<split>/<age_group>/<video_name>.wav
		img_path = os.path.join(self.data_dir, self.split, str(age_group), f'{video_name}.jpg')
		audio_path = os.path.join(self.data_dir, self.split, str(age_group), f'{video_name}.wav')
		text_path = os.path.join(self.data_dir, self.split, str(age_group), f'{video_name}.pkl')

		# Load transcription pkl file
		transcription = pd.read_pickle(text_path)

		# Load image
		image = Image.open(img_path).convert('RGB')
		if self.image_transform:
			image = self.image_transform(image)

		# Load audio
		waveform, sample_rate = torchaudio.load(audio_path)
		if self.audio_transform:
			waveform = self.audio_transform(waveform)

		return {
			'image': image,
			'audio': waveform,
			'text': transcription,
			'label': torch.tensor(age_group, dtype=torch.long)
		}

def collate_fn(batch):
	# Get max audio length in this batch
	max_audio_len = max([x['audio'].shape[1] for x in batch])

	# Pad audio sequences to the same length
	padded_audio = []
	for x in batch:
		audio = x['audio']
		padding_length = max_audio_len - audio.shape[1]

		# Pad with zeros at the end
		padded = torch.nn.functional.pad(audio, (0, padding_length))
		padded_audio.append(padded)

	return {
        'image': torch.stack([x['image'] for x in batch]),
        'audio': torch.stack(padded_audio),
		'text': [x['text'] for x in batch],
		'label': torch.stack([x['label'] for x in batch])
	}