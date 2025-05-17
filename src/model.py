import torch
import torch.nn as nn

from modules import ImageEncoder, AudioEncoder, TextEncoder


class MultimodalAgeClassifier(nn.Module):
	"""Multimodal age classifier combining image, audio, and text features."""

	def __init__(self, num_classes=8, feature_dim=128):
		super(MultimodalAgeClassifier, self).__init__()
		# Encoders for each modality
		self.image_encoder = ImageEncoder(output_dim=feature_dim)
		self.audio_encoder = AudioEncoder(output_dim=feature_dim)
		self.text_encoder = TextEncoder(output_dim=feature_dim)

		# Late fusion: concatenate features and classify
		self.fusion = nn.Sequential(
			nn.Linear(feature_dim * 3, 256),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(128, num_classes)
		)

	def forward(self, image, audio, text):
		# Ensure all inputs are tensors
		if not isinstance(image, torch.Tensor):
			print("Image tensor is not a torch.Tensor")
			image = torch.tensor(image)
		if not isinstance(audio, torch.Tensor):
			print("Audio tensor is not a torch.Tensor")
			audio = torch.tensor(audio)

		# Extract features from each modality
		image_features = self.image_encoder(image)
		audio_features = self.audio_encoder(audio)
		text_features = self.text_encoder(text)

		# Concatenate features
		combined_features = torch.cat([image_features, audio_features, text_features], dim=1)

		# Classify
		output = self.fusion(combined_features)
		return output