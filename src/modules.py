import torch
import torch.nn as nn
import torchvision.models as models
import torchaudio
import transformers


class ImageEncoder(nn.Module):
	"""Image encoder using pre-trained ResNet18."""

	def __init__(self, output_dim=128):
		super(ImageEncoder, self).__init__()
		# Load pre-trained ResNet model
		resnet = models.resnet18(weights='IMAGENET1K_V1')
		# Remove the final classification layer
		self.features = nn.Sequential(*list(resnet.children())[:-1])
		# Add a projection layer
		self.projection = nn.Linear(512, output_dim)

	def forward(self, x):
		x = self.features(x)
		x = torch.flatten(x, 1)
		x = self.projection(x)
		return x


class AudioEncoder(nn.Module):
	"""Audio encoder using wav2vec 2.0 from torchaudio."""

	def __init__(self, output_dim=128):
		super(AudioEncoder, self).__init__()
		# Load pre-trained wav2vec 2.0 model
		bundle = torchaudio.pipelines.WAV2VEC2_BASE
		self.wav2vec2 = bundle.get_model()

		# Freeze the model parameters (optional, can be fine-tuned)
		for param in self.wav2vec2.parameters():
			param.requires_grad = False

		# Wav2vec 2.0 produces 768-dimensional features
		self.projection = nn.Linear(768, output_dim)

	def forward(self, x):
		# x shape: (batch_size, channels, time)
		# Wav2vec expects (batch_size, time)
		if x.size(1) == 1:  # If mono audio with channel dim
			x = x.squeeze(1)
		elif x.size(1) == 2:  # If stereo, convert to mono
			x = torch.mean(x, dim=1)

		# Get wav2vec features - returns a dict with 'output_features' key
		with torch.no_grad():
			wav2vec_output = self.wav2vec2.extract_features(x)
			features = wav2vec_output[0][-1]

		# Average pooling across time dimension to get a fixed-length vector
		embeddings = torch.mean(features, dim=1)

		# Project to the desired output dimension
		x = self.projection(embeddings)
		return x


class TextEncoder(nn.Module):
	"""Text encoder using pre-trained DistilBERT."""

	def __init__(self, output_dim=128):
		super(TextEncoder, self).__init__()
		# Load pre-trained DistilBERT model
		self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
		self.model = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
		# Freeze the model parameters
		for param in self.model.parameters():
			param.requires_grad = False
		# Add a projection layer
		self.projection = nn.Linear(768, output_dim)

	def forward(self, text_list):
		# Tokenize the input texts
		encoded_input = self.tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')
		# Move to the same device as the model
		encoded_input = {k: v.to(next(self.model.parameters()).device) for k, v in encoded_input.items()}
		# Get BERT embeddings
		with torch.no_grad():
			outputs = self.model(**encoded_input)
		# Use the [CLS] token embedding as the sentence representation
		cls_embedding = outputs.last_hidden_state[:, 0, :]
		# Project to the desired output dimension
		x = self.projection(cls_embedding)
		return x