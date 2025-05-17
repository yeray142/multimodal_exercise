import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import MultimodalAgeClassifier
from dataset import MultimodalDataset
from src.dataset import collate_fn


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
	# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device = 'cpu'
	model = model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	for epoch in range(num_epochs):
		# Training
		model.train()
		running_loss = 0.0
		correct = 0
		total = 0

		for batch in tqdm(train_loader):
			images = batch['image'].to(device)
			audio = batch['audio'].to(device)
			text = batch['text']  # Text is processed in the text encoder
			labels = batch['label'].to(device)

			optimizer.zero_grad()

			outputs = model(images, audio, text)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

		train_loss = running_loss / len(train_loader)
		train_acc = correct / total

		# Validation
		model.eval()
		val_loss = 0.0
		correct = 0
		total = 0

		with torch.no_grad():
			for batch in tqdm(val_loader):
				images = batch['image'].to(device)
				audio = batch['audio'].to(device)
				text = batch['text']
				labels = batch['label'].to(device)

				outputs = model(images, audio, text)
				loss = criterion(outputs, labels)

				val_loss += loss.item()
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

		val_loss = val_loss / len(val_loader)
		val_acc = correct / total

		print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

	return model


def main():
	parser = argparse.ArgumentParser(description='Train a multimodal age classifier')
	parser.add_argument('--data_dir', type=str, default="/home/yeray142/Documents/projects/multimodal-exercise/data/first_Impressions_v3_multimodal", help='Directory with all the data files')
	parser.add_argument('--train_annotations', type=str, default="/home/yeray142/Documents/projects/multimodal-exercise/data/first_Impressions_v3_multimodal/train_set_age_labels.csv", help='Path to training annotations CSV file')
	parser.add_argument('--val_annotations', type=str, default="/home/yeray142/Documents/projects/multimodal-exercise/data/first_Impressions_v3_multimodal/valid_set_age_labels.csv", help='Path to validation annotations CSV file')
	args = parser.parse_args()

	image_transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	# No need for MFCC transform since we're using wav2vec

	# Define dataset and dataloaders
	train_dataset = MultimodalDataset(
		data_dir=args.data_dir,
		annotations_file=args.train_annotations,
		image_transform=image_transform,
		audio_transform=None  # No transform needed for wav2vec
	)

	val_dataset = MultimodalDataset(
		data_dir=args.data_dir,
		annotations_file=args.val_annotations,
		image_transform=image_transform,
		audio_transform=None  # No transform needed for wav2vec
	)

	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1, collate_fn=collate_fn)
	val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=1, collate_fn=collate_fn)

	# Get number of classes from annotations
	# Assuming the labels are in the last column of the CSV
	train_labels = pd.read_csv(args.train_annotations)
	num_classes = len(train_labels['AgeGroup'].unique())
	print(f"Number of classes: {num_classes}")

	# Initialize model
	model = MultimodalAgeClassifier(num_classes=num_classes)

	# Train model
	trained_model = train_model(model, train_loader, val_loader, num_epochs=10)

	# Save model
	torch.save(trained_model.state_dict(), 'multimodal_age_classifier.pth')


if __name__ == "__main__":
	main()