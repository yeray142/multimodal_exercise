import os, sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
import argparse
from torchinfo import summary # to print model summary
from tqdm.auto import tqdm # used in train function
from torchview import draw_graph # print model image
import random
from PIL import Image
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import InceptionResnetV1

# Note: this notebook requires torch >= 1.10.0
print("torch version: ", torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("devide: ", device)


def walk_through_dir(data_path):
  for dirpath, dirnames, filenames in os.walk(data_path):
    filenames = [f for f in filenames if not f[0] == '.']
    dirnames[:] = [d for d in dirnames if not d[0] == '.']
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def print_image_samples(data_path):
    # Set seed
    random.seed(42) 

    # 1. Get all image paths (* means "any combination")
    image_path_list= glob.glob(f"{data_path}/*/*/*.jpg")

    # 2. Get random image path
    random_image_path = random.choice(image_path_list)

    # 3. Get image class from path name (the image class is the name of the directory where the image is stored)
    image_class = Path(random_image_path).parent.stem

    # 4. Open image
    img = Image.open(random_image_path)

    # 5. Print metadata
    print(f"Random image path: {random_image_path}")
    print(f"Image class: {image_class}")
    print(f"Image height: {img.height}") 
    print(f"Image width: {img.width}")
    print(img)

    # Turn the image into an array
    img_as_array = np.asarray(img)

    # Plot the image with matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(img_as_array)
    plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
    plt.axis(False)
    plt.savefig("sample_data_intro.jpg")
    print("sample image saved as sample_data_intro.jpg")
    
    return image_path_list


def get_data_sets_path(data_path):
    train_dir = os.path.join(data_path,"train")
    valid_dir = os.path.join(data_path,"valid")
    test_dir = os.path.join(data_path,"test")
    print("train dir: ", train_dir)
    print("valid dir: ", valid_dir)
    print("test dir: ", test_dir)
    return train_dir, valid_dir, test_dir


def transform_data(IMAGE_WIDTH,IMAGE_HEIGHT):
    IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

    data_train_transform = transforms.Compose([
        v2.Resize(size=IMAGE_SIZE),
        v2.TrivialAugmentWide(),
        # Turn the image into a torch.Tensor
        v2.ToTensor(), # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
        # resnet50 normalization
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_valid_test_transform = transforms.Compose([
        # Resize the images to IMAGE_SIZE xIMAGE_SIZE 
        v2.Resize(size=IMAGE_SIZE),
        # Turn the image into a torch.Tensor
        v2.ToTensor(), # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
        # resnet50 normalization
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])    
    return data_train_transform, data_valid_test_transform


def plot_transformed_images(image_paths, transform, n=3, seed=42):
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for c, image_path in enumerate(random_image_paths):
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")
            plt.savefig("transformed_image.jpg")


def loadImageData(train_dir,valid_dir,test_dir,data_train_transform, data_valid_test_transform):
    # Creating training set
    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                    transform=data_train_transform, # transforms to perform on data (images)
                                    target_transform=None) # transforms to perform on labels (if necessary)

    # Creating validation set
    valid_data = datasets.ImageFolder(root=valid_dir, # target folder of images
                                    transform=data_valid_test_transform, # transforms to perform on data (images)
                                    target_transform=None) # transforms to perform on labels (if necessary)

    #Creating test set
    test_data = datasets.ImageFolder(root=test_dir, transform=data_valid_test_transform)

    print(f"Train data:\n{train_data}\nValidation data:\n{valid_data}\nTest data:\n{test_data}")

    # Get class names as a list
    class_names = train_data.classes
    print("Class names: ",class_names)

    # Check the lengths
    print("The lengths of the training, validation and test sets: ", len(train_data), len(valid_data), len(test_data))  

    return train_data, valid_data, test_data, class_names


def detail_one_sample_data(train_data, class_names):
    img, label = train_data[0][0], train_data[0][1]
    print(f"Image tensor:\n{img}")
    print(f"Image shape: {img.shape}")
    print(f"Image datatype: {img.dtype}")
    print(f"Image label: {label}")
    print(f"Label datatype: {type(label)}")

    # Rearrange the order of dimensions
    img_permute = img.permute(1, 2, 0)

    # Print out different shapes (before and after permute)
    print(f"Original shape: {img.shape} -> [color_channels, height, width]")
    print(f"Image permute shape: {img_permute.shape} -> [height, width, color_channels]")

    # Plot the image
    plt.figure(figsize=(10, 7))
    plt.imshow(img.permute(1, 2, 0))
    plt.axis("off")
    plt.title(class_names[label], fontsize=14);    
    plt.savefig("sample_data_detailed.jpg")


def myDataLoader(train_data, valid_data, test_data, NUM_WORKERS, BATCH_SIZE, BATCH_SIZE_VALID, BATCH_SIZE_TEST):

    # Turn train and test Datasets into DataLoaders
    train_dataloader = DataLoader(dataset=train_data, 
                                batch_size=BATCH_SIZE, # how many samples per batch?
                                num_workers=NUM_WORKERS,
                                shuffle=True) # shuffle the data?

    # Turn train and test Datasets into DataLoaders
    valid_dataloader = DataLoader(dataset=valid_data, 
                                batch_size=BATCH_SIZE_VALID, # how many samples per batch?
                                num_workers=NUM_WORKERS,
                                shuffle=True) # shuffle the data?

    test_dataloader = DataLoader(dataset=test_data, 
                                batch_size=BATCH_SIZE_TEST, 
                                num_workers=NUM_WORKERS,
                                shuffle=False) # don't usually need to shuffle testing data

    # Now let's get a batch image and check the shape of this batch.    
    img, label = next(iter(train_dataloader))

    # Note that batch size will now be 1.  
    print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
    print(f"Label shape: {label.shape}")

    return train_dataloader, valid_dataloader, test_dataloader    


def testSingleForwardPass(train_dataloader, model):
    # 1. Get a batch of images and labels from the DataLoader
    img_batch, label_batch = next(iter(train_dataloader))

    # 2. Get a single image from the batch and unsqueeze the image so its shape fits the model
    img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
    print(f"Single image shape: {img_single.shape}\n")

    # 3. Perform a forward pass on a single image
    model.eval()
    with torch.inference_mode():
        pred = model(img_single.to(device))
        
    # 4. Print out what's happening and convert model logits -> pred probs -> pred label
    print(f"Output logits:\n{pred}\n")
    print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
    print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
    print(f"Actual label:\n{label_single}")    


def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module):
    
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc  


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer):
    
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()
        
        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          early_stop_thresh: int = 5):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    best_accuracy = -1
    #early_stop_thresh = 5 # to-do: given as input parameter
    best_epoch = -1
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.6f} | "
            f"train_acc: {train_acc:.4f} | "
            f"valid_loss: {test_loss:.6f} | "
            f"valid_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if (test_acc > best_accuracy):
            best_accuracy = test_acc
            best_epoch = epoch
            #torch.save(model.state_dict(), 'best-model-parameters.pt')

            # save model checkpoint
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_fn,
                        }, 'best-model-parameters.pt')

            print("model saved")
                    
        # if did not improve in the last "early_stop_thresh" epochs, reduce learning rate
        elif epoch - best_epoch > early_stop_thresh:
            print("Early stop at epoch %d" % epoch)
            break  # terminate the training loop            

    # 6. Return the filled results at the end of the epochs
    return results


def save_loss_curves(model_results):
  
    results = dict(list(model_results.items()))

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig("training_history.jpg")


def my_test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, test_data):
    
    predictions = [['VideoName','ground_truth','prediction']]

    for f in test_data.imgs:
        f_name = f[0]
        f_name = f_name[len(f_name)-19:len(f_name)-4]+'.mp4'
        cat = f[1]+1 # f[1] from 0 to 6
        predictions.append([f_name,cat,'-1'])
        #print(predictions[-1])

    # Put model in test mode
    model.eval()
    
    # Setup test loss and test accuracy values
    test_acc = 0
    
    print("Evaluating on test set...")
    # Loop through data loader data batches
    counter = 1
    for batch, (X, y) in enumerate(dataloader):
        print(f"Batch: {batch} of {len(dataloader)}")
        # Send data to target device
        X, y = X.to(device), y.to(device)

        if(len(y)!=1):
            print("ERROR: batch size (test stage) =", len(y))
            print("ERROR: my_test_step function require batch size = 1 to generate the correct output file")
            exit()
        
        # 1. Forward pass
        y_pred = model(X)

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        test_acc += (y_pred_class == y).sum().item()/len(y_pred)

        predictions[counter][2] = y_pred_class.tolist()[0]+1 # y_pred_class.tolist()[0] from 0 to 6
        counter +=1

    # Adjust metrics to get average accuracy per batch 
    test_acc = test_acc / len(dataloader)
    print("Average accuracy = ", test_acc)

    np.savetxt("predictions_test_set.csv",
        predictions,
        delimiter =",",
        fmt ='% s')


def main(data_path, model_stage, parameters_dict, class_weights):

    # preliminaries
    walk_through_dir(data_path)
    train_dir, valid_dir, test_dir = get_data_sets_path(data_path)
    image_path_list = print_image_samples(data_path)

    # data transformation
    data_train_transform, data_valid_test_transform = transform_data(parameters_dict['image_size']['values'][0], parameters_dict['image_size']['values'][1])
    plot_transformed_images(image_path_list, transform=data_train_transform, n=3)
    
    # data loader
    train_data, valid_data, test_data, class_names = loadImageData(train_dir,valid_dir,test_dir,data_train_transform, data_valid_test_transform)
    num_classes = len(class_names)
    detail_one_sample_data(train_data, class_names)
    train_dataloader, valid_dataloader, test_dataloader = myDataLoader(train_data, valid_data, test_data, parameters_dict['num_workers']['values'][0], parameters_dict['batch_size']['values'][0], parameters_dict['batch_size_valid']['values'][0], parameters_dict['batch_size_test']['values'][0])

    # model definition
    model = InceptionResnetV1(classify=True, pretrained='vggface2', num_classes=num_classes).to(device)

    # model testing
    testSingleForwardPass(train_dataloader, model)

    # saving the model image as "model.gv.png"
    model_graph = draw_graph(model, input_size=(1,3,224,224), expand_nested=True)
    model_graph.visual_graph.render(format='pdf')

    #
    # TRAIN
    #
    if(model_stage=='train'):

        # Unfreeze layers
        for name, para in model.named_parameters():
            para.requires_grad = True

        # do a test pass through of an example input size 
        summary(model, input_size=[1, 3, parameters_dict['image_size']['values'][0], parameters_dict['image_size']['values'][1]])

        # Setup loss function and optimizer
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=parameters_dict['learning_rate']['values'][0])
        
        # Train model_0 
        model_results = train(model=model,
                            train_dataloader=train_dataloader,
                            test_dataloader=valid_dataloader,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            epochs=parameters_dict['num_epochs']['values'][0],
                            early_stop_thresh=parameters_dict['early_stopping']['values'][0])

        save_loss_curves(model_results)

    elif(model_stage=='resume'):

        # Unfreeze layers
        for name, para in model.named_parameters():
            para.requires_grad = True

        # do a test pass through of an example input size 
        summary(model, input_size=[1, 3, parameters_dict['image_size']['values'][0], parameters_dict['image_size']['values'][1]])

        # loading checkpoints
        checkpoint = torch.load('best-model-parameters.pt', map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])

        # reducing the original learning rate by 10
        optimizer = torch.optim.Adam(params=model.parameters(), lr=parameters_dict['learning_rate']['values'][0]/10)
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        for group in optimizer.param_groups:
            print("lr = ", group['lr'])

        # Train model_0 
        model_results = train(model=model,
                            train_dataloader=train_dataloader,
                            test_dataloader=valid_dataloader,
                            optimizer=optimizer,
                            loss_fn=checkpoint['loss'],
                            epochs=parameters_dict['num_epochs']['values'][0],
                            early_stop_thresh=parameters_dict['early_stopping']['values'][0])
        
        save_loss_curves(model_results) 


    elif(model_stage=='test'):

        # load the model
        checkpoint = torch.load('best-model-parameters.pt', map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])

        # evaluate on the test set and generate the 'predictions_test_set.csv' file (used later by the evaluation script)
        my_test_step(model, test_dataloader, test_data)
    

#------------------------------
# usage:
#
# python baseline_InceptionResnetV1.py ./data 'train' # train the model the first time
# python baseline_InceptionResnetV1.py ./data 'resume' # load the model previously trained, reduce the learning rate by 10, and keep training
# python baseline_InceptionResnetV1.py ./data 'test' # evaluate on the test set
#
# where, './data' is the path to the input data with the 'train', 'valid' 'test' directories
#
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Baseline InceptionResnetV1 model')
    parser.add_argument('--data_path', type=str, default='/home/yeray142/Documents/projects/multimodal-exercise/data/first_Impressions_v3_multimodal', help='path to the input data')
    parser.add_argument('--model_stage', type=str, default='train', help='model stage: train, resume or test')
    args = parser.parse_args()

    parameters_dict = {
        'image_size': {
            'values': [224, 224]
            },
        'num_workers': {
            'values': [0]
            },
        'batch_size': {
            'values': [256]
            },
        'batch_size_valid': {
            'values': [256]
            },
        'batch_size_test': {
            'values': [1]
            },
        'num_epochs': {
            'values': [300]
            },
        'learning_rate': {
            'values': [1e-6]
            },
        'early_stopping': {
            'values': [50]
            },
    }

    # train data distribution per category [10, 164, 1264, 2932, 1353, 232, 51]
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device) # can be changed to use class weights
    
    data_path = args.data_path # path to the input data
    model_stage = args.model_stage # 'train', 'resume' or 'test'

    main(data_path, model_stage, parameters_dict,class_weights)
