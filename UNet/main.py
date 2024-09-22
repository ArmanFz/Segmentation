from model import UNet
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import *
import torchvision.transforms as transforms
import os

# CONFIG
num_epochs = 15
batch_size = 1
n_classes = 1
starting_lr = 1e-4
final_lr = 1e-6
threshold = 0.5
data_path = 'data'

"""
DATA
├── train
│   ├── image
│   │   ├── *.png
│   │   ├── ...
│   ├── mask
│   │   ├── *.png
│   │   ├── ...
├── val
│   ├── image
│   │   ├── *.png
│   │   ├── ...
│   ├── mask
│   │   ├── *.png
│   │   ├── ...

"""
saved_model_path = 'best_models'
os.makedirs(saved_model_path, exist_ok=True)
result_dir = 'results'
os.makedirs(result_dir, exist_ok=True)




# DATA
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images and masks to the desired size
    transforms.ToTensor(),          # Convert images and masks to tensors
])
train_data = MyDataset(images_directory=os.path.join(data_path, 'train','image'),
                       masks_directory=os.path.join(data_path, 'train','mask'),
                       transform=image_transform)
val_data = MyDataset(images_directory=os.path.join(data_path, 'val','image'),
                     masks_directory=os.path.join(data_path, 'val','mask'),
                     transform=image_transform)

train_loader = DataLoader(train_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)

# MODEL
model = UNet(n_channels = 3, n_classes = 1, sigmoid=True)

# Train
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=starting_lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, min_lr=final_lr)
critrion = DiceLoss()
print('number of output classes:', n_classes)
print('Device:', device)
print('Optimizer:', type(optimizer).__name__)
print('Loss Func:', type(critrion).__name__)
print('Initial learning rate: {}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
print(40*'*', 'Training Started', 40*'*')


train_losses = []
val_losses = []
best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    print(40*'-')
    print(f"Epoch {epoch+1}")
    loop = tqdm(train_loader)
    for batch_idx, (image,mask) in enumerate(loop):
        inputs, masks = image.to(device), mask.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = critrion(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    train_losses.append(epoch_train_loss / len(train_loader))
    print('train loss:', train_losses[-1])

    model.eval()
    with torch.no_grad():
        epoch_val_loss = 0.0
        loop_val = tqdm(val_loader)
        for batch_idx, (image, mask) in enumerate(loop_val):
            inputs, masks = image.to(device), mask.to(device)
            outputs = model(inputs)
            loss = critrion(outputs, masks)
            epoch_val_loss += loss.item()
        val_losses.append(epoch_val_loss / len(val_loader))
        scheduler.step(epoch_val_loss)
        print('\033[1m', 'validation loss:', val_losses[-1], '\033[0m')
        print('learning rate={:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

        # Save the model if the validation loss is the best so far
        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
            torch.save(model.state_dict(), saved_model_path+f'/best_model_batch{batch_size}_{num_epochs}epochs.pth')
            print('best model updated')

print(40*'*', 'Training Finished', 40*'*')


fig, ax = plt.subplots( nrows=1, ncols=1 )
ax.plot(val_losses, label='validation')
ax.plot(train_losses, label='training')
ax.set_title('LOSS (Dice Loss)')
ax.legend()
fig.savefig(result_dir+'/loss_curve.png') 
plt.close(fig)

print('best model saved in:', saved_model_path)
print('reults saved in:', result_dir)


#save some validation output
results_folder = result_dir + '/test_results'
os.makedirs(results_folder, exist_ok=True)
for idx, (inputs, labels) in enumerate(val_loader):
    inputs = inputs.float().to('cuda')
    with torch.no_grad():
        outputs = model(inputs)
    masks = (outputs > threshold).float()  # Threshold the probabilities to create binary masks

    # Convert and save the masks as images
    for i in range(masks.size(0)):
        mask = masks[i].squeeze().cpu().numpy()
        label = labels.squeeze().cpu().numpy()
        pred_image = Image.fromarray(mask*255.0).convert('L')
        label_image = Image.fromarray(label*255.0).convert('L')
        pred_image.save(os.path.join(results_folder, f'test_{idx + i + 1}_pred.png'))
        label_image.save(os.path.join(results_folder, f'test_{idx + i + 1}_label.png'))

print(f"Testing complete. Results are saved in the '{result_dir}/test_results' folder.")
