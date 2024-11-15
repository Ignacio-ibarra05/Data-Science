import os
import cv2
import mediapipe as mp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from tqdm import tqdm

# Configuración
image_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
print("Using device:", device)

if torch.cuda.is_available():
    print("Nombre de la tarjeta gráfica:", torch.cuda.get_device_name(0))
    print("Memoria total de la tarjeta gráfica: {:.2f} GB".format(torch.cuda.get_device_properties(0).total_memory / 1e9))
    print("Memoria disponible antes del entrenamiento: {:.2f} GB".format(torch.cuda.memory_reserved(0) / 1e9))
    print("Memoria asignada antes del entrenamiento: {:.2f} GB".format(torch.cuda.memory_allocated(0) / 1e9))

mp_hands = mp.solutions.hands

def detect_and_crop_hand(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, c = image.shape
                x_min = min([landmark.x for landmark in hand_landmarks.landmark]) * w
                x_max = max([landmark.x for landmark in hand_landmarks.landmark]) * w
                y_min = min([landmark.y for landmark in hand_landmarks.landmark]) * h
                y_max = max([landmark.y for landmark in hand_landmarks.landmark]) * h
                x_min = max(0, int(x_min))
                x_max = min(w, int(x_max))
                y_min = max(0, int(y_min))
                y_max = min(h, int(y_max))
                cropped_image = image[y_min:y_max, x_min:x_max]
                return cropped_image
        return image  # Devuelve la imagen original si no se detecta ninguna mano

def preprocess_images(data_dir, output_dir):
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                image = cv2.imread(img_path)
                cropped_image = detect_and_crop_hand(image)
                if cropped_image is not None:
                    save_path = os.path.join(output_dir, file)
                    cv2.imwrite(save_path, cropped_image)

preprocess_images('datasets/ASL_Alphabet_Dataset/asl_alphabet_train', 'datasets/ASL_Alphabet_Dataset/processed_train')

# Definir transformaciones avanzadas
basic_transform = [
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]

augmentation_transform = [
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2)
]

transform = transforms.Compose(basic_transform + augmentation_transform)

# Cargar el conjunto de datos procesado
dataset = datasets.ImageFolder(root='datasets/ASL_Alphabet_Dataset/processed_train', transform=transform)

# Dividir el conjunto de datos en 95% entrenamiento y 5% prueba
train_size = int(0.95 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# Visualizar la primera imagen de cada clase antes de entrenar el modelo
def show_first_image_of_each_class(dataset, num_images=5):
    class_to_idx = dataset.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    first_images = []
    labels = []
    
    for class_name in class_to_idx.keys():
        class_index = class_to_idx[class_name]
        for idx in dataset.indices:
            img_path, target = dataset.dataset.samples[idx]
            if target == class_index:
                first_images.append(img_path)
                labels.append(class_name)
                break
    
    fig, axes = plt.subplots(1, len(first_images), figsize=(15, 15))
    for i, img_path in enumerate(first_images):
        img = Image.open(img_path).convert('L')
        img = transform(img)
        img = img.numpy()
        axes[i].imshow(img[0] * 0.5 + 0.5, cmap='gray')  # Des-normalizar y mostrar en escala de grises
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis('off')
    plt.show()

print("Primera imagen de cada clase preprocesada para entrenamiento:")
show_first_image_of_each_class(train_dataset)

class SignLanguageModel(nn.Module):
    def __init__(self):
        super(SignLanguageModel, self).__init__()
        self.base_model = resnet18(pretrained=True)
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 29)
    
    def forward(self, x):
        return self.base_model(x)

model = SignLanguageModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Función de entrenamiento con monitoreo de métricas
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({"loss": loss.item(), "accuracy": 100 * correct / total})
                pbar.update(1)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
        scheduler.step()
        if torch.cuda.is_available():
            print(f"Memoria asignada durante el entrenamiento: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"Memoria reservada durante el entrenamiento: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
            torch.cuda.empty_cache()

        # Evaluación en el conjunto de validación
        validate_model(model, test_loader, criterion)

# Entrenar el modelo
train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=50)

# Guardar el modelo entrenado
torch.save(model.state_dict(), f'ASL_language_model_{image_size}x{image_size}_resnet.pth')

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def validate_model(model, test_loader, criterion):
    model.eval()
    all_labels = []
    all_preds = []
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100 * correct / total
    print(f"Validation Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_acc:.2f}%")
    
    # Reporte de clasificación
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=dataset.classes))

    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

# Evaluar el modelo
validate_model(model, test_loader, criterion)
