'''

Author: Kunle Oguntoye
Date: 04 April 2023'''


from torch_snippets import *
import cv2
import torchvision
from torchvision import models


class ageGenderClassifier(nn.Module):
    def __init__(self):
        super(ageGenderClassifier, self).__init__()

        self.intermediate = nn.Sequential(
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.Dropout(0.4),  # dropout helps avoid overfitting
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128,64),
            nn.ReLU(),
        )
        self.age_classifier = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid() # brings your final output between 0 and 1
        )
        self.gender_classifier = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid() # brings your final output between 0 and 1
        )
    def forward(self, x):
        x = self.intermediate(x)
        age = self.age_classifier(x)
        gender = self.gender_classifier(x)
        return gender, age



class MODEL(nn.Module):
    IMAGE_SIZE = 224

    def __init__(self, fpath = './trained_model (1).pth'):
        super().__init__()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                                              std=[0.229, 0.224, 0.225])
        self.model = models.vgg16(pretrained = True)

        for param in self.model.parameters():
            param.requires_grad = False # all freeze

        self.model.avgpool = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten()
        )

    # further manipulate classifier layer to suit our tasks

        self.model.classifier = ageGenderClassifier()

        self.model.load_state_dict(torch.load(fpath))
        logger.info('Loaded Model')

    def preprocess_image(self, im): # returns image in standard channel
                                    # arrangement. that is [Number, C, H, W]

        im = resize(im, IMAGE_SIZE)
        im = torch.tensor(im).permute(2,0,1)
        im = self.normalize(im/255.) # normalize before you normalize. Dig it?
        return im[None]

    @torch.no_grad()
    def predict(self, path):
        image = cv2.imread(path, 0)
        image = self.preprocess_image(image)
        gender, age = self.model(image.to('cpu'))
        pred_gender = gender.to('cpu').detach().numpy()
        pred_age = age.to('cpu').detach().numpy()
        return { 'age': np.where(pred_gender[0][0]<0.5,'Male','Female'), 'gender': int(pred_age[0][0]*80)}


