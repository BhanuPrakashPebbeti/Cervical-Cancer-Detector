import cv2
import datetime
import numpy as np
import warnings
import pygame
import pygame.camera
import pygame.image
import sys
import time
from classifierLite import classification_module
warnings.filterwarnings("ignore")

class Button:
    def __init__(self,text,width,height,pos,elevation,button_font):
		#Core attributes 
        self.pressed = False
        self.elevation = elevation
        self.dynamic_elecation = elevation
        self.original_y_pos = pos[1]

		# top rectangle 
        self.top_rect = pygame.Rect(pos,(width,height))
        self.top_color = '#475F77'

		# bottom rectangle 
        self.bottom_rect = pygame.Rect(pos,(width,height))
        self.bottom_color = '#354B5E'
		#text
        self.button_font = button_font
        self.text_surf = button_font.render(text,True,'#FFFFFF')
        self.text_rect = self.text_surf.get_rect(center = self.top_rect.center)

    def draw(self,response):
        # elevation logic 
        self.top_rect.y = self.original_y_pos - self.dynamic_elecation
        self.text_rect.center = self.top_rect.center 
        self.bottom_rect.midtop = self.top_rect.midtop
        self.bottom_rect.height = self.top_rect.height + self.dynamic_elecation
        pygame.draw.rect(screen,self.bottom_color, self.bottom_rect,border_radius = 12)
        pygame.draw.rect(screen,self.top_color, self.top_rect,border_radius = 12)
        screen.blit(self.text_surf, self.text_rect)
        return self.check_click(response)

    def check_click(self,system):
        mouse_pos = pygame.mouse.get_pos()
        if self.top_rect.collidepoint(mouse_pos):
            self.top_color = '#D74B4B'
            if pygame.mouse.get_pressed()[0]:
                self.dynamic_elecation = 0
                self.pressed = True
            else:
                self.dynamic_elecation = self.elevation
                if self.pressed == True:
                    self.pressed = False
                    system = True
        else:
            self.dynamic_elecation = self.elevation
            self.top_color = '#475F77'
        return system
        
def cv2ImageToSurface(cv2Image):
    if cv2Image.dtype.name == 'uint16':
        cv2Image = (cv2Image / 255).astype('uint8')
    size = cv2Image.shape[1::-1]
    if len(cv2Image.shape) == 2:
        cv2Image = np.repeat(cv2Image.reshape(size[1], size[0], 1), 3, axis = 2)
    surface = pygame.image.frombuffer(cv2Image.flatten(), size, "RGB")
    return surface.convert()

print()
print("                ==========================================================================    ")
print("               |                             Starting Application                         |   ")
print("                ==========================================================================    ")
print()
print()
pygame.init()
pygame.font.init()
clock = pygame.time.Clock()
button_font = pygame.font.Font(None,30)
STAT_FONT = pygame.font.SysFont("comicsans", 40)
screen = pygame.display.set_mode((740, 512))
pygame.display.set_caption("Application")

pygame.camera.init()
cameras = pygame.camera.list_cameras()
webcam = pygame.camera.Camera(cameras[0])
webcam.start()

button1 = Button('Capture',200,40,(530,150),5,button_font)
button2 = Button('ROI',200,40,(530,250),5,button_font)
button3 = Button('Prediction',200,40,(530,350),5,button_font)
button4 = Button('Back',200,40,(530,450),5,button_font)

classifier = classification_module("models/modelB0.tflite")

input_size = (512,512)
run = True
live = True
capture = False
roi_image = False
predict = False
back = False
done_prediction = False

preroi = []
preclass = []
roicutting = []
classification = []

while run:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            print("ROI Preprocessing : ", np.mean(preroi), np.std(preroi, ddof=1))
            print("ROI Cutting : ", np.mean(roicutting), np.std(roicutting, ddof=1))
            print("Classification Preprocessing : ", np.mean(preclass), np.std(preclass, ddof=1))
            print("Classification : ", np.mean(classification), np.std(classification, ddof=1))
            run = False
            pygame.quit()
            break
    capture = button1.draw(capture)
    roi_image = button2.draw(roi_image)
    if roi_image:
        predict = False
    predict = button3.draw(predict)
    if predict:
        roi_image = False
    back = button4.draw(back)
        
    if capture:
        live = False
        screen.blit(rgbframe, (0,0))
        if not done_prediction:
            text = STAT_FONT.render("This Might take some time!",2,(255,255,255))
            screen.blit(text,(100,475))
            pygame.display.update()
            roi_pre_start = time.time()
            rgbframe_input_preprocessed = classifier.preprocess(rgbframe_input)
            roi_start = time.time()
            roi_cut = classifier.apply_roi_cutting(rgbframe_input_preprocessed,rgbframe_input)
            roi_end = time.time()
            image = classifier.preprocess(roi_cut)
            pre_end = time.time()
            label,score = classifier.classify(image)
            class_end = time.time()
            print("Done")
            print("ROI Preprocessing :",roi_start-roi_pre_start)
            preroi.append(roi_start-roi_pre_start)
            print("ROI Cutting :",roi_end-roi_start)
            roicutting.append(roi_end-roi_start)
            print("Classification Preprocessing :",pre_end-roi_end)
            preclass.append(pre_end-roi_end)
            print("Classification :",class_end-pre_end)
            classification.append(class_end-pre_end)
            done_prediction = True
            
    if roi_image:
        screen.blit(cv2ImageToSurface(cv2.resize(roi_cut,(512,512))), (0,0))
        capture = False
        
    if predict:
        screen.blit(rgbframe, (0,0))
        score = round(score, 2)
        result_text = label + ' (' + str(score) + ')'
        score_label = STAT_FONT.render(result_text,2,(25,255,255))
        screen.blit(score_label,(10,10))
        capture = False
            
    if back:
        live = True
        capture = False
        roi_image = False
        predict = False
        back = False
        done_prediction = False

    if live:
        frame = webcam.get_image()
        frame = pygame.surfarray.array3d(frame)
        frame = frame.transpose([1, 0, 2])
        rgbframe = cv2.resize(frame,input_size)
        rgbframe_input = cv2.cvtColor(rgbframe, cv2.COLOR_RGB2BGR)
        rgbframe = cv2ImageToSurface(rgbframe)
        screen.blit(rgbframe, (0,0))
    pygame.display.update()
    clock.tick(120)
    

