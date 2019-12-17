A0 = [.48, .14, .67, 1]
A1 = [.43, .09, .62, 1]

W0 = [1, 1, 1, 1]
W1 = [.95, .95, .95, 1]

G0 = [.75, .75, .75, 1]


B0 = [0.118, 0.016, 0.224, 1]
B1 = [0.204, 0.078, 0.337, 1]
B2 = [0.306, 0.173, 0.451, 1]
B3 = [0.427, 0.302, 0.565, 1]
B4 = [0.569, 0.467, 0.675, 1]

def add(A, B):
    return [i + j for i, j in zip(A, B)]
def sub(A, B):
    return [i - j for i, j in zip(A, B)]
def setAlpha(A, alpha):
    return A[:-1] + [alpha]
    

from kivy.graphics.texture import Texture
from PIL import Image, ImageDraw, ImageFilter

def createBoxShadow(w, h, r, alpha = 1.0):
    w, h, r = int(w), int(h), int(r) 

    texture = Texture.create(size=(w, h), colorfmt='rgba')

    im = Image.new('RGBA', (w, h))

    draw = ImageDraw.Draw(im)
    draw.rectangle((r, r, w - r, h - r), fill=(0, 0, 0, int(255 * alpha)))
    
    im = im.filter(ImageFilter.GaussianBlur(0.25 * r))

    texture.blit_buffer(im.tobytes(), colorfmt='rgba', bufferfmt='ubyte')

    return texture

def createInsetBoxShadow(w, h, r, alpha = 1.0):
    w, h, r = int(w), int(h), int(r) 

    texture = Texture.create(size=(w, h), colorfmt='rgba')
    
    im = Image.new('RGBA', (w, h), color=(0, 0, 0, int(255 * alpha)))

    draw = ImageDraw.Draw(im)
    draw.rectangle((r, r, w - r, h - r), fill=(255, 255, 255, 0))
    
    im = im.filter(ImageFilter.GaussianBlur(0.25 * r))

    texture.blit_buffer(im.tobytes(), colorfmt='rgba', bufferfmt='ubyte')

    return texture