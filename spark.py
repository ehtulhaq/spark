from flask import Flask, request
import mxnet as mx
from gluoncv import utils
import gluoncv as gc
import os

app = Flask(__name__)

PARTICLE = 'particle'
SAVE = 'save'

@app.route('/ignite', methods=['POST'])
def index():
    defaultResponse = {PARTICLE:'', 'flame':[]}
    if request.method =='POST':
        #Get the image URL and check
        particle = request.args.get(PARTICLE)
        save = request.args.get(SAVE)
        if save is 'False':
            save = False
        elif save is 'True':
            save = True
        if particle:
            try:
                return fire(particle,save)
            except Exception:
                return 'Some error when processing image. Please check passed image.'
        else:
            return defaultResponse


def fire(image_url, save):
    net = gc.model_zoo.get_model('resnet50_v1d',pretrained=True)
    im_fname = utils.download(image_url,PARTICLE)
    img = mx.image.imread(im_fname)
    transform_image = gc.data.transforms.presets.imagenet.transform_eval(img)
    pred = net(transform_image)
    prob = mx.nd.softmax(pred)[0].asnumpy()
    ind = mx.nd.topk(pred, k=5)[0].astype('int').asnumpy().tolist()
    respArray = []
    for i in range(5):
        respArray.append({net.classes[ind[i]]:prob[ind[i]]})
    if(not save):
        os.remove(im_fname)
    return str({PARTICLE:image_url,'flame':(respArray)})
    