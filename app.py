from fastai.vision.all import *
import gradio as gr

# def greet(name):
#     return "Hello " + name + "!!"
#
# iface = gr.Interface(fn=greet, inputs="text", outputs="text")
# iface.launch()

learn = load_learner('export.pkl')

def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


gr.Interface(fn=predict, inputs=gr.inputs.Image(shape=(512, 512)), outputs=gr.outputs.Label(num_top_classes=3)).launch(share=True)