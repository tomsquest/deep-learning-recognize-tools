from fastai.vision.all import *
import gradio as gr

# def greet(name):
#     return "Hello " + name + "!!"
#
# iface = gr.Interface(fn=greet, inputs="text", outputs="text")
# iface.launch()

learn = load_learner('export.pkl')
labels = learn.dls.vocab


def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


interface = gr.Interface(
    title="Recognize tools",
    description="This model recognizes tools between drill driver, hammer dill, screwdriver and screwgun.",
    examples=["image-examples/hammer-drill.jpg",
              "image-examples/screwgun.jpg",
              "image-examples/drill-driver.jpg",
              "image-examples/electric-screwdriver.jpg"
              "image-examples/toy.jpg",
              "image-examples/gremlin.jpg"],
    fn=predict,
    inputs=gr.Image(width=512,height=512),
    outputs=gr.Label(num_top_classes=3),
    allow_flagging="never",
    live=True,
)

interface.launch(share=True)