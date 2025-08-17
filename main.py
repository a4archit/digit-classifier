import streamlit as st
import numpy as np
import tensorflow as tf 
from random import choice
from PIL import Image

from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# used to diaplay some informaiton between program execution
INFO_POINTS = False # set 'False' if you don't want these informative points

# loading model
if INFO_POINTS: print("loading model ---", end="")
model = load_model("mnist_cnn_based_basic_model_9738.h5") #pickle.load(open('/home/archit-elitebook/workarea/whole working/deep learning/mnist cnn 972 model/mnist_cnn_based_basic_model.pkl', 'rb'))
if INFO_POINTS: print("> Complete")








# ***************** Streamlit working: starts ************** #

# ------------------ Sidebar -------------------------- #

st.sidebar.title("About the developer")
st.sidebar.divider()
st.sidebar.write("I am creating this web application with \
**Streamlit** and build an **Digit Classifier Model**, \
this model based on the **CNN** (Convolutional Neural Networks) \
with the **MNIST** Dataset. ")
st.sidebar.write("You can check my social media accounts: ")
st.sidebar.write("[Website](https://a4archit.github.io/my-portfolio)")
st.sidebar.write("[Kaggle](https://www.kaggle.com/architty108)")
st.sidebar.write("[Github](https://www.github.com/a4archit)")
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/archit-tyagi-191323296)")



# main heading
st.header("Number classifier")
st.divider()

st.write("Enter a single number here:")

canvas_tab1, canvas_tab2 = st.columns([1,1])

with canvas_tab1:
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=10,
        stroke_color="#fff",
        background_color="#000",
        background_image= None,
        height=150,
        width=150,
        drawing_mode="freedraw",
        key="full_app"
    )

with canvas_tab2:
    # Show the canvas result as an image or as raw data
    if canvas_result.image_data is not None:
        drawn_image = canvas_result.image_data.astype(np.uint8)

        # converting canvas image to a grayscale image
        img = Image.fromarray(drawn_image)
        img = img.convert("L").resize((28,28))

        img_array = np.array(img)

        # adding an extra dimension to the array (for batch size)
        img_array = np.expand_dims(img_array, axis=-1) # shape becomes (28, 28, 1)
        img_array = np.expand_dims(img_array, axis=0).reshape((-1, 28, 28)) # shape becomes (1, 28, 28, 1)

        # make the prediction
        if INFO_POINTS: st.write(f"{img_array.shape}, {img_array.min()}, {img_array.max()}")

        prediction_prob = model.predict(img_array)
        prediction_label = np.argmax(prediction_prob, axis=1)[0]
        
        # displaying the result 
        st.image(img, caption="Your drawing", width = 56)
        if prediction_prob.max() > 0.60:
            st.write(f"Prediction = **{prediction_label}**")
        else:
            bad_performance = [
                "please improve your handwritting",
                "write cleanly!",
                "your handwritting is very dirty",
                "chhoti bachhi ho kya *itna ganda bhi koi likhta hai bhala*"
            ]
            st.write(choice(bad_performance).capitalize())
 
# ***************** Streamlit working: ends ************** #
    
 
