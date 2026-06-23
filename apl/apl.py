import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Forces the script to look at the first GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' # Alternative memory growth enforcer

import tensorflow as tf
import streamlit as st

import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.io import loadmat
import math
import io
from PIL import Image

import gc
from tensorflow.keras import backend as K

class StreamlitProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar, total_batches):
        super().__init__()
        self.progress_bar = progress_bar
        self.total_batches = total_batches

    def on_predict_batch_end(self, batch, logs=None):
        # Calculate percentage complete based on the current batch
        percent_complete = int(((batch + 1) / self.total_batches) * 100)
        # Safely update the Streamlit UI (cap at 100 to prevent errors)
        self.progress_bar.progress(min(percent_complete, 100))


# FORCE GPU MEMORY GROWTH
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Tell TF to only use the memory it needs, rather than grabbing all of it
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        st.sidebar.success("✅ GPU connected and memory growth enabled!")
    except RuntimeError as e:
        st.sidebar.error(f"GPU Error: {e}")
else:
    st.sidebar.error("❌ TensorFlow cannot see the GPU in this process!")



# 1. App Setup
st.title("🛰️ HYPSO-1 Sea-Land-Cloud Hyperspectral Image Classifier")

# 2. Model Selection 
@st.experimental_singleton
def load_tf_model(model_name):
    model_paths = {
        "Autoencoder": "models/model_autoencoder.keras",
        "Hybrid 3D-2D CNN": "models/hybridcnn.keras",
        "Transformer": "models/model_transformer.keras"
    }
    model = tf.keras.models.load_model(model_paths[model_name], compile=False)
    return model

@st.experimental_singleton
def load_preprocessors():
    # Make sure paths match your lab PC folder structure
    scaler = joblib.load("preprocessors/scaler.pkl")
    pca = joblib.load("preprocessors/pca.pkl")
    return scaler, pca

# Call this right below where you select your model
scaler, pca = load_preprocessors()

model_choice = st.sidebar.selectbox("Select Model", ["Autoencoder", "Hybrid 3D-2D CNN", "Transformer"])
model = load_tf_model(model_choice)

# 3. File Upload
uploaded_file = st.file_uploader("Upload Hyperspectral Image (.mat)", type=["mat"])

if uploaded_file is not None:

    # 1. Read the .mat file
    mat_dict = loadmat(uploaded_file)
    
    # 2. Extract the actual image array using the exact same key you used in Jupyter
    # (Make sure to use whatever key your dictionary actually uses!)
    hsi_data = mat_dict['HYPSO_image_converted_from_bip_to_mat']
    
    # Optional: Ensure it's the exact same data type as training
    hsi_data = hsi_data.astype(np.float32)
    
    st.write(f"**Image Shape:** {hsi_data.shape}")
    
    # 4. The Execution Trigger
    if st.button(f"Run {model_choice} Classification"):

        try: 
        
            # YOU MUST DEFINE THESE TWO VARIABLES FIRST:
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            status_text.text("Extracting patches and running inference...")
            
            # --- YOUR PIPELINE GOES HERE ---

            status_text.text("Applying StandardScaler and PCA...")
            
            # Get original dimensions
            H, W, Bands = hsi_data.shape
            
            # 1. FLATTEN: Reshape from 3D to 2D (Pixels, Bands)
            # -1 automatically calculates H * W
            flat_pixels = hsi_data.reshape(-1, Bands) 
            
            # 2. TRANSFORM: Apply scaler and PCA
            # (This is usually very fast on CPU)
            scaled_pixels = scaler.transform(flat_pixels)
            pca_pixels = pca.transform(scaled_pixels)
            
            # 3. FOLD: Reshape back to 3D with the new number of PCA bands
            num_pca_bands = pca_pixels.shape[1]
            processed_hsi = pca_pixels.reshape(H, W, num_pca_bands)
            
            # Now your data is ready for patch extraction!
            
            status_text.text("Padding image and extracting patches...")
            
            margin=5

            # 1. Adăugăm padding imaginilor (Zero padding pe margini)
            padded_test_images = np.pad(processed_hsi, ((margin, margin), (margin, margin), (0, 0)), mode='constant')

            PATCH_SIZE=11
            nr_comp=20

            # Colectăm indicii pentru TEST
            # Assuming hsi_data shape is (H, W, Bands)
            h, w, bands = hsi_data.shape
            indices_test = []

            for x in range(h):
                for y in range(w):
                    # We use 0 because there is only 1 image being processed
                    indices_test.append((0, x, y))

            def patch_generator_test_input():
                for img_idx, x, y in indices_test:
                    patch = padded_test_images[x : x + PATCH_SIZE, y : y + PATCH_SIZE, :]
                    yield patch

            output_signature_input = (
                tf.TensorSpec(shape=(PATCH_SIZE, PATCH_SIZE, nr_comp), dtype=tf.float32)
            )

            test_input_ds = tf.data.Dataset.from_generator(
                patch_generator_test_input,
                output_signature=output_signature_input
            )

            batch_size_test=2048
            test_input_ds=test_input_ds.batch(batch_size_test).prefetch(tf.data.AUTOTUNE)

            h, w, bands = hsi_data.shape
            total_pixels = h * w

            total_batches = math.ceil(total_pixels / batch_size_test)
            
            status_text.text(f"Running {model_choice} inference on {total_pixels} pixels...")

            # Force the execution onto the first GPU
            with tf.device('/GPU:0'):
                status_text.text("Running inference strictly on GPU...")
                predictions = model.predict(
                                            test_input_ds,
                                            callbacks=[StreamlitProgressCallback(progress_bar, total_batches)]
                )
            
            st.write(f"**Predictions Shape:** {predictions.shape}")
            #st.write(f"**Sample raw prediction:** {predictions[0]}")

            #predictions = model.predict(test_input_ds)
            # -------------------------------
            
            # Mock prediction step to show how to update the UI
            # (Remove this in your actual code)
            import time
            for percent_complete in range(100):
                time.sleep(0.02) # Simulating batch processing time
                progress_bar.progress(percent_complete + 1)
                
            status_text.text("Inference Complete! Generating map...")
            progress_bar.progress(100) 
            
            # 5. Reshape and Display Results
            # 1. Convert the probabilities into a single class label per pixel
            predicted_labels = np.argmax(predictions, axis=1)

            # 2. Reshape the 1D list of labels back into the 2D image map
            class_map = predicted_labels.reshape(hsi_data.shape[0], hsi_data.shape[1])
            
            paleta = np.array([
            [ 18, 34, 102],        # Clasa 0: Albastru  Sea
            [ 79, 122,  48],       # Clasa 1: Verde     Land
            [161, 210, 212]        # Clasa 2: Alb-albastrui      Clouds
                
            ], dtype=np.uint8)

            image_rgb = paleta[class_map]
            

            # Render directly in Streamlit (No Matplotlib required!)
            st.subheader("Classification Result")

            # Creăm două coloane: col_img ocupă 75% din lățime (raport 3), col_leg ocupă 25% (raport 1)
            col_img, col_leg = st.columns([3, 1])

            # În prima coloană punem imaginea rezultată
            with col_img:
                st.image(image_rgb, width=400) #use_column_width=True, 
                
            # În a doua coloană construim legenda cu pătrățele colorate din HTML
            with col_leg:
                st.markdown("### 📋 Legend")
                
                # Clasa 0: Mare (Sea) - Albastru
                st.markdown(
                    '<div style="display: flex; align-items: center; margin-bottom: 12px;">'
                    '<div style="width: 22px; height: 22px; background-color: rgb(18, 34, 102); margin-right: 10px; border-radius: 4px;"></div>'
                    '<span><b>Sea</b></span>'
                    '</div>', 
                    unsafe_allow_html=True
                )
                
                # Clasa 1: Uscat (Land) - Verde
                st.markdown(
                    '<div style="display: flex; align-items: center; margin-bottom: 12px;">'
                    '<div style="width: 22px; height: 22px; background-color: rgb(79, 122, 48); margin-right: 10px; border-radius: 4px;"></div>'
                    '<span><b>Land</b></span>'
                    '</div>', 
                    unsafe_allow_html=True
                )
                
                # Clasa 2: Nori (Clouds) - Alb-albăstrui
                st.markdown(
                    '<div style="display: flex; align-items: center; margin-bottom: 12px;">'
                    '<div style="width: 22px; height: 22px; background-color: rgb(161, 210, 212); margin-right: 10px; border-radius: 4px; border: 1px solid #ccc;"></div>'
                    '<span><b>Clouds</b></span>'
                    '</div>', 
                    unsafe_allow_html=True
                )

            # 1. Convert the NumPy array into a PIL Image object
            result_image = Image.fromarray(image_rgb)

            # 2. Create an empty byte buffer in your RAM
            buf = io.BytesIO()

            # 3. Save the image into that buffer as a PNG
            result_image.save(buf, format="PNG")
            byte_im = buf.getvalue()

            # 4. Display the Streamlit Download Button
            st.download_button(
                label="📥 Download Classified Image",
                data=byte_im,
                file_name="hyperspectral_classification.png",
                mime="image/png"
            )

        finally:

                    # --- THE AGGRESSIVE MEMORY CLEANUP BLOCK ---
            #st.write("Cleaning up RAM...")

            # 1. Manually delete the massive arrays (use try/except in case they were already deleted)
            try: del mat_dict
            except NameError: pass
        
            try: del predictions
            except NameError: pass
        
            try: del class_map
            except NameError: pass

            # 2. Clear TensorFlow's background graph
            K.clear_session()

            # 3. Force Python to empty the trash instantly
            gc.collect()
