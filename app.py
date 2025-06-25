<<<<<<< HEAD
# import streamlit as st
# from PIL import Image
# import numpy as np
# from keras.models import load_model
# from keras.preprocessing.image import img_to_array
# import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go
# import pandas as pd
# import io
# import os
# os.environ["KERAS_BACKEND"] = "tensorflow"  # or "jax" or "torch"
# import keras

# @st.cache_resource
# def load_model_from_hf():
#     """Load specific model from Hugging Face Hub"""
#     try:
#         st.info("Loading model 250|15.h5 from Hugging Face...")
        
#         # Load specific model file from HF
#         model = keras.saving.load_model("hf://Ajay007001/Brain-Stroke-Prediction/model 250|15.h5")
        
#         st.success("✅ Model 250|15 loaded successfully!")
#         return model
        
#     except Exception as e:
#         st.error(f"❌ Error loading model: {str(e)}")
#         # Fallback: try with URL encoding
#         try:
#             st.info("Trying alternative loading method...")
#             from huggingface_hub import hf_hub_download
#             import tensorflow as tf
            
#             model_path = hf_hub_download(
#                 repo_id="Ajay007001/Brain-Stroke-Prediction",
#                 filename="model 250|15.h5"
#             )
#             model = tf.keras.models.load_model(model_path)
#             st.success("✅ Model loaded with fallback method!")
#             return model
#         except Exception as e2:
#             st.error(f"❌ Fallback also failed: {str(e2)}")
#             return None

# # Example usage:
# # Load one of your models


# # Load model with error handling
# try:
#     model = load_model("Model/model 250|15.h5")
#     model_loaded = True
# except Exception as e:
#     st.error(f"Error loading model: {str(e)}")
#     model_loaded = False

# # Standardized medical terminology
# class_labels = [
#     'Hemorrhagic Stroke', 
#     'Competition Dataset Session 1',
#     'Competition Dataset Session 2', 
#     'No Stroke', 
#     'Ischemic Stroke'
# ]

# img_siz = 250

# # Configure Streamlit page
# st.set_page_config(
#     page_title="Stroke MRI Detector",
#     layout='centered',
#     page_icon="🧠"
# )

# # Custom CSS for better styling
# st.markdown("""
#     <style>
#     .header {
#         font-size: 24px !important;
#         font-weight: bold !important;
#     }
#     .no-stroke {
#         color: #2ecc71 !important;
#         font-weight: bold !important;
#         font-size: 22px !important;
#     }
#     .stroke {
#         color: #e74c3c !important;
#         font-weight: bold !important;
#         font-size: 22px !important;
#     }
#     .confidence {
#         font-size: 18px !important;
#         color: #3498db !important;
#     }
#     .error {
#         color: #FF5252 !important;
#     }
#     .stImage>img {
#         border-radius: 10px;
#         box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
#     }
#     </style>
#     """, unsafe_allow_html=True)

# def create_prediction_chart(prediction_probs, class_labels):
#     """Create an interactive bar chart for prediction probabilities"""
    
#     # Convert to percentages
#     prob_percentages = [prob * 100 for prob in prediction_probs]
    
#     # Create DataFrame for easier handling
#     df = pd.DataFrame({
#         'Class': class_labels,
#         'Probability': prob_percentages
#     })
    
#     # Define colors based on class type
#     colors = []
#     for label in class_labels:
#         if label == "No Stroke":
#             colors.append('#2ecc71')  # Green for no stroke
#         elif "Stroke" in label:
#             colors.append('#e74c3c')  # Red for stroke types
#         else:
#             colors.append('#3498db')  # Blue for dataset entries
    
#     # Create the bar chart
#     fig = go.Figure()
    
#     fig.add_trace(go.Bar(
#         x=df['Class'],
#         y=df['Probability'],
#         marker_color=colors,
#         text=[f'{prob:.1f}%' for prob in prob_percentages],
#         textposition='auto',
#         textfont=dict(color='white', size=12, family='Arial Black'),
#         hovertemplate='<b>%{x}</b><br>Probability: %{y:.1f}%<extra></extra>',
#         name='Prediction Probability'
#     ))
    
#     # Update layout - disable zoom and pan
#     fig.update_layout(
#         title={
#             'text': '📊 Prediction Probabilities',
#             'x': 0.5,
#             'xanchor': 'center',
#             'font': {'size': 20, 'family': 'Arial Black'}
#         },
#         xaxis_title='Stroke Classification',
#         yaxis_title='Probability (%)',
#         xaxis_tickangle=-45,
#         height=500,
#         showlegend=False,
#         plot_bgcolor='rgba(0,0,0,0)',
#         paper_bgcolor='rgba(0,0,0,0)',
#         font=dict(family="Arial", size=12),
#         margin=dict(l=50, r=50, t=80, b=120),
#         # Disable interactions
#         dragmode=False
#     )
    
#     # Update axes - disable zoom
#     fig.update_xaxes(
#         showgrid=True,
#         gridwidth=1,
#         gridcolor='lightgray',
#         tickfont=dict(size=10),
#         fixedrange=True  # Disable zoom on x-axis
#     )
#     fig.update_yaxes(
#         showgrid=True,
#         gridwidth=1,
#         gridcolor='lightgray',
#         range=[0, max(prob_percentages) * 1.1],
#         fixedrange=True  # Disable zoom on y-axis
#     )
    
#     return fig

# def create_summary_metrics(predicted_label, confidence, prediction_probs, class_labels):
#     """Create summary metrics display"""
    
#     # Find top 3 predictions
#     top_indices = np.argsort(prediction_probs)[::-1][:3]
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.metric(
#             label="🎯 Top Prediction",
#             value=predicted_label,
#             delta=f"{confidence * 100:.1f}% confidence"
#         )
    
#     with col2:
#         second_best_idx = top_indices[1]
#         second_best_label = class_labels[second_best_idx]
#         second_best_prob = prediction_probs[second_best_idx] * 100
#         st.metric(
#             label="🥈 Second Highest",
#             value=second_best_label,
#             delta=f"{second_best_prob:.1f}%"
#         )
    
#     with col3:
#         third_best_idx = top_indices[2]
#         third_best_label = class_labels[third_best_idx]
#         third_best_prob = prediction_probs[third_best_idx] * 100
#         st.metric(
#             label="🥉 Third Highest",
#             value=third_best_label,
#             delta=f"{third_best_prob:.1f}%"
#         )

# st.title("🧠 Brain Stroke MRI Classifier")
# st.write("Upload an MRI image to detect stroke type or confirm no stroke")

# upload = st.file_uploader(
#     "Upload an MRI image",
#     type=['jpeg', 'png', 'jpg'],
#     help="Supported formats: JPEG, PNG, JPG"
# )

# if upload and model_loaded:
#     try:
#         # Process image
#         img = Image.open(upload).convert('RGB')
        
#         # Display images
#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(
#                 img,
#                 caption="Original Image",
#                 use_container_width=True
#             )
        
#         img_resized = img.resize((img_siz, img_siz))
#         with col2:
#             st.image(
#                 img_resized,
#                 caption=f"Resized to {img_siz}x{img_siz}",
#                 use_container_width=True
#             )
        
#         img_arr = img_to_array(img_resized) / 255.0
#         img_arr = np.expand_dims(img_arr, axis=0)

#         # Make prediction
#         prediction = model.predict(img_arr)
#         predicted_index = np.argmax(prediction)
#         confidence = float(np.max(prediction))
#         predicted_label = class_labels[predicted_index]

#         # Display results with conditional formatting
#         st.markdown("---")
        
#         if predicted_label == "No Stroke":
#             st.markdown(f'### <span class="no-stroke">✅ No Stroke Detected</span>', unsafe_allow_html=True)
#         else:
#             st.markdown(f'### <span class="stroke">⚠️ Detected: {predicted_label}</span>', unsafe_allow_html=True)
        
#         st.markdown(f'### <span class="confidence">🔍 Confidence: {confidence * 100:.2f}%</span>', unsafe_allow_html=True)
        
#         # Confidence progress bar
#         st.progress(confidence)
        
#         # Summary metrics
#         st.markdown("### 📈 Prediction Summary")
#         create_summary_metrics(predicted_label, confidence, prediction[0], class_labels)
        
#         # Interactive prediction chart (replaces the HTML/CSS bars)
#         st.markdown("---")
#         prediction_chart = create_prediction_chart(prediction[0], class_labels)
#         st.plotly_chart(prediction_chart, use_container_width=True)
        
#         # Additional insights
#         st.markdown("### 🔍 Detailed Analysis")
        
#         # Risk assessment - exclude "No Stroke" from stroke probability
#         stroke_probability = sum([prob for i, prob in enumerate(prediction[0]) 
#                                 if "Stroke" in class_labels[i] and class_labels[i] != "No Stroke"]) * 100
        
#         # Add positive message for No Stroke predictions
#         if predicted_label == "No Stroke":
#             no_stroke_probability = prediction[0][class_labels.index("No Stroke")] * 100
#             st.success(f"🎉 **Good News! No stroke detected** - {no_stroke_probability:.1f}% No Stroke vs {stroke_probability:.1f}% combined stroke probability")
#             # Add stroke risk assessment below the positive message
#             if stroke_probability > 50:
#                 st.error(f"⚠️ **High stroke risk detected**: {stroke_probability:.1f}% combined stroke probability")
#             elif stroke_probability > 20:
#                 st.warning(f"⚡ **Moderate stroke risk**: {stroke_probability:.1f}% combined stroke probability")
#             else:
#                 st.info(f"✅ **Low stroke risk**: {stroke_probability:.1f}% combined stroke probability")
#         elif stroke_probability > 50:
#             st.error(f"⚠️ **High stroke risk detected**: {stroke_probability:.1f}% combined stroke probability")
#         elif stroke_probability > 20:
#             st.warning(f"⚡ **Moderate stroke risk**: {stroke_probability:.1f}% combined stroke probability")
#         else:
#             st.success(f"✅ **Low stroke risk**: {stroke_probability:.1f}% combined stroke probability")
        
#         # Confidence level assessment
#         if confidence > 0.8:
#             confidence_level = "Very High"
#             confidence_color = "🟢"
#         elif confidence > 0.6:
#             confidence_level = "High"
#             confidence_color = "🟡"
#         elif confidence > 0.4:
#             confidence_level = "Moderate"
#             confidence_color = "🟠"
#         else:
#             confidence_level = "Low"
#             confidence_color = "🔴"
        
#         st.info(f"{confidence_color} **Model Confidence Level**: {confidence_level} ({confidence * 100:.1f}%)")

#         # Medical disclaimer
#         st.markdown("---")
#         st.warning("""
#         **⚕️ Important Medical Notice:** This tool is for preliminary assessment only. 
#         Always consult a qualified medical professional for proper diagnosis and treatment.
#         AI predictions should not replace professional medical evaluation.
#         """)

#     except Exception as e:
#         st.error(f"Error processing image: {str(e)}")

# elif not model_loaded:
#     st.error("Model failed to load. Please check the model file and try again.")

# # Footer
# st.markdown("---")
# st.markdown("""
#     <div style="text-align: center; color: gray; font-size: 14px;">
#     <p>Brain Stroke MRI Classifier v1.0 | Medical AI Tool</p>
#     </div>
#     """, unsafe_allow_html=True)

=======
>>>>>>> bf54e50a1f07a62adfca0d5c6a0935cac1469456
import streamlit as st
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import io
import os

# Set backend BEFORE importing keras
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from huggingface_hub import hf_hub_download
import tensorflow as tf

@st.cache_resource
def load_model_from_hf():
    """Load specific model from Hugging Face Hub"""
    try:
        st.info("Loading model 250|15.h5 from Hugging Face...")
        
        # Method 1: Try direct Keras loading
        try:
            model = keras.saving.load_model("hf://Ajay007001/Brain-Stroke-Prediction/model 250|15.h5")
            st.success("✅ Model 250|15 loaded successfully with Keras!")
            return model
        except Exception as e1:
            st.warning(f"Keras direct loading failed: {str(e1)}")
            
            # Method 2: Use huggingface_hub to download then load
            st.info("Trying huggingface_hub download method...")
            model_path = hf_hub_download(
                repo_id="Ajay007001/Brain-Stroke-Prediction",
                filename="model 250|15.h5",
                cache_dir="./hf_cache"
            )
            model = tf.keras.models.load_model(model_path)
            st.success("✅ Model loaded with huggingface_hub download!")
            return model
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Standardized medical terminology
class_labels = [
    'Hemorrhagic Stroke', 
    'Competition Dataset Session 1',
    'Competition Dataset Session 2', 
    'No Stroke', 
    'Ischemic Stroke'
]

img_siz = 250

# Configure Streamlit page
st.set_page_config(
    page_title="Stroke MRI Detector",
    layout='centered',
    page_icon="🧠"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .header {
        font-size: 24px !important;
        font-weight: bold !important;
    }
    .no-stroke {
        color: #2ecc71 !important;
        font-weight: bold !important;
        font-size: 22px !important;
    }
    .stroke {
        color: #e74c3c !important;
        font-weight: bold !important;
        font-size: 22px !important;
    }
    .confidence {
        font-size: 18px !important;
        color: #3498db !important;
    }
    .error {
        color: #FF5252 !important;
    }
    .stImage>img {
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

def create_prediction_chart(prediction_probs, class_labels):
    """Create an interactive bar chart for prediction probabilities"""
    
    # Convert to percentages
    prob_percentages = [prob * 100 for prob in prediction_probs]
    
    # Create DataFrame for easier handling
    df = pd.DataFrame({
        'Class': class_labels,
        'Probability': prob_percentages
    })
    
    # Define colors based on class type
    colors = []
    for label in class_labels:
        if label == "No Stroke":
            colors.append('#2ecc71')  # Green for no stroke
        elif "Stroke" in label:
            colors.append('#e74c3c')  # Red for stroke types
        else:
            colors.append('#3498db')  # Blue for dataset entries
    
    # Create the bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Class'],
        y=df['Probability'],
        marker_color=colors,
        text=[f'{prob:.1f}%' for prob in prob_percentages],
        textposition='auto',
        textfont=dict(color='white', size=12, family='Arial Black'),
        hovertemplate='<b>%{x}</b><br>Probability: %{y:.1f}%<extra></extra>',
        name='Prediction Probability'
    ))
    
    # Update layout - disable zoom and pan
    fig.update_layout(
        title={
            'text': '📊 Prediction Probabilities',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Arial Black'}
        },
        xaxis_title='Stroke Classification',
        yaxis_title='Probability (%)',
        xaxis_tickangle=-45,
        height=500,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial", size=12),
        margin=dict(l=50, r=50, t=80, b=120),
        # Disable interactions
        dragmode=False
    )
    
    # Update axes - disable zoom
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        tickfont=dict(size=10),
        fixedrange=True  # Disable zoom on x-axis
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        range=[0, max(prob_percentages) * 1.1],
        fixedrange=True  # Disable zoom on y-axis
    )
    
    return fig

def create_summary_metrics(predicted_label, confidence, prediction_probs, class_labels):
    """Create summary metrics display"""
    
    # Find top 3 predictions
    top_indices = np.argsort(prediction_probs)[::-1][:3]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🎯 Top Prediction",
            value=predicted_label,
            delta=f"{confidence * 100:.1f}% confidence"
        )
    
    with col2:
        second_best_idx = top_indices[1]
        second_best_label = class_labels[second_best_idx]
        second_best_prob = prediction_probs[second_best_idx] * 100
        st.metric(
            label="🥈 Second Highest",
            value=second_best_label,
            delta=f"{second_best_prob:.1f}%"
        )
    
    with col3:
        third_best_idx = top_indices[2]
        third_best_label = class_labels[third_best_idx]
        third_best_prob = prediction_probs[third_best_idx] * 100
        st.metric(
            label="🥉 Third Highest",
            value=third_best_label,
            delta=f"{third_best_prob:.1f}%"
        )

st.title("🧠 Brain Stroke MRI Classifier")
st.write("Upload an MRI image to detect stroke type or confirm no stroke")

# Load model with proper error handling
model = load_model_from_hf()
model_loaded = model is not None

if model_loaded:
    st.success("🎉 Model loaded successfully! Ready for predictions.")
else:
    st.error("❌ Model failed to load. Please check your internet connection and try again.")

upload = st.file_uploader(
    "Upload an MRI image",
    type=['jpeg', 'png', 'jpg'],
    help="Supported formats: JPEG, PNG, JPG"
)

if upload and model_loaded:
    try:
        # Process image
        img = Image.open(upload).convert('RGB')
        
        # Display images
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                img,
                caption="Original Image",
                use_container_width=True
            )
        
        img_resized = img.resize((img_siz, img_siz))
        with col2:
            st.image(
                img_resized,
                caption=f"Resized to {img_siz}x{img_siz}",
                use_container_width=True
            )
        
        img_arr = img_to_array(img_resized) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        # Make prediction
        with st.spinner("Analyzing MRI image..."):
            prediction = model.predict(img_arr, verbose=0)
            predicted_index = np.argmax(prediction)
            confidence = float(np.max(prediction))
            predicted_label = class_labels[predicted_index]

        # Display results with conditional formatting
        st.markdown("---")
        
        if predicted_label == "No Stroke":
            st.markdown(f'### <span class="no-stroke">✅ No Stroke Detected</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'### <span class="stroke">⚠️ Detected: {predicted_label}</span>', unsafe_allow_html=True)
        
        st.markdown(f'### <span class="confidence">🔍 Confidence: {confidence * 100:.2f}%</span>', unsafe_allow_html=True)
        
        # Confidence progress bar
        st.progress(confidence)
        
        # Summary metrics
        st.markdown("### 📈 Prediction Summary")
        create_summary_metrics(predicted_label, confidence, prediction[0], class_labels)
        
        # Interactive prediction chart
        st.markdown("---")
        prediction_chart = create_prediction_chart(prediction[0], class_labels)
        st.plotly_chart(prediction_chart, use_container_width=True)
        
        # Additional insights
        st.markdown("### 🔍 Detailed Analysis")
        
        # Risk assessment
        stroke_probability = sum([prob for i, prob in enumerate(prediction[0]) 
                                if "Stroke" in class_labels[i] and class_labels[i] != "No Stroke"]) * 100
        
        # Add positive message for No Stroke predictions
        if predicted_label == "No Stroke":
            no_stroke_probability = prediction[0][class_labels.index("No Stroke")] * 100
            st.success(f"🎉 **Good News! No stroke detected** - {no_stroke_probability:.1f}% No Stroke vs {stroke_probability:.1f}% combined stroke probability")
        
        # Risk level assessment
        if stroke_probability > 50:
            st.error(f"⚠️ **High stroke risk detected**: {stroke_probability:.1f}% combined stroke probability")
        elif stroke_probability > 20:
            st.warning(f"⚡ **Moderate stroke risk**: {stroke_probability:.1f}% combined stroke probability")
        else:
            st.info(f"✅ **Low stroke risk**: {stroke_probability:.1f}% combined stroke probability")
        
        # Confidence level assessment
        if confidence > 0.8:
            confidence_level = "Very High"
            confidence_color = "🟢"
        elif confidence > 0.6:
            confidence_level = "High"
            confidence_color = "🟡"
        elif confidence > 0.4:
            confidence_level = "Moderate"
            confidence_color = "🟠"
        else:
            confidence_level = "Low"
            confidence_color = "🔴"
        
        st.info(f"{confidence_color} **Model Confidence Level**: {confidence_level} ({confidence * 100:.1f}%)")

        # Medical disclaimer
        st.markdown("---")
        st.warning("""
        **⚕️ Important Medical Notice:** This tool is for preliminary assessment only. 
        Always consult a qualified medical professional for proper diagnosis and treatment.
        AI predictions should not replace professional medical evaluation.
        """)

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("Please try uploading a different image or check the image format.")

elif upload and not model_loaded:
    st.error("❌ Cannot make predictions - model failed to load.")

elif not upload and model_loaded:
    st.info("👆 Please upload an MRI image to start the analysis.")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray; font-size: 14px;">
    <p>Brain Stroke MRI Classifier v1.0 | Medical AI Tool</p>
    <p>Model: VGG19-based (250x15) | Source: Hugging Face</p>
    </div>
    """, unsafe_allow_html=True)
