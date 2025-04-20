import streamlit as st
import pandas as pd
import joblib  # Or import pickle if you used that

# --- Load Pre-trained Model, Encoder, and Columns ---
# IMPORTANT: Make sure you have saved these files from your notebook first!
# Example using joblib:
# model = joblib.load('random_forest_model.joblib')
# encoder = joblib.load('one_hot_encoder.joblib')
# X_final_columns = joblib.load('X_final_columns.joblib') # List of column names

# Placeholder loading - Replace with your actual file paths
try:
    model = joblib.load('random_forest_model.joblib')
    encoder = joblib.load('one_hot_encoder.joblib')
    X_final_columns = joblib.load('X_final_columns.joblib')
    # Ensure this list matches the categorical columns used during training
    categorical_cols = ['Soil_Type', 'Crop_Type']
except FileNotFoundError:
    st.error("Error: Model, encoder, or column list file not found.")
    st.error("Please ensure 'random_forest_model.joblib', 'one_hot_encoder.joblib', and 'X_final_columns.joblib' are in the same directory as app.py.")
    st.stop() # Stop execution if files are missing
except Exception as e:
    st.error(f"An error occurred loading files: {e}")
    st.stop()

# --- Fertilizer Recommendation Function (Adapted from Notebook) ---
def recommend_fertilizer(soil_type, crop_type, temperature, humidity, nitrogen, phosphorous, potassium, moisture):
    """Predicts fertilizer based on input features."""
    user_input = pd.DataFrame({
        'Soil_Type': [soil_type],
        'Crop_Type': [crop_type],
        'Temperature': [temperature],
        'Humidity': [humidity],
        'Nitrogen': [nitrogen],
        'Phosphorous': [phosphorous],
        'Potassium': [potassium],
        'Moisture': [moisture]
    })

    # Encode categorical features
    try:
        user_encoded = pd.DataFrame(encoder.transform(user_input[categorical_cols]))
        # Get feature names after encoding
        user_encoded.columns = encoder.get_feature_names_out(categorical_cols)
    except Exception as e:
        st.error(f"Error during encoding input: {e}")
        return "Encoding Error"

    # Combine encoded and numerical features
    # Ensure numerical columns are present before dropping
    cols_to_drop = [col for col in categorical_cols if col in user_input.columns]
    user_numeric = user_input.drop(columns=cols_to_drop)
    user_final = pd.concat([user_encoded, user_numeric], axis=1)

    # Reindex to match the columns used during training
    # Fill missing columns with 0 (important for one-hot encoding)
    try:
        user_final = user_final.reindex(columns=X_final_columns, fill_value=0)
    except Exception as e:
        st.error(f"Error during reindexing columns: {e}")
        return "Column Matching Error"

    # Make prediction
    try:
        prediction = model.predict(user_final)[0]
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Prediction Error"

# --- Fertilizer Image Mapping (Example - Customize this!) ---
# Replace keys with actual fertilizer names predicted by your model
# Replace values with paths to your images (e.g., in an 'images' folder)
fertilizer_images = {
    "Urea": "images/urea.jpg",
    "DAP": "images/dap.jpg",
    "Potash": "images/potash.jpg",
    "10/26/2026": "images/10-26-26.jpg",
    "17-17-17": "images/17-17-17.jpg",
    "20-20": "images/20-20.jpg",
    "28-28": "images/28-28.jpg",
    "14-35-14": "images/14-35-14.jpg",
    # Add entries for all possible fertilizer outputs from your model
    "Default": "images/default.jpg" # Fallback image
}

# --- Fertilizer Descriptions (Add your descriptions here!) ---
fertilizer_descriptions = {
    "Urea": {
        "en": "Urea is a nitrogen-rich fertilizer (46-0-0) primarily used to provide nitrogen to plants, promoting leafy green growth. Apply as per soil test recommendations.",
        "hi": "यूरिया एक नाइट्रोजन युक्त उर्वरक (46-0-0) है जिसका उपयोग मुख्य रूप से पौधों को नाइट्रोजन प्रदान करने के लिए किया जाता है, जिससे पत्तेदार हरी वृद्धि को बढ़ावा मिलता है। मिट्टी परीक्षण की सिफारिशों के अनुसार प्रयोग करें।"
    },
    "DAP": {
        "en": "Diammonium Phosphate (DAP) (18-46-0) is a source of both nitrogen and phosphorus. It's excellent for root development and early plant growth.",
        "hi": "डाईअमोनियम फॉस्फेट (डीएपी) (18-46-0) नाइट्रोजन और फास्फोरस दोनों का स्रोत है। यह जड़ विकास और पौधे की शुरुआती वृद्धि के लिए उत्कृष्ट है।"
    },
    "Potash": {
        "en": "Muriate of Potash (MOP) (0-0-60) provides potassium, essential for overall plant health, disease resistance, and fruit/flower development.",
        "hi": "म्यूरेट ऑफ पोटाश (एमओपी) (0-0-60) पोटेशियम प्रदान करता है, जो पौधे के समग्र स्वास्थ्य, रोग प्रतिरोधक क्षमता और फल/फूल विकास के लिए आवश्यक है।"
    },
    "10-26-26": { # Note: Key updated to match your image dictionary if it was '10/26/2026'
        "en": "A complex fertilizer providing Nitrogen (10%), Phosphorus (26%), and Potassium (26%). Suitable for crops requiring balanced nutrients, especially P and K.",
        "hi": "एक जटिल उर्वरक जो नाइट्रोजन (10%), फास्फोरस (26%), और पोटेशियम (26%) प्रदान करता है। संतुलित पोषक तत्वों, विशेष रूप से पी और के की आवश्यकता वाली फसलों के लिए उपयुक्त है।"
    },
    "17-17-17": {
        "en": "A balanced NPK fertilizer providing equal parts Nitrogen, Phosphorus, and Potassium. Good for general purpose use during vegetative growth.",
        "hi": "एक संतुलित एनपीके उर्वरक जो नाइट्रोजन, फास्फोरस और पोटेशियम के बराबर हिस्से प्रदान करता है। वानस्पतिक विकास के दौरान सामान्य प्रयोजन के उपयोग के लिए अच्छा है।"
    },
    "20-20": { # Assuming this means 20-20-0 or 20-20-20, adjust description accordingly
        "en": "Often refers to 20-20-0 (like Ammonium Phosphate Sulphate) or 20-20-20. Provides Nitrogen and Phosphorus (and Potassium if 20-20-20). Good for initial growth stages.",
        "hi": "अक्सर 20-20-0 (जैसे अमोनियम फॉस्फेट सल्फेट) या 20-20-20 को संदर्भित करता है। नाइट्रोजन और फास्फोरस (और पोटेशियम यदि 20-20-20) प्रदान करता है। प्रारंभिक विकास चरणों के लिए अच्छा है।"
    },
    "28-28": { # Assuming 28-28-0
        "en": "A fertilizer with high Nitrogen (28%) and Phosphorus (28%). Promotes vigorous vegetative growth and root development.",
        "hi": "उच्च नाइट्रोजन (28%) और फास्फोरस (28%) वाला उर्वरक। जोरदार वानस्पतिक विकास और जड़ विकास को बढ़ावा देता है।"
    },
    "14-35-14": {
        "en": "Provides Nitrogen (14%), high Phosphorus (35%), and Potassium (14%). Excellent for promoting root growth, flowering, and fruiting.",
        "hi": "नाइट्रोजन (14%), उच्च फास्फोरस (35%), और पोटेशियम (14%) प्रदान करता है। जड़ वृद्धि, फूल और फलन को बढ़ावा देने के लिए उत्कृष्ट है।"
    },
    # Add entries for all other possible fertilizer outputs from your model
    "Default": {
        "en": "No specific description available.",
        "hi": "कोई विशिष्ट विवरण उपलब्ध नहीं है।"
    }
}

# --- Streamlit UI ---
st.set_page_config(page_title="Fertilizer Recommender", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling (Optional)
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 15px;
        padding: 10px 0px;
        font-size: 1.1em;
        font-weight: bold;
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #2E8B57; /* SeaGreen */
    }
</style>
""", unsafe_allow_html=True)

st.title("🌾 Advanced Fertilizer Recommendation System 🌱")
st.markdown("Provide the details of your soil and crop conditions to receive a tailored fertilizer recommendation.")

# --- Input Form in Sidebar ---
st.sidebar.header("Input Parameters")
with st.sidebar.form("recommendation_form"):
    st.subheader("Soil & Crop")
    # --- Get unique values from your data or define them ---
    # Example: Replace with actual unique values from your 'Fertilizer Prediction.csv'
    soil_type_options = sorted(['Loamy', 'Sandy', 'Clayey', 'Black', 'Red']) # Add all types from your data
    crop_type_options = sorted(['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 'Pulses', 'Barley', 'Ground Nuts']) # Add all crops

    soil_type = st.selectbox("Soil Type", options=soil_type_options)
    crop_type = st.selectbox("Crop Type", options=crop_type_options)

    st.subheader("Environment")
    temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=60.0, value=26.0, step=0.5, format="%.1f")
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=52.0, step=0.5, format="%.1f")

    st.subheader("Nutrients & Moisture")
    nitrogen = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=40, step=1)
    phosphorous = st.number_input("Phosphorous (P)", min_value=0, max_value=200, value=60, step=1)
    potassium = st.number_input("Potassium (K)", min_value=0, max_value=200, value=40, step=1)
    moisture = st.number_input("Soil Moisture (%)", min_value=0, max_value=100, value=65, step=1) # Adjust label/range if needed

    submitted = st.form_submit_button("✨ Get Recommendation")

# --- Display Results ---
st.header("Recommendation Result")
if submitted:
    prediction = recommend_fertilizer(
        soil_type, crop_type, temperature, humidity, nitrogen, phosphorous, potassium, moisture
    )

    if prediction not in ["Encoding Error", "Column Matching Error", "Prediction Error"]:
        # Display recommendation text (using markdown for size)
        st.markdown(f"""
        <p style="font-size: 24px; color: #2E8B57; font-weight: bold;">
            Recommended Fertilizer: {prediction}
        </p>
        """, unsafe_allow_html=True)

        # Display image using the original prediction string as the key
        image_path = fertilizer_images.get(prediction, fertilizer_images["Default"])
        try:
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                 st.image(image_path, caption=f"{prediction}", use_container_width=True)
        except FileNotFoundError:
            st.warning(f"Image not found for {prediction} at path: {image_path}. Displaying default.")
            try:
                 col1, col2, col3 = st.columns([1, 3, 1])
                 with col2:
                    st.image(fertilizer_images["Default"], caption="Default Fertilizer Image", use_container_width=True)
            except FileNotFoundError:
                 st.error("Default image not found either.") # Handle missing default image
        except Exception as e:
             st.warning(f"Could not load image: {e}")

        # Display Fertilizer Descriptions
        st.markdown("---")
        st.subheader("Fertilizer Information")

        description_key = prediction
        if prediction == "10/26/2026":
            description_key = "10-26-26"
        description_info = fertilizer_descriptions.get(description_key, fertilizer_descriptions["Default"])

        with st.expander("English Description"):
            st.markdown(f"<p style='font-size: 16px;'>{description_info.get('en', 'Not available.')}</p>", unsafe_allow_html=True)
        with st.expander("हिंदी विवरण (Hindi Description)"):
            st.markdown(f"<p style='font-size: 16px;'>{description_info.get('hi', 'उपलब्ध नहीं है।')}</p>", unsafe_allow_html=True)

    else:
        st.warning("Prediction could not be generated due to errors.")

else:
    # --- MODIFIED: Display introductory text when no submission yet ---
    st.subheader("🌱 Why Use Fertilizers?")
    st.markdown("""
    <p style='font-size: 16px;'>
    Fertilizers play a crucial role in modern agriculture by supplying essential nutrients that crops need to grow strong and healthy. Soil often lacks sufficient amounts of these nutrients naturally, especially after repeated cultivation. Applying the right fertilizer at the right time ensures optimal plant growth, leading to higher yields and better quality produce, which is vital for feeding the world's growing population. This tool helps you choose the most suitable fertilizer based on your specific soil and crop conditions.
    </p>
    """, unsafe_allow_html=True)

    st.subheader("🌱 उर्वरकों का उपयोग क्यों करें?")
    st.markdown("""
    <p style='font-size: 16px;'>
    उर्वरक आधुनिक कृषि में महत्वपूर्ण भूमिका निभाते हैं क्योंकि वे फसलों को मजबूत और स्वस्थ रूप से विकसित होने के लिए आवश्यक पोषक तत्व प्रदान करते हैं। मिट्टी में अक्सर इन पोषक तत्वों की पर्याप्त मात्रा स्वाभाविक रूप से कम होती है, खासकर बार-बार खेती करने के बाद। सही समय पर सही उर्वरक का प्रयोग पौधों की इष्टतम वृद्धि सुनिश्चित करता है, जिससे उच्च पैदावार और बेहतर गुणवत्ता वाली उपज प्राप्त होती है, जो दुनिया की बढ़ती आबादी के लिए महत्वपूर्ण है। यह उपकरण आपकी विशिष्ट मिट्टी और फसल की स्थितियों के आधार पर सबसे उपयुक्त उर्वरक चुनने में आपकी सहायता करता है।
    </p>
    <br>
    <p style='font-size: 16px;'><i>Please input the parameters in the sidebar and click 'Get Recommendation' to proceed.</i></p>
    """, unsafe_allow_html=True)
    # --- END MODIFIED SECTION ---


st.markdown("---")
st.markdown("Developed by Shreya, Bhagyashree and Gaurvi")