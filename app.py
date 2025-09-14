import streamlit as st
import pandas as pd
import joblib

# Your prediction function
def make_predictions_from_excel(model_path, file):
    try:
        # Load the saved model
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"❌ Error: The model file at {model_path} was not found.")
        return None

    try:
        # Read uploaded Excel file into a DataFrame
        new_data_df = pd.read_csv(file)
        st.success(f"✅ New data loaded successfully.")
    except Exception as e:
        st.error(f"❌ Error while loading Excel file: {e}")
        return None

    # Make predictions
    predictions = model.predict(new_data_df)
    st.success("✅ Predictions completed.")

    # Put predictions into DataFrame
    predictions_df = pd.DataFrame(predictions, columns=['Predicted Target'])

    # Convert class indices back to labels (if classification)
    if hasattr(model.named_steps['model'], 'classes_'):
        class_mapping = {i: label for i, label in enumerate(model.named_steps['model'].classes_)}
        predictions_df['Predicted Target'] = predictions_df['Predicted Target'].map(class_mapping)

    return predictions_df


# -------- Streamlit UI --------
st.title("📊 Excel Prediction App")

# File uploader
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

# Model path input
model_path = st.text_input("Enter path to your saved model (.joblib)", "model.joblib")

if uploaded_file is not None and model_path:
    if st.button("Run Predictions"):
        predictions_df = make_predictions_from_excel(model_path, uploaded_file)

        if predictions_df is not None:
            st.subheader("Prediction Results")
            st.dataframe(predictions_df)

            # Option to download predictions
            csv = predictions_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Predictions as CSV",
                csv,
                "predictions.csv",
                "text/csv",
                key="download-csv"
            )
