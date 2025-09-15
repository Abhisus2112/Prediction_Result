import streamlit as st
import pandas as pd
import joblib
import io

# Prediction function
def make_predictions_from_file(model_file, data_file):
    try:
        # Load saved model
        # model_file is an UploadedFile, so use .read() and io.BytesIO
        model = joblib.load(model_file)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

    try:
        # Load new data
        if data_file.name.endswith(".csv"):
            new_data_df = pd.read_csv(data_file)
        else:
            new_data_df = pd.read_excel(data_file)
        st.success("✅ New data loaded successfully.")
    except Exception as e:
        st.error(f"❌ Error while loading new data file: {e}")
        return None

    try:
        # Make predictions
        predictions = model.predict(new_data_df)
        st.success("✅ Predictions completed.")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        return None

    # Wrap in DataFrame
    predictions_df = pd.DataFrame(predictions, columns=["Predicted Target"])

    # If classification, map class labels
    if hasattr(model.named_steps, "model") and hasattr(model.named_steps["model"], "classes_"):
        class_mapping = {i: label for i, label in enumerate(model.named_steps["model"].classes_)}
        predictions_df["Predicted Target"] = predictions_df["Predicted Target"].map(class_mapping)

    return predictions_df

# ===== Streamlit UI =====
st.title("📊 Prediction App: Upload Model + Data")

data_file = st.file_uploader("Upload data file (CSV or Excel)", type=["csv", "xlsx", "xls"])
model_file = st.file_uploader("Upload model file (.joblib)", type=["joblib"])

if data_file is not None and model_file is not None:
    if st.button("Run Predictions"):
        preds_df = make_predictions_from_file(model_file, data_file)
        if preds_df is not None:
            st.subheader("Prediction Results")
            st.dataframe(preds_df)

            csv = preds_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Predictions as CSV",
                csv,
                "predictions.csv",
                "text/csv"
            )
