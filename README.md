# ♻️ Sortify Trash Classifier

**Sortify** is an AI-powered trash classification app that helps users identify and sort waste into **four categories**:
- ♻️ **Recyclable**
- 🌿 **Organic**
- ☣️ **Hazardous**
- 🗑️ **Others**

The app also provides **smart disposal tips** to encourage proper waste handling and improve environmental awareness.

---

## 🚀 How It Works

Sortify uses a custom-trained **ResNet50 deep learning model** built with PyTorch. Users can upload images of trash through a simple web interface, and the model will:

1. Classify the trash into one of four categories.
2. Display the prediction confidence.
3. Suggest an appropriate action or disposal tip.

---

## 🧠 Model Performance

- ✅ **Test Accuracy:** `98.12%`
- ✅ **Validation Accuracy:** Consistent performance during training
- ✅ **No data leakage**: Strict separation between train, validation, and test datasets
- ✅ **Balanced dataset**: All 4 trash categories are well-represented
- ✅ **Real-world examples**: Test set contains actual images (not synthetic or overly clean)

> ⚠️ Note: This accuracy is meaningful because the dataset was carefully curated and balanced, and the model was evaluated on unseen, real-world images.

---

## 🧪 Tech Stack

- **Frontend:** Streamlit (previously HTML and CSS)
- **Backend (Model):** FastAPI (previously), now integrated into Streamlit
- **Model Architecture:** ResNet50 (with modified output layer)
- **Framework:** PyTorch
- **Deployment:** [Hugging Face Spaces](https://huggingface.co/spaces/marvelmmy/sortify-app)

---

## 📸 Try It Live

👉 [Click here to try Sortify on Hugging Face](https://huggingface.co/spaces/marvelmmy/sortify-app)

---

## 📂 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/marvelmmy/sortify-app.git
   cd sortify-app
