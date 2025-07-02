import streamlit as st
from PIL import Image
#from onnx_yolo_wrapper import YOLO
import cv2
import numpy
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
import os
import requests
import numpy as np

st.set_page_config(page_title="Image Input App", layout="centered")

language = st.selectbox("🌐 Choose Language / மொழியைத் தேர்ந்தெடுக்கவும் / Pilih Bahasa", ["English", "Tamil", "Malay"])

class_names = ["apple", "banana", "plastic", "glass"]  # Add your real class list here
#model = YOLO("best.onnx", class_names)



def predict_image_class(image, model):
    image = image.convert('RGB')
    img = np.array(image)
    img_resized = cv2.resize(img, (256, 256))
    img_normalized = img_resized / 255.0
    img_reshaped = img_normalized.reshape(1, 256, 256, 3)

    prediction = model.predict(img_reshaped)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    return predicted_class


if language == "English":
    st.title("Waste Classification App - Keep Environment Clean")

    

    def predict_image_class(image, model):
        image = image.convert('RGB')
        img = np.array(image)
        img_resized = cv2.resize(img, (256, 256))
        img_normalized = img_resized / 255.0
        img_reshaped = img_normalized.reshape(1, 256, 256, 3)
        prediction = model.predict(img_reshaped)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        return predicted_class

    with st.sidebar:
        st.subheader('What is Biodegradable material?')
        st.write('Biodegradable material is any substance that can be broken down into natural elements like water, carbon dioxide, and biomass by living organisms over time without harming the environment. Example: Apple, Bananas, etc')
        st.subheader('What is Non-biodegradable material?')
        st.write('Non-biodegradable material is any substance that cannot be decomposed naturally by microorganisms and remains in the environment for a very long time. Example: Plastic bags, plastic bottles, glass, etc')
        st.subheader('What is CNN')
        st.write('A class of deep neural networks commonly used in computer vision that automatically and adaptively learns spatial hierarchies of features through convolutional layers.')
        st.subheader('What is Machine learning?')
        st.write('A branch of artificial intelligence that uses statistical techniques to give computers the ability to learn from data without being explicitly programmed.')
        st.subheader('What is streamlit?')
        st.write('A Python framework that turns data scripts into interactive web applications with minimal code, supporting widgets, charts, and real-time updates.')
        st.subheader('What is YOLOv8?')
        st.write('YOLOv8 is a computer program that can see and detect objects in images or videos — like bananas, bottles, or people. YOLO stands for You Only Look Once, which means it looks at the image just one time and quickly finds all the objects.')

    a, b = st.tabs(['Home', 'Detection and report'])

    with b:
        st.title("📷 Image Input App")
        st.markdown("""
        **Choose how you'd like to provide an image:**
        - **Upload Image** to select from your device.
        - **Take Photo** to capture with your webcam.
        """)

        if 'mode' not in st.session_state:
            st.session_state.mode = None
        if 'image' not in st.session_state:
            st.session_state.image = None

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Upload Image 📁"):
                st.session_state.mode = 'upload'
        with col2:
            if st.button("Take Photo 📷"):
                st.session_state.mode = 'capture'

        if st.session_state.mode == 'upload':
            uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                st.session_state.image = Image.open(uploaded_file)
        elif st.session_state.mode == 'capture':
            camera_image = st.camera_input("Take a picture")
            if camera_image is not None:
                st.session_state.image = Image.open(camera_image)

        if st.session_state.image is not None:
            st.subheader("Selected Image")
            st.image(st.session_state.image, caption="Here is the image you provided", use_container_width=True)

            if st.button("🔍 Detect Objects"):
            
                st.subheader("YOLOv8 Detection Results")
                image_bytes = io.BytesIO()
                image_rgb = st.session_state.image.convert("RGB")
                image_rgb.save(image_bytes, format='JPEG')

                image_bytes.seek(0)
                
                # Send to FastAPI
                response = requests.post(
                    "https://fast-api-backend-8g48.onrender.com/predict",  # Change this to your deployed URL on Streamlit Cloud or public host
                    files={"file": ("image.jpg", image_bytes, "image/jpeg")},
                )
                boxes=[]
                if response.status_code == 200:
                    result = response.json()
                    boxes = result.get("results", [])
                    
                    # Draw boxes on the image
                    img = np.array(st.session_state.image.convert("RGB"))
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])
                        label = box["label"]
                        conf = float(box["confidence"])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, f"{label} {conf:.2%}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                    st.image(img, caption="Detected Objects", use_container_width=True)
                
                    # Continue with biodegradable / non-biodegradable logic using `boxes`
                    # ⬅ Your logic continues here...
                
                else:
                    st.error(f"❌ Error from detection server: {response.status_code}")
                    st.text(response.text)

                #st.image(result_img, caption="Detected Objects", use_container_width=True)
            
                biodegradable = {'apple', 'banana'}
                non_biodegradable = {'plastic', 'glass'}
                
                detected_labels = set()
                confidence_map = {}
                
                if boxes:
                    for box in boxes:
                        label = box["label"]
                        conf = float(box["confidence"])
                
                        detected_labels.add(label)
                        confidence_map[label] = f"{conf:.2%}"

            
                    st.subheader("♻️ Waste Type Classification")
                    waste_type = None
                    if detected_labels & biodegradable and not detected_labels & non_biodegradable:
                        st.success("This waste is **Biodegradable**.")
                        st.audio("biodegradable.mp3", format="audio/mp3")
                        waste_type = "Biodegradable"
                    elif detected_labels & non_biodegradable and not detected_labels & biodegradable:
                        st.error("This waste is **Non-Biodegradable**.")
                        st.audio("non-biodegradable.mp3", format="audio/mp3")
                        waste_type = "Non-Biodegradable"
                    elif detected_labels & biodegradable and detected_labels & non_biodegradable:
                        st.info("This waste contains **both Biodegradable and Non-Biodegradable** materials.")
                        waste_type = "Mixed"
                    else:
                        st.warning("No recognizable waste type found.")

                    

                    # PDF Generation
                    if waste_type in ["Biodegradable", "Non-Biodegradable"]:
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)

                        pdf.set_title("Waste Detection Report")
                        pdf.cell(200, 10, txt="Waste Detection Report", ln=True, align='C')
                        pdf.ln(10)

                        pdf.cell(200, 10, txt=f"Waste Type Detected: {waste_type}", ln=True)
                        pdf.ln(5)

                        

                        if waste_type == "Biodegradable":
                            pdf.multi_cell(0, 10, txt=(
                                "The CNN model based on deep learning accuracy: 90%\n"
                                "Yolov8 model accuracy: about 85%\n"
                                "Biodegradable waste such as apple and banana can naturally decompose.\n\n"
                                " Disposal Recommendation:\n"
                                "- Discard in green bins.\n"
                                "- Compost at home if possible.\n"
                                "- Do not mix with plastic or glass waste.\n\n"
                                " Environmental Measures:\n"
                                "- Promote organic composting.\n"
                                "- Keep surroundings clean.\n"
                                "- Educate others on separating biodegradable waste."
                            ))
                        else:
                            pdf.multi_cell(0, 10, txt=(
                                "The CNN model based on deep learning accuracy: 90%\n"
                                "Yolov8 model accuracy: about 85%\n"
                                "Non-biodegradable waste such as plastic and glass cannot decompose naturally.\n\n"
                                " Disposal Recommendation:\n"
                                "- Discard in blue bins or recycling centers.\n"
                                "- Never burn plastic waste.\n\n"
                                " Environmental Measures:\n"
                                "- Reduce plastic usage.\n"
                                "- Support recycling initiatives.\n"
                                "- Keep public areas free of litter."
                            ))

                        pdf_output = io.BytesIO()
                        pdf_bytes = pdf.output(dest='S').encode('latin-1')
                        pdf_output.write(pdf_bytes)
                        pdf_output.seek(0)

                        st.download_button(
                            label="📄 Download Waste Report as PDF",
                            data=pdf_output,
                            file_name="waste_report.pdf",
                            mime="application/pdf"
                        )

        st.divider()

    with a:
        st.write("## 🌍 Why It Matters")
        st.write("Every day, we generate tons of waste. Unfortunately, much of it ends up in the wrong places, harming our planet. Proper waste segregation is key to:")
        st.write("- Reducing environmental pollution")  
        st.write("- Enhancing recycling efficiency")  
        st.write("- Promoting a cleaner and greener future")
        st.write("Our system is designed to **distinguish between biodegradable and non-biodegradable items**, making waste disposal smarter and more sustainable.")
        st.write('---')
        st.write('#### Just capture or upload an image — and let the system do the rest!')
        st.write('---')
        st.write("### 🧠 How It Works")
        st.write("Using machine learning and image processing, our model analyzes images of waste and classifies them into two main categories:")
        st.markdown("- **Biodegradable** – Natural waste like fruits (apple, banana), leaves, etc.")
        st.markdown("- **Non-Biodegradable** – Artificial waste like plastic bottles, wrappers, etc.")



if language == "Tamil":
    st.title("கழிவு வகைப்பாடு செயலி - சுற்றுச்சூழலை சுத்தமாக வைத்திருங்கள்")

    def predict_image_class(image, model):
        image = image.convert('RGB')
        img = np.array(image)
        img_resized = cv2.resize(img, (256, 256))
        img_normalized = img_resized / 255.0
        img_reshaped = img_normalized.reshape(1, 256, 256, 3)
        prediction = model.predict(img_reshaped)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        return predicted_class

    with st.sidebar:
        st.subheader('என்ன இது – உயிர்ப்புச்சேரியக்கழிவு?')
        st.write('உயிர்ப்புச்சேரியக்கழிவு என்பது இயற்கையால் (நுண்ணுயிரிகள் மூலம்) அழிக்கக்கூடிய பொருள்கள் ஆகும். உதாரணம்: ஆப்பிள், வாழை பழம்.')
        st.subheader('என்ன இது – உயிர்ப் சாரா கழிவு?')
        st.write('உயிர்ப் சாரா கழிவுகள் என்பது இயற்கையாக அழிக்க முடியாத பொருள்கள். உதாரணம்: பிளாஸ்டிக் மூடி, கண்ணாடி.')
        st.subheader('CNN என்றால் என்ன?')
        st.write('கணினி காட்சி பயன்பாடுகளுக்கு பயன்படுத்தப்படும் ஒரு வகை ஆழமான நரம்பியல் வலைப்பின்னல்.')
        st.subheader('Machine Learning என்றால் என்ன?')
        st.write('தகவல்களில் இருந்து கற்றுக்கொள்வதற்கான செயற்கை நுண்ணறிவின் ஒரு கிளை.')
        st.subheader('Streamlit என்பது என்ன?')
        st.write('பைதான் சcrip்டுகளை எளிதில் இணைய செயலிகளாக மாற்ற உதவும் அமைப்பு.')
        st.subheader('YOLOv8 என்றால் என்ன?')
        st.write('YOLOv8 என்பது படங்கள் மற்றும் வீடியோக்களில் பொருள்களை அடையாளம் காணும் கணினி திட்டமாகும். "You Only Look Once" என்பது அதற்கு அர்த்தம்.')

    a, b = st.tabs(['முகப்பு', 'காண்பித்தல் மற்றும் அறிக்கை'])

    with b:
        st.title("📷 படம் உள்ளீடு செயலி")
        st.markdown("""
        **படத்தை எப்படிப் பெற விரும்புகிறீர்கள் என்பதைத் தேர்ந்தெடுக்கவும்:**
        - **படத்தை பதிவேற்று** உங்கள் சாதனத்தில் இருந்து தேர்ந்தெடுக்க.
        - **புகைப்படம் எடுக்கவும்** உங்கள் காமெராவைப் பயன்படுத்தி.
        """)

        if 'mode' not in st.session_state:
            st.session_state.mode = None
        if 'image' not in st.session_state:
            st.session_state.image = None

        col1, col2 = st.columns(2)
        with col1:
            if st.button("படத்தை பதிவேற்று 📁"):
                st.session_state.mode = 'upload'
        with col2:
            if st.button("புகைப்படம் எடு 📷"):
                st.session_state.mode = 'capture'

        if st.session_state.mode == 'upload':
            uploaded_file = st.file_uploader("படத்தை தேர்வு செய்யவும்", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                st.session_state.image = Image.open(uploaded_file)
        elif st.session_state.mode == 'capture':
            camera_image = st.camera_input("புகைப்படம் எடுக்கவும்")
            if camera_image is not None:
                st.session_state.image = Image.open(camera_image)

        if st.session_state.image is not None:
            st.subheader("தேர்ந்தெடுக்கப்பட்ட படம்")
            st.image(st.session_state.image, caption="நீங்கள் வழங்கிய படம்", use_container_width=True)

            if st.button("🔍 பொருள்களை கண்டறி"):
                st.subheader("YOLOv8 கண்டறிதல் முடிவுகள்")
                image_bytes = io.BytesIO()
                image_rgb = st.session_state.image.convert("RGB")
                image_rgb.save(image_bytes, format='JPEG')

                image_bytes.seek(0)
                
                response = requests.post(
                    "https://fast-api-backend-8g48.onrender.com/predict",  # உங்கள் FastAPI URL
                    files={"file": ("image.jpg", image_bytes, "image/jpeg")},
                )
                boxes = []
                if response.status_code == 200:
                    result = response.json()
                    boxes = result.get("results", [])
                    
                    img = np.array(st.session_state.image.convert("RGB"))
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])
                        label = box["label"]
                        conf = float(box["confidence"])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, f"{label} {conf:.2%}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    st.image(img, caption="கண்டறியப்பட்ட பொருட்கள்", use_container_width=True)
                else:
                    st.error(f"❌ சேவையகத்தின் பிழை: {response.status_code}")
                    st.text(response.text)

                biodegradable = {'apple', 'banana'}
                non_biodegradable = {'plastic', 'glass'}
                
                detected_labels = set()
                confidence_map = {}
                
                if boxes:
                    for box in boxes:
                        label = box["label"]
                        conf = float(box["confidence"])
                        detected_labels.add(label)
                        confidence_map[label] = f"{conf:.2%}"

                    st.subheader("♻️ கழிவின் வகை வகைப்படுத்தல்")
                    waste_type = None
                    if detected_labels & biodegradable and not detected_labels & non_biodegradable:
                        st.success("இது ஒரு **உயிர்ப்புச்சேரியக்கழிவு**.")
                        st.audio("biodegradable.mp3", format="audio/mp3")
                        waste_type = "Biodegradable"
                    elif detected_labels & non_biodegradable and not detected_labels & biodegradable:
                        st.error("இது ஒரு **உயிர்ப் சாரா கழிவு**.")
                        st.audio("non-biodegradable.mp3", format="audio/mp3")
                        waste_type = "Non-Biodegradable"
                    elif detected_labels & biodegradable and detected_labels & non_biodegradable:
                        st.info("இது **உயிர்ப்புச்சேரி மற்றும் உயிர்ப் சாரா** கழிவுகளை உள்ளடக்கியது.")
                        waste_type = "Mixed"
                    else:
                        st.warning("எந்த வகை கழிவும் அடையாளம் காணப்படவில்லை.")

                    if waste_type in ["Biodegradable", "Non-Biodegradable"]:
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)

                        pdf.set_title("Waste Detection Report")
                        pdf.cell(200, 10, txt="Waste Detection Report", ln=True, align='C')
                        pdf.ln(10)

                        pdf.cell(200, 10, txt=f"Waste Type Detected: {waste_type}", ln=True)
                        pdf.ln(5)

                        if waste_type == "Biodegradable":
                            pdf.multi_cell(0, 10, txt=(
                                "The CNN model based on deep learning accuracy: 90%\n"
                                "Yolov8 model accuracy: about 85%\n"
                                "Biodegradable waste such as apple and banana can naturally decompose.\n\n"
                                " Disposal Recommendation:\n"
                                "- Discard in green bins.\n"
                                "- Compost at home if possible.\n"
                                "- Do not mix with plastic or glass waste.\n\n"
                                " Environmental Measures:\n"
                                "- Promote organic composting.\n"
                                "- Keep surroundings clean.\n"
                                "- Educate others on separating biodegradable waste."
                            ))
                        else:
                            pdf.multi_cell(0, 10, txt=(
                                "The CNN model based on deep learning accuracy: 90%\n"
                                "Yolov8 model accuracy: about 85%\n"
                                "Non-biodegradable waste such as plastic and glass cannot decompose naturally.\n\n"
                                " Disposal Recommendation:\n"
                                "- Discard in blue bins or recycling centers.\n"
                                "- Never burn plastic waste.\n\n"
                                " Environmental Measures:\n"
                                "- Reduce plastic usage.\n"
                                "- Support recycling initiatives.\n"
                                "- Keep public areas free of litter."
                            ))

                        pdf_output = io.BytesIO()
                        pdf_bytes = pdf.output(dest='S').encode('latin-1')
                        pdf_output.write(pdf_bytes)
                        pdf_output.seek(0)

                        st.download_button(
                            label="📄 கழிவு அறிக்கையை PDF ஆக பதிவிறக்கவும்",
                            data=pdf_output,
                            file_name="waste_report.pdf",
                            mime="application/pdf"
                        )

        st.divider()

    with a:
        st.write("## 🌍 ஏன் இது முக்கியம்?")
        st.write("நாம் தினமும் உருவாக்கும் கழிவுகளில் பெரும்பான்மையானவை தவறான இடங்களில் கிடப்பதால் சுற்றுச்சூழலுக்கு தீங்கு விளைவிக்கின்றன.")
        st.write("- சுற்றுச்சூழல் மாசுபாட்டை குறைக்கும்")  
        st.write("- மறுசுழற்சி திறனை அதிகரிக்கும்")  
        st.write("- சுத்தமான மற்றும் பசுமையான எதிர்காலத்தை உருவாக்கும்")
        st.write("இந்த செயலி **உயிர்ப்புச்சேரி மற்றும் உயிர்ப் சாரா கழிவுகளை** அடையாளம் காணுகிறது, மேலும் கையாளுவதில் உதவுகிறது.")
        st.write('---')
        st.write('#### படமொன்றை பதிவேற்றவும் அல்லது எடுக்கவும் — உங்கள் வேலை முடிந்தது!')
        st.write('---')
        st.write("### 🧠 இது எப்படிச் செயல்படுகிறது?")
        st.write("இயந்திரக் கற்றல் மற்றும் படம் பகுப்பாய்வைப் பயன்படுத்தி, இந்த முறை உங்கள் படங்களில் உள்ள கழிவுகளை இரண்டு வகைகளில் வகைப்படுத்துகிறது:")
        st.markdown("- **உயிர்ப்புச்சேரி** – இயற்கை கழிவுகள்: பழங்கள் (ஆப்பிள், வாழைப்பழம்), இலைகள்.")
        st.markdown("- **உயிர்ப் சாரா** – செயற்கை கழிவுகள்: பிளாஸ்டிக் பாட்டில்கள், மூடிகள்.")
    
    


if language == "Malay":
    st.title("Aplikasi Pengelasan Sisa - Kekalkan Alam Sekitar Bersih")

    def predict_image_class(image, model):
        image = image.convert('RGB')
        img = np.array(image)
        img_resized = cv2.resize(img, (256, 256))
        img_normalized = img_resized / 255.0
        img_reshaped = img_normalized.reshape(1, 256, 256, 3)
        prediction = model.predict(img_reshaped)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        return predicted_class

    with st.sidebar:
        st.subheader('Apa itu bahan biodegradasi?')
        st.write('Bahan biodegradasi ialah bahan yang boleh diuraikan secara semula jadi oleh mikroorganisma menjadi unsur seperti air, karbon dioksida, dan bahan organik tanpa mencemarkan alam sekitar. Contoh: Epal, Pisang.')
        st.subheader('Apa itu bahan tidak biodegradasi?')
        st.write('Bahan tidak biodegradasi tidak boleh diuraikan oleh mikroorganisma dan kekal dalam alam sekitar untuk jangka masa yang lama. Contoh: Beg plastik, botol plastik, kaca.')
        st.subheader('Apa itu CNN?')
        st.write('Rangkaian neural mendalam yang digunakan dalam penglihatan komputer untuk mengenal pasti ciri dalam imej secara automatik.')
        st.subheader('Apa itu Pembelajaran Mesin?')
        st.write('Cabang kecerdasan buatan yang membolehkan komputer belajar dari data tanpa diprogramkan secara langsung.')
        st.subheader('Apa itu Streamlit?')
        st.write('Kerangka kerja Python untuk membina aplikasi web interaktif daripada skrip data dengan mudah.')
        st.subheader('Apa itu YOLOv8?')
        st.write('YOLOv8 adalah program komputer yang boleh mengenal pasti objek dalam imej atau video seperti pisang, botol, atau manusia. YOLO bermaksud "You Only Look Once".')

    a, b = st.tabs(['Laman Utama', 'Pengesanan & Laporan'])

    with b:
        st.title("📷 Aplikasi Input Imej")
        st.markdown("""
        **Pilih cara untuk memuatkan imej:**
        - **Muat Naik Imej** dari peranti anda.
        - **Ambil Gambar** menggunakan kamera web anda.
        """)

        if 'mode' not in st.session_state:
            st.session_state.mode = None
        if 'image' not in st.session_state:
            st.session_state.image = None

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Muat Naik Imej 📁"):
                st.session_state.mode = 'upload'
        with col2:
            if st.button("Ambil Gambar 📷"):
                st.session_state.mode = 'capture'

        if st.session_state.mode == 'upload':
            uploaded_file = st.file_uploader("Pilih fail imej", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                st.session_state.image = Image.open(uploaded_file)
        elif st.session_state.mode == 'capture':
            camera_image = st.camera_input("Ambil gambar")
            if camera_image is not None:
                st.session_state.image = Image.open(camera_image)

        if st.session_state.image is not None:
            st.subheader("Imej Dipilih")
            st.image(st.session_state.image, caption="Inilah imej yang anda berikan", use_container_width=True)

            if st.button("🔍 Kesan Objek"):
                st.subheader("Keputusan Pengesanan YOLOv8")
                image_bytes = io.BytesIO()
                image_rgb = st.session_state.image.convert("RGB")
                image_rgb.save(image_bytes, format='JPEG')

                image_bytes.seek(0)

                response = requests.post(
                    "https://fast-api-backend-8g48.onrender.com/predict",
                    files={"file": ("image.jpg", image_bytes, "image/jpeg")},
                )
                boxes = []
                if response.status_code == 200:
                    result = response.json()
                    boxes = result.get("results", [])

                    img = np.array(st.session_state.image.convert("RGB"))
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])
                        label = box["label"]
                        conf = float(box["confidence"])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, f"{label} {conf:.2%}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    st.image(img, caption="Objek Dikenal Pasti", use_container_width=True)
                else:
                    st.error(f"❌ Ralat dari pelayan pengesanan: {response.status_code}")
                    st.text(response.text)

                biodegradable = {'apple', 'banana'}
                non_biodegradable = {'plastic', 'glass'}

                detected_labels = set()
                confidence_map = {}

                if boxes:
                    for box in boxes:
                        label = box["label"]
                        conf = float(box["confidence"])
                        detected_labels.add(label)
                        confidence_map[label] = f"{conf:.2%}"

                    st.subheader("♻️ Pengelasan Jenis Sisa")
                    waste_type = None
                    if detected_labels & biodegradable and not detected_labels & non_biodegradable:
                        st.success("Sisa ini adalah **Biodegradasi**.")
                        st.audio("biodegradable.mp3", format="audio/mp3")
                        waste_type = "Biodegradable"
                    elif detected_labels & non_biodegradable and not detected_labels & biodegradable:
                        st.error("Sisa ini adalah **Tidak Biodegradasi**.")
                        st.audio("non-biodegradable.mp3", format="audio/mp3")
                        waste_type = "Non-Biodegradable"
                    elif detected_labels & biodegradable and detected_labels & non_biodegradable:
                        st.info("Sisa ini mengandungi **Biodegradasi dan Tidak Biodegradasi**.")
                        waste_type = "Mixed"
                    else:
                        st.warning("Tiada jenis sisa dapat dikenalpasti.")

                    if waste_type in ["Biodegradable", "Non-Biodegradable"]:
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)

                        pdf.set_title("Laporan Pengesanan Sisa")
                        pdf.cell(200, 10, txt="Laporan Pengesanan Sisa", ln=True, align='C')
                        pdf.ln(10)

                        pdf.cell(200, 10, txt=f"Jenis Sisa Dikesan: {waste_type}", ln=True)
                        pdf.ln(5)

                        if waste_type == "Biodegradable":
                            pdf.multi_cell(0, 10, txt=(
                                "Ketepatan model CNN berdasarkan pembelajaran mendalam: 90%\n"
                                "Ketepatan model YOLOv8: sekitar 85%\n"
                                "Sisa biodegradasi seperti epal dan pisang boleh terurai secara semula jadi.\n\n"
                                " Cadangan Pelupusan:\n"
                                "- Buang dalam tong hijau.\n"
                                "- Kompos di rumah jika boleh.\n"
                                "- Elakkan mencampur dengan plastik atau kaca.\n\n"
                                " Langkah Alam Sekitar:\n"
                                "- Galakkan kompos organik.\n"
                                "- Kekalkan kebersihan sekitar.\n"
                                "- Didik masyarakat tentang pengasingan sisa."
                            ))
                        else:
                            pdf.multi_cell(0, 10, txt=(
                                "Ketepatan model CNN berdasarkan pembelajaran mendalam: 90%\n"
                                "Ketepatan model YOLOv8: sekitar 85%\n"
                                "Sisa tidak biodegradasi seperti plastik dan kaca tidak boleh terurai secara semula jadi.\n\n"
                                " Cadangan Pelupusan:\n"
                                "- Buang dalam tong biru atau pusat kitar semula.\n"
                                "- Jangan sekali-kali membakar sisa plastik.\n\n"
                                " Langkah Alam Sekitar:\n"
                                "- Kurangkan penggunaan plastik.\n"
                                "- Sokong inisiatif kitar semula.\n"
                                "- Pastikan kawasan awam bebas sampah."
                            ))

                        pdf_output = io.BytesIO()
                        pdf_bytes = pdf.output(dest='S').encode('latin-1')
                        pdf_output.write(pdf_bytes)
                        pdf_output.seek(0)

                        st.download_button(
                            label="📄 Muat Turun Laporan Sisa (PDF)",
                            data=pdf_output,
                            file_name="laporan_sisa.pdf",
                            mime="application/pdf"
                        )

        st.divider()

    with a:
        st.write("## 🌍 Kenapa Ia Penting?")
        st.write("Setiap hari kita menghasilkan sisa. Jika tidak diurus dengan betul, ia boleh mencemarkan alam sekitar.")
        st.write("- Mengurangkan pencemaran")  
        st.write("- Meningkatkan keberkesanan kitar semula")  
        st.write("- Menyokong masa depan yang bersih dan hijau")
        st.write("Sistem kami dapat **membezakan antara bahan biodegradasi dan tidak biodegradasi**, menjadikan pelupusan lebih pintar.")
        st.write('---')
        st.write('#### Hanya muat naik atau ambil gambar — sistem akan mengurus selebihnya!')
        st.write('---')
        st.write("### 🧠 Bagaimana Ia Berfungsi?")
        st.write("Menggunakan pembelajaran mesin dan pemprosesan imej, model kami menganalisis imej sisa dan mengelaskannya kepada dua kategori:")
        st.markdown("- **Biodegradasi** – Sisa semula jadi seperti buah-buahan, daun, dll.")
        st.markdown("- **Tidak Biodegradasi** – Sisa buatan seperti botol plastik, pembungkus, dll.")

    
    
    
    
