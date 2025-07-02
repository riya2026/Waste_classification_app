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

#from tensorflow.keras.models import load_model



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
                    "https://fast-api-backend-dv4g.onrender.com/predict",  # Change this to your deployed URL on Streamlit Cloud or public host
                    files={"file": ("image.jpg", image_bytes, "image/jpeg")},
                )
                
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
                    
                    # Continue with your biodegradable / non-biodegradable logic here
                else:
                    st.error("❌ Error from detection server.")
                st.image(result_img, caption="Detected Objects", use_container_width=True)
            
                boxes = results.boxes
                biodegradable = {'apple', 'banana'}
                non_biodegradable = {'plastic', 'glass'}
            
                # ✅ Define sets and maps BEFORE the loop
                detected_labels = set()
                confidence_map = {}
            
                if boxes is not None and boxes.cls is not None:
                    for i in range(len(boxes.cls)):
                        class_id = int(boxes.cls[i])
                        conf = float(boxes.conf[i])
            
                        # ✅ Safe check to avoid index out of range
                        if class_id < len(results.names):
                            label = results.names[class_id]
                        else:
                            label = f"Unknown Class {class_id}"
            
                        detected_labels.add(label)
                        confidence_map[label] = f"{conf:.2%}"
            
                    st.subheader("♻️ Waste Type Classification")
                    waste_type = None
                    if detected_labels & biodegradable and not detected_labels & non_biodegradable:
                        st.success("This waste is **Biodegradable**.")
                        waste_type = "Biodegradable"
                    elif detected_labels & non_biodegradable and not detected_labels & biodegradable:
                        st.error("This waste is **Non-Biodegradable**.")
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

if language == 'Tamil':
    st.title("கழிவுகள் வகைப்பாடு செயலி - சூழலினை சுத்தமாக வைத்திருக்கவும்")
    with st.sidebar:
        st.subheader("சிதையக்கூடிய பொருள் என்றால் என்ன?")
        st.write("சிதையக்கூடிய பொருள் என்பது இயற்கையாகவே நீர், கார்பன் டை ஆக்ஸைடு, மற்றும் உயிரியல் மேம்பொருட்களாகப் பாகுபடுத்தக்கூடிய ஒரு பொருள் ஆகும். உதாரணம்: ஆப்பிள், வாழைப்பழம் ஆகியவை.")

        st.subheader("சிதையாத பொருள் என்றால் என்ன?")
        st.write("சிதையாத பொருள் என்பது நுண்ணுயிரிகளால் இயற்கையாகவே அழிக்க முடியாத மற்றும் நீண்ட காலம் சுற்றுச்சூழலில் நிலைத்திருக்கும் ஒரு பொருள். உதாரணம்: பிளாஸ்டிக் பைகள், பிளாஸ்டிக் பாட்டில்கள், கண்ணாடி போன்றவை.")

        st.subheader("CNN என்றால் என்ன?")
        st.write("கணினி பார்வையில் பொதுவாகப் பயன்படுத்தப்படும் ஒரு ஆழமான நரம்பியல் வலைப்பின்னல்களின் வகை. இது தனக்கே உரிய வடிவங்களின் அடுக்குகளை கற்றுக்கொள்கிறது.")

        st.subheader("மெஷின் லெர்னிங் என்றால் என்ன?")
        st.write("இது ஒரு செயற்கை நுண்ணறிவின் கிளை. இது தரவுகளிலிருந்து நேரடியாக கற்றுக்கொண்டு முடிவெடுக்கக் கூடிய கணினிகளுக்கு திறன் அளிக்கிறது.")

        st.subheader("Streamlit என்பது என்ன?")
        st.write("பைதான் நிரலாக்க மொழியில் எழுதப்பட்ட தரவுப் பதிவுகளை எளிதாக இணையப் பயன்பாடுகளாக மாற்றும் ஒரு வடிவமைப்பு சூழல்.")

        st.subheader("YOLOv8 என்பது என்ன?")
        st.write("YOLOv8 என்பது படங்கள் அல்லது வீடியோக்களில் பொருட்களை கண்டறியக்கூடிய கணினி காட்சி முறை. இது 'You Only Look Once' என்ற முறைப்படி ஒரு முறையே பார்க்கிறது.")


    a, b = st.tabs(['முகப்பு', 'கண்டறிதல் மற்றும் அறிக்கை'])

    with b:
        st.title("📷 படம் உள்ளீட்டு செயலி")
        st.markdown(
        """
        **படத்தை வழங்க விரும்பும் முறையைத் தேர்ந்தெடுக்கவும்:**
        - **படத்தை பதிவேற்று** உங்கள் சாதனத்தில் இருந்து தேர்ந்தெடுக்க.
        - **புகைப்படம் எடு** உங்கள் வலைக்கேமரா மூலம் பதிவு செய்ய.
        """
        )
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
            uploaded_file = st.file_uploader("படக் கோப்பைத் தேர்ந்தெடுக்கவும்", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                st.session_state.image = Image.open(uploaded_file)
        
        elif st.session_state.mode == 'capture':
            camera_image = st.camera_input("புகைப்படம் எடுக்கவும்")
            if camera_image is not None:
                st.session_state.image = Image.open(camera_image)


        
        if st.session_state.image is not None:
            st.subheader("தேர்ந்தெடுக்கப்பட்ட படம்")
            st.image(
                st.session_state.image,
                caption="நீங்கள் வழங்கிய படம் இங்கே",
                use_container_width=True
            )
        
            # Detection Button
            if st.button("🔍 பொருட்களை கண்டறி"):
                st.subheader("கண்டறியப்பட்ட விளைவுகள்")

                # Run detection
                results = model.predict(source=st.session_state.image)
    
                # Get result image with boxes
                result_img = results[0].plot()
    
                # Show image with detections
                st.image(result_img, caption="Detected Objects", use_container_width=True)
    
                # Waste classification logic
                boxes = results[0].boxes
                biodegradable = {'apple', 'banana'}
                non_biodegradable = {'plastic', 'glass'}
                detected_labels = set()
                confidence_map = {}
    
                if boxes is not None and boxes.cls is not None:
                    for i in range(len(boxes.cls)):
                        label = results[0].names[int(boxes.cls[i])]
                        conf = float(boxes.conf[i])
                        detected_labels.add(label)
                        confidence_map[label] = f"{conf:.2%}"
    
                    # Waste type classification
                    st.subheader("♻️ Waste Type Classification")
                    waste_type = None
    
                    if detected_labels & biodegradable and not detected_labels & non_biodegradable:
                        st.success("இந்தக் கழிவு **சிதையக்கூடியது**.")
                        waste_type = "Biodegradable"
                    elif detected_labels & non_biodegradable and not detected_labels & biodegradable:
                        st.error("இந்தக் கழிவு **சிதையாதது**.")
                        waste_type = "Non-Biodegradable"
                    elif detected_labels & biodegradable and detected_labels & non_biodegradable:
                        st.info("இந்தக் கழிவில் **சிதையக்கூடிய மற்றும் சிதையாத** பொருட்கள் இரண்டும் உள்ளன.")
                        waste_type = "Mixed"
                    else:
                        st.warning("எந்த வகையான கழிவும் அடையாளம் காணப்படவில்லை.")
            
                    # PDF report
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
        st.write("## 🌍 ஏன் இது முக்கியம்")
    
        st.write("தினமும் நாங்கள் டன் கணக்கில் கழிவுகளை உருவாக்குகிறோம். அதில் பெரும்பாலானவை தவறான இடங்களில் முடிவடைகின்றன மற்றும் பூமிக்கு சேதம் ஏற்படுத்துகின்றன. சரியான வகைப்படுத்தல் முறையே முக்கியமானது:")
    
        st.write("- சுற்றுச்சூழல் மாசுபாட்டை குறைக்கும்")  
        st.write("- மறுசுழற்சி செயல்திறனை மேம்படுத்தும்")  
        st.write("- சுத்தமான மற்றும் பசுமையான எதிர்காலத்தைக் ஏற்படுத்தும்")
    
        st.write("நமது முறைமை **சிதையக்கூடிய மற்றும் சிதையாத பொருட்களை** வேறுபடுத்தக்கூடிய வகையில் வடிவமைக்கப்பட்டுள்ளது, இது கழிவு அகற்றத்தை புத்திசாலித்தனமாகவும் நிலைத்தன்மையுடனும் மாற்றுகிறது.")
    
        st.write('---')
    
        st.write('#### படம் எடுக்கவும் அல்லது பதிவேற்றவும் – உங்கள் சார்பில் அமைப்பு வேலை செய்யும்!')
    
        st.write('---')
    
        st.write("### 🧠 இது எப்படி செயல்படுகிறது")
        st.write("மெஷின் லெர்னிங் மற்றும் படத்தை செயலாக்குதல் மூலம், எங்கள் மாடல் கழிவுப் படங்களை பகுப்பாய்வு செய்து இரு முக்கிய வகைகளாக வகைப்படுத்துகிறது:")
        st.markdown("- **சிதையக்கூடியது** – ஆப்பிள், வாழைப்பழம், இலைகள் போன்ற இயற்கை கழிவுகள்.")
        st.markdown("- **சிதையாதது** – பிளாஸ்டிக் பாட்டில்கள், உலுப்பு கவர்கள் போன்ற செயற்கை கழிவுகள்.")
    
    
    
    
    
if language == 'Malay':
    st.title("Aplikasi Pengelasan Sisa - Kekalkan Alam Sekitar Bersih")

    with st.sidebar:
        st.subheader("Apa itu bahan terbiodegradasi?")
        st.write("Bahan terbiodegradasi ialah sebarang bahan yang boleh diuraikan secara semula jadi kepada unsur seperti air, karbon dioksida dan bahan organik oleh mikroorganisma tanpa mencemarkan alam sekitar. Contoh: Epal, pisang, dan lain-lain.")
    
        st.subheader("Apa itu bahan tidak terbiodegradasi?")
        st.write("Bahan tidak terbiodegradasi ialah bahan yang tidak boleh diuraikan oleh mikroorganisma secara semula jadi dan kekal lama dalam alam sekitar. Contoh: Beg plastik, botol plastik, kaca dan sebagainya.")
    
        st.subheader("Apa itu CNN?")
        st.write("Satu jenis rangkaian neural dalam yang biasa digunakan dalam visi komputer yang mempelajari hierarki ciri melalui lapisan konvolusi.")
    
        st.subheader("Apa itu Pembelajaran Mesin?")
        st.write("Satu cabang kecerdasan buatan yang membolehkan komputer belajar daripada data tanpa diprogram secara eksplisit.")
    
        st.subheader("Apa itu Streamlit?")
        st.write("Rangka kerja Python yang membolehkan anda menukar skrip data menjadi aplikasi web interaktif dengan kod yang minimum.")
    
        st.subheader("Apa itu YOLOv8?")
        st.write("YOLOv8 ialah sistem visi komputer yang boleh mengesan objek dalam imej atau video — seperti pisang, botol, atau manusia. YOLO bermaksud 'You Only Look Once', yang bermakna ia melihat imej sekali sahaja dan mengesan semua objek dengan cepat.")
    
    a, b = st.tabs(['Laman Utama', 'Pengesanan dan Laporan'])
    
    with b:
        st.title("📷 Aplikasi Input Imej")
        st.markdown(
            """
            **Pilih cara anda ingin menyediakan imej:**
            - **Muat Naik Imej** dari peranti anda.
            - **Ambil Gambar** menggunakan kamera web anda.
            """
        )

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
            st.subheader("Imej Terpilih")
            st.image(
                st.session_state.image,
                caption="Ini adalah imej yang anda berikan",
                use_container_width=True
            )
    
            # Butang Pengesanan
            if st.button("🔍 Kesan Objek"):
                st.subheader("Keputusan Pengesanan")
                    
                
                # Run detection
                results = model.predict(source=st.session_state.image)
    
                # Get result image with boxes
                result_img = results[0].plot()
    
                # Show image with detections
                st.image(result_img, caption="Detected Objects", use_container_width=True)
    
                # Waste classification logic
                boxes = results[0].boxes
                biodegradable = {'apple', 'banana'}
                non_biodegradable = {'plastic', 'glass'}
                detected_labels = set()
                confidence_map = {}
    
                if boxes is not None and boxes.cls is not None:
                    for i in range(len(boxes.cls)):
                        label = results[0].names[int(boxes.cls[i])]
                        conf = float(boxes.conf[i])
                        detected_labels.add(label)
                        confidence_map[label] = f"{conf:.2%}"
    
                    # Waste type classification
                    st.subheader("♻️ Waste Type Classification")
                    waste_type = None
                    if detected_labels & biodegradable and not detected_labels & non_biodegradable:
                        st.success("Sisa ini adalah **Terbiodegradasi**.")
                        waste_type = "Biodegradable"
                    elif detected_labels & non_biodegradable and not detected_labels & biodegradable:
                        st.error("Sisa ini adalah **Tidak Terbiodegradasi**.")
                        waste_type = "Non-Biodegradable"
                    elif detected_labels & biodegradable and detected_labels & non_biodegradable:
                        st.info("Sisa ini mengandungi **kedua-dua bahan Terbiodegradasi dan Tidak Terbiodegradasi**.")
                        waste_type = "Mixed"
                    else:
                        st.warning("Tiada jenis sisa yang dapat dikenalpasti.")
        
    
                    # PDF report
                    if waste_type in ["Biodegradable", "Non-Biodegradable"]:
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                    
                        pdf.set_title("Laporan Pengesanan Sisa")
                        pdf.cell(200, 10, txt="Laporan Pengesanan Sisa", ln=True, align='C')
                        pdf.ln(10)
                    
                        pdf.cell(200, 10, txt=f"Jenis Sisa Dikenalpasti: {waste_type}", ln=True)
                        pdf.ln(5)
                    
                        pdf.cell(200, 10, txt="Item Dikesan dan Tahap Keyakinan:", ln=True)
                        for item, conf in confidence_map.items():
                            pdf.cell(200, 10, txt=f"- {item} - {conf}", ln=True)
                    
                        pdf.ln(5)
                    
                        if waste_type == "Biodegradable":
                            pdf.multi_cell(0, 10, txt=(
                                "Sisa terbiodegradasi seperti epal dan pisang boleh terurai secara semula jadi.\n\n"
                                " Cadangan Pelupusan:\n"
                                "- Buang ke dalam tong hijau.\n"
                                "- Komposkan di rumah jika boleh.\n"
                                "- Jangan campur dengan sisa plastik atau kaca.\n\n"
                                " Langkah Alam Sekitar:\n"
                                "- Galakkan pengkomposan organik.\n"
                                "- Kekalkan kebersihan persekitaran.\n"
                                "- Didik orang ramai tentang pengasingan sisa terbiodegradasi."
                            ))
                        else:
                            pdf.multi_cell(0, 10, txt=(
                                "Sisa tidak terbiodegradasi seperti plastik dan kaca tidak boleh terurai secara semula jadi.\n\n"
                                " Cadangan Pelupusan:\n"
                                "- Buang ke dalam tong biru atau pusat kitar semula.\n"
                                "- Jangan bakar sisa plastik.\n\n"
                                " Langkah Alam Sekitar:\n"
                                "- Kurangkan penggunaan plastik.\n"
                                "- Sokong inisiatif kitar semula.\n"
                                "- Kekalkan kebersihan tempat awam."
                            ))
                    
                        pdf_output = io.BytesIO()
                        pdf_bytes = pdf.output(dest='S').encode('latin-1')
                        pdf_output.write(pdf_bytes)
                        pdf_output.seek(0)
                    
                        st.download_button(
                            label="📄 Muat Turun Laporan Sisa sebagai PDF",
                            data=pdf_output,
                            file_name="waste_report.pdf",
                            mime="application/pdf"
                        )
                
        st.divider()
                
                        
                        
    with a:
        st.write("## 🌍 Kenapa Ia Penting")
    
        st.write("Setiap hari, kita menghasilkan bertan-tan sisa. Malangnya, kebanyakannya dibuang di tempat yang salah dan merosakkan bumi kita. Pengasingan sisa yang betul adalah kunci kepada:")
    
        st.write("- Mengurangkan pencemaran alam sekitar")  
        st.write("- Meningkatkan kecekapan kitar semula")  
        st.write("- Menyokong masa depan yang bersih dan hijau")
    
        st.write("Sistem kami direka bentuk untuk **membezakan antara sisa terbiodegradasi dan tidak terbiodegradasi**, menjadikan pelupusan sisa lebih bijak dan mampan.")
    
        st.write('---')
    
        st.write('#### Ambil atau muat naik gambar – sistem akan melakukannya untuk anda!')
    
        st.write('---')
    
        st.write("### 🧠 Bagaimana Ia Berfungsi")
        st.write("Dengan menggunakan pembelajaran mesin dan pemprosesan imej, model kami menganalisis imej sisa dan mengelaskannya kepada dua kategori utama:")
        st.markdown("- **Terbiodegradasi** – Sisa semula jadi seperti buah-buahan (epal, pisang), daun, dll.")
        st.markdown("- **Tidak Terbiodegradasi** – Sisa buatan seperti botol plastik, pembungkus, dll.")
    
    
    
    
    
